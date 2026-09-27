import argparse
from copy import deepcopy
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from model.util.cr_bci import CR_BCI_Module
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d, IgnoreSafeCrossEntropyLoss
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed, DistributedEvalSampler, update_atc_statistics


parser = argparse.ArgumentParser(description='FTC-Seg: Semi-Supervised Semantic Segmentation Training')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def initialize_atc_statistics(nclass, device, checkpoint=None):
    if checkpoint is None:
        return (torch.ones(nclass, device=device) / nclass,
                torch.ones(nclass, device=device) / nclass,
                torch.ones(nclass, device=device))
    state = checkpoint.get('atc_state')
    if not isinstance(state, dict) or set(state) != {'p_l', 'p_u', 'acc_l'}:
        raise ValueError('Checkpoint has no complete ATC EMA state; cannot resume continuously.')
    result = []
    for name in ('p_l', 'p_u', 'acc_l'):
        value = state[name]
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (nclass,)
                or not torch.isfinite(value).all()):
            raise ValueError('Invalid ATC statistic: ' + name)
        result.append(value.detach().to(device=device, dtype=torch.float32).clone())
    return tuple(result)


def main():
    parser.add_argument('--num-opr-layers', type=int, default=1,
                        help='Number of layers to apply OPR. 0: none, 1: layer4(default), 2: layers 3+4, etc.')
    parser.add_argument('--atc-mode', '--acr-mode', dest='atc_mode', type=str, default='full',
                        choices=['full', 'mdiff_only', 'mbias_only'],
                        help='Ablation mode for ATC: full=Mdiff+Mbias, mdiff_only=only difficulty term, mbias_only=only distribution-bias term.')
    parser.add_argument('--disable-opr-gamma-k', action='store_true',
                        help='Disable gamma_k gating inside OPR.')
    parser.add_argument('--disable-opr-residual', action='store_true',
                        help='Disable residual connection inside OPR.')
    parser.add_argument('--disable-opr-loss-o', action='store_true',
                        help='Disable orthogonality loss L_O during training.')
    
    parser.add_argument('--foreground-class-ids', nargs='+', type=int, default=None,
                        help='Explicit OPR foreground class IDs; overrides YAML.')
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    foreground_class_ids = args.foreground_class_ids
    if foreground_class_ids is None:
        foreground_class_ids = cfg.get('opr_foreground_class_ids')
    if (args.num_opr_layers > 0 and not args.disable_opr_gamma_k
            and foreground_class_ids is None):
        raise ValueError(
            'Define opr_foreground_class_ids in the YAML or pass --foreground-class-ids. '
            'Do not assume class 0 is background: ACDC/Cityscapes class 0 is road.')
    cfg['opr_foreground_class_ids'] = foreground_class_ids
    cfg['opr_gate_mode'] = 'paper'

    checkpoint_path = os.path.join(args.save_path, 'latest.pth')
    resume_checkpoint = None
    if os.path.exists(checkpoint_path):
        resume_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        initialize_atc_statistics(cfg['nclass'], 'cpu', resume_checkpoint)
        if args.num_opr_layers > 0:
            if resume_checkpoint.get('opr_gate_mode') != 'paper':
                raise ValueError(
                    'Historical OPR checkpoint: use a new --save-path for the corrected '
                    'mechanism. infer_unified.py still supports historical inference.')
            if resume_checkpoint.get('opr_foreground_class_ids') != foreground_class_ids:
                raise ValueError('Checkpoint foreground classes differ from this run.')
            for key in ('disable_opr_gamma_k', 'disable_opr_residual'):
                if resume_checkpoint.get(key) != getattr(args, key):
                    raise ValueError('Checkpoint OPR option differs: ' + key)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)

        file_handler = logging.FileHandler(os.path.join(args.save_path, 'train.log'))
        formatter = logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{
        **model_configs[cfg['backbone'].split('_')[-1]],
        'nclass': cfg['nclass'],
        'num_opr_layers': args.num_opr_layers,
        'disable_gamma_k': args.disable_opr_gamma_k,
        'disable_residual': args.disable_opr_residual,
        'foreground_class_ids': foreground_class_ids,
        'opr_gate_mode': 'paper',
    })
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
        
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = IgnoreSafeCrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val'
    )
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )
    
    valsampler = DistributedEvalSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    
    if min(len(trainloader_l), len(trainloader_u)) == 0:
        raise ValueError('No complete training batch per rank; reduce batch_size or provide more training samples.')
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
        del checkpoint
    
    nclass = cfg['nclass']
    atc_p_l, atc_p_u, atc_acc_l = initialize_atc_statistics(
        nclass, torch.device('cuda', local_rank), resume_checkpoint)
    del resume_checkpoint
    
    atc_momentum = 0.999
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_o = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            with torch.no_grad():
                pred_u_w, feat_u_w_raw, _ = model_ema(img_u_w, mode='unlabeled', return_feat=True)
                
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
                
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x, feat_x_raw, aux_logits_x = model(img_x, mode='labeled', return_feat=True)
            

            with torch.no_grad():
                atc_p_l, atc_p_u, atc_acc_l = update_atc_statistics(
                    atc_p_l, atc_p_u, atc_acc_l, mask_x, pred_x.argmax(dim=1),
                    mask_u_w, ignore_mask, atc_momentum)

                base_thresh = cfg['conf_thresh']
                
                difficulty = 1.0 - atc_acc_l
                mdiff = -0.1 * difficulty

                ratio = atc_p_u / (atc_p_l + 1e-6)
                mbias = 0.05 * (ratio - 1.0)
                mbias = torch.clamp(mbias, -0.1, 0.1)

                if args.atc_mode == 'full':
                    adaptive_thresh = base_thresh + mdiff + mbias
                elif args.atc_mode == 'mdiff_only':
                    adaptive_thresh = base_thresh + mdiff
                elif args.atc_mode == 'mbias_only':
                    adaptive_thresh = base_thresh + mbias
                else:
                    raise ValueError(f'Unsupported atc_mode: {args.atc_mode}')
                
                adaptive_thresh = torch.clamp(adaptive_thresh, 0.5, 0.98)
                
                current_iter = epoch * len(trainloader_u) + i
                if i % 500 == 0 and rank == 0:
                    logger.info(f"\n[ATC Stats] Iter {current_iter} | Mode={args.atc_mode}")
                    for c in range(nclass):
                        logger.info(
                            f"  Class {c:2d} ({CLASSES[cfg['dataset']][c]:10s}): "
                            f"Acc={atc_acc_l[c]:.3f}, Ratio={ratio[c]:.3f}, "
                            f"Mdiff={mdiff[c]:.3f}, Mbias={mbias[c]:.3f}, Thresh={adaptive_thresh[c]:.3f}"
                        )


            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True, mode='unlabeled').chunk(2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)
            
            if aux_logits_x is not None:
                if isinstance(aux_logits_x, list):
                    loss_aux = 0.0
                    for aux_l in aux_logits_x:
                        aux_l = F.interpolate(aux_l, size=mask_x.shape[-2:], mode='bilinear', align_corners=True)
                        loss_aux += criterion_l(aux_l, mask_x)
                    loss_x = loss_x + 0.4 * (loss_aux / len(aux_logits_x))
                else:
                    aux_logits_x = F.interpolate(aux_logits_x, size=mask_x.shape[-2:], mode='bilinear', align_corners=True)
                    loss_aux = criterion_l(aux_logits_x, mask_x)
                    loss_x = loss_x + 0.4 * loss_aux

            thresh_map_1 = adaptive_thresh[mask_u_w_cutmixed1]
            thresh_map_2 = adaptive_thresh[mask_u_w_cutmixed2]

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= thresh_map_1) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().clamp_min(1)
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= thresh_map_2) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().clamp_min(1)
            
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            loss_ortho = 0.0
            if (not args.disable_opr_loss_o) and hasattr(model.module, 'cr_bci_modules') and model.module.cr_bci_modules is not None:
                for cr_bci in model.module.cr_bci_modules:
                    loss_ortho += cr_bci.get_ortho_loss()
                loss_ortho = loss_ortho / len(model.module.cr_bci_modules)
            elif (not args.disable_opr_loss_o) and hasattr(model.module, 'cr_bci') and model.module.cr_bci is not None:
                loss_ortho = model.module.cr_bci.get_ortho_loss()
            
            loss = (loss_x + loss_u_s) / 2.0 + 0.01 * loss_ortho
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            if isinstance(loss_ortho, torch.Tensor):
                total_loss_o.update(loss_ortho.item())
            else:
                total_loss_o.update(loss_ortho)
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum().clamp_min(1)
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            
            for (name, buffer), (_, buffer_ema) in zip(model.named_buffers(), model_ema.named_buffers()):
                if buffer.is_floating_point():
                    buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
                else:
                    buffer_ema.copy_(buffer)
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                if isinstance(loss_ortho, torch.Tensor):
                    writer.add_scalar('train/loss_o', loss_ortho.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % max(1, len(trainloader_u) // 8) == 0) and (rank == 0):
                max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss o: {:.3f}, Mask ratio: '
                            '{:.3f}, Max Mem: {:.0f}MB'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, total_loss_o.avg, total_mask_ratio.avg, max_mem_mb))
        
        eval_mode = 'sliding_window' if cfg['dataset'] in ('ACDC', 'cityscapes') else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=14)
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)

        is_best = max(mIoU, mIoU_ema) >= max(previous_best, previous_best_ema)
        
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'selected_branch': 'model' if mIoU >= mIoU_ema else 'model_ema',
                'selected_miou': max(mIoU, mIoU_ema),
                'student_miou': mIoU,
                'ema_miou': mIoU_ema,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema,
                'atc_state': {
                    'p_l': atc_p_l.detach().cpu().clone(),
                    'p_u': atc_p_u.detach().cpu().clone(),
                    'acc_l': atc_acc_l.detach().cpu().clone(),
                },
                'opr_gate_mode': 'paper',
                'opr_foreground_class_ids': foreground_class_ids,
                'disable_opr_gamma_k': args.disable_opr_gamma_k,
                'disable_opr_residual': args.disable_opr_residual,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
                logger.info('Saved best.pth: %s, mIoU %.4f, epoch %d',
                            checkpoint['selected_branch'], checkpoint['selected_miou'], epoch)


if __name__ == '__main__':
    main()
