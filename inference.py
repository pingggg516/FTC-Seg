#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import socket
import sys
import time
import traceback

sys.dont_write_bytecode = True


def arguments():
    p = argparse.ArgumentParser(description='FTC-Seg unified inference: explicit inputs, local prediction and mIoU.\n\nRun from the FTC-Seg checkout. Only model definitions are imported from the\nproject; no training, dataset, evaluation or other inference script is called.\nThe RN101 compatibility model below preserves the historical checkpoint layout.\nOutputs go into a NEW directory; checkpoints and datasets are read-only.\n')
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--checkpoint-key', required=True, choices=['model', 'model_ema'])
    p.add_argument('--backbone', required=True, choices=['dinov2_small', 'resnet101'])
    p.add_argument('--num-opr-layers', type=int, default=1)
    p.add_argument('--nclass', required=True, type=int)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--split-path', required=True, type=Path,
                   help='Text file: image_path mask_path on each line.')
    p.add_argument('--mode', choices=['original', 'sliding_window'], default='original')
    p.add_argument('--crop-size', type=int, default=798)
    p.add_argument('--mask-override-dir', type=Path,
                   help='Optional GT directory; match mask basenames, never modify original GT.')
    p.add_argument('--expected-overrides', type=int, default=0)
    p.add_argument('--output-dir', required=True, type=Path,
                   help='Must not already exist. Saves console.log and result.json.')
    p.add_argument('--save-predictions', action='store_true',
                   help='Also save original-size class-ID PNG predictions.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    if not 2 <= a.nclass <= 255 or not 0 <= a.num_opr_layers <= 4:
        p.error('nclass must be 2..255; num-opr-layers must be 0..4')
    if a.backbone == 'resnet101' and a.num_opr_layers != 1:
        p.error('Historical RN101 checkpoints require exactly one OPR layer')
    if a.crop_size <= 0 or a.crop_size % 14 or a.workers < 0:
        p.error('crop-size must be a positive multiple of 14; workers >= 0')
    if bool(a.mask_override_dir) != (a.expected_overrides > 0):
        p.error('Set mask-override-dir and a positive expected-overrides together')
    return a


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_opr_settings(state, nclass, num_opr_layers):
    import torch
    classifier = state.get("head.scratch.output_conv.2.weight")
    if num_opr_layers == 0:
        encoder = state.get("backbone.patch_embed.proj.weight")
        if classifier is None or encoder is None:
            raise ValueError("Cannot identify checkpoint decoder dimensions.")
        return ("paper" if classifier.shape[1] == encoder.shape[0] else "legacy"), None
    legacy = any(".proto_classifier." in key for key in state)
    masks = [value for key, value in state.items()
             if key.endswith(".foreground_mask")]
    if legacy:
        if masks:
            raise ValueError("Checkpoint mixes legacy and shared-head OPR parameters.")
        return "legacy", None
    if not masks:
        raise ValueError("OPR checkpoint has no recorded semantic gate configuration.")
    for mask in masks:
        if tuple(mask.shape) != (nclass,) or mask.dtype != torch.bool:
            raise ValueError("Invalid foreground mask in checkpoint.")
        if not torch.equal(mask, masks[0]):
            raise ValueError("Inconsistent foreground classes across OPR modules.")
    ids = masks[0].nonzero(as_tuple=False).flatten().tolist()
    if not ids:
        raise ValueError("Checkpoint foreground class set is empty.")
    prototypes = [v for k, v in state.items() if k.endswith(".prototypes")]
    if (classifier is None or classifier.ndim != 4 or tuple(classifier.shape[2:]) != (1, 1)
            or not prototypes
            or any(p.ndim != 2 or p.shape[1] != classifier.shape[1] for p in prototypes)):
        raise ValueError("Checkpoint used the superseded decoder-query adaptation, "
                         "not direct paper OPR; it cannot be silently converted.")
    return "paper", ids


def main(a):
    a.output_dir.mkdir(parents=True, exist_ok=False)
    fd = os.open(str(a.output_dir / 'console.log'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    started = time.time()
    events = (a.output_dir / 'progress.jsonl').open('x', buffering=1)

    def event(state, **fields):
        record = dict(state=state, time=time.time(), **fields)
        line = json.dumps(record, ensure_ascii=False)
        events.write(line + '\n')
        print(line, flush=True)

    try:
        event('loading', host=socket.gethostname(), pid=os.getpid(),
              gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
              command=shlex.join([sys.executable] + sys.argv))
        import numpy as np
        from PIL import Image
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from model.semseg.dpt import DPT
        from model.util.blocks import FeatureFusionBlock, _make_scratch
        from model.util.legacy_cr_bci import CR_BCI_Module as LegacyCRBCI

        torch.set_num_threads(4)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        device = torch.device(a.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        class RNBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet101(pretrained=False)
                del self.model.fc
                del self.model.avgpool

            def forward(self, x):
                m = self.model
                x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
                l1 = m.layer1(x)
                l2 = m.layer2(l1)
                l3 = m.layer3(l2)
                return [l1, l2, l3, m.layer4(l3)]

        class RNHead(nn.Module):
            def __init__(self):
                super().__init__()
                channels = [256, 512, 1024, 2048]
                self.projects = nn.ModuleList([nn.Conv2d(c, c, 1) for c in channels])
                self.resize_layers = nn.ModuleList([nn.Identity() for _ in channels])
                self.scratch = _make_scratch(channels, 256, groups=1, expand=False)
                self.scratch.stem_transpose = None
                for i in range(1, 5):
                    setattr(self.scratch, 'refinenet' + str(i), FeatureFusionBlock(
                        256, nn.ReLU(False), deconv=False, bn=False, expand=False,
                        align_corners=True, size=None))
                self.scratch.output_conv = nn.Sequential(
                    nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.Conv2d(256, a.nclass, 1))

            def forward(self, features):
                levels = [self.resize_layers[i](self.projects[i](x)) for i, x in enumerate(features)]
                s = self.scratch
                l1, l2, l3, l4 = [getattr(s, 'layer%d_rn' % (i + 1))(x)
                                   for i, x in enumerate(levels)]
                p4 = s.refinenet4(l4, size=l3.shape[2:])
                p3 = s.refinenet3(p4, l3, size=l2.shape[2:])
                p2 = s.refinenet2(p3, l2, size=l1.shape[2:])
                return s.output_conv(s.refinenet1(p2, l1))

        class RNDPT(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = RNBackbone()
                self.head = RNHead()
                self.cr_bci = LegacyCRBCI(dim=2048, nclass=a.nclass)

            def forward(self, x):
                features = self.backbone(x)
                features[-1] = self.cr_bci(features[-1].clone(), mode='labeled')
                return F.interpolate(self.head(features), x.shape[-2:],
                                     mode='bilinear', align_corners=True)

        before = a.checkpoint.stat()
        checkpoint = torch.load(str(a.checkpoint), map_location='cpu')
        state = {k[7:] if k.startswith('module.') else k: v
                 for k, v in checkpoint[a.checkpoint_key].items()}
        epoch = int(checkpoint['epoch']) if 'epoch' in checkpoint else None
        aliases = []
        opr_gate_mode, foreground_class_ids = checkpoint_opr_settings(
            state, a.nclass, a.num_opr_layers)
        disable_gamma_k = bool(checkpoint.get('disable_opr_gamma_k', False))
        disable_residual = bool(checkpoint.get('disable_opr_residual', False))
        if a.backbone == 'resnet101':
            if opr_gate_mode != 'legacy':
                raise ValueError("RN101 support is for historical checkpoints only.")
            model = RNDPT()
        else:
            model = DPT(encoder_size='small', nclass=a.nclass, features=64,
                        out_channels=[48, 96, 192, 384], num_opr_layers=a.num_opr_layers,
                        disable_gamma_k=disable_gamma_k, disable_residual=disable_residual,
                        foreground_class_ids=foreground_class_ids,
                        opr_gate_mode=opr_gate_mode)
            last_prefix = 'cr_bci_modules.%d.' % (a.num_opr_layers - 1)
            for key, value in list(state.items()):
                if key.startswith('cr_bci.') and a.num_opr_layers > 0:
                    alias = last_prefix + key[len('cr_bci.'):]
                    if alias in state:
                        if not torch.equal(value, state[alias]):
                            raise ValueError('Inconsistent OPR alias: ' + key)
                    else:
                        state[alias] = value
                        aliases.append(alias)
        model.load_state_dict(state, strict=True)
        del state, checkpoint
        model.to(device).eval()

        rows = []
        applied = []
        used_overrides = set()
        overrides = {}
        if a.mask_override_dir:
            overrides = {p.name: p.resolve() for p in a.mask_override_dir.glob('*.png')}
            if len(overrides) != a.expected_overrides:
                raise ValueError('Override directory PNG count does not match expected-overrides')
        for line_no, line in enumerate(a.split_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError('Expected image and mask paths at split line %d' % line_no)
            image, original = [(a.data_root / f).resolve() for f in fields]
            mask = overrides.get(original.name, original)
            if mask != original:
                if original.name in used_overrides:
                    raise ValueError('Ambiguous/repeated override basename: ' + original.name)
                used_overrides.add(original.name)
                applied.append(dict(original=str(original), override=str(mask), sha256=sha256(mask)))
            for p in (image, original, mask):
                if not p.is_file():
                    raise FileNotFoundError(p)
            rows.append((image, mask))
        if not rows or len(applied) != a.expected_overrides:
            raise ValueError('Empty split or incorrect number of applied GT overrides')

        class ImagesAndMasks(Dataset):
            def __len__(self):
                return len(rows)

            def __getitem__(self, i):
                image_path, mask_path = rows[i]
                with Image.open(image_path) as im:
                    image = transforms.ToTensor()(im.convert('RGB'))
                image = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(image)
                with Image.open(mask_path) as im:
                    mask = np.array(im, dtype=np.int64)
                if mask.ndim != 2 or tuple(image.shape[-2:]) != mask.shape:
                    raise ValueError('Image/mask shape mismatch: ' + str(mask_path))
                valid = ((mask >= 0) & (mask < a.nclass)) | (mask == 255)
                if not valid.all():
                    raise ValueError('Invalid class IDs: ' + str(mask_path))
                return image, torch.from_numpy(mask)

        loader = DataLoader(ImagesAndMasks(), batch_size=1, shuffle=False,
                            num_workers=a.workers, pin_memory=device.type == 'cuda')

        def predict(image):
            h, w = image.shape[-2:]
            if a.mode == 'original':
                size = (max(14, int(h / 14 + .5) * 14), max(14, int(w / 14 + .5) * 14))
                resized = F.interpolate(image, size, mode='bilinear', align_corners=True)
                logits = F.interpolate(model(resized), (h, w), mode='bilinear', align_corners=True)
            else:
                grid = a.crop_size
                if min(h, w) < grid:
                    raise ValueError('Sliding-window image is smaller than crop-size')
                logits = torch.zeros(1, a.nclass, h, w, device=device)
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        patch = model(image[:, :, row:row + grid, col:col + grid])
                        logits[:, :, row:row + grid, col:col + grid] += patch.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)
            return logits.argmax(dim=1).cpu().numpy()[0]

        intersection = np.zeros(a.nclass, dtype=np.int64)
        union = np.zeros_like(intersection)
        target = np.zeros_like(intersection)
        if a.save_predictions:
            (a.output_dir / 'predictions').mkdir()
        event('evaluating', processed=0, total=len(rows), strict_load=True,
              epoch_index=epoch, checkpoint_key=a.checkpoint_key, overrides=len(applied))
        with torch.no_grad():
            for i, (image, mask_tensor) in enumerate(loader):
                prediction = predict(image.to(device))
                mask = mask_tensor.numpy()[0]
                keep = mask != 255
                pred_valid, gt_valid = prediction[keep], mask[keep]
                area_i = np.bincount(gt_valid[pred_valid == gt_valid], minlength=a.nclass)
                area_p = np.bincount(pred_valid, minlength=a.nclass)
                area_t = np.bincount(gt_valid, minlength=a.nclass)
                intersection += area_i
                union += area_p + area_t - area_i
                target += area_t
                if a.save_predictions:
                    name = '%06d_%s.png' % (i, rows[i][0].stem)
                    with (a.output_dir / 'predictions' / name).open('xb') as f:
                        Image.fromarray(prediction.astype(np.uint8)).save(f, format='PNG')
                if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(rows):
                    event('evaluating', processed=i + 1, total=len(rows), elapsed=time.time() - started)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        after = a.checkpoint.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Checkpoint changed during evaluation')
        iou = intersection / (union + 1e-10) * 100
        result = dict(state='completed', miou=float(iou.mean()), class_iou=iou.tolist(),
                      pixel_counts=dict(intersection=intersection.tolist(), union=union.tolist(), target=target.tolist()),
                      processed=len(rows), checkpoint=str(a.checkpoint.resolve()), checkpoint_key=a.checkpoint_key,
                      checkpoint_epoch_index=epoch, strict_load=True, aliases_added=aliases,
                      backbone=a.backbone, nclass=a.nclass, num_opr_layers=a.num_opr_layers,
                      opr_gate_mode=opr_gate_mode, opr_foreground_class_ids=foreground_class_ids,
                      disable_opr_gamma_k=disable_gamma_k, disable_opr_residual=disable_residual,
                      data_root=str(a.data_root.resolve()), split_path=str(a.split_path.resolve()),
                      split_sha256=sha256(a.split_path), applied_overrides=applied,
                      mode=a.mode, crop_size=a.crop_size if a.mode == 'sliding_window' else None,
                      ignore_index=255, include_background=True, absent_class_iou=0,
                      flip=False, multiscale=False, resize_multiplier=14,
                      script_sha256=sha256(__file__), host=socket.gethostname(),
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      torch_version=torch.__version__, torchvision_version=torchvision.__version__,
                      elapsed_seconds=time.time() - started)
        with (a.output_dir / 'result.json').open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        event('completed', processed=len(rows), total=len(rows), miou=result['miou'])
    except BaseException as exc:
        traceback.print_exc()
        event('failed', error=repr(exc))
        raise
    finally:
        events.close()


if __name__ == '__main__':
    main(arguments())
