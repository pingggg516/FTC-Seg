import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.dinov2 import DINOv2
from model.util.blocks import FeatureFusionBlock, _make_scratch
from model.util.cr_bci import CR_BCI_Module


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024],
        classifier_dim=None,
    ):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.classifier_dim = features if classifier_dim is None else classifier_dim
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, self.classifier_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.classifier_dim, nclass, kernel_size=1, stride=1, padding=0)
        )
    
    def forward_features(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv[0](path_1)
        return self.scratch.output_conv[1](out)

    def forward(self, out_features, patch_h, patch_w):
        features = self.forward_features(out_features, patch_h, patch_w)
        return self.scratch.output_conv[2](features)

    @torch.no_grad()
    def classify_prototypes(self, prototypes):
        classifier = self.scratch.output_conv[2]
        if prototypes.ndim != 2 or prototypes.shape[1] != classifier.in_channels:
            raise ValueError("Raw prototype and pixel classifier dimensions must match.")
        return F.linear(prototypes, classifier.weight[:, :, 0, 0], classifier.bias)


class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=21,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        num_opr_layers=1,
        disable_gamma_k=False,
        disable_residual=False,
        foreground_class_ids=None,
        opr_gate_mode="paper",
    ):
        super(DPT, self).__init__()
        
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
            'large': [4, 11, 17, 23], 
            'giant': [9, 19, 29, 39]
        }
        
        self.encoder_size = encoder_size
        self.backbone = DINOv2(model_name=encoder_size)
        
        if opr_gate_mode not in ("paper", "legacy"):
            raise ValueError("Unknown OPR gate mode: " + str(opr_gate_mode))
        self.head = DPTHead(
            nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels,
            classifier_dim=self.backbone.embed_dim if opr_gate_mode == "paper" else features)
        
        self.num_opr_layers = num_opr_layers
        self.disable_gamma_k = disable_gamma_k
        self.disable_residual = disable_residual
        if not 0 <= self.num_opr_layers <= 4:
            raise ValueError("num_opr_layers must be between 0 and 4.")
        self.opr_gate_mode = opr_gate_mode
        opr_class = CR_BCI_Module
        opr_kwargs = {"foreground_class_ids": foreground_class_ids}
        if opr_gate_mode == "legacy":
            from model.util.legacy_cr_bci import CR_BCI_Module as opr_class
            opr_kwargs = {}

        if self.num_opr_layers > 0 and (opr_gate_mode == 'paper' or self.encoder_size == 'small'):
            self.cr_bci_modules = nn.ModuleList([
                opr_class(
                    dim=self.backbone.embed_dim,
                    nclass=nclass,
                    disable_gamma_k=self.disable_gamma_k,
                    disable_residual=self.disable_residual,
                    **opr_kwargs,
                )
                for _ in range(self.num_opr_layers)
            ])
            self.cr_bci = self.cr_bci_modules[-1]
        else:
            self.cr_bci_modules = None
            self.cr_bci = None
            
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)
        
    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x, comp_drop=False, mode='labeled', return_feat=False):
        h, w = x.shape[-2], x.shape[-1]
        patch_h = h // 14 if isinstance(h, int) else torch.div(h, 14, rounding_mode='trunc')
        patch_w = w // 14 if isinstance(w, int) else torch.div(w, 14, rounding_mode='trunc')
        
        features = self.backbone.get_intermediate_layers(
            x, self.intermediate_layer_idx[self.encoder_size]
        )
        
        feat_raw = None
        aux_logits_list = []
        
        if self.cr_bci_modules is not None:
            features = list(features)
            
            start_idx = 4 - self.num_opr_layers
            
            for i, cr_bci_module in enumerate(self.cr_bci_modules):
                feat_idx = start_idx + i
                feat_r = features[feat_idx].clone()
                
                B, N, C = feat_r.shape
                f_hw = feat_r.permute(0, 2, 1).reshape(B, C, patch_h, patch_w)
                
                if self.opr_gate_mode == "legacy":
                    f_enhanced = cr_bci_module(f_hw, mode=mode)
                else:
                    prototype_logits = None
                    if cr_bci_module.requires_semantic_gate:
                        prototype_logits = self.head.classify_prototypes(cr_bci_module.prototypes)
                    f_enhanced = cr_bci_module(
                        f_hw, mode=mode, prototype_logits=prototype_logits)
                
                f_enhanced = f_enhanced.flatten(2).transpose(1, 2)
                
                features[feat_idx] = f_enhanced
                
                if feat_idx == 3:
                    feat_raw = f_hw
                    
            features = tuple(features)
            
            aux_logits = None
        else:
            aux_logits = None
            
        if comp_drop:
            bs, dim = features[0].shape[0], features[0].shape[-1]
            
            dropout_mask1 = self.binomial.sample((bs // 2, dim)).to(features[0].device) * 2.0
            dropout_mask2 = 2.0 - dropout_mask1
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            dropout_mask1[kept_indexes, :] = 1.0
            dropout_mask2[kept_indexes, :] = 1.0
            
            dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
            
            features = (feature * dropout_mask.unsqueeze(1) for feature in features)
            
            out = self.head(features, patch_h, patch_w)
            
            out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
            
            if return_feat:
                return out, feat_raw, aux_logits
            return out
        
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 14, patch_w * 14), mode='bilinear', align_corners=True)
        
        if return_feat:
            return out, feat_raw, aux_logits
        return out
