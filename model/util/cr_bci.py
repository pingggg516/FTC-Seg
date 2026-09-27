import torch
import torch.nn as nn
import torch.nn.functional as F


class CR_BCI_Module(nn.Module):
    def __init__(self, dim, nclass=2, disable_gamma_k=False,
                 disable_residual=False, foreground_class_ids=None):
        super().__init__()
        self.dim = dim
        self.nclass = nclass
        self.disable_gamma_k = disable_gamma_k
        self.disable_residual = disable_residual
        self.num_prototypes = 16
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, dim))
        nn.init.orthogonal_(self.prototypes)

        if foreground_class_ids is None:
            if not disable_gamma_k:
                raise ValueError("OPR requires explicit foreground_class_ids for this dataset.")
            foreground_class_ids = range(nclass)
        ids = tuple(foreground_class_ids)
        if not ids or any(isinstance(c, bool) or not isinstance(c, int)
                          or c < 0 or c >= nclass for c in ids):
            raise ValueError("foreground_class_ids must contain valid class IDs.")
        if len(set(ids)) != len(ids):
            raise ValueError("foreground_class_ids must not contain duplicates.")
        mask = torch.zeros(nclass, dtype=torch.bool)
        mask[list(ids)] = True
        self.register_buffer("foreground_mask", mask)

    @property
    def requires_semantic_gate(self):
        return not self.disable_gamma_k and not bool(self.foreground_mask.all())

    def get_ortho_loss(self):
        p = F.normalize(self.prototypes, dim=1)
        sim = torch.mm(p, p.t())
        return torch.triu(sim, diagonal=1).square().sum()

    def forward(self, x, mode="labeled", prototype_logits=None):
        x_norm = F.normalize(x, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        sim_map = F.conv2d(
            x_norm, proto_norm.view(self.num_prototypes, self.dim, 1, 1))
        spatial_weights = sim_map

        if self.requires_semantic_gate:
            if prototype_logits is None:
                raise ValueError("Pass prototype logits from the shared segmentation head.")
            if tuple(prototype_logits.shape) != (self.num_prototypes, self.nclass):
                raise ValueError("prototype_logits must have shape [K, nclass].")
            dominant_class = prototype_logits.argmax(dim=1)
            gamma = self.foreground_mask[dominant_class].to(spatial_weights.dtype)
            spatial_weights = spatial_weights * gamma.view(1, -1, 1, 1)

        x_proto = F.conv2d(
            spatial_weights,
            self.prototypes.view(self.num_prototypes, self.dim, 1, 1).transpose(0, 1))
        return x_proto if self.disable_residual else x + x_proto
