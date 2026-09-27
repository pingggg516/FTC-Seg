
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CR_BCI_Module(nn.Module):
    def __init__(self, dim, nclass=2, disable_gamma_k=False, disable_residual=False):
        super(CR_BCI_Module, self).__init__()
        self.dim = dim
        self.nclass = nclass
        self.disable_gamma_k = disable_gamma_k
        self.disable_residual = disable_residual

        self.num_prototypes = 16
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, dim))
        nn.init.orthogonal_(self.prototypes)
        
        self.proto_classifier = nn.Linear(dim, nclass)

    def get_ortho_loss(self):
        p = F.normalize(self.prototypes, dim=1)
        sim = torch.mm(p, p.t())
        I = torch.eye(self.num_prototypes, device=sim.device)
        loss = torch.norm(sim - I, p='fro')
        return loss

    def forward(self, x, mode='labeled', classifier_weights=None):
        B, C, H, W = x.shape
        N = H * W
        
        x_norm = F.normalize(x, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)
        
        sim_map = F.conv2d(x_norm, proto_norm.view(self.num_prototypes, self.dim, 1, 1))
        
        proto_class_score = self.proto_classifier(self.prototypes)
        
        suppression_factor = None
        if (not self.disable_gamma_k) and proto_class_score.shape[1] > 1:
            foreground_score = proto_class_score[:, 1:].max(dim=1)[0]
            suppression_factor = F.relu(foreground_score).view(1, self.num_prototypes, 1, 1)
        
        spatial_weights = F.softmax(sim_map / 0.1, dim=1)
        if suppression_factor is not None:
            spatial_weights = spatial_weights * suppression_factor
        
        x_proto = F.conv2d(spatial_weights, self.prototypes.view(self.num_prototypes, self.dim, 1, 1).transpose(0, 1))
        
        global_sim_max = F.adaptive_max_pool2d(sim_map, 1).flatten(1)
        if suppression_factor is not None:
            global_sim_max = global_sim_max * suppression_factor.squeeze(-1).squeeze(-1)

        filters = torch.mm(global_sim_max, self.prototypes)
        channel_weights = torch.sigmoid(filters).unsqueeze(2).unsqueeze(3)
        
        if self.disable_residual:
            x_recal = x_proto
        else:
            x_recal = x + x_proto
        
        return x_recal

