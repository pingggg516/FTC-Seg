import os
import subprocess

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def setup_distributed(backend="nccl", port=None):
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port) if port is not None else "12359"
        os.environ["LOCAL_RANK"] = "0"
    
    if num_gpus > 0:
        torch.cuda.set_device(rank % num_gpus)

    if os.name == 'nt' and backend == 'nccl':
        backend = 'gloo'
        print("Warning: Windows does not support NCCL, switched to GLOO.")

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size


class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        distributed = dist.is_available() and dist.is_initialized()
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else (
            dist.get_world_size() if distributed else 1)
        self.rank = rank if rank is not None else (dist.get_rank() if distributed else 0)
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError('Invalid evaluation sampler rank or world size.')

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.num_replicas - 1) // self.num_replicas)


@torch.no_grad()
def update_atc_statistics(p_l, p_u, acc_l, labels, predictions,
                          pseudo_labels, ignore_mask, momentum=0.999):
    nclass = p_l.numel()
    valid_l = labels != 255
    valid_u = ignore_mask != 255
    counts = torch.stack((
        torch.bincount(labels[valid_l], minlength=nclass),
        torch.bincount(pseudo_labels[valid_u], minlength=nclass),
        torch.bincount(labels[valid_l & (predictions == labels)], minlength=nclass),
    ))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
    labeled, unlabeled, correct = counts.float()
    if labeled.sum() > 0:
        current = labeled / (labeled.sum() + 1e-6)
        p_l = momentum * p_l + (1 - momentum) * current
    if unlabeled.sum() > 0:
        current = unlabeled / (unlabeled.sum() + 1e-6)
        p_u = momentum * p_u + (1 - momentum) * current
    current_acc = torch.where(labeled > 0, correct / labeled.clamp_min(1), acc_l)
    acc_l = momentum * acc_l + (1 - momentum) * current_acc
    return p_l, p_u, acc_l
