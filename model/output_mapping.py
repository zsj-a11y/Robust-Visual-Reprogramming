import torch
import torch.nn as nn

from torch.cuda.amp import autocast

import torch
from torch.nn.functional import one_hot
import torch.distributed as dist
from tqdm import tqdm
import numpy as np


def linear(input_size, output_size):
    return nn.Linear(input_size, output_size)

class LabelMappingBase(nn.Module):
    def __init__(self, mapping_sequence, map_weights=None, mapping_method="default"):
        super(LabelMappingBase, self).__init__()
        self.register_buffer("mapping_sequence", mapping_sequence)
        self.mapping_method = mapping_method

    def upgrade_mapping_sequence(self, mapping_sequence):
        self.mapping_sequence = mapping_sequence


    def forward(self, logits):
        modified_logits = logits[:, self.mapping_sequence]
        return modified_logits

def get_dist_matrix(fx, y):
    fx = one_hot(torch.argmax(fx, dim = -1), num_classes=fx.size(-1))
    dist_matrix = [fx[y==i].sum(0).unsqueeze(1) for i in range(len(y.unique()))]
    dist_matrix = torch.cat(dist_matrix, dim=1)
    return dist_matrix

def predictive_distribution_based_multi_label_mapping(dist_matrix, mlm_num: int):
    assert mlm_num * dist_matrix.size(1) <= dist_matrix.size(0), "source label number not enough for mapping"
    mapping_matrix = torch.zeros_like(dist_matrix, dtype=int)
    dist_matrix_flat = dist_matrix.flatten()
    for _ in range(mlm_num * dist_matrix.size(1)):
        loc = dist_matrix_flat.argmax().item()
        loc = [loc // dist_matrix.size(1), loc % dist_matrix.size(1)]
        mapping_matrix[loc[0], loc[1]] = 1
        dist_matrix[loc[0]] = -1
        if mapping_matrix[:, loc[1]].sum() == mlm_num:
            dist_matrix[:, loc[1]] = -1
    return mapping_matrix


def gather_x(x):
    gather_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, x)
    gathered_x = torch.cat(gather_list, dim=0)
    return gathered_x

def generate_label_mapping_by_frequency(ddp_net, data_loader, attack, rank, mapping_num = 1): # mapping_num=1: 1V1 match
    if hasattr(ddp_net, "eval"):
        ddp_net.eval()
    fx0s = []
    ys = []
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Frequency Label Mapping", ncols=100) if len(data_loader) > 20 else data_loader
    for x, y in pbar:
        x, y = x.cuda(), y.cuda()
        # inputs_adv = attack(x, y)
        with torch.no_grad():
            with autocast():
                fx0 = ddp_net.module.s_model(ddp_net.module.visual_prompt(x))
        gathered_fx0 = gather_x(fx0)
        gathered_y = gather_x(y)
        if rank == 0:
            fx0s.append(gathered_fx0.cpu())
            ys.append(gathered_y.cpu())
    if rank == 0:
        print(fx0s[0].shape, ys[0].shape)
        fx0s = torch.cat(fx0s).cpu().float()
        ys = torch.cat(ys).cpu().int()
        if ys.size(0) != fx0s.size(0):
            assert fx0s.size(0) % ys.size(0) == 0
            ys = ys.repeat(int(fx0s.size(0) / ys.size(0)))
        dist_matrix = get_dist_matrix(fx0s, ys)
        pairs = torch.nonzero(predictive_distribution_based_multi_label_mapping(dist_matrix, mapping_num))
        mapping_sequence = pairs[:, 0][torch.sort(pairs[:, 1]).indices.tolist()]
        ddp_net.module.label_mapping.upgrade_mapping_sequence(mapping_sequence.cuda())
    print(ddp_net.module.label_mapping.mapping_sequence, rank, 'aaa')
    
    dist.barrier()
    dist.broadcast(ddp_net.module.label_mapping.mapping_sequence, src=0)
        
    dist.barrier()
    print(ddp_net.module.label_mapping.mapping_sequence, rank)
    return None
