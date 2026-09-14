# Adapted from the preserved rtn_eval_tp8/resident.py; new GPTQ path only.
"""Contiguous expert ownership; packed weights stay on the owning GPU."""
import torch
from torch import nn


def bounds(experts,rank,world):
    if experts%world or not 0<=rank<world:raise ValueError('Invalid expert partition')
    n=experts//world
    return rank*n,(rank+1)*n


class Bank(nn.Module):
    def __init__(self,checkpoint,base,device,selection=None):
        super().__init__()
        p=checkpoint.tensor(base+'.int4_packed',selection)
        s=checkpoint.tensor(base+'.scale',selection)
        a=checkpoint.tensor(base+'.input_scale',selection)
        if p.dtype!=torch.uint8 or s.dtype!=torch.float32 or a.dtype!=torch.float32:
            raise ValueError('Packed dtype mismatch')
        if p.shape[:-1]!=s.shape or a.shape!=p.shape[:-2]+(p.shape[-1]*2,):
            raise ValueError('Packed shape mismatch')
        if not torch.isfinite(s).all() or (s<=0).any() or not (a==1).all():
            raise ValueError('Invalid scales')
        self.register_buffer('packed',p.to(device).contiguous())
        self.register_buffer('scale',s.to(device).contiguous())

    def weights(self,expert=None):
        return (self.packed,self.scale) if expert is None else (self.packed[expert],self.scale[expert])
