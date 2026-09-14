# Adapted from the preserved rtn_eval_tp8/kernel.py; new GPTQ path only.
"""Experimental H20 INT4-storage / INT8-compute expert prototype."""
import argparse, hashlib, json, math, os, platform, statistics, subprocess
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors.torch import load_file


@triton.jit
def mm_kernel(A, W, SA, SW, C, M, N:tl.constexpr, K:tl.constexpr,
              RAW:tl.constexpr, BM:tl.constexpr=16, BN:tl.constexpr=64, BK:tl.constexpr=64):
    m = tl.program_id(0)*BM+tl.arange(0,BM)
    n = tl.program_id(1)*BN+tl.arange(0,BN)
    kk = tl.arange(0,BK)
    acc = tl.full((BM,BN),0,tl.int32)
    for i in range(tl.cdiv(K,BK)):
        k = i*BK+kk
        a = tl.load(A+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0)
        b = tl.load(W+n[None,:]*tl.cdiv(K,2)+k[:,None]//2,
                    (n[None,:]<N)&(k[:,None]<K),0).to(tl.int32)
        b = (b >> ((k[:,None]%2)*4)) & 15
        b = tl.where(b>=8,b-16,b).to(tl.int8)
        acc = tl.dot(a,b,acc,out_dtype=tl.int32)
    if RAW:
        out = acc
    else:
        sa = tl.load(SA+m,m<M,0)
        sw = tl.load(SW+n,n<N,0)
        out = acc.to(tl.float32)*sa[:,None]*sw[None,:]
    tl.store(C+m[:,None]*N+n[None,:],out,(m[:,None]<M)&(n[None,:]<N))


def quant(x, bound):
    scale = x.float().abs().amax(-1,keepdim=True)/bound
    scale = torch.where(scale==0,torch.ones_like(scale),scale)
    return torch.round(x.float()/scale).clamp(-bound,bound).to(torch.int8).contiguous(),scale.flatten().contiguous()


def pack(q):
    if q.shape[1]%2: q=F.pad(q,(0,1))
    return ((q[:,::2].to(torch.int16)&15)|((q[:,1::2].to(torch.int16)&15)<<4)).to(torch.uint8).contiguous()


def unpack(w,k):
    lo=(w&15).to(torch.int8); hi=(w>>4).to(torch.int8)
    q=torch.stack((lo,hi),-1).flatten(1)[:,:k]
    return torch.where(q>=8,q-16,q).to(torch.int8)


COMPILED=[]
def mm(a,w,sa,sw,raw=False):
    m,k=a.shape; n=w.shape[0]
    out=torch.empty((m,n),device=a.device,dtype=torch.int32 if raw else torch.bfloat16)
    kernel=mm_kernel[(triton.cdiv(m,16),triton.cdiv(n,64))](a,w,sa,sw,out,m,n,k,raw)
    if not COMPILED or (not raw and len(COMPILED)==1): COMPILED.append(kernel)
    return out


def projection(x,w,s):
    a,sa=quant(x,127)
    return mm(a,w,sa,s)
