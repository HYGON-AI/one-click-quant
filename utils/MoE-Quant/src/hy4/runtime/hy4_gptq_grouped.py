# Adapted from the preserved rtn_eval_tp8/grouped.py; new GPTQ path only.
"""Graph-safe routed W4A8. Capacity per expert equals input token count."""
import torch
import re
import triton as tr
import triton.language as tl
from .hy4_gptq_kernel import quant

_INTEGER_EVIDENCE = None


def integer_evidence():
 return _INTEGER_EVIDENCE

@tr.jit
def route(IDS, ROWS, POS, COUNT, T:tl.constexpr, TOP:tl.constexpr, LO:tl.constexpr, B:tl.constexpr):
 e=tl.program_id(0)
 t=tl.arange(0,B)
 slot=tl.full((B,),-1,tl.int32)
 for j in range(TOP):
  v=tl.load(IDS+t*TOP+j,t<T,-1)
  slot=tl.where(v==e+LO,j,slot)
 valid=(t<T)&(slot>=0)
 p=tl.cumsum(valid.to(tl.int32))-1
 tl.store(ROWS+e*T+p,t,valid)
 tl.store(POS+t*TOP+slot,p,valid)
 tl.store(COUNT+e,tl.sum(valid.to(tl.int32),0))

@tr.jit
def grouped_mm(A,SA,W,SW,ROWS,COUNT,C,T:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
               GATHER:tl.constexpr, RAW:tl.constexpr, BN:tl.constexpr=64,BK:tl.constexpr=64):
 e=tl.program_id(0); block=tl.program_id(1); nb=tl.program_id(2)
 count=tl.load(COUNT+e)
 if block*16<count:
  m=block*16+tl.arange(0,16); n=nb*BN+tl.arange(0,BN); kk=tl.arange(0,BK)
  if GATHER:
   row=tl.load(ROWS+e*T+m,m<count,0)
  else:
   row=e*T+m
  acc=tl.full((16,BN),0,tl.int32)
  for start in range(tl.cdiv(K,BK)):
   k=start*BK+kk
   a=tl.load(A+row[:,None]*K+k[None,:],(m[:,None]<count)&(k[None,:]<K),0)
   b=tl.load(W+(e*N+n[None,:])*(K//2)+k[:,None]//2,(n[None,:]<N)&(k[:,None]<K),0).to(tl.int32)
   b=(b>>((k[:,None]%2)*4))&15
   b=tl.where(b>=8,b-16,b).to(tl.int8)
   acc=tl.dot(a,b,acc,out_dtype=tl.int32)
  if RAW:
   out=acc
  else:
   sa=tl.load(SA+row,m<count,0)
   sw=tl.load(SW+e*N+n,n<N,0)
   out=acc.to(tl.float32)*sa[:,None]*sw[None,:]
  tl.store(C+(e*T+m[:,None])*N+n[None,:],out,(m[:,None]<count)&(n[None,:]<N))

@tr.jit
def combine(Y,IDS,WEIGHTS,POS,OUT,T:tl.constexpr,H:tl.constexpr,TOP:tl.constexpr,LO:tl.constexpr,E:tl.constexpr,B:tl.constexpr):
 t=tl.program_id(0); h=tl.program_id(1)*B+tl.arange(0,B)
 acc=tl.full((B,),0,tl.float32)
 for j in range(TOP):
  e=tl.load(IDS+t*TOP+j)-LO
  p=tl.load(POS+t*TOP+j)
  valid=(e>=0)&(e<E)
  y=tl.load(Y+(e*T+p)*H+h,valid&(h<H),0).to(tl.float32)
  w=tl.load(WEIGHTS+t*TOP+j)
  acc=acc+y*w
 tl.store(OUT+t*H+h,acc,h<H)

def routed(x,ids,weights,gate_up,down,lo,limit):
 global _INTEGER_EVIDENCE
 t,h=x.shape;e,n,_=gate_up.packed.shape;i=n//2;top=ids.shape[1]
 rows=torch.empty((e,t),device=x.device,dtype=torch.int32)
 pos=torch.zeros((t,top),device=x.device,dtype=torch.int32)
 counts=torch.empty(e,device=x.device,dtype=torch.int32)
 route[(e,)](ids,rows,pos,counts,t,top,lo,tr.next_power_of_2(t))
 a,sa=quant(x,127)
 gu=torch.empty((e,t,n),device=x.device,dtype=torch.bfloat16)
 compiled=grouped_mm[(e,tr.cdiv(t,16),tr.cdiv(n,64))](a,sa,gate_up.packed,gate_up.scale,rows,counts,gu,t,n,h,True,False)
 if _INTEGER_EVIDENCE is None:
  lines=sorted(set(re.findall(r'mma[^;\n]*s32[^;\n]*',compiled.asm['ptx'])))
  if not any('.s8.s8.' in line for line in lines):raise RuntimeError('Missing grouped INT8 tensor-dot evidence')
  _INTEGER_EVIDENCE=dict(backend='Triton grouped_mm',instructions=lines,weight_bits=4,activation_bits=8,accumulator_bits=32)
 g,u=gu.chunk(2,-1)
 if limit and limit>0:g,u=g.clamp(max=limit),u.clamp(-limit,limit)
 mid,sm=quant(torch.nn.functional.silu(g)*u,127)
 y=torch.empty((e,t,h),device=x.device,dtype=torch.bfloat16)
 grouped_mm[(e,tr.cdiv(t,16),tr.cdiv(h,64))](mid,sm,down.packed,down.scale,rows,counts,y,t,h,i,False,False)
 out=torch.empty((t,h),device=x.device,dtype=torch.float32)
 combine[(t,tr.cdiv(h,256))](y,ids,weights,pos,out,t,h,top,lo,e,256,enable_fp_fusion=False)
 return out
