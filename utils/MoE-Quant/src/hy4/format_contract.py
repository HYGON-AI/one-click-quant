"""Version-one on-disk INT4 contract; not SGLang registration."""
import torch

FORMAT = 'hy4_w4a8_v1'

def pack(q):
    if q.dtype != torch.int8 or q.ndim != 2 or bool(((q < -7) | (q > 7)).any()):
        raise ValueError('Expected narrow signed INT4 matrix')
    if q.shape[1] % 2:
        q = torch.nn.functional.pad(q, (0, 1))
    return ((q[:, ::2].to(torch.uint8) & 15) |
            ((q[:, 1::2].to(torch.uint8) & 15) << 4)).contiguous()

def unpack(packed, width):
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] != (width+1)//2:
        raise ValueError('Invalid packed shape')
    q=torch.stack((packed & 15, packed >> 4),dim=-1).reshape(packed.shape[0],-1).to(torch.int8)
    q=torch.where(q>=8,q-16,q)[:,:width]
    if bool((q == -8).any()):
        raise ValueError('Reserved narrow-range nibble')
    return q

def narrow_grid(weight):
    scale=weight.float().abs().amax(dim=-1,keepdim=True)/7
    scale=torch.where(scale==0,torch.ones_like(scale),scale)
    return scale, torch.full_like(scale,7), torch.full_like(scale,14)

if __name__ == '__main__':
    for width in (1,2,3,127,128,129):
        q=(torch.arange(3*width).reshape(3,width)%15-7).to(torch.int8)
        assert torch.equal(unpack(pack(q),width),q)
    s,z,m=narrow_grid(torch.zeros(3,5,dtype=torch.bfloat16))
    assert s.dtype==torch.float32 and bool((s==1).all())
    print('FORMAT_FIXTURE_PASSED_NOT_GPU_OR_MODEL_VALIDATION')
