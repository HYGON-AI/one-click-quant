import torch
import torch.distributed as dist
import argparse
import os
from src.hy4.hf_hy4_distributed_hessian import merge_to_owner

torch.set_num_threads(1)
parser = argparse.ArgumentParser()
parser.add_argument('--backend', choices=['gloo', 'nccl'], default='gloo')
args = parser.parse_args()
device = 'cpu'
if args.backend == 'nccl':
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = 'cuda:'+str(local_rank)
dist.init_process_group(args.backend)
try:
    rank, world = dist.get_rank(), dist.get_world_size()
    # Rank zero has no hits, other ranks have different token counts.
    rows = [torch.arange(r*3, dtype=torch.float32).reshape(r,3)/7 for r in range(world)]
    x = rows[rank].to(device)
    local = None if rank == 0 else 2*x.T@x/len(x)
    for owner in (0, world-1):
        merged, count = merge_to_owner(local, len(x), 3, owner, device, 'expert.gate')
        assert count == sum(range(world))
        if rank == owner:
            all_x = torch.cat(rows)
            torch.testing.assert_close(merged.cpu(), 2*all_x.T@all_x/len(all_x), rtol=1e-6, atol=1e-6)
        else:
            assert merged is None
    empty, count = merge_to_owner(None, 0, 3, 0, device, 'expert.empty')
    assert empty is None and count == 0
    try:
        merge_to_owner(None if rank==0 else local, 1 if rank==0 else len(x),
                       3, 0, device, 'expert.invalid')
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid rank was accepted')
    if rank==0:
        print('PASS: weighted Hessian merge across '+str(world)+' '+args.backend+' ranks; zero coverage and collective rejection; synthetic only', flush=True)
finally:
    dist.destroy_process_group()
