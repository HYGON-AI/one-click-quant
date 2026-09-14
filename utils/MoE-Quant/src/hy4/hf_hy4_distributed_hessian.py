"""Collectively validate and merge native GPTQ H=2*X.T@X/token_count."""
import torch
import torch.distributed as dist


def merge_to_owner(hessian, count, width, owner, device, projection):
    """All ranks call in the same projection order; only owner gets merged H.

    Device must match the process-group backend. No equal-rank averaging and
    no fabricated samples for uncovered experts. H is not mutated in place.
    """
    if not dist.is_initialized():
        raise RuntimeError('Initialized process group required')
    signature = (projection, width, owner)
    signatures = [None] * dist.get_world_size()
    dist.all_gather_object(signatures, signature)
    if any(value != signature for value in signatures):
        raise ValueError('Ranks disagree on projection merge order/shape/owner')
    valid = (type(count) is int and count >= 0 and type(width) is int and width > 0
             and type(owner) is int and 0 <= owner < dist.get_world_size())
    if valid:
        if count == 0:
            valid = hessian is None
        else:
            valid = (isinstance(hessian, torch.Tensor) and hessian.dtype == torch.float32
                and hessian.shape == (width, width) and bool(torch.isfinite(hessian).all()))
    error = torch.tensor([not valid], dtype=torch.int32, device=device)
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    if error.item():
        raise ValueError('Invalid Hessian on one or more ranks')
    total = torch.tensor([count], dtype=torch.int64, device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    total_count = int(total.item())
    if not total_count:
        return None, 0
    # Weight local normalized H by its actual token count before summing.
    value = (torch.zeros((width, width), dtype=torch.float32, device=device)
             if count == 0 else hessian.to(device).clone().mul_(count))
    dist.reduce(value, dst=owner, op=dist.ReduceOp.SUM)
    if dist.get_rank() == owner:
        return value.div_(total_count), total_count
    return None, total_count
