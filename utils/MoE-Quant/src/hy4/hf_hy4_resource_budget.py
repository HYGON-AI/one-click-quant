"""Collective, read-only budget gate before each native layer allocation."""
from pathlib import Path
import shutil
import torch.distributed as dist


def check(output,reserve_gib=300,stage_headroom_gib=50):
    error=None
    try:
        required=(reserve_gib+stage_headroom_gib)*(1<<30)
        if min(reserve_gib,stage_headroom_gib)<0 or shutil.disk_usage(output).free<required:
            raise OSError('Disk reserve/headroom exhausted before loading next layer')
        # v2 preferred, with the known v1 memory-controller layout as fallback.
        for used,maximum in ((Path('/sys/fs/cgroup/memory.current'),Path('/sys/fs/cgroup/memory.max')),
                (Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'))):
            if used.is_file() and maximum.is_file():
                limit=maximum.read_text().strip()
                if limit!='max' and int(used.read_text())>=int(limit)*.9:
                    raise MemoryError('Container memory reached 90%; next layer not allocated')
                break
    except Exception as failure:
        error=repr(failure)
    errors=[None]*dist.get_world_size(); dist.all_gather_object(errors,error)
    if any(errors):
        raise RuntimeError('Native resource gate stopped before layer allocation: '+repr(errors))
