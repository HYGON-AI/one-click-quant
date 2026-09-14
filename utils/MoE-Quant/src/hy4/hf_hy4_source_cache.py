"""Advisory release of old layer source pages, not source files/global cache."""
import json
import os
import struct
from pathlib import Path


def advise_old_layer(source_root,layer,main_layers):
    root=Path(source_root).resolve()
    prefix=f'model.layers.{layer}.' if layer<main_layers else f'model.mtp_layers.{layer-main_layers}.'
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    grouped={}
    for name,filename in index.items():
        if name.startswith(prefix):
            grouped.setdefault(filename,[]).append(name)
    advised=0
    for filename,names in grouped.items():
        path=(root/filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source cache hint escapes model root')
        with path.open('rb') as stream:
            length=struct.unpack('<Q',stream.read(8))[0]
            if not 0<length<=64<<20:
                raise ValueError('Invalid source header')
            header=json.loads(stream.read(length))
            size=os.fstat(stream.fileno()).st_size
            for name in names:
                begin,end=header[name]['data_offsets']
                if not 0<=begin<end<=size-length-8:
                    raise ValueError('Invalid source byte range')
                os.posix_fadvise(stream.fileno(),8+length+begin,end-begin,os.POSIX_FADV_DONTNEED)
                advised+=end-begin
    return advised
