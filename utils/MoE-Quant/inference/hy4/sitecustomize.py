"""Explicit opt-in installation in SGLang parent and spawned interpreters."""
import os

if os.environ.get('HY4_GPTQ_ROOT'):
    try:
        from src.hy4.runtime.hy4_gptq_sglang_adapter import install
        install(os.environ['HY4_GPTQ_ROOT'])
    except BaseException:
        import traceback
        traceback.print_exc()
        # Python ignores ordinary sitecustomize exceptions. Never fall through
        # to constructing the full native BF16 expert model on adapter failure.
        os._exit(78)
