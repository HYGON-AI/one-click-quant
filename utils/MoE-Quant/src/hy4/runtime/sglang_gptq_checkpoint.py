"""Strict full-model GPTQ reader; never presents GPTQ as RTN metadata."""
import json
from pathlib import Path
from src.hy4.checkpoint_loader import CheckpointLoader
from src.hy4.checkpoint_writer import digest
from src.hy4.sglang_full_inventory import validate_full_state
from .sglang_bank_view import PackedBankView


class Checkpoint(PackedBankView):
    def __init__(self, root, source_index_path=None):
        root = Path(root).resolve()
        source_index_path = source_index_path or root/'hy4-source-model.index.json'
        state = json.loads((root/'hy4-checkpoint.index.json').read_text())
        config_path = root/'config.json'
        identity = state['identity']
        if digest(source_index_path) != identity['source_index_sha256']:
            raise ValueError('Wrong original source inventory')
        if digest(config_path) != identity['source_config_sha256']:
            raise ValueError('Model config differs from quantization source')
        config = json.loads(config_path.read_text())
        source = json.loads(Path(source_index_path).read_text())
        expected, physical = validate_full_state(state, config, source)
        # Performs payload checks. Do not instantiate separately in eight ranks
        # until a safe preflight/verified-handle sharing path is implemented.
        loader = CheckpointLoader(root, identity, expected)
        super().__init__(loader, config['n_routed_experts'])
        self.root = root
        self.index_sha256 = digest(root/'hy4-checkpoint.index.json')
        self.targets = physical
        self.meta = dict(format=state['format'], identity=identity,
                         accuracy='NOT_EVALUATED')

    def retained(self, mtp=False):
        for name, record in self.loader.state['parameters'].items():
            if record['kind'] == 'retained' and name.startswith('model.mtp_layers.') == mtp:
                yield name, self.loader.retained(name)
