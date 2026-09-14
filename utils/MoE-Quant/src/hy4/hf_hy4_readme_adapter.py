"""ModelAdapter integration for the README quant.py entry, under development.

Uses the installed tool's interface, not the historical sequence controller.
Execution gates fail closed until the native stage hooks are installed.
"""
import re
from transformers import AutoModelForCausalLM
from torch import nn
from src.models.base import ModelAdapter
from .hf_hy4_mtp import HFHy4MTP, MTPState, prepare_boundary, move_mtp_state, forward_mtp
from .hf_hy4_source_loader import SourceLoader, translate
from .hf_hy4_targets import projection_views
from .hf_hy4_layer_state import prepare_embeddings, move_state, forward_block
import torch


class HFHy4ReadmeAdapter(ModelAdapter):
    name = 'hy4_hf_main_mtp'

    @classmethod
    def matches(cls, config):
        return (getattr(config, 'model_type', None) == 'hy_v4'
                and 'HYV4ForCausalLM' in (getattr(config, 'architectures', None) or []))

    def prepare_config(self, config, world_size):
        if config.num_hidden_layers != 78 or getattr(config, 'num_nextn_predict_layers', None) != 1:
            raise ValueError('Only the verified Hy4 78-main/1-MTP layout is supported')
        if world_size not in (1, 8):
            raise ValueError('Only single-rank bring-up and eight-rank layout are supported')
        # No fake EP: calibration ranks will own distinct documents, then merge
        # token-weighted statistics and assign one writer per projection.

    def build_empty_model(self, config, dtype, attn_implementation=None):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype,
            attn_implementation=attn_implementation or 'eager')
        model.model.mtp_layers = nn.ModuleList([HFHy4MTP(config).to(dtype=dtype)])
        return model.eval()

    def get_transformer_layers(self, model):
        return list(model.model.layers) + list(model.model.mtp_layers)

    def get_layer_prefix(self, block_idx):
        if 0 <= block_idx < 78:
            return f'model.layers.{block_idx}.'
        if block_idx == 78:
            return 'model.mtp_layers.0.'
        raise ValueError('Invalid Hy4 block index')

    def get_block_index_from_layer_name(self, name):
        match = re.match(r'^model\.(layers|mtp_layers)\.(\d+)(?:\.|$)', name)
        if match is None:
            raise ValueError('Invalid Hy4 checkpoint block name')
        kind, number = match.groups()
        number = int(number)
        if kind == 'mtp_layers' and number == 0:
            return 78
        if kind == 'layers' and 0 <= number < 78:
            return number
        raise ValueError('Invalid Hy4 checkpoint block index')

    def expected_quantized_layer_names(self, model):
        names = set()
        for index, block in enumerate(self.get_transformer_layers(model)):
            names.update(projection_views(block, self.get_layer_prefix(index)))
        return names

    def load_current_block(self, block, block_idx, model_path):
        return SourceLoader(model_path).load(block, self.get_layer_prefix(block_idx))

    def checkpoint_keys_for_model_key(self, model_key, weight_map):
        candidates = [source for source in weight_map if translate(source) == model_key]
        if len(candidates) != 1:
            raise ValueError('Missing or ambiguous Hy4 source: '+model_key)
        return candidates

    def validate_quantization_args(self, args):
        # These are not accepted by unpatched quant.py. Reject rather than
        # silently applying its global Linear selection or W4A16 defaults.
        required = dict(bits=4, group_size=None, sym=True, dtype='bfloat16',
            quantize_scope='routed_shared_experts', include_mtp=True,
            activation_bits=8, weight_range='narrow', rel_damp=0.01,
            block_size=128, quantization_scale='absmax', quantization_order='default')
        for name, expected in required.items():
            if getattr(args, name, None) != expected:
                raise ValueError(f'Hy4 requires {name}={expected!r}')
        if getattr(args, 'quantize_only_experts', False):
            raise ValueError('quantize_only_experts excludes required shared experts')

    def materialize_state_dict(self, *args, **kwargs):
        raise RuntimeError('Native quant.py must use load_current_block for bounded fused loading')

    def create_block_state(self, hidden_states, position_ids):
        from transformers.models.hy_v4.modeling_hy_v4 import HYV4RotaryEmbedding
        with torch.device(hidden_states.device):
            rotary = HYV4RotaryEmbedding(self.config)
        _, state = prepare_embeddings(self.config, rotary, hidden_states, position_ids)
        return state

    def move_block_state(self, block_state, device):
        if isinstance(block_state, MTPState):
            return move_mtp_state(block_state, device)
        return move_state(block_state, device)

    def prepare_mtp_document(self, model, input_ids, backbone_hidden):
        return prepare_boundary(model.model, input_ids, backbone_hidden)

    def forward_block(self, block, hidden_states, position_ids, block_state=None):
        if isinstance(block, HFHy4MTP):
            if not isinstance(block_state, MTPState):
                raise ValueError('MTP requires explicit document handoff before collection')
            return forward_mtp(block, hidden_states, position_ids, block_state)
        if block_state is None or not torch.equal(position_ids, block_state.positions):
            raise ValueError('Missing or mismatched Hy4 document state')
        if block_state.next_layer == 0 and hidden_states.ndim == 3:
            hidden_states = hidden_states.unsqueeze(2).expand(
                -1, -1, self.config.hc_mult, -1).contiguous()
        return forward_block(block, hidden_states, block_state)
