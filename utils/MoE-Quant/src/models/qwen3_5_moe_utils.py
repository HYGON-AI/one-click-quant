"""
Qwen3.5-MoE (Qwen3.8-2.4T-A95B-FP8) 适配工具。

背景
----
transformers 5.8.1 的 Qwen3_5Moe 建模把全部专家权重打包成单个 3D ``nn.Parameter``
（``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``），而 Qwen3.8-2.4T-A95B-FP8
的 checkpoint 是“逐专家、gate/up 分离”的布局
（``mlp.experts.<idx>.(gate|up|down)_proj.weight``，FP8 时还带 ``weight_scale_inv``）。
MoE-Quant 的加载 / 量化 / 专家并行都基于“每个专家是独立 ``nn.Linear``”的假设，
因此这里把模型结构转换为逐专家 MLP，使 ``state_dict`` 命名与 checkpoint 完全一致。

本模块提供：
1. 逐专家 MoE 结构 ``Qwen3_5MoeSparseMoeBlockUnpacked``；
2. 模型结构转换 + decoder layer forward 兼容层（``quant.py`` 直接调用
   ``block(inputs, position_ids=...)``，而 Qwen3.5 的 decoder layer 需要
   ``position_embeddings`` 与因果 mask，这里复刻 ``Qwen3_5MoeTextModel.forward``
   的准备逻辑，包括 mrope 计算）；
3. MTP 权重原样转存（transformers 主模型不加载 MTP，打包时需保留）。

注意：本模块只针对纯文本 ``Qwen3_5MoeForCausalLM``（model_type
``qwen3_5_moe_text``），多模态 / 其它模型不受影响。
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.activations import ACT2FN
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTopKRouter,
)


class Qwen3_5MoeExpertMLP(nn.Module):
    """单个专家：gate/up/down 三个标准 ``nn.Linear``，与 checkpoint 命名一致。"""

    def __init__(self, hidden_dim: int, intermediate_dim: int, act_fn, torch_dtype: torch.dtype):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=torch_dtype)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False, dtype=torch_dtype)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False, dtype=torch_dtype)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5MoeSparseMoeBlockUnpacked(nn.Module):
    """
    逐专家版 MoE block，forward 语义与 transformers 的 ``Qwen3_5MoeSparseMoeBlock`` 等价：
    top-k 路由 -> 命中专家逐个计算 -> 叠加经 sigmoid 门控的 shared expert。
    """

    def __init__(self, config, torch_dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.gate = Qwen3_5MoeTopKRouter(config)
        # transformers 的 router 参数默认 float32，这里显式改为量化目标 dtype，
        # 避免加载 BF16 checkpoint 后 forward 出现 dtype 不匹配。
        self.gate.weight = nn.Parameter(
            torch.empty((config.num_experts, config.hidden_size), dtype=torch_dtype)
        )
        # Expert parallelism: 全长 ModuleList，非本 rank 槽位为 None。
        # state_dict key 保持全局专家编号（与 checkpoint 一致），每个 rank 只输出
        # 自己专家的参数，quant.py 现有的"master 收集 keys -> 按 key 分发"流程直接复用。
        ep_size = getattr(config, "ep_size", 1)
        if ep_size <= 1 and dist.is_available() and dist.is_initialized():
            ep_size = dist.get_world_size()
        if ep_size > 1:
            if not (dist.is_available() and dist.is_initialized()):
                raise RuntimeError(
                    "Expert parallelism requires torch.distributed to be initialized."
                )
            assert dist.get_world_size() == ep_size, (
                f"config.ep_size ({ep_size}) must equal the distributed world size "
                f"({dist.get_world_size()})."
            )
            assert self.num_experts % ep_size == 0, (
                f"num_experts ({self.num_experts}) must be divisible by ep_size ({ep_size})."
            )
            self.ep_size = ep_size
            self.experts_per_rank = self.num_experts // ep_size
            self.ep_rank = dist.get_rank()
        else:
            self.ep_size = 1
            self.experts_per_rank = self.num_experts
            self.ep_rank = 0
        self.experts_start_idx = self.ep_rank * self.experts_per_rank
        self.experts_end_idx = self.experts_start_idx + self.experts_per_rank
        self.experts = nn.ModuleList(
            [
                (
                    Qwen3_5MoeExpertMLP(
                        self.hidden_dim, self.intermediate_dim, self.act_fn, torch_dtype
                    )
                    if self.experts_start_idx <= i < self.experts_end_idx
                    else None
                )
                for i in range(self.num_experts)
            ]
        )
        self.shared_expert = Qwen3_5MoeExpertMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            self.act_fn,
            torch_dtype,
        )
        self.shared_expert_gate = nn.Linear(
            config.hidden_size, 1, bias=False, dtype=torch_dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

        if self.ep_size > 1:
            expert_output = self._moe_infer_ep(
                hidden_states_reshaped, selected_experts, routing_weights
            )
        else:
            expert_output = self._dispatch_to_local_experts(
                hidden_states_reshaped, selected_experts, routing_weights
            )

        shared_expert_output = (
            F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        )
        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output

    def _dispatch_to_local_experts(
        self,
        hidden_states_reshaped: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """非 EP（单卡）路径：对本地全部专家逐专家 dispatch（与原实现等价）。"""
        expert_output = torch.zeros_like(hidden_states_reshaped)
        with torch.no_grad():
            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, num_tokens)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx[0])
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_reshaped[token_idx]
            current_hidden_states = self.experts[expert_idx](current_state)
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            expert_output.index_add_(0, token_idx, current_hidden_states.to(expert_output.dtype))
        return expert_output

    @torch.no_grad()
    def _moe_infer_ep(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        EP 版专家计算（all-to-all token dispatch），移植自 DeepSeek-V3 官方
        ``modeling_deepseek.py::MoE.moe_infer``：按路由专家排序 token ->
        交换计数与 token -> 本地专家计算 -> 反向交换 -> 加权求和。
        语义与单卡逐专家 dispatch 等价（浮点求和顺序除外）。
        """
        import numpy as np

        cnts = topk_ids.new_zeros((topk_ids.shape[0], self.num_experts))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)  # [num_experts]
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape

        tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
        output_splits = (
            tokens_per_expert_group.view(self.ep_size, -1)
            .sum(1)
            .cpu()
            .numpy()
            .tolist()
        )
        gathered_tokens = sorted_tokens.new_empty(
            tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
        )
        input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
        dist.all_to_all(
            list(gathered_tokens.split(output_splits)),
            list(sorted_tokens.split(input_split_sizes)),
        )
        tokens_per_expert_post_gather = tokens_per_expert_group.view(
            self.ep_size, self.experts_per_rank
        ).sum(dim=0)
        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % self.experts_per_rank
            s += k
        gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[gatherd_idxs]
        tokens_per_expert = tokens_per_expert_post_gather

        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[self.experts_start_idx + i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0, sorted_tokens.shape[1])
        new_x = torch.empty_like(outs)
        new_x[gatherd_idxs] = outs
        gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
        dist.all_to_all(
            list(gathered_tokens.split(input_split_sizes)),
            list(new_x.split(output_splits)),
        )
        outs = gathered_tokens

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


def _apply_interleaved_mrope(freqs: torch.Tensor, mrope_section) -> torch.Tensor:
    """与 transformers ``Qwen3_5MoeTextRotaryEmbedding.apply_interleaved_mrope`` 等价。"""
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def _compute_rope_embeddings(
    config, hidden_states: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    复刻 ``Qwen3_5MoeTextRotaryEmbedding.forward`` 的核心逻辑（mrope，float32 计算）。
    ``position_ids`` 形状与官方一致：2D ``[batch, seq]`` 或 3D ``[3, batch, seq]``。
    仅支持 ``rope_type == "default"``（当前模型 config 即为此类型）。
    """
    rope_params = config.rope_parameters
    theta = rope_params["rope_theta"]
    partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64)
            .to(device=hidden_states.device, dtype=torch.float)
            / dim
        )
    )
    if position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(
        3, position_ids.shape[1], -1, 1
    )
    position_ids_expanded = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    mrope_section = rope_params.get("mrope_section", [11, 11, 10])
    freqs = _apply_interleaved_mrope(freqs, mrope_section)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)


def _patch_decoder_layer_forward(model, config):
    """
    给每个 decoder layer 打 forward 兼容补丁：
    ``quant.py`` 直接调用 ``block(inputs, position_ids=...)``，而 Qwen3.5 的
    decoder layer 需要 ``position_embeddings`` 与因果 mask。这里复刻
    ``Qwen3_5MoeTextModel.forward`` 中对 decoder layer 调用前的准备逻辑。
    """
    for block in model.model.layers:
        original_forward = block.forward

        def make_forward(blk, orig):
            def forward(
                hidden_states,
                position_embeddings=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                **kwargs,
            ):
                if position_ids is not None:
                    if position_ids.ndim == 2:
                        position_ids_4 = position_ids[None, ...].expand(
                            4, position_ids.shape[0], -1
                        )
                    else:
                        position_ids_4 = position_ids
                    text_position_ids = position_ids_4[0]
                    rope_position_ids = position_ids_4[1:]
                else:
                    text_position_ids = None
                    rope_position_ids = None

                if blk.layer_type == "linear_attention":
                    # 与 TextModel.forward 一致：linear attention 层传 None mask。
                    result = orig(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=None,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        **kwargs,
                    )
                else:
                    # full_attention 层：补齐 position_embeddings 与因果 mask。
                    if position_embeddings is None:
                        position_embeddings = _compute_rope_embeddings(
                            config, hidden_states, rope_position_ids
                        )
                    if attention_mask is None:
                        attention_mask = create_causal_mask(
                            config=config,
                            inputs_embeds=hidden_states,
                            attention_mask=None,
                            past_key_values=None,
                            position_ids=text_position_ids,
                        )
                    result = orig(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        position_ids=text_position_ids,
                        past_key_values=past_key_values,
                        **kwargs,
                    )
                # quant.py 逐层更新激活时用 block(...)[0] 取 hidden_states
                # （DeepSeek 的 decoder layer 返回 tuple）；Qwen3.5 的 decoder
                # layer 返回纯 tensor，这里统一包装成 tuple 以保持兼容。
                return result if isinstance(result, tuple) else (result,)

            return forward

        block.forward = make_forward(block, original_forward)


def prepare_qwen3_5_moe_model(model, config, torch_dtype: torch.dtype) -> None:
    """
    把 Qwen3.5-MoE 模型转换为逐专家 MLP 结构并打上 decoder layer forward 兼容补丁。

    必须在 ``accelerate.init_empty_weights()`` 上下文内调用（meta 阶段，只改结构不搬数据）。
    """
    for block in model.model.layers:
        if isinstance(block.mlp, Qwen3_5MoeSparseMoeBlock):
            block.mlp = Qwen3_5MoeSparseMoeBlockUnpacked(config, torch_dtype)
    _patch_decoder_layer_forward(model, config)


def count_mtp_shards(weight_map: dict[str, str]) -> int:
    """统计 checkpoint 中 MTP 权重涉及的 safetensors shard 数。"""
    return len({fname for key, fname in weight_map.items() if key.startswith("mtp.")})


def save_mtp_weights(
    weight_dir: str,
    weight_map: dict[str, str],
    packed_model_path: str,
    next_shard_id: int,
    num_output_shards: int,
    safetensors_index: dict[str, str],
) -> int:
    """
    把 checkpoint 中的 ``mtp.*`` 权重原样转存到打包目录（key、dtype、数值均不变），
    并按输入 shard 分组保存，更新输出 safetensors index。返回下一个 shard id。
    """
    mtp_files = sorted({fname for key, fname in weight_map.items() if key.startswith("mtp.")})
    for fname in mtp_files:
        fpath = os.path.join(weight_dir, fname)
        mtp_tensors = {}
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    mtp_tensors[key] = f.get_tensor(key)
        if not mtp_tensors:
            continue
        shard_path = f"model-{next_shard_id:05}-of-{num_output_shards:05}.safetensors"
        save_file(mtp_tensors, os.path.join(packed_model_path, shard_path))
        for key in mtp_tensors:
            safetensors_index[key] = shard_path
        next_shard_id += 1
    return next_shard_id
