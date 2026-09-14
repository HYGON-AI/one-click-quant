import os
import gc
import argparse
import time
from dataclasses import dataclass

from tqdm import tqdm
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer

try:
    import wandb
    wandb_enabled = True
except:
    wandb_enabled = False


from src import dist_utils, data_utils, model_utils, quant_utils, loading_utils, gptq
from src.models import get_model_adapter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize_scope', choices=['routed_shared_experts'])
    parser.add_argument('--include_mtp', action='store_true')
    parser.add_argument('--activation_bits', type=int, choices=[8])
    parser.add_argument('--weight_range', choices=['narrow'])
    parser.add_argument('--calibration_manifest')
    parser.add_argument('--hessian_budget_gib', type=float, default=44.0)
    parser.add_argument('--stop_after_layer', type=int, default=None,
                        help='Commit this layer and stop; incomplete candidate, not a full model.')
    parser.add_argument('--gpu_memory_budget_gib', type=float, default=64.0)
    parser.add_argument('--disk_reserve_gib', type=float, default=300.0)
    parser.add_argument('--stage_disk_headroom_gib', type=float, default=50.0)
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the DeepSeek model",
    )
    # Data params
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="The name or path to calibration dataset",
    )
    parser.add_argument("--num_calibration_samples", default=128, type=int, help="Number of samples for calibration.")
    parser.add_argument("--max_sequence_length", default=8192, type=int, help="Calibration sequence length.")
    # Quantization params
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bitwidth. Each adapter re-validates the bit widths it supports.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help=(
            "Weight quantization granularity. Omit this option for channel-wise "
            "quantization (one scale per output channel); specify a positive "
            "integer to quantize groups of that many input columns. The value "
            "must divide each quantized layer's input dimension."
        ),
    )
    parser.add_argument("--sym", action="store_true", help="Whether to use symmetric quantization")
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--quantization_scale", type=str, default="absmax", choices=["absmax", "mse"])
    parser.add_argument("--quantization_order", type=str, default="default", choices=["default", "activation"])
    parser.add_argument(
        "--quantize_only_experts",
        default=False,
        action="store_true",
        help="Whether to quantize only routed (non-shared) experts.",
    )
    # Save params
    parser.add_argument("--save_dir", type=str, default=None, help="where to save quantized model.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Log to W&B")
    parser.add_argument("--log_error", default=False, action="store_true", help="Whether to log relative L2 error")
    # Misc params
    parser.add_argument("--offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--tie_gptq_handles", action="store_true", help="whether to reuse hessian between gate and up projections.")
    parser.add_argument("--resume", action="store_true", help="whether to resume quantization from latest checkpoint.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--dtype", default="float16", type=str, choices=["float16", "bfloat16"], help="Torch dtype used."
    )
    parser.add_argument(
        "--attn_implementation",
        "--attn-implementation",
        dest="attn_implementation",
        type=str,
        default="flash_attention_2",
        metavar="BACKEND",
        help=(
            "Attention backend passed to Transformers; examples are "
            "flash_attention_2, eager, and sdpa."
        ),
    )
    return parser.parse_args()


def get_resume_block_idx(save_dir: os.PathLike, adapter) -> int:
    resume_block_idx = 0
    if os.path.exists(save_dir):
        for layer_name in os.listdir(save_dir):
            if not os.path.isdir(os.path.join(save_dir, layer_name)):
                continue
            block_idx = adapter.get_block_index_from_layer_name(layer_name)
            resume_block_idx = max(resume_block_idx, block_idx)
    return resume_block_idx


@dataclass
class RuntimeContext:
    world_size: int
    rank: int
    device: str
    offload_device: str | None
    dtype: torch.dtype


@dataclass
class ModelContext:
    config: object
    adapter: object
    model: object
    tokenizer: object


@dataclass
class CheckpointContext:
    weight_dir: str
    param_buffer: dict
    weight_map: dict | None = None
    loaded_shards: set | None = None


@dataclass
class CalibrationContext:
    dataset: list
    num_seq_per_rank: int
    inputs: list
    position_ids: list
    block_states: list
    frozen_identity: dict | None = None


def initialize_runtime(args):
    if dist.is_available() and all(var in os.environ for var in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")):
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    if args.group_size is None:
        dist_utils.print_on_main("[INFO] weight quantization strategy=channel-wise (one scale per output channel)")
    else:
        dist_utils.print_on_main(f"[INFO] weight quantization strategy=group-wise, group_size={args.group_size}")
    device = f"cuda:{rank}"
    torch.set_grad_enabled(False)
    torch.cuda.set_device(device)
    if args.gpu_memory_budget_gib <= 0:
        raise ValueError('Positive GPU memory budget required')
    capacity = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0,args.gpu_memory_budget_gib*(1<<30)/capacity),device)
    torch.manual_seed(args.seed)
    offload_device = "cpu" if args.offload_activations else None
    dtype = getattr(torch, args.dtype)
    # Init W&B logger
    if args.log_wandb and dist_utils.is_main():
        assert wandb_enabled, "wandb not installed. try `pip install wandb`"
        wandb.init(config=args)
    return RuntimeContext(world_size, rank, device, offload_device, dtype)


def build_model_context(args, runtime):
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    adapter = get_model_adapter(config)
    print(f"[INFO] model adapter={adapter.name}")
    adapter.validate_quantization_args(args)
    adapter.prepare_config(config, runtime.world_size)
    with init_empty_weights():
        model = adapter.build_empty_model(config, runtime.dtype, attn_implementation=args.attn_implementation).eval()
        model.config.use_cache = False
        adapter.prepare_model(model, config, runtime.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    return ModelContext(config, adapter, model, tokenizer)


def prepare_calibration_context(args, runtime, model_context):
    print("[INFO] Preparing calibration dataset...")
    if getattr(model_context.adapter, 'name', None) == 'hy4_hf_main_mtp':
        from src.hy4.hf_hy4_frozen_data import load
        if not args.calibration_manifest:
            raise ValueError('Hy4 requires --calibration_manifest')
        dataset, identity = load(args.calibration_manifest, args.model_name_or_path,
            args.dataset_name_or_path, runtime.rank, runtime.world_size,
            args.num_calibration_samples, args.max_sequence_length)
        dist_utils.barrier(device_ids=[runtime.rank])
        return CalibrationContext(dataset, len(dataset), [], [], [], identity)
    dataset = data_utils.prepare_calibration_dataset(
        args.dataset_name_or_path, model_context.tokenizer, args.max_sequence_length,
        args.num_calibration_samples, args.seed
    )
    print(f"[INFO] Calibration dataset prepared, {len(dataset)} sequences.")
    num_seq_per_rank = len(dataset) // runtime.world_size
    dataset = dataset[runtime.rank * num_seq_per_rank : (runtime.rank + 1) * num_seq_per_rank]
    dist_utils.barrier(device_ids=[runtime.rank])
    return CalibrationContext(dataset, num_seq_per_rank, [], [], [])


def prepare_checkpoint_context(args, runtime, model_context):
    checkpoint = CheckpointContext(args.model_name_or_path, {})
    if getattr(model_context.adapter, 'name', None) == 'hy4_hf_main_mtp':
        # Each rank loads only the current final storage; no full shard buffer.
        return checkpoint
    if dist_utils.is_main():
        checkpoint.weight_map = loading_utils.load_safetensors_weight_map(checkpoint.weight_dir)
        checkpoint.loaded_shards = set()
        loaded = loading_utils.ensure_model_params_loaded(
            checkpoint.weight_dir, checkpoint.param_buffer,
            [model_context.adapter.embedding_weight_key()], checkpoint.weight_map,
            model_context.adapter, checkpoint.loaded_shards,
        )
        dist_utils.print_on_main(f"Loaded embedding parameter from shards: {loaded}")
    dist_utils.barrier(device_ids=[runtime.rank])
    return checkpoint


def initialize_embeddings(args, runtime, model_context, calibration, checkpoint):
    adapter = model_context.adapter
    embedding_module = adapter.get_embedding_module(model_context.model)
    embedding_module.to_empty(device=runtime.device)
    if getattr(adapter, 'name', None) == 'hy4_hf_main_mtp':
        from src.hy4.hf_hy4_source_loader import SourceLoader
        SourceLoader(args.model_name_or_path).load(embedding_module, 'model.embed_tokens.')
        for tokens in calibration.dataset:
            calibration.inputs.append(embedding_module(tokens.to(runtime.device)).to(runtime.offload_device))
            calibration.position_ids.append(torch.arange(tokens.shape[1], device=runtime.device).unsqueeze(0))
        embedding_module.to(device='meta')
        calibration.block_states = [adapter.create_block_state(hidden, positions.to(hidden.device))
            for hidden, positions in zip(calibration.inputs, calibration.position_ids)]
        return
    if dist_utils.is_main():
        embedding_state_dict = loading_utils.materialize_model_state_dict(
            checkpoint.param_buffer,
            [adapter.embedding_weight_key()],
            checkpoint.weight_map,
            adapter,
            runtime.dtype,
            expected_shapes={adapter.embedding_weight_key(): tuple(embedding_module.weight.shape)},
        )
        embedding_module.weight.data = embedding_state_dict[adapter.embedding_weight_key()].to(
            device=runtime.device, dtype=runtime.dtype
        )
    if dist_utils.is_dist_available_and_initialized():
        dist_utils.broadcast_parameters(embedding_module)
    for i in range(calibration.num_seq_per_rank):
        seq_length = calibration.dataset[i].shape[1]
        calibration.inputs.append(
            embedding_module(calibration.dataset[i].to(runtime.device)).to(runtime.offload_device)
        )
        calibration.position_ids.append(
            torch.arange(0, seq_length, dtype=torch.long, device=runtime.device).unsqueeze(0)
        )
    embedding_module.to(device="meta")
    checkpoint.param_buffer.pop(adapter.embedding_weight_key(), None)
    checkpoint.param_buffer.pop(adapter.embedding_weight_key() + "_scale_inv", None)
    calibration.block_states = [
        adapter.create_block_state(calibration.inputs[i], calibration.position_ids[i])
        for i in range(calibration.num_seq_per_rank)
    ]


def collect(block, prefix, model, args, adapter, inputs, position_ids, block_states, device, rank):
    if getattr(adapter, 'name', None) == 'hy4_hf_main_mtp':
        from src.hy4.hf_hy4_collect import collect as collect_hy4
        def forward_documents():
            for hidden, positions, state in zip(inputs, position_ids, block_states):
                adapter.forward_block(block, hidden.to(device), positions.to(device),
                                      adapter.move_block_state(state, device))
        handles = collect_hy4(block, prefix, args, forward_documents,
                             int(args.hessian_budget_gib * (1 << 30)))
        dist_utils.barrier(device_ids=[rank])
        return handles, {}
    layers = model_utils.select_layers(model, prefix, ".*", model_utils.LINEAR_LAYERS)
    handles = {}
    hooks = {}

    def update_handle_hook(name):
        def _hook(_, inp, out):
            handles[name].update(inp[0])

        return _hook

    for layer_name, layer in layers.items():
        if args.quantize_only_experts and not adapter.is_routed_expert(layer_name):
            continue
        tied_gptq_handle = None
        if args.tie_gptq_handles:
            tied_layer_name = adapter.get_tied_gptq_source(layer_name)
            if tied_layer_name is not None:
                if tied_layer_name not in handles:
                    raise KeyError(
                        f"Cannot tie GPTQ handle {layer_name!r}: source "
                        f"{tied_layer_name!r} has not been created."
                    )
                tied_gptq_handle = handles[tied_layer_name]
        handles[layer_name] = gptq.GPTQ(
            layer, args.group_size, args.sym, args.rel_damp, args.block_size,
            args.quantization_order, args.quantization_scale,
            is_distributed=not adapter.is_routed_expert(layer_name),
            tied_gptq_handle=tied_gptq_handle,
        )
        if tied_gptq_handle is None:
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))

    for i in range(len(inputs)):
        adapter.forward_block(
            block, inputs[i].to(device), position_ids[i].to(device),
            adapter.move_block_state(block_states[i], device),
        )
    for hook in hooks.values():
        hook.remove()
    dist_utils.barrier(device_ids=[rank])
    return handles, hooks


def quantize(handle, bits):
    qweight, scale, zero = handle.quantize(bits)
    dequantized_weight = quant_utils.dequantize_linear_weight(qweight, scale, zero)
    assert torch.isfinite(dequantized_weight).all().item(), "weight is broken after quantization."
    return qweight, scale, zero, dequantized_weight


def propagate(block, adapter, inputs, position_ids, block_states, device, offload_device):
    for i in range(len(inputs)):
        next_inputs, next_state = adapter.forward_block(
            block, inputs[i].to(device), position_ids[i].to(device),
            adapter.move_block_state(block_states[i], device),
        )
        inputs[i] = next_inputs.to(offload_device)
        block_states[i] = adapter.move_block_state(next_state, offload_device)
        assert torch.isfinite(inputs[i]).all().item(), "NaN of inf encountered."


def release(block, prefix, checkpoint, runtime):
    block.to(device="meta")
    if dist_utils.is_main():
        for key in list(checkpoint.param_buffer):
            if key.startswith(prefix):
                checkpoint.param_buffer.pop(key, None)
    torch.cuda.empty_cache()
    gc.collect()


def save_quantization_metadata(args):
    if args.save_dir:
        torch.save(
            {"bits": args.bits, "group_size": args.group_size, "quantize_only_experts": args.quantize_only_experts},
            os.path.join(args.save_dir, "metadata.pt"),
        )


def cleanup_runtime():
    if dist_utils.is_dist_available_and_initialized():
        dist.destroy_process_group()


def run(args):
    runtime = initialize_runtime(args)
    model_context = build_model_context(args, runtime)
    if getattr(model_context.adapter, 'name', None) == 'hy4_hf_main_mtp':
        if args.stop_after_layer is not None and not 0 <= args.stop_after_layer < 79:
            raise ValueError('stop_after_layer must be within the 78+1 Hy4 layer sequence')
        print('[Hy4 native] candidate conversion only; full runtime/accuracy NOT_EVALUATED',flush=True)
    calibration = prepare_calibration_context(args, runtime, model_context)
    checkpoint = prepare_checkpoint_context(args, runtime, model_context)
    return process(args, runtime, model_context, calibration, checkpoint)


def prepare_block_keys(block_idx, block, prefix, world_size, checkpoint, adapter):
    rank_block_keys = [k for k in block.state_dict()]
    if dist_utils.is_main():
        block_keys_with_prefix = [f"{prefix}{k}" for k in rank_block_keys]
        other_ranks_keys = []
        for i in range(1, world_size):
            other_rank_keys = [None for _ in rank_block_keys]
            dist.recv_object_list(other_rank_keys, src=i)
            block_keys_with_prefix.extend([f"{prefix}{k}" for k in other_rank_keys])
            other_ranks_keys.append(other_rank_keys)
        block_keys_with_prefix = set(block_keys_with_prefix)
    else:
        block_keys_with_prefix = []
        other_ranks_keys = []
        dist.send_object_list(rank_block_keys, dst=0)

    has_routed_experts = adapter.has_routed_experts(rank_block_keys)

    if dist_utils.is_main():
        loaded = loading_utils.ensure_model_params_loaded(
            checkpoint.weight_dir,
            checkpoint.param_buffer,
            block_keys_with_prefix,
            checkpoint.weight_map,
            adapter,
            checkpoint.loaded_shards,
        )
        if loaded:
            dist_utils.print_on_main(f"Loaded block {block_idx} parameters from shards: {loaded}")

    return rank_block_keys, block_keys_with_prefix, other_ranks_keys, has_routed_experts


def load_dense_block(block, prefix, block_keys_with_prefix, checkpoint, adapter, dtype):
    if dist_utils.is_main():
        expected_shapes = {
            f"{prefix}{key}": tuple(tensor.shape)
            for key, tensor in block.state_dict().items()
        }
        materialized_block = loading_utils.materialize_model_state_dict(
            checkpoint.param_buffer,
            block_keys_with_prefix,
            checkpoint.weight_map,
            adapter,
            dtype,
            expected_shapes=expected_shapes,
        )
        block_state_dict = {
            key[len(prefix):]: tensor
            for key, tensor in materialized_block.items()
            if key.startswith(prefix)
        }
        block.load_state_dict(block_state_dict)
        del materialized_block, block_state_dict
    if dist_utils.is_dist_available_and_initialized():
        dist_utils.broadcast_parameters(block)


def load_moe_block(
    block, prefix, rank_block_keys, other_ranks_keys, checkpoint, adapter, dtype, device
):
    if dist_utils.is_main():
        rank_key_sets = [rank_block_keys] + other_ranks_keys
        for target_rank, target_keys in enumerate(rank_key_sets):
            target_model_keys = [f"{prefix}{key}" for key in target_keys]
            expected_shapes = None
            if target_rank == 0:
                expected_shapes = {
                    f"{prefix}{key}": tuple(tensor.shape)
                    for key, tensor in block.state_dict().items()
                }
            materialized_rank = loading_utils.materialize_model_state_dict(
                checkpoint.param_buffer,
                target_model_keys,
                checkpoint.weight_map,
                adapter,
                dtype,
                expected_shapes=expected_shapes,
            )
            rank_state_dict = {
                key[len(prefix):]: tensor
                for key, tensor in materialized_rank.items()
            }
            if target_rank == 0:
                block.load_state_dict(rank_state_dict)
            else:
                local_state_dict = block.state_dict()
                for key in target_keys:
                    tensor = rank_state_dict[key]
                    target_tensor = local_state_dict.get(key)
                    if target_tensor is not None and tensor.shape != target_tensor.shape:
                        raise RuntimeError(
                            f"Shape mismatch for {key}: "
                            f"send={tuple(tensor.shape)}, "
                            f"recv={tuple(target_tensor.shape)}"
                        )
                    send_dtype = tensor.dtype if target_tensor is None else target_tensor.dtype
                    dist.send(tensor.to(device=device, dtype=send_dtype), dst=target_rank)
                del local_state_dict
            del materialized_rank, rank_state_dict
            gc.collect()
    else:
        rank_state_dict = block.state_dict()
        for key in rank_state_dict:
            dist.recv(rank_state_dict[key], src=0)
        block.load_state_dict(rank_state_dict)
        del rank_state_dict


def quantize_shared_handle(handle_name, handle, args):
    dist_utils.print_on_main(f"Quantizing layer {handle_name}")
    qweight, scale, zero, dequantized_weight = quantize(handle=handle, bits=args.bits)

    if args.log_error:
        if handle.has_hessian_issues():
            dist_utils.print_on_main(
                "An issue occured on Hessian computation. Output error cannot be estimated."
            )
        else:
            relative_mse = quant_utils.get_relative_mse_error(
                dequantized_weight.float(), handle.layer.weight.float(), handle.H
            )
            dist_utils.print_on_main(f"Relative error: {relative_mse.item():.2e}")
            if args.log_wandb and dist_utils.is_main():
                wandb.log({f"relative_error/{handle_name}": relative_mse.item()}, step=0)

    if args.save_dir and dist_utils.is_main():
        os.makedirs(os.path.join(args.save_dir, handle_name), exist_ok=True)
        torch.save(
            {"qweight": qweight, "scale": scale, "zero": zero},
            os.path.join(args.save_dir, handle_name, "quantized_weight.pt"),
        )
    handle.layer.weight.data = dequantized_weight
    handle.reset()
    return (
        handle.issue_zero_samples,
        handle.issue_nan_hessian,
        handle.issue_non_invertible,
    )


def quantize_expert_handle(handle_name, handle, args):
    message = f"Quantizing layer {handle_name}\n"
    qweight, scale, zero, dequantized_weight = quantize(handle, args.bits)
    message += f"Tokens collected: {handle.tokens_collected}.\n"

    if args.log_error:
        if handle.has_hessian_issues():
            message += "Hessian issue. Output error cannot be estimated.\n"
        else:
            relative_mse = quant_utils.get_relative_mse_error(
                dequantized_weight.float(), handle.layer.weight.float(), handle.H
            )
            message += f"Relative error: {relative_mse.item():.2e}\n"
            if args.log_wandb and dist_utils.is_main():
                wandb.log({f"relative_error/{handle_name}": relative_mse.item()}, step=0)

    if args.save_dir:
        os.makedirs(os.path.join(args.save_dir, handle_name), exist_ok=True)
        torch.save(
            {"qweight": qweight, "scale": scale, "zero": zero},
            os.path.join(args.save_dir, handle_name, "quantized_weight.pt"),
        )
    handle.layer.weight.data = dequantized_weight
    handle.reset()
    return message, handle.issue_zero_samples, handle.issue_nan_hessian, handle.issue_non_invertible


def process(args, runtime=None, model_context=None, calibration=None, checkpoint=None):
    if runtime is None:
        runtime = initialize_runtime(args)
    if model_context is None:
        model_context = build_model_context(args, runtime)
    if calibration is None:
        calibration = prepare_calibration_context(args, runtime, model_context)
    if checkpoint is None:
        checkpoint = prepare_checkpoint_context(args, runtime, model_context)
    world_size = runtime.world_size
    rank = runtime.rank
    device = runtime.device
    offload_device = runtime.offload_device
    dtype = runtime.dtype
    adapter = model_context.adapter
    model = model_context.model
    hy4_native = getattr(adapter, 'name', None) == 'hy4_hf_main_mtp'
    run_identity = None
    if hy4_native:
        from src.hy4.hf_hy4_run_state import prepare as prepare_run
        run_identity, resume_block_idx = prepare_run(args, calibration, model.config.num_hidden_layers, __file__)
    else:
        resume_block_idx = get_resume_block_idx(args.save_dir, adapter) if args.resume else 0
    if not hy4_native or resume_block_idx == 0:
        initialize_embeddings(args, runtime, model_context, calibration, checkpoint)
    inputs = calibration.inputs
    position_ids = calibration.position_ids
    block_states = calibration.block_states

    transformer_layers = adapter.get_transformer_layers(model)
    for block_idx, block in tqdm(
        enumerate(transformer_layers), desc="Processing transformer blocks", total=len(transformer_layers)
    ):
        if hy4_native and block_idx < resume_block_idx:
            continue
        if hy4_native and args.stop_after_layer is not None and block_idx > args.stop_after_layer:
            break
        prefix = adapter.get_layer_prefix(block_idx)
        if hy4_native:
            from src.hy4.hf_hy4_resource_budget import check as check_resource_budget
            check_resource_budget(args.save_dir,args.disk_reserve_gib,args.stage_disk_headroom_gib)
        stage_start=time.monotonic()
        if hy4_native:
            print(f'[Hy4 native rank {rank}] layer={block_idx} stage=load',flush=True)
        if hy4_native and block_idx == model.config.num_hidden_layers:
            from src.hy4.hf_hy4_mtp_stage import prepare as prepare_mtp_stage
            prepare_mtp_stage(adapter, model, calibration, args.model_name_or_path,
                              device, offload_device)
            inputs = calibration.inputs
            position_ids = calibration.position_ids
            block_states = calibration.block_states
        # Collect state dict keys from all processes
        if not hy4_native:
            rank_block_keys, block_keys_with_prefix, other_ranks_keys, has_routed_experts = prepare_block_keys(
                block_idx, block, prefix, world_size, checkpoint, adapter
            )

        # Put block onto GPU
        block.to_empty(device=device)

        # Dense blocks are replicated on all ranks. MoE blocks may contain rank-local experts.
        if hy4_native:
            adapter.load_current_block(block, block_idx, args.model_name_or_path)
            print(f'[Hy4 native rank {rank}] layer={block_idx} stage=loaded elapsed={time.monotonic()-stage_start:.3f}s allocated_gib={torch.cuda.memory_allocated(device)/(1<<30):.3f}',flush=True)
        elif not has_routed_experts:
            load_dense_block(block, prefix, block_keys_with_prefix, checkpoint, adapter, dtype)
        # Materialize one rank at a time so packed experts are never all expanded on rank 0.
        else:
            load_moe_block(
                block,
                prefix,
                rank_block_keys,
                other_ranks_keys,
                checkpoint,
                adapter,
                dtype,
                device,
            )
        # Clear memory before calibration
        torch.cuda.empty_cache()
        gc.collect()

        if block_idx >= resume_block_idx:
            if hy4_native:
                print(f'[Hy4 native rank {rank}] layer={block_idx} stage=collect documents={len(inputs)}',flush=True)
            # Collect GPTQ Hessian statistics for the current block.
            handles, hooks = collect(
                block, prefix, model, args, adapter, inputs, position_ids, block_states, device, rank
            )
            if hy4_native:
                from src.hy4.hf_hy4_layer_stage import quantize_and_save
                print(f'[Hy4 native rank {rank}] layer={block_idx} stage=gptq projections={len(handles)}',flush=True)
                quantize_and_save(handles, args, calibration, block_idx, rank, world_size)
                del handles, hooks
                torch.cuda.empty_cache()
                print(f'[Hy4 native rank {rank}] layer={block_idx} stage=propagate',flush=True)
                propagate(block, adapter, inputs, position_ids, block_states, device, offload_device)
                release(block, prefix, checkpoint, runtime)
                from src.hy4.hf_hy4_layer_commit import commit as commit_layer
                from src.hy4.hf_hy4_run_state import status as save_native_status
                commit_layer(args.save_dir, block_idx, calibration, run_identity)
                print(f'[Hy4 native rank {rank}] layer={block_idx} stage=committed elapsed={time.monotonic()-stage_start:.3f}s peak_gib={torch.cuda.max_memory_allocated(device)/(1<<30):.3f}',flush=True)
                resume_block_idx = block_idx + 1
                save_native_status(args.save_dir, run_identity, resume_block_idx, len(transformer_layers))
                dist_utils.print_on_main(f'[Hy4 native] globally committed layer {block_idx}; next={resume_block_idx}')
                from src.hy4.hf_hy4_cache_retention import retire as retire_old_boundary
                retired=retire_old_boundary(args.save_dir,block_idx,run_identity,args.model_name_or_path)
                if retired:
                    print(f'[Hy4 native rank {rank}] retired_reconstructible_cache_bytes={retired}',flush=True)
                continue
            shared_handles = {k: v for k, v in handles.items() if not adapter.is_routed_expert(k)}
            expert_handles = {k: v for k, v in handles.items() if k not in shared_handles}

            # Quantized shared handles first
            num_issue_zero_samples = 0
            num_issue_nan_hessian = 0
            num_issue_non_invertible = 0
            for handle_name, handle in shared_handles.items():
                issue_zero_samples, issue_nan_hessian, issue_non_invertible = quantize_shared_handle(
                    handle_name, handle, args
                )
                num_issue_zero_samples += issue_zero_samples
                num_issue_nan_hessian += issue_nan_hessian
                num_issue_non_invertible += issue_non_invertible

            dist_utils.print_on_main("-" * 10)
            dist_utils.print_on_main(f"GPTQ calibration issues for shared modules:")
            dist_utils.print_on_main(f"Zero Hessian: {num_issue_zero_samples}")
            dist_utils.print_on_main(f"Non-invertible: {num_issue_non_invertible}")
            dist_utils.print_on_main(f"NaN Hessian: {num_issue_nan_hessian}")
            dist_utils.print_on_main("-" * 10)

            # Quantize experts
            num_issue_zero_samples = 0
            num_issue_nan_hessian = 0
            num_issue_non_invertible = 0
            if len(expert_handles) > 0:
                dist_utils.print_on_main(f"Processing experts")

                expert_messages = None
                if dist_utils.is_main():
                    expert_messages = [None for _ in range(world_size)]
                rank_expert_message = ""

                for handle_name, handle in expert_handles.items():
                    message, issue_zero_samples, issue_nan_hessian, issue_non_invertible = quantize_expert_handle(
                        handle_name, handle, args
                    )
                    rank_expert_message += message
                    num_issue_zero_samples += issue_zero_samples
                    num_issue_nan_hessian += issue_nan_hessian
                    num_issue_non_invertible += issue_non_invertible

                dist_utils.barrier(device_ids=[rank])

                if dist_utils.is_dist_available_and_initialized():
                    dist.gather_object(rank_expert_message, expert_messages)
                    if dist_utils.is_main():
                        for expert_message in expert_messages:
                            dist_utils.print_on_main(expert_message)
                else:
                    dist_utils.print_on_main(rank_expert_message)

                # TODO sync data from other processes
                dist_utils.print_on_main("-" * 10)
                dist_utils.print_on_main(f"GPTQ calibration issues for expert modules:")
                dist_utils.print_on_main(f"Zero Hessian: {num_issue_zero_samples}")
                dist_utils.print_on_main(f"Non-invertible: {num_issue_non_invertible}")
                dist_utils.print_on_main(f"NaN Hessian: {num_issue_nan_hessian}")
                dist_utils.print_on_main("-" * 10)

            del handles
            del shared_handles
            del expert_handles
            del hooks
            torch.cuda.empty_cache()
            gc.collect()
        else:
            dist_utils.print_on_main(f"Block {block_idx} is already quantized. Skipping quantization.")

        propagate(block, adapter, inputs, position_ids, block_states, device, offload_device)

        release(block, prefix, checkpoint, runtime)

    if not hy4_native:
        save_quantization_metadata(args)
    cleanup_runtime()


def main():
    return run(parse_args())


if __name__ == "__main__":
    main()
