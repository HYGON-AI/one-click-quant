#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ─────────────────────────────────────────────────────────────────────────────
# GLM-5.3 W8A16 级联精度扫描脚本（原始 fp8-block vs MoE-Quant W8A16 packed）
#
# 循环遍历层区间, 每轮构建 2 个 mini 模型 (orig fp8-block / w8a16 packed), 用级联模式比对:
#   每段输入 = 上一段 orig mini 的 hidden state 输出（--cascade）
#   消除路由分叉（--fix-routing）并打印逐层 cos（--dump-per-layer）
#   W8A16 packed 权重必须走 --cpu-convert-then-dispatch（ct_decompress 路径）
#
# 磁盘策略:
#   - mini 目录命名稳定 (GLM53-mini-<fmt>-L<start>-<end>), 跨 run 可复用
#   - 若某轮 2 个 mini 目录都已存在 → 直接复用 (跳过 build)
#   - 若都不存在 → 正常 build; 本次 build 的产物进入延迟清理队列
#   - 延迟队列保留最近 KEEP_ROUNDS 轮产物 (只清 mini 目录, embeds 全部保留)
#   - 中断 (Ctrl+C) 立即退出, 磁盘上留下的文件由用户手动清理
#
# embeds 文件:
#   - 保存到 EMBEDS_DIR, 命名为 <label>_bs{bs}_len{seq}_hs.pt (label = orig mini 目录名)
#   - 下一段通过 --load-embeds-prefix <EMBEDS_DIR>/<prev_label> 自动加载
#
# 结果汇总到 summary.csv, 每轮独立日志到 logs/sweep_cascade_TIMESTAMP/
#
# 用法示例:
#   bash sweep_cascade_w8a16.sh                          # 按 indexer_types 自动切段
#   bash sweep_cascade_w8a16.sh --start 40 --total 60    # 只测中间段 [40, 60)
#                                                        # 注意: --start 非 0 时首段无法加载上一段 hs，
#                                                        #       退回 embed_tokens 作为输入（非真实 hidden state）
#   bash sweep_cascade_w8a16.sh --src-w8a16 /other/path  # 自定义 W8A16 路径
#   bash sweep_cascade_w8a16.sh --embeds-dir /ssd/embeds # 自定义 embeds 保存目录
#   bash sweep_cascade_w8a16.sh --fix-routing 0          # 关闭路由固定, 测路由+权重综合误差
#
# 层组切分:
#   不再传 --step。脚本读 config.json 的 indexer_types, 按 "每段 = 一个 full
#   indexer + 其后连续的 shared" 自动切分, 保证每段 forward 内 shared 层能拿
#   到同段内前置 full 层产生的 top-k, 不再抛 "Shared DSA layers require
#   top-k indices from a previous full indexer layer."
# ─────────────────────────────────────────────────────────────────────────────

set -u

# ── 默认参数 ──
SRC_ORIG="./GLM-5.3"
SRC_W8A16="./GLM-5.3-w8a16-group32-packed"
WORK_DIR="./glm_mini"
EMBEDS_DIR="./glm_mini/embeds"
LOG_ROOT="./logs"
START=0
TOTAL=""    # 空则从 SRC_ORIG/config.json 自动读取
KEEP_ROUNDS=2
FIX_ROUTING=1   # 1=启用 --fix-routing (推荐, 消除路由分叉干扰); 0=不加, 观察路由+权重综合误差

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_MINI="${SCRIPT_DIR}/build_mini_glm5_3.py"
COMPARE="${SCRIPT_DIR}/compare_mini_models.py"

# ── 参数解析 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            echo "⚠ --step 已废弃; 现按 config.json 的 indexer_types 自动切分 (每段 = 一个 full + 其之后连续的 shared)。忽略 --step $2" >&2
            shift 2
            ;;
        --start)         START="$2";      shift 2 ;;
        --total)         TOTAL="$2";      shift 2 ;;
        --src-orig)      SRC_ORIG="$2";   shift 2 ;;
        --src-w8a16)     SRC_W8A16="$2";  shift 2 ;;
        --work-dir)      WORK_DIR="$2";   shift 2 ;;
        --embeds-dir)    EMBEDS_DIR="$2"; shift 2 ;;
        --log-dir)       LOG_ROOT="$2";   shift 2 ;;
        --keep-rounds)   KEEP_ROUNDS="$2"; shift 2 ;;
        --fix-routing)
            case "$2" in
                0|1) FIX_ROUTING="$2"; shift 2 ;;
                *) echo "❌ --fix-routing 只接受 0 或 1, 收到: $2" >&2; exit 1 ;;
            esac
            ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \?//' | head -40
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ── 前置检查 ──
for f in "$BUILD_MINI" "$COMPARE"; do
    if [[ ! -f "$f" ]]; then
        echo "❌ 缺少脚本: $f" >&2
        exit 1
    fi
done

for d in "$SRC_ORIG" "$SRC_W8A16"; do
    if [[ ! -d "$d" ]]; then
        echo "❌ 源模型目录不存在: $d" >&2
        exit 1
    fi
done

# ── 自动读取 total_layers ──
if [[ -z "$TOTAL" ]]; then
    TOTAL=$(python3 -c "
import json, sys
with open('${SRC_ORIG}/config.json') as f:
    cfg = json.load(f)
tc = cfg.get('text_config', cfg)
print(tc['num_hidden_layers'])
" 2>/dev/null) || {
        echo "❌ 无法从 ${SRC_ORIG}/config.json 读取 num_hidden_layers" >&2
        exit 1
    }
fi

# ── 按 indexer_types 计算 DSA-aware 层组 ──
# GLM-5.3 中 "shared" indexer 层依赖同一次前向内前置 "full" 层生成的 top-k;
# 若 mini 从 shared 层开始 forward 会抛
#   "Shared DSA layers require top-k indices from a previous full indexer layer."
# 因此每段 mini 必须从 full 起, 到下一个 full 前止。连续 full 前缀
# (如 layers 0-2) 归入首段, 让首段含 full+shared 而不是全 full。
# 输出格式: "s1:e1,s2:e2,..." (半开区间)
RANGES=$(python3 - "$SRC_ORIG" "$TOTAL" 2>&1 <<'PYEOF'
import json, sys
src_dir, total_str = sys.argv[1], sys.argv[2]
with open(f"{src_dir}/config.json") as f:
    cfg = json.load(f)
tc = cfg.get("text_config", cfg)
num_hidden = int(total_str) if total_str else int(tc["num_hidden_layers"])
itypes = list(tc.get("indexer_types", []))[:num_hidden]
if not itypes:
    sys.exit("indexer_types 缺失, 无法自动切分")
if itypes[0] != "full":
    sys.exit(f"第 0 层 indexer_type={itypes[0]!r}, 不是 full, 无法从此起切分")
# 找每个组的起点 = 每个 "本 full 之前不是 full" 的位置
group_starts = []
for i, t in enumerate(itypes):
    if t == "full" and (i == 0 or itypes[i - 1] != "full"):
        group_starts.append(i)
# 每组 [start, next_start) 或到 num_hidden
group_starts.append(num_hidden)
ranges = [(group_starts[i], group_starts[i + 1]) for i in range(len(group_starts) - 1)]
print(",".join(f"{s}:{e}" for s, e in ranges))
PYEOF
)
py_rc=$?
if (( py_rc != 0 )) || ! [[ "$RANGES" =~ ^[0-9]+:[0-9]+(,[0-9]+:[0-9]+)*$ ]]; then
    echo "❌ 计算 DSA 层组失败 (rc=$py_rc):" >&2
    echo "$RANGES" >&2
    exit 1
fi

IFS=',' read -ra RANGE_ARR <<< "$RANGES"
NUM_RANGES=${#RANGE_ARR[@]}
if (( NUM_RANGES == 0 )); then
    echo "❌ 未能生成任何层组" >&2
    exit 1
fi

# ── 日志目录 ──
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_ROOT}/sweep_cascade_${TIMESTAMP}"
mkdir -p "$LOG_DIR"
mkdir -p "$EMBEDS_DIR"
MAIN_LOG="${LOG_DIR}/main.log"
SUMMARY="${LOG_DIR}/summary.csv"

echo "start,end,cos_min,max_diff_max,cos_all,diff_all,build_status,compare_status" > "$SUMMARY"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

log "============================================================"
log "GLM-5.3 W8A16 级联精度扫描（orig fp8-block vs w8a16 packed）"
log "  SRC_ORIG:   $SRC_ORIG"
log "  SRC_W8A16:  $SRC_W8A16"
log "  WORK_DIR:   $WORK_DIR"
log "  EMBEDS_DIR: $EMBEDS_DIR"
log "  LOG_DIR:    $LOG_DIR"
log "  START=$START  TOTAL=$TOTAL  KEEP_ROUNDS=$KEEP_ROUNDS  FIX_ROUTING=$FIX_ROUTING"
log "  按 indexer_types 自动切分, 共 $NUM_RANGES 段: $RANGES"
log "============================================================"

# ── mini 目录命名 ──
mini_orig()  { echo "${WORK_DIR}/GLM53-mini-orig-L${1}-${2}"; }
mini_w8a16() { echo "${WORK_DIR}/GLM53-mini-w8a16-L${1}-${2}"; }

# ── embeds 文件前缀（与 compare_mini_models.py 的 save_embeds_dir 逻辑一致）
# 保存时文件名为 <label>_bs{bs}_len{seq}_hs.pt，label = orig mini 目录名
embeds_prefix() { echo "${EMBEDS_DIR}/GLM53-mini-orig-L${1}-${2}"; }

# ── 从 compare 日志提取指标 ──
# 输出 3 行:
#   line 1: "<cos_min> <max_diff_max>"
#   line 2: "<cos1>,<cos2>,..."
#   line 3: "<diff1>,<diff2>,..."
extract_metrics() {
    local logfile="$1"
    python3 - "$logfile" <<'PYEOF' 2>/dev/null || printf "NA NA\nNA\nNA\n"
import re, sys
p = sys.argv[1]
try:
    txt = open(p).read()
except Exception:
    print("NA NA"); print("NA"); print("NA"); sys.exit(0)
cos_vals  = [float(m.group(1)) for m in re.finditer(r"Cosine similarity:\s*([\d.eE+\-]+)", txt)]
diff_vals = [float(m.group(1)) for m in re.finditer(r"最大绝对差异:\s*([\d.eE+\-]+)", txt)]
cos_min  = min(cos_vals)  if cos_vals  else float("nan")
diff_max = max(diff_vals) if diff_vals else float("nan")
print(f"{cos_min:.6f} {diff_max:.6e}")
print(",".join(f"{v:.6f}" for v in cos_vals)  if cos_vals  else "NA")
print(",".join(f"{v:.3e}" for v in diff_vals) if diff_vals else "NA")
PYEOF
}

# ── 检查 mini 目录复用状态 ──
# 返回 0: 都不存在 → 需要构建
# 返回 1: 都存在   → 直接复用
# 返回 2: 部分存在 → 状态混乱, 报错退出
check_reuse_state() {
    local exist=0 miss=0
    for d in "$@"; do
        if [[ -d "$d" ]]; then exist=$((exist + 1)); else miss=$((miss + 1)); fi
    done
    if   (( exist == 0 )); then return 0
    elif (( miss  == 0 )); then return 1
    else                        return 2
    fi
}

# ── 清理延迟队列头部 1 轮（只删 mini 目录，embeds 文件始终保留）──
# 队列元素格式: "<range_tag>|<dir_orig>|<dir_w8a16>|<embeds_prefix>"
PENDING_ROUNDS=()

drain_one_round() {
    (( ${#PENDING_ROUNDS[@]} == 0 )) && return
    local entry="${PENDING_ROUNDS[0]}"
    PENDING_ROUNDS=("${PENDING_ROUNDS[@]:1}")
    IFS='|' read -r rtag d1 d2 ep <<< "$entry"
    log "  [清理] 释放旧轮 mini: ${rtag}"
    [[ -n "$d1" && -d "$d1" ]] && rm -rf "$d1"
    [[ -n "$d2" && -d "$d2" ]] && rm -rf "$d2"
}

trap 'echo; echo "⚠ 收到中断信号, 立即退出 (mini 目录和 embeds 文件未清理, 请手动清理)" >&2; exit 130' INT TERM

# ── 主循环 ──
prev_start=""   # 上一段的 start（用于生成 --load-embeds-prefix）
prev_end=""     # 上一段的 end

for range in "${RANGE_ARR[@]}"; do
    start=${range%:*}
    end=${range#*:}
    # 支持 --start N: 完全在 N 之前的组直接跳过
    (( end <= START )) && continue

    range_tag="L${start}-${end}"
    round_log="${LOG_DIR}/range_${range_tag}.log"
    log "── [Range ${start}:${end}] 开始 ──  详见 ${round_log}"

    # 进本轮之前, 若延迟队列已达上限, 先释放最旧的一轮
    while (( ${#PENDING_ROUNDS[@]} >= KEEP_ROUNDS )); do
        drain_one_round
    done

    dir_orig=$(mini_orig  "$start" "$end")
    dir_w8a16=$(mini_w8a16 "$start" "$end")
    ep=$(embeds_prefix "$start" "$end")

    build_status="ok"
    compare_status="ok"
    cos_min="NA"
    diff_max="NA"
    cos_all="NA"
    diff_all="NA"

    # ─ 复用检查 ─
    check_reuse_state "$dir_orig" "$dir_w8a16"
    reuse_rc=$?
    if (( reuse_rc == 1 )); then
        log "  [复用] 2 个 mini 目录已存在, 跳过 build 直接 compare"
        echo "=== REUSE (skip build) ===" >> "$round_log"
        build_status="reused"
    elif (( reuse_rc == 2 )); then
        log "❌ 部分 mini 已存在 (状态混乱), 跳过本轮; 请手动清理后再运行:"
        for d in "$dir_orig" "$dir_w8a16"; do
            log "     $d $([[ -d $d ]] && echo '[存在]' || echo '[缺失]')"
        done
        build_status="skip_partial"
        echo "${start},${end},NA,NA,NA,NA,${build_status},${compare_status}" >> "$SUMMARY"
        # 更新 prev 指针：跳过的段不产生 hs 文件，下一段 compare 会因找不到文件而退回 embed_tokens
        prev_start=$start
        prev_end=$end
        continue
    fi

    # ─ Build 2 mini models ─
    if [[ "$build_status" != "reused" ]]; then
        {
            echo "=== BUILD orig (fp8-block dequant → bf16) ==="
            python3 "$BUILD_MINI" --src "$SRC_ORIG" --dst "$dir_orig" \
                --src-format fp8-block --layer-range "${start}:${end}"
        } >> "$round_log" 2>&1 || build_status="fail_orig"

        if [[ "$build_status" == "ok" ]]; then
            {
                echo "=== BUILD w8a16 (keep-quant → HF ct_decompress) ==="
                python3 "$BUILD_MINI" --src "$SRC_W8A16" --dst "$dir_w8a16" \
                    --src-format w8a16 --layer-range "${start}:${end}" \
                    --keep-quant
            } >> "$round_log" 2>&1 || build_status="fail_w8a16"
        fi
    fi

    # ─ Compare ─
    if [[ "$build_status" == "ok" || "$build_status" == "reused" ]]; then
        cmp_log="${LOG_DIR}/cmp_${range_tag}.log"

        # 首段（start=0）没有上一段的 hs 文件，不传 --load-embeds-prefix
        embeds_args=()
        if [[ -n "$prev_end" ]]; then
            prev_ep=$(embeds_prefix "$prev_start" "$prev_end")
            embeds_args=(--load-embeds-prefix "$prev_ep")
        fi

        # 是否加 --fix-routing
        if (( FIX_ROUTING == 1 )); then
            fix_routing_arg="--fix-routing"
        else
            fix_routing_arg=""
        fi

        {
            echo "=== COMPARE orig vs w8a16 (cascade, fix_routing=${FIX_ROUTING}) ==="
            python3 "$COMPARE" \
                --model-a "$dir_orig" \
                --model-b "$dir_w8a16" \
                --cascade ${fix_routing_arg:+"$fix_routing_arg"} --dump-per-layer \
                --cpu-convert-then-dispatch \
                --save-embeds-dir "$EMBEDS_DIR" \
                "${embeds_args[@]}"
        } > "$cmp_log" 2>&1 || compare_status="fail"
        cat "$cmp_log" >> "$round_log"

        mapfile -t _lines < <(extract_metrics "$cmp_log")
        read -r cos_min diff_max <<< "${_lines[0]}"
        cos_all="${_lines[1]}"
        diff_all="${_lines[2]}"
    fi

    if [[ "$compare_status" == "ok" && "$build_status" != "skip_partial" ]]; then
        log "    [orig vs w8a16] cos_all=${cos_all}"
        log "    [orig vs w8a16] diff_all=${diff_all}"
    fi
    log "  build=${build_status}  compare=${compare_status}  cos_min=${cos_min}  max_diff=${diff_max}"
    echo "${start},${end},${cos_min},${diff_max},${cos_all},${diff_all},${build_status},${compare_status}" >> "$SUMMARY"

    # ─ 本轮结束: 把本次 build 的产物挂到延迟队列（复用的不入队）─
    if [[ "$build_status" == "ok" ]]; then
        PENDING_ROUNDS+=("${range_tag}|${dir_orig}|${dir_w8a16}|${ep}")
    fi

    # 推进 prev 指针（供下一段构建 --load-embeds-prefix）
    prev_start=$start
    prev_end=$end
done

# ── 循环结束: 剩余队列中的 mini 目录不再清理，embeds 全部保留 ──
log "── 主循环结束 (队列中剩余 ${#PENDING_ROUNDS[@]} 轮 mini 目录保留，embeds 全部保留) ──"

log "============================================================"
log "全部完成. Summary: $SUMMARY"
log "============================================================"
