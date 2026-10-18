#!/bin/bash
# The baseline: one SGLang server per GPU, behind sglang-router's
# cache-aware policy (SGLang's own data-parallel serving with prefix-aware
# routing). Same model, context and greedy outputs as the pie run.
#   sglang_up.sh [model] [context]   -> router on 127.0.0.1:30100
set -e
MODEL=${1:-Qwen/Qwen3-14B}; CTX=${2:-12288}; PY=${SGLANG_PY:-/root/venvs/sglang/bin/python}
LOG=${LOG_DIR:-.}
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g nohup $PY -m sglang.launch_server --model-path "$MODEL" \
    --host 127.0.0.1 --port $((30000 + g)) --context-length $CTX \
    --mem-fraction-static ${MEM_FRACTION:-0.88} --enable-cache-report \
    --served-model-name default > $LOG/sgl$g.log 2>&1 &
done
for g in 0 1; do
  timeout 900 bash -c "until curl -sf 127.0.0.1:$((30000 + g))/health >/dev/null; do sleep 3; done"
done
nohup $PY -m sglang_router.launch_router --worker-urls http://127.0.0.1:30000 http://127.0.0.1:30001 \
  --policy cache_aware --host 127.0.0.1 --port 30100 > $LOG/router.log 2>&1 &
timeout 120 bash -c "until curl -sf 127.0.0.1:30100/health >/dev/null; do sleep 2; done"
grep -h "max_total_num_tokens" $LOG/sgl0.log $LOG/sgl1.log | head -2 || true
echo up
