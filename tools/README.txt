From GPU (2026-11-25). Run on a build that contains 109e5bb89 (no patch needed).
Build: cargo build --release --features cuda (pie) and examples' text-completion-bench wasm.
Import Qwen3-8B, Qwen3-0.6B, gpt-oss-20b into one fresh PIE_HOME. Per model:
python3 sweep.py --pie ./target/release/pie --wasm .../text_completion_bench.wasm --home $PIE_HOME \
  --client <checkout>/packages/python/client/src --model Qwen/Qwen3-8B --tops 2048,4096,8192,16384 --conc 1 --reps 6
Report per model/top: c1 (6 values), c1_mean, c1_range_pct, boot_s, laid_top, device_pages, bodies, refused.
Tie rule: per model the smallest top within 2x the larger rep range of the best mean (drift counts where wider);
across models the smallest worst loss, ties to smaller.
GPU lock per server run; the script edits only $PIE_HOME/profiles/cuda/user/*.toml (fire_rows).
