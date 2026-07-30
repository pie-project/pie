# Stage 6 — what the recon changed, and the plan that survives it

Date: 2026-07-30. Recon over the graph machinery, the hook execution path,
and §3.4's design, against the tree with stages 1–5 landed.

## The finding that reorders everything

Pie's only Structural divergence site is the hook site (`fire_plan.rs`:
`qkv_postprocess`; `Lowering::Guard` is dead code by design). §3.4's
conditional regions are built for genuinely different operator sequences
per lane — MoD, per-request depth — and none exist here yet. For the one
live site, the numbers go the other way: hook-vs-no-hook as a SWITCH would
pay the idle-node floor (1.2–1.4 µs × 28 layers ≈ 40 µs, `l40s_cond.txt`)
to save ~35 cache entries, while enumerating it as a key bit costs 2× a
lattice that sits at a quarter of `kMaxEntries`. And `fast_rows` — the
dimension that genuinely will not fit the key (~600 triangular entries) —
dissolves without any conditional: bake the grid at the lattice R and make
the split a device-resident count on the `row_valid` pattern.

**So: conditional nodes stay unbuilt until a second Structural site
exists.** §3.4 remains correct as design; it has no consumer yet.

## The actual wall: hooks are host code

A hook fire is eager because `run_generated_stage` has no prepare/body
split — the same defect the attention planner had before it was hoisted
(`forward_graph.hpp:18-24`). Per fire it: blocks on events inside the
would-be capture (`fused_runtime.cuh:109`, `:192` — illegal in capture,
full stop), rebuilds lane/channel/readiness metadata on the host and
uploads it, launches with grid = lane count, allocs per-layer sidebands
(`LayerScoreCapture` ×3, `FirePageMask` ×5 cudaMallocAsync per fire), and
— the silent one — bakes the channel protocol cursors
(`expected_head`/`expected_tail`/`committed_cell`,
`fused_runtime.cuh:1024-1049`) into uploaded arguments. A replayed graph
would re-execute a previous fire's channel view: issue #24's failure class
with no static_assert available, because the aliasing lives in runtime
data.

Also refuted en route: splitting a mixed fire so the hook-free prefix
replays while the tail runs eager. One body call per fire, the split is
per-layer inside it, and temporal segmentation at hook boundaries needs
2L+1 = 57 segments ≈ 137 µs of launch overhead — strictly worse than
fixing the hook body.

## The increments (in order; each useful alone)

1. **PTIR prepare/body split** (the bulk, ~2–3 wks): hoist the host
   metadata build out of `run_generated_stage` into a per-fire prepare
   writing one stable pinned + one stable device buffer per
   (stage, R-bucket); retire the rotating rings on the captured path
   (removes the event-sync violations with them). Go/no-go spike ~1 wk:
   capture one ON_ATTN stage, replay twice, diff.
2. **Stable-address sidebands** (~1 wk): fire-scoped lattice-sized
   workspace for score capture + page mask. Deletes ~3 allocs/layer/fire
   from the hot path — worth landing even if 1 stalls.
3. **Grid padding**: lane grids baked at lattice max, idle blocks exit
   early (`test_graph_padding_kv_canary` is the harness pattern).
4. **The key bit**: `kGvHasHooks` in `graph_variant` (layout shifts to 4 —
   re-prove the #24 static_asserts), `fast_rows` device-resident, drop
   `!has_stage_hooks` from eligibility for llama_like only.

Total ≈ 5–7 wks to a hook fire replaying a graph.

## The two risks that gate shipping

- **Channel-cursor bake** (above): any version that does not make the
  channel view device-resolved or re-uploaded per replay into a stable
  address is shipping a miscompute, not a bug that crashes.
- **TP branch agreement**: `tp.cpp:1357` hardcodes followers hook-free.
  Rank 0 replaying a hook graph while followers replay plain ones must
  keep NCCL op order identical; a conditional wrapping a collective would
  deadlock. Capture lockstep (`tp_graph_capture_barrier`) does not cover
  replay-time branch agreement.
