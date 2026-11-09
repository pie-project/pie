# The role-split wafer program (Cerebras CS-3)

The role-split program runs one bf16 model on a CS-3 wafer at batch 1. It is a second wafer layout
next to `serve8b_region`. It is built for Qwen3.5 (9B, 4B and 0.8B), Qwen3-8B, Gemma 4 (E2B, E4B) and gpt-oss-20b
(section 7). This file covers its design, results, build and run, the parameters and environment variables, the
metrics, and open items.
Serving is Rust only: `SplitModel` in this module behind the engine's `Wafer`. The Python under
`crates/engine-cerebras/tools/` runs at build time only.

## 1. Design

**Why a second layout.** In `serve8b_region`, every layer PE carries the whole layer unit's code and
buffers: about 22 KB of each 48 KB PE is not weights. That is why the bf16 9B models do not fit
there. The role-split layout gives PEs fixed roles. Most PEs hold only weights, plus a small matvec
kernel of about 8 KB, so about 40 KB of each PE holds weights. The element-wise work and the roots
of every collective move onto a few control PEs.

**Unit.** A unit is PX x PY weight PEs, a control column (CX) and a control row (CY), so it spans
(PX + 1) x (PY + 1) PEs. Each unit runs K layers. The units are visited as a snake: down unit
column 0, then up column 1, and so on.

- **Weight PE** (`csl/split/weight_q35.csl`). Holds its K layers' four tiles: the row's x chunk
  (Hc dims) by the column's outputs, as f16 bits. A layer runs as:
  1. x from CX's row multicast
  2. the qkv partial goes into the column reduction to CY
  3. o comes back in CY's column multicast
  4. the o_proj partial goes into the row reduction to CX
  5. the MLP repeats steps 1-4

  The PE also holds the mixer's per-PE parts:
  - GDN: its slice of the recurrent state.
  - Attention: NE KV entries a row; position p lives on row p % PY.
- **CX** (`ctl_cx.csl`). Holds x's chunk, the residuals and the norms' gammas. It is the root of the
  row reductions and does the unit handoff.
- **CY** (`ctl_cy_q35.csl`). The root of the column reductions. It does the rms scale, the GDN conv,
  SiLU, gates, delta and the gated norm, RoPE and the q/k norms, the softmax's o / l with its redo
  flag and log-sum-exp steps, and the MLP SiLU.
- **Head** (`head_pe.csl`, `head_ctl.csl`). HC columns of HDc dims by HR rows of VR vocabulary rows.
  The head holds lm_head and the embedding tiles of the same vocabulary slice. Each row's control
  PE reduces the logits and takes the row argmax. The head has HR x VR rows, more than the
  vocabulary (9B / 4B / 0.8B: 249,048 over 248,320). The prep fills the lm_head rows past the
  vocabulary with copies of its last row, so they tie with it and the first-maximum argmax keeps
  the real, lower id; zero rows would win whenever every real logit is negative. The host also
  truncates logits rows to the vocabulary and refuses a ring id past it.
- **Wafer token loop** (`WL 1`). The rows' (max, id) pairs are max-reduced down the head control
  column. The root records the winner in `ring`, or takes `force[s]` for a forced prompt. The
  winner row's head PEs send their embedding slices back to unit 0's CX rows, so a launch runs
  many tokens without the host.
- **Collectives.** Every reduction is a two-level streamed chain (groups of G2) into a control
  PE. The control PE merges both halves through a FIFO and multicasts the result. Routes and
  queue colours are static per PE: `rcfg` words from `geo.rs`. There are no axis switches, so one
  image serves every PE of a kind.

**Queue and colour rules learned on the CS-3.** The simulator accepted every one of these
violations; the chip did not.

- **Each flow gets its own queue.** An async op runs on its queue id's microthread, which iq N and
  oq N share. Example: the row add-forward reads iq2 and sends on oq5, not oq2. A segment
  all-reduce that borrowed a live reduction queue gave wrong sums.
- **Restore a route or queue only after its output queue drains.** A send returns once its
  wavelets enter the queue, not when they leave it. Restoring right after the send stranded a
  tail and hung the token loop. The code waits for oq4 to be empty first.
- **Park input queues only on colours nothing routes to the PE.** Never park one on the PE's own
  send colour; that hung the row reductions. Never let two input queues share a colour, even
  briefly (`bind_iqs`).
- **Keep at least 768 B of stack room in every image.** The linker does not count the stack, which
  grows down from 0xC000. `tools/split/chk_stack.py` checks this after every link.
- **Name chip payloads `*_ieee`.** The bench harness otherwise compiles with
  `--fp16-format=bf16`, which turns the f16 kernels into garbage.

**Numerics.** IEEE f16 weights and x, with f16 accumulation on each PE and f32 across PEs. The MLP's
SiLU is in f16; the KV cache and the softmax probabilities are in f16. The KV matvecs accumulate in
f32. Every lever was gated in host emulation before it went to the chip
(`tools/emu/tf_q35.py`, teacher-forced against the HF f32 golden): it ships only if it keeps 0
mismatches. Two levers were left out on that basis: xg16, which flips a near tie on the 9B, and
one-pass bf16 x.

## 2. Results (CS-3, B1 greedy, all golden)

"bare" is the program driven by its own host loop. "Pie" is `pie serve` (headline metric: section
5). Every row's greedy output equals the HF f32 golden: 256 tokens for Qwen3.5 and 512 for Llama
and Qwen3-8B.

| model | layout | bare tok/s | Pie tok/s | notes |
|---|---|---|---|---|
| Llama 3.1 8B | serve8b_region (t29, TRM 32767) | 2,044 | ~2,000 e2e (1,981-2,002; decode 2,058-2,080) | depth 127, `posted:127` |
| Qwen3.5-9B | role-split (wl11: 8 units 256 x 256, K 4) | 1,961 | 1,908 (0.5241 ms/step, 97.3%) | loop 381.7k cycles; GDN 11.1k / attention 11.45k a layer |
| Qwen3.5-4B | role-split, the 9B geometry (Hc 10, Ic 36) | 2,057 | 1,998 (0.5005 ms/step, 97.1%) | GDN 10.6k / attention 10.96k |
| Qwen3.5-0.8B | role-split, 6 units of 256 x 128 (Hc 8, Ic 14, NE 128) | 3,390 | 3,286 (0.3043 ms/step, 96.9%) | GDN 8.2k / attention 9.2k |
| Qwen3-8B | role-split (9 units 256 x 228, Hc 18, K 4, gap slot; GATE 0, SLW 1) | 1,825 (0.548 ms, G2 12) | 1,811 (0.5521 ms, 99.3%, L64, G2 12) | greedy 512 and TF 512 golden; Pie 12 of 12 reps golden (G2 16 and 12). `PIE_CEREBRAS_SPLIT=<prep>:q84_:G2=12,WL=1`. serve8b_region (qw1): 1,642 |
| Gemma 4 E2B | role-split Gemma (7 units 256 x 128, Hc 12, K 5; WSPEC 2) | 1,431 (0.6986 ms) | 1,418 (0.7053 ms, 99.0%) | greedy 720 golden; TF 256 0 mismatches |
| Gemma 4 E4B | role-split Gemma (7 units 256 x 160, Hc 16, K 6) | 1,072 (0.9327 ms) | 1,063 (0.9404 ms, 99.2%) | global context 2,560 positions (NEG 16 x 160) |
| Gemma 4 12B | role-split 12B (8 units 344 x 283: 128 attention rows Hc 12, plain rows Hc 15; K 6; RING 1) | 1,057 (0.9462 ms) | 1,055 (0.9483 ms, 99.8%) | TF 212 0 mismatches, greedy golden; TF 1,600 past the window: 3 bf16 near-ties (below); global context 4,096 |
| gpt-oss-20b | role-split gpt-oss (12 units, K 2; CYV 2, GLUV 1) | 1,555 (0.643 ms) | 1,522 (0.6569 ms, 97.9%) | MXFP4 experts as E5M2, lossless; Pie text = bare greedy |

Notes on the table:

- The 9B, 4B and 0.8B builds share kernel sources; only the params differ.
- Pie runs 32-step loops (`PIE_CEREBRAS_SPLIT_LOOP=32`). Longer loops lower the per-step cost but
  lose wall time at the tail.
- Pie's own cost is about 0.45 ms per posted launch, so Pie reaches 97% of bare.
- Qwen3-8B: 36 dense attention layers in 9 K4 units on a 2 x 5 grid. The slot (1, 1) stays out of the snake so
  the last unit ends on row 0 (the wafer loop's return); its PEs (`relay.csl`) only carry the vertical handoff
  past it (`GAPX` / `GAPY`, prep meta). The hidden size is padded to PY Hc (4,104): zero weights, gammas and
  embedding past 4,096, the norms divide by H. A 12-unit K3 layout (PY 188, Hc 22) ran at 0.559 ms and flipped the
  golden's token 389, an exact bf16 tie (1424 and 97613 both 32.0 in HF bf16); the K4 build keeps it.
- Gemma 4 and gpt-oss-20b serve with 64-step loops, their prep meta's `LOOP`. gpt-oss at a fixed L32 / L64 posted
  0.6634 / 0.6619 ms (1,511 tok/s); with the meta's doubling loops up to 64, 0.6569 ms. Pie's overhead in ms equals
  the previous gpt-oss build's (gpo6: 0.6779 / 0.6723 ms against 0.661 bare): the ratio fell because bare got faster.
- Llama stays on `serve8b_region`: the role-split layout forecasts 2,029-2,057 for it, which is no
  gain.

## 3. Build and run

All tools are under `crates/engine-cerebras/tools/`. Set `CSL_SDK` to the SDK directory, which holds
`cslc` and `cs_readelf`.

1. **Prep** (checkpoint -> tiles): writes `<prefix>meta.json`, `w<u>.npy`, `hw`, `hemb`, ... and
   `params.txt`, the shape params.
   - Any Qwen3.5 size and geometry:
     `tools/split/prep_q35.py <model> <out> <golden.json> PX PY UNX UNY K HR VR NE [prefix]`.
     The 9B: `256 256 2 4 4 1153 216 64 s9b_`. The 4B: the same with its own model. The 0.8B:
     `256 128 2 3 4 1153 216 128 s08b_`.
   - The 9B in Rust: `cargo run --release -p engine-cerebras --example split_prep -- <ckpt> <out> s9b_`.
     `--check` compares it word for word against a Python prep.
2. **Numerics gate** (before any new lever): `tools/emu/tf_q35.py <model> <golden> <meta.json>
   ref,f16acc_s16_kv16 256 <out>`. This needs `torch` and `transformers` with Qwen3.5. For
   serve8b_region's levers on Qwen3-8B: `tools/emu/tf_qwen3_8b.py`.
3. **Sources**: `tools/split/mkfull.py <build dir>` writes `layout.csl` and one source a PE kind
   (weight, cx, cy, head, hctl, filler). It reads `src/spatial/csl/split`, or `$SPLIT_SRC` if set.
4. **Link and check**: `tools/split/link.sh <build dir> <prep>/<prefix>params.txt [K:V,...]`. This is
   a full local link at the chip's fabric dims, followed by the stack-room check. Run it on every
   build before a chip request. Name the chip payload `*_ieee`.
5. **Tiny end-to-end on the simulator** (no chip):
   1. `tools/split/mk_tinyhf.py <dir> t08` (or `t4`) builds a tiny HF checkpoint and its golden.
   2. `prep_q35.py` on it, then `mkfull.py` and `link.sh` with the tiny's params.
   3. `split_sim <build>/out <prep> "" G2=2,WL=1,G2W=2 5` (`--example split_sim`; `SPLIT_NTOK` caps
      the tokens) runs `SplitModel` against the golden: two sequences, a rollback and a logits row.
      Run it where the SDK runtime libraries are: inside the SDK container, through the container's
      glibc loader for a host-built binary, or with `PIE_CEREBRAS_SDK_LIB`.
   4. `examples/split_place.rs` dumps a placement copy by copy (sha256 each), so two hosts or
      builds can be compared.
6. **Serve**: `pie serve` with the cerebras engine and `PIE_CEREBRAS_SPLIT=<prep dir>:<prefix>:G2=16,WL=1`.
   The 0.8B uses `G2=16,G2Y=8,WL=1`. Pie config: `frame_size 4`, `frame_dispatch_depth 63`,
   `spatial_lookahead 254`. The program is the linked build. `Wafer::open` sees a split program and
   drives it through `SplitModel`.

## 4. Parameters, levers and diagnostics (one place)

The shape params come from the prep's `params.txt`. Every other param has a default, and the
default is the shipped build. Each kernel's header describes its own params and interface.

| param | where | default | meaning |
|---|---|---|---|
| `NPX` | head_pe | 1 | x in NPX f16 parts. 1 was emulated exact on all three goldens. |
| `MV` | weight | 1 (mkfull) | f16 tiles with f16 accumulation (0: bf16 tiles, f32 accumulation) |
| `SILU16`, `CONVQK`, `ATTV`, `GDEC`, `ROPEP`, `SILUQ` | CY (GDEC also weight) | 1 | the layer trims: f16 MLP SiLU; q/k conv first; vector gate; decay off the path; next position's RoPE tables early; lean f32 SiLU. The 0 forms are the earlier, slower code. |
| `ATTL` | CY, weight | 2 | the log-sum-exp steps go in a second multicast after o (0: they ride with o) |
| `SSQL` | CX, weight | 1 | x's sum of squares follows x |
| `XG16` | CX | 0 | 1: x g in f16. Inexact; flips a near tie on the 9B. |
| `VARG` | head_ctl | 2 | the row argmax by block maxima (0: the scalar scan; both exact) |
| `CAP` | head_ctl | 256 | ring / force length (steps a launch) |
| `TSTAMP` | weight, CY | 0 | diagnostic phase stamps into `tdbg`. On the path: TSTAMP 1 and XKEEP 1 together cost about 8k cycles a token on the 9B. |
| `XKEEP` | CX | 0 | diagnostic copies of every reduction (`xred`) and of x after each layer (`xdbg`) |
| `DIAG` | weight, CX, CY | 0 | bisection bits: 1 no layers, 2 one layer, 4 no MLP, 8 column reduction only, 16 row reduction only |
| `HOSTX` | CX | 0 | simulator tests: x is the host's, no handoff |
| `WFORCE`, `WFN` | head_ctl | 0 | diagnostics: a forced winner sequence `(s WFORCE + 7) % (VR WFN)` that exercises every row's return path |
| `force` / `nforce` | head_ctl (symbols) | - | the host forces the first `nforce` steps' tokens: a whole prompt through the loop in one launch (`PIE_CEREBRAS_SPLIT_FORCE`), and teacher-forced checks |
| `HACC16` | head_pe | 0 | measurement: f16 head accumulation, about -1.7k cycles. It halves the 9B golden's tightest margin and, stacked with other levers, flipped it. |
| `FIFOW` | all | 32 | the reductions' FIFO depth |
| `SLW` | serve8b_region regions; role-split CY | 0 | the f16 SiLU without its subnormal flush (the tail to 2^-24). Planner key `silu16w`; Qwen3-8B needs it on both layouts (`tools/emu/silu16.py`): the flush costs 2 golden tokens in emulation. |
| `GATE` | CY | 1 | attention's q_proj carries the output gate (Qwen3.5). Qwen3 (dense): 0, NQA = Qh Dc + 2 Dc. |
| `GAPX`, `GAPY` | layout | 255 | the unit slot left out of the snake (relay PEs); from the prep meta |
| `RLV`, `RBITS` | CY | 4, 3 | RoPE tables: a position is RLV digits of RBITS bits, cos / sin composed from one table entry a digit; positions < 2^(RBITS RLV). The prep takes the levels its context needs (9B / 4B / 0.8B: 5) and writes them to the meta; `SplitModel` bounds its positions there. |

`head_ctl`, `head_pe` and `CX` always write a few `tdbg` stamps. These are off the critical path:
`head_ctl`'s `tdbg[2 + s]` is the token loop's clock.

These variants were removed after they failed:
- ATTL 1: the logs on the weight PE do not fit the 9B weight PE.
- One Newton step in SILUQ: it flipped the 9B golden's tightest tie.
- VARG 1: superseded.
- A streamed act-into-down variant: a DSR-driven `@map` does not take a fabin DSD.

## 5. Environment (Pie host) and metrics

| variable | meaning |
|---|---|
| `PIE_CEREBRAS_SPLIT=<prep dir>:<prefix>[:k=v,...]` | selects the split program. Overrides: `G2` (`G2X` / `G2Y` / `G2H`) chain groups, `WL` the token loop, `G2W` the max-reduce groups, `CAP`, `EPS`. |
| `PIE_CEREBRAS_SPLIT_LOOP` | steps a posted loop runs past its row (default: the prep meta's `LOOP`, else 32; gpt-oss preps say 64). Loops double from 8 up to this value. |
| `PIE_CEREBRAS_SPLIT_POST=0` | one loop at a time (synchronous, bounded by the lane's pages) |
| `PIE_CEREBRAS_LOOP_ROOM=off` | synchronous mode with no page bound |
| `PIE_CEREBRAS_SPLIT_FORCE=1` | the prompt is forced through the token loop (one launch: prefill and the first token). Needs `force` / `nforce`. |
| `PIE_CEREBRAS_CONTINUATION=posted:<n>` | serve8b_region (Llama, Qwen3-8B): posted continuations, with the matching `frame_dispatch_depth` |

**Posted loops.** Greedy decoding runs as token loops posted back to back. The next loop goes onto
the wafer while the previous loop's rows are answered. The wafer idles about 0.02 ms between loops.

Non-greedy sampling stays on the host path: logits rows are `lgt` scaled by the last unit's rms.
They match HF f32 logprobs to within 0.004 on the chosen token.

**Metrics:**
- `posted_ms_per_step` (headline): the posted loops' launch wall time over their steps, from the
  engine log. tok/s = 1000 / it.
- `decode_tok_s`: (tokens - 1) / (last streamed chunk - first chunk), measured on the client. It
  flatters loops: the first loop's rows arrive together.
- `wall_tok_s`: tokens / request time, prefill included.
- bare tok/s: tokens / wall time of the program's own host loop at 255-step launches.
- cycles: `head_ctl`'s `tdbg[2 + s]` differences at the root, at about 0.745 GHz.

## 6. Open items

- **gpt-oss CYV 1** (the attention CY's exp and reciprocal both as vector ops, 0.632 ms) flips teacher-forced
  position 35 of the 256 golden, a bf16 near tie where the chip agrees with the f32 argmax. It stays off (CYV 2:
  vector exp only, bit-identical) until the pass criterion is decided.
- **gpt-oss units 0-2 stall.** On the last profiled step only, data-dependent, ~0.75% of a typical step. `PFTZ`
  and the other diagnostics are off by default.

- **Seed divergence.** Sampled requests with different seeds (7 and 11; T 0.8 and 1.0) return
  identical text on the split program. The seed path was traced through the interpreter's
  RngKeyed and engine-cerebras's per-bind channel seeds, and no loss was found. It needs an A/B on
  a CUDA or Metal Pie.
- **Qwen3-8B qw2 mismatch on chip.** qw2 is the Llama levers without the f16 SiLU (TRM 29695). On
  the CS-3 it gave 15 teacher-forced mismatches, with margins up to 2.77, although host emulation
  gives 0. It is a real bug, not yet found. qw1, which ships, is exact.
- **Rep-2 near-tie flip** (serve8b_region). On Qwen3-8B in Pie, 1 of 6 reps leaves the golden at token ~389, the
  golden's tightest margin (0.0043, an exact bf16 tie). The role-split build serves Qwen3-8B now: 6 of 6 reps golden.

- **Gemma 4 12B speed.** 0.946 ms against a ~0.86 ms forecast. A sliding layer is ~13.3k cycles
  (0.982 ms and 14.1k before the exact ring). The attention-row PE has 1,776 B of stack room, too
  little for `WSPEC 1`. Splitting the act into chunks, so the down matvec overlaps the GELU, was
  slower (0.995 ms). Next: fine stamps of the attention pass and of the CY GELU phase.
- **Gemma 4 12B long context.** Teacher-forced over 1,600 tokens (1,660 positions), 3 of 1,600 differ
  from the HF f32 golden, at i 372, 925 and 1301 (f32 gaps 0.042, 0.0034 and 0.0053). At each, the
  chip's pick is f32's runner-up. At 372 and 1301 HF bf16 picks the chip's token. At 925 the gap is
  far below a bf16 step at that logit (0.125).

## 7. Gemma 4 and gpt-oss-20b

Both reuse the role-split machinery (`geo.rs`'s chains, segments, handoffs, head and token loop; `head_ctl.csl`)
with their own kernels, geometry (`geo/gemma.rs`, `geo/gptoss.rs`: rcfg words identical to the Python tools on
every PE), placement (`place_gemma.rs`, `place_gptoss.rs`) and `SplitModel` paths. The prep meta selects the path:
`NEX` is gpt-oss, `geo` without it is Gemma. Tools: `tools/split/gemma`, `tools/split/gptoss` (prep, mkfull,
`run_*.py` bare runner, `gotiny.sh` tiny simulation), `tools/split/split_geo.py` (the Python geometry both build
on), `tools/emu/tf_gemma.py` and `tf_gptoss.py` (teacher-forced emulation of the kernels' number formats).

**Gemma 4** (`*_gemma.csl`, E2B and E4B; 12B below).
- Units at explicit origins: 7 units (E2B 256 x 128, Hc 12, K 5; E4B 256 x 160, Hc 16, K 6) placed around the
  per-layer-embedding (PLE) storage, which fills the PEs outside the units and the head.
- PLE: the host precomputes every token's per-layer input (f16, 4.7 GB for E2B) into storage PEs above each unit
  column. The head control rows send the winning id west (TB); the storage owner sends its words down the weight
  column on a bus (PB) that the CY PEs read with pre-posted receives, step 0 included. One more round a layer.
- Shared KV (E2B layers 15-34, E4B 24-41): the source unit's CY row sends k and raw v north on a static colour (KVS)
  through the consumer units, which store the entries of their rows.
- Sandwich norms without a new collective: CX keeps x and a pending term p whose scale is only known after the next
  reduction; the next phase multicasts both with four partial sums, CY combines them (deferred scalars).
- Sliding caches are rings of `window / PY` entries a row (`WIN` masks a longer ring); `WSPEC 1` specializes the
  attention code per layer kind and `WSPEC 2` also keeps the sliding ring whole (E2B; E4B has no room). GELU stays
  f32 (an f16 GELU flipped the long golden); the final softcap is applied on the host (greedy-invariant).
- Diagnostics: `DIAG 1` (no layers: handoffs, head, PLE and KVS words checked at every consumer), `PLOG 1` (the full
  model logs each step's PLE / KVS words: the hang-localizing probe).

**Gemma 4 12B** (`*_g12.csl`, bf16; no PLE or shared KV, `attention_k_eq_v` on the global layers). 21.9 GB of
layers and a 2 GB tied head fill 98% of the wafer, which only fits with a row-role split.
- Units: 8 of 344 x 283 weight PEs, K 6 (S S S S S G). The first 128 rows of a unit are attention rows (Hc 12; the KV,
  position p on row p % 128; the scores segments). The other 155 are plain rows (Hc 15, no attention code: they send
  zeros into the values reduction). Hidden 3,861 slots for 3,840; padded dims carry zero weights and gammas.
- Attention: every segment is 43 columns, so sliding has 8 kv heads x 43 columns (Dc 6) and global has the one kv head
  replicated 8 times (Dc 12), with Qh 2 on both kinds. A global projection is q | raw k, and CY makes v = the raw k
  (the weight PE stores it v-normalised). The sliding ring is 8 x 128 entries, exactly the 1024 window, so `RING 1`
  runs a sliding layer's mixer over the whole ring with fixed-length vector ops (no window mask). The
  global context is 32 x 128 = 4,096.
- Head: 54 columns + control (its tiles in two files, `hw0` / `hw1`: one is over the 2 GiB a staged file may take), tied (`head_g12.csl` reads the winner's embedding from its f16 lm_head tile). A head
  column takes a group of CX rows: attention rows in sixes, plain rows in fives, no group across the CY row. The
  group's CX PEs chain their x to the group's root on the last unit's CX column, and the root sends it east. The
  wafer loop's embedding returns to unit 0's root row and down the same chain.
- Tools: `tools/split/g12`. `geo/g12.rs` rcfg words are identical to `split_geo_g12.py` on all 846,094 PEs.

**gpt-oss-20b** (`*_gptoss*.csl`).
- Experts: the released MXFP4 values re-encoded losslessly as FP8 E5M2 (the high byte of an f16) with one power of
  two per (layer, expert, matrix); `prep_gptoss.py` checks every value. All 32 experts are resident on every MoE PE.
- Unit: attention columns | CX | MoE columns (12 units, K 2: a sliding and a full layer), PY 160, Hc 18. Attention
  64 columns (kv head h on 8 columns, Dc 8), with sinks and YaRN RoPE on its CY; MoE 288 columns (Ic 10) with a
  vector top-4 router in f32 on its CY (a max tree, only the taken leaf's path redone).
- Head: 201,088 untied rows in a band below the units; logits are prescaled (`HSCALE`, 2^15) and the host divides.
- Pass criterion: the bf16 golden, with mismatches classified by `tools/emu/classify_tf.py` (bf16 top-1 tie, router
  tie at the step or before it); the f32 reference is secondary.
- Serving build (the mkfull defaults): `GLUV 1` (SwiGLU as vector ops), `CYV 2` (the attention CY's exp as a vector
  op, bit-identical). `CYV 1` / `3` and the stamps (`TSTAMP`) are opt-in.
