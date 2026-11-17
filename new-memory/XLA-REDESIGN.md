# XLA: structure is compiled, sizes and cases are data

Draft, 2026-11-17. TPU v6e, libtpu 0.0.48. Starts from a measured census of
what an XLA load compiles today, then states the design and what each step
of it cuts. Companion to `REDESIGN.md` (the ledger) and `XLA-POOL.md` (what
lives in the pool).

## North star

> An XLA load compiles a handful of programs, each in seconds, and a fire
> never compiles. A program's structure is compiled; every size and every
> case a fire can take is data.

The contract poem states for XLA: **a tensor dimension is a model constant
or T** (the fire's total rows, from a short ladder). Everything else (how
many lanes, which lanes decode or prefill, how long each one's context is,
which pages it holds, which recurrent slot it owns) is a ragged descriptor:
scalars and small integer arrays the program reads, not shapes it is
compiled for. vLLM-TPU and SGLang-JAX serve TPUs this way: one shape axis
(total tokens), everything else data.

## 1. Census: what a load compiles today

Method: a probe (not committed) on `sc/xla-int-divide` 4a144d566 logs every
program the warm ladder compiles: the ladder shape that traced it, its
StableHLO text bytes, the generated code and temporaries
(`GetCompiledMemoryStats`), its compile time, and its `case` / `while`
ops by the source op they lower (the `loc` of the op). Each load ran
cold (`PIE_XLA_CACHE` empty), default config (`prewarm = true`, c64, the
default token ladder up to 2560 rows, page 16, context 4096).

### 1.1 Totals

| | Qwen3-0.6B (dense, 28 layers) | Qwen3.5-0.8B (hybrid) |
|---|---|---|
| ladder fires | 202 | 275 |
| distinct programs | 143 | 253 |
| StableHLO text | 1.9 GB | 2.7 GB |
| generated code | 27.3 GB | 42.1 GB |
| largest temporaries | 179 MiB | 353 MiB |
| compile, summed over programs | 2989 s | 5321 s |
| compile, wall (8 threads) | 1454 s | 2150 s |

Generated code is resident in HBM once a program first executes (measured
11-15: `bytes_in_use` grows by exactly the program's code), and PJRT holds
about 2.1x it on the host per loaded program. A load cannot hold its whole
ladder resident: 27 GB of code is most of a 32 GiB v6e.

### 1.2 By family: which axes multiply

| family | Qwen3-0.6B programs | code | axes it is compiled for | Qwen3.5-0.8B programs | code |
|---|---|---|---|---|---|
| decode | 35 | 2.2 GB | lane rung (7: 1..64) × page rung (5: 16..256) | 35 | 0.8 GB |
| prefill | 17 | 3.0 GB | row rung (17), 1 lane | 110 | 18.5 GB |
| mixed (prefill beside decodes) | 91 | 22.1 GB | decode lane rung (7) × row rung (13) | 108 | 22.8 GB |

- A decode program is compiled per lane count and per page bound: the page
  bound is a shape (the walk's extent), not data.
- A mixed fire multiplies the decode lane rung by the prefill row rung:
  64% of the 0.6B's programs and 81% of its code.
- A hybrid's prefills multiply rows by prompt count (1, 3, 5, 7, 9, 15, 17,
  31, 33, 63 lanes): its chunked scan is compiled per lane count
  (`canonical_pads`), so the prefill family grows 6.5x over the dense one.
- Mean code per program: decode 63 MB, prefill 176 MB, mixed 243 MB (0.6B).

### 1.3 Runtime branches compiled in

Counted by the op each `case` / `while` lowers, every program of the 0.6B
ladder:

| op | `case` | `while` |
|---|---|---|
| `attn.kv_append` | 2996 | 0 |
| `attn.decode` | 504 | 2240 |
| `attn.prefill` | 364 | 5768 |

- Every branch is per layer: the largest program holds 56 `case` and 56
  `while` (28 layers × 2). Each `case` compiles all its arms into every
  layer, and the code of the arms not taken is resident too.
- `attn.kv_append`'s arms (rows in place, page rows, kernel) are already
  one static body per program on `sc/xla-append-static` 32f594393 (not in
  this tree): that cuts 2996 `case` to 0 and was the main cause of the
  ladder OOM of 11-16.
- `attn.decode` / `attn.prefill` choose flash against the paged walk at
  run time, per layer; the `while` ops are the walks over pages.
- The hybrid's programs hold 6-12 `case` (no append arms there) and up
  to 397 custom calls (its gated-delta kernels).

### 1.4 Compiles during serving

With the ladder warm, a fire still compiles when its shape is off the
ladder. Measured 11-17 on the 0.6B (`prewarm = true`):
`mtp_speculative_decoding`'s draft and verify fires compiled model programs
of 5 MB and 14 MB text at 15 s and 24 s, on the engine thread, each
stalling every lane (a 10 s watchdog, and a frame held 25 s). Guest stages
no longer compile in a fire (f346ef5f3); model programs still can.

## 2. The design

### (a) One attention kernel, cases as data

One Pallas kernel in the style of `tpu-inference`
`kernels/ragged_paged_attention/v3` (Apache-2.0, may be read or reused):
the fire's decode, prefill and mixed ranges come from a `distribution`
vector `[decode_end, prefill_end, mixed_end]`, and each request's lengths
and page table are scalar-prefetched data. No `case`: one body serves every
mix, and the walk over older pages is inside the kernel.

Cuts: the 868 attention `case` and the 8008 `while` of the 0.6B ladder,
the arms' resident code, and the flash / walk crossover decision (it stops
being a choice). Prerequisite for (e): a mixed fire stops being its own
program family once attention reads the mix as data.

### (b) Pages off the ladder

Page tables at a fixed maximum capacity (`pages_per_slot` at the load's
context), page counts as data. The decode family stops multiplying by its
page rung.

Cuts: 0.6B decode 35 → 7 programs; Qwen3.5 decode 35 → 7. Measured cost
to settle first (11-15): a full-width walk cost 0.6B d1 −7% and 30B d64
−1.6% at 512 context, which the in-kernel walk of (a) bounds by the
request's real length instead of the table's width.

### (c) Recurrent and conv state as one array a layer, indexed by slot

One `[slots, ...]` array per layer, read and written at each lane's slot
index (data). Lane count stops being a compiled axis of the hybrid's scan.
The arena A/B of 11-17 showed weight placement is not the cause of
Qwen3.5's two per-load speeds; the profiler diff decides that, separately.

Cuts: the hybrid's prefill family multiplies by prompt count today (110
programs, 18.5 GB); with slot-indexed state and per-request lengths as data
it is one program a row rung (17). The chunked scan's speed with lane count
as data was measured as the open risk (11-15, `canonical_pads`): it must be
measured again with the state layout of (c), not assumed.

### (d) A compile during serving is an error

Counted (like `BuiltAfterReady`) and named, never silent. Guest stages:
done (`guest_fire_compiles`, f346ef5f3). Model programs: today the
ladder enumerates shapes by hand (`serve/warm.rs`, 300 lines of cases from
measured misses); with (a)-(c) the only compiled axis is T, so the ladder is
the T rungs and nothing a fire can take is off it.

### (e) Split programs where axes differ

The backbone (embedding through the last layer) is compiled per T rung;
the head (final norm, logits, sampling, guest epilogues) per lane capacity
rung, reading the backbone's last rows by an index vector. Axes add instead
of multiplying.

Cuts, on the 0.6B: today 143 programs (7 × 5 decode + 17 prefill + 7 × 13
mixed). With (a)-(e): about 24 backbone programs (the union of lane rungs
and row rungs as T rungs) plus 7 head programs, about 31. At today's mean
code per program the backbone set is ~4 GB of code against 27 GB; the
layer loop measured 11-15 (one layer body under `stablehlo.while`, code
7-13x smaller, compile 4-7x faster, decode d64 +7% / prefill +13-30%
slower at the time, bitwise only with template weights) is a further cut,
held until (a) removes the per-layer kernel offsets that blocked it.

## 3. Order

1. (a) the RPA-style kernel, gated bitwise (contract A: lanes and adapters
   equal up to the stated rounding) and on speed against today's flash and
   walk at the 0.6B, 8B and hybrid decode / prefill points.
2. (b) pages as data, on (a)'s in-kernel walk.
3. (e) the backbone / head split; the ladder becomes the T rungs.
4. (c) slot-indexed recurrent state; the hybrid's prefill family collapses.
5. (d) with the ladder at T rungs, any fire compile fails by name.

Each step reports this census again: programs, code, temporaries, compile
seconds, and branches by op.
