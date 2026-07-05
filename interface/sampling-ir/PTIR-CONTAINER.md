# PTIR Trace Container — wire format (`PTIR` v1)

The host↔driver encoding for a **traced pass** (thrust-3 P0.2): stage-tagged
programs over the closed first-party op set, channel declarations,
descriptor-port bindings, and the name table for second-party kernels/sinks.
Realizes `docs/ptir/overview.md` §1–§5 + §7.1; supersedes nothing — the shipped
PSIR v4 sampler bytecode (`BYTECODE.md`) is untouched and its programs keep
their hashes.

Producer/consumer is `pie-sampling-ir::ptir` (`src/ptir/{op,container}.rs`);
the C++ driver implements an independent reader against *this document* plus
the generated `include/ptir_abi.h` (op tags, stage/port/intrinsic ids — do not
hand-copy ids; include the header). On disagreement, this doc + the round-trip
tests in `src/ptir/container.rs` are the source of truth.

**Identity (contract C3):** `container_hash = FNV-1a64(canonical bytes)` — the
same FNV as `program_hash` (offset `0xcbf29ce484222325`, prime `0x100000001b3`).
Canonical = the encoder's deterministic output under the sortedness rules in
§2. Channel **seed values are not in the container** (per-instance data, D2):
two instances differing only in seeds share one identity by construction.

---

## 1. Conventions

- **Endianness:** little-endian; tightly packed, no padding; forward-cursor
  reader; self-delimiting (every count explicit).
- **Primitives:** `u8`, `u16` (2 LE), `u32` (4 LE — value ids, channel indices,
  counts, dims), `f32` (raw IEEE-754 bits).
- **`Shape`** — `rank:u8 | dims[rank]:u32`; rank ≤ 4 (reject larger).
- **`Literal`** — 5 bytes, `dtype:u8 | value:u32` (raw bits per dtype).

### Enum tags

| Enum | Tag → value |
|---|---|
| **DType** | `0`=F32, `1`=I32, `2`=U32, `3`=Bool, `4`=**ACT** (channel decls only: late-bound activation dtype; program-side ops see F32) |
| **Predicate** | `0`=RankLe, `1`=CummassLe, `2`=ProbGe (payload = a value id, 5 bytes total) |
| **RngKind** | `0`=Uniform, `1`=Gumbel |
| **Stage** | `0`=prologue, `1`=on_attn_proj, `2`=on_attn, `3`=epilogue |
| **Port** | `0`=embed_tokens, `1`=embed_indptr, `2`=positions, `3`=pages, `4`=page_indptr, `5`=kv_len, `6`=w_slot, `7`=w_off, `8`=readout, `9`=attn_mask |
| **HostRole** | `0`=none (device-private), `1`=host-writes, `2`=host-reads |
| **Intrinsic** (u16) | `0`=logits, `1`=mtp_logits, `2`=hidden, `3`=query, `4`=value_head, `5`=layer |

## 2. File structure

```
Header
Name[n_names]
ChannelDecl[n_channels]
PortBinding[n_ports]      (sorted by port tag, unique)
StageProgram[n_stages]    (sorted by stage tag, unique — one program per stage)
```

### 2.1 Header (24 bytes)

| Offset | Field | Type | Notes |
|---|---|---|---|
| 0x00 | `magic` | 4×u8 | ASCII `"PTIR"` |
| 0x04 | `version` | u16 | `1` |
| 0x06 | `flags` | u16 | reserved, `0` |
| 0x08 | `n_names` | u32 | name-table entries |
| 0x0C | `n_channels` | u32 | |
| 0x10 | `n_ports` | u32 | |
| 0x14 | `n_stages` | u32 | |

### 2.2 Name — `len:u16 | utf8[len]`

Second-party kernel/sink names (`kernel_call`/`sink_call` reference by index).
First-party sink names are ordinary entries (`"attn_page_mask"`, `"lora"`,
`"minference_sparse"` — scopes are spec-owned, see `ptir_abi.h`).

### 2.3 ChannelDecl — `dtype:u8 | shape:Shape | capacity:u32 | host_role:u8 | seeded:u8`

`capacity ≥ 1` (a capacity-N channel lowers to a ring of N+1 cells, §7.1).
`seeded = 1` ⇔ `Channel::from(v)`: the cell starts full; the **value** arrives
at instantiation (per-instance data). SPSC endpoints: `host_role` names the
host end; the pass is the other. Bind rejects a stage `put` on a host-written
channel and any stage/port consumption of a host-read channel.

### 2.4 PortBinding — `port:u8 | src:u8 | payload`

- `src = 0` (channel): `chan:u32`. Contents are read at execution time (C1).
- `src = 1` (const): `dtype:u8 | shape:Shape | data[numel × elem]` where
  `elem` = 4 (F32/I32/U32) or 1 (Bool). Trace-known contents (e.g. a
  rectangular `indptr` folded to a constant).

Consumption is fixed per port (§5.1): the **token family takes** —
`embed_tokens`, `positions`, `w_slot`, `w_off` (a token is spent by the pass
that embeds it); geometry and masks (`pages`, indptrs, `kv_len`, `readout`,
`attn_mask`) **read** (peek). This feeds the descriptor row of the readiness
table.

### 2.5 StageProgram — `stage:u8 | n_ops:u32 | Op[n_ops]`

A flat SSA op list. No input/output tables: values enter via `chan_take` /
`chan_read` / `intrinsic_val` / `const`, effects leave via `chan_put` /
`sink_call`. Op at position `p` defines `next_id .. next_id+results` ids
(`sort_desc`/`top_k` = 2, `chan_put`/`sink_call` = 0, else 1); operands
reference earlier ids only. Per-layer stages (`on_attn_proj`, `on_attn`) run
once per layer; boundary stages once per pass.

## 3. Op records

Tag byte then the fixed operand layout. Tags are **shared with PSIR v4** where
the op coincides — a v4 reader's table extends, it does not fork. Value
operands are `u32` value ids. `0x80` (v4 `input`) is reserved-unused in PTIR.

| tag | op | operands after tag | results |
|---|---|---|---|
| 0x01–0x06 | `exp log neg recip abs sign` | `a:u32` | 1 |
| 0x07 | `cast` | `value:u32, dtype:u8` | 1 |
| 0x10–0x15 | `add sub mul div max_elem min_elem` | `a:u32, b:u32` | 1 |
| 0x16–0x1B | `gt ge eq ne lt le` | `a:u32, b:u32` | 1 (Bool) |
| 0x1C/0x1D | `and or` | `a:u32, b:u32` | 1 (Bool) |
| 0x1E | `not` | `a:u32` | 1 (Bool) |
| 0x1F | `rem` | `a:u32, b:u32` | 1 |
| 0x20 | `select` | `cond:u32, a:u32, b:u32` | 1 |
| 0x30–0x33 | `reduce_sum/max/min/argmax` | `v:u32` | 1 |
| 0x38 | `broadcast` | `value:u32, shape:Shape` | 1 |
| 0x39 | `reshape` | `value:u32, shape:Shape` | 1 |
| 0x3A | `transpose` | `v:u32` | 1 |
| 0x40/0x41 | `cumsum cumprod` | `v:u32` | 1 |
| 0x50 | `sort_desc` | `v:u32` | **2** |
| 0x51 | `top_k` | `input:u32, k:u32` (k trace-known) | **2** |
| 0x55 | `matmul` | `a:u32, b:u32` | 1 |
| 0x58 | `pivot_threshold` | `input:u32, predicate:5B` | 1 (Bool) |
| 0x60 | `gather` | `src:u32, idx:u32` | 1 |
| 0x61 | `gather_row` | `src:u32, idx:u32` | 1 |
| 0x62/0x63 | `scatter_add scatter_set` | `base:u32, idx:u32, vals:u32` | 1 |
| 0x64 | `iota` | `len:u32` | 1 (U32 `[len]`) |
| 0x65 | `mask_apply_packed` | `logits:u32, mask:u32` | 1 |
| 0x70 | `rng` | `stream:u32, shape:Shape, kind:u8` | 1 |
| 0x71 | `rng_keyed` | `state:u32, shape:Shape, kind:u8` | 1 |
| 0x81 | `const` | `literal:5B` | 1 |
| 0x90 | `chan_take` | `chan:u32` | 1 |
| 0x91 | `chan_read` | `chan:u32` | 1 |
| 0x92 | `chan_put` | `chan:u32, value:u32` | **0** |
| 0xA0 | `intrinsic_val` | `intr:u16, dtype:u8, shape:Shape` | 1 |
| 0xA1 | `kernel_call` | `name:u16, dtype:u8, shape:Shape, n_args:u8, args:u32[n]` | 1 |
| 0xA2 | `sink_call` | `name:u16, n_args:u8, args:u32[n]` | **0** |

## 4. Typing rules (bind re-derives; container does not carry types)

- `cast`: any → declared dtype, shape preserved. `rem`/`div`: same numeric
  dtype (int `%` / truncating int division, 0 on divide-by-zero; F32 `fmod` /
  division). PTIR `div` is numeric-wide — §6.2's `parent = div(i, V)` is id
  math (PSIR v4's `div` stays F32-only).
- `and/or/not`: Bool. Comparisons: same numeric dtype → Bool.
- `reshape`: numel preserved. `transpose`: rank-2 `[m,n] → [n,m]`.
- `top_k`: F32 rank 1/2, `1 ≤ k ≤ last_len`; `[n]→[k]` / `[m,n]→[m,k]`,
  values F32 + indices U32 (value-first, two consecutive ids).
- `matmul`: `[m,k]×[k,n]→[m,n]` F32.
- **`gather` (axis-0, generalized):** `src [n, rest..]`, `idx` int any rank
  (scalar ok) → `idx.dims ++ src.dims[1..]` (result rank ≤ 4), dtype = src.
  Rank-1 src + rank-1 idx ≡ the v4 element gather. Out-of-range → fill-0.
- **`scatter_add`/`scatter_set` (axis-0, generalized):** `base [n, rest..]`,
  `idx` int shape `S`, `vals` shape `S ++ base.dims[1..]` **or scalar**
  (broadcast) → base's type. Out-of-range index skips; duplicates: `set` =
  index order, **last wins** (load-bearing, §6.2's `heir`); `add`
  accumulates. `scatter_add` requires numeric base.
- `chan_take`/`chan_read` yield the channel's declared `(shape, dtype)` with
  ACT materialized as F32; `chan_put` requires exactly that type.
- `intrinsic_val`/`kernel_call` declare their (trace-known) result types;
  bind cross-checks intrinsics against the model (e.g. `logits` =
  `[n_out, vocab]` F32) and checks stage scope: `logits`/`mtp_logits`/
  `hidden`/`value_head` epilogue-only, `query`/`layer` attn-taps-only.
  `mtp_logits`/`value_head` are model-gated; kernel/sink names are bind-time
  availability; a non-replayable kernel is rejected (T10).
- Sinks (T11): pass-wide scope ⇒ prologue only; attention scope ⇒ prologue
  (all layers) or `on_attn_proj` (that layer). Sinks never appear at
  `on_attn`/`epilogue`.

## 5. Numeric + RNG contract (T8 — replay determinism; pinned)

The tier-0 reference interpreter (`ptir::interp`, feature `eval`) is the
normative golden model; every backend matches it **bit-for-bit on integer
outputs and selections** (argmax/top-k indices, sort orders, mask bits) and to
ULP-level on floats:

- **argmax:** lower index wins ties; **NaN is never selected** (an all-NaN row
  yields index 0).
- **sort_desc / top_k:** descending; ties → lower original index first; NaN
  orders below −inf (i.e. last). `top_k` ≡ first k of `sort_desc` order.
- **rank_le(k):** keep element iff `#{strictly greater} < k` — ties at the
  boundary may admit more than k elements. **cummass_le(p):** inclusive
  nucleus over the sort_desc order (keep while *exclusive* prefix mass < p).
  **prob_ge(t):** `x >= t`.
- **`rng` (0x70, ambient seed):** unchanged from BYTECODE.md §5
  (`seed_eff(S, stream)`, SplitMix64 mix, `sample_temp.cu` bit-parity).
- **`rng_keyed` (0x71, `state = [key, ctr]` U32):**
  `seed64 = splitmix64(((key as u64) << 32) | ctr)`; element `j` (row-major
  flat index over `shape`) draws `u = hash_uniform(seed64, j)` with the same
  `splitmix64` / `hash_uniform` as BYTECODE.md §5; `uniform = u`,
  `gumbel = -log(-log(u))`. Pure function of `(key, ctr, j)` — a replayed
  pipeline reproduces its tokens exactly.

## 6. Execution semantics the container does not encode

Overview §1 + §7.1, normative recap: per-**phase** readiness in the order
`prologue → descriptor → on_attn_proj → on_attn → epilogue` (the first op per
channel across that order names the required bit: take/read ⇒ full, leading
put ⇒ empty — the emitted readiness table, `ptir_abi.h` direction enum); a
missing input dummy-runs the instance on each cell's last committed value;
pass-atomic commit (no take consumes, no put lands unless every stage's check
passed); within a pass a channel is a register (put-then-take reads pending,
double-put last-wins); epoch-ring bump at pass end; faults/deadline poison.
In-place lowering classes (`full_ring` / `in_place` / `in_place_undo`) are
computed at bind (`classify_channels`) — semantics-invisible, perf-only.

## 7. Worked example

See `src/ptir/validate.rs::tests::section3()` — the overview §3
greedy+grammar-mask pipeline (5 channels, 3 ports, 1 epilogue) — and the
golden vectors under `tests/golden-ptir/` (hex container + expected verdicts +
tier-0 results), the conformance suite for any backend.
