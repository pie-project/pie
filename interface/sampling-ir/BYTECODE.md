# Sampling IR Bytecode — wire format (`PSIR` v4, shape-typed, binding-free)

The **internal host↔driver encoding** for the Pie Sampling IR. The inferlet front
door is the structured WIT `op-kind` surface (`tensor.wit`); the host lowers
WIT-ops → this IR (1:1, see the `Op↔op-kind` oracle in `src/witmap.rs`) →
`encode()` → these bytes → bridge → driver. Producer/consumer is
`pie-sampling-ir` (`src/bytecode.rs`); the C++ driver implements an independent
reader against *this document*. On disagreement, this doc + the round-trip tests
in `src/bytecode.rs` are the source of truth.

**Status: v4 (shape-typed, binding-free).** A `program` is typed **input slots**
+ a flat SSA **op list** + declared **outputs**. The **binding** (which slot is
logits vs a host tensor, and submit/late readiness) is **attach-time** (the WIT
forward-pass) and is **NOT in the bytecode** — so one compiled program reuses
across attaches ("construct once, attach by handle"). Layout differs from v1–v3,
so a v4 reader accepts **only `version == 4`**. Within v4, evolution is additive.

---

## 1. Conventions

- **Endianness:** little-endian. **Tightly packed, no padding.** Reader = forward
  cursor.
- **Primitives:** `u8`, `u16` (2 LE), `u32` (4 LE — all value ids, input indices,
  counts, dims), `f32` (raw IEEE-754 bits).
- The encoded program is **self-delimiting** (every count explicit).

### Compound primitives

**`Shape` — `rank:u8 | dims[rank]:u32`:** rank 0 = scalar `[]`, 1 = `[n]`, 2 =
`[m,n]`. Rank must be `≤ MAX_RANK` (4); a reader rejects larger.

**`Literal` — 5 bytes, `dtype:u8 | value:u32`** (raw bits per dtype).

### Enum tag tables (in the bytecode)

| Enum            | Tag → value |
|-----------------|-------------|
| **DType**       | `0`=F32, `1`=I32, `2`=U32, `3`=Bool |
| **Predicate**   | `0`=RankLe, `1`=CummassLe, `2`=ProbGe |
| **RngKind**     | `0`=Uniform, `1`=Gumbel |
| **OutputKind**  | `0`=Token, `1`=Distribution, `2`=Logits, `3`=Logprobs, `4`=Entropy, `5`=Scalar, `6`=Embedding |

(Binding / Readiness are **attach-time**, not encoded here.) `OutputKind`
discriminants are a frozen wire contract mirroring the SDK enum; the WIT front
door carries bare output value-ids (typed-tensor outputs), but the bytecode keeps
`OutputKind` for the host's typed-channel marshaling until WS5.

---

## 2. File structure

```
Header
InputDecl[n_inputs]
Op[n_ops]
Output[n_outputs]
```

### 2.1 Header (20 bytes)

| Offset | Field        | Type | Notes                          |
|--------|--------------|------|--------------------------------|
| 0x00   | `magic`      | 4×u8 | ASCII `"PSIR"` = `50 53 49 52` |
| 0x04   | `version`    | u16  | `4`                            |
| 0x06   | `flags`      | u16  | reserved, must be `0`          |
| 0x08   | `n_inputs`   | u32  | number of `InputDecl` records  |
| 0x0C   | `n_ops`      | u32  | number of `Op` records         |
| 0x10   | `n_outputs`  | u32  | number of `Output` records     |

Reject magic ≠ `"PSIR"` or `version != 4`.

### 2.2 InputDecl record

`dtype:u8 | shape:Shape` — a **typed input slot**. Slot `i` is referenced by
`op-kind input(i)` / [`Op::Input`]`(i)`. **Bit 7 of the `dtype` byte is the
readiness flag** (DType tags are `0..=3`, so the high bit is free): clear ⇒
`Submit`, set (`tag | 0x80`) ⇒ `Late` (the value is injected per-fire before its
first consuming op — e.g. a grammar mask computed post-logits). **Additive:** v4
bytecode never set the bit, so it decodes as `Submit`; a Late-input program is a
distinct recognized shape (it rides the bytecode `program_hash` hashes over). The
*source* binding (logits vs a host tensor) is still attach-time.

### 2.3 Op record

`op_tag:u8` + a fixed operand layout (§4). `input`/`const` are value-producing
ops. Result value ids are implicit (§3).

### 2.4 Output record (5 bytes)

`value:u32 | kind:u8` (`OutputKind`).

---

## 3. SSA value-id model

One flat SSA space over the op list. `next_id` starts at `0`; for each op:
`result_id_0 = next_id; next_id += result_count(op)` (2 for `SortDesc` `0x50`,
1 otherwise). `input(index)` (`0x80`) and `const` (`0x81`) are leaf ops that
define ids; `input(index)` materializes input slot `index`. `SortDesc` reserves
two consecutive ids, value-first (`r` = sorted F32 `[n]`, `r+1` = indices U32
`[n]`). Operands reference earlier ids (no forward refs).

---

## 4. Op table

| tag   | Op             | Operands (after tag)                  | Results | Result type |
|-------|----------------|---------------------------------------|---------|-------------|
| `0x80`| `Input`        | `index:u32`                           | 1 | `inputs[index]` type |
| `0x81`| `Const`        | `literal:Literal(5B)`                 | 1 | `scalar(literal.dtype)` |
| `0x01`–`0x06`| `Exp/Log/Neg/Recip/Abs/Sign` | `a:u32`         | 1 | same as `a` |
| `0x10`–`0x15`| `Add/Sub/Mul/Div/MaxElem/MinElem` | `a:u32, b:u32` | 1 | broadcast(a,b) |
| `0x16`–`0x18`| `Gt/Ge/Eq`     | `a:u32, b:u32`                  | 1 | broadcast(a,b), **Bool** |
| `0x20`| `Select`       | `cond:u32, a:u32, b:u32`              | 1 | broadcast; `cond` Bool |
| `0x30`–`0x32`| `ReduceSum/Max/Min` | `v:u32`                    | 1 | drop-last(v) |
| `0x33`| `ReduceArgmax` | `v:u32`                               | 1 | drop-last(v), **I32** |
| `0x38`| `Broadcast`    | `value:u32, shape:Shape`              | 1 | `{shape, value.dtype}` |
| `0x40`/`0x41`| `CumSum/CumProd` | `v:u32`                         | 1 | same as `v` (F32) |
| `0x50`| `SortDesc`     | `v:u32`                               | **2** | `([n],F32)`,`([n],U32)` |
| `0x58`| `PivotThreshold`| `input:u32, predicate:Predicate(5B)` | 1 | `{input.shape, Bool}` |
| `0x60`| `Gather`       | `src:u32, idx:u32`                    | 1 | `[idx.len]`, dtype=src |
| `0x61`| `GatherRow`    | `src:u32, idx:u32`                    | 1 | `[src.rows]`, dtype=src (per-row pick) |
| `0x62`/`0x63`| `ScatterAdd/Set` | `base:u32, idx:u32, vals:u32`   | 1 | `[base.len]`, dtype=base |
| `0x70`| `Rng`          | `stream:u32, shape:Shape, kind:u8`    | 1 | `{shape, F32}` |

`Predicate` (5 bytes): `tag:u8 | payload:u32` — `payload` is a **value id** for all
three: RankLe = a Scalar/`[rows]`-`U32` `k`; CummassLe/ProbGe = a Scalar/`[rows]`-F32
threshold. (#25: `k` is host-submit like top-p `p`, so top-k bytecode is k-invariant.)
"drop-last" = last axis removed.

---

## 5. Semantics the bytecode does not encode

- **Indexing:** `ScatterAdd/Set` SKIP an index `<0` or `>= len(base)`; `Gather`/
  `GatherRow` FILL-0 an out-of-range index.
- **`CummassLe(p)`** = inclusive nucleus (keep token iff exclusive prefix mass
  `< p`); **`ProbGe`** inclusive `>=`; **`RankLe(k)`** ties → lower index. For a
  matrix input the `k`/threshold operand may be a shared **scalar** *or* a per-row
  **`[rows]`** vector (one `k`/threshold per row — batched top-k/top-p/min-p).
- **RNG (ambient seed + static stream):** no seed operand; the per-fire seed `S`
  is the runtime's per-row `sample_seed`, folded by codegen. `stream` decorrelates
  multiple `Rng` ops. `splitmix64(x): x^=x>>27; x*=0x3C79AC492BA7B653; x^=x>>33;
  x*=0x1C69B3F74AC4AE35; x^=x>>27`. `seed_eff(S,stream) = (S ^ 0xA5A5A5A5) ^
  splitmix64(stream * 0x9E3779B97F4A7C15)`. **`stream=0` ⇒ `S ^ 0xA5A5A5A5`
  exactly** (`sample_temp.cu` bit-parity). The **RNG axis is a lowering choice**,
  not in the wire format: the standard de-hardwiring samplers lower **batch-axis**
  — per-row seed `S[r]` (ambient `row_seeds[r]`) over the in-row index `col=j`
  (`sample_temp.cu` parity, what hotel's removal-gate proved). The flattened
  `row*len+col` single-seed form is **spec-verify-ONLY** (the `[rows,len]`
  draft-verification geometry — one `S` over draft positions, not independent
  batch rows). Same `Rng{stream:0}` bytecode either way; the axis follows the
  lowering (`batched=true ⇒ col=j`) + seed source, never the program.
- **`Broadcast`** = left-aligned replicate (`value` dims left-aligned vs `shape`,
  trailing axes padded with `1`; each equal or `1`). Folds scalar-broadcast and
  per-row `[m] → [m,n]`. Preserves `value`'s dtype.
- **`GatherRow`** = per-row column pick `out[i] = src[i, idx[i]]` (`src` `[m,n]`,
  `idx` `[m]`), the lossless accept-ratio `p[i, draft[i]]` — **not** whole-row
  select.
- **Matrix (rank ≥ 2)** reductions/scans/argmax/pivot are **per-row**; no
  cross-row reduce. Binary ops: `Matrix⊕Matrix` (same shape) or `Matrix⊕Scalar`;
  lift a `[m]` to `[m,n]` with `Broadcast` first.
- **Binding & late-bind (attach-time):** binding (logits / tensor key / submit-
  late readiness) is supplied at the forward-pass attach, per input slot — not in
  the bytecode. **A host binding MUST cover the input decl's full extent**: the
  bound tensor's byte length equals `numel(decl.shape) × dtype_size(decl.dtype)`
  (a `[k, n]` matrix decl means k FULL rows). The runtime/executor rejects a
  short binding loudly at bind/fire time — it never launches; a silently short
  matrix input (e.g. a 2-row grammar mask against a `[4, vocab]` decl) is
  undefined on-device behavior (the §6.1 psir_k0 OOB incident, 2026-07-05). For a slot bound to a *late* tensor, `input_first_use(program,
  index)` gives the first consuming op = the runtime's inject-before barrier
  (miss = skip).
- **Output-kind ⟂ dtype:** `Token` ⇒ integer value; every other kind ⇒ F32; a
  `Bool` cannot be a declared output.
- **No automaton/grammar ops** — masks enter as host tensor inputs.

---

## 6. Worked example (byte-accurate)

Greedy **argmax** over `logits[32000]` (input slot 0), output kind `Token`.

```
inputs:  [ {[32000], F32} ]                 // slot 0
ops:     [ Input(0),                        // id 0
           ReduceArgmax(0) ]                // id 1 : scalar i32 token
outputs: [ (value:1, kind:Token) ]
```

Encoded (41 bytes):

```
0000: 50 53 49 52 04 00 00 00 01 00 00 00 02 00 00 00
0010: 01 00 00 00 00 01 00 7d 00 00 80 00 00 00 00 33
0020: 00 00 00 00 01 00 00 00 00
```

| Bytes                       | Meaning |
|-----------------------------|---------|
| `50 53 49 52`               | magic |
| `04 00`                     | version = 4 |
| `00 00`                     | flags |
| `01 00 00 00`               | n_inputs = 1 |
| `02 00 00 00`               | n_ops = 2 |
| `01 00 00 00`               | n_outputs = 1 |
| `00`                        | inputs[0].dtype = F32 |
| `01`                        | inputs[0].shape rank = 1 |
| `00 7d 00 00`               | inputs[0].dims[0] = 32000 |
| `80`                        | op0 = Input |
| `00 00 00 00`               | input index = 0 |
| `33`                        | op1 = ReduceArgmax |
| `00 00 00 00`               | operand v = 0 |
| `01 00 00 00`               | output[0].value = 1 |
| `00`                        | output[0].kind = Token |

---

## 7. Minimal C++ reader sketch

```cpp
struct Cursor { const uint8_t* p; /* u8/u16/u32 LE readers */ };
struct Shape { uint8_t rank; uint32_t dims[4]; };
Shape read_shape(Cursor& c) {
  Shape s; s.rank = c.u8();                  // reject s.rank > 4
  for (uint8_t i=0;i<s.rank;++i) s.dims[i]=c.u32();
  return s;
}

if (memcmp(c.p, "PSIR", 4) != 0) fail(); c.p += 4;
if (c.u16() != 4) fail();                    // version
c.u16();                                     // flags
uint32_t n_inputs  = c.u32();
uint32_t n_ops     = c.u32();
uint32_t n_outputs = c.u32();
// inputs:  n_inputs × { uint8_t dtype = c.u8(); Shape sh = read_shape(c); }
// ops:     switch(op_tag) — 0x80 Input{index:u32}, 0x81 Const{literal}, …
//          next_id = 0; advance by 2 for SortDesc (0x50) else 1.
// outputs: n_outputs × { uint32_t value = c.u32(); uint8_t kind = c.u8(); }
```
