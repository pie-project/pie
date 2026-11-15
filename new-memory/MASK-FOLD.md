# Mask along the fold direction: step 1 contract

Status (2026-11-15): design accepted by the user; this text carries Pie Leader's rulings on the draft's eight questions (section 7).
Nothing here is built. Step 1 lands on top of the unfolded-contract merge, behind the serve failures and the rung branches.
Owners read: section 2 (every engine), 2.5 (gates per engine), 3 (SDK and the examples), 4 (runtime), 5 (rs, step-2 material).

Cites are `path:line` on origin/memory-redesign at 10d708421, the tip named in the task.
The ref moved to cec91bb4b while this was written; that push changes only `crates/engine-cuda/src/{api.rs,record.rs,serve/enqueue.rs}`,
and cites into those files are at 10d708421. "NP" is origin/sc/np-unfolded-contract (d3bc4cf7b), unmerged, merge-base f8d9104ac.
`†` marks a cite a sub-agent read and I did not re-read. `p/fire.rs` is `crates/runtime/src/pipeline/fire.rs`; `e/fire.rs` is `crates/engine/src/fire.rs`.

Framing fact: the tip still carries `UNFOLDED_ROWS = 16` (e/fire.rs:590), enforced by the runtime (p/fire.rs:608-650) and by engines
(engine-cuda serve/prepare.rs:1903-1914). `rung_rows`, `FireRequest.carried` and `KvWrites.end` exist on NP only (NP e/fire.rs:569-607;
tip `KvWrites` has no `end`, crates/engine/src/engine.rs:29-40).

## 1. Reading check

**a. Working set = frozen `[0, folded)` + buffer `[folded, reserved)`; `fold` acts on kv, rs, window lists. Confirmed, wording imprecise.**
Layout: crates/inferlet/wit/working-set.wit:6-15. `fold` is not a working-set verb: it is the per-lane `fold` channel of `state-geometry`
(forward.wit:25-43), a count of slots frozen after the fire (SDK `Fold::{None, All, Count}`, crates/inferlet/src/eta.rs:441-475). kv: freeze
plus seal of whole logged pages (crates/runtime/src/store/kv.rs:1795-1824; page_table.rs:478-483), applied when the fire settles
(p/fire.rs:4035-4096, kv at 4067-4069). rs: `frozen` advances and buffer pages wholly below it go (store/rs.rs:1873-1896); the scan is the
engine's. Window lists commit through `WindowAdvance` (p/fire.rs:469-506). The host cursor advances `folded` at submit for a host-known
count (p/fire.rs:2549-2552) and marks it inexact for a device-counted one (2513-2519) until settle (4083-4092).

**b. Engines are not told the kv boundary. Confirmed.** `KvDelta { held, pages, translation, window }` (e/fire.rs:169-180); `folded` is only in
`RsVerb::Buffer` (553-567). `held` is the lane's first written slot, not the boundary (pipeline/fire/geometry.rs:477-481). Engines read `held`
as `have`: engine-gpu api.rs:1100-1122†, engine-cuda api.rs:1487-1500, engine-xla api.rs:1394-1460†, engine-metal api.rs:1174-1207†.
`KvDelta.window` has no reader on Vulkan/wgpu or XLA†.

**c. The mask is expanded over `[0, have + rows)` on every engine. Confirmed, ROCm imprecise.**
- Vulkan/wgpu: byte plane, all fire rows x stride, stride = widest `have + rows` (crates/engine-gpu/src/mask.rs:23-124).
- CUDA: per lane a bit rectangle `rows x (have + rows)`, i32 byte offset per lane (crates/engine-cuda/src/mask.rs:20-114).
- ROCm: same staging (engine-cuda drives it), but the paged prefill refuses a mask (crates/kernels-rocm/src/attn.rs:228-230); only the kv-less ragged kernel reads bits†.
- XLA: byte plane rows x stride, stride = widest `have + rows` rounded up to a power of two, at least 256 (crates/engine-xla/src/mask.rs:29-80).
- Metal: byte plane rows x stride, stride = widest exactly (crates/engine-metal/src/mask.rs:22-148†).
- CPU takes no custom mask (engine-cpu forward.rs:601-602†); Cerebras accepts only a complete causal mask (engine-cerebras spatial/engine/fire.rs:283-311†).
Every engine faults when `mask.total < have + rows` (`Fault::Mask`). The guest's key axis is its own (`span`, `kv_max`, `pool_len`), wider
than the extent; the runtime lowers each row to `Mask::new(runs, total = guest key axis)` with no check against kv-len (geometry.rs:591-601).

**d. Fork shares the whole buffer copy-on-write. Confirmed.** kv: page_table.rs:226-245, copy at the first write (store/kv.rs:2147-2198;
pipeline/fire/kv.rs:13-27, test 116-145). rs: `fork` clones the entry and refs every slot (store/rs.rs:850-856). The host child clones the
parent's cursor (inferlet/host/working_set.rs:183-254). WIT and SDK say "a copy of the buffer" (working-set.wit:27-29; eta.rs:416-421).

**e. Stored K is RoPE-rotated on every backend. Confirmed; the FP8 parenthesis is wrong as stated.** Programs rotate, then append
(models/lib/blocks.poem:24-25†, models/qwen_3.poem:1217-1224†); appends copy verbatim (kv_append.slang:47-48†; kernels-xla attn/append.rs:1-60†;
CUDA fused qkv_append.cu:163-196†). `KvScheme::Fp8PerTokenHead` exists in kernels-cuda only (src/attn/kv.rs:13-19†); its scale arrays are
indexed by physical page and offset (kv_append.cu:210-219†), so a copied row needs its scale entry copied to the destination; the engine binds
none at the tip (`key_scales: Tensor::new(0, 0, 0, U8)`, `scheme_byte: 0`, engine-cuda store.rs:1702-1715). Metal's quantized rows keep scales
inside the row (kernels-metal attn/kv_codec.rs:1-32†). A moved row keeps its stated position, so slot != position afterwards (see f).

**f. Metal and wgpu take the causal bound from stated positions; CUDA and XLA from the cache row. Confirmed, and Vulkan is with wgpu.**
engine-gpu serve.rs:2921-2934 (stated position, else `have + at`; cache row under mrope), shader `kp > q_pos` (kernels-slang attn/sdpa.slang:229-242).
Metal serve.rs:3743-3754 (same rule). XLA serve.rs:1921-1929 (`cache_rows = have + at`; positions feed rotary only). CUDA `kv_len = have + rows`
(poem-exec store/kv.rs:437, 491-495†; fa2/prefill.cuh:1256-1260†). On CUDA the masked arm has no causal test in the kernel: causality is in the
staged bits, by row order (engine-cuda mask.rs:89-111).

**g. No kernel reads an unmasked committed run plus a masked tail; lower bounds exist. Confirmed.** A masked row tests a bit on every key:
sdpa.slang:229-242; fa2/variants.cuh:79-93; kernels-xla attn/paged.rs:697-737†; sdpa_paged.metal:208-216, 736-741†. Lower bounds: window
start on all; `Attn::Prefill.floor`/`ReplayFloor` on CUDA (poem-ir ops/attn.rs:69-73, 492-500†) and `ReplayFloor` on XLA and Metal†;
Vulkan/wgpu refuse `KvEnds | WindowSink | ReplayFloor` (engine-gpu dispatch/attn.rs:349-352†). XLA's flash kernel reads two segments
(older pages, fresh rows) but neither masked, and it declines any flagged row (kernels-xla attn/flash.rs:622-625, 676-779†).

**What the reading missed.**
1. Positions of buffered rows: default `slot - lag`, or stated per row (geometry.rs:457-472). Beam states position = depth and slot = pool
   cell (examples/beam-search/src/lib.rs:180-182, 225-226), so slot > position for width > 1. On Vulkan, wgpu and Metal the bound is the
   stated position, which is below a beam row's own slot. Read from code, not run: whether beam is correct there today is not verified.
2. `held` is `start` of the lane's written span (geometry.rs:477-481); `have` on a device-placed lane is device reach minus rows.
   The boundary the runtime holds is `Place.folded` with `folded_exact` (geometry.rs:350-357), from the cursor (working_set.rs:18-31).
3. Window lists and a buffer: on a window model each lane needs a working set of its own (pipeline/fire/rs.rs:443-452) and a windowed
   set holds no rs buffer (store/rs.rs:247-249). Engines take window ids from `Lane::tables`.
4. Guest mask coordinates today: absolute slots, `[rows, n]` with n the guest's choice (section 3). The comments "A bound mask replaces
   the causal bound" (snapkv-eviction:244, trackb-h2o:310, trackb-snapkv:236) describe the guest's view; engines already clip every row at
   `have + q` by row order when staging (engine-gpu mask.rs:105-124; engine-cuda mask.rs:89-111; XLA and Metal the same†).
5. Lanes of one working set in one fire are the same resource repeated in `lanes` (p/fire.rs:2454-2491, `shared` at 2476); they share one
   cursor and must state slots (geometry.rs:420-425). An rs group's first lane writes the state and carries the fold; the others read it
   (pipeline/fire/rs.rs:490-518, 555-559). The engine has no set id on a lane: it sees equal pages.
6. `buffered_reads` already holds rs lanes to row order: a row that hides an earlier row of its own lane is refused (e/fire.rs:150-164),
   and every row must admit the same buffered slots (143-149). It reads the mask in absolute slots (131-135).
7. A device-resolved mask (`FireAttnMask::Device`, geometry.rs:558-562; p/fire.rs:4462-4467) reaches the engine as a port; the engine
   lowers it with `stride = dense.len() / spanned`, the guest's key axis (crates/engine/src/ports.rs:244-274, 318-343).

## 2. `engine::fire` contract text for step 1

### 2.1 The boundary: `KvDelta::folded`

```rust
pub struct KvDelta {
    pub held: u32,
    /// The set's committed boundary, in the set's own slot numbering. Every
    /// slot below it is committed: the lane's rows read `[0, folded)` under
    /// the model's own bounds (causal, window, sink) and no mask bit. The
    /// lane's mask speaks of the slots from `folded` on and of nothing else.
    /// `folded <= held`: a lane writes no committed slot.
    ///
    /// It is the boundary as the runtime knows it when it stages the fire.
    /// While a device-counted fold is in flight it is the boundary before
    /// that fold; the slots the fold freezes are then read through the mask
    /// like any buffered slot. A lane that carries `RsVerb::Buffer` states
    /// the same `folded` there. `KvWrites::end` reports where the lane's
    /// rows ended, in the same numbering.
    pub folded: u32,
    pub pages: Vec<u32>,
    pub translation: Vec<u32>,
    pub window: Vec<u32>,
}
```

Type `u32`, `#[serde(default)]`; the runtime fills it from `Place.folded` where it fills `held` (geometry.rs:477-481) and per piece of a cut
pass. `KvWrites::end` is NP's field (NP crates/engine/src/engine.rs, "One past the last slot the lane's rows landed at, in the set's own slot
numbering"); it comes with NP or with this step. Not verified: that `Place.folded` equals the rs store's `frozen` on every path; rs fires drain
device-counted folds until the cursor is exact (p/fire.rs:662-685), which is the case where they could differ.

### 2.2 What a lane's mask is (doc text for `Lane::mask`)

```rust
/// What the lane's rows read of the slots from `kv.folded` on. Column `c` of
/// a mask is slot `kv.folded + c`. The extent is `end - kv.folded` columns,
/// `end` one past the lane's last row: the buffer the fire found, then the
/// lane's own rows in row order. A set bit admits the slot, a clear bit
/// hides it. A mask's `total` is at least the extent; columns past it are
/// not read. A plane is `rows x (buffer + rows)` bits and no more.
///
/// Among the lane's own rows the order is the row order: row `q` reads own
/// rows `0..=q` its bits admit and no later row, whatever its bits or its
/// stated position say. A wide lane's rows read every own row their bits
/// admit.
///
/// Bits of own rows above the diagonal are not read.
///
/// Lanes of one set in one fire read one buffer, as the fire found it, and
/// their own rows. A slot a sibling lane writes in the same fire is hidden
/// by the guest's mask; where a mask admits it, the lane reads <this fire's
/// row | the slot's content before the fire>, as the engine states (2.5).
///
/// `None`: every row reads every slot before its own and its own.
/// `Masking::Extent`: one mask for every row. `Masking::Rows`: one per row,
/// as many as the lane's rows. A kv-less lane's mask is over its attention
/// group's rows, as before.
pub mask: Option<Masking>,
```

The causal bound is the row order (the cache row) on every backend; a stated position feeds rotation only. This changes Vulkan, wgpu
and Metal for a guest whose stated positions differ from its slots (beam), where they bound by stated position today.

`Mask` keeps its run-length form; `Mask::total` changes meaning from the context to the buffer extent. `buffered_reads` (e/fire.rs:119-167)
reads columns `[0, first - folded)` for the replay and `[first - folded, ..)` for the own-row check.

### 2.3 What an engine refuses

Mirrors NP's wording ("a fault for a wrong count only ... refused by name at prepare and writes nothing", NP e/fire.rs:569-607).
- Fault (a wrong count, the runtime's bug): `mask.total < end - kv.folded` (`Fault::Mask`, extent now the buffer extent); `Masking::Rows`
  count != rows (`Fault::MaskRows`; also e/fire.rs:324-334); `kv.folded > kv.held`.
- Refused by name at prepare, nothing written: a fire whose mask planes, summed over its lanes, pass the mask room of the rung it runs at
  (2.4; the runtime cuts to fit, so this is the runtime's bug or a device-placed fire past the largest rung). This replaces the per-engine ceilings "mask bits" (engine-cuda inputs.rs:919-927), "mask plane bytes" and "key positions in one
  mask row" (engine-gpu inputs.rs:581-600; engine-metal inputs.rs:773-796†), "bytes of one fire's mask plane" (engine-xla mask.rs:69-80).
- Not refused: a long committed part. No figure of this step grows with `kv.folded`.

### 2.4 The mask-plane room

Where engines state fire figures today: `FireLimits` (crates/engine/src/caps.rs:20-40: `max_lanes, max_tokens, max_page_refs, max_context,
prefill_rows, prefill_lanes, max_readouts`); per rung only `ChunkFacts::act_chunks: Vec<(rung tokens, chunks)>` (crates/engine/src/chunk.rs:65-69,
empty where activation has a buffer of its own); `LoadFacts::input_bytes` is one total (crates/engine/src/load.rs:203; XLA reports 0†).
The ledger files staged inputs under `Class::Fire` (crates/engine/src/memory.rs:33-34). **No contract figure bounds the mask plane today.**
The implicit bound is `max_tokens x max_context`, sized in each engine: bytes at load on Vulkan/wgpu (engine-gpu inputs.rs:291, 318-323), bits at
load on CUDA (engine-cuda inputs.rs:630-636), bytes per rung and arm on Metal (engine-metal serve.rs:912-930, inputs.rs:468-471), none reserved
on XLA (mask.rs:62-80). Metal is the one engine whose plane already scales with the rung.

**The room is derived in the contract; no engine states a figure and `FireLimits` gains nothing.**

> A rung of `t` tokens holds `max(t * t, max_context)` mask bits, summed over a fire's masked lanes (`rows * (buffer + rows)` each).

`t * t` is one full-width masked prefill of the rung's own rows over an empty buffer, the thing a rung must be able to run;
`max_context` is one row over a buffer of any length. An engine that stages bytes, a common stride or a rounded stride sizes its own
room to cover that, for its masked classes only: a load whose model has no masked class reserves none. At a top rung of 10240 tokens
the room is 10240 * 10240 bits = 13.1 MB (bytes: 105 MB on an engine that stages a byte a key). An engine that cannot afford `t * t`
at its top rung says so with the figure before step 1 is built.

### 2.5 Gates each engine adds with step 1

- Two-segment read: `[0, folded)` unmasked, bits only at `kp >= folded`, plane indexed by `kp - folded`. ROCm's paged prefill gets the
  same read (it refuses a mask today, crates/kernels-rocm/src/attn.rs:228-230); AMD's.
- A two-lane fire on one set: a lane does not see the row its sibling writes in the same fire when its mask hides it; and what it sees
  when the mask admits it (this fire's row, or the old content), stated in the engine's doc and in 2.2 so it is defined.
- Beam on Vulkan, wgpu and Metal (no test under Vulkan/wgpu today): the row-order causal bound.
- A/B against main, +-1% paired.

## 3. WIT and SDK wording

Current WIT (crates/inferlet/wit/forward.wit:87-90): "`[rows, n]` bools: per row, the slots `[0, kv-len)` it attends when lanes read kv, else
its attention group's rows in lane order. Set before `program`." Current SDK (crates/inferlet/src/eta.rs:650-660): "Per row, the rows it
attends: kv slots if the lanes read kv, else its group's rows." Bindings: packages/javascript/inferlet/src/eta/bridge.ts:605-607;
packages/python/inferlet/src/inferlet/eta/bridge.py:655.

Proposed WIT:

```wit
/// `[rows, n]` bools. Where lanes read kv: per row, which slots of its
/// working set's buffer and of the pass's own rows it attends. Column `c`
/// is slot `folded + c`, `folded` the set's frozen length when the pass
/// runs; `n` covers the buffer and the lane's rows. The frozen part is
/// not the mask's: every row attends it under the model's own bounds. A
/// row attends no row of its lane after it, whatever its bits say. Left
/// out: causal. Where lanes read no kv: per row, its attention group's
/// rows in lane order. Set before `program`.
mask: func(rows: borrow<channel>) -> result<_, error>;
```

Proposed SDK: "Per row, what it attends past the frozen part: column `c` is slot `folded() + c` of its lane's working set, the buffer and then
the pass's rows. Frozen slots are always read. Kv-less lanes: its group's rows." `state-geometry.kv-len` ("the lane reads slots `[0, kv-len)`",
forward.wit:25-43) is unchanged; kv-len beside a mask already leaves the decode envelope (geometry.rs:49-55).

Call sites (`fwd.mask`), and whether the mask hides a committed row today:

| Site | Mask | Hides committed rows |
|---|---|---|
| examples/attention-sink/src/lib.rs:142 | prefill `[n, span]` causal | no (own rows) |
| examples/attention-sink/src/lib.rs:169 | decode `[1, span]` sink + window, device-carried, `folding()` (170) | yes: between sink and window |
| examples/contrastive-decoding/src/lib.rs:128, 160 | `[2n, span]`, `[2, span]`; amateur lane windowed (117-120), `folding()` | amateur lane yes; expert lane no |
| examples/naive-masked/src/lib.rs:138 | prefill causal `[n, span]` | no |
| examples/naive-masked/src/lib.rs:179 | decode `[1, span]`; modes dense/structured `j <= base` | no; mode doc-isolation yes (hides `[0, doc_start)`, 166-171) |
| examples/trackb-h2o/src/lib.rs:323 | `[1, kv_max]` keep-set and causal, `folding()` | yes |
| examples/trackb-snapkv/src/lib.rs:249 | `[1, kv_max]` keep-set and causal, `folding()` | yes |
| examples/snapkv-eviction/src/lib.rs:258 | `[q, kv_max]`; prefill folds `ask_at` rows (179-185), question rows stay buffered | yes: un-kept haystack rows |
| examples/beam-search/src/lib.rs:185 | `[b, pool_len]` device-carried ancestry, `Fold::None`, one set for all lanes (172) | no: "Every mask admits the prompt's slots" (13) |
| examples/beam-search-js/index.js:126, examples/beam-search-py/main.py:140 | same as beam | no |

Under step 1 beam's mask becomes `[b, pool_len - n]`; nothing else in it changes. The six that hide committed rows stop folding the
rows they later hide until step 2 (ruling 1): their masks are then valid in buffer coordinates, and the runtime's cut keeps the planes
inside the room, so they get slower, not refused. No old path is kept.

## 4. The runtime's cut

The cut today: a host-placed pass is cut to `fire_cap(engine)` (crates/runtime/src/store/shared.rs:141-147) by `pass_pieces` with
`prefill_share` (p/fire.rs:3114-3144; 693-709; 1115-1169), re-cut when a raise lands (3277-3291), and what cannot be cut is refused by name
(3199-3240). A piece keeps its rows' masks unchanged: "A row's mask names absolute keys: it is the same in any piece." (1071-1074), with
`lane.kv.held += from` (1075). The carried-rows ask is NP only: `FireRows { own, carried }` and `ask() = rung_rows(own, carried)`
(NP p/fire.rs:602-627), carried = `start - frozen` (629-638), the ask at 3350-3368, `req.carried` at 3554, refusal only past the largest
rung (`check_largest_rung`, 727-740), the batch fit at NP scheduler/batch.rs:28-40, 157.

Proposed check, kv lanes only:
- A masked lane with buffer `b = kv.held - kv.folded` and `r` own rows stages `r * (b + r)` bits. An unmasked lane stages none.
- The cut takes the largest `r` with `r <= fire_cap` and `r * (b + r) <= room`, `room = max(t * t, max_context)` at a rung of `t` tokens:
  `r = floor((sqrt(b*b + 4*room) - b) / 2)`. `r >= 1` at every rung by the `max_context` term in 2.4. A decode row over 100,000 buffered keys is 12.5 KB.
- Nothing is refused for a long buffer: the pass takes more fires. Where a wider piece would fit a higher rung the pass asks for it as a
  prompt asks today (`prefill_want`, `pass_asks`, `raise_now`, 3120-3134).
- Later pieces: piece k's buffer is `b` plus the rows of the pieces before it, unless the pass folds as it goes (`FoldBy::To` on each
  piece when the count is host-known, p/fire.rs:2560-2608), where each piece restates `kv.folded` and its buffer starts empty. The slice at
  1071-1074 becomes rows `a..b` with columns counted from the piece's own `kv.folded`.
- Several masked decode lanes in one fire are not cut (one row each): their bits, `sum(b_i + 1)`, are a fit the scheduler counts as NP
  counts carried rows (NP scheduler/batch.rs:157, 177, 184).

Where `b` is known at staging: host-placed lanes, from `Place.folded` and the lane's `start` (geometry.rs:350-357, 477-481). Device-placed
fires (slots or mask resolved on the device; lanes built as `tokens: vec![0; rows]`, p/fire.rs:4483-4493): the runtime knows the rows, the
cursor's `folded` and `reserved`, and the bound mask channel's shape `[tokens, keys]` (geometry.rs:191-217), so the plane is at most
`rows * keys` bits, known before the fire. Such a fire is not cut ("its geometry or mask resolves on the device", p/fire.rs:3215-3216): its
bits are an ask for a rung, and it is refused by name only past the largest rung's room, as `check_largest_rung` refuses carried rows.

## 5. rs: can a replay longer than a rung be split across fires?

**Facts.**
- A `Buffer` lane replays every buffered slot `folded..first` its mask admits in one run, then its own rows (e/fire.rs:117-167, 557-558;
  `LanePlan::of`, crates/engine/src/rs.rs:20-89; engine-cuda calls `buffered_reads` directly, serve/prepare.rs:1050).
- The run starts from the lane's state slot every fire (`rs_slot`, e/fire.rs:203-206; CUDA serve/prepare.rs:987, 1008-1031; GPU
  gdn_recurrent.slang:259-335†; XLA kernels-xla ssm.rs:988-1014†; Metal conv1d.metal:348-394†).
- State is written at one point, the commit (`to = folded + fold`): rs.rs:62-67; CUDA serve/prepare.rs:1056-1078, and a lane with fold 0
  writes nothing (serve/enqueue.rs:304-309); XLA `keep = min(commit, span)` (kernels-xla lanes.rs:508-557†); Metal gated_delta.metal:306-321†.
- A fire carries own rows: an empty lane is refused (p/fire.rs:529-537). The fold point is the inferlet's: "the runtime never folds more
  behind its back" (p/fire.rs:613-616).
- Replay scratch holds own rows plus replayed rows (`rows_ext`): GPU serve.rs:3172-3193†; XLA rs.rs:378†; Metal per rung
  `tokens + lanes * UNFOLDED_ROWS` (serve.rs:1055-1074†); CUDA in the activation span (store/rs.rs:47-51†).

**Answer.** Not today: the state after k of n buffered rows is written nowhere unless the set folds those k rows, which moves the
inferlet's boundary. Every fire replays the whole admitted buffer from the record, so the replay length is the buffer length.

### Step 2 material: snapshots without folding (Mac's proposal, checked against the runtime and CUDA)

Proposal: land the lane's state in a snapshot slot every chunk of rows without folding; a later fire's replay starts from the latest
snapshot at or before the rows it needs. Scratch is then own rows plus one chunk.

**Runtime: what snapshot slots are today.**
- A snapshot slot is an ordinary state slot of the rs store's slot pool: `prepare_snapshots` takes one per row from the fire's grant or
  `alloc` (store/rs.rs:1689-1732, 2298-2305); the fire's demand counts them (`total.slots += group.snapshots.len()`, pipeline/fire/rs.rs:135).
  In a shared pool the record is the fixed shape's slot refs, named in `RsSnapshot::{pieces, fixed}` (e/fire.rs:528-543; pipeline/fire/rs.rs:327-332).
- On publish each becomes a hidden working set, `RsEntry { folded: Some(slot), buffer: [], base: frozen + row, frozen: frozen + row }`
  (store/rs.rs:1789-1806; `RsSnapshot { ws, row, slot, snapshot }`, store/rs/write.rs:93-101).
- The pipeline keeps it as a checkpoint under the kv prefix key of that row's page (`key: keys[pages - 1]`, p/fire.rs:3372-3382, 3534-3539,
  1884-1889), or releases it if the fire fails (1873-1879; pipeline/fire/rs.rs:9-15). Checkpoints are cache: `checkpoints: HashMap<Hash256,
  RsCheckpoint>`, "hidden forks no process holds" (store/rs.rs:306-307, 320-325), dropped by reclaim under their retention key (893-913, 948-952).
- The guest's `checkpoints(slots)` (forward.wit:60-62; host forward.rs:585-598) are these: rows floored to whole pages within the keyed
  pages (`checkpoint_rows`, p/fire.rs:1201-1221); those inside a fire become `Lane::rs_snapshots`, up to `Capabilities::rs_snapshots`,
  and the rest force a cut (3091-3106, 3405-3410); one at a piece's end is a fork of the set's folded state (`checkpoint`, store/rs.rs:1024-1047).
- A checkpoint is adopted only by a fresh set, by key (`adopt_checkpoint`, store/rs.rs:1210-1241; p/fire.rs:1500-1505). It is not tied to
  the set that landed it. **Nothing holds "the latest snapshot at or before row r of this set's buffer"**: `RsEntry` has one `folded` slot
  and `frozen` (store/rs.rs:251-259). A fork, a fold or a rewrite of the parent does nothing to a checkpoint; `fork` would share a per-set
  snapshot table by refcount as it shares the record (850-856).
- There is no rewind verb: a rewind is a pass that states slots at or above `folded` (below is refused, geometry.rs:433-439).

**Engine contract today.**
- `RsVerb::{Fold, Buffer { pages, base, folded, fold: FoldLen }}` (e/fire.rs:553-567). The state slot is `Lane::rs_slot`, or the lane's
  `fixed` refs to an engine that states chunks.
- Snapshots are taken only by a lane that folds every row: `rs_snapshots` beside anything but `RsVerb::Fold` with an `rs_slot` is invalid,
  and a snapshot row is interior (`row < rows`) (e/fire.rs:341-362). The runtime states them only for a lone lane that starts at `frozen`
  and folds all (p/fire.rs:548-563). CUDA requires the lane to run whole (serve/prepare.rs:988-999), lands up to 16 (`SNAPSHOTS`,
  store/rs.rs:595-621: only where every state-reading op is a scan whose chunked arm lands them; "a single-token arm never meets one").
  Metal lands 4, "A lane that snapshots folds its own rows, none replayed" (conv1d.metal:360-375†); XLA runs them as separate piece fires
  (snapshot.rs:1-14†); GPU only under `Fold` (serve.rs:2885-2890†). So every snapshot today is of a folded row.
- A scan that starts from a slot other than the record exists in contract form: `Lane::fixed` `[run, cow]` with `cow != run` "has the
  engine copy `run` onto `cow` on the stream before the fire, which then reads and writes `cow` (a fork, or a snapshot restored, within
  the fire)" (e/fire.rs:243-249); outside a shared pool the same is `engine::StateCopy` (pipeline/fire/rs.rs:192-200, 505-513).
- So the slot half of a state source is already stated (`rs_slot` / `[run, cow]`). The row half is `Buffer::folded`, which today is at
  once the row the state stands at, the origin a `fold` count is taken from, and the origin `buffered_reads` reads the mask from.

**What the contract change would be.**
1. `RsVerb::Buffer` gains the row the lane's state slot stands at, apart from `folded` (for example `state_at: u32`, `folded <=
   state_at <= first`): the replay runs `state_at..first`; `folded` stays the set's boundary for the fold count and the mask. The slot
   stays `rs_slot` / `[run, cow]`. Not verified: that any engine takes a `Buffer` lane whose state is ahead of `folded` without this
   (CUDA keeps per-slot `rs_ahead`, serve/prepare.rs:1026-1031).
2. `Lane::rs_snapshots` is admitted beside `RsVerb::Buffer`, with rows counted over the extended run and the lane's end allowed; today's
   one state write per lane (the commit) becomes the commit plus the snapshots. On CUDA this reaches the single-token arms.
3. Runtime: a per-set table of `(row, slot)` above `frozen`, shared by fork, dropped past a rewritten slot, dropped at or below a fold.

**The carried-rows contract, if this holds.** Carried rows a lane are below one chunk, so NP's `rung_rows(own, carried) = max(own,
carried)` and `check_largest_rung` have nothing left to refuse where the chunk is at most the smallest rung's tokens. With chunk =
`UNFOLDED_ROWS` the tip's fixed bound holds by construction and `FireRequest.carried` is not needed. Candidates for the chunk: the constant
(16), the smallest activation rung's tokens, the tokens of the rung the fire runs at, or an engine-stated `FireLimits` figure. Records held
are `ceil(unfolded rows / chunk)` a set, one state record each.

**Form to evaluate first, with no chunk figure (Pie Leader).** A buffering lane lands its state at the end of its own rows every fire:
one snapshot a lane a fire, replacing the previous one unless a fork or a sibling still needs it. The next fire replays nothing unless
its mask admits a different set of buffered rows than the snapshot covers. A beam child inherits its parent's snapshot by the same
parent index that inherits its mask. Cost: one state record a live lane (and a live beam cell), not one a chunk. To state for the
step-2 design: what that costs per model in slot-pool room, and what breaks.

**Open.**
- A fold to a row `t` behind the latest snapshot: the state at `t` comes from a snapshot at exactly `t` (a store move, no scan) or from a
  replay from the snapshot before `t`; a fire that also has own rows far past `t` then runs two disjoint scans or carries `first - t` rows.
- Device-counted folds (`FoldLen::Device`): `t` is not known at staging, so the runtime cannot pick the snapshot before it.
- Masks: a snapshot is the state after the buffered rows that fire's mask admitted. It serves a later fire only where that fire admits the
  same rows below it. Lanes of one group that read one state with different masks have no common snapshot.
- Record size per model and the slot pool's room for `buffer / chunk` records: not measured.

## 6. What step 1 deletes, at contract level

- The extent rule "a mask covers `[0, have + rows)`": `Mask::total` as a context length and `Fault::Mask` with `extent = have + rows`
  (engine-gpu mask.rs:36-42; engine-cuda mask.rs:40-46; engine-xla mask.rs:41-48).
- Whole-mask expansion with 32-bit offsets: `Mask::words`/`expand_into` (e/fire.rs:55-90) and the runtime's `MaskWords { request_indptr,
  word_indptr, words }` / `bitmask_words` (crates/runtime/src/engine/fire.rs:113-140, exported at engine.rs:26; no caller found in `crates/`
  outside that file).
- Context-sized planes: `max_tokens * context` bytes (engine-gpu inputs.rs:318-323; engine-metal inputs.rs:468-471 per rung),
  `(max_tokens * context) / 8 + lanes` (engine-cuda inputs.rs:630-636), and their ceilings (2.3); replaced by the derived room (2.4).
- 32-bit offsets into a context-sized plane: CUDA's i32 `mask_indptr` (mask.rs:17, 114; inputs.rs:640); XLA's 2^31 check (kernels-xla
  attn/paged.rs:2093-2105†). They stay as types and stop being a limit: a plane is bounded by the rung's room.
- The runtime's lowering over the guest's key axis (geometry.rs:591-601) and the engine's port lowering (ports.rs:244-274) become
  lowerings over the buffer extent.

Where each backend expands the mask today: Vulkan/wgpu crates/engine-gpu/src/mask.rs:23-124 (called serve.rs:3087†); CUDA and ROCm
crates/engine-cuda/src/mask.rs:20-114 (called serve/prepare.rs:1528†); XLA crates/engine-xla/src/mask.rs:29-145 (called serve.rs:2041-2054†);
Metal crates/engine-metal/src/mask.rs:22-148 (called serve.rs:3907†). Each kernel reads `[0, folded)` unmasked and tests bits only at
`kp >= folded` with the plane indexed by `kp - folded`.

## 7. Rulings on the draft's questions (Pie Leader, 2026-11-15)

1. **Between steps 1 and 2.** Step 1 lands first; the six examples that hide committed rows stop folding those rows until step 2.
   No old path kept.
2. **Mask origin while a device-counted fold is in flight.** The stated boundary; a lower bound is valid (the slots are read through
   the mask). A selecting fold (step 2) settles before the set's next fire is staged.
3. **Own-row bits above the diagonal.** Not read.
4. **A slot another lane of the same fire writes in the same set.** Contract: a fire's lanes of one set read one buffer as the fire
   found it plus their own rows; a guest's mask hides siblings. Each engine gates the hidden case and states what an admitted read sees (2.5).
5. **The mask room.** Derived, no engine figure: `max(t * t, max_context)` bits at a rung of `t` tokens (2.4).
6. **Hybrid lanes.** One boundary, stated twice and equal. To verify before building: `Place.folded` against the rs store's `frozen`
   on every path, and where they can differ (rs fires drain device-counted folds until the cursor is exact, p/fire.rs:662-685).
7. **Causal bound.** Row order / cache row on every backend; stated position is rotation only. A behaviour change on Vulkan, wgpu and
   Metal for guests whose positions differ from their slots; each gates beam.
8. **ROCm's paged prefill.** Gets the two-segment read.

Accepted as written: the boundary (2.1), the mask text (2.2), the refusals (2.3), the runtime's cut (4), the WIT and SDK wording (3),
what step 1 deletes (6). Step 2's rs shape (`state_at` on `RsVerb::Buffer`, `rs_snapshots` beside `Buffer`, a per-set `(row, slot)`
table above `frozen`) is the right shape; the chunk question is replaced by the no-figure form in section 5, to be evaluated.
