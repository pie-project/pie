# Follow-ups to MASK-FOLD.md: A (Q6, before step 1) and B (step-2 form with no chunk figure)

Read at `cec91bb4b` (origin/memory-redesign; the runtime files are the same as at 10d708421, which MASK-FOLD.md cites). Nothing was built
or run: every statement is from reading the code. Paths: `p/` = crates/runtime/src/pipeline/, `s/` = crates/runtime/src/store/,
`h/` = crates/runtime/src/inferlet/host/, `e/` = crates/engine/src/. "Not verified" marks what was not read to the end.

---

## A. `Place.folded` against the rs store's `frozen`

### A.0 Verdict

They are not equal by construction. There are three counters, moved at three different times, and the rs lane already states the
second while step 1 would state the first:

| Counter | Read by | Moves |
|---|---|---|
| host cursor `Cursor.folded` (+ `folded_exact`, `unsettled`) | `Place.folded` (p/fire.rs:2470-2478), the guest's `working-set.folded()` (h/working_set.rs:179-181) | at submit, once a pass, after its last piece (`advance_cursors`, p/fire.rs:2496-2554, called 3623); or at settle (4083-4092) |
| kv page table `entry.folded` (`kv.folded(ws)`) | the frozen-prefix log check (p/fire.rs:3455-3485), `settle_fold` | at settle, on success only (`kv.fold`, p/fire.rs:4067-4069; s/kv.rs:1797-1824); at `attach_run` (s/kv.rs:1875-1877) |
| rs store `RsEntry.frozen` | `RsVerb::Buffer.folded` (p/fire/rs.rs:532-566), `host_rs_plan` / `device_rs_plan` (p/fire.rs:539-541, 590-592) | at prepare of each piece, for a host-known count (`commit_folds`, p/fire/rs.rs:569; s/rs.rs:1772-1776, 1874-1896); at settle for a device count (p/fire.rs:4076-4081) |

Equal on: a one-lane host-placed pass with a host-known count (single piece or cut), a device-counted fold (both old until settle),
rewind, fork of a settled set, park and restore of a settled set. They differ on the six paths in A.2. The one that is live in
ordinary serving is D2 (a device-placed pass with a host-known fold, pipelined): kv would state `F`, rs states `F + c`.

All writers of rs `frozen`: create 0 (s/rs.rs:733), export/import (746, 844), `adopt_checkpoint` (1230-1236), `forget_state` (1247),
the snapshot's hidden set (1795-1796), `fold` (1882), handle save/load (crates/runtime/src/handle/snapshot.rs:426, 1021). No residency
move, offload or park writes it (grep of `frozen` over s/rs*, handle, offload, inferlet).

### A.1 Path by path

Notation: `C` cursor, `K` kv table, `R` rs frozen, all equal to `F` before the pass; `c` the fold count of the cursor's first lane.

**Submit, host-placed, host-known count.**
- `Place` is a clone of each lane's cursor taken once a pass, before any cut (p/fire.rs:2820; device-placed 4293).
- `R` moves at each piece's prepare: `R += min(c, E_first - R)`, `E_first` the end of the group's first lane in that piece
  (`host_rs_plan` 539-543, `group_fold` 652-660; committed p/fire/rs.rs:569).
- `C` moves once, after the last piece: `C = max(C, min(C + c, E_max))`, `E_max` the largest landing end over the cursor's lanes
  (p/fire.rs:2524-2529, 2549-2552).
- `K` moves at settle, on success: `min(F + c, reach, end)`, `reach` the first lane's landing end (`fold_at` 2580-2589; call 3502-3514;
  `settle_fold` 4067-4069).
- One lane a set: `C == R` after submit and `K` joins at settle. Equal by construction (given equal before).
- Several lanes on one set ending at different slots, count past the first lane's end: differ, D1.

**Cut passes and their pieces.**
- An rs pass is cut only as one lane that folds 0 or all (`folds_all`, p/fire.rs:2940-2943; `cuttable` 2958-2964; 3107-3108).
- Fold all: `R` = the piece's end after each piece's prepare; `K` = the piece's end at that piece's settle; `C` = the pass's end after
  the last piece. Fold 0: nothing moves. Equal when the pass has gone out whole.
- A piece after the first does not start from `Place.folded`: `Place` is the pass's, not the piece's. Step 1's "per piece of a cut pass"
  must compute the piece's boundary: `max(F, min(F + c, piece start))`. With `c` in {0, all} that is `R` as piece k finds it.
- An early return between a piece's prepare and `advance_cursors` (p/fire.rs:3442, 3477-3485, 3528-3530, 3574-3577) leaves `R` (and,
  when those pieces settle, `K`) advanced by the pieces submitted and `C` not advanced. The pipeline is failed at the same time for the
  submit failure (`record_submit_failure`, 4219-4234); for the other returns that was not verified. D5.

**Device-counted folds in flight (`Fold::Device`).**
- Nothing moves at submit. The cursor is marked `unsettled += 1; folded_exact = false` (p/fire.rs:2513-2519).
- At settle, in `settle_fold` (4061-4092): `to = K + reported count, at most reach`; `K = to` on success; `R += to - K_old` on success
  (4076-4081); `C = K` and exact once `unsettled` is 0.
- In flight `Place.folded == R == F`: equal, and both a lower bound of what the engine holds (ruling 2).
- Host-placed rs fires drain first: `settle_device_folds` (664-685; called 2821-2825 when the pass has rs sets) awaits the pipeline's
  fires while any place is inexact, re-reads the places, and refuses by name if one is still inexact.
- Device-placed fires do not drain: "the fire goes ahead of the settle" (4291-4293).
- `R` follows `K` here (`to - K_old`), so a gap between `K` and `C` or `R` that another path left is carried forward (D4).
- Not verified: that the three updates in `settle_fold` are atomic against a pass being staged by the same process at that moment.

**Device-placed pass with a host-known count** (slots from a channel, `Fold::All` or a count).
- `R` moves at prepare: `R += min(c, reserved - R)` (`device_rs_plan`, p/fire.rs:576-601).
- `C` does not move at submit: `advance_cursors` runs with no landings and leaves the loop before the fold (2520-2523; call 4641).
  The fold goes to settle as `FoldBy::Settled { count: Some, rs: None }` (`fold_at` 2590-2607; call 4588-4599), where `K` and then `C` move.
- While that fire is in flight `C == K == F` and `R == F + c`. The next device-placed fire is staged ahead of the settle. Differ, D2.

**Float lanes.** `fold_one_plan` folds 1 a fire (p/fire/float.rs:17-30) and the cursor adds 1 at submit (326-337). Equal by construction.

**Fork.**
- rs `fork` clones the entry with its `frozen` (s/rs.rs:850-856); the host fork clones the cursor's value into a new `Arc`
  (h/working_set.rs:196-198, 249-253); kv `fork` gives the child the table's `folded` as it is (s/kv/page_table.rs:226-245).
- Equal in the child exactly where equal in the parent at that moment. `fork` refuses a closed or failed pipeline (h/working_set.rs:218-220);
  no drain of fires in flight was seen in it.
- A fork taken while a fold is unsettled: the child's cursor keeps `folded_exact = false, unsettled = k` and no settle ever reaches it
  (the settle holds the parent's `Arc`). D3.
- A handle fork is of a parked set, after every fire has settled (crates/runtime/src/handle/lease.rs:317; held.rs:112-128): as the parent.

**Rewind (slots stated at or above folded).** No counter moves: the cursor's `tail` and spans change, rs buffer pages are rewritten. A
write below is refused three times: `Place.folded` when exact (p/fire/geometry.rs:433-439), `K` (p/fire.rs:3470-3485), `R`
(`BelowFrozen`, s/rs.rs:1531-1534). Equal by construction. Where the counters already differ the three refusals disagree on the row.

**Checkpoints adopted (`resume`).**
- Only a fresh set at `held == 0` on an unmasked, keyed, cuttable pass (p/fire.rs:1301-1312, 2990-2998; `plain_rows` 1009-1013).
- `K = pages * page_size` (`attach_run`, s/kv.rs:1875-1877). `R = the checkpoint's frozen` (s/rs.rs:1230-1236). `C` is not touched by
  `supply` or `resume` (1256-1321, 1492-1530).
- The checkpoint's `frozen` is its row for a snapshot (s/rs.rs:1785-1797) and for a fork of a set folded through the row. A checkpoint
  taken at a piece's end is a fork of the set as it is (`checkpoint`, s/rs.rs:1028-1047), with no test that `frozen` reached the row;
  the pipeline takes it whether or not the pass folds (p/fire.rs:3396-3398). Then `R < K` after adoption. Not verified whether a
  non-folding pass can reach that arm with a state the store accepts.
- After the adopting pass: fold all gives `C = end = R`; fold 0 leaves `C = 0` with `K = R = S`. Differ, D6. The same lag of `C`
  follows a kv-only `Attach::Kv`.

**A failed fire's rollback.** There is none: `finalize_fire_await` settles sequences and releases, and no code restores `frozen` or the
cursor (p/fire.rs:4098-4200; `RsTxnsGuard::drop` 263-278 abandons snapshots only).
- Host-known count, host-placed: `C` and `R` stay advanced, `K` does not fold (4067). `C == R != K`.
- Host-known count, device-placed: `R` advanced at prepare; at settle `C = K = F`. `C != R`.
- Device count: nothing moves; a fold never reported leaves the cursor inexact.
- The set does not fire again: `open_pipeline` refuses a failed pipeline (2674-2677), fork refuses it, a parked set becomes `Slot::Dead`
  ("A fire that failed left the cursor ahead of what was written", handle/lease.rs:398-405). So no lane states either value after it.

**Park, move, offload and restore.**
- Park copies the cursor's value (handle/lease.rs:385-406) after the fires settled. Residency moves and offload write no counter.
- Save writes `cursor.folded`, the kv `folded` and rs `frozen` as three fields (handle/snapshot.rs:790-797, 420-429); load restores each
  as saved, with `folded_exact: true, unsettled: 0` (945-952, 1017-1024). Equal where equal at save; a difference is preserved and the
  cursor is then called exact. Not verified: that `save` requires every fire settled.

**Windowed sets** (`window_tokens > 0`: Gemma 4 and the other sliding models). They carry no `RsVerb::Buffer` (verbs default to `Fold`,
p/fire/rs.rs:458; `Buffer` is written only on the group path, 560) and the rs fold at settle is skipped for them (`rs` filtered by
`!windowed`, p/fire.rs:3503, 4588). The store's position is `window.committed` (s/rs.rs:879-883), moved per lane at prepare by
`window_advance` (1436-1451): by the lane's own count, or with a device count up to the first row the fire writes. It is the retire
reach, not a mask origin, so Q6 does not arise for them. Not verified: `committed` against `C` on the device-placed windowed path.

### A.2 Every place they can differ

| | Path | Cite | kv boundary (`Place.folded`) | rs `folded` (`frozen`) |
|---|---|---|---|---|
| D1 | Several lanes share a set, end at different slots, and the first lane's count reaches past its own end (lanes writing `n` and `n+1`, `Fold::All`) | cursor: p/fire.rs:2524-2552 (max end over lanes); rs: 543, 652-660 (first lane's end); kv: 3502-3514 | `n + 2`, called exact; ahead of both stores from then on | `n + 1` |
| D2 | Device-placed pass with a host-known count, next pass staged before the settle | p/fire.rs:576-601, 2520-2523, 4291-4293, 4588-4599, 4641 | `F` (stale by `c` for each such fire in flight) | `F + c` |
| D3 | Fork while a fold is unsettled | h/working_set.rs:196-253; s/rs.rs:850-856; s/kv/page_table.rs:226-245 | in case D2: `F`, never exact, never settled in the child | `F + c` |
| D4 | After D3 with a host-placed `To` fold in flight (child `C == R == F + c`, child `K == F`), a later device-counted fold in the child | p/fire.rs:4061-4092 (`C = K`, `R += to - K_old`) | `K`, which is `c` short | `c` ahead of it |
| D5 | A cut pass that returns early after some pieces were prepared | p/fire.rs:3442, 3477-3485, 3528-3530, 3574-3577 against 3623 | `F` | the last prepared piece's end. Reachable by a later lane only if the pipeline is not failed (not verified for the non-submit returns) |
| D6 | Resume or kv attach, then a pass that folds 0 | p/fire.rs:1256-1321, 1492-1530; s/rs.rs:1230-1236; s/kv.rs:1875-1877 | `0` (and the guest's `folded()` says 0) | `S`, the adopted row; a write below `S` is refused by the stores |

No example in the repo was found on D1 (lanes that share a set fold 0 in beam; the other multi-lane examples bind one set a lane), but
the search was by `fwd.state` call sites only. D2 is every pipelined decode loop whose slots are device-carried and whose fold is `All`
or a count. D6 is every prompt-cache hit followed by a pass that does not fold.

### A.3 What a hybrid lane states today, and what step 1 has to decide

- The rs side states `store.frozen(ws)` as the fire finds it, read after the fire's publish and before its own fold is committed
  (p/fire/rs.rs:532-566). The recurrence starts there, so that value cannot be a bound: it has to be exact.
- The kv side, as MASK-FOLD.md 2.1 is written, would state `Place.folded`. On D1-D6 the attention layers and the recurrent layers of
  one lane would then read one mask from two origins, `c` columns apart.
- "The set's frozen length when the pass runs" (the proposed WIT text) is `R` on D2, not `Place.folded`: the fire in flight folds
  the kv to the same row before this one runs.
- The guest builds its mask from `working-set.folded()`, which is the cursor. On D2 and D6 that is neither what the pass will find
  nor what the rs lane states. A settle between the guest's `folded()` call and its submit also moves the cursor (`open_pipeline`
  drains settled fires first, p/fire.rs:2674), so with a device-counted fold in flight the guest cannot know the origin it will be
  given. No example combines `fwd.mask` with a device-counted fold today (MASK-FOLD.md section 3 table).

Observations for the step-1 build (not designs):
1. A hybrid lane's two `folded` values can be made one by construction only by filling both from one read. The rs read is the one
   that must be exact, so the kv side would take it (or the runtime asserts equality and refuses by name).
2. D2 goes away if the cursor is advanced at submit on the device-placed path for a host-known count, with the bound the rs plan
   already uses (`min(c, reserved - frozen)`). Not verified: that `reserved` is never past the rows the device then writes.
3. D1 goes away if `advance_cursors` bounds the fold by the first lane's end, as the rs plan and `fold_at` do.
4. D6 goes away if `supply` / `resume` advance the cursor to what they attached, or if a pass after an attach must fold through it.
5. D3/D4 need either a drain in `fork` or the child's cursor made exact from the stores.
6. The checkpoint fork with `frozen` short of its row (s/rs.rs:1028-1047) is a separate question from the mask: `R < K` after adoption.

---

## B. Step 2 with no chunk figure: a landing a lane a fire

Terms used below. *Landing*: the state record a buffering lane writes at the end of its own rows. *Cover*: the set of buffered rows
`frozen..row` whose contribution the landing holds; it is one set a lane, since "every row must admit the same buffered slots"
(e/fire.rs:117-118).

### B.1 Cost in slot-pool room

A landing is one state record: one state slot of the rs store (`prepare_snapshots`, s/rs.rs:1689-1732; demand p/fire/rs.rs:135), which
in a shared pool is `pieces` slots of each fixed shape (s/shared.rs:526-533, 1241-1265) and takes `ceil(n * pieces / slots_per_chunk)`
chunks a shape (`record_chunks`, 1076-1084). Its bytes are the sum of the model's `State` rows (CUDA `state_slot_bytes`,
crates/engine-cuda/src/store.rs:1495-1500; `one_slot_bytes` 316-321).

Bytes computed from the `state(...)` declarations and dimension tables in models/ (qwen_3.poem:320-369, 524, 554-556;
nemotron_h.poem:59-61, 63-110, 143-147, 171; kimi_k3.poem:131, lib/kda.poem:38-39). Chunks are bytes over the profile's target chunk
(CUDA and XLA 16 MiB, Metal 2 MiB: crates/poem-compiler/src/chunk/profile.rs:88-131; GPU 4 MiB: crates/engine-gpu/src/store/layout.rs:79).
The compiler picks the real chunk between half and twice the target (profile.rs:172-175) and rounds pieces per layer, so the chunk
columns are estimates; the recipe was not run. One figure agrees with a measured note: Qwen3.5-0.8B "one record (~10.37 MB)"
(new-memory/PROGRESS.md:1333) against 10.32 MB computed.

| Model | Recurrent layers | One record | Chunks at 16 MiB | at 4 MiB | at 2 MiB | Same bytes as kv of |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 18 GDN | 9.8 MiB | 0.6 | 2.5 | 5 | ~840 tokens |
| Qwen3.5-35B-A3B | 30 GDN | 31.9 MiB | 2 | 8 | 16 | ~1630 tokens |
| Qwen3-Next-80B-A3B | 36 GDN | 38.3 MiB | 2.4 | 10 | 19 | |
| Qwen3.5-27B | 48 GDN | 75.8 MiB | 4.7 | 19 | 38 | ~1210 tokens |
| Qwen3.5-122B-A10B | 36 GDN | 75.4 MiB | 4.7 | 19 | 38 | |
| Qwen3.5-397B-A17B | 45 GDN | 94.2 MiB | 5.9 | 24 | 47 | |
| Qwen3.8 Max (A95B) | 69 GDN | 286.8 MiB | 18 | 72 | 143 | |
| Nemotron-3 Nano | 23 Mamba-2 | 47.1 MiB | 2.9 | 12 | 24 | |
| Nemotron-3 Super | 40 Mamba-2 | 163.1 MiB | 10 | 41 | 82 | |
| Nemotron-3 Ultra | 48 Mamba-2 | 390.8 MiB | 24 | 98 | 195 | |
| Kimi K3 2.8T | about 69 KDA (6.3 MiB a layer, f32) | about 433 MiB | 27 | 108 | 217 | |
| Gemma 4, gpt-oss, MiMo, Muse, dots3 (windowed) | none | 0 | 0 | 0 | 0 | |

Not computed: Qwen 3.8 Flash Next (qwen_4.poem:377-378, 426-427), Inkling (inkling.poem:154-157), DeepSeek V4's `ids_state` and
`replay` (a few words), the Kimi K3 KDA layer count and width (taken as `heads * head_dim`), Wan's frame cache. The kv column assumes
bf16 kv and the poem's attention layer count.

Times lanes:
- Only a lane that buffers holds a landing. A lane that folds all its rows writes its record in place, as today.
- A lone buffering lane: the record at `frozen` plus one landing, so its state room doubles. A second landing is held while a fork or a
  fire in flight still reads the first (the store already takes a fresh slot when the held one has another reference, s/rs.rs:1487).
- A beam of `b` cells on one set: the record plus `b` landings where the engine copies each parent into its child's slot before the
  fire, or plus `2b` where it reads one generation and writes the next. Qwen3.5-35B-A3B at `b = 4`: 128 MiB or 255 MiB beside the
  32 MiB record; at `b = 8`: 255 or 510 MiB. Nemotron-3 Super at `b = 4`: 652 MiB or 1.3 GiB.
- 64 buffering lanes of Qwen3.5-35B-A3B: 2.0 GiB of landings, the kv of about 104k tokens.
- Against the chunked form (`ceil(unfolded rows / chunk)` records a set): one record a lane whatever the buffer's length.
- Windowed models hold no record, so the form costs them nothing and gives them nothing. Their step-1 cost is elsewhere: windows retire
  behind `committed` (s/rs.rs:1452-1470), which only a fold moves, so a lane that stops folding keeps every row of its window lists.

### B.2 What breaks or needs a rule

**Finding that shapes the rest.** In the steady case (one lane a set, cover unchanged, no fold) the landing is not a snapshot in the
engine's sense. It is the lane's one state write, at the end of its rows, into the lane's own slot: what every `RsVerb::Fold` lane
does today, single-token arms included. What differs is only that the set's boundary does not move and the rows' inputs still go to
the buffer pages. So the form needs a state row apart from the boundary, and does not need `rs_snapshots` beside `Buffer`. The
snapshot machinery comes back only where a rule below asks for a second state write in one lane.

1. **A mask that admits a different set than the landing covers: replay from where.**
   - A recurrence cannot drop a row. With one landing a lane the only other source is the record at `frozen`, so the fire replays every
     admitted row `frozen..first`: the whole buffer, the length the form set out to avoid.
   - The source that bounds it is the newest landing whose cover equals the new mask below its row. That exists only if older landings
     are kept: "replacing the previous one" has to become "replacing it unless the next cover differs below it". Landings are then a
     short per-lane stack ordered by row, and which to keep under pressure is retention, which is PLEX's to key and the runtime's to
     drop (as checkpoints are cache today, s/rs.rs:306-325, 893-913).
   - A sliding mask changes the cover every fire: attention-sink's decode (examples/attention-sink/src/lib.rs:169) and the amateur
     lane of contrastive decoding (117-120, 160). They replay the window from a landing at the last stable row (the sink's end) every
     fire. The tip landing is never used. The replay is the window's length and is inherent.
   - A keep-set that shrinks (trackb-h2o:323, trackb-snapkv:249) replays from the record, or from a landing before the evicted row,
     at each eviction. A cover that is set once (snapkv-eviction:258, naive-masked doc isolation:166-171) replays once.
   - The runtime can compare covers only for a mask the host holds. attention-sink's decode mask and beam's are device-carried
     (channels re-put in the epilogue), so at staging the runtime does not know the cover. Either the engine compares the bitmap on
     the device (and then the rung and scratch were chosen without knowing the replay length), or the guest states that the pass's
     cover continues the previous one and the engine fails the fire by name where it does not.
   - A landing therefore carries its cover (a bitmap of `row - frozen` bits, or a digest), re-based when the set folds.

2. **A rewind below the landing's row.** The landing covers rows being rewritten, so it is dropped; the fire replays from an older
   landing at or below the rewind row, else from the record (`start - frozen` rows). A loop that rewinds every fire (speculative
   drafts: rows past the accepted one are rewritten) never uses a tip landing. The row it needs is the accepted one, which is where
   its fold commits today. Rule: a lane that folds (a count above 0 or a device count) commits as today and lands nothing.

3. **A fold behind the landing** (to `t` below its row).
   - `t` equal to a landing's row with the same cover: a store move, the landing becomes the record, no scan.
   - `t` below it: the record at `t` is the record plus the admitted rows `frozen..t`, a replay of up to `t - frozen` rows from the
     record. If the lane's own rows also continue from the landing, that is two disjoint scans in one lane, or a piece with no own
     rows (an empty lane is refused today, p/fire.rs:529-537), or today's single run from the record carrying `first - frozen` rows.
   - The landing survives the fold where the folding fire admitted the same rows below `t` as the cover.
   - Simplest rule: a fire that folds short of a landing runs as today (from the record, commit at `t`, no landing), and the fire
     after it replays `t..first` once and lands. Both replays are as long as the buffer.

4. **Device-counted folds.** `t` is unknown at staging, so the runtime cannot name the source: at `t` at or past the landing's row the
   landing serves, below it only the record does. With rule 2 (a folding lane lands nothing) the case is today's. A set that holds a
   landing and then takes a device-counted fold needs ruling 2's settle before its next fire is staged, and the count taken from the
   record. Finding A applies: the runtime's `frozen` is a lower bound while such a fold is in flight (CUDA keeps the difference per
   slot, `rs_ahead`, serve/prepare.rs:1026-1031), and a landing's cover is stated against `frozen`.

5. **Lanes of one group reading one state with different masks.**
   - Today a group reads one record; only its first lane commits and the others' end states are dropped (`FoldLen::Host(0)`,
     p/fire/rs.rs:554-559). Under the form every lane lands, so the lanes need a slot each and are no longer one state.
   - Which landing a lane of the next fire continues is not stated by anything today. Host-known: a per-lane `[run, cow]`, `run` the
     parent's landing, already in the contract ("a fork, or a snapshot restored, within the fire", e/fire.rs:243-249).
   - Beam's parent is not in the contract at all. It is a value in the guest's epilogue (`parent = i / v`,
     `gather(mask.take(), &parent)`, examples/beam-search/src/lib.rs:201, 216), computed on the device and never drained by the host.
     "Inherits its parent's snapshot by the same parent index that inherits its mask" needs the guest to state that index to the
     pass (a `[lanes]` channel beside the mask), and the engine to take a lane's state source by a device value. This is new in the
     WIT, the SDK and the engine contract.
   - Beam's rows are device-placed, so the runtime does not know a landing's row (`writes: Some(frozen..reserved)`, p/fire.rs:595-603).
     Its landings are keyed by lane of the previous fire, not by row.
   - Two children of one parent read one slot and write two, so the fire cannot run in place: `b` copies of a record a fire
     (128 MiB a step for Qwen3.5-35B-A3B at `b = 4`), or two generations of slots.

6. **CUDA's single-token arms landing a snapshot.** Not needed where the landing is the lane's commit into its own slot: the arm writes
   its state as it does for a `Fold` lane. Needed only if one lane both commits to the record and lands (rule 3 excludes it). Not
   verified: that CUDA runs a `Buffer` lane with nothing to replay on its single-token arms, and that a `Buffer` lane whose fold covers
   its own rows still writes those rows to the buffer pages (a lane with fold 0 writes no state today, serve/enqueue.rs:304-309).

7. **Metal's 4 and CUDA's 16 snapshot limits.** Both are per lane per fire (crates/engine-cuda/src/store/rs.rs:595-598;
   crates/engine-metal/src/rs.rs:48-49; the check is `rs_snapshots.len()` against `serves.rs_snapshots`, e/fire.rs:341-348). One landing
   a lane a fire is inside both, and as a commit it does not count against them. The chunked form would have been bound by them: at
   most 4 chunks of rows a lane a fire on Metal.

8. **XLA's separate piece fires.** XLA runs each snapshot row as a piece fire of its own (crates/engine-xla/src/snapshot.rs, `MOST = 4`
   at 21; api.rs:367). A landing at the lane's end is not an interior row (`row < rows`, e/fire.rs:352-362), so it adds no piece. The
   chunked form would have multiplied XLA's fires by the chunks a fire.

9. **Fork, failure.** A fork shares the landing by reference and the first writer takes a fresh slot, as the record does. A failed fire
   that wrote a landing in place is not rolled back; the set is dead with its pipeline (section A).

**The engine contract change, against the three in MASK-FOLD.md section 5.**

| Section 5 | Under this form |
|---|---|
| 1. `state_at` on `RsVerb::Buffer`, replay `state_at..first` | Kept, and it carries more: the lane's slot (`rs_slot` / `cow`) stands at `state_at`; the replay is the admitted rows `state_at..first` (empty in the steady case); the state after the lane's last row is written to that slot; `folded` does not move and is only the mask's origin and the fold count's origin. After step 1 the boundary is already stated in `KvDelta::folded`, so `Buffer.folded` could become the state row with no new field, which undoes "stated twice and equal" on purpose. |
| 2. `rs_snapshots` beside `Buffer`, the lane's end allowed | Dropped. One state write a lane stays one. No change to the snapshot limits, CUDA's single-token arms or XLA's piece fires. |
| 3. A per-set `(row, slot)` table above `frozen` | Becomes per-lane landings `(row or previous lane, slot, cover)`: at least one a buffering lane, more where rule 1 keeps older ones; shared by fork, dropped past a rewritten slot or a changed cover, turned into the record by a fold to its row. |
| (none) | New: a lane's state source named by a device value (beam), with the guest stating the parent index. New: for a device-carried mask, who tests the cover. |

### B.3 Whether the carried-rows contract has anything left to bound

Yes, less than before. With landings, a fire carries no rows in for: a lone buffering lane whose cover is unchanged and that does
not fold, and a beam cell given its parent's landing. `rung_rows(own, carried)` is then `own` and `check_largest_rung` has nothing
to refuse.

It still bounds, by the buffer's length unless a rule shortens it:
1. the first fire after a cover change below the newest usable landing (every fire of a sliding mask: the window's length);
2. the first fire after a rewind below the landing;
3. a fold short of a landing, and the fire after it (B.2 rule 3);
4. a set with a device-counted fold in flight;
5. any lane whose mask is device-carried, unless the guest states the cover continues: the runtime must otherwise size the rung for
   the whole buffer;
6. a group whose lanes were not given a slot each (a beam without the stated parent index).

The contract goes away entirely only with one more change: a replay cut into pieces by the runtime, each at most a rung of rows and
each landing where it ends. The buffered rows' inputs are in the buffer pages, so such a piece runs the recurrent scans alone and has
no own rows; the engine refuses an empty lane today and has no body for it. Without that, `FireRequest.carried` and the largest-rung
check stay as the refusal for cases 1-6, and the tip's fixed `UNFOLDED_ROWS = 16` (e/fire.rs:569-590) cannot hold for them: a sink
window or a keep-set is longer than 16 rows.

### B.4 Not verified

- Chunk and piece counts of a record on each backend (the recipe was not run); the record bytes of the models listed as not computed.
- That any engine takes a `Buffer` lane whose slot stands ahead of `folded` today (it should not: `folded` is the state row).
- CUDA's arm choice for a `Buffer` lane with no replay, and the buffer write of rows a lane folds in the same fire.
- The time of a record copy on each device (beam's `b` copies a fire).
- Whether any guest other than beam shares one set across lanes with masks that differ.
