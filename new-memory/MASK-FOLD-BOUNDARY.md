# One committed boundary per set

Answer to Pie Leader's ruling on MASK-FOLD-FOLLOWUP.md. Read-only; nothing was built or run.

Tree read: `cec91bb4b` (the tree of the follow-up). `451233bfd` on memory-redesign differs from it only by
clippy `cfg_attr` attributes in 19 files; no line cited here changes behaviour there, and line numbers below
are those of `cec91bb4b`.

Paths are under `crates/runtime/src/` unless they start with `crates/` or `examples/`. Short names:

| short | file |
|---|---|
| pf | pipeline/fire.rs |
| pfrs | pipeline/fire/rs.rs |
| float | pipeline/fire/float.rs |
| geo | pipeline/fire/geometry.rs |
| hws | inferlet/host/working_set.rs |
| skv | store/kv.rs |
| pt | store/kv/page_table.rs |
| kws | store/kv/working_set.rs |
| srs | store/rs.rs |
| held, lease, snap | handle/held.rs, handle/lease.rs, handle/snapshot.rs |

Counters: **C** is the host cursor `Cursor.folded` (hws:20-61), **K** is the kv page table's
`WorkingSetEntry.folded` (pt:132), **R** is the rs entry's `RsEntry.frozen` (srs:256). **B** is the single
boundary the ruling asks for. F is the boundary a pass finds, c the count it states.

## 0. Corrections to FOLLOWUP.md

1. "A failed fire rolls nothing back and the set dies with its pipeline" is half true. Nothing is rolled
   back. The set dies only (a) at handle park, where a failed fire makes the slot `Dead` (lease:398-405),
   and (b) while the failed pipeline is still the set's scope (`open_pipeline`, pf:2675; host fork,
   hws:218-220). Once the failed pipeline is closed and drained, `PipelineScope::claim` hands the set to a
   new open pipeline without looking at the failure (crates/runtime/src/store.rs:90-108, the arm at
   97-100; `failure()` is at 63-65 and is not called there).
2. The early returns of a host-placed pass that are not submit failures do not fail the pipeline
   (pf:3325, 3442, 3477, 3479-3485, 3528-3530; device-placed 4412, 4440). Only pf:3574-3577, pf:4632 and
   float:317-321 call `record_submit_failure` (pf:4219-4234).
3. A seventh path differs: a float pass advances C and R by one a fire and never K (float:267-274,
   334-336, `fold: None` at 352).
4. "With a device-counted fold in flight a host count becomes `To(F + c)`" was wrong. `FoldBy::To` is used
   only when `place.folded_exact` and the count and reach are known (pf:2580-2588). Otherwise a host count
   is `Settled{count: Some(c)}`, an increment applied at settle in pipeline order.
5. "No engine has an empty-lane refusal of its own" was wrong. The refusal is one shared site, poem-exec's
   `seriate` (crates/poem-exec/src/fire/compose.rs:399-401, `Fault::EmptyLane`, message at
   crates/poem-exec/src/fire.rs:176-178), and every engine reaches it through `compose_axes` /
   `compose_axes_at` (compose.rs:346-357). See Part 3.

---

## Part 1. What stands in the way of the single boundary

### 1a. Why kv freezes at settle and rs at prepare

**What `KvStore::fold` does at settle.** Two things in one call (skv:1797-1824): it advances the counter
(`table.freeze`, a max, pt:479-483) and it seals the whole pages below `min(logged.len(), folded)` under
content keys and extends the prompt reach (`reach_prompt`). Sealing needs the fire's rows in the log, and
the log is closed at finalize (`log_end`, pf:4146-4171; "What the fire froze seals now that its rows are
logged.", pf:4172-4175). Sealing also needs success, because a sealed page is offered to other processes
through the shared trie; `settle_fold` calls `kv.fold` only on success (pf:4067-4069).

So the settle timing protects **sealing**. It does not protect the counter. The readers of K between a
fire's prepare and its settle are:

| reader | cite | does it need the old value |
|---|---|---|
| frozen-prefix log check of the next piece or pass | pf:3456, 3470-3472, 3479-3485 | No. The old value is only more permissive than the cursor's rewind check (geo:433-439), which already uses the advanced C. |
| device-placed log cut | pf:4421-4423 | No. It already takes `max(K, C)`. |
| `settle_fold`'s base for a device count | pf:4045, 4061 | No. It wants every earlier fold applied; a stale K is the cause of D4. |
| sealing bound inside `fold` | skv:1805-1808 | No. It is bounded by the log as well, and the log of an unsettled fire is cut at its pending mark (pt:287-296). |
| page-table fork | pt:226-245 (235-239) | No. It copies the stale K while the rs fork copies the advanced R (srs:850-856); that is D3. |
| export for save | pt:1871-1872 | Not reachable with a fire in flight (see 1c). |

No reader needs the old value.

**What rs's prepare timing protects.** The rs publish at prepare swaps the state slot and `commit_folds`
advances `frozen` and drops the buffer pages behind it (pfrs:569; srs:1772, 1874-1896). The pair (state
slot, `frozen`) is one fact: the slot holds the state as of `frozen`. The next pipelined fire reads that
pair at its own prepare, before the first fire has run: `host_rs_plan` (pf:539-543), `device_rs_plan`
(pf:590-594), the `Buffer` verb, which hands `frozen` to the engine as the lane's `folded` (pfrs:532-566,
read at 543), `check_unfolded` / `check_carried` (pf:617-650), `BelowFrozen` (srs:1531-1534) and the
hidden snapshot sets placed at `frozen + row` (srs:1778-1800). These need the **new** value. Moving rs to
settle would make every rs fire wait for the one before it; pipelining on rs models would be lost, which
the no-regression gate forbids.

**Conclusion.** The boundary advances at prepare. kv is the odd one, and only its counter: the change is
to advance the counter at prepare and leave `KvStore::fold` at settle for sealing (its `freeze` is then a
no-op for a host-known count, because `freeze` is a max). This is the smaller change of the two the
ruling names: rs keeps its timing, C's advance moves from the end of submit to each piece's prepare, and
K gains one `freeze` call in a kv critical section the pass already takes.

**The ordinary failure path today.** Nothing un-advances. After a failed fire:

- R and the state slot stay as published. `RsTxnsGuard`'s drop abandons the transaction (releases
  snapshot sets, settles) and does not move `frozen` back or un-publish the slot (pf:247-278; pfrs:9-16).
- C stays advanced ("A fire that failed left the cursor ahead of what was written.", lease:398).
- The log is cut back (`log_end(None)`, pf:4147-4148, 4159-4167), `kv.fold` is skipped (pf:4067-4069),
  checkpoints are released (`take_checkpoint`, pf:1868-1907).
- The pipeline's failure is set (`complete_finalize`, pf:3992-4011; detached, pf:4013-4033; or
  `record_submit_failure`).

Two of the three counters are already advanced after a failed fire, so advancing K at prepare adds no new
exposure. "Un-advance on failure" is not something the code has, and it cannot be added cheaply: a later
fire of the same pipeline may already have been prepared against the advanced R and state slot.

**Is "the set dies with its pipeline" true for every failure?** No. Three cases survive:

1. *The set on a new pipeline.* After the failed pipeline is closed and drained, `claim` gives the set to
   a new pipeline (crates/runtime/src/store.rs:97-100). That pass reads C and R advanced past rows that
   were never written. Fix site: that arm, add `existing.failure().is_none()` (or return the failure).
   Not verified whether any program does this; none was found that fires a set after its pipeline failed.
2. *Forks.* A child carries the parent's scope (`forked`, kws:207-210), so it dies with the parent under
   the same rule and survives under the same hole.
3. *A piece prepared and never fired, on a pipeline that is not failed.* `reserve` publishes rs and
   commits folds (inside `prepare_reserved`, pf:1670-1690). The returns after it at pf:3477, 3479-3485
   and 3528-3530 leave R advanced and the state slot published for a fire that does not exist, and the
   pipeline stays usable. To reach 3479-3485 a pass needs R <= start < K on an rs set; no program found
   reaches it. Under the single boundary the 3456 check is the same check as geo:433-439 and goes before
   `reserve`; the two remaining returns (3477, 3528-3530) should call `record_submit_failure`.

With the claim fix and those two returns failing the pipeline, "a failed fire leaves the boundary
advanced and the set is dead" is true for every failure, and nothing live reads an advanced boundary.

### 1b. Every reader and writer of the three counters

"Under B" says what the site reads or does when `WorkingSetEntry.folded` is the one boundary (1c).

**C — `Cursor.folded` (hws:20-61)**

| site | cite | R/W, when | under B |
|---|---|---|---|
| `pass_lanes` | pf:2454-2491 (calls 2820, 4293, 674) | read, top of a pass | reads B under the kv lock; `folded_exact` means no device-counted fold in flight |
| guest `folded()` | hws:179-181 | read | reads B |
| rewind refusal | geo:433-439 | read (`folded_exact && start < folded`) | reads B; subsumes the log check at pf:3456 |
| `fold_at`, the F of `To(F + c)` | pf:2560-2608 (2580-2588) | read | B, read once at the top of the pass |
| `fold_at`, reach | pf:2593-2596 | reads `reserved` | not the boundary; unchanged |
| `advance_cursors`, the fold | pf:2496-2554 (2549-2552), call 3623 | write, once at the end of submit | deleted; the advance is the freeze at each piece's prepare |
| `advance_cursors`, unsettled / exact | pf:2513-2519 | write | stays; counts device-counted folds in flight |
| `advance_cursors`, device branch | pf:2520-2523, call 4641 | no write (`tail = None; continue`) | B advances at prepare for a host-known count |
| `settle_fold`, cursor part | pf:4083-4092 | write, `C = K` when the last fold settles | only `unsettled -= 1` |
| device log cut | pf:4421 | read | B |
| float pass | float:334-336 | write, `+1` | B `+= 1` in the step of the rs fold-one (float:267-274) |
| host fork | hws:183-254 (196-199, 249-253) | value copy into a new cursor | child reads its own entry's B |
| handle fork | held:111-135 (128) | clone | same |
| park / adopt | lease:361-407, held:43-84; lease:214-215 | moved by value | cursor carries tail, reserved, spans only |
| stat | held:164-190 | read | B from the page table |
| save / load | snap:781-793; 946-953 | write to / read from the image | B from the page-table export (pt:1871-1872, 1906-1907) |
| `settle_device_folds` | pf:664-685, called 2821-2825 | reads `folded_exact` | unchanged |
| `open_pipeline`'s drain | pf:2658-2682 (`drain_settled` 2674, pf:3920-3938) | runs `settle_fold` for settled fires before the pass reads the cursor | unchanged |

**K — `WorkingSetEntry.folded` (pt:132)**

| site | cite | R/W, when | under B |
|---|---|---|---|
| `PageTable::freeze` | pt:479-483 | write (max) | the one writer of B |
| `KvStore::fold` from `settle_fold` | skv:1797-1824; pf:4067-4069 | write and seal, at settle on success | seal only, for a host-known count; the advance, for a device count |
| `settle_fold`, device-count base | pf:4045, 4061 | read | B |
| `settle_fold` into the cursor | pf:4071, 4089 | read | gone |
| frozen-prefix log check | pf:3456, 3470-3472, 3479-3485 | read, at each piece's prepare | B before this piece's own advance; moved before `reserve` |
| device log cut | pf:4423 | read | B |
| `attach_run` (supply, resume) | skv:1839-1879 (1875-1877); pf:1285, 1516 | write, before the pass's fold is placed | see D6 in Part 2 |
| page-table fork | pt:226-245 (235-239) | copy | B |
| export / import | pt:1871-1872; 1906-1907 | copy | B |
| sealing bound | skv:1805-1808 | read | B |
| `sealed_prefix` | pt:299-301 | not a read of K (it reads the sealed pages) | unchanged |

**R — `RsEntry.frozen` (srs:256)**

| site | cite | R/W, when | under B |
|---|---|---|---|
| `RsStore::fold` via `commit_folds` | srs:1874-1896 (1882), 1772; pfrs:569 | write, at prepare | follows B in the same step |
| `settle_fold`, rs part | pf:4076-4081 | write, at settle, device count, success | follows B in the same step |
| windowed sets | srs:879-883; pfrs:412-415 | `position` is the window's `committed`, else `frozen` | unchanged; the window commit is the rs follower |
| `host_rs_plan` | pf:510-572 (539-543) | read | B |
| `device_rs_plan` | pf:576-606 (590-594, `writes` 600) | read | B |
| `Buffer` verb (the engine's `folded`) | pfrs:532-566 (543) | read | B |
| `check_unfolded` / `check_carried` | pf:617-650 | read | B |
| `BelowFrozen` | srs:94-95, 1531-1534 | read | B |
| `publish_prevalidated` | srs:1778-1800 (1795-1796) | read | B |
| rs fork | srs:850-856 | clone | follows the kv fork |
| `adopt_checkpoint` | srs:1210-1241 (1234) | write, `= checkpoint's frozen`, into a fresh set only (1211-1217) | must agree with what kv attached; see D6 |
| `forget_state` | srs:1243-1250 (1247) | write, `= 0` | see below |
| `take_checkpoint`, `checkpoint`, `checkpoint_at` | pf:1868-1907; srs:1028, 1056 | read | B |
| rs export / import | srs:740-746; 838-848 | copy | B |
| save / load | snap:426; 1021 | copy | B |
| `empty_base` | srs:1507-1517 | read | B |

An eighth divergence found while filling the table: on a resume mismatch (`mapped != pages`) `resume`
calls `forget_state`, which sets R to 0, after `attach_run` has already frozen K to what it mapped
(pf:1492-1530; skv:1875-1877; srs:1247). K and R then differ by the mapped run. Under B the attach must
not move the boundary unless both halves resumed.

### 1c. Where the value lives, who advances it, and the step in which the others follow

**Home.** `WorkingSetEntry.folded` (pt:132), read by `KvStore::folded` (skv:1826-1828), written by
`PageTable::freeze` (pt:479-483). Reasons: every `WorkingSet` has a kv entry and only some have an rs
entry (hws:63-67); the field is already copied by fork and carried by export and import; and the kv lock
is the first lock taken wherever a pass takes more than one (kv, then rs, then cursor; never nested in
the code read).

`Cursor.folded` is deleted. The cursor keeps `unsettled` / `folded_exact`, `tail`, `reserved`, `spans`.
`RsEntry.frozen` stays as the rs store's follower: the rs code needs it under the rs mutex for buffer
paging, and it is advanced in the same step as B, never on its own.

**Who advances it, and the step.**

| fold | advance point | the step |
|---|---|---|
| host-known count, host-placed pass | the kv section the piece already takes right after `reserve` returns (pf:3455-3474), with the `fold_at` call (pf:3504) moved above it | read B, log check, `log_begin`, `freeze(to)`; rs committed just before by `prepare_reserved` (pf:1670-1690, rs at 1688) with the same `to`. There is no await between `reserve` returning (pf:3443) and that section. |
| host-known count, device-placed pass | the kv section at pf:4416-4442, after the reserve at pf:4393-4413 | same, with `to = B + min(c, rows, reserved - B)` (see below) |
| float pass | float:334-336 | `freeze(B + 1)` beside the rs fold-one (float:267-274); one more kv lock a float fire |
| device count | `settle_fold` (pf:4036-4096), as today | kv, then rs, on success |
| host-known count while a device-counted fold is in flight | `settle_fold`, as `Settled{count: Some(c)}` | the lower-bound case the ruling names |

`to` stays absolute, `min(F_pass + c, first lane's end, piece end)`, so the pieces of a cut pass are
idempotent under `freeze`'s max. `FoldBy::To(to)` still travels to settle, for sealing only.

**Device-placed host-known count: the bound.** The ruling's interim states `min(c, reserved - frozen)`.
That is not what the engine commits. `reserved` can be past the rows the device then writes: `fold_at`
on a device-placed pass sets `reach = cursor.reserved` (pf:2593-2596), `settle_fold` takes
`min(K + c, reach)` (pf:4061), `device_rs_plan` passes `reserved` to `group_fold` (pf:593-594), and
`Fold::All` is the host count `u32::MAX`. A set that reserves 17 slots and fires one row with `Fold::All`
gets K = R = F + 17 after one fire (examples/qwen3-tts/src/lib.rs:640-672, 807, 369-382, by reading; not
run). The engine commits at `min(folded + fold, first + rows)`
(crates/engine/src/rs.rs:62). The host knows `rows` on a device-placed pass (the first lane's token
count) and does not know `first`. So:

- `c <= rows` is exact at prepare.
- `c > rows`, `Fold::All` included, is exact as `B + rows` when the lane carries no unfolded rows in.
- A lane that carries unfolded rows in and folds past its own rows is exact only from the engine's
  report; it stays a settle-time fold (the existing `report_on`, pf:2599).

Recommended bound: `min(c, rows of the first lane, reserved - B)`, passed to `fold_at` at pf:4589-4599 so
it returns `To` (no `unsettled += 1`, so host-placed rs passes stop draining behind such a fire), with the
same bound given to `group_fold` at pf:594. The third case is a choice for Pie Leader: settle-time, or
refused by name. No program found reaches it.

**Size.** Runtime crate only. No engine, WIT or SDK change; the SDK doc of `folded()`
(crates/inferlet/src/eta.rs:410-414) stays true. About 12 to 15 sites in 7 files:

- pipeline/fire.rs: `pass_lanes`, `advance_cursors`, the two kv sections (3455-3474, 4416-4442), the
  `fold_at` call at 4589-4599, `device_rs_plan`'s bound, `settle_fold`'s cursor part, `supply` / `resume`,
  the two early returns, and the unit tests that build cursors with `folded` (pf:5540-5786);
- pipeline/fire/float.rs: 334-336;
- store/kv.rs: a counter-only freeze beside `fold`; `attach_run`;
- inferlet/host/working_set.rs: `Cursor`, `folded()`, `fork`;
- handle/held.rs: stat; handle/snapshot.rs: save and load (the saved cursor field needs a format note);
- crates/runtime/src/store.rs: the claim arm.

**Can step 1 carry it.** Yes, as its first commit. It is smaller than the interim fixes of Part 2 taken
together: those touch the same functions to keep three counters agreeing, and four of them (D2, D5, the
cut-piece origin, most of D3) disappear under B instead of being fixed.

**The "not verified" items that bear on it.**

- *`settle_fold` atomicity against a pass being staged.* Not atomic. It is three critical sections taken
  in turn: kv (pf:4044-4072), rs (4076-4081), cursor (4083-4092). Finalizers exclude each other through
  the queue's finalizer mutex (pf:60, 79-81; held at pf:2023, 3927, 3945,
  inferlet/host/forward.rs:272, 326, planner/exec.rs:449). A pass being staged does not hold that mutex
  across its length; `open_pipeline` takes it for `drain_settled` and lets it go. The planner finalizes
  settled host-placed fires on its own task during eviction (planner/exec.rs:446-471, called at 488;
  host-placed only, pf:1723-1731), so a `settle_fold` can run between two awaits of a submit. What keeps
  this sound today: a `To` fold is a max; a `Settled` fold updates the cursor last and the cursor is
  inexact until then, so `folded_exact == true` implies all three are updated; host-placed rs passes drain
  first (pf:664-685). The open window is between 4072 and 4076: K is advanced and R is not, and a
  device-placed pass staged there reads R one fold behind (it goes ahead of the settle by design,
  pf:4291-4293). Under B the structure is the same and the window is the ruling's own lower-bound case.
  Closing it means taking the rs lock inside the kv lock in `settle_fold`; that is a new nesting and
  needs a lock-order check. Not verified: whether a submit can be mid-await while the planner's eviction
  runs (the fence is raised and the process notified first, planner/exec.rs:486-487).
- *Can fork run with fires in flight.* Yes, and it is meant to. The host fork has no wait (hws:183-254;
  its gates are residency 188, closed pipeline 193-195, failed pipeline 218-220), and the page-table fork
  is written for it: "The child holds what the parent's log knows, short of what the parent's fires on
  their way may still change, and what it froze." (pt:229-230; the log is cut at the least pending mark,
  231-234). A program does it: examples/consensus-decoding/src/lib.rs:135-141. The handle-level fork
  waits for settled moves (held:111-135, 210-262).
- *Does save require every fire settled.* Yes. `save` takes a `&Held` (snap:543); a `Held` is made by
  park, where "no fire may be in flight" (held:44), after the lease's `settle`, which "Runs once every
  fire has settled." (lease:316-358, 317). Load restores `folded_exact: true, unsettled: 0`
  (snap:946-953).
- *Can `reserved` be past the rows the device then writes.* Yes; see the bound above.
- *Do the non-submit early returns of a cut pass fail the pipeline.* No; see correction 2.

---

## Part 2. The interim

Each fix is stated with its site and a test. Tests are unit tests in pipeline/fire.rs beside the existing
ones (helpers `cursor`, `lanes`, `request`, `frozen_at`, `taken`, `report` at pf:5540-5606) unless said
otherwise. Under the single boundary of Part 1, D2, D5 and the cut-piece origin need no fix of their own;
the tests still apply.

### D2. Device-placed pass, host-known count

- **Site.** `fold_at` call, pf:4589-4599: pass the reach `min(c, rows, reserved - F)` instead of
  `u64::MAX` so a host-known count on an exact cursor is `To`. `advance_cursors`, pf:2520-2523 (call
  4641): advance C by the same amount instead of `continue`. `device_rs_plan`, pf:594: the same bound.
- **Why `rows` is in the bound.** See 1c. With `min(c, reserved - frozen)` alone, `Fold::All` puts C, K
  and R at `reserved`, past what was written.
- **Test.** Beside `a_fold_the_host_knows_is_placed_at_submit` (pf:5653) and
  `a_device_fold_is_reported_and_placed_when_it_settles` (pf:5681): a host count on device-placed rows
  gives `cursor.folded == F + min(c, rows, reserved - F)` at submit, stays exact, and after `settle_fold`
  K and R equal it; a second such pass staged before the first settles reads the advanced value.
- **Reached by.** examples/qwen3-tts/src/lib.rs:640-672 (device-carried `slots`, `Fold::All`, a fresh set
  a frame, one row a fire, pipelined by `submit_frame` 825-832). Not verified by running that it takes the
  device-geometry path, nor whether its model has recurrent layers.

### D6. Supply or resume, then a pass folding less than what was attached

- **Site.** The ruling's form ("supply/resume advance the cursor to what they attached") must be applied
  after the pass's fold is placed: carry `supplied` (pf:3044-3072) into `advance_cursors` and take
  `C = max(F + min(c, end), S)` at pf:2549-2552; the same for `resume` (pf:1492-1530). Advancing C before
  `fold_at` would turn the pass's fold into `To(S + c)`.
- **What this costs.** The boundary becomes `max(F + c, S)`, and S depends on what the cache happens to
  hold. That contradicts "the guest's origin is the value the pass states".
- **The form that fits one boundary.** kv attach maps without freezing (drop skv:1875-1877; the mapped
  pages are sealed and so copy-on-write, skv:2177-2178, whatever K says), and rs resume adopts a
  checkpoint only at or below what the pass folds (cap the reach at `(F + min(c, rows)) / page`). Then
  the boundary after the pass is what the pass states. With `Fold::All` prefills, which is every builtin
  (crates/builtins/inferlets/compat/src/generate.rs:436, 625, 708; prefill/src/lib.rs:41), the cap never
  binds, so nothing is lost for them. The resume-mismatch case of 1b goes away with it.
- **Test.** Store-level: seal a prompt in one set, supply it to a fresh set with a pass that folds 0 and
  with one that folds a short count; C, K and R agree, and under the second form equal the stated fold.
- **Reached by.** `supply` runs for every keyed host-placed single-lane pass (keyability at pf:2949-2954,
  1180-1198); kv attach is not a guest call. A keyed prefill that folds short of its rows:
  examples/snapkv-eviction/src/lib.rs:172-182 (`Fold::Count`), with a later rewrite of the rows past the
  fold (252-265). Not verified whether that pass is keyable.

### D1. Several lanes on one set, different ends

- **Site.** `advance_cursors`, pf:2524-2529: the fold's bound is the first lane's landing end
  (`landings[mine[0]].end()`), as `host_rs_plan` bounds it (pf:543) and as the kv `To` should (the reach
  given to `fold_at`). `tail` and `reserved` keep the max over lanes (2530-2531).
- **Test.** Two lanes on one set writing n and n + 1 rows with `Fold::All`: `cursor.folded == F + n`,
  equal to the rs plan's fold and to the kv `To`.
- **Reached by.** No program found. Several lanes on one set appear in beam search
  (examples/beam-search/src/lib.rs:172, with `Fold::None` at 192 and device slots; the JS and Python
  twins; the SDK unit tests), never with a non-zero host fold. The fix is three lines, so a refusal by
  name is not needed; if the fix is deferred this is the first candidate for one.

### D3 / D4. Fork with fires in flight

- **The event.** `drain_pipeline_fires` (pf:2008-2036) awaits the front op's `completion_signal()`
  (pf:1715-1721) under the finalizer guard and finalizes in pipeline order. It is the wait
  `settle_device_folds` already uses (pf:673). It is an event wait and it drives finalization.
  `KvSuspendHandle::quiesce` (kws:141-157, notified from the lease's drop at 112) is an event too, but it
  only waits; someone else must finalize, so it is not enough alone.
- **Site.** Host `fork`, hws:183-254, before the kv fork at 222-226: when the set has fires in flight,
  await `drain_pipeline_fires(self, &pipeline.fires)`. The host there is `ProcessCtx`, the context
  `drain_pipeline_fires` takes. A set is scoped to one pipeline, so its fires are all on `on`'s queue.
  The test for "in flight" in the interim is the kv set's `active_leases` (kws:135-139), because
  `cursor.unsettled` counts only `Settled` folds and a `To` fold in flight also leaves K behind.
- **Cost.** examples/consensus-decoding/src/lib.rs:136-139 forks before taking the prefill at 141; its
  first fork would wait for the prefill it takes right after. The wait moves; it is not added.
- **Under B.** Only `unsettled > 0` needs the wait, since a host-known fold has already advanced B. No
  program found forks with a device-counted fold in flight (the one program that folds by device count,
  examples/mtp-speculative-decoding/src/lib.rs:247-266, 344, forks at 363 and 397 after the fire is
  taken), so under B consensus-decoding does not wait at all.
- **Test.** With the pending-op harness (`PendingOp::TestStub`, as in pf:5010): fork with a stub op in
  flight; the fork completes when the op settles, and the child's cursor, K and R agree and are exact.

### D5. A cut pass that returns early

- **Site.** Advance the cursor per piece, right after the piece is pushed (pf:3594-3609), instead of once
  at pf:3623. "Exact from the stores" works only for rs sets: K has not moved for the pieces in flight,
  so the stores do not hold the answer for a kv-only set. The per-piece advance works for both.
- **The piece prepared and not pushed.** The returns at pf:3477 and 3528-3530 call
  `record_submit_failure`; the check at 3479-3485 moves before `reserve` (see 1a, case 3).
- **Test.** Factor the per-piece advance as a function and test it like
  `a_fire_carrying_a_span_leaves_its_lag` (pf:5615): after k of n pieces, `cursor.folded` is the fold of
  those k pieces and `tail` / `reserved` cover them.
- **Reached by.** Any cut pass whose later piece fails to reserve (pf:3442). Cutting applies to keyed and
  long prefills (pf:2958-2964).

### Cut passes: the origin of each piece

- **Site.** Where a piece's request is built (pf:3327-3340): the piece states
  `max(F, min(F + c, piece start))`.
- **Test.** F = 10, pieces [10, 74) and [74, 138): `All` gives 10 and 74; c = 20 gives 10 and 30; c = 0
  gives 10 and 10. For an rs set the value equals `frozen` as that piece's prepare finds it, for c in
  {0, all}; rs passes are cut only then (pf:2940-2943, refusal at 3226-3240).

### Refusals by name

None is required once the fixes above are in. The paths no existing program reaches, by a search of
examples/, crates/builtins/inferlets/, the three SDKs (crates/inferlet, packages/javascript/inferlet,
packages/python/inferlet; there is no `sdk/` directory and models/ holds only `.poem` files) and the
tests' guest programs:

1. several lanes on one set with a non-zero host fold (D1);
2. a fork taken while a device-counted fold is in flight, and a device-counted fold in such a child (D4);
3. on a device-placed pass, a host count past the first lane's rows on a set that carries unfolded rows
   in (the third case of the bound in 1c);
4. a pass on a set whose earlier pipeline failed (1a, case 1). This one is a refusal in any case.

No guest reads `folded()` outside the SDK wrappers (crates/inferlet/src/eta.rs:410-414 and the JS and
Python bridges).

---

## Part 3. May the runtime cut replay-only pieces

**Answer.** Not today, and not by lifting the two refusals. The landing a replay-only piece needs is a
state row apart from the set's boundary, which is step-2 material (FOLLOWUP B.2). So the carried-rows
contract cannot go entirely in step 1; `check_carried` / `check_unfolded` (pf:617-650) and
`UNFOLDED_ROWS` (crates/engine/src/fire.rs:569-590) stay as the refusal until step 2.

**What the runtime refusal protects (pf:529-537).** The fold contract. It is reached only when the pass
has rs sets (the plan returns early without them, pf:518-520). Under today's `Buffer` verb a zero-row
lane has two outcomes. With fold 0 the engine replays and keeps nothing: the commit is 0
(crates/engine/src/rs.rs:62, 67) and a fold-0 lane never commits. With fold > 0 it is a fire that moves
the state and writes no rows, which is the runtime folding behind the guest ("the runtime never folds
more behind its back", pf:610-616) or the guest folding in a pass that writes nothing ("fold with the
pass that writes it", pf:529-537). It is a contract check, not a kernel limit.

**What the engine refusal protects.** There is one refusal, in shared code:
crates/poem-exec/src/fire/compose.rs:399-401, reached by CUDA and ROCm
(crates/engine-cuda/src/serve/prepare.rs:477; ROCm is the same crate over `kernels_rocm`,
crates/engine-cuda/src/lib.rs:6), Vulkan and wgpu (crates/engine-gpu/src/serve.rs:2590), Metal
(crates/engine-metal/src/serve.rs:3350) and XLA (crates/engine-xla/src/serve.rs:1610). `Lane::validate_for`
has no empty-token check of its own (crates/engine/src/fire.rs:305-370). The refusal itself is a sanity
check ("a lane is its rows"). What it stands in front of, per engine, by reading (two read-only audits;
the sites marked * were re-read here):

| engine | what sits behind the refusal | kind |
|---|---|---|
| shared walk | a class window with zero own rows runs only collective nodes (crates/poem-exec/src/fire/walk.rs:178*, 203-219), so the replay and the commit would be skipped silently | run rule |
| CUDA, ROCm | readout defaults to the lane's last own row and refuses `row >= owned` (prepare.rs:495-507*); `Readout::None` and `Last` are one value (api.rs:1523-1526), so a lane cannot say "no readout" | readout assumption |
| CUDA, ROCm | a window whose summed replays exceed `span.rows.max(REPLAYED)` walks unbodied (prepare.rs:1563-1570*) and is tallied `Unrecorded` (enqueue.rs:908-926); several replay-only lanes in one window hit it. Under the graph-capture rule that is a bug, not a fallback | record key |
| CUDA, ROCm | an all-empty fire reads `live_rows == 0` as "every staged row valid" (inputs.rs:947-951) | staging sentinel |
| Metal | readout refuses `owned == 0` even for `Readout::None` (serve.rs:4204-4217*) | readout assumption |
| Metal | replay scratch is sized at boot as `tokens + lanes * UNFOLDED_ROWS` with `lanes <= rows` assumed (serve.rs:1058-1074) | sizing |
| Vulkan, wgpu | nothing found. Readout skips a lane with no own rows (serve.rs:3744-3746*); scratch is sized at the fire (3174-3192). Vulkan's record key includes per-lane rows (crates/engine-vulkan/src/serve.rs:372-396), so a zero-row lane is a new recording | record key (new shape only) |
| XLA | nothing refuses. Readout skips zero-row lanes (serve.rs:3275-3277*); an empty lane turns off the decode fast path (1396) and makes a new compiled program (2373-2410) | record key (new shape only) |

No ssm kernel needs own rows > 0: the CUDA chunk kernels clamp to the commit length and skip a
non-positive span (crates/kernels-cuda/kernels/ssm/chunk.cuh:29-39); the Metal and Slang conv kernels
take `span = own + replay` and return when it is not positive
(crates/kernels-metal/kernels/ssm/conv1d.metal:321-328; crates/kernels-slang/kernels/ssm/conv1d.slang:204-218);
XLA writes when `span > 0` and `keep > 0` (crates/kernels-xla/src/lanes.rs:539-549). On the kv side a
zero-row lane still takes a seat and at least one page (crates/poem-exec/src/store/kv.rs:438, 469-474;
CUDA prepare.rs:906-910). engine-cpu has no recurrent state (crates/engine-cpu/src/forward.rs:512-517).

The refusal is not side-effect free on CUDA and Metal: reference takes, slot copies and the state fork
run before it (CUDA api.rs:1420-1445, prepare.rs:141-145; Metal api.rs:1163-1173 against 1214-1225).

**What a replay-only piece would have to state.**

1. The rows replayed: `state_at..x`, at most one rung of `UNFOLDED_ROWS`.
2. The state landing, as a slot apart from the set's boundary. If the piece instead moved `frozen`, it
   would be the runtime folding behind the guest and a second writer of the boundary, against the ruling
   above and pf:610-616.
3. No kv writes: no seat, no page, no log, no lease on kv pages.
4. No readout: a real "none", which CUDA and Metal cannot say today.
5. A record and arm key for the shape, so the fire is captured and not walked eagerly.

**Smallest change set if it is pursued in step 2.** pf:529-537; compose.rs:399-401 with a run rule at
walk.rs:178; CUDA prepare.rs:494-510, enqueue.rs:987, prepare.rs:1563-1570, inputs.rs:947-951; Metal
serve.rs:4204-4217 and 1062. No site found on engine-gpu or engine-xla.

---

## Not verified

- Nothing here was built or run. Every "by reading" verdict on an engine with the refusal lifted is
  untested.
- qwen3-tts: that its predictor pass takes the device-geometry path, and whether its model has recurrent
  layers.
- snapkv-eviction: whether its prefill is keyable, so whether it reaches D6.
- Whether any program fires a set on a new pipeline after its first pipeline failed.
- Whether a submit can be mid-await while the planner's eviction finalizes detached fires.
- Whether the kv fire lease is dropped exactly at finalize; assumed from the lease riding in the fire's
  per-set record (pf:4144), which makes `active_leases` the test for "fires in flight" in D3.
- Whether `reserve` failing (pf:3442, 4412) can leave rs published; the three returns listed in 1a are
  the ones read to come after a successful `reserve`.
- Engines, with an empty lane: ROCm ssm kernels (`mixers.hip`); attention kernels with a zero-row seat;
  CUDA's `Ladder::of` for an all-empty class and whether arming records such a shape; XLA's warm ladder
  coverage; Metal's `mask::stage` and `Windows::of`; engine-gpu's harvest of an empty read span.
