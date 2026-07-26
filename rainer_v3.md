# Project Rainer v3: Divisible Residency and a Typed Liveness Core

Date: 2026-07-26
Status: **PROPOSAL — nothing here is implemented.** Every measurement cited
is from the 33-case pie-vs-vLLM matrix and the two request-destruction
root-causes recorded in `CONTENTION_FOLLOWUP.md` §18–§18.10.
Relationship to `rainer.md`: v3 **does not replace v2's boundary pass** —
it re-founds its justification, and adds the three things v2 leaves
untouched. Read v2 (§11–§17) first; v3 is a delta on it.

---

## 1. Why v3 exists: the justification changed, not the design

`rainer.md` §10 parked v2 with an explicit verdict:

> v2's remaining value is the *seam-class* simplification of §14
> (membership pushed, not inferred) — **a robustness refactor to take
> incrementally, ... not a rewrite justified by throughput.**

That verdict was correct on its own terms and **remains correct**: v2 is
not justified by throughput (§4.3 below argues it moves the contended
number less than v2 §15 claims). But it weighed the robustness half
against no evidence, because none existed yet.

There is evidence now. In one session, at one config, the shipped v1 tree:

- **destroyed a live request in 4 of 7 runs** (§18.7), then
- **destroyed one in 1 of 26 runs** after the first fix (§18.9), from a
  *different* state, then
- **took a third fix** to reach 45 of 45 (§18.10),
- and the first, natural attempt at fix #2 introduced a **livelock**
  (654 → 49 tok/s) that the existing tests did not catch (§18.9).

All four events are the same defect class — a decision taken on state that
was already stale when it was acted on — and all four live in the seam v2
deletes. The claim of v3 is therefore narrow and, I think, hard to argue
with:

> **v2 should be scheduled as a correctness project, not a performance
> project. Its performance case is weak; its correctness case is now
> measured.**

---

## 2. What the session actually measured

Grounding, so v3 is not argued from taste. One RTX 4090, Qwen3-0.6B,
tp=1, greedy, both engines on a token-for-token identical KV budget
(pie `total_pages = P`, vLLM `num_gpu_blocks_override = P, block_size 16`).

| finding | number | source |
|---|---|---|
| parity at/below capacity | 1.000x / 0.987x | §18.2 A1/A2 |
| single oversubscribed wave | **0.967x** | §18.2 F4 |
| sustained churn, worst cell | **0.705x** | §18.2 E3 |
| gap tracks turnover, not pressure | 0 refills 0.967x → 3 refills 0.744x | §18.3 |
| evictions per request | 0.52 → 1.56 across the same axis | §18.3 |
| PCIe traffic per page of real demand | **1.48 pages** | §18.4 |
| pages moved to hand the head ONE page | **~33** (16.6 evicted + 16.6 restored; ~70 if the victim is at full length) | §18.4 |
| run-ahead overlap, roomy → 4x oversub | **185% → 83%** | §18.5 |
| pie's best over ALL admission levels | 8,770 vs vLLM 11,226 = **0.78x** | §18.6 |
| vLLM's mechanism | `Running: 24, Waiting: 107`, zero preemptions | §18.4 |

Two consequences worth stating plainly, because they close off the two
most natural "fixes":

- **Admission tuning cannot close the gap.** Sweeping pie's offered
  concurrency across every value at a fixed budget peaks at 0.78x. The
  remaining 22% is mechanism, not policy.
- **Contention is not the cost; *turnover* is.** A fully oversubscribed
  wave that never refills runs at 0.967x. The bill arrives when new
  processes enter a full pool.

---

## 3. The four structural defects

### 3.1 Decisions taken on stale snapshots (v2 fixes this)

`plan_eviction` samples the candidate set under the lock, releases it to
quote (quoting takes store locks), and only later reaches the endgame
predicates. A process that was `Evicting`/`Restoring` at sample time is in
**neither** the eligible nor the vetoed list; moments later it is
`Resident` holding pages. The kill is then decided against a scan taken
before that process existed as a candidate. Two requests died to exactly
this (§18.9 repro A, §18.10 repro B — the second holding `Pages(13)` and
eligible by every rule while the planner destroyed a request for want of
one page).

The same shape produced the historical failures: the §1 livelock (a victim
that held nothing), and the §12 three-party frame⇄copy⇄resize deadlock.

**v2's answer is the right one**: one boundary pass per frame is the only
writer, and every endgame predicate becomes an assertion over a single
consistent global snapshot (v2 §13–§14). Nothing in v3 improves on it.

**What v3 adds:** the fix that shipped (`last_resort_evict()`, §18.10)
should be understood as a *manual emulation of one property of the
boundary pass* — re-read a consistent snapshot at the instant of an
irreversible decision. It is a patch that v2 deletes. That is the
strongest available argument that v2 is the correct target rather than
more patching.

### 3.2 Liveness and policy share one funnel — enforced by nothing

`rainer.md` §15 already states the criterion:

> No timers, no knobs, **no heuristics in any liveness path** (the §6
> criterion).

It is a criterion, not a mechanism. E6 — one line of *hysteresis* — sat in
the liveness path unnoticed until it destroyed requests: a restored
process that parks on an unmet ask never reaches another `acquire`, so its
veto never lifts, and it was the only process that could fund the head
(§18.9). v2 keeps E6 (its §14 "remains" table) and keeps the criterion as
prose, so **v2 does not prevent this recurrence.**

**v3 proposal — make the criterion a type, not a comment.** Split the
question the planner is really asking into two, and give them different
types and different freshness contracts:

| question | kind | freshness | may be wrong? |
|---|---|---|---|
| *Does a legal victim exist?* | liveness | must be a boundary-pass snapshot | **no** |
| *Which legal victim is best?* | policy | may be stale/racy | yes — costs perf only |

Concretely: a `VictimSet` produced by the boundary pass (exact, total,
snapshot-consistent) and an `Ordering` applied to it (lease-quiescence,
E6 hysteresis, youngest-first). Endgame predicates may consume only
`VictimSet`; heuristics may only reorder within it and can never empty it.
`Impossible`/`Hog`/`Starved` become assertions over `VictimSet::is_empty()`
and nothing else. E6 then *cannot* be load-bearing for liveness, because
it lives on the wrong side of the type boundary — the recurrence is
unrepresentable rather than merely discouraged.

This subsumes the ad-hoc split the §18.10 fix created by hand
(`last_resort_evict` = existence, `quote_and_pick` ordering = policy),
which today exists at exactly one call site and is not enforced anywhere.

### 3.3 `ReclaimQuote` conflates a durable fact with a transient one

**This is a latent bug that v2 does not fix** — its §14 keeps the "hog
endgame ... unchanged".

```rust
// check_hog
let held = kv_reclaim_quotes(&[head_pid])… .map_or(0, ReclaimQuote::pages)
         + swapped_page_count(head_pid, model, driver);
…
if outcome.is_some() || demand.kv_pages.saturating_add(held) <= total { return None; }
```

`ReclaimQuote::pages()` returns **0 for every `Nothing` variant**. So a
head holding 39 pages reads as holding **0** the moment one in-flight pin
overlaps its residency (`Nothing(Pinned)`) or its pages are shared
(`Nothing(AllShared)`). The hog predicate then passes trivially and the
hog is never detected; control falls through to the starvation rung, which
destroys the **innocent youngest** instead of the actual hog.

Status: **code-evident, not yet observed at runtime.** I found it by
reading while root-causing §18.10 and did not build a repro.

The irony is that this is the exact failure `NoReclaim` was introduced to
prevent — `page_table.rs` says so:

> These are NOT interchangeable: each is cleared by a different event, and
> victim selection has to know which. **Collapsing them into one boolean is
> what allowed the reclaim ladder to re-pick a useless victim forever.**

The *reason* was split into variants; the *quantity* is still overloaded.
`pages()` answers "how much would suspending free right now" (transient,
policy) and is being read as "how much does this process hold" (durable,
liveness) — §3.2's confusion in miniature.

**v3 proposal:** two accessors with no implicit coercion —
`held_pages()` (durable: everything this process owns, resident or
swapped, pinned or not) for liveness predicates, and `reclaimable_now()
-> ReclaimQuote` (transient) for victim selection. `check_hog` takes the
former. Cheap, local, independent of everything else in this document,
and it should land first.

### 3.4 The eviction unit is not the ask unit — REJECTED 2026-07-26

Measured (§18.4): the head asks for **one page** — `head_pages=1
head_kind=allocation`, a decode crossing a page boundary. Funding it
evicts a **whole working set** (16.6 pages average) which then re-queues
for a **whole restore**. ~33 pages of PCIe move to deliver one (52,984
pages / 1,608 cycles; ~70 when the victim is at full length), and the
fleet pays 1.48 pages of traffic per page of real demand.

Residency is all-or-nothing because membership is defined over whole
working sets. That is a *unit mismatch*, not a bug, and no amount of
victim-selection cleverness removes it — §18.6's admission sweep is the
proof that the tuning branch tops out at 0.78x.

**What v2 does and does not do here.** v2's membership rule (longest FCFS
prefix whose Σ(held + declared) fits) genuinely helps: a new arrival is
simply out-of-prefix rather than implicitly admitted by acquiring, so the
"young arrival takes pages, then the head evicts a young member" cycle
(§18.3's turnover cost) is largely removed by construction. That is a real
improvement and v3 keeps it.

But it does **not** remove rotation, because members *grow*. Membership is
recomputed from `held + declared`, and `declared` is deliberately the next
fire's demand, never an extrapolation (v2 §2 as revised). So the prefix is
admitted on current size, grows past capacity, and must shrink again —
evicting whole working sets. The unit is unchanged, therefore §18.4's
amplification is unchanged.

> **DECISION 2026-07-26: rejected. Both parts, not deferred-with-intent.**
> The diagnosis below stands — this *is* where the contended gap
> physically is — but neither proposed cure survived scrutiny:
>
> - **(a) is not clearly positive in sign.** Attention needs every past
>   page each step, so a partially-evicted victim cannot run either way;
>   (a) buys only a cheaper *reinstatement* (~1 page instead of ~33).
>   Against that, the victim keeps ~32 pages it cannot use, where a whole
>   eviction would have freed them to feed other parked asks. Bytes moved
>   fall, effective capacity falls too, and §17.1's g4 is a measured
>   instance of the same trade going the wrong way (cheaper transfers →
>   rotation 280→374 → press 11,253→10,721).
> - **(b) costs batch size.** Admitting on final declared size holds
>   P=1092 to 32 processes; the measured admission sweep puts pie at
>   conc 32 at **7,190 tok/s, below today's 8,043**. A request's mean
>   working set over its life is about half its final size, and optimistic
>   admission is what buys the larger batch.
> - Both add substantial mechanism (page-run addressing, partial fences,
>   mask implications / a declaration protocol and its under-declaration
>   path) to a codebase whose measured problem is that it is already too
>   intricate to reason about (§18.7–§18.10).
>
> The contended 0.70–0.82x therefore **stands as accepted cost** for now
> — §17.1(c). The underlying constraint is real and worth stating: vLLM's
> relief valve is recompute-from-prompt, which costs zero data movement
> but is only sound because a vLLM sequence is a pure function of its
> prompt. An inferlet's KV is arbitrary guest-constructed state, so pie
> has no cheap relief valve, and every proposal to build one has so far
> cost more than it saved.

**Superseded proposal, kept for the record (two independent parts):**

**(a) Divisible residency.** Let a working set be *partially* resident:
evict a page *run* from the tail of a victim rather than the whole set,
and lift the fence at run granularity. Funding a 1-page ask should cost
~1 page, not ~33. This is `CONTENTION_FOLLOWUP.md` §17.1(b) ("restore
chunking / partial fence lift"), promoted here from a lever to the
central structural change, because §18.4 identifies it as where the gap
physically is. It is also the largest piece of work in this document and
the one most likely to have consequences I have not foreseen — attention
masks, page-run addressing and the fence's granularity all move.

> **Counter-evidence, and it is direct.** §17.1's g4 made transfers
> cheaper (batched copies + swap-stream completion) and press got
> *worse*: 11,253 → 10,721, because "cheap transfers sped up the
> evict⇄restore ROTATION (evictions 280→374): the slow copies had been an
> accidental thrash brake." Divisible residency makes each eviction
> cheaper in exactly the same way, so it inherits exactly that risk — it
> could raise rotation *frequency* faster than it lowers rotation *cost*.
>
> This is why (a) must not ship without (b) or an equivalent stabiliser:
> cheapening the mechanism while leaving membership churn-driven is the
> experiment that already failed once. **(a) alone is not a safe bet, and
> §18.4's arithmetic is necessary but not sufficient justification for
> it.**

**(b) Declared intent, not extrapolation.** Stable membership needs the
*final* working-set size, but declarations only carry the next step, so
the prefix necessarily overshoots and then contracts. The guest, however,
already knows: `max_tokens` is an inferlet input. Letting an inferlet
declare an intended total residency is **not** extrapolation — it is more
of exactly what v2 already calls "the only information source", and it
keeps the no-inference rule intact. Membership computed on declared
intent is stable, and stable membership is what makes turnover cheap.
(Unverified: a declaration is a hint, and a guest that under-declares must
be handled — presumably by re-declaring and re-entering the prefix at its
existing spawn position, which preserves FCFS.)

---

## 4. What v3 does not claim

Being explicit, because v2 §15's performance claim is the part of that
document this session did not reproduce.

1. **v3 does not claim the boundary pass fixes the contended gap.** v2 §15
   predicts coherent full-width waves pricing *above* vLLM. The measured
   gap under churn is 0.70–0.82x and is rotation economics (§18.4), which
   §3.4 argues v2 reduces but does not remove. The wave-coherence effect
   is real and worth having; it is not the same lever.
2. **v3 does not claim vLLM's design is the target.** vLLM never commits
   KV it cannot sustain, so it never moves any — but its preemption is
   *recompute from the prompt*, valid only because a vLLM sequence is a
   pure function of its prompt. **An inferlet's KV is arbitrary
   guest-constructed state, so recompute is not generally available to
   pie.** This is a real constraint, not an oversight, and anyone
   "fixing" the gap by copying vLLM will break the product's premise.
   (A guest-declared *reconstructible* hint would open the door for the
   subset that qualifies. Speculative; nothing here depends on it.)
3. **v3 does not claim starvation becomes unreachable.** With FCFS's
   "never evict your senior" rule, a state in which no victim younger than
   the head holds pages is still expressible, and a request still dies
   there. None has been observed since §18.10, but nothing proves it
   unreachable. Only §17.1(a) (restores funded from organic frees, with
   eviction redirected to the oldest *allocation* ask) removes it
   structurally — and that needs its own liveness argument against an
   immortal fleet starving an evictee.

---

## 5. What dies

Beyond v2 §14's list, v3 deletes the scaffolding this session added:

| dies | why |
|---|---|
| `last_resort_evict()` | a hand-rolled consistent snapshot at decision time; the boundary pass provides it (§3.1) |
| `is_wedged()` as a shared helper, and its double evaluation | one snapshot, one evaluation |
| the `e6_relaxed` flag threaded into `commit_evictions` | E6 cannot be liveness-bearing once typed out of that path (§3.2) |
| `StarveCause` | with one writer there is one reason, computed where it is known |
| `ReclaimQuote::pages()` as a holdings measure | replaced by `held_pages()` (§3.3) |

That six of these exist *only* because of defects found in one session is
itself the argument.

## 5b. Complexity ledger — does this make the code simpler?

Honestly: **three of the four proposals simplify; one does not.** Bundling
them under one title obscures that, so the ledger is explicit. Sizes are
from the shipped tree (`planner.rs` 2,000 lines + `exec.rs` 468 +
`grant.rs` 197 = 2,665).

| proposal | LOC effect | new heuristics/knobs? | verdict |
|---|---|---|---|
| §3.3 `ReclaimQuote` split | ~neutral (one accessor becomes two) | none | **strictly clearer** |
| §3.2 typed liveness boundary | **+** (a `VictimSet`/`Ordering` pair is new machinery) | none — it *removes* heuristics from the liveness path | **more code, less to reason about** |
| §3.1 / v2 boundary pass | **−− large** | none — deletes knobs and inference | **the real simplification** |
| ~~§3.4 divisible residency~~ | **+ +** | — | **REJECTED as too complex for its uncertain payoff** |

Three points a reader should not have to dig for:

1. **§3.1 is where the simplification comes from — but it is moderate,
   not dramatic.** An earlier draft of this section claimed planner.rs
   would lose ~565 lines (−27%). **That was wrong**, and the error is
   instructive: it counted the whole of `acquire()` (141) and `plan()`
   (151) as dying, when v2 still needs both — an acquire-equivalent
   (zero-demand path, uncontended reserve, `Impossible` checks, a wait on
   membership) and a boundary pass (collect, membership, fund, publish).
   Re-measured against the current tree:

   | | lines |
   |---|---|
   | deleted outright (`Waiter`, `WaitKind`, `Step`, `collect_outcome`, `cancel_waiter`, `WaitRegistration`, `wait_resident`) | **117** |
   | `acquire()` + `plan()` shrink (292 → ~145) | ~130 |
   | added (member set + publication) | +~50 |
   | **net in planner.rs** | **≈ −200 (~10%)** |

   planner.rs would go 2,080 → ~1,880, still ABOVE origin/dev's 1,749.
   The real prize is not line count but the **seams**: the waiter queue as
   a data structure, the park/serve/collect three-phase handshake, and the
   park → `notify_lane_close` → frame-quorum coupling. Every bug found in
   this session lived in a seam of exactly that kind.
2. **§3.2 adds code to remove a bug class.** It is not a LOC win and
   should not be sold as one. The trade is: a newtype pair, in exchange
   for making "a heuristic became load-bearing for liveness" a compile
   error instead of a code-review rule. The rule already existed as prose
   (`rainer.md` §15) and E6 violated it undetected until requests died —
   that is the evidence the type is worth its weight, not an aesthetic
   preference.
3. **§3.4 does not simplify anything, and it is the only item that moves
   the throughput number.** Its *policy* can stay knob-free — take exactly
   `deficit` pages from the victim's tail, which is strictly more
   arithmetic than today's "evict whole sets until `covered >= deficit`",
   an overshoot of up to a full working set — so it does not violate the
   no-knobs criterion. But partial fences and page-run addressing are real
   new mechanism, and §3.4's own counter-evidence (g4) says cheapening
   transfers has already backfired once. **If the goal is simpler and
   safer code, do §3.3 → §3.2 → §3.1 and stop.** §3.4 is a separate
   project with a separate risk profile and should be decided on its own.

---

## 6. Ordering

Each step is independently landable and independently benchmarked
(roomy A/B + h2h + press gates, as v2 §16 prescribes):

1. **`ReclaimQuote` accessor split (§3.3).** ✅ **DONE 2026-07-26.**
   `PageTable::held_pages()` added (durable: resident + swapped, pinned
   and shared alike); `check_hog` now calls it. `ReclaimQuote::pages()`
   **deleted** rather than merely documented — it had no other caller, so
   the misuse is now unrepresentable instead of discouraged. Guarded by
   `held_pages_is_a_durable_fact_where_a_reclaim_quote_is_not`, which
   asserts the exact divergence that caused the bug: a pin collapses the
   quote to `Nothing(Pinned)` while holdings stay at 10.
   Verified: suite green (344 lib + 43 integration), F2 10/10, throughput
   in band on 7 cases.
2. **Typed liveness/policy boundary (§3.2).** ✅ **DONE 2026-07-26.**
   `VictimSet` (all FCFS-legal victims from one snapshot) + `Victim`
   carrying `e6_fresh` as a *tag*, not a filter. `preferred()` is the
   routine path, `all_hysteresis_waived()` the last rung, `is_empty()`
   the only fact the endgame may act on. Both `plan_eviction` and
   `last_resort_evict` now build the set through one constructor, so the
   endgame taking a FRESH snapshot is structural rather than remembered.
   Verified: suite green, F2 12/12, throughput in band on 8 cases.

   *Honest accounting:* this did not shrink `planner.rs` — the file is
   1,749 → 2,043 lines (+253 code, −61 code, +102 comment across the
   whole session's planner work, much of it extraction). §5b predicted
   exactly this: steps 1–2 buy enforceability, step 3 buys the deletion.
3. **v2 boundary pass (v2 §16's four steps).** Large. Absorbs 1–2's call
   sites and removes the seam of §3.1. Justified as correctness (§1).
4. ~~Divisible residency / declared intent~~ — **REJECTED** (§3.4).
   Both parts add more mechanism than they are worth, and neither is
   clearly positive in sign. The contended 0.70–0.82x is accepted cost.

**Steps 1–3 are the project.** All three simplify or make an existing
rule enforceable, none adds a heuristic or a knob, and together they
delete more than they add (§5b).

---

## 7. Open questions

1. §3.3's hog under-count needs a runtime repro before it is called a bug
   in anything but code review.
2. Divisible residency's fence granularity is unscoped: what a partial
   working set means to the attention mask, to page-run addressing, and to
   the restore head's accumulation ledger.
3. Does declared intent (§3.4b) interact with the hog predicate? A guest
   declaring more than the pool must fail loud at declaration time, not on
   its 300th fire.
4. v2 §17's four open questions are unaffected and still open.
5. Does divisible residency (§3.4a) reproduce g4's regression? The
   mechanism is the same — cheaper eviction — and g4 lost 5% on press to
   it. This must be answered by measurement on a prototype before the
   full fence/addressing work is committed to.
6. The intermittent seal-watchdog wedge (`frame k=1 lanes=64 awaited=64
   sealed=0`, planner uninvolved) observed once and not reproducible —
   0 hangs in 5 traced runs on both the baseline and fixed builds — is
   unexplained and untouched by any of this.

---

## 8. How to actually do §3.1 (implementation plan, 2026-07-26)

Written after mapping the tree. **It corrects an earlier assessment in this
session that called step 1 infeasible.** That assessment claimed phase A
would have to be re-architected to emit an exact demand before the build.
It does not: `fire.rs:1018-1040` already computes `Demand` in a pure phase
("Phase-A demand: pure computation, holding no grant, no pins, no txn")
*before* `acquire_grant(...).await`. **v2's "declaration" already exists**;
what is distributed is not the demand, it is the decision.

### 8.1 What is already v2-shaped

Three of v2's properties are in the shipped tree and need no work:

- **Single writer already exists.** E5's `arm_drain_task` (planner.rs
  705-723) runs `plan()` on one dedicated task; `poke()` (725-734) only
  notifies it, and "the caller NEVER runs the drain itself".
- **Exact declarations already exist** (above).
- **Membership is already FCFS-by-spawn**, keyed on `proc.seq`.

So v2 is not a rewrite of the whole planner. The gap is narrower than
`rainer.md` §16 makes it sound.

### 8.2 What actually causes the bug class

The stale-snapshot defects (§18.9, §18.10) have one cause, and it is not
"many writers":

> `plan_eviction` must **release the planner lock to quote**, because
> reclaimability lives in the KV store behind a different lock.

Everything else follows. The candidate set is sampled at T0, the quotes
come back at T1, the endgame fires at T2, and a process that changed state
in between is invisible or misjudged. `last_resort_evict()` (§18.10) is a
manual re-read at T2 — a patch on the symptom.

**The cure is to stop needing the store lock for the decision**, which is
exactly what `rainer.md` §14 means by "accumulation becomes a **ledger**
number".

### 8.3 Step 1 — one atomic snapshot for the endgame ✅ DONE 2026-07-26

**The ledger route below was proposed, then rejected on contact with the
code.** `pages_freed()` (planner.rs:809) carries neither a pid nor a count
— it is a bare "something freed pages" poke from 8 sites
(`fire.rs:227,266,1818`, `kv_working_set.rs:121,159,191`,
`working_set.rs:92`, `exec.rs:241`), and at several of them the freed
count is not in hand. A ledger fed from those sites would drift, and a
drifting ledger making *liveness* decisions is worse than no ledger.

**What shipped instead achieves the same property directly.** §8.2 says the
cure is to stop needing the store lock *during* the decision. The tree's
documented lock order already permits that — `inner` is innermost — so the
endgame now takes the **store lock outside, the planner lock inside**, and
reads procs, queue and quotes at one instant:

```
kv_working_sets_for(pids)                 // RESIDENCIES, before either lock
with_kv_lock(store, |kv|                  // outer
    with_inner(|inner| {                  // inner — documented order
        head, deficit, legal victims, quotes   // ALL at one instant
    }))
commit_evictions(...)                     // after both locks release
```

- `residency::kv_reclaim_quotes` was split into `kv_working_sets_for`
  (RESIDENCIES only) + `quote_locked` (already-locked store), because
  `RESIDENCIES` is acquired before the KV lock everywhere and taking them
  the other way round would invert that order.
- Working sets are gathered before either lock; a process registered after
  that gather cannot invalidate the decision, because it starts with no
  pages and cannot acquire any while the planner lock is held.
- The rare KV-lock hold does not touch the per-fire convoy §16 measured:
  this runs only in the wedge.

`VictimSet` (§3.2) remains the routine path's membership fact; its
`is_empty`/`all_hysteresis_waived` were deleted, since the endgame now
computes the same membership rule fused with quoting inside the atomic
section, and its doc says so rather than claiming a role it no longer has.

Verified: suite green (344 lib + 43 integration), **F2 20/20**, throughput
in band on 8 cases against a 6-sample pre-band.

*What is still deferred:* the ledger would additionally let the ROUTINE
path decide without the store lock, which is what would allow §8.4's
batched funding to compute membership cheaply. If §8.4 is attempted, the
ledger returns as a prerequisite — and the `pages_freed` signature is the
work item.

### 8.3b The ledger (proposed, NOT implemented)

Give the planner authoritative per-process page accounting in its own
state, so membership and every endgame predicate are computable under one
lock with no store round-trip.

Feasible because the planner already observes every page movement through
its own port:

| event | already seen at |
|---|---|
| grant issued | `acquire` / serve path |
| unconsumed pages returned | `AllocationGrant` Drop → port |
| pages freed (fire finalize, process exit, ws release) | `pages_freed()` planner.rs:812 |
| eviction committed | `commit_evictions` |
| restore landed | `report_restored` |

Then:

- `victim_set()` (added in §3.2) reads `held > 0 && seq > head_seq &&
  Resident` from the ledger — **no quotes, no lock release**.
- `check_hog` reads the ledger instead of `held_page_count()`'s store
  round-trip (the §3.3 accessor becomes a debug cross-check).
- `check_starvation` becomes an assertion over the same snapshot.
- **`last_resort_evict()` is deleted** — with no lock release there is no
  stale snapshot to re-read.
- `ReclaimQuote` survives, demoted to what §3.2 already types it as:
  *ordering only*. A stale quote can now only pick a worse victim, never
  invent or hide one.

This is the step that pays the correctness debt, and it is self-contained
inside `planner.rs` — no fire.rs, frame.rs or worker.rs change. Gate:
lib+integration suite, F2 ×20, roomy/h2h/press.

### 8.4 Step 2 — boundary-triggered batched funding

> **Attempted and REVERTED 2026-07-26 (the funding-batch half).** Measured
> first, then removed, because it failed the only test that mattered.
>
> The cascade re-absorbs per entry, so N parked asks cost ~N port
> reservations. Pulling the whole servable FCFS prefix's need in ONE
> `reserve_device_up_to` (head-first untouched; the prefix stops at a
> restore entry, which must be sized against the store) is a ~35-line
> change. Result, 3 samples each against the step-1 numbers and the
> pre-band:
>
> | case | step-1 | batched | pre-band |
> |---|---|---|---|
> | A4 | 11,731 | 11,389-11,740 | 10,867-11,618 |
> | A6 | 8,713 | 8,080-8,360 | 7,689-8,486 |
> | E2 | 10,309 | 10,466-10,864 | 10,296-10,946 |
> | E3 | 7,659 | 7,677-7,729 | 6,724-7,902 |
>
> All inside the bands: **no gain, no loss.** Expected in hindsight —
> §18.5 located the bottleneck at eviction latency, not at port round
> trips. So the change bought +35 lines for nothing measurable and was
> reverted.
>
> **What this tells us about step 2's decomposition.** Step 2 has two
> halves and they are not equal: batching the *funding* is worthless, and
> **all of step 2's value is in computing and publishing a member set** —
> because that is what step 3's 405-line deletion depends on. A future
> attempt should skip the funding batch entirely and go straight at
> membership. The boundary trigger likewise: `poke()` already fires on
> every free, so adding a per-frame trigger changes nothing until
> membership is what the frame consumes.

#### Original sketch (kept for the record)

Only now move *when* the pass runs and *how much* it serves at once:

- attach the pass to the per-frame decision point, `frame.rs:735-767
  plan_dispatch` (the worker consumes it at `worker.rs:3482-3489`);
- replace the one-head-at-a-time serve cascade (`plan()`'s
  `Step::{Absorb, ServeAllocation, ServeRestore}` loop, planner.rs
  1120-1270) with: compute the FCFS member prefix from the ledger, then
  fund every member in ONE batched KV-lock acquisition.

**Keep `acquire()`'s signature and the fire's await point exactly where
they are.** The fire cannot build without pages, so it must block
somewhere; whether the wait is satisfied by the serve cascade or by a
batch pass is invisible to the 63 grant-consumption sites in `fire.rs`.
Not touching them is the single biggest risk reduction available.

**Keep `poke()` alive for the urgent paths in this step.** Boundary-only
triggering adds up to a frame (~7 ms) of page-supply latency; on a tree
tuned to 0.96x that is a plausible regression, so it must be measured, not
assumed. Gate: h2h ×6 + press ×2 + roomy A/B, per `rainer.md` §16.

### 8.5 Step 3 — membership push, then delete the park machinery

> **Started 2026-07-26. One deletion landed; the rest is blocked on
> membership.**
>
> **Landed — the elder bypass is gone (−23 lines).** `acquire`'s fast path
> let a process OLDER than the queue head reserve directly instead of
> parking. It was justified by "the 47% inter-batch gap measured
> 2026-07-25" — i.e. *before* the §17 mechanism fixes. Re-measured after
> them, it earns nothing:
>
> | case | without elder bypass | band |
> |---|---|---|
> | A4 | 11,370 / 11,633 | 10,867-11,740 |
> | A6 | 8,163 / 8,789 | 7,689-8,713 |
> | E3 | 7,707 / 7,737 | 6,724-7,902 |
>
> F2 stays 12/12 (elders now park, so the wedge dynamics change — this had
> to be re-verified, not assumed). Lib-suite flake rate is unchanged:
> baseline 1 failure in 10 runs, with the change 1 in 10, all in
> `scheduler::worker::tests` and all pre-existing.
> `note_ask_and_check_elder` deleted; its E6-progress side effect moved to
> an explicit `note_progress` call. v2 lists the bypass under "dies —
> derivatives of implicit membership", and it turned out to be removable
> *before* membership rather than after.
>
> **Blocked — and the sequencing is worse than §8.5's sketch says.**
> Digging in to implement it established one fact that changes the plan:
> **membership push on its own deletes ZERO of the 117 lines.** It removes
> the leave *coupling* (two call sites plus the frame.rs handlers) and
> nothing else — `Waiter`, `WaitKind`, `Step`, `collect_outcome`,
> `cancel_waiter` and `WaitRegistration` (86 lines) are the waiter queue's
> *funding protocol*, and `wait_resident` (21) is still needed by the
> residency gate. The 117 lines come only from deleting the waiter queue,
> which requires the boundary pass to fund a fire BEFORE its build — the
> fire.rs restructuring this document's §8.1 waved away.
>
> And the queue is doing real work, so it does not collapse into a
> per-process slot: `EntryKey = (spawn_seq, insertion_id)` because a
> single process can hold CONCURRENT asks from sibling pipelines
> (`residency::pipelines_of` returns a Vec;
> `scoped_leave_does_not_remove_a_sibling_pipeline_of_the_same_process`
> exists precisely for that). Folding the queue into `Proc` rebuilds the
> same map under another name.
>
> Net effect on the estimate: **the ≈ −200 lines in §5b is optimistic.**
> Membership push alone is net-ADDITIVE; the 117 lines arrive only with
> the full v2 funding model, as one indivisible change.
>
> **Everything else.** The remaining lines (waiter queue,
> park/serve/collect, `WaitRegistration`, the leave contract's
> park-driven re-posts) all depend on a published member set, because
> that is what lets the frame seal without an unfunded lane. There is no
> further net-deleting slice: every intermediate state is additive until
> membership lands. Note also that the "five re-post sites" headline
> overstates the prize — only two are park/suspend-driven
> (`planner.rs:936`, `planner.rs:1091`); `fire.rs:347` and `gate.rs:19`
> reach them indirectly and `fire.rs:1571` is a genuine graceful close
> that must stay.

#### Original sketch (kept for the record)

- Publish the member set to the frame policy so `awaited ⟺ member`,
  replacing inference from leave/arrival events. Transitions to replace:
  `frame.rs:335-339` (set true), `478-489` (graceful leave), `513-520`
  (process suspend).
- That retires the five re-post sites: `planner.rs:936-948` (acquire
  park), `planner.rs:1091-1107` (`wait_resident`), `fire.rs:347-357`
  (`settle_and_wait_resident`), `fire.rs:1571-1590` (`pipeline_close`),
  `gate.rs:19-38` (`residency_gate`).
- Then the waiter queue, park/serve/collect and `WaitRegistration` can go:
  an unfunded fire is simply not a member, and the seal does not wait for
  it.

Publish whole-set per boundary, never a delta stream (`rainer.md` §17.2).

### 8.6 Step 4 — demote lease/fence to assertions

Once eviction never targets a member of the executing or next frame,
frame structure *is* the quiescence and `ws.fire_lease()` (fire.rs:1043)
becomes a debug assertion.

### 8.7 Why this order differs from `rainer.md` §16

§16 bundles the ledger and boundary-batched grants into one step. Splitting
them matters: **the ledger alone closes the bug class** (§8.2) and is
planner-local, while the boundary trigger is the part that carries
throughput risk. Landing them together would make a correctness fix
hostage to a performance gate. Land §8.3 first, on its own.

### 8.8 Honest risks

1. §8.4 is where a regression would appear (page-supply latency). It is
   also the only step whose value is performance-shaped, and §4.1 already
   says v2's throughput claim was not reproduced — so if it regresses,
   **stopping after §8.3 is a legitimate end state**, not a failure.
2. The ledger must not drift from the store's truth. Keep `held_pages()`
   (§3.3) as a `debug_assert` cross-check for a release cycle.
3. Demand can drift while a fire awaits (fire.rs:188 comments on exactly
   this); the re-declare loop already handles it and must be preserved.
