# Beam page lists: `Port::Parents` (spec)

Status: agreed (GPU, Mac, Leader, ROCm), 2026-11-19; reviewed edits folded in.
Hybrid models (§11) and a search's finish (§6.6): agreed (GPU, Leader), 2026-11-20. The single written spec the
device kernels (engine-cuda CUDA/HIP, engine-gpu Vulkan/wgpu, engine-metal)
and the host replay all follow. Where an implementation and this text
disagree, this text is right and the implementation is the bug.

## 1. What it is

A beam search of `B` beams runs `B` lanes of one decode pass over **one
working set** (the *group's* set). The prompt is folded into the set; the
pages of the set's reserve past its frozen full pages are a *pool* from
which each beam holds its own *page list*: the pages its own generated
slots lie in. A beam's rows read the frozen prompt and its own pages only
(maskless, a read extent of its own length), and write only its own pages.

The guest states one thing per fire: each beam's **parent** (which beam of
the previous fire it continues). The engine keeps every list and the free
pages as resident state of the instance, and at the head of each fire
reorders the lists by the parents, copies the tail pages forks need, and
resolves the fire's page tables over the lists. Nothing of it is guest
memory, and nothing waits on the host.

On a hybrid model (recurrent layers beside full attention, §11) the same
lists page each beam's buffered recurrent rows as well: the recurrent state
stays the set's one shared state, committed at `folded` and never written
during the search, and a beam's generated rows are its own rows of the set's
rs buffer, on the same pool pages as its kv slots.

## 2. Guest surface

- WIT `forward.wit` `state-geometry.parents: option<borrow<channel>>` —
  `[lanes]` u32. eta-ir `Port::Parents`. Every lane of the pass names the
  same working set; lane `b` is beam `b`.
- The lanes' `slots` and `positions` stay host-derivable (a decode
  envelope): every beam writes the same slot `s_t` at fire `t` (its depth)
  and ropes at the same position. No `kv-len`, no `mask`.
- The set reserves its first column's pool with `working-set.reserve(end)`
  before the first decode fire, and the runtime grows the reservation a
  column at a time (§3).
- The parents channel is written by the guest's epilogue and read by the
  next fire (loop-carried, device only: `HostRole::None`). Its initial value
  is ignored (the first fire primes, §5). Engines never read the parents
  cell on the host: the head kernel reads it at the staging cursor's head,
  so an engine that resolves envelope ports on the host excludes `Parents`
  from them.
- SDK: `inferlet::beam::BeamGroup::new(&ws, beams, folded, steps)` reserves
  the pool; `group.geometry()` binds `positions`, `slots`, `parents`;
  `group.reorder(&parent)` in the epilogue puts the parents and advances
  `slots` / `positions`.
- An engine whose capabilities lack `Port::Parents` refuses the bind by name
  (`HostGeometry::EngineLacks`). A pass binding `parents` whose lanes name
  more than one working set, or that also binds `kv-len`, `mask` or `fold`,
  is refused by name.

## 3. Numbers

- `S`: the engine's kv page size (tokens). `folded`: the set's frozen slots
  when the pass first fires. `f0 = folded / S` (frozen **full** pages).
  `tail = folded % S`: the prompt's slots on page `f0`.
- Pool page `q` (`0 <= q < Q`) is the set's page `f0 + q`. `Q = reserved
  pages - f0` (`reserved` from `reserve`). Page 0 is the prompt's own last
  page when `tail > 0`: its first `tail` slots are frozen.
- `s_t`: the slot every beam writes at fire `t` (`s_0 = folded`). Column
  `c_t = s_t / S - f0`, offset `o_t = s_t % S`. Entry `j` of a list covers
  slots `[(f0 + j) S, (f0 + j + 1) S)`.
- `P` (list width): columns a list may hold, `P = ceil(reserved / S) - f0`
  bounds it; an engine may size by its context pages (`pages_per_slot`).
- A pool page `q` is the set's page `f0 + q` of every paged plane the set
  holds: its kv pages and, on a hybrid (§11), its rs buffer pages, one
  reservation for both. `Q_t` below counts pool pages, each priced at the
  bytes of all its planes.
- `F`: the fires a freed page waits before it is handed out again (§6.3).
  The engine states its `F` at bind (its frames in flight); the guest sizes
  with `run_ahead_window()` (the fires its pipeline holds in flight), which
  is never less, and never with a constant.
- The pool grows a column at a time (incremental reservation): the guest
  (`BeamGroup`) reserves what the first column needs, `Q_0 = 2B + B F + 1`
  pages past `f0`, and before each fire the runtime grows the set's
  reservation to what that fire's column needs, `Q_t = B (c_t + 2) + B F + 1`
  (each column's pages, a beam's each at most; the most one fire hands out;
  the pages pending across `F` fires; the frozen prompt page), waiting like
  any reservation, and backs exactly that; a fire never opens a column it
  cannot back. The step hands out the lowest free pages, so every page it
  names or holds pending lies below `Q_t`. A group whose length could never
  fit alone is refused by name. A fire that finds fewer free pages than it
  needs faults the lanes that needed one (`RESERVE_EXHAUSTED`).

## 4. Resident state (per instance)

    map[B][P]      u32   pool page of each beam's column (columns > c_t: unspecified)
    free[Q]        bit   page may be handed out
    pending[F][Q]  bit   a ring: slot `t mod F` holds the pages fire `t` freed
    fired          u32   fires run; 0 until the first fire primes
    pool           u32   the pool's pages at the last fire

Stated in `engine::GuestState::resident`, part of the instance's bytes at
its bind extents (bind to unbind; `scratch` and `scratch_arms` unchanged):
`B*P*4 + (F + 1)*ceil(Q/32)*4 + 16` bytes, by the engine's bounds on `P`
and `Q` (an engine may hold a word where a bit is stated, and may keep two
copies of `map` to reorder in place). It is one copy, device-owned, changed
only by head kernels in queue order: never part of what an engine
double-buffers for fires in flight, and never touched by staging a later
fire on the host. It is pool-relative: it never names a physical page, so
the runtime may move, suspend and restore the set's pages and the
instance's resident bytes freely (§8).

## 5. Priming (the instance's first fire, `primed == 0`)

    for b in 0..B: map[b][0] = b                (columns 1.. unspecified)
    free[q]       = (q >= B) && !(q == 0 && tail > 0)
    pending[k][q] = 0
    copies: for b in 1..B, if tail > 0 and o_0 > 0: copy(src = 0, dst = b, tokens = tail)
    primed = 1

The parents are not read. `c_0` must be 0 (the first write lies on page
`f0`); otherwise every lane faults `SHORT`. Beam 0 keeps page 0 (the
prompt's own when `tail > 0`), every other beam gets the prompt's tail
slots copied into its first page. With `tail == 0`, page 0 is an ordinary
pool page, and beams take pages `0 .. B-1`.

## 6. Each later fire `t >= 1` (in this order, before the fire appends or reads)

Inputs: the state after fire `t-1`, `parent[B]` (the cell the epilogue of
fire `t-1` wrote), `s_t` (host-derived; `c = c_t`, `o = o_t`, `cp = c_{t-1}`),
and the pool's pages now `Q`. First, the pages that joined the pool since
the last fire are free: `free[q] = !(q == 0 && tail > 0)` for
`pool <= q < Q`; the step ends with `pool = Q`.

**6.1 Check.** For each `b`: `parent[b] < B`, else lane `b` faults
`PARENT`: it continues itself (`parent[b] := b`, keeping `map[b]`), so its
pages stay named and none is freed under it.

**6.2 Reorder.**

    inh[b]   = map[parent[b]]                                   (whole row)
    first[b] = no b' < b with parent[b'] == parent[b]
    need[b]  = (o == 0) || !first[b]
    rank[b]  = #{ b' < b : need[b'] }
    hand[k]  = the k-th lowest q with free[q] (k = 0, 1, ...)
    fresh[b] = hand[rank[b]]          (fewer free than needed: lanes past them fault RESERVE_EXHAUSTED)
    map'[b]  = inh[b] with map'[b][c] = fresh[b] where need[b]
    copy[b]  = (src = inh[b][c], dst = fresh[b], tokens = o)  where need[b] && o > 0, else none

  The first child of a parent keeps the parent's pages, its tail included;
  every other child (and every beam at a page boundary, `o == 0`) takes a
  fresh page at column `c`. A child mid-page has its parent's `o` tail slots
  copied into its fresh page.

**6.3 Free.** A page lies in one column of every list that names it.

    k        = t mod F
    freed[q] = exists (p, j), j <= cp, map[p][j] == q, and no b with map'[b][j] == q
    taken[q] = exists b, need[b] && fresh[b] == q
    free'    = (free & !taken) | pending[k]      (the pages fire t - F freed)
    pending'[k] = freed                          (other slots unchanged)
    (when tail > 0: free'[0] = 0 and page 0 is never freed)

  A page freed by fire `t` is handed out no earlier than fire `t + F + 1`'s
  reorder, whatever the engine's queue model.

**6.4 Copy.** For each `b` with a copy: copy the first `tokens` slots of the
set's page `f0 + src` to page `f0 + dst`, in every plane a page holds (keys
and values of every full-context layer, and on a hybrid every rs buffer
plane of every recurrent layer: its conv input, `[b|a]`, keys and `u` rows,
§11), before any lane of the fire appends or reads. Checks (else lane `b`
faults `COPY_BOUNDS` and nothing is copied): `src < Q`, `dst < Q`, `tokens <= S`, `(f0 + dst) S >= folded`.

**6.5 Resolve.** Over the page tables the fire stages for each lane as for
any envelope lane (its slots `[0, s_t + 1)`): entry `f0 + j` of lane `b`'s
list becomes the set's page `f0 + map'[b][j]` for `j <= c`, resolved
through the set's **whole reserved translation** (its pages `f0 .. f0 +
Q - 1`, which the host stages each fire as one per-group input: host-known,
no wait), not through the lane's own staged list (`map'` may name any pool
page), and its row's
write page becomes the set's page `f0 + map'[b][c]` (write offset `o`
unchanged). Checks (else lane `b` faults and keeps the staged words, which
name its own set's pages): `map'[b][j] < Q` (`PAST_RESERVE`), `c < P`
(`SHORT`), and the absolute write slot `(f0 + map'[b][c]) S + o >= folded`
and `< reserved` (`BELOW_FLOOR`).

On a hybrid it also resolves each lane's **rs page table** the same way:
entry `f0 + j` becomes the set's rs buffer page `f0 + map'[b][j]` for
`j <= c`, through the set's whole reserved rs translation (staged with the kv
one), and the row's rs write page `f0 + map'[b][c]`; the checks are the kv
table's, one fault a lane for both.

Commit the state (`map = map'`, `free = free'`, `pending = pending'`) for
every lane, faulted or not. A faulted lane fails its fire by name
(`Overreach: lane b's page list ...`); the others proceed.

**6.6 Finish.** A search ends when the guest names its chosen beam `b*`
(`group.finish(b*)`) after its last fire lands: the runtime reads
`map[b*][0..=c]` once (the search's one host read, off the per-fire path)
and re-lists the set, its page `f0 + j` becoming pool page `map[b*][j]` of
every plane for `j <= c`; every other pool page is released. The set then
holds the chosen hypothesis at slots `[folded, s_T + 1)` in its own order,
and later passes bind it as any set (no `parents`).

## 7. Faults (one word a lane, read at settle)

| code | name              | meaning |
|------|-------------------|---------|
| 1    | PAST_RESERVE      | an entry names a page past the pool |
| 2    | SHORT             | the lane reads past its list's width (or the first write is not on page `f0`) |
| 3    | BELOW_FLOOR       | a row would land below the set's frozen slots or past its reserve |
| 4    | COPY_BOUNDS       | a copy past the pool, longer than a page, or into frozen slots |
| 5    | PARENT            | `parent[b] >= B` |
| 6    | RESERVE_EXHAUSTED | fewer free pages than the fire hands out: unreachable with the exact sizing of §3, so a bug, logged loudly; it still faults only that lane |

The fault words are a `[B]` array per group the head kernel writes and the
settle reads, **one set per fire in flight**: the next fire's head kernel
may run after this fire completes but before its settle reads them.
Nothing outside the group's own working set is read or written in any
case; the staged words a faulted lane keeps are its own set's pages.

## 8. Ordering, suspend, restore

- Stream order: the reorder + copy + resolve run in the fire's own stream
  ahead of its first append or read (one launch, or launches separated by
  the backend's barrier), never behind a host wait. The parents cell they
  read was written by the previous fire's epilogue, earlier in the stream.
- The copy-landing rule holds by construction: a copy's source page is
  named by `map'` (the first child's), so it is not freed by this fire, and
  a freed page is handed out a fire later (§6.3).
- Suspend moves the set's pages and the instance's resident bytes only
  after the fires in flight have landed (their copies with them). Restore
  re-resolves pool pages to physical pages at the next fire: the state names
  pool pages only, so a restored search continues identically.

## 9. Host replay and its test

The host replay is §5–6 as a pure function, in Rust:
`step(state, parents, s_t) -> (state', copies, lists, faults)`. It is not on
the serving path. The test, on every backend that implements the port:
widths 2, 4, 8; at least `3S` steps from a `folded` with `tail > 0` and one
with `tail == 0`; parents drawn with forks (repeated parents) and
retirements (beams no child continues). After every step the device's
`map` (columns `<= c`), `free`, `pending`, the copies it made and the
resolved table entries equal the replay's; and a run suspended and restored
mid-search continues with beams identical to an unsuspended run.

A hybrid case runs beside it: qwen35-tiny at widths 2, 4, 8 from a `folded`
with `tail > 0` and one with `tail == 0`, the device's rs page tables and rs
copies equal to the replay's after every step, and the chosen beam's logits
after `finish` and its fold equal to an unbranched decode of the same tokens.

## 10. Costs (MI300X, Qwen3-0.6B)

- Copy: one launch per fire for all of its copies, 2.3–4.1 µs (1–3 copies of
  1–15 slots, 0.34–5.2 MB); the reorder and resolve add `B * P` work in the
  same launch.
- Attention keys per beam row: its own length (`s_t + 1`), against
  `folded + B t` with the earlier shared pool and ancestry mask.
- Memory: live pages at most `B (c_t + 1)` plus `2B + 1`, against
  `folded + B T` slots held to the end before.

## 11. Hybrid models (recurrent layers beside full attention)

- State and rows: a recurrent layer keeps the set's state `h`, committed at
  `folded` (the prompt), and its buffered rows: per generated slot its conv
  input, `[b|a]`, keys and `u`, the rows a step folds into `h`. On a parents
  pass `h` is read-only: every beam's step reads `h` and its own buffered
  rows `[folded, s_t]` through its rs page table (§6.5), in one kernel over
  the state and the rows, and writes its row `s_t` to its rs write page.
  Nothing writes `h` while the pass is bound.
- Pages: the rs buffer is paged like the kv slots, page size `S`: beam `b`'s
  slot `s` lies on rs page `map[b][s / S - f0]` (`f0`, `map` as §3–4; one
  `map` for both). The prompt's tail rows (`tail > 0`) on pool page 0 are
  committed into `h`, so no beam reads them as buffered rows: priming's
  copies (§5) need cover only the kv planes, and the rs rows of page 0's
  first `tail` slots are never read.
- No fold: a pass binding `parents` binds no `fold` (§2), and the recurrent
  state is not folded by it. The chosen beam's rows fold into `h` after
  `finish` (§6.6) through an ordinary fire that folds the set's rows
  `[folded, s_T + 1)`, as any pass that commits buffered rows does.
- Pricing: one rule for both planes. A pool page is one reservation of its
  kv and rs bytes (§3), priced in the ledger as part of the set's working
  set, and the pool grows a column at a time through it, as §3 grows the kv
  pool. At bind the runtime checks that the search's full depth (`steps`:
  the `Q_t` of its last column, both planes) could fit alone, and refuses
  the group by name if it never could. A group that cannot grow a column
  mid-search waits like any reservation (served slower), never fails
  partway and never silently.
- Engines: the state + buffer decode kernel reads a lane's rows through its
  rs page table when the pass binds `parents`, and through the set's own
  rows otherwise; the head step (§6) resolves both tables and makes both
  planes' copies in its one launch.
