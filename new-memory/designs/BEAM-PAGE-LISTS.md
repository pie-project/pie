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
- `end`: the slot past the search's last row (`folded` plus its fires), the
  depth the guest states at the bind (`state-geometry.parents-end`, required
  with `parents`, refused by name without it; the SDK's `BeamGroup::geometry`
  states `folded + steps`). It reaches the engine as the bind's `key_len`
  (the most keys a lane reads).
- Readout rows: one figure, the instance's. A group's instance reads its `B`
  beams' rows (the bind's per-instance readout rows, `b` for the group), one
  a lane; every engine checks its readout seats against that figure.
- `P` (list width): columns a list may hold, `ceil(end / S)` pages a beam,
  laid for the bind's depth, never the context's. A search whose rows pass
  `end` faults `SHORT` by name (§6.5).
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

It is the instance's **resident state** (`engine::GuestState::resident`),
sized by the engine's `guest_state(program, extents)` from the bind's
`key_len` (`end`, §3), never in the per-program instance item (one program
binds searches of many depths): `B*P*4 + (F + 1)*ceil(Q/32)*4 + 16` bytes, by
the engine's bounds on `P` and `Q` (`Q <= B P + B (F + 1) + 1`; an engine may
hold a word where a bit is stated, and may keep two copies of `map` to
reorder in place). The runtime takes it at the bind as a live part of the
instance: one run of adjacent `Instance` slots, at most a chunk (past one the
bind is refused by name), priced in the ledger at the bind like any guest
slot and named to the engine as `InstanceBinding::resident`. It is not the
guest's scratch, which is dead between fires and given back on a park. A
park copies the resident run wherever the instance's tables go (the host,
else the disk tier's slot file) and the unpark copies it back into fresh
device slots, named in `Rebind::Instance::resident`; the engine clears it
once, at the bind, follows it to wherever a rebind names, and clears nothing
there. So it stands from the bind to the search's finish. engine-cuda lays
the group's state at the start of that run; engine-gpu, Metal and XLA state
the same bytes as resident and read them where the binding names. It is one copy, device-owned, changed
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
set's page `f0 + src` to page `f0 + dst`, in every kv plane a page holds
(keys and values of every full-context layer), before any lane of the fire
appends or reads. Checks (else lane `b` faults `COPY_BOUNDS` and nothing is
copied): `src < Q`, `dst < Q`, `tokens <= S`, `(f0 + dst) S >= folded`. On a
hybrid the same copy moves the beam's rs rows too, slot by slot (§11.2).

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

On a hybrid it also resolves each lane's **rs page table**, slot by slot
(§11.1), one fault word a lane for both tables.

Commit the state (`map = map'`, `free = free'`, `pending = pending'`) for
every lane, faulted or not. A faulted lane fails its fire by name
(`Overreach: lane b's page list ...`); the others proceed.

**6.6 Finish.** A search ends when the guest names its chosen beam `b*`
(`forward-pass.adopt-beam(b*, end)`, `group.finish`) after its last fire
lands: the runtime reads `map[b*][0..=c]` once (the search's one host read,
off the per-fire path, behind the pass's fires on the engine's lane) and
re-lists the set, its page `f0 + j` becoming pool page `map[b*][j]` of every
plane for `j <= c` (on a hybrid, its rs entries slot by slot through
`σ_{b*}`, §11); every other pool page is let go once the fires that may read
it retire. Nothing is copied. The set then holds the chosen hypothesis at
slots `[folded, end)` in its own order, and later passes bind it as any set
(no `parents`; a hybrid folds the rows by an ordinary pass that folds them).

`map` is the last fire's: its beams are those the last fire read, before its
epilogue picked the next parents. A guest whose final pick is beam `l` of
that epilogue (its hypothesis the parent's plus one token no fire fed)
adopts `b* = parent[l]`, whose pages hold every row of that hypothesis.
`end = folded + fires` (each fire writes one slot a beam).

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
resolved table entries equal the replay's.

Parked mid-search: the runtime's roles take the resident run live (the
roles test: with a host tier and with none, a parked instance's resident run
is copied out whole, copied back from where it went into a fresh run that is
neither its old one nor another instance's, the engine told that run, and
nothing of the scratch copied). Served, the `test-pause` worker policy
pauses a process from `pause-at-ms` after its arrival: beam `--verify` at
width 4 over 256 steps, paused inside the search (MI300X Qwen3-0.6B: +200,
+300 and +450 ms; one process and four at once; a host tier, and a pool
with none whose disk tier holds it) finishes verified, its resident run back
in other slots each time. The same pause with the state in the scratch, or
with its copy left out, faults the next fire by name (`Overreach`).

A hybrid case runs beside it: widths 2, 4, 8 from a `folded` with
`tail > 0` and one with `tail == 0`, rs pages of `T` = 1, 2 and 4 rows from
a buffer base off a kv page boundary (as a frozen slot of 38 at `T` = 2,
`S` = 16); after every step each beam's resolved rs entries equal §11.1's
over the replay's `map'`. The copy is checked alone: a fork whose copy ends
mid rs page, one from the prompt's own page (its tail's slots below the
buffer skipped), and one out of bounds that faults and moves nothing; every
destination row equals its source row and every other byte stays. Served
(qwen35 at widths 2, 4, 8): beam `--verify` passes, and the chosen beam's
logits after `finish` and its fold equal an unbranched decode of the same
tokens.

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
- Rows by slot: the set's rs buffer is paged as the engine's chunk recipe
  pages it, `T` rows a page from slot `base` (the slot its buffer starts at,
  `<= folded`; Qwen3.5-0.8B: `T` = 2 from the frozen slot), not as the kv
  slots. One `map` serves both: beam `b`'s buffered row of slot `s`
  (`s >= f0 S`) is the set's row of slot
  `σ_b(s) = (f0 + map[b][s / S - f0]) S + s mod S`, that is the slot at the
  same offset of the pool page its list names for `s`'s column. The set's
  row of slot `σ` lies on its rs entry `(σ - base) / T`, row `(σ - base) mod
  T`. Slots below `base` (the prompt's tail on pool page 0) are in `h`: no
  beam reads them as buffered rows, and no copy moves them.
- Divisibility: `T` divides `S`, and `T` divides `base` (so no entry
  straddles two of a beam's pages). An engine refuses a `parents` bind on a
  hybrid whose recipe has `T ∤ S` by name with `T` and `S`, before any beam
  fires; a buffer base off a row page is refused by name at the fire. A
  recipe that cannot meet it chooses its `T` to divide `S`: there is no
  fallback.
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
  rows otherwise; the head step (§6) resolves both tables, and the copies
  (§6.4, §11.2) run after it, ahead of the fire's first append or read.
- Staging: a lane's rs page table holds the entries its replayed rows and
  its own rows touch this fire (`[base, s_t + 1)`); the set's whole rs
  translation (every entry the pool's reservation backs) is staged once per
  group for the resolve and the copies.

### 11.1 Resolve (rs)

Inputs per beam `b`: its rs table's entries `pages[from ..]` (as the host
staged them: the set's), `entries = (s_t - base) / T + 1`, `base`, `T`, the
set's rs entries' addresses `set[0 .. E)`, and `map'[b]` (§6.2). For each
`e < entries`, `slot = base + e T`:

    if slot < f0 S: leave pages[from + e]          (a frozen page: the set's)
    j = slot / S - f0                               (j >= cap: SHORT)
    q = map'[b][j]                                  (q >= the pool's staged: PAST_RESERVE / EXHAUSTED as §6.5)
    σ = (f0 + q) S + slot mod S
    a = (σ - base) / T                              (a >= E: PAST_RESERVE)
    pages[from + e] = set[a]

A faulted entry keeps its staged word. The row's write entry is among them
(`e = (s_t - base) / T`), so the beam's row lands on its own page.

### 11.2 Copy (rs): `copy_rs_rows`

One run per beam with a copy, over every rs plane (a block per (run,
plane), a plane's row per step). Inputs: the copy `[src, dst, tokens]` the
step left (§6.4), the pool's staged pages `Q`, `S`, `T`, `base`, `f0`,
`floor_rel = folded - f0 S`, the set's rs entries `set[0 .. E)`, and the
planes: each `(offset, cell)`, the byte offset of its rows in an rs page and
the bytes one row takes (row `r` of a plane at `set[a] + offset + r cell`).

    if tokens == 0 or src == dst: nothing
    if src >= Q or dst >= Q or tokens > S or dst S < floor_rel:
        fault COPY_BOUNDS, copy nothing
    for k in 0 .. tokens:
        from = (f0 + src) S + k;  to = (f0 + dst) S + k
        if from < base or to < base: next           (in the state, no row)
        fe = (from - base) / T;  te = (to - base) / T
        if fe >= E or te >= E: fault COPY_BOUNDS, stop
        copy plane.cell bytes from set[fe] + offset + ((from - base) mod T) cell
                           to   set[te] + offset + ((to - base) mod T) cell

It runs after the head step and its kv copies, in the fire's stream, ahead
of the fire's first append or read; its fault word is the lane's (§7).

### 11.3 The common-ancestor fold

A group's rows below its common ancestor are the same in every live beam, so
they fold into the set's state once, and each beam carries only the rows after
it: with one beam every row is common, and beam-1 carries the natural window a
plain decode does.

- Definition. After fire `t`'s head step has reordered the lists (`map'`,
  §6.2), let `j*` be the first column at which the live beams' lists name
  different pool pages (`cap` where every column agrees), and `end_t` the slot
  past the fire's rows. The group's common slot is
  `c_t = min((f0 + j*) S, end_t)`. Every slot below it lies, for every beam,
  on the same pool page: one row, never a copy that could differ (a slot a
  fork copied is not counted: whole shared pages only, at most `S - 1` rows
  more carried). `c_t` never decreases: a child keeps its parent's columns
  below `j*`, and `end_t` only grows.
- Computed on the device, where the lists live: the head step writes `c_t` to
  the group's state header (`header[3]`, beside `fired`). The host, which
  does not know the lists under `Port::Parents`, reads it when it harvests the
  fire, through the readback its outputs land by. Monotone, a value read late
  (the lag `W`) is still common: nothing waits on it.
- Stated to engines as a fold. The group's set keeps one committed length,
  `folded_g`. The next fire the runtime stages on the group states
  `fold = c - folded_g`, `c` the largest `c_t` harvested, by the verb every
  decode folds by (`RsVerb::Buffer { folded, fold: FoldLen::Host(n) }`):
  host-known, run-ahead safe. It folds the set's state `h`, which every lane of
  the group reads, once a group and never once a lane: the engine folds rows
  `[folded_g, folded_g + n)` into `h` before any lane of that fire reads it,
  reading them through any beam's rs page table (below `c_t` every beam's
  `sigma` names the same row, §11.1), by the rank-1 updates a decode's
  committed rows take from the kept planes (gated delta `S <- g S + k u^T`,
  Mamba-2 `S <- a S + (dt x) B^T`, KDA `M <- M diag(alpha) + u k^T`). An engine
  may run it as one fold over the group's recurrent layers at the fire's
  start, or inside each layer's step, so long as every lane's read of `h`
  follows it. After it `folded_g = c`, each beam lane replays `[c, s_t]`, and
  lanes still commit nothing (`commit_len` 0): `h` changes only by the group's
  fold.
- Pages below the fold point. kv: unchanged (attention reads every slot, and
  the pages below `c` are already in every list; the lists, `f0`, `free` and
  `pending` do not move). rs: no beam's replay reaches below `folded_g`, so the
  set's rs buffer base moves to `floor(folded_g / T) T` and the rs pages wholly
  below it go back to the rs store, the truncate from below a set's committed
  length advancing takes. `sigma` (§11.1) and `copy_rs_rows` (§11.2) apply at
  slots at or past the base; a copy's slot below it is skipped, as a prompt
  tail's is.
- A search owns its set's state. A parents pass used to leave the set's state
  as the search found it; it now advances it to the common ancestor. §6.6's
  adopt is unchanged (the chosen beam's list holds every common page, and the
  set's committed length is already `folded_g`: the relist covers kv whole and
  rs from the base). A retract of the set below `folded_g` is refused by name,
  as any past a fold is. A search left without adopting a beam leaves `h` at
  `folded_g`, not the prompt: a guest that wants the prompt state back forks
  the set before the search (`forward.wit`, `state-geometry.parents`). The
  common rows are in every hypothesis, so whichever beam finishes adopts them:
  accepted history, never a speculative tail.
- Tests, beside §9. `c_t` against the host replay's lists after every step
  (forks, retirements, one beam); a search with folds verifies as one without
  (`--verify` and the finish check, widths 1, 2, 4, 8, on a hybrid); the rs
  pages below the base are given back; beam-1-parents' step cost and carried
  rows equal a plain decode's at the same lag, and beam-4's step lies within
  the user's rule of greedy's.
