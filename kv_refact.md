# KV / Memory Refactor Sketch

## Goals

- Preserve inferlet expressiveness for beam search, constrained decoding, Quest,
	MCTS, speculative decoding, and custom attention layouts.
- Avoid sending a full page-id vector on every fire; contexts may contain tens
	of thousands of pages.
- Make logical discard, transparent residency management, prefix deduplication,
	and disaggregated prefill first-class.
- Replace the current KV-page-sized unified `Arena` with resource-specific
	stores and typed static backing pools initially, with a path to dynamic
	backing (MemoryBroker, CUDA VMM / Metal sparse mapping) as a later
	evolution.

## Core Model

```text
inferlet / PTIR
		|
		| logical ranges, masks, cursors, cache recipes
		v
WorkingSet ---------------------- CacheFabric
		|                                |
		| logical mappings               | local hit / remote hit /
		|                                | remote compute / local compute
		v                                v
typed stores <---------------- cache bundles
		|
		v
MemoryBroker
		|
		v
driver backing (CUDA VMM / Metal sparse mapping)
```

### WorkingSet

A WorkingSet is independent of any Pipeline. It is:

- a logical address space;
- a migration and memory-accounting domain;
- a handle to one terminal in a persistent KV mapping trie;
- a holder of optional recurrent-state handles.

It is not a physical allocator; its flattened physical-id table (below) is a
derived cache of the trie mapping, not authoritative state. A WorkingSet
may be used by multiple Pipelines. `KvPageTable` maps its id to a terminal node.
Launches bind that persistent mapping while PTIR supplies WorkingSet-relative
page indexes rather than physical page vectors.

```text
WorkingSetId -> TerminalNode
WorkingSetPageIndex -> node page runs -> PhysicalKvPageId
PhysicalKvPageId -> pool_base + page_size * id
```

The trie is radix-compressed: each node adds a run of pages rather than one node
per page. WorkingSets with the same semantic prefix share the same root and
prefix nodes and point at different terminals only after they diverge. A node
may be terminal for one WorkingSet while remaining an internal ancestor for
others.

```text
Root P [P0, P1, P2] <- WS C
       /             \
      v               v
Tail A [P3] <- WS A   Tail B [P4] <- WS B
```

Root and terminal are roles rather than exclusive node kinds. A root is any node
without a parent; each WorkingSet points to the node where its current mapping
ends. A terminal may still have children for longer WorkingSets. Local cache
matching extends a WorkingSet from its own captured terminal and follows matching
child runs; discovering a prefix owned by an unrelated WorkingSet is a
`CacheFabric` lookup by page hash, not a local trie scan. The trie itself
canonicalizes shared prefixes. *(Amended in implementation: an EMPTY WorkingSet
has no terminal to extend from and roots are implicit, so the store keeps one
bounded local boundary index — canonical full page's boundary chain value `->`
trie location, validated on lookup and lazily pruned — for same-store empty-WS
prefill adoption. Cross-store discovery remains `CacheFabric`.)*

Each WorkingSet keeps a flattened physical-id table: a pinned, address-stable,
device-visible buffer holding `WorkingSetPageIndex -> PhysicalKvPageId` for its
current mapping. It is not a growable host vector; reallocation would move the
address under in-flight kernels. The runtime and drivers share one process, so
kernels read this table directly; a fire binds the WorkingSet's current table
version and does not upload a page vector. Appends write new tail entries in
place. Discard suffix rewrites and CoW entry patches publish a new table
version; a fire binds the version current at prepare time, and old versions are
recycled after their completion epochs retire. The table is derived host-owned
state, not authoritative page-table state; a driver may additionally snapshot a
`(WorkingSet, version)` copy into device memory as an internal optimization
with no ABI change.

### Pipeline

A Pipeline is an ordering domain, not the owner of a WorkingSet. Any
device-visible WorkingSet operation takes `borrow<pipeline>` and is ordered:

```text
prior command on that Pipeline
		-> WorkingSet operation
		-> later command on that Pipeline
```

Different Pipelines have no promised relative order beyond what the single
per-driver sequencer queue provides: all commands, across all Pipelines, enter
one queue in program order and are linearized there. Creating a WorkingSet on
one Pipeline and firing it from another is therefore safe by construction, and
there is no hazard tracker. The batcher applies one rule: two write-intent
fires binding the same KV WorkingSet are never merged into one batch. Beyond
that, an inferlet that constructs conflicting overlapping accesses across its
own Pipelines gets undefined values in its own outputs only; runtime metadata
integrity and cross-inferlet isolation are unconditional, because CoW
decisions are made by the runtime and slot reclamation is completion-epoch
gated.

Pure logical reservation needs no Pipeline. Physical or mapping mutations do.

```wit
resource kv-working-set {
		constructor();

		reserve: func(pages: u64) -> result<page-range, error>;

		discard: func(
				on: borrow<pipeline>,
				ranges: list<page-range>,
		) -> result<_, error>;

		fork: func(
				on: borrow<pipeline>,
		) -> result<kv-working-set, error>;

		slice: func(
				on: borrow<pipeline>,
				range: page-range,
		) -> result<kv-working-set, error>;
}
```

Operations involving two WorkingSets, such as KV copy, remain Pipeline methods.
Today's WIT dependency is one-way (`forward.wit` uses `working-set.wit`); the
`borrow<pipeline>` parameters above would introduce the reverse edge. Hoist the
`pipeline` resource into its own WIT interface and let both `forward.wit` and
`working-set.wit` use it.

## PTIR Owns Attention Geometry

Do not add a public WorkingSet View resource. Trie nodes have internal ordered
page runs used to express structural range/sparse sharing for `slice` and
`discard`, but PTIR still owns the complete per-forward attention view:

- `Pages`, `PageIndptr`, and `KvLen` describe ragged logical page selection;
- `WSlot` and `WOff` describe logical write targets;
- `AttnMask` describes token/cell visibility;
- the layer-scoped `attn_page_mask` sink describes Quest-style dynamic page
	selection.

These values remain in persistent device channels and stage programs, so the
inferlet does not upload a large page vector on every fire. PTIR values and
device-produced geometry are WorkingSet-relative logical indexes end to end;
the single logical-to-physical translation point is the kernel's lookup into
the bound WorkingSet's shared flattened table. Device stages may select,
permute, and mask logical indexes but never mint physical ids; new entries
enter the table only through host mapping publication.

**Implemented composed batch (2026-07).** "Device-geometry" is ONE predicate
shared verbatim by the runtime (`detect_device_geometry`) and the driver
(`is_device_geometry_trace`): `WSlot`/`WOff` write descriptors plus a
channel-bound `[B, P>1]` `Pages` port. A merely non-const port (a host-put
`KvLen`) does not make a pass device-geometry — its geometry is host-known
and rides the wire. A batch may carry any mix: the launch ships a program →
wire-request attribution CSR (`ptir_program_row_indptr`; a device-geometry
fire's row is an empty placeholder), the driver resolves EVERY
device-geometry program's channels (translated per program through its
`kv_translation` segment) and composes wire rows first, resolved programs
after (`batch_compose.hpp`), synthesizing standard-append `WSlot`/`WOff`
targets for wire rows when any program in the batch carries explicit write
descriptors. The post-forward dispatch receives per-PROGRAM gathered-logits
offsets, not the per-request sampling CSR. v1 mask scope, enforced by the
scheduler and defended in the driver: dense device masks (`AttnMask`
channel) batch solo; custom BRLE wire masks never co-batch with
device-geometry fires.

- Beam/MCTS: branch topology, scores, logical pages, masks, and append cursors
	stay in PTIR.
- Quest: computes `attn_page_mask` in `on_attn_proj`; the WorkingSet and memory
	allocator know nothing about the selection algorithm.
- Constrained decoding: remains logits/channel logic and does not enter the
	memory model.
- Reordering for attention is a PTIR tensor operation over logical page ids, not
	a WorkingSet remap.
- Backing reclaim and pool resize do not renumber live `PhysicalKvPageId`s or
	change PTIR state.

`fork` creates a child over the complete logical address space. Parent and child
initially point at the same terminal. The operation is O(1). When they diverge,
fresh child nodes are attached below the shared terminal. This is the normal
primitive for beam/MCTS branching and self-correction.

`slice` creates a structurally shared child from one range and rebases that range
to page zero in the child. Rebasing needs no node chain: lookup anchors at the
terminal, so the front cut is implicit in the child's smaller `mapped_len` (see
Page Table and Node Pages). A slice ending at a node boundary points directly at
that node, whether or not the slice starts at page zero; a slice ending inside a
node creates one prefix `Pages::ParentSelection` of that node as the child's
terminal. Pages in front of the sliced range remain on the child's ancestor
path: they are invisible to lookup but stay reachable, and therefore
unreclaimable, while the child lives. A child may be returned immediately in a
pending state; later commands on the same Pipeline depend on the fork/slice
command automatically.

## Discard and Residency

`reserve(n)` is purely logical: it extends the WorkingSet's index space by `n`
and returns the new WorkingSet-relative index range. No physical slot is
allocated. Physical `PhysicalKvPageId`s are allocated only at forward
preparation, for the rows that actually compute after cache matching, so a
fully cached request never holds pool slots for its prefix. Pool exhaustion
therefore surfaces only at the scheduler, never from `reserve`.

`discard(ranges)` removes those ranges from the WorkingSet mapping and leaves no
tombstone. Ranges are interpreted against pre-discard indexes and applied
atomically. Over a private region (no other WorkingSet terminal, cache/checkpoint
root, parent selection, or in-flight snapshot reaches the affected nodes), the
`ids`, `token_hashes`, and `page_hashes` vectors of owned nodes are drained over
the affected ranges, across node boundaries if needed, and the removed slots are
released; a private `Pages::ParentSelection` rewrites its runs in place and owns
no slots to release.

On a shared path, a discard composes three mechanisms, and a range is legal only
if they cover it:

- A range falling within the terminal node's local contribution becomes a
	`Pages::ParentSelection` of the terminal node (ordered runs cover interior
	holes) serving as the new terminal.
- A tail-reaching range whose boundary lies above the terminal node moves the
	terminal up to the node at the boundary, or to a prefix selection of the
	node containing it.
- A front-reaching range whose boundary lies above the terminal node is
	expressed by reducing the mapped extent alone; the excluded pages remain on the
	ancestor path, invisible to lookup but not reclaimable through this
	WorkingSet.

Any other interior range on a shared path is rejected with an error: honoring it
would re-parent surviving shared suffix nodes under a new selection, which the
growth-boundary invariant forbids (see Trie Split, Merge, and Lifetime). When a
shared discard creates a selection and that selection later becomes the only
remaining user of the owner's node-local pages, owner compaction takes ownership
of the selected entries and releases the excluded slots.

Because suffix indexes shift, all old `WorkingSetPageIndex`s at or after a
removed range become invalid; the inferlet must publish new PTIR geometry after
the ordered discard. If stable interior indexes become a requirement, add an
explicit sparse namespace rather than hiding one in this design.

Residency is not a WorkingSet API. The runtime MemoryBroker may reclaim,
transfer, or restore a KV page's backing after all in-flight users retire. The
WorkingSet mapping retains the same stable `PhysicalKvPageId`. CUDA VMM or Metal
sparse mapping may unmap and later restore backing beneath that slot, but a live
page is never renumbered. No public residency-control methods are exposed.

## Minimal WorkingSet Specification

### Page Table and Node Pages

`KvPageTable` owns the hash-labeled prefix trie and the WorkingSet terminal
registry. It does not maintain a reverse hash index or a second logical-to-
physical page directory. A node either owns aligned physical-id, token-hash, and page-hash
vectors or selects ordered runs from a parent node; there is no separate segment,
selection, or path arena.

```rust
#[repr(transparent)]
struct WorkingSetPageIndex(u32);

#[repr(transparent)]
struct PhysicalKvPageId(u32);

struct WorkingSet {
	id: WorkingSetId,
}

struct KvPageTable {
	working_sets: SlotMap<WorkingSetId, WorkingSetEntry>,
	nodes: SlotMap<NodeId, KvTrieNode>,
}

struct WorkingSetEntry {
	terminal: Option<NodeId>,
	// Logical extent including pending (reserved, unpublished) space.
	page_len: u64,
	// Exclusive end of the published mapping; the lookup anchor.
	mapped_len: u64,
	// Derived, device-visible flattened mapping. Pinned and address-stable
	// (never a reallocating Vec), versioned, epoch-retired on rewrite.
	flat_table: Option<FlatTableHandle>,
}

struct KvTrieNode {
	parent: Option<NodeId>,
	children: SmallVec<[NodeId; 2]>,
	pages: Pages,
	cached_path_hash: Option<[u8; 32]>,
}

enum Pages {
	Owned {
		ids: Vec<PhysicalKvPageId>,
		token_hashes: Vec<Vec<Option<[u8; 32]>>>,
		page_hashes: Vec<[u8; 32]>,
	},
	ParentSelection {
		runs: SmallVec<[Range<u32>; 2]>,
	},
}
```

For every owned node, `ids.len() == page_hashes.len()` and equal indexes describe
the same page; `token_hashes[i]` holds one `Option<[u8; 32]>` per token slot of
page `i`, where `None` means unwritten or invalid. Keeping the vectors separate
lets host cache matching scan only page hashes while driver mapping publication
reads only physical ids. An owned node needs no run metadata. A `ParentSelection`
applies the same ordered runs to the `ids`, `token_hashes`, and `page_hashes`
vectors of its parent; its source is therefore implicit in `parent`. The selected
entries replace the parent's local contribution on that branch rather than being
appended to it. Its ordered ranges cover both contiguous ranges and structural
sparse selection.

Creating a `ParentSelection` from another `ParentSelection` composes their runs
against the original owned ancestor and inserts the result as another child of
that owner. Parent selections therefore form siblings rather than a recursive
selection chain. Dynamic sparse attention and per-fire reorder remain PTIR
operations; they do not create or mutate `Pages::ParentSelection`.

Trie edges are structural and do not require a reproducible semantic hash;
opaque passes and rows outside the canonical KV-production shape (see Matching
a Submitted Forward) may still extend and branch a WorkingSet. Such
pages receive fresh opaque token-slot hashes that preserve concrete identity across
forks/parent selections but cannot be reproduced by an unrelated forward.
Recipe-derived and opaque token-slot hashes occupy the same `token_hashes` entries;
only the former support automatic matching from semantic inputs.

A `WorkingSetEntry` keeps two lengths. `page_len` is the logical extent
including pending reservations: `reserve(n)` returns the range starting at
`page_len` and advances it. `mapped_len <= page_len` is the exclusive end of
the published mapping and the lookup anchor; publication advances it,
`discard` reduces both, `slice` sets both to the selected range length, and
`fork` copies both. The anchor must exclude reserved-but-unpublished space:
with a purely logical `reserve`, anchoring at `page_len` would shift every
committed index while a reservation is outstanding. Because lookup anchors at
the terminal, `mapped_len` also encodes front truncation: ancestor pages
beyond it are never reached by lookup, so a front cut from `slice` or a
front-reaching `discard` needs no structural node.

Given `WorkingSetPageIndex(i)`, lookup starts from the terminal with an exclusive
end equal to `mapped_len` (an index at or past it is unwritten or out of
range). At each node it subtracts the node-local page count to
derive that node's start. If `i` is within the resulting span,
`Pages::Owned` reads the node-local id directly and `Pages::ParentSelection`
resolves the offset through `runs` against its parent's id vector. Page hash
lookup applies the same offset/runs to the parallel `page_hashes` vector. Otherwise lookup
continues toward the logical predecessor with the derived start as its new
exclusive end. A derived start may be negative when `mapped_len` truncates the
front of the path; every valid index resolves before the walk passes index zero,
and pages beyond the truncation are invisible to this WorkingSet. When
stepping past a `ParentSelection`, it skips the parent
contribution that the selection replaces. The runtime performs this walk once per
mapping change to refresh the WorkingSet's flattened table; kernels and drivers
only read the table and never walk the trie. The result is already a
`PhysicalKvPageId`; the kernel address is:

```text
kv_pool_base + PhysicalKvPageId * kv_page_bytes
```

There is no software `KvPageId -> PhysicalKvPageId` lookup. Hardware VMM address
translation remains below the pool address. Mapping changes are published by
refreshing the WorkingSet's flattened table in Pipeline order; there are no
separate driver-side mirrors.

Full-prefix sharing points multiple WorkingSets at the same trie nodes and needs
no selection node. A partial `slice` or a shared discard materialized at the
terminal node creates another node with `Pages::ParentSelection`, without
copying KV data, physical ids,
token hashes, or page hashes. The owned parent and all its slots remain live
while any parent selection references it; there is no per-page coverage refcount.
If exactly one direct parent selection is the only remaining consumer of the
parent's node-local pages, compact all owned vectors to that selection, release
the excluded slots,
and rewrite its runs against the compacted vectors. The node remains a
`ParentSelection`; this operation does not merge semantic node identities.

### Trie Split, Merge, and Lifetime

`fork` at a WorkingSet terminal needs no split. A split is required only when a
new terminal, branch, slice boundary, or discard boundary falls inside a
radix-compressed node. A private full-run node may be split by moving identical
ranges from its `ids`, `token_hashes`, and `page_hashes` vectors into new owned
vectors. A shared node is never split physically, because other selections' runs
and in-flight snapshots index its vectors; it is split logically with parent
selections. The two scenarios produce different shapes:

```text
Node N pages = Owned {
	ids:         [P0, P1, P2, P3, P4],
	page_hashes: [H0, H1, H2, H3, H4],
}

Branch at offset 3 (divergence or cache-hit attach):
prefix node pages     = ParentSelection { runs: [0..3] }      // child of N
new suffix node pages = Owned(fresh ids, prospective hashes)  // child of prefix
WorkingSets holding N's full extent keep terminal N unchanged; no node for N's
old suffix is created.

Discard of [0..3) by a WorkingSet whose terminal is N:
suffix node pages     = ParentSelection { runs: [3..5] }      // child of N,
new terminal of the discarding WorkingSet
```

A `ParentSelection` is always created at the growth boundary of some WorkingSet
mapping: the point where that WorkingSet's use of shared history ends and its
new content begins. Everything below a selection is created after it; a
pre-existing shared node is never re-parented under a selection. Later appends
make a selection an interior node, but only by attaching fresh growth below it.
This invariant is what keeps `parent` sufficient as the implicit selection
source, and what rejects interior discards on shared paths.

Unary nodes are allowed, but unobserved unary nodes should be compressed. A node
with one child may merge with that child only when it is not a WorkingSet
terminal, cache/checkpoint boundary, slice boundary, root, or in-flight snapshot;
the merged extent must remain within the configured node size. Parent selections
of the same owned parent merge by coalescing adjacent runs. Owned nodes merge
only when both are private so their physical ids, token hashes, and page hashes
can be moved into the corresponding vectors. Neither case copies KV bytes.

The owner-compaction rule is separate from unary semantic-node merging. It is
allowed only when the sole direct parent selection is the only live consumer of
the owner's node-local pages; a WorkingSet terminal, semantic descendant,
cache/checkpoint boundary, another parent selection, or in-flight snapshot that
observes the owner's complete extent blocks compaction. Compaction changes both
owner vectors and the sole selection's runs, but does not rewire trie children
or remove either node. After compaction the selection's runs are the identity
run over the compacted vectors, and the owner/selection pair becomes an ordinary
unary-merge candidate under the merge guards above; slot reclamation does not
wait for that merge.

A `ParentSelection` is never removed merely because its owner is compacted. It
becomes reclaimable only when its own subtree is unreachable from every
WorkingSet terminal, cache/checkpoint root, and in-flight snapshot. Reclaiming it
then means removing that ordinary child from `parent.children`; there is no
separate `ParentSelection` registry or unregister operation.

Lifetime is tracked by reachability from WorkingSet terminals, strong
cache/checkpoint roots, and in-flight snapshot pins. A cache lease is what keeps
an otherwise unused root alive. The node has no embedded terminal or external
reference counters. The normal parent/child edge keeps an owned parent alive
while a `ParentSelection` child exists. The sequencer owns mutation ordering, and
completion epochs delay reclamation until all submitted users retire. There are
no page-level `refs`, `readers`, or `writer` fields.

### Token-Slot Hashes, Page Hashes, and Trie Matching

Three levels of hash capture semantic identity at different granularities.

**Token-slot hash.** Any two semantically identical committed token slots carry
the same `[u8; 32]` value. The cache domain (model/weight identity, adapters, KV
format, relevant model configuration, and other pass-wide inputs), the recipe,
and all causal token/embedding inputs, positions, masks, and other cache-producing
inputs jointly determine it. Where a reproducible recipe exists, the hash is
derived from those inputs; where no recipe covers the slot, a fresh opaque
`[u8; 32]` is assigned. Opaque values preserve concrete identity across forks and
parent selections but cannot be matched by an unrelated forward.

**Page hash.** Each physical page's entry in `page_hashes` is a `[u8; 32]` that
deterministically folds the page's format and validity state together with the
ordered sequence of its token-slot hashes. Recomputing a page hash requires only
that page's slot hashes; no other page's KV bytes or hashes are involved.

**Cached path hash.** `KvTrieNode.cached_path_hash` is a `[u8; 32]` that
deterministically folds all visible page hashes from the trie root through and
including the node's own pages, independent of where radix node boundaries fall.
It summarizes the complete causal prefix at that node boundary. A
`ParentSelection` contributes by folding only the page hashes of its selected
runs, starting from the path hash before the parent's local contribution, so the
path hash is always a function of a well-defined ordered set of page hashes.
`cached_path_hash` is `None` when any contributing page hash is not yet valid or
committed; it is recomputed lazily on demand and cached until invalidated.

**Scope and limitation.** The cached path hash is designed for complete-path and
causal-prefix hashing: a full prefix in the trie followed by sequentially appended
pages. Arbitrary sparse or reordered PTIR page selections do not use this cache;
they fold only the relevant `page_hashes` entries directly. If the recipe cannot
express a selection's dependencies, reproducible hashing is skipped for that
selection. No additional data structure is introduced for arbitrary selections.

The hash is a semantic fingerprint, not a hash of KV bytes. An optional device-
computed content hash may verify transferred bytes or detect semantic-hash bugs,
but it is not required on the normal lookup path.

Local content-addressable lookup is the trie traversal itself. An extension of a
WorkingSet starts at its captured terminal and uses the cached path hash (or
recomputes it from visible page hashes) to derive the next candidate. The planner
compares candidate page hashes against visible node-local `page_hashes` runs.
After consuming a run, it selects the child whose first visible page hash matches
the next candidate. On a mismatch inside a compressed run, it also checks for a
prefix `ParentSelection` that materializes a branch at the matched offset and
continues through that selection's children. A mismatch with no such branch ends
the local hit. A candidate ending inside a radix-compressed node identifies that
transient `(NodeId, offset)` boundary; a split or prefix `ParentSelection` is
materialized only if a terminal or branch must be attached there. No persistent
reverse map from a page hash to a `(NodeId, offset)` exists; a prefix held only
by an unrelated WorkingSet is found through `CacheFabric`, not a local scan.

A private terminal node may update ids and hashes in place while no WorkingSet,
cache/checkpoint root, parent selection, semantic descendant, or in-flight
snapshot requires the old path. A hit becomes such an observer before mutation
through the same sequencer, so a path that must remain observable uses CoW rather
than changing under its existing page-hash and cached-path-hash sequence.

### Matching a Submitted Forward

There is no separate recipe artifact. The cache recipe is the canonical
KV-production shape the planner recognizes directly in the submitted pass:
token-input rows written to sequential `WSlot` targets under the standard
causal mask. For rows in that shape, token-slot hashes derive from the cache
domain, token ids, and positions already present in the pass inputs. Rows
outside the shape (embedding inputs, custom write patterns) are opaque and
skip automatic matching.

Cache hits are applied by trimming the pass, invisibly to the inferlet, under
four rules:

1. Only a prefix of query rows is trimmed; KV-side geometry (`Pages`,
	`PageIndptr`, `KvLen`) is untouched, so the residual pass runs the ordinary
	chunked-prefill kernel path.
2. A row referenced by an observed output (readout or host-visible channel
	write) is never trimmed. The trim bound is
	`min(hit boundary, earliest observed row)`; a pass observing all rows trims
	nothing and needs no special casing.
3. All row-indexed ports slice uniformly over the same row range; stage
	programs must be row-count-parametric and never hardcode row counts.
4. Trimmed rows must target exactly the logical slots the attached hit
	provides; otherwise the pass is opaque.

Before allocating output slots, the forward planner performs:

1. Bind the submitted PTIR pass to a WorkingSet and capture its immutable
	terminal snapshot.
2. Recognize the canonical KV-producing row range. An entirely opaque pass
	skips automatic cache matching.
3. Derive candidate page hash values for complete output pages in causal order.
4. Start at the captured terminal and use its cached path hash (or recompute from
	visible page hashes) to derive the next candidate. Compare the candidate
	sequence against node-local `page_hashes` runs, including branches anchored by
	prefix `ParentSelection` nodes inside compressed runs. At the first local
	mismatch, ask `CacheFabric` for a remote location or in-flight producer.
5. For each prefix range, choose local hit, remote hit plus transfer, remote
	prefill plus transfer, or local compute. Attach the longest contiguous hit
	by sharing the matching trie path, materializing an interior boundary only
	when required, and trim the covered rows under the rules above.
6. At the first causal miss, allocate fresh physical slots and a pending child
	node, then submit only the residual rows.
7. After successful compute or transfer, publish aligned ids, token hashes, and
	page hashes with the mapping in Pipeline order.

The same trim machinery serves disaggregated prefill: a remote provider covers
a prefix range and the local pass runs only the residual rows.

For a hybrid model, a match is a cache bundle: the KV prefix and recurrent-state
checkpoint at the same semantic boundary must both be available.

**Implemented recognition and chain rules (2026-07, increments 1–2a).**
Recognition is runtime-decided in two halves; a guest declaration could poison
a cross-inferlet cache and is rejected as a design option.

- *Bind time* (`canonical_kv_shape`): no `AttnMask`/`Positions`/device-geometry
	ports, no `on-attn`/`on-attn-proj` stage programs, no extern channels, a
	single trace-const lane, and a `KvLen` port present. Prologue/epilogue-only
	passes (grammar, watermark, samplers) remain canonical.
- *Fire time* (`canonical_fire_evidence`): the embed value THIS fire consumes
	must be host-known — a trace constant, the staged Writer put shipping with
	the fire, or (first fire) the seed — and the host-known kv-len must equal
	`committed + new` (full-context attention). Device-carried decode tokens are
	unknowable at prepare and hash opaque; prefix-cache value concentrates in
	host-known prefill.
- *Chain state* (per WorkingSet): the identity the next appended slot chains
	from. Commit sets it to the fire's highest slot hash; fork and full-range
	slice inherit it (continuations hash identically); tail-only surgery
	continues from the last surviving slot hash; front/interior surgery refolds
	the visible page identities so post-surgery appends never impersonate the
	unedited continuation.
- *Retention*: releasing a WorkingSet with canonical content leases its
	terminal as a cache root (bounded FIFO, `PIE_KV_CACHE_ROOTS_MAX`); ladder
	rung 1 reclaims retained roots under pressure, and the fire path runs rung 1
	inline even in Error mode so retention never surfaces as guest OOM.
- *Landed*: empty-WS longest-prefix graft (`adopt_path_prefix` /
	`ptir_kv_match_prefix`, always leaving the readout token to compute).
- *Landed (matching 2b)*: the fire-path trim, and it required NO driver or
	compiled-program change — correcting the earlier "descriptor-level driver
	rewrite" assessment, which conflated device-geometry programs with
	canonical ones. Canonical programs are never device-geometry (the shape
	gate forbids exactly the ports that would let the program own geometry),
	so their forward geometry is runtime-derived launch DATA, and cache
	eligibility (the two gates above) is exactly trim eligibility: token
	values host-known, geometry host-owned, `KvLen` (total attended context)
	invariant under the trim, post-forward programs row-relative over their
	sampled-logits segment. `submit_pass` probes the CAS on the first
	canonical fire of a fresh WorkingSet (probe capped so no adopted page
	covers a sampled row), grafts, and slices the launch to the suffix with
	`prepare(committed = grafted tokens)` — indistinguishable from a
	continuation fire. Misses and ineligible shapes fall through to full
	compute, so the cache stays a pure opportunistic optimization.
	Device-geometry (Design-B) passes are not canonical and not cacheable by
	construction.

### CoW

Every PTIR KV output is a write intent. Preparing a write chooses:

```text
fresh reserved slot
	-> fresh backing, no copy

private terminal node with `Pages::Owned` and no
parent-selection/cache/in-flight observer
    -> write in place

shared or cache/checkpoint-retained node, or `Pages::ParentSelection`
	-> reserve a fresh child node and CoW
```

CoW copies only cells that must survive the write. A full-page overwrite needs no
copy. A partial append to a shared tail copies the preserved prefix cells before
the driver writes new cells. The old branch retains all of its cached page hashes
and `cached_path_hash` unchanged. The new branch node starts with
`cached_path_hash: None`.

Multiple PTIR selections reading one WorkingSet do not trigger CoW by themselves.
Across Pipelines, the sequencer serializes conflicting commands. CoW is required
when an old trie path must remain independently observable, not merely because a
page has an in-flight reader counter.

The destination mapping is pending until the copy and forward commit. On failure
or dummy-run, the reservation is released and the committed mapping remains
unchanged. No pool-global lock or generic pool transaction is required.

### Hash Update Lifecycle

Hash and mapping publication follows the Pipeline command order:

1. `reserve`: extend the logical index space only. Physical slots and the
	pending private node are created at forward preparation for the miss
	suffix; prospective token-slot hashes (recipe-derived or fresh opaque) and
	their derived page hashes remain pending.
2. Forward preparation: keep the destination node update pending; the committed
	WorkingSet terminal and published flattened table remain unchanged.
3. Trie hit or completed transfer: attach the matching path and publish the new
	terminal snapshot.
4. Successful private in-place write: the physical page id remains unchanged. For
	each committed page, replace the affected entries in `token_hashes` with the
	committed token-slot hashes and recompute only that page's entry in
	`page_hashes`; then set `cached_path_hash = None` on the owning node and
	propagate that invalidation to all trie descendants. No other page's KV bytes,
	token hashes, or page hashes are rewritten or recomputed. Affected token-slot
	hashes remain pending until this commit; on failure they are not published (see
	step 6). This is consistent with the existing rule that partial or failed
	writes do not alter the committed state.
5. Successful fresh/CoW write: publish the child node, terminal, and aligned
	`ids`, `token_hashes`, and `page_hashes` vectors atomically in Pipeline order.
6. Failure, poison, or readiness dummy-run: release pending slots and metadata;
	the committed terminal, topology, token hashes, page hashes, and cached path
	hashes do not change. Pending token-slot hashes from an in-place write are
	discarded and not published; the existing committed content and hashes remain
	authoritative.
7. `discard`: drain the same ranges from a private node's `ids`, `token_hashes`,
	and `page_hashes` vectors, publish `Pages::ParentSelection` over the
	existing owner, or reduce the mapped extent alone for a front-reaching range
	above the terminal node. No tombstone remains; suffix `WorkingSetPageIndex`s
	shift.
8. Backing reclaim/restore and pool resize preserve live physical ids and update
	only driver/VMM backing state after in-flight users retire.

PTIR output declarations are conservatively treated as write intents, but page
table topology and artifact identity change only for outputs the driver reports
as committed.

## Typed Stores and Pools

Do not use a common KV-page-sized block or generic `ArenaKind`.

```text
KvStore                  -> KvBackingPool
RecurrentStateStore      -> StateBackingPool
RecurrentBufferStore     -> BufferBackingPool
ScratchStore             -> ScratchBackingPool
```

Each Store owns its semantics:

- KV: semantic trie/node sharing, content-addressable dedup, CoW, and stable
	physical-slot allocation.
- Main recurrent state: model-defined composite slot, reset, copy, checkpoint.
- Recurrent buffer: candidate allocation, copy, promotion.
- Scratch: transient reserve/release only.

In the initial implementation, backing pools only reserve and release stable
slots over driver-preallocated static GPU memory. There is no dynamic resize,
`MemoryBroker`, or CUDA VMM / Metal sparse map/unmap. OOM from a pool is
surfaced to the inference scheduler, which applies the contention ladder
described under Scheduler in the module architecture: drop unused cache leases
first, then FCFS preemption.

**Backing evolution (later phase).** A `MemoryBroker` arbitrates a shared byte
budget between pools. The driver performs CUDA VMM or Metal sparse-buffer
map/unmap to grow or shrink mapped capacity on demand. Live `PhysicalKvPageId`s
remain stable across resize; only device-side backing changes.

## Deduplication and Disaggregation

Cache artifacts identify an immutable semantic trie boundary. A boundary is
addressed by the resulting output page hash sequence and path provenance: the
ordered `page_hashes` of the pages up to that boundary, summarized for local
lookup by the node's `cached_path_hash`. The cache domain (model/weights,
adapters, dtype/layout, RoPE/configuration, and other pass-wide inputs) is folded
into the token-slot hashes and therefore into the page hashes; it is not stored
as a separate field. A hybrid-model cache bundle contains both KV pages and the
recurrent-state checkpoint required at the same prefix boundary.

`CacheFabric.ensure_local(requirement)` resolves one of:

1. local cache hit;
2. remote cache hit plus transfer;
3. remote prefill plus transfer;
4. local suffix compute.

Local prefix traversal happens before physical allocation. A local hit attaches
existing immutable nodes or inline parent selections to the WorkingSet. On a
local miss, `CacheFabric` uses the output page hash or `cached_path_hash` for
remote and in-flight lookup; only the missing suffix is sent for compute. Its in-flight entries provide single-flight
deduplication so concurrent identical prefills do not all compute. This external
directory is not a reverse index in `KvPageTable`. Cross-node transfer creates a
local replica; physical slots and node references never cross nodes.

Remote materialization is a Pipeline command. The Pipeline remains non-blocking;
dependent launches wait in order until transfer completes.

## Runtime Module Architecture

### Module Tree

```text
runtime/engine/src/
  store/
    mod.rs            # re-exports; KvStore, RsStore, pool types
    registry.rs       # WorkingSet registry; maps WorkingSetId to store entry
    pool.rs           # typed physical-ID free list over driver static pools
    kv/
      mod.rs          # KvStore: WorkingSets, page table, alloc, CoW, CAS, lifecycle
      working_set.rs  # KvWorkingSet: thin WIT/resource handle; delegates to KvStore
      page_table.rs   # KvPageTable: trie, ParentSelection, path-hash invalidation/recompute
      hash.rs         # pure token-slot/page/cached-path hash calculations
      cas.rs          # CAS integration; cross-WorkingSet prefix via CacheFabric
      write.rs        # KvPreparedWrite: per-fire prepared operation
    rs/
      mod.rs          # RsStore: RS WorkingSets, folded state, CoW/reset/fold/promote
      working_set.rs  # RsWorkingSet: thin WIT/resource handle; delegates to RsStore
      write.rs        # RsPreparedWrite: per-fire prepared operation
  ptir/
    ptir_host.rs      # PendingFire lifecycle; owns Option<KvPreparedWrite/RsPreparedWrite>
    ptir_kv.rs        # PTIR geometry/intents -> KvStore prepare calls
    ptir_rs.rs        # PTIR geometry/intents -> RsStore prepare calls
  driver/             # mostly unchanged for initial static-pool deployment
  scheduler/          # OOM surfacing and contention routing changes
```

The existing `working_set/` directory and generic `arena/` directory are
replaced by the `store/` hierarchy. `Arena`, `ArenaKind`, `ArenaHandle`,
`ArenaTxn`, `ObjectId`, generic `CowPlan`, generic `MovePlan`, `Residency`,
and generic object refcounts/pinning are deleted. Low-level free-list mechanics
may survive in `store/pool.rs` only if directly reusable without semantic
changes.

### Responsibility and Ownership Boundaries

**Driver.** Owns the preallocated static GPU memory pools and maps typed
physical IDs to device addresses. Does not know WorkingSet, CoW, CAS/hashes,
trie topology, or allocation ownership. Dynamic backing is a later evolution
(see the Backing Evolution note in [Typed Stores and Pools](#typed-stores-and-pools)).

**`store/pool.rs`.** A typed physical-ID free list over driver static pools,
with `try_alloc` and completion-epoch-delayed recycle. It must not become a
renamed generic `Arena` and must not own CoW logic, hash maintenance,
mapping/residency, or object refcounts. OOM propagates up to the scheduler.

```rust
struct KvPool { free: Vec<PhysicalKvPageId> }

impl KvPool {
    fn try_alloc(&mut self) -> Option<PhysicalKvPageId> { ... }
    fn recycle_after_epoch(&mut self, id: PhysicalKvPageId, epoch: u64) { ... }
}
```

**`store/registry.rs`.** Maintains the mapping from `WorkingSetId` to its owning
store. Used by `ptir_host.rs` and the driver to resolve handles without each
component holding a direct store reference.

**`store/kv/mod.rs` — `KvStore`.** Owns KV WorkingSets, `KvPageTable`, typed
pool access, implicit CoW allocation, hash lifecycle, CAS integration,
reachability-based lifetime, and the prepare/commit/abort protocol. `KvStore`
is the single authority over which `PhysicalKvPageId`s are live.

**`store/kv/working_set.rs` — `KvWorkingSet`.** A thin WIT/resource handle
containing a `WorkingSetId` and a reference to its `KvStore`. All substantive
operations delegate to `KvStore`. The `kv-working-set` WIT resource maps
directly to this type.

**`store/kv/page_table.rs` — `KvPageTable`.** Owns the mapping trie, WorkingSet
terminal registry, `ParentSelection` composition, and `cached_path_hash`
invalidation and lazy recomputation. Does not allocate `PhysicalKvPageId`s or
call driver APIs. `KvStore` passes freshly allocated IDs in; `KvPageTable`
places them into node vectors. `ParentSelection`-based path discovery and local
cache matching (trie traversal by page hash) live here. There is no reverse
hash index in `KvPageTable`; cross-WorkingSet prefix discovery goes through
`CacheFabric` / local CAS in `cas.rs`.

**`store/kv/hash.rs`.** Pure functions: token-slot hash derivation (recipe-based
or fresh opaque), page hash folding from token-slot hashes, and cached-path hash
computation up through a node. No mutable state. Invoked by `KvPageTable` and
`KvStore`; results are stored in trie node vectors.

**`store/kv/cas.rs`.** CAS integration: verifying CAS conditions against a
prepared write before commit, and dispatching cross-WorkingSet prefix discovery
through `CacheFabric` or a local CAS index. Kept separate from `KvPageTable`;
the CAS index (if local) lives here, not in the trie.

**`store/kv/write.rs` — `KvPreparedWrite`.** Defines the per-fire prepared
operation: newly allocated `PhysicalKvPageId`s, copy plans, and pending
mapping/hash deltas, retained until async driver completion confirms the epoch.
This is not a transaction manager or operation framework. `KvStore` commits the
prepared write on success and aborts/releases on failure. `PendingFire` in
`ptir_host.rs` owns `Option<KvPreparedWrite>`.

```rust
struct KvPreparedWrite {
    allocated:    Vec<PhysicalKvPageId>,
    copy_plan:    Vec<CopySegment>,
    pending_node: Option<NodeId>,
    // pending token_hashes / page_hash deltas until driver epoch
}
```

**`store/rs/mod.rs` — `RsStore`.** Separately owns RS WorkingSets, folded
recurrent state and buffers, typed static pools, CoW/reset/fold/promotion, and
its own prepare/commit/abort protocol. Does not reuse `KvPageTable`, trie
structure, or KV hash semantics.

**`store/rs/working_set.rs` — `RsWorkingSet`.** Thin WIT/resource handle;
delegates to `RsStore`.

**`store/rs/write.rs` — `RsPreparedWrite`.** Per-fire prepared RS operation;
same lifecycle discipline as `KvPreparedWrite`.

**`ptir/ptir_kv.rs`.** Translates PTIR geometry and KV output intents into
`KvStore::prepare` calls. Does not own allocation, CoW logic, or publication;
that authority remains in `KvStore`.

**`ptir/ptir_rs.rs`.** Translates PTIR recurrent-state intents into
`RsStore::prepare` calls. Same boundary as `ptir_kv.rs`.

**`ptir/ptir_host.rs`.** Keeps the existing `PendingFire` lifecycle unchanged.
Adds `Option<KvPreparedWrite>` and `Option<RsPreparedWrite>` fields. No new
queue, transaction framework, or operation registry is introduced.

**Scheduler.** Receives OOM signals from `KvPool::try_alloc` or RS pool
equivalents, raised only at forward preparation because `reserve` is logical.
Contention policy is a two-step ladder. First, drop unused cache-lease roots:
prefixes retained only by a lease are reclaimed with no work lost. If pressure
remains, preempt inferlets FCFS, most recently started first. Preemption in
this model means releasing the victim's WorkingSets: their nodes become
unreachable and their slots recycle after in-flight epochs retire; there are
no refcounts to decrement. Victim selection is sized by exclusive footprint,
the pages reachable only from that inferlet's terminals, which the trie yields
cheaply as its private suffix nodes; shared prefixes are not freed by
preempting one sharer. Without residency in the initial phase, a preempted
inferlet's KV is discarded and recomputed on resume, where it may hit
surviving prefix cache. The scheduler does not call pool or store internals
directly beyond the OOM signal and lease/preempt decisions.

### Migration from Arena

| Symbol / directory | Disposition |
|--------------------|-------------|
| `Arena`, `ArenaKind`, `ArenaHandle` | Delete; replaced by `KvStore`/`RsStore` |
| `ArenaTxn` | Delete; replaced by `KvPreparedWrite`/`RsPreparedWrite` |
| `ObjectId` | Delete; use `PhysicalKvPageId` or RS-specific typed IDs |
| generic `CowPlan`, generic `MovePlan` | Delete; CoW logic moves to `KvStore`/`RsStore` |
| `Residency` | Delete; residency is a later-phase driver/MemoryBroker concern |
| generic object refcounts / pinning | Delete; lifetime is reachability-based (see Trie Lifetime) |
| `working_set/` directory | Split into `store/kv/working_set.rs` and `store/rs/working_set.rs` |
| `arena/` directory | Delete; free-list mechanics only survive in `store/pool.rs` |

## Completion and Direct FFI

Inferlets do not receive per-operation completion objects. WorkingSet operations
and launches return enqueue/validation success; values and asynchronous errors
remain observable through channels.

Internally, the driver publishes channel/pipeline epochs and invokes a payload-
free callback. A later launch completion may retire preceding copies, mapping
updates, and pool operations on the same ordered stream. The direct FFI exposes
typed operations (`copy_kv`, `copy_state`, `resize_pool`, `launch`) and the
per-driver sequencer remains the only local operation queue. The flattened
WorkingSet tables are pinned host regions shared with kernels in process; they
fall under the persistent-region ownership rule (addresses stable until
ordered close), not the copy-on-call descriptor rule, and publishing them
requires no new driver operation.

## Key Invariants

1. Physical page ids never cross the inferlet API.
2. A launch binds a persistent WorkingSet; PTIR provides logical attention
	geometry and never exposes physical page ids.
3. Same-Pipeline order is deterministic; cross-Pipeline commands are
	 linearized by the single per-driver sequencer queue, and two write-intent
	 fires on the same WorkingSet never co-batch. There is no hazard tracker;
	 conflicting overlapping accesses an inferlet constructs beyond this
	 corrupt only its own output values.
4. Trie and remote-cache hits attach only immutable node boundaries proven by
	matching page hashes and `cached_path_hash` provenance.
5. Mapping publication happens only after copy, compute, or transfer succeeds.
6. A live `PhysicalKvPageId` is a stable pool offset; backing reclaim, restore,
	and resize never renumber it or change its device address.
7. Runtime backing reclaim waits for all in-flight users, regardless of
	Pipeline.
8. Sharing is represented by shared trie nodes or a `Pages::ParentSelection`
	child over its parent's aligned `ids`, `token_hashes`, and `page_hashes`
	vectors; page-level refcounts, reader counts, writer flags, and
	mapping/content generations are not part of the KV page metadata model.
9. A `Pages::ParentSelection` is created only at the growth boundary of a
	WorkingSet mapping, and everything below a selection is created after it.
	Pre-existing shared nodes are never re-parented under a selection, so a
	mapping edit that needs a rerouted shared suffix (an interior discard on a
	shared path) is rejected.
