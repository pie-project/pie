/** @module Interface pie:inferlet/working-set@0.3.0 **/
export type Error = import('./pie-inferlet-types.js').Error;
export type Pipeline = import('./pie-inferlet-pipeline.js').Pipeline;
/**
 * A contiguous, half-open span [start, start + len) of WorkingSet-relative
 * page indexes. The ONLY references that ever cross this API are these
 * relative indexes — never physical page ids.
 */
export interface PageRange {
  start: number,
  len: number,
}
/**
 * A half-open declaration [start, end) of WorkingSet-relative page
 * indexes. Unlike page-range, an absent end follows later lease growth at
 * the storage contract. A pass using SDK-generated dense page geometry
 * must be recreated if growth exceeds its bind-time page envelope. When
 * end is present, start must not exceed it. This type is only for forward-
 * pass readable/writable declarations; reserve/discard/slice continue to
 * use finite page-range values.
 */
export interface PageSpan {
  start: number,
  end?: number,
}

export class KvWorkingSet {
  constructor()
  /**
  * Tokens per KV page for this working set's model/driver.
  */
  pageSize(): number;
  /**
  * Current logical extent in pages, including reserved space whose
  * pages have not been written yet.
  */
  pageLen(): number;
  /**
  * Extend the logical address space by `pages`; returns the added
  * range. Purely logical: physical pages are allocated only when a
  * forward writes them.
  */
  reserve(pages: number): PageRange;
  /**
  * Atomically insert or replace `key` with a structurally retained view
  * of this working set. Keys are opaque bytes scoped to the active
  * model/driver store. The working set must have no in-flight fire and
  * every logical page must be physically mapped.
  */
  updateIndex(key: Uint8Array): void;
  /**
  * Exact best-effort lookup. A missing or pressure-evicted key returns
  * none. The returned working set structurally shares the indexed
  * mapping and participates in the normal copy-on-write rules.
  */
  static fromIndex(key: Uint8Array): KvWorkingSet | undefined;
  /**
  * Remove only the named index root. Working sets already returned by
  * from-index remain valid. Returns false when the key is absent.
  */
  static removeIndex(key: Uint8Array): boolean;
  /**
  * Remove `ranges` (pre-discard indexes, applied atomically) from the
  * mapping, ordered on `on`. Suffix indexes shift; no tombstone
  * remains. Rejected when an interior range on a shared path would
  * reroute a shared suffix (growth-boundary invariant).
  */
  discard(on: Pipeline, ranges: Array<PageRange>): void;
  /**
  * O(1) copy-on-write child over the complete logical address space,
  * ordered on `on`. The normal primitive for beam/MCTS branching and
  * self-correction.
  */
  fork(on: Pipeline): KvWorkingSet;
  /**
  * Structurally shared child over `range`, rebased to page zero,
  * ordered on `on`. Pages in front of the range stay reachable (and
  * unreclaimable) while the child lives.
  */
  slice(on: Pipeline, range: PageRange): KvWorkingSet;
  /**
  * Ordered KV cell move within this working set (Design-B lazy KV
  * GC): move token cells, for ALL layers, from
  * (src-page-ids[i], src-tok-idx[i]) -> (dst-page-ids[i],
  * dst-tok-idx[i]); the four lists are parallel. Page ids are
  * WorkingSet-relative indexes. Rides the same in-flight FIFO as
  * submissions on `on`: ordered after every prior fire's KV write and
  * before every later fire's descriptor read. The caller guarantees
  * DISJOINT src/dst spans and computes the post-move layout itself.
  */
  copyInto(on: Pipeline, dstPageIds: Uint32Array, dstTokIdx: Uint32Array, srcPageIds: Uint32Array, srcTokIdx: Uint32Array): void;
}

export class RsWorkingSet {
  constructor()
  /**
  * Size in bytes of one folded recurrent-state object for this model.
  */
  stateSize(): bigint;
  /**
  * Current number of buffered page slots.
  */
  bufferSize(): number;
  /**
  * Tokens per buffered RS page for this working set's model/driver.
  */
  bufferPageSize(): number;
  /**
  * Append `n` fresh buffered page slots; returns the contiguous range.
  * Slots are materialized lazily on their first write.
  */
  allocBuffer(n: number): PageRange;
  /**
  * Remove buffered slots at `indices` and densely compact (call-time
  * interpretation). Invalid or duplicate indices return `error`.
  */
  freeBuffer(indices: Uint32Array): void;
  /**
  * Forget the last `count` buffered tokens: they never happened.
  * 
  * The twin of `rs-geometry.fold-len` on the other end of the buffer.
  * A fold moves the folded boundary RIGHT and is irreversible; this
  * moves the live end LEFT and is free, because the slots it releases
  * are simply overwritten by the next append.
  * 
  * Not part of `rs-geometry`, deliberately. `fold-len` is there because
  * it changes what the DEVICE does -- it is where the recurrent state
  * snapshot lands. This changes nothing on the device at all; it is a
  * statement about the working set, which is what a resource method is
  * for. It is also the token-granular sibling of `free-buffer`: that
  * one releases CAPACITY, this one releases CONTENT.
  * 
  * A speculative decoder is why it exists. A verify fire buffers a
  * whole window and a prefix of it is accepted; without this, the only
  * way to drop the rejected tail is `free-buffer`, which empties the
  * buffer wholesale and so forces the accepted prefix to be folded
  * away first -- in a second fire, since its length is not known until
  * the verify has run. Discard the tail instead and the NEXT window's
  * fire folds the previous prefix while writing its own tokens.
  */
  discardBuffered(count: number): void;
  /**
  * Reorder the buffered slots by the full bijection `perm`.
  */
  reorderBuffer(perm: Uint32Array): void;
  /**
  * Copy-on-write child sharing the folded state and buffered slabs,
  * ordered on `on`.
  */
  fork(on: Pipeline): RsWorkingSet;
}
