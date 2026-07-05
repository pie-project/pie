/** @module Interface pie:core/working-set **/
export type Error = import('./pie-core-types.js').Error;
/**
 * Shared by both working-set kinds. A contiguous, half-open span
 * [start, start + len) of relative page slots inside a working set's dense
 * ordered array. `start`/`len` are in page-slot units. The ONLY references
 * that ever cross this API are these relative indices into the array —
 * never physical page ids, never vpage ids.
 */
export interface PageRange {
  start: number,
  len: number,
}

export class KvWorkingSet {
  /**
  * Fresh, empty working set bound to `model` (and thus its device /
  * unified arena). The model is captured at construction so the
  * structural mutators (`alloc`/`free`/`slice`/`append`/`fork`) can
  * resolve the arena without a forward pass; `slice`/`fork` inherit this
  * model. In a single-model runtime every set binds to the single
  * bound model.
  */
  constructor()
  /**
  * Current number of page slots in the array.
  */
  size(): number;
  /**
  * Monotonic counter bumped on every STRUCTURAL mutation (alloc, free,
  * reorder, append). Lets the SDK/inferlet detect that cached relative
  * indices were invalidated by a compaction/reorder before it submits a
  * forward pass, so a stale-index use surfaces as a clean caller-side
  * error rather than silent corruption. (CoW page writes do NOT bump it.)
  */
  generation(): number;
  /**
  * Tokens per KV page for this working set's model/driver.
  */
  pageSize(): number;
  /**
  * Append `n` fresh page slots to the end of the array; returns the
  * contiguous relative range that was added. `alloc(0)` returns an
  * empty range at the current end.
  */
  alloc(n: number): PageRange;
  /**
  * Remove the slots at `indices` and densely compact the array.
  * `indices` are interpreted against the array AT CALL TIME; all
  * removals are applied together (call-time dense compaction). Indices
  * that follow a removed entry are invalidated afterwards — the caller
  * recomputes its own bookkeeping. Out-of-range or duplicate indices
  * return `error` (the call never traps).
  */
  free(indices: Uint32Array): void;
  /**
  * Reorder the array by the full bijection `perm` over `0..size`: new
  * slot `i` takes old slot `perm[i]`. `perm` must list every current
  * index exactly once; otherwise returns `error`.
  */
  reorder(perm: Uint32Array): void;
  /**
  * Create a new working set whose array is the slots in
  * [start, start + len), sharing those page objects by reference (lazy
  * CoW; first divergent write copies). Out-of-range spans return `error`.
  */
  slice(start: number, len: number): KvWorkingSet;
  /**
  * Append the slots of `other` onto the end of this array, sharing the
  * page objects by reference (lazy CoW).
  */
  append(other: KvWorkingSet): void;
  /**
  * Fork into a new working set that shares all current page objects by
  * reference (lazy CoW; first divergent write copies).
  */
  fork(): KvWorkingSet;
}

export class RsWorkingSet {
  /**
  * Fresh, empty RS working set bound to `model` (device / arena),
  * captured at construction so `alloc-buffer`/`fork`/`fold` resolve the
  * arena without a forward pass; `fork` inherits this binding.
  */
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
  */
  allocBuffer(n: number): PageRange;
  /**
  * Remove buffered slots at `indices` and densely compact (call-time
  * interpretation, same rules as `kv-working-set.free`). Invalid or
  * duplicate indices return `error`.
  */
  freeBuffer(indices: Uint32Array): void;
  /**
  * Reorder the buffered slots by the full bijection `perm`.
  */
  reorderBuffer(perm: Uint32Array): void;
  /**
  * Fork into a new RS working set sharing the folded state and buffered
  * slabs by reference (lazy CoW; first fold/write copies).
  */
  fork(): RsWorkingSet;
}
