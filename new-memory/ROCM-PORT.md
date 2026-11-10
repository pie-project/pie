# ROCm on the chunk pool: what serves today, and what the kernel port needs

State on new-memory after `sc/l40s-rocm`. Written on a machine with no ROCm
toolchain and no AMD GPU: `cargo check -p engine-cuda --features rocm` and
`cargo check -p kernels-rocm --features rocm --tests` pass, and nothing in
`crates/kernels-rocm/kernels/*.hip` was compiled or run (hiprtc compiles those
at run time). No `.hip` file was edited.

## What the ROCm build does now

The HIP kernels still address memory the way they did before the chunk pool:

- a kv plane's pages lie one right after another (page `q` of a plane at
  `plane + q * page_size * row`);
- a recurrent row lies whole in one slot, slots `slot_stride_elems` apart;
- an embedding table's rows are contiguous.

So a ROCm build lays the chunk pool the one way those kernels can read it
(`store.rs`, `PACKED_KERNELS` and `pack`; `layout.rs`, `Layout::pack`):

- Each kv shape (the kv list, the window list) steps its page ids by one
  plane's packed page instead of by `gcd(chunk, slot)`. A plane's pointer
  carries its offset inside the slot, so page id `q` times the packed page
  lands on the slot. This needs every plane read through the shape's page
  table (keys, values, compressor state) to have the same packed page, and a
  slot and a chunk to be whole numbers of it.
- A recurrent row's place in its slot is folded into the slab pointer, and
  the row is stated as one piece (`Segments::ONE`).
- No table streams by its rows (`layout::streamed_rows` is empty): the
  weight cache holds no shape, the embedding table stays dense on device.

A load that cannot be laid so is refused at load with the reason:

- kv planes of different packed page sizes in one shape (mixed head widths,
  index keys beside pools, compressor state beside kv rows);
- a slot or a chunk that is no whole number of the packed page;
- a recurrent row cut into more than one piece.

Wrappers in kernels-rocm also refuse what their kernels cannot address, should
anything else call them: `conv::cache_gather` / `cache_store` (a pieced or
segmented record, slots further apart than a row), `attn::compressed::pool`
`gather` / `state_write` (state pages not packed), `gather::embed_tabled` (a
located row table).

### Checked on CUDA

The same layout can be forced on a CUDA build with the diagnostic
`packed-pages` (`PIE_GOLDEN_PACKED=1` for the golden probe). On the L40S,
`golden_logits` under it:

- bit-identical to base: qwen3-micro-text, gemma4-e4b-mini-l6 (window list),
  qwen35-tiny (recurrent record), nemotron3-nano-mini, mimo26-flash-mini;
- refused at load, planes of mixed packed pages: dsv4-flash-mini,
  dots3-note-mini, minimax-m3-mini.

That checks the engine's arithmetic (page ids, plane offsets, the folded
record offset) against kernels that honour a stated stride. It does not check
that the HIP kernels read the same addresses: that needs AMD hardware.

### Not checked at all

- Any HIP kernel on the new layout.
- The rs buffer (rs-list slots) and whole-slot copies on ROCm.
- Slot-ref mode, evacuation and segment release on ROCm.
- Whether the models refused above served on ROCm before the chunk pool;
  if they did, they are a regression until the kernel port below.

## What the kernel port needs (option A)

Each item mirrors a kernels-cuda change; the CUDA commit is the reference.

1. **Page stride** (`401269dd4`, `4e9dabc9b`; `KvPool::page_stride`, bytes
   between two pages' rows of a plane, per plane). Every page address in the
   HIP kernels is `page * page_size * row`; each becomes
   `page * stride + slot * row`. Sites:
   - `attn/kv.hip` `kv_at` (and its 14 callers across `kv.hip`,
     `paged.hip`, `rel.hip`, `score.hip`, `replay.hip`);
   - `attn/qkv_fused.hip` (its own NHD/HND offset);
   - `attn/kv_envelope.hip` (page base);
   - `attn/pool.hip` (`page * page_size + pos` twice: entry rows and state);
   - `attn/mla.hip` (two slot computations);
   - `attn/index.hip` (key page base);
   - `attn/sdpa_selected.hip` (slot, and a slot table built as element
     indices, which becomes byte offsets or page/slot pairs);
   - `elemwise/rope.hip` (the paged write).
   The Rust wrappers pass `pages.page_stride` where they pass `page_size`.
2. **Compressor state stride** (`537c2f0f1`; `pool::gather`,
   `pool::state_write`, the fused compress): `state_at(page, off, page_size,
   pitch, stride)` in place of `(page * page_size + off) * pitch`.
3. **Recurrent rows through pieces and a segment table** (`2fd2b2c20`,
   `c474e3d29`; `Segments`, `StateRows`, `prim/state_rows.cuh`). Sites:
   `ssm/mixers.hip`, `conv/cache.hip`, `gather/ngram.hip`, `attn/replay.hip`
   and their wrappers in `ssm.rs`, `conv/cache.rs`, `gather/ngram.rs`. A row
   is `pieces` pieces, each in a chunk named by its own id, `piece_off`
   elements in; `piece_atoms` where the last piece is padded.
4. **Generated kernels' paged store** (`ff5e836f8`; `Store.page_stride`,
   `store_row` in `prim/store.cuh`): the HIP skeletons' paged epilogue.
5. **Row table for the embedding gather** (`5bcc67884`; `RowTable`, per-block
   hit counters): `gather/rows.hip` `embed`. With it, drop the early return
   in `layout::streamed_rows`.
6. **Expert tables** (`ba9f6b9f2` and after): kernels-rocm already carries
   `ExpertTable` as a refusing stub; the routed selects need the table read.

Once 1 to 3 run on AMD hardware, remove `PACKED_KERNELS`, `pack` and the
wrapper refusals; the `packed-pages` diagnostic can go with them.

Tests to port alongside: kernels-cuda `common/segments.rs`, `ssm/segments.rs`,
`conv/frame_cache.rs`, the spread-page cases of the paged kv tests
(`4e9dabc9b`), `elemwise/synth_paged.rs`, `gather/embed_table.rs`.
