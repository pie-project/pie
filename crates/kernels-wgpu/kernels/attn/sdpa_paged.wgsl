// Paged attention, read side: decode and tiled prefill from one body.
//
// The page-table arithmetic is Metal's, unchanged: request -> CSR page list ->
// physical page + in-page offset, then NHD element addressing. Eleven storage
// buffers, which is over WebGPU's guaranteed floor of eight -- that is the
// HOST's problem (it must request the adapter's real limits rather than
// `downlevel_defaults`, see `over_downlevel_storage_limit`) and not a reason to
// bind fewer. Six of the eleven are the fire's own tables and the ROW is the
// only place they are written down. Read that row through
// `kernels_wgpu::bindings` rather than guessing: the deleted `dump_layout`
// example only printed the same derivation. The numbers below are NOT Metal's,
// because this backend sends the row's five interleaved scalars to a uniform
// block and the buffer run closes up around them: `attention_mask` is the row's
// thirteenth operand and this file's binding 8.
//
// `_p32` and `_sg8` are ABI points inherited from Metal's table, not claims
// about hardware. `_p32` compiles the page arithmetic against a page size of
// 32 (a shift instead of a division); `_sg8` names a subgroup width this body
// does not read, because nothing here is subgroup-shaped.
//
// One lane owns a channel PAIR: bf16 crosses as `array<u32>`, two values to a
// word, and a lane owning one channel would read-modify-write a word its
// neighbour writes at the same instant with no sub-word atomic to arbitrate.
// It also keeps every workgroup at or under WebGPU's guaranteed 256
// invocations -- at `d_512` this body runs exactly 256 lanes where a
// channel-per-lane body would ask for 512 and fail to create a pipeline.

//#include "common/bf16.inc.wgsl"
//#include "attn/sdpa_online.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> queries: array<u32>;
@group(0) @binding(1) var<storage, read_write> k_pages: array<u32>;
@group(0) @binding(2) var<storage, read_write> v_pages: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_: array<u32>;
@group(0) @binding(4) var<storage, read_write> position_ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> req_of_token: array<i32>;
@group(0) @binding(6) var<storage, read_write> kv_page_indices: array<u32>;
@group(0) @binding(7) var<storage, read_write> kv_page_indptr: array<u32>;
// `U8s` in the row, and WGSL has no eight-bit storage element any more than it
// has a sixteen-bit one -- the smallest is a `u32`. Both mask buffers are
// therefore four bytes to a word and a byte is a shift, the same divergence
// bf16 makes and for the same reason.
@group(0) @binding(8) var<storage, read_write> attention_mask: array<u32>;
@group(0) @binding(9) var<storage, read_write> attention_mask_enabled: array<u32>;
@group(0) @binding(10) var<storage, read_write> sinks: array<u32>;
//#if defined(PIE_SPLITK)
// THE SPLIT'S PARTIAL SOFTMAX STATES, and the only buffer in this file that
// belongs to neither the statement nor the model.
//
// It is the FIRE's, handed over as `keys::AttnScratch` the way the page CSR
// and the rope ladder are, because a split of the key range is a decision
// this backend makes about ITS occupancy and no authored trace should have to
// know it happened. `f32` and not bf16: these are a running maximum and a
// denominator, and rounding them to eight mantissa bits before the merge
// would lose exactly the precision the online recurrence exists to keep.
//
// `2 + PIE_HEAD_DIM` floats per (row, query head, split), laid out in that
// order so that one split's whole state is contiguous and the merge below
// walks it with a stride it can hoist.
@group(0) @binding(11) var<storage, read_write> pie_split_state: array<f32>;
//#endif

// The row's scalars in ROW order, at 0, 4, 8, 12, 16 and 20 -- and 24 under
// `PIE_TILED`. All four bytes wide, so this block is the one place in the
// family where the naive sum of widths happens to be the right answer --
// `kv_write`'s is not, and the difference is `Usize`.
struct Params {
    gqa_factor: i32,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask_stride: u32,
    window: i32,
//#if defined(PIE_TILED)
    // `sdpa_paged_tiled` and `sdpa_paged_tiled_sink` STATE this one: it is
    // `Source::Rows`, the eighteenth and last operand of a row that is
    // otherwise `sdpa_paged_decode`'s seventeen, so offset 24 here is the row's
    // arithmetic and not a convention. It carries the fire's TRUE row count:
    // the grid rounds the rows up to whole tiles -- see
    // `kernels::LaunchRule::SdpaTiled` -- so a partial last tile's threads sit
    // past the end, and this is what tells them so.
    n_rows: i32,
//#if defined(PIE_STRIDED)
    // These two are appended here in the order `kernels-vulkan`'s push block
    // states them, which is the order `attn::sdpa_paged_tiled_strided`'s
    // signature takes them in. Nothing derived them: the row named no operands
    // for this variant, so the lowered plan's own argument order was the only
    // description, and the routine is that order written down. The same
    // reasoning fixes `n_rows` above at 24 for THIS variant; every other
    // variant agrees with it, which is why one struct
    // serves both.
    q_row_pitch: i32,
    o_row_pitch: i32,
//#endif
//#endif
//#if defined(PIE_SPLITK)
    // How many ways the key range is cut. At offset 24, which is where
    // `PIE_TILED`'s `n_rows` sits -- the two are mutually exclusive arms and
    // neither variant compiles the other's scalar, so the offset is reused
    // rather than shared.
    //
    // A COUNT AND NOT A LENGTH. The kernel cannot be told how many keys a row
    // has: that is the page CSR's answer and it is per REQUEST, while a fire
    // launches one grid for every row it carries. So the host states how many
    // slices to cut and each workgroup derives its own from the row's own
    // `q_pos`, which means a short row simply leaves some splits empty. An
    // empty split writes the identity of the merge -- `PIE_SDPA_NEG_INF`, a
    // zero denominator and a zero accumulator -- so the combine needs no
    // count of which ones were live.
    splits: i32,
//#endif
}
@group(1) @binding(0) var<uniform> params: Params;

// One lane per output word.
const PIE_PAIRS: u32 = PIE_HEAD_DIM / 2;

// The bf16 half-index unpack, per buffer. `pie_load_bf16(&queries, i)` is the
// shared answer and cannot be CALLED: its `ptr<storage, array<u32>, read>`
// parameter is WGSL's `unrestricted_pointer_parameters`, which naga does not
// implement, so a module that calls it parses and then fails
// `create_shader_module`. The CONVERSION keeps one definition in
// `common/bf16.inc.wgsl`; only the address arithmetic is restated.
fn q_at(i: u32) -> f32 {
    let word = queries[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn k_at(i: u32) -> f32 {
    let word = k_pages[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn v_at(i: u32) -> f32 {
    let word = v_pages[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn sink_at(i: u32) -> f32 {
    let word = sinks[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

// The diagnosis is `kernels-metal`'s: this reloads `kv_page_indices`, and the
// address of the key load that follows depends on what comes back, so a loop
// over positions is a chain of dependent round trips. Metal measured its own
// version over Llama-3.2-1B at 1024 of context -- 1.033 ms for sixteen
// launches reading 2 MB each, 32 GB/s on a 400 GB/s part -- and fixed it by
// striding the PAGES rather than the positions.
//
// HALF OF IT APPLIED HERE, and the other half measured and declined.
//
// * `kv_page_indptr[req]` is loop-invariant and was the FIRST of the two
//   dependent loads. It is hoisted out of both key loops now
//   ([`page_slot_at`]), which removes a load per position and shortens the
//   chain from three links to two. Worth about 1.5 ms of a 512-context
//   decode.
//
// * Caching the PHYSICAL PAGE across the `page_size` consecutive positions
//   that share it was written, measured and reverted. Metal's loop is
//   simdgroup-STRIDED, so its consecutive iterations are different pages and
//   the reload is unavoidable without restructuring; these loops are
//   SEQUENTIAL, so the repeated load is the same address every time and
//   whatever is between the shader and the memory was already absorbing it.
//   The change measured inside the run-to-run noise.
//
//   That is only knowable by measuring, and by measuring more than once: the
//   same binary gave 22.36 ms and 23.77 ms on consecutive runs of
//   `what_a_decode_costs_at_length`, which is larger than the effect being
//   looked for. The first reading said it helped.
fn page_slot(req: i32, kp: i32) -> u32 {
    return page_slot_at(kv_page_indptr[u32(req)], kp);
}

// The same, with the request's page-table base already in hand.
//
// `kv_page_indptr[req]` is LOOP-INVARIANT and was being loaded once per key
// position, as the first of two DEPENDENT loads: the index of the page could
// not be fetched until the base came back, and the key could not be fetched
// until the index did. Hoisting the base out of a key loop removes one load
// per position and shortens the chain from three links to two.
// # THE INTEGER DIVIDE IN HERE COSTS NOTHING, AND THE PROBE THAT SAID IT DID
// # WAS READING DIFFERENT MEMORY
//
// The generic arm below runs `kp / params.page_size` and `kp % params.page_size`
// once per key on all 256 lanes of the split kernel -- the kernel every decode
// actually runs, since `attn.rs`'s `workgroups < 128` test sends it there. An
// integer divide by a runtime value is tens of instructions where a shift is
// one, so this looked like the obvious thing in a kernel measured at 92 us a
// fire against 0.64 ms of traffic.
//
// It was priced by adding `PIE_PAGE_SIZE=32` to the split points, which takes
// the `_p32` arm: 9.828 -> 9.463 ms, 101.8 -> 105.7 tok/s, a clean 3.8%. The
// suite was not run, because the point of the probe was the clock.
//
// THAT NUMBER WAS AN ARTIFACT. This engine's page size is 16, not 32. Baking 32
// in does not just remove a divide, it changes every address the loop computes:
// `kp >> 5` indexes half as far into `kv_page_indices` and `phys * 32` lands
// somewhere else entirely. The probe was faster because it read a smaller,
// denser region of memory, and it was reading the wrong keys while it did.
//
// Done correctly -- a `_pow2` point taking `firstTrailingBit(page_size)` as the
// shift and `page_size - 1` as the mask, which is correct for any power of two
// and therefore for 16 -- the answer is **9.837 ms against a 9.828 baseline**.
// Confirmed firing with `PIE_WGPU_PROBE=1 PIE_WGPU_DUMP=470`, which named
// `sdpa_paged_decode_split_bfloat16_d_128_pow2` on all 28 launches. Zero. The
// divide is hidden behind the two dependent loads that follow it, and Apple's
// integer unit is not what this loop is waiting on.
//
// So the `_pow2` points were written, measured and removed, and the `_p32`
// points above stay unfired. Two things to take from it:
//
//   * A WRONG-ANSWER PROBE CAN BE FASTER FOR THE WRONG REASON. Every other
//     probe in this tree removes work while leaving the addresses alone -- half
//     the dot's words, duplicate a fire, add a second reduction tree. This one
//     moved the addresses, which is a different experiment than the one
//     intended. Prefer a probe that changes the arithmetic and not the memory,
//     and if it must change the memory, run the suite before believing it.
//   * The reduction tree is the same story: a second full tree per block, six
//     more barriers, measured 9.828 -> 9.964, so the whole tree is 0.14 ms and
//     5% of this kernel. Neither the barriers nor the integer unit is where the
//     92 us goes.
fn page_slot_at(page_base: u32, kp: i32) -> u32 {
//#if defined(PIE_PAGE_SIZE) && PIE_PAGE_SIZE == 32
    // The `_p32` points: a 32-entry page is a shift and a mask, and the
    // division this replaces is the inner loop's only integer divide.
    let page_ix = u32(kp) >> 5u;
    let page_off = u32(kp) & 31u;
    let phys = kv_page_indices[page_base + page_ix];
    return phys * 32u + page_off;
//#else
    let page_ix = u32(kp / params.page_size);
    let page_off = u32(kp % params.page_size);
    let phys = kv_page_indices[page_base + page_ix];
    return phys * u32(params.page_size) + page_off;
//#endif
}

// One byte out of a `u32` array. Module-scope storage is addressable from any
// function here, so these take the row index rather than a pointer -- a
// `ptr<storage, ...>` parameter is a WGSL language extension `naga` does not
// implement.
fn mask_enabled(row: u32) -> bool {
    let word = attention_mask_enabled[row >> 2u];
    return ((word >> ((row & 3u) * 8u)) & 0xffu) != 0u;
}

fn mask_allows(at: u32) -> bool {
    let word = attention_mask[at >> 2u];
    return ((word >> ((at & 3u) * 8u)) & 0xffu) != 0u;
}

fn keeps(row: u32, kp: i32, q_pos: i32, start: i32) -> bool {
    if (kp > q_pos || kp < start) { return false; }
    if (mask_enabled(row)) {
        // The mask's own stride bounds it: a request whose history is longer
        // than the mask the fire supplied has no entry to read, and reading
        // past the row would pick up the NEXT row's mask.
        if (u32(kp) >= params.attention_mask_stride) { return false; }
        if (!mask_allows(row * params.attention_mask_stride + u32(kp))) { return false; }
    }
    return true;
}

fn q_base_for(row: u32, q_head: u32, n_q_heads: u32) -> u32 {
//#if defined(PIE_TILED) && defined(PIE_STRIDED)
    var base = row * n_q_heads * PIE_HEAD_DIM;
    if (params.q_row_pitch > 0) { base = row * u32(params.q_row_pitch); }
    return base + q_head * PIE_HEAD_DIM;
//#else
    return (row * n_q_heads + q_head) * PIE_HEAD_DIM;
//#endif
}

fn o_base_for(row: u32, q_head: u32, n_q_heads: u32) -> u32 {
//#if defined(PIE_TILED) && defined(PIE_STRIDED)
    var base = row * n_q_heads * PIE_HEAD_DIM;
    if (params.o_row_pitch > 0) { base = row * u32(params.o_row_pitch); }
    return base + q_head * PIE_HEAD_DIM;
//#else
    return (row * n_q_heads + q_head) * PIE_HEAD_DIM;
//#endif
}

fn dot_row(q_base: u32, k_base: u32) -> f32 {
    var acc = 0.0;
    // A WORD at a time, which is two channels, and NOT a different sum.
    //
    // `q_at(i)` and `k_at(i)` each load `buf[i >> 1]` and then select a half,
    // so a scalar loop over `PIE_HEAD_DIM` issues two loads per channel and
    // loads every word TWICE -- 512 loads at `d_128` where 128 would do. Both
    // bases are a multiple of `PIE_HEAD_DIM` and that is even, so a word is
    // exactly two consecutive channels of one row and the halves are the even
    // and odd channel in that order.
    //
    // The multiply-adds below are kept as SEPARATE `acc = acc + ...`
    // statements in the original order. Folding them into one expression
    // would associate the pair before the running sum, which is a different
    // f32 rounding, and this kernel's answers are walked against
    // `kernels-metal` and `kernels-vulkan` NUMBER BY NUMBER. The scale stays
    // per term for the same reason -- hoisting it out is also a rounding.
    //
    // # Two words an iteration, and two is the largest this tree can take
    //
    // The loop is unrolled by two: both words are LOADED, then the four
    // multiply-adds run in the source order above. The order is untouched, so
    // this is not a re-association at the WGSL level -- the win is that two
    // independent loads are in flight where one was. That is worth 10.6% of a
    // 512-row prefill (1444 -> 1597 tok/s), which says the dot was waiting on
    // memory ISSUE, not on memory bandwidth and not on the dependent chain
    // through `acc`.
    //
    // Unrolling by FOUR is worth 17% (1690 tok/s) and this tree cannot take
    // it. It breaks `a_two_level_prefix_tree_reads_what_a_seat_that_never_
    // forked_reads`: a forked leaf drifts 2.05% of the row's peak from an
    // unforked seat that heard the same tokens, against a 0.79% bar. Metal
    // compiles MSL with fast-math, so it is free to re-associate whatever the
    // source says, and at four words wide it vectorises -- differently for
    // different variants, so two batch shapes of the SAME prompt stop
    // agreeing. Carrying four explicit accumulators instead measures the same
    // 1690, which is the proof: at width four the compiler had already made
    // the four partial sums by itself.
    //
    // So the limit here is not correctness in the abstract, it is that a
    // serving system must answer a prompt the same way regardless of who else
    // is in its batch. Two words is where that still holds. Re-check the
    // batch-dependence guard, not just the suite, before widening it.
    //
    // Every instantiated `PIE_HEAD_DIM` is 64, 128, 256 or 512, so `PIE_PAIRS`
    // is 32, 64, 128 or 256 and the step of two always divides it. A new head
    // dimension with an odd `PIE_PAIRS` would run off the end of the row.
    let q_word = q_base >> 1u;
    let k_word = k_base >> 1u;
    for (var w = 0u; w < PIE_PAIRS; w = w + 2u) {
        let q0 = queries[q_word + w]; let k0 = k_pages[k_word + w];
        let q1 = queries[q_word + w + 1u]; let k1 = k_pages[k_word + w + 1u];
        acc = acc + params.scale * pie_bf16_to_f32(q0 & 0xffffu) * pie_bf16_to_f32(k0 & 0xffffu);
        acc = acc + params.scale * pie_bf16_to_f32(q0 >> 16u) * pie_bf16_to_f32(k0 >> 16u);
        acc = acc + params.scale * pie_bf16_to_f32(q1 & 0xffffu) * pie_bf16_to_f32(k1 & 0xffffu);
        acc = acc + params.scale * pie_bf16_to_f32(q1 >> 16u) * pie_bf16_to_f32(k1 >> 16u);
    }
    return acc;
}

// STAGING THE QUERY IN WORKGROUP MEMORY WAS TRIED AND IS A LOSS. DO NOT REDO IT.
//
// The loop above re-reads all `PIE_PAIRS` words of the query for every key,
// and the query does not change across that loop -- at `d_128` with a 512-row
// prompt that is 64 words read ~256 times per (row, lane). It reads like the
// largest single term in the loop's traffic and like pure repetition.
//
// It is neither. Staging the group's 32 query rows into a `var<workgroup>`
// array once per workgroup, filled flat and coalesced across all lanes with
// one barrier, and reading the dot's `q` from there instead, measured on an
// M4 Pro against the same fixture that produced the table in
// `what_a_prefill_costs_at_length`:
//
// | rows | 32 | 64 | 128 | 256 | 512 |
// | --- | --- | --- | --- | --- | --- |
// | global Q (this file) | 40.0 | 56.5 | 94.2 | 179.3 | **356.7** |
// | staged Q | 40.8 | 56.8 | 98.6 | 187.9 | **414.2** |
//
// Sixteen percent SLOWER at 512, and slower at every length. Two reasons, and
// the second is the one to carry forward:
//
//  * A row's query is 256 bytes at `d_128`. Every one of those "repeated"
//    reads was already an L1 hit, so the stage removes no memory traffic --
//    it moves a hit from one fast path to another. This is exactly what the
//    page-table cache above found, for exactly the same reason, and it is now
//    twice-confirmed: NOTHING IN THIS KERNEL'S INNER LOOP IS FAR FROM THE
//    CORE. Do not price a load in this file by counting how many times it is
//    issued.
//
//  * The stage costs 8 KB of workgroup memory at `d_128`, and that is a
//    charge with no offsetting credit. Fewer workgroups fit a core at once,
//    so there is less to overlap the loop's latency with, and this kernel is
//    latency-bound (see below). A staging that buys nothing still costs
//    occupancy, so it comes out behind rather than level.
//
// This narrows what the attention's 13x deficit against this backend's own
// scalar GEMM can be. It is not redundant traffic. The remaining candidate is
// the DEPENDENCE of the loop: `acc` above is a single accumulator threaded
// through 128 multiply-adds, and `max_score`/`sum_exp` are a serial chain
// across every key. The GEMM in `qmm_t.wgsl` hit the same wall and cleared it
// by carrying several accumulators at once -- see the prefill-GEMM unrolling
// in this tree's history.
//
// The blocker there is not performance, it is parity. Splitting `acc` re-
// associates the sum, and this dot is walked term by term against
// `kernels-metal` and `kernels-vulkan`; a different association is a
// different f32. Any attempt has to move all three backends together, which
// is why it is written down here rather than attempted.

//#if !defined(PIE_TILED)

// One (row, head), computed by the whole workgroup together.
//
// # Why the two arms have separate bodies
//
// They used to share `compute_one`, one call per output pair, and neither arm
// wanted that: this one because the whole workgroup was recomputing one
// scalar, the tiled one because each lane was. The two fixes are different
// shapes -- a barrier tree here, a hoisted pair loop there -- so the shared
// body is gone rather than parameterised.
//
// The tiled arm cannot barrier at all: one workgroup there spans thirty-two
// different rows with different positions, windows and masks, and its lanes
// `continue` past rows that are not theirs. A `workgroupBarrier()` reached by
// some lanes and not others is a hang. This arm is the opposite shape -- one
// workgroup is one row and one head -- so `q_pos`, `start` and every `keeps`
// answer are uniform and a reduction is safe by CONSTRUCTION. That is checked
// rather than argued: WGSL's uniformity analysis rejects a barrier in
// non-uniform control flow, so the module compiling at all is the proof.
//
// # What it is worth, measured on THIS kernel and nothing else
//
// `how_long_a_decodes_kernels_take`, which times the same work at two repeat
// counts and subtracts, so pipeline creation and submission fall out:
//
// | keys | one dot per lane | this | |
// |---|---|---|---|
// | 256 | 1.443 ms | 0.204 ms | 7.1x |
// | 2048 | 11.722 ms | 1.420 ms | 8.3x |
//
// And it is the kernel that matters. A Qwen3-0.6B decode is 28 layers, so at
// 2048 keys this is 28 x 1.42 = 40 ms against about 3.4 ms for all 196
// projection dispatches put together (`affine_qmv_fast` measures 0.021 to
// 0.056 ms at this model's shapes). **Attention IS the decode** at any real
// context length, which is why the sibling bothered.
//
// # This was reverted once, on a measurement that was noise
//
// The first attempt timed it through `wgpu_many_conversations` -- the whole
// engine, on a SHARED machine -- read 601.9 s against 342.8 s, and took the
// change out as a "1.75x regression". Three runs of one identical binary on
// that gate gave 586 s, 343 s and 357 s: the spread WAS the result. The bench
// exists so the next person measures the kernel instead of the machine, and
// its own first version measured the SHADER COMPILER instead of the kernel,
// which is recorded there.
// One partial dot per lane. `PIE_PAIRS` is `PIE_HEAD_DIM / 2` and every
// instantiated head dim is a power of two, so the tree halves exactly and
// needs no odd-lane tail.
// Keys carried by one pass of the reduction, and the workgroup's y extent.
//
// WebGPU guarantees 256 invocations per workgroup, and `PIE_PAIRS` of them go
// on the head dimension, so what is left is what can go on the KEYS: 8 at
// d_64, 4 at d_128, 2 at d_256 and 1 at d_512, where the shape degenerates to
// the one this arm had.
// AND 256 IS AN OPTIMUM, NOT JUST A GUARANTEE. Swept against `PIE_SPLITS` so
// that the TOTAL invocation count is held constant -- the grid is one workgroup
// per query head per split, so narrowing the group without widening the split
// axis just removes threads and proves nothing. p50 means, interleaved:
//
// | invocations/group | 64 | 128 | **256** | 512 | 1024 |
// | --- | --- | --- | --- | --- | --- |
// | splits | 32 | 16 | **8** | 4 | 2 |
// | ms | 13.40 | 10.54 | **9.49** | 10.46 | 10.82 |
//
// A real interior peak, 10% clear on both sides, and it lands exactly on the
// number WebGPU guarantees -- so the portable shape and the fast shape are the
// same shape here and no tier is wanted. (`maxComputeInvocationsPerWorkgroup`
// is 1024 on this part; the two right-hand columns are what asking for it
// buys, which is nothing.)
//
// Both sides of the peak have a reason and they are different reasons, which is
// why it is a peak. Narrower groups put fewer keys in flight per rendezvous, so
// the same 63-add tree is amortised over less; wider ones make every
// `workgroupBarrier()` a rendezvous of more simdgroups and hold more of the
// core's state per group, which is the residency finding above arriving from
// the other direction.
// LANES ON THE CHANNEL AXIS, AND WHY THIS IS NOT `PIE_PAIRS`.
//
// A row of `PIE_HEAD_DIM` channels is `PIE_PAIRS` bf16 pairs, and this kernel
// gave each pair its own lane. At `d_128` that is 64 lanes to a row against a
// 32-lane simdgroup, so a row's dot straddles TWO subgroups -- and a butterfly
// cannot cross a subgroup, so the `@subgroup` arm still had to put both block
// sums through workgroup memory and fence them, twice per key block.
//
// Priced before it was fixed, by the truncation probe this file now trusts:
// delete the staging, the two `workgroupBarrier()`s and the cross-block fold,
// take the wrong answer, and read the token. Three interleaved rounds at 200
// samples:
//
// | | shipped (ms) | no cross-block fold (ms) |
// | --- | --- | --- |
// | | 7.789 | 7.374 |
// | | 7.764 | 7.346 |
// | | 7.719 | 7.305 |
// | mean | **7.757** | **7.342** |
//
// **0.42 ms, 5.4%**, and the two distributions do not touch. That is the
// price of the last rendezvous in the decode's largest kernel, and the only
// way to collect it is to make a row FIT in a subgroup.
//
// So a lane owns `PIE_DP` pairs instead of one and a row is `PIE_DX` lanes.
// 32 is the simdgroup width on every part this backend has run on, and where
// a row is already narrower than that -- `d_64`, whose 32 pairs are exactly
// one subgroup -- `PIE_DX` is `PIE_PAIRS` and `PIE_DP` is 1, which is the
// shape this kernel already had. Nothing at `d_64` moves.
//
// The bound on `PIE_DP` is the ACCUMULATOR, exactly as it is for the tiled
// arm's `PIE_LANE_PAIRS` below: a lane holds `PIE_DP` `vec2<f32>` live across
// the whole key loop, and another `PIE_KR * PIE_DP` for the values it has
// loaded but not yet scaled. At `d_512` that is 8 and 8, which is the same
// register bill `PIE_TX = 16` already pays on the prefill.
const PIE_DX: u32 = min(PIE_PAIRS, 32u);
const PIE_DP: u32 = PIE_PAIRS / PIE_DX;
// AND THE WORKGROUP DOES NOT CHANGE SHAPE, WHICH IS THE POINT.
//
// The obvious way to spend the saving is to declare `@workgroup_size(PIE_DX,
// 256 / PIE_DX)` and put the freed invocations on the y axis. That would have
// cost two invariants this tree is not willing to sell. The host computes this
// grid a SECOND time in `kernels-wgpu::attn` and a THIRD in
// `driver-wgpu::geometry`'s `Rule::SdpaVector`, and the rule's whole guard is
// that TWICE a module's workgroup width is the head width it was built for --
// which is what stops a `_d_256` module being handed a 128-wide head and
// answering plausible nonsense. Cap the width at 32 and `d_128`, `d_256` and
// `d_512` all declare `(32, 8)`, so the guard cannot tell them apart and two
// host copies of the arithmetic have to move with the shader.
//
// So the group stays `PIE_PAIRS x PIE_KB` and the X AXIS IS REINTERPRETED
// instead: `PIE_DX` of its lanes are the channel lanes and the remaining
// `PIE_DP` slices of it are MORE KEYS. `lane % PIE_DX` is the channel and
// `lane / PIE_DX` is a second key index that composes with `ky`, so a block
// carries `PIE_KB * PIE_DP` keys where it carried `PIE_KB`. Identical
// invocation count, identical work per invocation, identical grid, and the
// module still declares the width the guard reads.
//
// It is also what makes the butterfly correct at every subgroup width without
// an alignment argument of its own. The linear invocation index is
// `lid.x + lid.y * PIE_PAIRS`, so a `PIE_DP` slice of x is an ALIGNED
// `PIE_DX`-lane block of it, and a butterfly over `lim = min(PIE_DX, sg)`
// cannot leave one -- whether the subgroup is 16 lanes, 32, or 64 straddling
// two slices.
const PIE_KB: u32 = 256u / PIE_PAIRS;
// Keys a block carries: the y axis times the key slices of x.
const PIE_KY: u32 = PIE_KB * PIE_DP;
// Blocks of keys one tree serves, on top of the `PIE_KB` the y axis already
// gives it. A rendezvous of every simdgroup in the workgroup is the cost this
// kernel pays most often, and a lane can carry several keys' partials as
// cheaply as one, so the tree folds `PIE_KR` blocks at a time and the barrier
// count falls by the same factor.
//
// `PIE_KB * PIE_PAIRS` is 256 at every instantiated head dim, so the score
// array is `PIE_KR` KiB everywhere, plus 2 KiB of merge state, against the
// 16352-byte floor `wgpu`'s downlevel defaults guarantee. Sixteen does not
// fit that floor at all.
//
// THIS IS A RESIDENCY LIMIT, NOT A BARRIER-COUNT ONE, which is the whole
// reason it is not simply as large as it fits. Apple's GPU splits a fixed
// threadgroup-memory budget between the workgroups it keeps resident on a
// core, and this kernel is only 32 workgroups wide to begin with -- one per
// query head -- so a workgroup that claims more memory takes away the only
// thing there was to run while a key load is in flight. Swept whole, every
// point correct at all 56 gpu tests:
//
//   PIE_KR   workgroup memory   tg128   tg256@2048
//        2              4 KiB   104.0         71.8
//        4              6 KiB   109.2         73.3
//        8             10 KiB   110.4         75.3
//       12             14 KiB    80.1         63.2
//
// Two is too few barriers-worth of work and twelve falls off the residency
// cliff, so the peak is interior and it is at EIGHT -- the most memory a
// workgroup can hold while two of them still fit a core.
//
// READ THE 2048 COLUMN MORE THAN ONCE. A single run at `PIE_KR = 4` read
// 77.4 there and would have carried the decision; four more runs of the same
// binary read 73.2, 73.5, 73.0 and 73.5. The long-context bench is one
// sequence and one sample, and its spread is several percent -- wide enough
// to invert this table's top two rows. The 512 column is stable to a tenth.
//
// This is the same occupancy wall `quant/qmv.wgsl`'s column count found from
// the other side. Do not raise either without measuring: the arithmetic
// argument says go up and the machine says no.
//
// # AND THE SPLIT ARM WANTS TWO, BECAUSE THE SWEEP ABOVE PREDATES SPLIT-K
//
// Read the residency argument again: "this kernel is only 32 workgroups wide
// to begin with -- one per query head -- so a workgroup that claims more
// memory takes away the only thing there was to run while a key load is in
// flight". That is why eight won. It is also a statement about a grid that no
// longer exists. `attn.rs`'s `workgroups < 128` branch sends every real decode
// down the SPLIT arm, whose grid is `q_heads x rows x splits` -- 128
// workgroups at this model's shapes, four times what the sweep above was
// choosing for. With four times the workgroups there is plenty to run while a
// load is in flight, and the 10 KiB each one claims stops buying anything and
// starts costing residency.
//
// Re-swept on the split arm, qwen3-0.6b, `what_a_decode_costs_at_length`:
//
// | PIE_KR | 1 | 2 | 4 | 8 |
// | --- | --- | --- | --- | --- |
// | ms @512 keys | 9.323 | **9.077** | 9.154 | 9.828 |
// | tok/s | 107.3 | **110.2** | 109.2 | 101.8 |
//
// The peak is interior again and it has moved from eight to TWO. One is too
// few -- a barrier every four keys instead of every eight -- which is the same
// shape of curve as before, shifted.
//
// THAT TABLE IS ONE RUN A POINT AND IT IS OPTIMISTIC. Taken as an interleaved
// A/B against the tree it replaces, four runs alternating, it is 9.894 and
// 9.851 at eight against 9.484 and 9.574 at two: **9.873 -> 9.529 ms, 101.3 ->
// 104.9 tok/s, 3.5%**. That is the number this change ships, and the 8.3% the
// single-run sweep implied is what a sweep taken in order looks like on a
// machine that drifts. The sweep still chose the right point; it just could
// not say by how much. Interleave before quoting.
//
// RE-CONFIRMED ON THE REPAIRED HARNESS, and this is the number to trust. The
// 3.5% above was two pairs on a bench that took 40 samples and repeated to
// only 3.1%, so the effect and the instrument's own noise were the same size.
// `serving.rs` now takes 200 and repeats to ~1.7%. Three interleaved pairs:
//
// | pair | 1 | 2 | 3 | mean |
// | --- | --- | --- | --- | --- |
// | KR 2, p50 ms | 9.641 | 9.691 | 9.692 | **9.675** |
// | KR 8, p50 ms | 9.952 | 10.042 | 9.967 | **9.987** |
//
// **3.1%**, and every run at two beats every run at eight with nothing
// between the two groups -- where five pairs on the old bench could not
// separate a 2.7% change at all. The absolute figures are higher than the
// ones above because a 200-sample run warms the part and reports its steady
// state; only compare within a block of this table, never across.
//
// At a longer context the two are level: at ~960 keys, 10.667 ms at eight
// against 10.687 at two, one run each. Each workgroup has more keys to walk there, so the
// block loop amortises what the barrier costs and residency stops being the
// binding thing. Neutral long, 8% short, so two.
//
// # WHY THIS AND NOT ANY OF THE FIVE THINGS TRIED FIRST
//
// The split kernel costs 92.1 us a fire against 0.64 ms of traffic, and every
// obvious explanation for the gap was measured and is zero:
//
// | probe | ms | verdict |
// | --- | --- | --- |
// | (baseline) | 9.828 | |
// | a second reduction tree per block | 9.964 | the tree is 0.14 ms, 5% |
// | the online softmax replaced by a plain add | 9.936 | the `exp` is free |
// | every K and V load issued twice | 9.744 | load ISSUE is free |
// | `kv_head = 0`, an 8x cut in unique traffic | 9.476 | 14%, not bandwidth |
// | `_pow2` page walk, no integer divide | 9.837 | the divide is free |
//
// Nothing INSIDE the loop costs anything, and yet the kernel scales with the
// key count -- 7.901 ms at a 128-key context against 9.828 at 512. A body
// whose every part is free but whose total is not is a body that is waiting,
// and what a workgroup waits on is decided by how many other workgroups the
// core can hold. That is `PIE_KR`, and it is the only knob in this file that
// the five probes above do not touch.
// AND THEN 2 -> 1, ON THE REPAIRED BENCH, WHICH REVERSES THE SWEEP ABOVE.
//
// The single-run sweep read 1 as 9.323 against 2's 9.077 and the note below it
// explains the loss -- "one is too few, a barrier every four keys instead of
// every eight". That reasoning was built on a number the old 40-sample bench
// could not produce. Three interleaved pairs at 200 samples:
//
// | pair | 1 | 2 | 3 | mean |
// | --- | --- | --- | --- | --- |
// | KR 1, p50 ms | 9.582 | 9.425 | 9.534 | **9.514** |
// | KR 2, p50 ms | 9.898 | 9.815 | 9.858 | **9.857** |
//
// **3.5%**, non-overlapping, in the direction the sweep said was uphill. Every
// interior peak this constant has ever shown was an artifact of taking one run
// a point in order; the curve is monotone and it wants the smallest state.
//
// AND IT DOES NOT REVERSE LONG, which the previous retune had to be careful
// about -- 8 against 2 was level at ~960 keys, so that change was neutral long
// and worth taking only short. This one is not. At a 1024-key context, two
// interleaved pairs:
//
// | | KR 1 | KR 2 |
// | --- | --- | --- |
// | p50 ms | **10.822 / 10.560** | 11.507 / 11.506 |
//
// **7.1%**, twice the gap at 512. The gap grows with the context because the
// attention's share of the token grows with it, so whatever residency buys is
// bought on more of the token. There is no length at which to prefer 2.
//
// # AND THE REGISTERS COUNT TOO, WHICH IS WHY 1 IS NOT A SURPRISE
//
// `PIE_KR` sizes two things at once and they were assumed to pull opposite
// ways: the staging array, which decides how many of these workgroups a core
// holds, and the depth of the load loop, which decides how many key loads are
// in flight before a lane must wait. Residency wants it small, latency hiding
// wants it deep, and the interior peak was read as the trade between them.
//
// So they were SEPARATED -- a `PIE_KL` holding `PIE_KL` keys in registers and
// draining them through a `PIE_KR`-deep array, identical barriers, identical
// arithmetic, identical fold order, the previous body exactly at
// `PIE_KL == PIE_KR`. Three interleaved rounds, p50 means:
//
// | PIE_KL | 2 | 8 | 16 |
// | --- | --- | --- | --- |
// | ms | **9.858** | 10.413 | 11.478 |
//
// Monotone and steep the wrong way: 16% worse at 16. There is no trade. Deeper
// flight does not help this kernel at all, and the registers it costs are
// themselves a residency cost, the same one the workgroup array is. Both knobs
// say the identical thing -- HOLD LESS STATE -- which is why the two of them
// bottom out together at one.
//
// That also closes the "what is it waiting on" question the six probes below
// opened. It is not waiting on memory it could have prefetched, because giving
// it eight loads in flight instead of two made it worse. It is waiting because
// too few of these workgroups fit on a core, and every byte of state a lane
// holds is what stops another one fitting. The code that measured this is
// reverted -- at `PIE_KL == PIE_KR` it is a no-op, and below that it cannot
// exist -- and only this table is kept.
//
// # WHAT THE DECODE ARM ACTUALLY COSTS, TAKEN APART WITHOUT AN INSTRUMENT
//
// Everything above prices this arm against the OTHER knob settings, which is
// what a sweep can do and no more. It never said how much of the 1.838 ms this
// pair of kernels spends in a token is work and how much is the launch, and
// the numbers that claimed to -- the `[cost]` shares from `PIE_WGPU_STAMP` --
// are taken on a token the instrument itself inflates from 7.4 ms to 13.8. So
// it was re-taken with the trick `quant/qmv.wgsl` uses: a uniform-false early
// return at the top of `decode_row`,
//
// ```wgsl
//   if params.page_size >= 0 { return; }
// ```
//
// which keeps every binding live (naga drops a binding whose last use goes,
// and the fire comes back `Unfired(Refused { Bindings { .. } })`), keeps all
// 128 workgroups launching and all 256 threads of each starting, and deletes
// the entire body. Plain wall-clock medians, three rounds, no instrument:
//
// ```text
//   configuration                             decode p50 (512 keys)
//   shipped                                        7.451 ms
//   `decode_row` body deleted, merge intact        5.993
//   PIE_WGPU_SKIP=sdpa_paged                       5.613
//   PIE_WGPU_SKIP=sdpa_paged_decode_merge          7.390
// ```
//
// Which decomposes the whole of the attention in a decode:
//
// ```text
//   split body       1.458 ms   19.6% of the token   28 fires   52.1 us each
//   split launch     0.270       3.6%                28          9.6 us each
//   merge, all of it 0.110       1.5%                28          3.9 us each
//   -------------------------------------------------------------------
//   sdpa_paged       1.838      24.7%
// ```
//
// Three things follow and none of them was known.
//
// **The merge is not worth looking at.** It is 1.5% of a token and 6% of the
// attention. `PIE_SPLITS` has now been swept four times and lands on 8 every
// time; this is why the eight-way fold it forces is affordable.
//
// **The launch is 9.6 us a fire, and that is the most expensive dispatch in
// the decode by a factor of three.** `qmv` fires 196 times for 1.085 ms of
// launch; `silu_mul` returns 1.4 us a fire and `kv_append` 2.6. The ordering
// is not by kernel, it is by GRID: this arm dispatches 128 workgroups of 256
// threads, the largest launch in the token, and `qmv.wgsl`'s three-mechanism
// table established that the ramp is charged PER WORKGROUP. 9.6 us for 128
// workgroups is 75 ns each, which is the same rate `qmv` pays. There is
// nothing anomalous here and nothing to reclaim without shrinking the grid --
// and the grid is `PIE_SPLITS` x `q_heads`, both of which are swept shut.
//
// **So the target is 1.458 ms of body and it is the second-largest single
// thing in the token, after `qmv`'s 2.80.** Priced in bytes it is not obviously
// bad: 16 q heads x 512 keys x 128 channels x 2 (K and V) x 2 B is 4.19 MB a
// layer and 117 MB a token, so 1.458 ms is **80 GB/s** against a 273 GB/s
// part. But the traffic is a fiction -- the UNIQUE footprint is 2.1 MB a
// layer, `gqa_factor` = 2 means every KV byte is read by two query heads, and
// the `kv_head = 0` probe (an 8x cut in unique bytes) buys nothing at all. All
// of it is a cache hit. The 80 GB/s is a LOAD-ISSUE rate wearing a byte
// costume, exactly as the prefill arm's table below concludes, and the only
// lever that has ever moved this kernel is issuing fewer, wider loads.
//
// That lever is `PIE_DP`, the channel pairs a lane owns, and it is not free:
// `PIE_DX`, `PIE_DP`, `PIE_KB` and `PIE_KY` are all derived from `PIE_PAIRS`,
// `PIE_PAIRS` is half the head width, and `driver-wgpu::geometry`'s
// `Rule::SdpaVector` guards that twice the module's workgroup width IS the
// head width -- with a third copy of the arithmetic in `kernels-wgpu::attn`.
// The x axis has to be REINTERPRETED, not reshaped. Nothing here has been
// tried; this section exists so the next attempt starts from 1.458 and not
// from a stamped share.
//
// # SIX PROBES INTO THAT LOOP AND FIVE OF THEM ARE FREE
//
// The 1.458 ms was then cut apart the same way, one deletion at a time. Every
// probe below gives the WRONG ANSWER, every one is reverted, and every one
// keeps the loop trips, the grid and the bindings intact so that only the
// named thing goes. Plain medians, three rounds each, no instrument; the
// baseline cluster that day was 7.429-7.525 with a mean of 7.446.
//
// ```text
//   probe                                          decode p50   delta
//   shipped                                          7.446         --
//   the V load deleted -- HALF of every load          7.458       0
//   channel map contiguous instead of strided         7.442       0
//   the butterfly deleted -- 5 shuffles a key         7.556       0
//   both `exp()` deleted from the online softmax      7.521       0
//   the page-table load deleted                       7.249     -0.197
//   `PIE_DX` 16 / `PIE_DP` 4 -- a lane owns 16 B      8.001     +0.55
// ```
//
// **THIS ARM IS NOT PAID PER LOAD ISSUED, AND THAT IS THE OPPOSITE OF WHAT THE
// PREFILL ARM'S TABLE SAYS.** `v_pages` and `k_pages` are read at the same
// index in the same loop body, so deleting the V fetch removes exactly half of
// the kernel's scalar loads -- and it returns NOTHING. The two arms are the
// same file and they are not the same kernel: the prefill arm has 512 rows of
// work per fire and saturates the issue port, and the decode arm has one.
//
// **Nor is it paid per load ADDRESS.** A lane's `PIE_DP` words are strided by
// `PIE_DX` and so are 128 B apart; remapping all six sites to `cl * PIE_DP + p`
// makes them adjacent, which is what a `vec4` load would need. Three
// INTERLEAVED rounds -- strided 7.429/7.454/7.456, contiguous
// 7.483/7.404/7.439 -- and the two means are 7.446 and 7.442. The pattern is
// free, so the "fewer, wider loads" idea has nothing to buy even if naga
// emitted the wide load.
//
// **And a lane cannot own more anyway.** `PIE_DX = 16` is the shape in which a
// lane holds four CONTIGUOUS words, one 16-byte vector. It self-derives --
// `PIE_DP` 4, `ks` still covers `PIE_KY`, `lim` 16 is still inside a subgroup,
// `@workgroup_size` unchanged -- and it costs 0.55 ms. The reason is two lines
// away: `PIE_KY = PIE_KB * PIE_DP` doubles, so `pie_sdpa_macc` doubles to 8
// KiB, and this is the identical residency cliff `quant/qmv.wgsl` fell off
// when its `qmv_partials` grew. Every kernel in this tree that has been asked
// to hold more state has said no.
//
// # THE PAGE WALK IS THE ONLY THING THAT ANSWERS, AND IT CANNOT BE COLLECTED
//
// 0.197 ms is 2.6% of a token and 17% of the loop, and all three readings sit
// below every baseline reading, so it is real. It is also the only DEPENDENT
// load in the body: `kv_page_indices` must return before the K and V addresses
// exist. That is why it prices and the other four loads do not -- it is not
// bytes and not issue slots, it is a serial chain of two loads per key.
//
// Four ways of shortening that chain, all measured, all reverted:
//
// ```text
//   fix                                            decode p50
//   shipped                                          7.446
//   cache the entry in a register across a page      7.561
//   stage the split's slice into workgroup memory    7.714
//   the same, with the fallback branch removed       7.637
//   prefetch the entry one block ahead               7.567
// ```
//
// The register cache is right about the redundancy -- this lane's keys advance
// by `PIE_KSPAN` a block, so at `page_size` 16 it re-reads the same entry four
// times -- and the `if` that skips the re-read costs more than the read. The
// staged version says the thing worth remembering: **workgroup memory is
// SLOWER here than the storage load it replaces**, by 0.19 ms, with the
// fallback branch removed so nothing else is in the way. And the prefetch says
// the chain was already being hidden by whatever the compiler does.
//
// So the probe bounds the fix at 0.197 and four fixes collect zero, which is
// this file's sixth rule for the third time.
//
// # WHAT THAT LEAVES
//
// Nothing in the body prices. Deleting half the loads, all the shuffles, all
// the transcendentals or the address arithmetic each returns zero, and the
// loop is still 1.15 ms. A kernel whose parts are all free and whose whole is
// not is bound at the LOOP, not at any instruction in it: 64 keys of a serial
// online-softmax dependence with one row of work to hide it behind, at an
// occupancy the `PIE_KR` and `PIE_DX` sweeps have both already pinned to the
// smallest state a lane can hold. The next idea has to change how many
// independent things are in flight -- and `PIE_SPLITS`, which is exactly that
// knob, has been swept four times and says eight.
//#if defined(PIE_SPLITK)
const PIE_KR: u32 = 1u;
//#else
const PIE_KR: u32 = 8u;
//#endif
// NO `enable subgroups;` HERE, AND THAT IS NOT AN OVERSIGHT. naga 30 refuses
// the enable-extension outright -- "specifies standard functionality which is
// not yet implemented in Naga" -- while parsing and lowering the subgroup
// builtins themselves perfectly well, gated on `wgpu::Features::SUBGROUP`
// instead. `common/reduce.inc.wgsl` writes the enable and has therefore never
// been compiled by anything; it is dead at this version.
const PIE_KSPAN: u32 = PIE_KR * PIE_KY;
var<workgroup> pie_sdpa_part: array<f32, PIE_KSPAN * PIE_DX>;
// The per-y-lane softmax states, staged ONCE at the end of the row so the y
// axis can be folded away. See "THE Y AXIS CARRIES ITS OWN SOFTMAX" below for
// why the accumulator needs a slot per lane and the max and sum need one per
// y lane: the running max and denominator are functions of the SCORES, which
// every lane of a y row shares, while the accumulator is this lane's two head
// elements and nobody else's.
var<workgroup> pie_sdpa_macc: array<vec2<f32>, PIE_KY * PIE_PAIRS>;
var<workgroup> pie_sdpa_mmax: array<f32, PIE_KY>;
var<workgroup> pie_sdpa_msum: array<f32, PIE_KY>;

// `sp` and `n_splits` are the SLICE of this row's keys this workgroup owns.
// One and zero is the whole range, which is the unsplit kernel exactly; the
// split arm dispatches `n_splits` workgroups per (row, head) and each takes
// its own slice, leaving the merge to `PIE_COMBINE`.
fn decode_row(row: u32, q_head: u32, lane: u32, ky: u32, n_q_heads: u32, sp: u32, n_splits: u32, sg: u32) {
    // THE X AXIS, CUT IN TWO. `cl` is the channel lane and `kx` is the key
    // slice; together with `ky` they name this invocation's state. The
    // channel pairs a lane owns are strided by `PIE_DX` and not contiguous,
    // so that for any fixed `p` the channel lanes still walk consecutive
    // words: a coalesced load is a property of what the LANES do at one
    // instant, not of what one lane does over time.
    let cl = lane % PIE_DX;
    let kx = lane / PIE_DX;
    // This invocation's key slot inside a block, and the slot its softmax
    // state merges from. `ky` varies fastest so that the `PIE_KB` invocations
    // of one x slice take CONSECUTIVE keys, which is what keeps the page walk
    // and the K/V loads contiguous across the y axis.
    let ks = kx * PIE_KB + ky;
    let req = req_of_token[row];
    let q_pos = position_ids[row];
//#if defined(PIE_FAST_FULL)
    var start = 0;
//#else
    var start = 0;
    if (params.window > 0 && q_pos >= params.window) { start = q_pos - params.window + 1; }
//#endif
    let q_base = q_base_for(row, q_head, n_q_heads);
    let o_base = o_base_for(row, q_head, n_q_heads);
    let kv_head = i32(q_head) / params.gqa_factor;

    // THIS SPLIT'S SLICE, rounded to whole `PIE_KSPAN` blocks.
    //
    // Rounded, and not divided evenly, because the block is the unit the loop
    // below stages and folds: a boundary that cut one would leave a partial
    // block at both ends of it and the `kn` clamp would have to know about
    // two limits instead of one. The cost is that the last split can be
    // empty, which costs a workgroup that exits immediately.
    //
    // At `n_splits == 1` this is `[start, q_pos]` -- the whole range, no
    // arithmetic changed -- which is what makes the unsplit arm and the split
    // arm one body.
    let span = q_pos + 1 - start;
    let per = (span + i32(n_splits) - 1) / i32(n_splits);
    let per_blocks = ((per + i32(PIE_KSPAN) - 1) / i32(PIE_KSPAN)) * i32(PIE_KSPAN);
    let lo = start + i32(sp) * per_blocks;
    let hi = min(q_pos, lo + per_blocks - 1);

    // # Three things measured here and NOT done, so nobody re-measures them
    //
    // All three with `how_long_a_decodes_kernels_take`, at d_128 and 2048 keys,
    // interleaved against the current body. Note the drift first: the SAME
    // binary read 1.420 ms and 1.637 ms an hour apart on this shared machine,
    // so only an interleaved A/B means anything, and a 15% effect is not one.
    //
    // **It is not bandwidth-bound.** 16 query heads over 16 KV heads, over 8,
    // and over 1 -- which is 16 MiB of distinct key/value traffic against 1 MiB
    // read sixteen times -- give 1.416, 1.417 and 1.416 ms. Sharing the KV
    // reads between the heads of a GQA group therefore buys nothing, which is
    // the obvious next optimisation and is dead.
    //
    // (SAME INSTRUMENT AS THE PARAGRAPH BELOW, WHICH WAS OVERTURNED. This was
    // llvmpipe, which has a CPU's cache hierarchy and no graphics part's
    // bandwidth wall, so "16 MiB and 1 MiB read alike" is exactly what it
    // would say whether or not the claim held on a GPU. Re-taken on an M4 by
    // the probe table above: forcing `kv_head = 0` -- an EIGHT-fold cut in
    // unique traffic -- reads 9.476 against 9.828, 14% of the kernel. So the
    // conclusion survives its instrument this time, but only just, and the
    // arithmetic it licenses is different: this model's `gqa_factor` is 2, so
    // sharing K and V across a GQA group is a TWO-fold cut, not eight, and at
    // that slope it is worth a few percent of a kernel that is itself 19% of a
    // token. Still dead, now for a reason that was measured on the machine it
    // is claimed about.)
    //
    // **The barrier COUNT is not the cost.** Blocking the key loop so one tree
    // serves eight keys -- eight times fewer barriers, identical arithmetic --
    // measured 1.601 ms against 1.637 ms. llvmpipe runs a whole workgroup on
    // one thread and vectorises across its invocations, so a
    // `workgroupBarrier()` is close to a loop boundary rather than a thread
    // rendezvous. What the tree costs is its ADDS: 63 of them to fold 64
    // lanes, against 128 multiply-adds for the dot itself. That is a third of
    // the kernel and it is inherent to reducing 64 lanes.
    //
    // (This also corrects the reasoning in the commit that first reverted the
    // reduction, which blamed barrier synchronisation. The conclusion was
    // wrong and so was the mechanism.)
    //
    // # AND THE CORRECTION ABOVE IS ITSELF WRONG ON METAL
    //
    // Everything in the paragraph before this was measured on llvmpipe, which
    // the text says, and the conclusion it reached -- "what the tree costs is
    // its ADDS, 63 of them, and that is inherent" -- does not survive a real
    // GPU's own clock.
    //
    // `PIE_WGPU_STAMP` prices this kernel at 98-108 us a launch and 31% of a
    // decode, the largest single kernel in a token. Truncating the ladder from
    // six levels to three -- which gives the WRONG ANSWER and was reverted, and
    // which removes only 4 + 2 + 1 = 7 of the 63 adds, ELEVEN PERCENT of them
    // -- took 18 us off the launch, a THIRD of the kernel. Eleven percent of
    // the adds cannot buy a third of the time. **The cost is the LEVEL, not
    // the add**: a barrier and a workgroup-memory round trip, paid six times
    // for every four keys, and the levels with four active lanes cost the same
    // as the level with thirty-two.
    //
    // llvmpipe could not have seen this. It runs a whole workgroup on one
    // thread and vectorises across the invocations, so a `workgroupBarrier()`
    // there is close to a loop boundary and the adds really are all that is
    // left. On an M4 the same barrier is a rendezvous of eight simdgroups.
    // The old measurement was sound and its conclusion was portable to exactly
    // one machine.
    //
    // WHAT WAS DONE. The `@subgroup` tier replaces `log2(PIE_PAIRS)` levels
    // with `log2(lim)` register exchanges and ONE store -- eight barriers a
    // block become two -- and the fold walks `PIE_PAIRS / lim` block sums
    // instead of reading one slot. (Both of those two are gone as well now;
    // the section after next is how, and `PIE_PAIRS` there has become
    // `PIE_DX`.) See the reduction itself for why the
    // butterfly is confined to an aligned block and therefore correct at any
    // subgroup width.
    //
    // Three interleaved rounds, 200 decodes at 512 keys, `PIE_WGPU_TIER`
    // switching the tier inside one binary:
    //
    // ```text
    //   round        1       2       3
    //   subgroup   8.980   9.051   9.210   ms a token
    //   baseline   9.593   9.566   9.664
    // ```
    //
    // **0.54 ms a token, 5.7%, and every round wins.** By the stamp table the
    // kernel goes 86.6/106.1/105.1 us to 76.1/91.5/96.4 across the bench's
    // three windows and its share of a decode falls 31% to 26%.
    //
    // NOTE WHAT THE TRUNCATION PROBE OVERSTATED. It said three levels were
    // worth 18 us; removing five bought about 11 us by the same instrument.
    // A probe that deletes work measures more than a fix that reorganises it,
    // because the fix still pays for the exchanges. The probe was right about
    // the SHAPE -- levels, not adds -- and wrong about the size, which is the
    // sixth rule wearing a new hat: a probe's number bounds the fix, it does
    // not predict it.
    //
    // # AND THEN THE LAST TWO BARRIERS WENT TOO
    //
    // "Eight barriers a block become two" is the sentence the next sitting
    // attacked, because on this finding two is still two levels and a level is
    // what costs. The two survived for a reason that had nothing to do with
    // the reduction: `PIE_PAIRS` is 64 at `d_128` and a simdgroup is 32, so a
    // ROW STRADDLED TWO SUBGROUPS and a butterfly cannot cross one. The block
    // sums had to meet in workgroup memory, with a fence on each side of the
    // meeting.
    //
    // Priced first, by the same truncation probe: delete the staging, both
    // `workgroupBarrier()`s and the cross-block fold, take the wrong answer,
    // read the token. Then fixed, by giving a lane `PIE_DP` channel pairs so a
    // row is `PIE_DX = 32` lanes -- see `PIE_DX` for why the workgroup keeps
    // its declared shape and the X AXIS is reinterpreted instead. Interleaved,
    // 200 decodes at 512 keys:
    //
    // ```text
    //   shipped    7.789  7.764  7.719   7.737  7.766  7.774   mean 7.758
    //   probe      7.374  7.346  7.305                         mean 7.342
    //   fixed      7.484  7.497  7.488                         mean 7.490
    // ```
    //
    // **0.27 ms, 3.5%**, and the nine baseline samples and the three fixed
    // ones do not overlap. The probe said 0.42 and the fix collected 0.27,
    // which is the sixth rule for the second time in one kernel -- and here
    // the gap has a name: the probe kept 64 lanes each holding one pair, the
    // fix holds two pairs and two accumulators in half as many lanes, so it
    // pays in registers what it saved in rendezvous.
    //
    // The decode's largest kernel has now had its ladder folded (5.7%) and
    // then emptied of barriers entirely (3.5%), and the token has gone 9.64 ->
    // 7.49 ms across the two. There is no rendezvous left to remove.
    //
    // # WHERE THE 85 us THAT IS LEFT ACTUALLY IS, CUT AT THE LOOP
    //
    // The kernel is still ~33% of a token by the stamp table, ~85 us a launch
    // against ~7 us of unique traffic, and six probes into its BODY have all
    // come back free. So this sitting cut it somewhere else: at the loop
    // boundary. Force `hi` below `lo` so the key loop runs ZERO iterations --
    // wrong answers, reverted -- and everything else about the kernel stands:
    //
    // ```text
    //   shipped                7.484  7.497  7.488   mean 7.490
    //   zero key iterations    6.339  6.343  6.334   mean 6.339
    // ```
    //
    // **THE ENTIRE 64-KEY LOOP IS 1.15 ms OF A TOKEN.** Everything else this
    // kernel does -- launch, query load, the merge tail beside
    // `pie_sdpa_macc`, the split-state writeout -- is what is left, and the
    // rest of this note prices those one at a time. They come to about 0.25
    // ms.
    //
    // # WHICH MAKES THIS KERNEL 19% OF A TOKEN AND NOT 33%, AND THE STAMP
    // # TABLE IS WHAT WAS WRONG
    //
    // 1.15 + 0.25 is ~1.4 ms of a 7.490 ms decode. `PIE_WGPU_STAMP`'s `[cost]`
    // table has this kernel at 32-35% of a window, which would be ~2.5 ms, and
    // the two cannot both be true. The probes win, and they agree with a third
    // measurement that was taken for another purpose: doubling the split count
    // doubles the WORKGROUPS without changing the key work, and 8 -> 16 costs
    // only 0.17 ms end to end. A kernel with 1.3 ms of per-workgroup fixed
    // cost could not add 128 workgroups a layer for 0.17.
    //
    // The stamp table's SHARES are distorted, and this kernel is the one they
    // distort most. Apple has no `TIMESTAMP_QUERY_INSIDE_PASSES`, so stamping
    // forces one compute pass per launch, and the cost of a pass scales with
    // how much state the launch establishes. This kernel has the widest grid
    // in a decode -- 16 heads x 8 splits x 256 invocations against a `qmv`'s
    // one-dimensional 256 -- so it collects more of that surcharge than
    // anything else in the table and its share is inflated against kernels
    // that collect less.
    //
    // The seventh rule already said a profiler's window must be cut on the
    // work rather than the wall clock, and the standing note says the stamp
    // table RANKS kernels while the interleaved bench PRICES changes. This is
    // the sharper version of both: **the stamp table's ranking is sound and
    // its shares are not a budget.** The 33% led four sittings to treat this
    // kernel as a third of the problem. It is a fifth.
    //
    // AND THE PIECES OUTSIDE THE LOOP, PRICED. The merge tail is 0.215 ms and
    // has a section of its own beside `pie_sdpa_macc`, including the two
    // restructures that failed to collect it. The split-state writeout is
    // FREE: let only `sp == 0` write it -- same buffer, same binding, an
    // eighth of the traffic -- and the decode reads 7.439/7.506/7.474 against
    // 7.490, which is a tie. 130 floats a workgroup is 1.9 MB a token written
    // and read back, ~14 us of bandwidth, and it measures like it.
    //
    // The paged gather, which was the leading suspect, is inside the loop and
    // is not much of it either.
    // Replace `page_slot_at` with a lookup that always resolves to one page --
    // same load, same binding, no scatter -- and the decode reads
    // 7.326/7.363/7.388 against 7.490. **0.13 ms, 1.7%**, for deleting the
    // indirection entirely. (The first version of this probe dropped the
    // reference to `kv_page_indices` altogether, which drops the BINDING and
    // the fire is refused with a module/bound mismatch. The second rule --
    // change arithmetic, not addresses -- has a corollary in this driver: a
    // probe may not change what a shader touches.)
    //
    // # Two loop invariants are deliberately NOT hoisted here
    //
    // This lane's two query elements are the same for every key, and so is
    // `mask_enabled(row)` inside `keeps`. Hoisting both is bit-identical --
    // `params.scale * q * k` parses as `(params.scale * q) * k`, so pulling
    // `params.scale * q` out is the same arithmetic in the same order -- and
    // it was written, measured and thrown away: 1.415 ms to 1.390 ms at 2048
    // keys, against a 4% run-to-run spread on the same binary. LLVM's LICM is
    // already doing it; the storage buffers it would have to prove non-aliasing
    // for are all `read`.
    //
    // Recorded so the next person spends the ten minutes on the bench rather
    // than on the edit. `how_long_a_decodes_kernels_take` is how to check.
    //
    // # The restructure none of the above covers, and what was DONE instead
    //
    // The note used to propose full flash-decoding: give each of the 32 x
    // lanes its own KEYS over the whole head dim, and merge once at the end.
    // That trades a perfectly coalesced 128-byte key load for 32 per-lane
    // contiguous ones, which is why it was never taken.
    //
    // The Y axis gives the same win at none of that cost, and is what this
    // body now does. `ky` ALREADY owns keys -- it always did, that is what the
    // second workgroup axis is for -- so letting it own a softmax STATE as
    // well changes no load at all. What it removes is the redundancy that sat
    // between the two: the fold used to walk all `PIE_KSPAN` staged slots on
    // every one of the `PIE_KB` y lanes, so a `PIE_KB`-deep serial dependence
    // chain ran `PIE_KB` times over and every staged value pair was read
    // `PIE_KB` times from workgroup memory. Now each y lane folds only the
    // `PIE_KR` keys it loaded, V never reaches workgroup memory at all, and
    // `PIE_KB` states merge once after the last block.
    //
    // Measured on an M4 with Llama-3.2-1B, end to end through the engine:
    // decode 89.8 -> 102.9 tok/s at 512 context and 52.7 -> 71.2 at 2048.
    // The gain grows with the context because that is where this kernel's
    // share of the token grows. Workgroup memory fell from 12 KiB to 6 KiB,
    // which is a second effect: two of these fit in an Apple core's 32 KiB
    // where two-and-a-bit did before.
    //
    // What is left in here is the 63-add tree, still once per key. That is
    // the restructure above, and it is still not obviously worth its load.
    var max_score = PIE_SDPA_NEG_INF;
    var sum_exp = 0.0;
    var acc: array<vec2<f32>, PIE_DP>;
    for (var p = 0u; p < PIE_DP; p = p + 1u) { acc[p] = vec2<f32>(0.0, 0.0); }
    // Loop-invariant, and it was the first of two dependent loads per key.
    let page_base = kv_page_indptr[u32(req)];
    // AND IT BOUGHT NOTHING MEASURABLE HERE, which is worth writing down.
    //
    // The same three edits on the TILED arm took a 512-row prefill from 571 ms
    // to 445. On this arm, `what_a_decode_costs_at_length` reads 10.092 ms
    // before and 10.017 ms after, one sitting -- under 1%, which is this
    // harness's noise. The reason is that a decode at 512 keys is not this
    // kernel: one row against 512 keys is a thousandth of the prefill's key
    // work, while the 196 projection dispatches and the per-fire cost are
    // unchanged. It is kept because it is strictly fewer loads for identical
    // answers, and because the decode's share of the token GROWS with the
    // context -- the note above measures exactly that at 2048 -- but nobody
    // should expect a number from it at 512.
    //
    // THIS LANE'S TWO QUERY CHANNELS, ONCE.
    //
    // `d_out` is `lane * 2` and `q_base` is a multiple of `PIE_HEAD_DIM`, so
    // this pair is one word and the lane needs the same word for every key it
    // ever touches. It was re-read inside the key loop, twice -- `q_at` loads
    // `queries[i >> 1]` and selects a half, so a pair is two loads of one
    // word -- which is `2 * keys` loads of a value that does not change.
    var q_lo: array<f32, PIE_DP>;
    var q_hi: array<f32, PIE_DP>;
    for (var p = 0u; p < PIE_DP; p = p + 1u) {
        let q_word = queries[(q_base + (cl + p * PIE_DX) * 2u) >> 1u];
        q_lo[p] = pie_bf16_to_f32(q_word & 0xffffu);
        q_hi[p] = pie_bf16_to_f32(q_word >> 16u);
    }
    // HOW WIDE A SHUFFLE MAY REACH, at the `@subgroup` tier. A butterfly over
    // `off < lim` only ever exchanges with `lane ^ off`, which stays inside the
    // aligned `lim`-lane block, so taking the smaller of the row's width and
    // the subgroup's makes it correct on BOTH sides of the comparison: a
    // subgroup wider than a row never crosses into the neighbouring row, and a
    // row wider than a subgroup never reaches outside its own subgroup. The
    // remainder -- `PIE_DX / lim` block sums -- goes through workgroup
    // memory exactly once, and on every part this backend has run on there is
    // no remainder because `PIE_DX` is 32 and so is the simdgroup. `lim` is a
    // runtime value and it is uniform across the workgroup, which is what lets
    // the loops below carry a barrier.
    let lim = min(PIE_DX, sg);
    // ONE TREE FOR PIE_KB KEYS.
    //
    // The note above records this as measured and rejected -- 1.601 ms against
    // 1.637 ms, "the barrier COUNT is not the cost" -- and that measurement is
    // sound for the machine it was taken on. It was llvmpipe, where the note
    // says why: a whole workgroup runs on one thread, so a barrier is close to
    // a loop boundary. On a real GPU it is a rendezvous of every simdgroup in
    // the workgroup, and this loop reached six of them PER KEY.
    //
    // Measured on an M4 with Llama-3.2-1B: decode ran at 72 tok/s against an
    // 8-token context and 40.7 against 512, and the whole of that fall is this
    // kernel. Eight keys to a tree does not change one multiply, one add or one
    // online update -- the tree's shape and the order the keys update the
    // running max are exactly as they were -- it just stops paying the
    // rendezvous seven times out of eight.
    var kp0 = lo;
    while (kp0 <= hi) {
        let kn = min(i32(PIE_KSPAN), hi + 1 - kp0);
        // ONE KEY PER Y LANE. The block's keys are loaded and dotted at the
        // same time rather than one after another, which is the whole point of
        // the second workgroup axis: this kernel dispatches one workgroup per
        // query head and nothing else, so at d_64 it had 32 x 32 = 1024
        // invocations for the entire decode's attention.
        //
        // V is read here, before the reduction, which is the old body's finding
        // and is kept: it does not depend on the score, and the reduction is a
        // long wait to start the load after.
        // THIS LANE'S `PIE_KR` KEYS, staged before any reduction. Lane `ky`
        // owns offsets `ky`, `ky + PIE_KB`, ... so the slot a key lands in IS
        // its offset in the block, and the fold below can read its own scores
        // without arithmetic.
        //
        // V never reaches workgroup memory. The lane that loads key `q`'s
        // value pair for head elements `2*lane, 2*lane+1` is exactly the lane
        // that accumulates them, so the pair stays in registers.
        var v_keep: array<vec2<f32>, PIE_KR * PIE_DP>;
//#if defined(PIE_SUBGROUP)
        var p_keep: array<f32, PIE_KR>;
//#endif
        for (var r = 0u; r < PIE_KR; r = r + 1u) {
            let q = r * PIE_KY + ks;
            var part = 0.0;
            if (i32(q) < kn) {
                let slot = page_slot_at(page_base, kp0 + i32(q));
                let k_base = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM;
                // ONE word each, where this was six loads for three words.
                // The two products and their sum are associated exactly as
                // the scalar form associated them, so no answer moves.
                //
                // `PIE_DP` of them now, because a lane owns that many pairs.
                // The partials are summed in the lane before the butterfly
                // sees them, which is the whole saving: the pairs a lane holds
                // are the pairs it no longer has to exchange for.
                for (var p = 0u; p < PIE_DP; p = p + 1u) {
                    let at = (k_base + (cl + p * PIE_DX) * 2u) >> 1u;
                    let v_word = v_pages[at];
                    let k_word = k_pages[at];
                    v_keep[r * PIE_DP + p] = vec2<f32>(
                        pie_bf16_to_f32(v_word & 0xffffu), pie_bf16_to_f32(v_word >> 16u));
                    part = part
                        + params.scale * q_lo[p] * pie_bf16_to_f32(k_word & 0xffffu)
                        + params.scale * q_hi[p] * pie_bf16_to_f32(k_word >> 16u);
                }
            } else {
                for (var p = 0u; p < PIE_DP; p = p + 1u) {
                    v_keep[r * PIE_DP + p] = vec2<f32>(0.0, 0.0);
                }
            }
//#if defined(PIE_SUBGROUP)
            p_keep[r] = part;
//#else
            pie_sdpa_part[q * PIE_DX + cl] = part;
//#endif
        }
//#if defined(PIE_SUBGROUP)
        // THE SCORE REDUCTION, WITHOUT THE LADDER AND WITHOUT A RENDEZVOUS.
        // The baseline arm below folds `PIE_DX` lanes through workgroup memory
        // in `log2(PIE_DX)` levels, which is five barriers and five shared
        // round trips for every block of keys. This arm does the whole fold in
        // `log2(lim)` register exchanges, and because `PIE_DX` is capped at a
        // simdgroup width the result never has to leave the registers at all:
        // the store and the two fences below are reached only by a part whose
        // subgroup is NARROWER than 32.
        //
        // The arithmetic is NOT the same and does not claim to be: a butterfly
        // associates the sum as a balanced tree over the whole block where the
        // ladder associates it as a balanced tree over halves. Both are
        // balanced trees over the same 64 f32 products, so the difference is
        // at most a few ulps of a score that then goes through `exp`, and
        // `arena`'s tolerance covers it. What must NOT move is the order the
        // KEYS update the running max, and that is the fold below, untouched.
        for (var r = 0u; r < PIE_KR; r = r + 1u) {
            var v = p_keep[r];
            for (var off = 1u; off < lim; off = off << 1u) {
                v = v + subgroupShuffleXor(v, off);
            }
            p_keep[r] = v;
            // ONLY IF THE SUBGROUP IS NARROWER THAN THE ROW. `PIE_DX` is 32,
            // and every part this backend has run on is 32 lanes wide, so this
            // branch is not taken and the block loop carries NO rendezvous at
            // all. It is kept because `subgroup_size` is a runtime value and a
            // 16-lane part would otherwise read a third of its dot; `lim` is
            // workgroup-uniform, so both arms are uniform control flow and the
            // barriers inside are legal.
            if (lim < PIE_DX && cl % lim == 0u) {
                pie_sdpa_part[(r * PIE_KY + ks) * PIE_DX + cl] = v;
            }
        }
        if (lim < PIE_DX) {
            workgroupBarrier();
            for (var r = 0u; r < PIE_KR; r = r + 1u) {
                var score = 0.0;
                for (var g = 0u; g < PIE_DX; g = g + lim) {
                    score = score + pie_sdpa_part[(r * PIE_KY + ks) * PIE_DX + g];
                }
                p_keep[r] = score;
            }
            workgroupBarrier();
        }
//#else
        workgroupBarrier();
        for (var half = PIE_DX >> 1u; half > 0u; half = half >> 1u) {
            if (cl < half) {
                for (var r = 0u; r < PIE_KR; r = r + 1u) {
                    let at = (r * PIE_KY + ks) * PIE_DX + cl;
                    pie_sdpa_part[at] = pie_sdpa_part[at] + pie_sdpa_part[at + half];
                }
            }
            workgroupBarrier();
        }
//#endif
        // Uniform: every lane of this workgroup has the same row and the same
        // position, so a masked key is masked for all of them. That is what
        // makes the reduction above legal, and it is why the mask is applied
        // HERE and not as a `-inf` score -- a score equal to the running max
        // contributes `exp(0)` to the denominator, and the running max starts
        // at a finite floor.
        //
        // THE Y AXIS CARRIES ITS OWN SOFTMAX. This loop used to walk all
        // `PIE_KSPAN` slots on every one of the `PIE_KB` y lanes, which meant
        // `PIE_KB` copies of one serial dependence chain and `PIE_KB` reads of
        // every staged value. Now y lane `ky` folds only the `PIE_KR` keys it
        // loaded, into a softmax state of its own, and the states meet once
        // after the last block. The partition is exact: an online softmax over
        // a disjoint cover merges to the same numerator and denominator as one
        // pass over the union, which is what flash-decoding rests on.
        for (var r = 0u; r < PIE_KR; r = r + 1u) {
            let q = r * PIE_KY + ks;
            let kp = kp0 + i32(q);
            if (i32(q) < kn && keeps(row, kp, q_pos, start)) {
//#if defined(PIE_SUBGROUP)
                let score = p_keep[r];
//#else
                let score = pie_sdpa_part[q * PIE_DX];
//#endif
                let step = pie_sdpa_online_update(score, max_score, sum_exp);
                max_score = step.max_score;
                sum_exp = step.sum_exp;
                for (var p = 0u; p < PIE_DP; p = p + 1u) {
                    acc[p] = acc[p] * step.history_scale
                        + step.score_scale * v_keep[r * PIE_DP + p];
                }
            }
        }
//#if defined(PIE_SUBGROUP)
        // NO BARRIER HERE, and that is the point of `PIE_DX`. The scores this
        // fold read came out of registers, so the next block has nothing to
        // overwrite. On a subgroup narrower than a row the branch above staged
        // them and fenced them itself, on both sides of its own read.
//#else
        // The next block's partials overwrite what this fold just read.
        workgroupBarrier();
//#endif
        kp0 = kp0 + i32(PIE_KSPAN);
    }
    // THE MERGE. `PIE_KY` states, each over a disjoint set of this row's keys,
    // become one. Every lane runs the same fold so every lane leaves holding
    // the answer, which is what the sink merge and the single writer below
    // already assumed.
    //
    // # WHAT THIS MERGE COSTS, AND TWO WAYS OF NOT COLLECTING IT
    //
    // Truncate the fold below to a SINGLE state -- wrong answers, reverted --
    // and a 7.490 ms decode reads 7.285/7.248/7.292. **0.215 ms, 2.9% of a
    // token**, for seven links of a loop that runs ONCE per workgroup. Set
    // beside the key loop that runs 64 keys through the same workgroup, which
    // the empty-loop probe below prices at 1.15 ms, this tail is a SEVENTH of
    // the body it exists to summarise.
    //
    // The obvious reading is the dependence: this is `pie_sdpa_online_update`'s
    // shape, a running max rescaling the history at every step, so it is a
    // serial chain `PIE_KY` deep holding two `exp` and a workgroup load per
    // link. The key loop has no choice about that shape -- it meets its scores
    // one block at a time -- but this loop has every state in front of it
    // before it starts, and a softmax merge over a disjoint cover is
    // associative. So it can be written in two phases with no chain at all:
    // take the max, then the denominator against it, then `PIE_KY`
    // INDEPENDENT multiply-adds per pair.
    //
    // Both versions were written and both are slower:
    //
    // ```text
    //   online fold (shipped)          7.484  7.497  7.488   mean 7.490
    //   two phases, exp recomputed     7.537  7.538  7.496   mean 7.524
    //   two phases, scales hoisted     7.526  7.541  7.516   mean 7.528
    // ```
    //
    // So the 0.215 ms is NOT the dependence. Both shapes read the same
    // `PIE_KY * PIE_DP` slots out of `pie_sdpa_macc` and the same `2 * PIE_KY`
    // out of the state arrays, and that traffic is what the truncation removed
    // -- it removes the loads along with the links. This is the same finding
    // as the ladder's, arriving from the other side: what costs in this kernel
    // is touching workgroup memory, and rearranging the arithmetic AROUND the
    // touches buys nothing because the arithmetic was never the price.
    //
    // Which says where the remaining lever is, and it is not here: the only
    // way to shrink this tail is to have FEWER STATES, and `PIE_KY` is
    // `PIE_KB * PIE_DP` -- the workgroup's own shape. Narrowing it trades this
    // 0.2 ms against the key loop's parallelism, and the invocation sweep
    // beside `PIE_KB` says 256 is a real interior optimum. Both ends would
    // have to move together and neither can be swept alone.
    for (var p = 0u; p < PIE_DP; p = p + 1u) {
        pie_sdpa_macc[ks * PIE_PAIRS + cl + p * PIE_DX] = acc[p];
    }
    if (cl == 0u) {
        pie_sdpa_mmax[ks] = max_score;
        pie_sdpa_msum[ks] = sum_exp;
    }
    workgroupBarrier();
    max_score = PIE_SDPA_NEG_INF;
    sum_exp = 0.0;
    for (var p = 0u; p < PIE_DP; p = p + 1u) { acc[p] = vec2<f32>(0.0, 0.0); }
    for (var t = 0u; t < PIE_KY; t = t + 1u) {
        let other_max = pie_sdpa_mmax[t];
        let merged_max = max(max_score, other_max);
        // Both are `PIE_SDPA_NEG_INF` until a state with keys arrives, and
        // that floor is finite for the reason the header of the online include
        // gives: `exp(-inf - -inf)` is NaN and `exp(0)` is 1.
        let history_scale = exp(max_score - merged_max);
        let other_scale = exp(other_max - merged_max);
        max_score = merged_max;
        sum_exp = sum_exp * history_scale + pie_sdpa_msum[t] * other_scale;
        for (var p = 0u; p < PIE_DP; p = p + 1u) {
            acc[p] = acc[p] * history_scale
                + other_scale * pie_sdpa_macc[t * PIE_PAIRS + cl + p * PIE_DX];
        }
    }
//#if defined(PIE_SPLITK)
    // STOP HERE AND STATE THE PARTIAL. No sink and no normalization: both are
    // functions of the FINAL denominator, and this workgroup has only its own
    // slice's. `PIE_COMBINE` does them once, after the states meet.
    //
    // Written unconditionally, including by a split whose slice was empty --
    // the identity of the merge, which is what saves the combine from needing
    // to know how many splits were live. Stale state from the previous layer
    // sitting in an unwritten slot would otherwise be read as attention.
    let stride = 2u + PIE_HEAD_DIM;
    let base = ((row * n_q_heads + q_head) * n_splits + sp) * stride;
    // `ky == 0 && kx == 0`: one writer per channel, once. See the unsplit
    // arm's writeout below for why the x axis now needs a guard of its own.
    if (ky == 0u && lane < PIE_DX) {
        if (lane == 0u) {
            pie_split_state[base] = max_score;
            pie_split_state[base + 1u] = sum_exp;
        }
        for (var p = 0u; p < PIE_DP; p = p + 1u) {
            let d_out = (cl + p * PIE_DX) * 2u;
            pie_split_state[base + 2u + d_out] = acc[p].x;
            pie_split_state[base + 3u + d_out] = acc[p].y;
        }
    }
}
//#else
//#if defined(PIE_WITH_SINK)
    let merged = pie_sdpa_merge_sink(sink_at(q_head), max_score, sum_exp);
    for (var p = 0u; p < PIE_DP; p = p + 1u) { acc[p] = acc[p] * merged.output_scale; }
    sum_exp = merged.sum_exp;
//#endif

    // Every y lane ran the merge above, so they all hold this answer and one
    // writes it. The redundancy is `PIE_KB` fused multiply-adds, once per row.
    for (var p = 0u; p < PIE_DP; p = p + 1u) {
        var norm = acc[p];
        if (sum_exp != 0.0) { norm = acc[p] / sum_exp; }
        let at = (o_base + (cl + p * PIE_DX) * 2u) >> 1u;
        // `lane < PIE_DX` is `ky == 0 && kx == 0`: ONE channel lane per
        // channel, once. Every state ran the merge above so they all hold the
        // answer, and the x axis now holds `PIE_DP` copies of each channel
        // lane on top of the `PIE_KB` the y axis already held.
        if (lane < PIE_DX && ky == 0u && at < arrayLength(&out_)) {
            out_[at] = pie_pack_bf16(norm.x, norm.y);
        }
    }
}
//#endif

//#endif

//#if defined(PIE_TILED)

// How many output pairs one x-lane of a tile owns.
//
// LANES PER ROW, on the channel axis: a row has `PIE_PAIRS` pairs and a lane
// takes every `PIE_TX`th one. Every one of them walks the whole key
// history and every one of them computes the SAME `dot_page`, so this number
// is the redundancy factor on the dot -- and the dot is 128 multiply-adds a
// key at `d_128` against the two the accumulator costs.
//
// It was 32 by inheritance: the group is `(32, 8)` because 32 is a subgroup
// and 8 is what 256 invocations leave. Nothing about the arithmetic wanted
// it. Lowering it trades registers for that redundancy -- a lane holds
// `PIE_LANE_PAIRS` `vec2<f32>` accumulators, so halving the lanes doubles
// them -- and the trade is decided by measurement, not by counting.
//
// The row axis stays 32 wide whatever this is: the TILE is 32 rows, which the
// `rr < 32u` sweep below and the host's `ceil(n_rows / 32)` in y both state,
// and it does not move with the lane extents. `PIE_TX * PIE_TY <= 256` and
// `PIE_TY <= 32` together bound this from below at 8 while the group stays
// full; below that the group is smaller than 256, which is legal and which
// the sweep covers.
//
// Swept at 512 prefill rows of qwen3-0.6b, Apple M4 Pro, `--release`, one
// sitting, `what_a_prefill_costs_at_length` (see `driver-wgpu/tests/serving.rs`):
//
// ```text
//   PIE_TX      ms     tok/s
//       32    3037       169
//       16    1665       307
//        8    1028       498
//        4     750       683
//        2     571       897
//        1     595       860
// ```
//
// Re-taken after `dot_page` went word-at-a-time, which halved that loop's
// loads and moved the whole curve: 1 and 2 now READ THE SAME, 445 ms against
// 541 at 4. 2 is kept because it is the optimum on both sweeps and because
// the tie at 1 is a tie rather than a win.
//
// Monotone from 32 down to 2 and then back up, which is what a redundancy
// traded against a register file should look like: every halving of the lanes
// halves the `dot_page` work and doubles `PIE_LANE_PAIRS`, and at 1 the
// accumulator finally costs more than the dot it saved. **5.3x**, and the
// per-row cost turns from rising to flat.
//
// EVERY CELL BUT `32` AND `2` IN THE FIRST VERSION OF THIS TABLE WAS AN
// ARTEFACT, and the artefact read 1623 tok/s -- nearly twice the real
// optimum -- so it is worth stating how. `apply` hands a `Fire` LANES and the
// `Fire` divides by the module's own `@workgroup_size`; `kernels-wgpu`'s
// `tiled_lanes` is a SECOND copy of the arithmetic below and it was not moved
// with it. A host saying 2 against a shader saying 8 asks for a quarter of
// the query heads, so three quarters of the attention was never computed --
// fast, and wrong, and only wrong in the heads. `arena`'s workgroup census is
// what caught it (16,332,666 against 16,338,066) after
// `the_tiled_gemm_answers...` had already failed at 216% of peak. The two
// copies now agree by construction and the sweep above was re-taken with both
// ends moved together; `32` reproduces the pre-existing 3037 ms exactly,
// which is what says the new numbers are the same measurement.
//
// Derived from the head dim rather than flat, and the bound is the
// ACCUMULATOR: a lane holds `PIE_LANE_PAIRS` `vec2<f32>` live across the
// whole key loop, so this floors at 2 -- the measured optimum, and the whole
// story at `d_64` and `d_128` -- and rises only once 32 pairs a lane would be
// exceeded. That gives 2, 2, 8, 16 at `d_64`, `d_128`, `d_256`, `d_512`.
// Only `d_128` is measured; the wider two are an extrapolation of the shape
// above and the first Qwen-class checkpoint at `d_256` should re-take it.
const PIE_TX: u32 = max(2u, PIE_PAIRS / 32u);
// 32 rows is the most the group covers, and 256 invocations is the most
// WebGPU guarantees; the smaller of the two wins. At `d_128` this is a
// 64-invocation group, which is legal and which the sweep says is faster than
// the full one it replaced.
const PIE_TY: u32 = min(32u, 256u / PIE_TX);
const PIE_LANE_PAIRS: u32 = PIE_PAIRS / PIE_TX;

// Every pair one lane owns, over ONE walk of the keys.
//
// The body this replaced was called once per pair, and each call walked the
// whole key history -- and the whole query and key vector at each key -- over
// again. That is the same redundancy the decode arm had, arrived at from the
// other side: there it was across LANES, here it was within one.
//
// The saving is exact and needs no cooperation, because the online softmax
// state does not depend on the output channel. `max_score` and `sum_exp` are
// functions of the SCORES, and every pair of one (row, head) sees the same
// scores; only `acc` is per pair. So one key loop can carry all of a lane's
// pairs and the dot product is computed `PIE_LANE_PAIRS` times fewer.
//
// # Why this and not the workgroup reduction the decode arm took
//
// Because this arm cannot barrier. One workgroup here spans thirty-two
// different rows -- `row` is derived from `lid.y` -- with different positions,
// windows and masks, and lanes `continue` past rows that are not theirs. A
// `workgroupBarrier()` that some lanes reach and others do not is a hang, and
// WGSL's uniformity analysis would reject it rather than let one ship.
// `kernels-vulkan` makes the loop uniform to get the other factor of 32 (a
// shared atomic for the tile's largest position, rows past `n_rows` staying in
// with `q_pos = -1`, the mask as a predicate rather than a `continue`); that is
// a separate change with a separate proof, and this one is free of it.
// # WHAT THE PREFILL ARM COSTS, AND THE FIRST PROBE INTO IT
//
// `device.rs`'s `PIE_WGPU_SKIP` prices a prefill the same way it prices a
// decode -- by dropping a kernel's fires and re-timing. A 512-row prefill is
// 311.7 ms, and TWO kernels are 95% of it:
//
// ```text
//   skipped                       ms      saved     share
//   (baseline)                 311.7
//   affine_qmm_t (the GEMM)    127.4    184.3 ms    59.1%
//   sdpa_paged_tiled_d_128     201.0    110.7 ms    35.5%
// ```
//
// **The attention arm is 110.7 ms and it should not be.** A causal 512-row
// prefill is about 30 GFLOP of QK and PV across 28 layers -- the loop below
// runs `kp <= q_pos`, so the triangle is already the only thing computed --
// and 30 GFLOP in 110.7 ms is 271 GFLOP/s against this part's ~7.7 TFLOP/s.
// **3.5% of the machine.** Nothing else in either phase is that far from its
// wall; the GEMM beside it runs at about a third of peak.
//
// FIRST PROBE, AND IT IS NOT THE KEY TRAFFIC. Every (row, head) lane walks the
// paged cache itself, so a layer's 2.1 MiB of K and V is re-read by 512 rows
// times 16 heads -- on paper about 15 GB a prefill, which at 273 GB/s would be
// half the 110.7 ms on its own. Forcing `kv_head = 0` cuts the UNIQUE footprint
// eightfold at an identical issue count, identical arithmetic and identical
// bindings. Two rounds: 312.1 and 313.1 ms against baselines of 311.4 and
// 311.7. **A tie, and if anything slower.**
//
// So the re-reads are absorbed -- a layer's whole K and V is 2.1 MiB and the
// part's caches hold it -- and the paper figure was never traffic that moved.
// Same shape of answer as `qmv.wgsl`'s activation probe and the decode arm's
// own `kv_head = 0`: this family's loads keep looking like the cost and keep
// measuring free.
//
// AND THE PER-KEY ALU IS NOT IT EITHER, SO "3.5% OF ALU PEAK" WAS THE WRONG
// DENOMINATOR AND IS WITHDRAWN. The paragraph above used to end by naming the
// online softmax and the bf16 unpack as the next things to cut. They were
// priced by stripping them: the four unpacks, the four `params.scale`
// multiplies and the four products in `dot_row`'s unrolled body were replaced
// by two `bitcast<f32>(q ^ k)` adds, which keeps every load and every loop
// trip and deletes about 85% of the dot's arithmetic. Two rounds: 305.7 and
// 307.6 ms against 311.4 and 311.7. **4.9 ms -- 1.6% of a prefill and 4.4% of
// this arm.**
//
// Put the three probes in one line and the currency is not in doubt:
//
// ```text
//   probe                                  arm's share   what it removes
//   kv_head = 0 (8x unique footprint)          0%        bytes
//   dot ALU stripped (~85% of it)              4.4%      arithmetic
//   w < PIE_PAIRS / 2 (halve the dot)         45%        loads AND trips
//   p < PIE_LANE_PAIRS / 2 (halve the V)      40%        loads AND trips
// ```
//
// **Nothing here is paid for in bytes and nothing is paid for in arithmetic.
// It is paid for per LOAD ISSUED.** Count them: 512 rows x 16 heads x ~256
// causal keys x `PIE_TX` = 2 lanes is 4.2M lane-keys a layer, each issuing 128
// word loads in the dot and 32 in the V accumulate, so 671M loads a layer and
// 18.8 G over 28 -- delivered in 110.7 ms at **170 G scalar loads/s**. Every
// one of them is a cache hit, per the traffic paragraph above and per the
// `kv_head` probe.
//
// # Which makes the `PIE_TX` redundancy the whole target, and it is reachable
//
// Both lanes of a `PIE_TX` pair compute the SAME dot from the SAME words, so
// **half of the 537M dot loads a layer fetch a number the neighbouring lane is
// fetching in the same cycle.** The note further up prices a perfect split at
// about 8% of a prefill and then dismisses it, because splitting one dot
// across two lanes re-associates a sum that `kernels-metal` and
// `kernels-vulkan` are walked against number by number.
//
// THAT DISMISSAL IS TOO QUICK, AND SPLITTING IS WHAT SHIPPED. The re-
// association objection is the same one the split decode arm's butterfly
// answers a few hundred lines up: both orders are balanced trees over the same
// products, a few ulps of a score that then goes through `exp`, inside
// `arena`'s tolerance -- and the batch-determinism hazard recorded in
// `dot_row` is about the COMPILER choosing different vectorisations for
// different variants, which a stripe written out in the source cannot do.
// `dot_row_split` below stripes the dot across the `PIE_TX` lanes and folds it
// with a `log2(PIE_TX)` butterfly, under a new `@subgroup` tier for
// `sdpa_paged_tiled`.
//
// **311.7 -> 285.9 ms. 25.8 ms, 8.3% of a prefill, 1641 -> 1791 tok/s.** Three
// rounds at 285.8/285.9/286.2. The predicted bound was 22 ms and the measured
// win is slightly larger, which is the butterfly costing less than the loads
// it replaced plus the stripe halving the QUERY loads as well as the key ones.
// 300 tests pass at both tiers; the baseline arm is untouched and still reads
// 312.5 ms, and the decode is unmoved at 7.49 ms.
//
// WHAT IS LEFT ON THIS ARM. The V accumulate is 40% and is NOT redundant --
// each lane already owns a distinct channel stripe of it -- so the same trick
// does not apply twice. The remaining idea, unpriced, is below.
//
// THE DOT DOES NOT HAVE TO BE SPLIT AT ALL, EITHER.
// Give each lane of the pair a DIFFERENT KEY instead: lane 0 runs the whole
// serial dot for `kp`, lane 1 runs the whole serial dot for `kp + 1`, and the
// pair exchanges the two scalars. Each dot is bit-for-bit the sum this file
// computes today -- same terms, same order, no re-association anywhere -- and
// the pair retires two keys for the loads of one. The online fold then applies
// `kp` and then `kp + 1` on both lanes, which is also the order it runs in
// now. The V accumulate is already channel-split by lane and does not change.
//
// That would halve the same loads WITHOUT re-associating anything, which is
// strictly better on the numerics -- but it is now competing with a shipped
// 8.3% rather than with nothing, so what it can still add is only the ulps,
// and it costs a `keeps` mask and a page boundary that stop agreeing across
// the pair. Left undone deliberately, and recorded so it is not rediscovered
// as a fresh idea.
//
//#if defined(PIE_SUBGROUP)
// THE DOT, SPLIT ACROSS THE `PIE_TX` LANES THAT USED TO DUPLICATE IT.
//
// `dot_row` above is called by every one of a row's `PIE_TX` lanes with the
// same two bases, so each of them issues the same `PIE_PAIRS` query loads and
// the same `PIE_PAIRS` key loads and arrives at the same scalar. The probe
// table in `compute_lane` prices that: the dot is 45% of this arm, its
// arithmetic is 4.4%, and its footprint is nothing -- so what the second lane
// costs is purely the loads it issues for a number the first lane already has.
//
// Here each lane walks a `PIE_TX`-strided stripe of word PAIRS and the pair is
// folded by a `log2(PIE_TX)` butterfly, which leaves every lane holding the
// whole sum -- which is what they need, because each then scales its own half
// of the value row by the same softmax weight.
//
// THE STRIPE IS IN PAIRS, NOT SINGLE WORDS, for the reason `dot_row` is
// unrolled by two: two independent loads in flight is worth 10.6% here and
// this loop is issue-bound. `PIE_PAIRS` is 32, 64, 128 or 256 and `PIE_TX` is
// 2, 2, 4 or 8, so `PIE_TX * 2` divides `PIE_PAIRS` at every head dimension.
//
// THE SUM IS RE-ASSOCIATED AND THAT IS ALLOWED HERE, on exactly the ground the
// split decode arm's butterfly already stands on a few hundred lines up: both
// orders are balanced trees over the same products, the difference is a few
// ulps of a score that then goes through `exp`, and `arena`'s tolerance covers
// it. What must not move is the order the KEYS fold into the running max, and
// that loop is untouched. This is also NOT the unroll-by-four hazard recorded
// in `dot_row`: that one broke because the compiler chose a different
// vectorisation for different variants of the same prompt, so two batch shapes
// disagreed. A stripe written out in the source is the same stripe in every
// batch.
//
// THE PARTNER LANES ARE ALWAYS ACTIVE TOGETHER. A row's lanes are linear
// indices `[lid.y * PIE_TX, lid.y * PIE_TX + PIE_TX)`, so an xor by anything
// below `PIE_TX` stays inside one row -- and every branch in the key loop
// (`keeps`, the page walk, the loop bound) depends only on `row` and `kp`,
// which the lanes of a row agree on. Rows diverge from each other, and that is
// fine, because no exchange ever crosses a row.
// AND THE QUERY STRIPE STAYS IN THE BUFFER. THIRD TIME.
//
// The query row is loop-invariant across every key, so each lane re-loads the
// same `PIE_PAIRS / PIE_TX` words -- 32 at `d_128` -- once per key, a third of
// what this function issues. Hoisting them into a `var<private>` array before
// the key loop and reading `pie_q_stripe[j]` here measured **300.7 ms against
// 285.9**: 5% SLOWER. A private array indexed by a loop variable is scratch
// memory, not registers, so the "hoist" traded a coalesced buffer load that
// hits L1 for a thread-local one that does not.
//
// This is the third independent time staging something in this loop has lost:
// the page table (a tie), the query in workgroup memory (16% slower), and now
// the query in private storage. The rule that keeps coming back is that
// nothing here is far from the core, so REMOVING a load wins and MOVING one
// does not -- which is exactly why the stripe below wins and this did not.
fn dot_row_split(q_base: u32, k_base: u32, lane: u32) -> f32 {
    var acc = 0.0;
    let q_word = q_base >> 1u;
    let k_word = k_base >> 1u;
    for (var w = lane * 2u; w < PIE_PAIRS; w = w + PIE_TX * 2u) {
        let q0 = queries[q_word + w]; let k0 = k_pages[k_word + w];
        let q1 = queries[q_word + w + 1u]; let k1 = k_pages[k_word + w + 1u];
        acc = acc + params.scale * pie_bf16_to_f32(q0 & 0xffffu) * pie_bf16_to_f32(k0 & 0xffffu);
        acc = acc + params.scale * pie_bf16_to_f32(q0 >> 16u) * pie_bf16_to_f32(k0 >> 16u);
        acc = acc + params.scale * pie_bf16_to_f32(q1 & 0xffffu) * pie_bf16_to_f32(k1 & 0xffffu);
        acc = acc + params.scale * pie_bf16_to_f32(q1 >> 16u) * pie_bf16_to_f32(k1 >> 16u);
    }
    for (var off = 1u; off < PIE_TX; off = off << 1u) {
        acc = acc + subgroupShuffleXor(acc, off);
    }
    return acc;
}
//#endif

fn compute_lane(row: u32, q_head: u32, lane: u32, n_q_heads: u32) {
    let req = req_of_token[row];
    let q_pos = position_ids[row];
//#if defined(PIE_FAST_FULL)
    var start = 0;
//#else
    var start = 0;
    if (params.window > 0 && q_pos >= params.window) { start = q_pos - params.window + 1; }
//#endif
    let q_base = q_base_for(row, q_head, n_q_heads);
    let o_base = o_base_for(row, q_head, n_q_heads);
    let kv_head = i32(q_head) / params.gqa_factor;

    var max_score = PIE_SDPA_NEG_INF;
    var sum_exp = 0.0;
    var acc: array<vec2<f32>, PIE_LANE_PAIRS>;
    for (var p = 0u; p < PIE_LANE_PAIRS; p = p + 1u) { acc[p] = vec2<f32>(0.0, 0.0); }

    // Loop-invariant, and it was the first of two dependent loads per key.
    let page_base = kv_page_indptr[u32(req)];
    // THE PAGE, HELD ACROSS THE KEYS THAT SHARE IT.
    //
    // `page_slot_at` divides by the page size and then loads
    // `kv_page_indices[page_base + page_ix]`, and that load's address depends
    // on the loop variable, so it is a dependent round trip. It was made
    // TWICE per key here -- once inside `dot_page` for the key row and once
    // again below for the value row, which compute the SAME `slot` from the
    // same arguments. This loop walks `kp` upward one at a time, so the page
    // index changes once every `params.page_size` keys and the physical page
    // is otherwise a constant: 2 divides and 2 dependent loads a key become
    // one divide and, at a 32-entry page, one load every 32.
    //
    // K and V share the row: both `dot_page`'s `k_base` and the `v_row` below
    // were `(slot * n_kv_heads + kv_head) * PIE_HEAD_DIM`, so it is computed
    // once now and the dot takes the row rather than the position.
    //
    // **AND IT IS WORTH NOTHING MEASURABLE**, which is the second time this
    // idea has been tried and the first time it has been priced properly.
    // `what_a_prefill_costs_at_length` reads 445.0 ms before and 443.3 after,
    // one sitting -- 0.4%, inside this harness's spread. The note in the
    // decode arm above records the same finding from a single noisy sample
    // and reverted on it; this keeps the change instead, because it is
    // strictly fewer divides and fewer dependent loads for identical answers
    // and because the reason it buys nothing is now known rather than
    // guessed: the page table is a few hundred bytes and every one of the 32
    // rows of every group walks it in the same order, so it is resident after
    // the first row and the "dependent round trip" is an L1 hit.
    //
    // The lesson is about the model of the machine, not the code. This note
    // used to end by naming the fix: Q and K stream from GLOBAL, 256 keys x
    // 64 words x two lanes a row with no reuse across the 32 rows of a group
    // that all read the SAME keys, so stage them in workgroup memory.
    //
    // THAT WAS WRONG, and it was wrong for the reason the paragraph above it
    // had already given. Staging the query was tried and measured 16% SLOWER
    // (see `dot_row`); the reuse it removes was L1 all along, exactly like
    // the page table, and the 8 KB of workgroup memory it costs is real. Two
    // independent findings now say the same thing: NOTHING IN THIS LOOP IS
    // FAR FROM THE CORE, and counting how many times a load is issued does
    // not price it here.
    //
    // What did work was issuing more loads at once rather than fewer in
    // total -- two words an iteration in `dot_row`, two values an iteration
    // in the V accumulate below, together 1444 -> 1634 tok/s at 512 rows.
    // The loop was short of memory-level parallelism, not of locality.
    //
    // # Where this loop's time actually goes, measured rather than argued
    //
    // The attention rectangle is 4.1 ms a layer and this model has 28 of
    // them, so 114.8 ms of a 313.4 ms 512-row prefill is spent right here.
    // Each term below was priced by HALVING that part of the loop and
    // re-running `what_a_prefill_costs_at_length`. The answers are wrong
    // while the probe is in, which is fine -- the probe is a stopwatch, not
    // a candidate.
    //
    // | part | probe | prefill ms | this loop's share |
    // | --- | --- | --- | --- |
    // | (baseline) | | 313.4 | |
    // | QK dot loads | `w < PIE_PAIRS / 2` | 287.5 | **45%** |
    // | V accumulate | `p < PIE_LANE_PAIRS / 2` | 290.7 | **40%** |
    // | everything else | by subtraction | | 15% |
    // | per-term `params.scale` | hoisted out of the dot | 312.0 | **~1%** |
    //
    // Read those four rows together and the kernel stops being mysterious:
    //
    //  * IT IS ALL LOAD. The dot and the V accumulate are 85% of it and both
    //    are word-at-a-time streaming; the remaining 15% covers the online
    //    softmax, the page walk and the loop itself TOGETHER.
    //
    //  * IT IS NOT ARITHMETIC. Hoisting `params.scale` out of the dot turns
    //    a multiply-and-fma per term into one fma -- half the inner loop's
    //    ALU -- and buys 0.4% of a prefill. Anything that only makes this
    //    loop's arithmetic cheaper, the hand-rolled bf16 unpack included, is
    //    already priced at approximately nothing. Do not go there.
    //
    //  * IT IS NOT THE SOFTMAX EITHER. The serial `max`/`exp` chain across
    //    keys is inside that 15% along with two other things, so it cannot
    //    be worth more than a few percent of a prefill however it is
    //    restructured.
    //
    // # And the loads are served by cache, not by DRAM
    //
    // Count the bytes this arm asks for at 512 rows: 512^2/2 causal pairs a
    // head, 16 heads, a 256-byte K row read by each of `PIE_TX` = 2 lanes
    // and a V row read once between them, is 1.61 GB a layer and 45 GB over
    // 28. Delivered in 114.8 ms that is **392 GB/s**, about twice this
    // part's DRAM bandwidth. So the working set is resident and every load
    // in here is already a hit -- which is the same conclusion the page
    // cache and the query staging each reached the hard way, now arrived at
    // from the traffic side.
    //
    // The one real inefficiency left is that `PIE_TX` lanes each compute the
    // WHOLE dot, so K is read `PIE_TX` times over. Splitting it and
    // combining with a subgroup reduction is bounded above by the 45% row:
    // at `d_128`, where `PIE_TX` is 2, a perfect free split is 25.9 ms of
    // 313, about 8% -- before paying for the reduction, and it re-associates
    // a sum that three backends agree on. Priced, and not worth it here. It
    // is a different question at `d_256` and `d_512`, where `PIE_TX` is 4
    // and 8 and the same redundancy is 4x and 8x.
    //
    // What WOULD change the shape is reading each K element once per several
    // query rows instead of once per row -- which is what a matrix unit
    // does, and `tests/cooperative.rs` is the evidence this part has one.
    var page_held = 0xffffffffu;
    var page_phys = 0u;
    for (var kp = start; kp <= q_pos; kp = kp + 1) {
        if (!keeps(row, kp, q_pos, start)) { continue; }
//#if defined(PIE_PAGE_SIZE) && PIE_PAGE_SIZE == 32
        let page_ix = u32(kp) >> 5u;
        let page_off = u32(kp) & 31u;
        let page_len = 32u;
//#else
        let page_ix = u32(kp / params.page_size);
        let page_off = u32(kp % params.page_size);
        let page_len = u32(params.page_size);
//#endif
        if (page_ix != page_held) {
            page_held = page_ix;
            page_phys = kv_page_indices[page_base + page_ix];
        }
        let slot = page_phys * page_len + page_off;
        let v_row = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM;
        // ONE dot for every pair this lane owns, which is the whole point.
//#if defined(PIE_SUBGROUP)
        let step = pie_sdpa_online_update(dot_row_split(q_base, v_row, lane), max_score, sum_exp);
//#else
        let step = pie_sdpa_online_update(dot_row(q_base, v_row), max_score, sum_exp);
//#endif
        max_score = step.max_score;
        sum_exp = step.sum_exp;
        // ONE load for the pair. `d_out` is even by construction, so both
        // halves live in `v_pages[(v_row + d_out) >> 1]` and the two `v_at`
        // calls this replaced fetched the same word twice.
        //
        // Unrolled by two for the reason `dot_row` is: the loads are
        // independent of each other and of the accumulate, so issuing two
        // before either arrives is free. `PIE_LANE_PAIRS` is `PIE_PAIRS /
        // PIE_TX`, which is 16 at `d_64` and 32 at every other head
        // dimension, so two always divides it.
        for (var p = 0u; p < PIE_LANE_PAIRS; p = p + 2u) {
            let d0 = (lane + p * PIE_TX) * 2u;
            let d1 = (lane + (p + 1u) * PIE_TX) * 2u;
            let v0 = v_pages[(v_row + d0) >> 1u];
            let v1 = v_pages[(v_row + d1) >> 1u];
            acc[p] = acc[p] * step.history_scale
                + step.score_scale
                    * vec2<f32>(pie_bf16_to_f32(v0 & 0xffffu), pie_bf16_to_f32(v0 >> 16u));
            acc[p + 1u] = acc[p + 1u] * step.history_scale
                + step.score_scale
                    * vec2<f32>(pie_bf16_to_f32(v1 & 0xffffu), pie_bf16_to_f32(v1 >> 16u));
        }
    }
//#if defined(PIE_WITH_SINK)
    let merged = pie_sdpa_merge_sink(sink_at(q_head), max_score, sum_exp);
    sum_exp = merged.sum_exp;
//#endif

    for (var p = 0u; p < PIE_LANE_PAIRS; p = p + 1u) {
        let d_out = (lane + p * PIE_TX) * 2u;
        var norm = acc[p];
//#if defined(PIE_WITH_SINK)
        norm = norm * merged.output_scale;
//#endif
        // A masked-out row keeps a zero denominator, and zero over zero is NaN
        // where the reference gives zero.
        if (sum_exp != 0.0) { norm = norm / sum_exp; }
        let at = (o_base + d_out) >> 1u;
        if (at < arrayLength(&out_)) { out_[at] = pie_pack_bf16(norm.x, norm.y); }
    }
}

// 32 x 8 and not 32 x 32: WebGPU guarantees only 256 invocations per workgroup,
// so the y lanes sweep the group's 32 rows four at a time instead of one lane
// per row. The GROUP still covers 32 rows, which keeps the host's grid
// arithmetic -- `ceil(n_rows / 32)` in y -- exactly as `kernels-vulkan` states
// it. Nothing in this arm barriers, so the `continue` below is a skip and not a
// hang.
@compute @workgroup_size(PIE_TX, PIE_TY)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let q_head = wg.x;
    for (var rr = lid.y; rr < 32u; rr = rr + PIE_TY) {
        let row = wg.y * 32u + rr;
        if (row >= u32(params.n_rows)) { continue; }
        compute_lane(row, q_head, lid.x, groups.x);
    }
}

//#else

//#if defined(PIE_COMBINE)

// THE SECOND HALF OF THE SPLIT: merge what the slices left.
//
// One workgroup per (row, query head) and `PIE_PAIRS` lanes in it, which is
// the same width the split arm's x axis has -- a lane owns the same channel
// pair here that it accumulated there, so its two accumulator floats are the
// only ones it reads.
//
// No workgroup memory and no barrier. The running max and the denominator are
// scalars every lane needs, and reading them from storage `n_splits` times per
// lane is cheaper than a rendezvous to share them: they are two floats at the
// head of a state the lane is already reading.
//
// The merge is the same online-softmax fold the split arm ends with, over
// `n_splits` states instead of `PIE_KB` -- and it is exact for the same
// reason. Merging softmax states over a disjoint cover of the keys gives the
// numerator and denominator of one pass over the union, which is what
// flash-decoding rests on.
@compute @workgroup_size(PIE_PAIRS, 1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let q_head = wg.x;
    let row = wg.y;
    let lane = lid.x;
    let n_q_heads = groups.x;
    let d_out = lane * 2u;
    let n_splits = max(u32(params.splits), 1u);
    let stride = 2u + PIE_HEAD_DIM;

    var max_score = PIE_SDPA_NEG_INF;
    var sum_exp = 0.0;
    var acc = vec2<f32>(0.0, 0.0);
    for (var t = 0u; t < n_splits; t = t + 1u) {
        let base = ((row * n_q_heads + q_head) * n_splits + t) * stride;
        let other_max = pie_split_state[base];
        // Both are the floor until a slice with keys arrives, and that floor
        // is FINITE for the reason the online include's header gives:
        // `exp(-inf - -inf)` is NaN where `exp(0)` is 1.
        let merged_max = max(max_score, other_max);
        let history_scale = exp(max_score - merged_max);
        let other_scale = exp(other_max - merged_max);
        max_score = merged_max;
        sum_exp = sum_exp * history_scale + pie_split_state[base + 1u] * other_scale;
        let other = vec2<f32>(
            pie_split_state[base + 2u + d_out],
            pie_split_state[base + 3u + d_out],
        );
        acc = acc * history_scale + other_scale * other;
    }

    var norm = acc;
    if (sum_exp != 0.0) { norm = acc / sum_exp; }
    let o_base = o_base_for(row, q_head, n_q_heads);
    let at = (o_base + d_out) >> 1u;
    if (at < arrayLength(&out_)) { out_[at] = pie_pack_bf16(norm.x, norm.y); }
}

//#else

@compute @workgroup_size(PIE_PAIRS, PIE_KB)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
//#if defined(PIE_SUBGROUP)
    @builtin(subgroup_size) sg: u32,
//#endif
) {
//#if !defined(PIE_SUBGROUP)
    // The baseline arm never shuffles, so the width is a placeholder that
    // makes `lim` the row width and the fold a single slot.
    let sg = PIE_DX;
//#endif
//#if defined(PIE_SPLITK)
    // THE SPLIT ON Z. The x and y axes are the unsplit arm's exactly -- one
    // workgroup per query head, one per row -- so only the third axis is new,
    // and `num_workgroups.z` is where the count comes from rather than a
    // second reading of the scalar.
    decode_row(wg.y, wg.x, lid.x, lid.y, groups.x, wg.z, groups.z, sg);
//#else
    decode_row(wg.y, wg.x, lid.x, lid.y, groups.x, 0u, 1u, sg);
//#endif
}

//#endif

//#endif

// pie:instantiate sdpa_paged_decode_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_paged_decode_bfloat16_d_128 PIE_HEAD_DIM=128
// pie:instantiate sdpa_paged_decode_bfloat16_d_256 PIE_HEAD_DIM=256
// pie:instantiate sdpa_paged_decode_bfloat16_d_512 PIE_HEAD_DIM=512
// pie:instantiate sdpa_paged_decode_bfloat16_d_64_p32 PIE_HEAD_DIM=64 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1
// pie:instantiate sdpa_paged_decode_bfloat16_d_128_p32 PIE_HEAD_DIM=128 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1
// pie:instantiate sdpa_paged_decode_bfloat16_d_64_p32_sg8 PIE_HEAD_DIM=64 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1 PIE_SHORT_GROUP=8
// pie:instantiate sdpa_paged_decode_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_WITH_SINK=1
// The split pair, at the widths a decode reaches. No sink point: a sink is a
// logit that joins the FINAL softmax, so it belongs to the combine arm alone,
// and no deployment with sinks has yet been narrow enough to want the split.
// pie:instantiate sdpa_paged_decode_split_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_SPLITK=1
// pie:instantiate sdpa_paged_decode_split_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_SPLITK=1
// pie:instantiate sdpa_paged_decode_split_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_SPLITK=1
// pie:instantiate sdpa_paged_decode_split_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_SPLITK=1
// THE SAME KERNEL WITH THE LADDER TAKEN OUT, at the one width a decode of the
// models this tree is tuned against actually reaches. `serve::pick` takes this
// when the adapter has `SUBGROUP` and falls back to the baseline line above
// when it does not, so nothing here is a requirement.
//
// Only `d_128` is minted. A tier variant costs a pipeline and seven pinned
// counts, and the other three widths are not on any measured path -- add one
// the day a model needs it and the bench says the same thing.
// pie:instantiate sdpa_paged_decode_split_bfloat16_d_128 @subgroup PIE_HEAD_DIM=128 PIE_SPLITK=1
// pie:instantiate sdpa_paged_decode_merge_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_SPLITK=1 PIE_COMBINE=1
// pie:instantiate sdpa_paged_decode_merge_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_SPLITK=1 PIE_COMBINE=1
// pie:instantiate sdpa_paged_decode_merge_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_SPLITK=1 PIE_COMBINE=1
// pie:instantiate sdpa_paged_decode_merge_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_SPLITK=1 PIE_COMBINE=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_64 @subgroup PIE_HEAD_DIM=64 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_128 @subgroup PIE_HEAD_DIM=128 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_256 @subgroup PIE_HEAD_DIM=256 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_512 @subgroup PIE_HEAD_DIM=512 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_WITH_SINK=1
// pie:instantiate sdpa_paged_tiled_strided_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_STRIDED=1
