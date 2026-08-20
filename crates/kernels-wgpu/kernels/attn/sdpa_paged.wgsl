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

fn dot_page(q_base: u32, page_base: u32, kv_head: i32, kp: i32) -> f32 {
    let slot = page_slot_at(page_base, kp);
    let k_base = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM;
    var acc = 0.0;
    for (var d = 0u; d < PIE_HEAD_DIM; d = d + 1u) {
        // Scale per term, where `kernels-metal` and `kernels-vulkan` put it.
        // Hoisting it out of the loop is a different rounding, and a parity
        // walk between backends compares numbers.
        acc = acc + params.scale * q_at(q_base + d) * k_at(k_base + d);
    }
    return acc;
}

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
const PIE_KB: u32 = 256u / PIE_PAIRS;
// Blocks of keys one tree serves, on top of the `PIE_KB` the y axis already
// gives it. A rendezvous of every simdgroup in the workgroup is the cost this
// kernel pays most often, and a lane can carry several keys' partials as
// cheaply as one, so the tree folds `PIE_KR` blocks at a time and the barrier
// count falls by the same factor.
//
// Eight, not four: the value staging that used to sit beside the scores is
// gone (see the y-axis note in `decode_row`), so the budget it held pays for
// twice the blocking instead. `PIE_KB * PIE_PAIRS` is 256 at every
// instantiated head dim, so the score array is `PIE_KR` KiB everywhere -- 8
// KiB, plus 2 KiB of merge state, against the 16352-byte floor `wgpu`'s
// downlevel defaults guarantee. Sixteen does not fit that floor.
//
// Measured at 8 against 4: decode 102.9 -> 104.0 tok/s at 512 context and
// 71.2 -> 74.0 at 2048.
const PIE_KR: u32 = 8u;
const PIE_KSPAN: u32 = PIE_KR * PIE_KB;
var<workgroup> pie_sdpa_part: array<f32, PIE_KSPAN * PIE_PAIRS>;
// The per-y-lane softmax states, staged ONCE at the end of the row so the y
// axis can be folded away. See "THE Y AXIS CARRIES ITS OWN SOFTMAX" below for
// why the accumulator needs a slot per lane and the max and sum need one per
// y lane: the running max and denominator are functions of the SCORES, which
// every lane of a y row shares, while the accumulator is this lane's two head
// elements and nobody else's.
var<workgroup> pie_sdpa_macc: array<vec2<f32>, PIE_KB * PIE_PAIRS>;
var<workgroup> pie_sdpa_mmax: array<f32, PIE_KB>;
var<workgroup> pie_sdpa_msum: array<f32, PIE_KB>;

fn decode_row(row: u32, q_head: u32, lane: u32, ky: u32, n_q_heads: u32) {
    let d_out = lane * 2u;
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
    var acc = vec2<f32>(0.0, 0.0);
    // Loop-invariant, and it was the first of two dependent loads per key.
    let page_base = kv_page_indptr[u32(req)];
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
    var kp0 = start;
    while (kp0 <= q_pos) {
        let kn = min(i32(PIE_KSPAN), q_pos + 1 - kp0);
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
        var v_keep: array<vec2<f32>, PIE_KR>;
        for (var r = 0u; r < PIE_KR; r = r + 1u) {
            let q = r * PIE_KB + ky;
            var part = 0.0;
            var v_pair = vec2<f32>(0.0, 0.0);
            if (i32(q) < kn) {
                let slot = page_slot_at(page_base, kp0 + i32(q));
                let k_base = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM;
                v_pair = vec2<f32>(v_at(k_base + d_out), v_at(k_base + d_out + 1u));
                part = params.scale * q_at(q_base + d_out) * k_at(k_base + d_out)
                    + params.scale * q_at(q_base + d_out + 1u) * k_at(k_base + d_out + 1u);
            }
            v_keep[r] = v_pair;
            pie_sdpa_part[q * PIE_PAIRS + lane] = part;
        }
        workgroupBarrier();
        for (var half = PIE_PAIRS >> 1u; half > 0u; half = half >> 1u) {
            if (lane < half) {
                for (var r = 0u; r < PIE_KR; r = r + 1u) {
                    let at = (r * PIE_KB + ky) * PIE_PAIRS + lane;
                    pie_sdpa_part[at] = pie_sdpa_part[at] + pie_sdpa_part[at + half];
                }
            }
            workgroupBarrier();
        }
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
            let q = r * PIE_KB + ky;
            let kp = kp0 + i32(q);
            if (i32(q) < kn && keeps(row, kp, q_pos, start)) {
                let step = pie_sdpa_online_update(pie_sdpa_part[q * PIE_PAIRS], max_score, sum_exp);
                max_score = step.max_score;
                sum_exp = step.sum_exp;
                acc = acc * step.history_scale + step.score_scale * v_keep[r];
            }
        }
        // The next block's partials overwrite what this fold just read.
        workgroupBarrier();
        kp0 = kp0 + i32(PIE_KSPAN);
    }
    // THE MERGE. `PIE_KB` states, each over a disjoint set of this row's keys,
    // become one. Every lane runs the same fold so every lane leaves holding
    // the answer, which is what the sink merge and the single writer below
    // already assumed.
    pie_sdpa_macc[ky * PIE_PAIRS + lane] = acc;
    if (lane == 0u) {
        pie_sdpa_mmax[ky] = max_score;
        pie_sdpa_msum[ky] = sum_exp;
    }
    workgroupBarrier();
    max_score = PIE_SDPA_NEG_INF;
    sum_exp = 0.0;
    acc = vec2<f32>(0.0, 0.0);
    for (var t = 0u; t < PIE_KB; t = t + 1u) {
        let other_max = pie_sdpa_mmax[t];
        let merged_max = max(max_score, other_max);
        // Both are `PIE_SDPA_NEG_INF` until a state with keys arrives, and
        // that floor is finite for the reason the header of the online include
        // gives: `exp(-inf - -inf)` is NaN and `exp(0)` is 1.
        let history_scale = exp(max_score - merged_max);
        let other_scale = exp(other_max - merged_max);
        max_score = merged_max;
        sum_exp = sum_exp * history_scale + pie_sdpa_msum[t] * other_scale;
        acc = acc * history_scale + other_scale * pie_sdpa_macc[t * PIE_PAIRS + lane];
    }
//#if defined(PIE_WITH_SINK)
    let merged = pie_sdpa_merge_sink(sink_at(q_head), max_score, sum_exp);
    acc = acc * merged.output_scale;
    sum_exp = merged.sum_exp;
//#endif

    var norm = acc;
    if (sum_exp != 0.0) { norm = acc / sum_exp; }
    let at = (o_base + d_out) >> 1u;
    // Every y lane ran the merge above, so they all hold this answer and one
    // writes it. The redundancy is `PIE_KB` fused multiply-adds, once per row.
    if (ky == 0u && at < arrayLength(&out_)) { out_[at] = pie_pack_bf16(norm.x, norm.y); }
}

//#endif

//#if defined(PIE_TILED)

// How many output pairs one x-lane of a tile owns.
//
// The tile is 32 lanes wide on x and a row has `PIE_PAIRS` pairs, so a lane
// takes every 32nd one: 1 at `d_64`, 2 at `d_128`, 4 at `d_256`, 8 at `d_512`.
const PIE_LANE_PAIRS: u32 = PIE_PAIRS / 32u;

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
    for (var kp = start; kp <= q_pos; kp = kp + 1) {
        if (!keeps(row, kp, q_pos, start)) { continue; }
        // ONE dot for every pair this lane owns, which is the whole point.
        let step = pie_sdpa_online_update(
            dot_page(q_base, page_base, kv_head, kp), max_score, sum_exp);
        max_score = step.max_score;
        sum_exp = step.sum_exp;
        let slot = page_slot_at(page_base, kp);
        let v_row = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM;
        for (var p = 0u; p < PIE_LANE_PAIRS; p = p + 1u) {
            let d_out = (lane + p * 32u) * 2u;
            acc[p] = acc[p] * step.history_scale
                + step.score_scale * vec2<f32>(v_at(v_row + d_out), v_at(v_row + d_out + 1u));
        }
    }
//#if defined(PIE_WITH_SINK)
    let merged = pie_sdpa_merge_sink(sink_at(q_head), max_score, sum_exp);
    sum_exp = merged.sum_exp;
//#endif

    for (var p = 0u; p < PIE_LANE_PAIRS; p = p + 1u) {
        let d_out = (lane + p * 32u) * 2u;
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
@compute @workgroup_size(32, 8)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let q_head = wg.x;
    for (var rr = lid.y; rr < 32u; rr = rr + 8u) {
        let row = wg.y * 32u + rr;
        if (row >= u32(params.n_rows)) { continue; }
        compute_lane(row, q_head, lid.x, groups.x);
    }
}

//#else

@compute @workgroup_size(PIE_PAIRS, PIE_KB)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    decode_row(wg.y, wg.x, lid.x, lid.y, groups.x);
}

//#endif

// pie:instantiate sdpa_paged_decode_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_paged_decode_bfloat16_d_128 PIE_HEAD_DIM=128
// pie:instantiate sdpa_paged_decode_bfloat16_d_256 PIE_HEAD_DIM=256
// pie:instantiate sdpa_paged_decode_bfloat16_d_512 PIE_HEAD_DIM=512
// pie:instantiate sdpa_paged_decode_bfloat16_d_64_p32 PIE_HEAD_DIM=64 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1
// pie:instantiate sdpa_paged_decode_bfloat16_d_128_p32 PIE_HEAD_DIM=128 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1
// pie:instantiate sdpa_paged_decode_bfloat16_d_64_p32_sg8 PIE_HEAD_DIM=64 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1 PIE_SHORT_GROUP=8
// pie:instantiate sdpa_paged_decode_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_WITH_SINK=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_WITH_SINK=1
// pie:instantiate sdpa_paged_tiled_strided_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_STRIDED=1
