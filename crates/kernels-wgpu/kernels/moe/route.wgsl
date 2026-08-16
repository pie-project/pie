// Generic MoE routing: choose experts, group the rows by expert, gather them,
// and combine the sorted results back.
//
// Five kernels in one file, the way `moe/route.comp` and `route.metal` have
// them: they share the three param blocks and the routing vocabulary, and a
// split would put `route_sort`'s output layout and `combine_sorted`'s reading
// of it in two places that can drift.
//
// Three things here are this backend's and not the port's.
//
// **A workgroup is 256 invocations, not 1024.** WebGPU's guaranteed
// `maxComputeInvocationsPerWorkgroup` is 256, where the GLSL sibling declares
// `local_size_x = 1024` and gives every expert of a 1024-expert router its own
// lane. Every loop over experts here therefore STRIDES by the workgroup width
// instead of assuming one item per lane -- which is the `gated_rms` lesson
// (`.wiki/new-driver/vulkan.md` §9) applied before it can bite: a body that
// wrote one slot per lane would leave three quarters of a 1024-slot staging
// array holding whatever the last dispatch left there, and the selection would
// read it as a logit.
//
// **`atomicAdd` needs an `atomic<u32>`, and the counting sort's counters are
// WORKGROUP memory.** That is what makes the sort expressible at all: WGSL
// forbids an atomic operation on a plain `array<u32>` and forbids a non-atomic
// read of an `array<atomic<u32>>`, and the prefix pass reads the counters
// plainly. In `var<workgroup>` that is one declaration and one `atomicLoad`;
// had the row put the counters in a storage buffer, the same buffer could not
// be both in one module and the sort would need a second binding or a second
// pass. The row does not, so it does not.
//
// **A bf16 tensor is `array<u32>`, two values to a word.** WGSL has no 16-bit
// storage type at all (`common/bf16.inc.wgsl` says why), so every bf16 index in
// the GLSL body is a HALF-index here that has to be split -- and every bf16
// STORE in this file lands in a word whose other half belongs to a different
// invocation, sometimes in a different workgroup. Each store says what it
// concluded; the short version is that WGSL has no sub-word atomic, so a
// read-modify-write of a shared word loses one of the two values, and the
// stores are spelled as an atomic AND (clear my half) followed by an atomic OR
// (set my half) instead. Those two touch only the writer's sixteen bits, so
// whatever order the four operations of a shared word land in, each half ends
// at its own writer's value.

//#include "common/bf16.inc.wgsl"
//#include "moe/params.inc.wgsl"

// The widest routing the tree compiles for. A `var<workgroup>` is sized by a
// const-expression in WGSL, so the staging arrays are sized for the ceiling and
// the launch's own `n_experts` bounds the loops.
const ROUTER_MAX_TOPK = 16u;
const ROUTER_MAX_EXPERTS = 1024u;

// The workgroup width the two whole-routing kernels run at, named once because
// every stride below is it. 256 is WebGPU's floor and 1024 is the sibling's;
// see this file's header.
const ROUTER_LANES = 256u;

//#if defined(PIE_ROUTER_TOPK)
@group(0) @binding(0) var<storage, read_write> logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> expert_ids: array<i32>;
// ATOMIC, and it is the one declaration in this arm that is not obvious.
//
// Lane 0 of a row's workgroup writes that row's `k` weights, so the halves of
// one word belong to two DIFFERENT workgroups whenever `k` is odd: row r's last
// weight and row r+1's first weight land in one word, and two workgroups doing
// a read-modify-write of it keep one value and drop the other. Every
// checkpoint the tree has seen routes to an even `k` (2, 4 or 8) and would
// never show it. `atomicAnd` + `atomicOr` costs two instructions per weight --
// `k` of them per row -- and is correct for every `k` rather than for the even
// ones.
@group(0) @binding(2) var<storage, read_write> expert_weights: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> params: RouterParams;
@group(0) @binding(4) var<storage, read_write> per_expert_scale: array<u32>;

var<workgroup> s_logits: array<f32, ROUTER_MAX_EXPERTS>;
var<workgroup> chosen: array<f32, ROUTER_MAX_TOPK>;
var<workgroup> chosen_i: array<u32, ROUTER_MAX_TOPK>;

// The bf16 half-index split, spelled per buffer.
//
// `common/bf16.inc.wgsl` has `pie_load_bf16(&logits, i)` and it cannot be
// called: naga 30 rejects a `ptr<storage, ...>` FUNCTION PARAMETER outright
// (`unrestricted_pointer_parameters` is unimplemented), and the module would
// parse and then fail `create_shader_module` on every device. So the widening
// -- which is the part that can be got wrong -- still goes through the
// fragment's `pie_bf16_to_f32`, and only the subscript is local.
fn load_logit(i: u32) -> f32 {
    let word = logits[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

//#if defined(PIE_SCALED)
fn load_scale(i: u32) -> f32 {
    let word = per_expert_scale[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}
//#endif

// See the declaration of `expert_weights` for why this is two atomics and not
// `pie_store_bf16`.
fn store_weight(i: u32, x: f32) {
    let at = i >> 1u;
    let v = pie_f32_to_bf16(x);
    if ((i & 1u) == 1u) {
        atomicAnd(&expert_weights[at], 0x0000ffffu);
        atomicOr(&expert_weights[at], v << 16u);
    } else {
        atomicAnd(&expert_weights[at], 0xffff0000u);
        atomicOr(&expert_weights[at], v);
    }
}

@compute @workgroup_size(ROUTER_LANES)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let lane = lid.x;
    // One workgroup per ROW, on the grid's y -- `LaunchRule::RouterLane`. The
    // row axis is load-bearing and was once missing: at `grid.y = 1` a mixture
    // prefill routed row 0 and left every other row's ids whatever the previous
    // layer wrote.
    let row = wid.y;
    let n = min(params.n_experts, ROUTER_MAX_EXPERTS);
    let k = min(params.experts_per_token, ROUTER_MAX_TOPK);
    let pitch = select(params.logits_pitch, n, params.logits_pitch == 0u);
    let neg = -3.0e38;

    // Four slots per lane at the 1024 ceiling, not one: see the header.
    // Everything past `n` is stamped with the neutral element rather than left
    // alone, because the selection below scans the whole array.
    for (var e = lane; e < ROUTER_MAX_EXPERTS; e = e + ROUTER_LANES) {
        var v = neg;
        if (e < n) {
            v = load_logit(row * pitch + e);
        }
        s_logits[e] = v;
    }
    workgroupBarrier();

    // The selection is one lane's, as it is in both siblings: `k` passes over
    // `n` logits is a few thousand comparisons for a router, and a parallel
    // top-k would need a second staging array and three more barriers to answer
    // the same question. The barrier above is what the other 255 lanes are for.
    if (lane == 0u) {
        // The softmax over ALL experts, taken before the selection eats the
        // array: the winners are stamped with `neg` as they are chosen, so a
        // sum computed afterwards would be a sum over the losers.
        var all_max = neg;
        for (var e = 0u; e < n; e = e + 1u) {
            all_max = max(all_max, s_logits[e]);
        }
        var all_sum = 0.0;
        for (var e = 0u; e < n; e = e + 1u) {
            all_sum = all_sum + exp(s_logits[e] - all_max);
        }

        for (var r = 0u; r < k; r = r + 1u) {
            var best = neg;
            var best_i = 0xffffffffu;
            for (var e = 0u; e < n; e = e + 1u) {
                let v = s_logits[e];
                if (v > best) {
                    best = v;
                    best_i = e;
                }
            }
            chosen[r] = best;
            chosen_i[r] = best_i;
            if (best_i < n) {
                s_logits[best_i] = neg;
            }
        }

        var mx = all_max;
        var sum = all_sum;
        if (params.softmax_over_all == 0u) {
            mx = neg;
            for (var r = 0u; r < k; r = r + 1u) {
                mx = max(mx, chosen[r]);
            }
            sum = 0.0;
            for (var r = 0u; r < k; r = r + 1u) {
                sum = sum + exp(chosen[r] - mx);
            }
        }
        for (var r = 0u; r < k; r = r + 1u) {
            let e = chosen_i[r];
            var w = exp(chosen[r] - mx) / sum;
//#if defined(PIE_SCALED)
            // The per-expert scale is indexed by the EXPERT, not by the slot:
            // it is a property of the expert the row chose, and `r` is only
            // where in this row's top-k that expert came.
            w = w * load_scale(e);
//#endif
            expert_ids[row * k + r] = i32(e);
            store_weight(row * k + r, w);
        }
    }
}

//#elif defined(PIE_ROUTE_SORT)
@group(0) @binding(0) var<storage, read_write> expert_ids: array<i32>;
@group(0) @binding(1) var<storage, read_write> perm: array<i32>;
@group(0) @binding(2) var<storage, read_write> row_expert: array<i32>;
@group(0) @binding(3) var<storage, read_write> tile_expert: array<i32>;
@group(0) @binding(4) var<storage, read_write> params: MoeRouteParams;
@group(0) @binding(5) var<storage, read_write> inv: array<i32>;

// The counting sort's two arrays. See the header for why `counts` being
// workgroup memory is what makes an atomic expressible here.
var<workgroup> counts: array<atomic<u32>, ROUTER_MAX_EXPERTS>;
var<workgroup> expert_base: array<u32, ROUTER_MAX_EXPERTS>;

// ONE workgroup for the whole routing -- `LaunchRule::RouterSort`. Not one per
// row: N copies of this would each clear and rewrite the permutation the others
// are reading, which is why the rule was split from `RouterLane`.
@compute @workgroup_size(ROUTER_LANES)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let experts = min(params.n_experts, ROUTER_MAX_EXPERTS);
    let tile = max(params.tile_rows, 1u);
    let tiles = params.padded / tile;

    // Every one of these four is a stride over the workgroup width, and that is
    // not an optimisation: at 256 lanes and a `padded` in the thousands, a body
    // that wrote index `lane` alone would leave the tail of the permutation
    // holding the previous routing's rows -- which are valid indices, so the
    // gather would read real activations for the wrong rows.
    for (var e = lane; e < experts; e = e + ROUTER_LANES) {
        atomicStore(&counts[e], 0u);
    }
    for (var i = lane; i < params.padded; i = i + ROUTER_LANES) {
        perm[i] = -1;
        row_expert[i] = 0;
    }
    for (var t = lane; t < tiles; t = t + ROUTER_LANES) {
        tile_expert[t] = -1;
    }
    for (var i = lane; i < params.n; i = i + ROUTER_LANES) {
        inv[i] = -1;
    }
    workgroupBarrier();

    for (var i = lane; i < params.n; i = i + ROUTER_LANES) {
        let e = expert_ids[i];
        if (e >= 0 && u32(e) < experts) {
            atomicAdd(&counts[u32(e)], 1u);
        }
    }
    workgroupBarrier();

    // The exclusive scan is one lane's, and it has to be: each expert's base is
    // the running sum of every earlier expert's PADDED span, so the pass is
    // sequential by construction. It also stamps the tile ownership and resets
    // the counters for their second use as cursors.
    if (lane == 0u) {
        var at = 0u;
        for (var e = 0u; e < experts; e = e + 1u) {
            let c = atomicLoad(&counts[e]);
            var span = 0u;
            if (c > 0u) {
                // Rounded UP to a whole number of tiles: a routed GEMM reads
                // one expert per row tile, so an expert whose rows half-fill a
                // tile must still own the whole tile or the tail rows would be
                // multiplied by its neighbour's weights.
                span = ((c + tile - 1u) / tile) * tile;
            }
            expert_base[e] = at;
            for (var t = at / tile; t < (at + span) / tile && t < tiles; t = t + 1u) {
                tile_expert[t] = i32(e);
            }
            atomicStore(&counts[e], 0u);
            at = at + span;
        }
    }
    workgroupBarrier();

    // The placement. WHICH slot inside an expert's span a row lands in is
    // decided by the order the atomic increments happen to land in, exactly as
    // it is in the GLSL and Metal siblings -- the sort is stable in neither, and
    // the 256-lane stripe here only changes which unstable order comes out.
    // What IS guaranteed, and what the GPU suite checks, is the STRUCTURE: every
    // routed row gets exactly one slot, inside its own expert's span, and `inv`
    // is the exact inverse of `perm` for every row that got one. A race cannot
    // hide behind that, because `combine_sorted` reads back through `inv` and a
    // lost or doubled slot shows up as a row that never comes back.
    for (var i = lane; i < params.n; i = i + ROUTER_LANES) {
        let e = expert_ids[i];
        if (e < 0 || u32(e) >= experts) {
            continue;
        }
        let at = expert_base[u32(e)] + atomicAdd(&counts[u32(e)], 1u);
        if (at < params.padded) {
            perm[at] = i32(i);
            row_expert[at] = e;
            inv[i] = i32(at);
        }
    }
}

//#elif defined(PIE_ROUTE_GATHER)
@group(0) @binding(0) var<storage, read_write> x: array<u32>;
// Atomic for the same reason `router_topk`'s weights are: this writes ONE bf16
// per invocation at `r * width + c`, so the two halves of a word are two
// invocations -- `c` and `c + 1`, or the last column of a row and the first of
// the next when `width` is odd. Both cases put them in different workgroups for
// some `c`, and a read-modify-write then keeps one and drops the other.
@group(0) @binding(1) var<storage, read_write> out_: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> perm: array<i32>;
@group(0) @binding(3) var<storage, read_write> params: MoeRouteParams;

// The gather moves BITS, not numbers: the GLSL sibling assigns `out_[..] =
// x[..]` with no widening, and a round trip through f32 would be a rounding
// step on a value that is only being copied. So these two work on the raw half
// and the only bf16 arithmetic here is `pie_f32_to_bf16(0.0)` for a pad row.
fn load_x_bits(i: u32) -> u32 {
    let word = x[i >> 1u];
    return select(word & 0xffffu, word >> 16u, (i & 1u) == 1u);
}

fn store_out_bits(i: u32, b: u32) {
    let at = i >> 1u;
    if ((i & 1u) == 1u) {
        atomicAnd(&out_[at], 0x0000ffffu);
        atomicOr(&out_[at], b << 16u);
    } else {
        atomicAnd(&out_[at], 0xffff0000u);
        atomicOr(&out_[at], b);
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let r = gid.y;
    // `dispatch_workgroups` counts WORKGROUPS, so both axes round up to 16 and
    // the last group of each runs past the data.
    if (c >= params.width || r >= params.padded) {
        return;
    }
    let sel = perm[r];
    let k = max(params.experts_per_token, 1u);
    let pitch = select(params.x_pitch, params.width, params.x_pitch == 0u);
    // A padding slot is written with zero rather than skipped: the row exists in
    // the gathered tensor, the GEMM will read it, and whatever the buffer held
    // is not zero.
    var v = pie_f32_to_bf16(0.0);
    if (sel >= 0) {
        // `sel` is a (row, slot) PAIR index and `x` is indexed by ROW, which is
        // what the division by `k` is. The two are not the same number and the
        // gather is the only place that has to know it.
        v = load_x_bits((u32(sel) / k) * pitch + c);
    }
    store_out_bits(r * params.width + c, v);
}

//#elif defined(PIE_COMBINE_SORTED)
@group(0) @binding(0) var<storage, read_write> y: array<u32>;
@group(0) @binding(1) var<storage, read_write> expert_weights: array<u32>;
// Atomic: one bf16 per invocation again, and with a row pitch the pairing is
// worse than the gather's -- the partner half of a boundary word can be a row's
// PADDING, which no invocation writes, or the next row's first column, which
// another workgroup writes. Two atomics settle both without the body having to
// know which case it is in.
@group(0) @binding(2) var<storage, read_write> out_: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> params: ExpertCombineParams;
@group(0) @binding(4) var<storage, read_write> inv: array<i32>;

fn load_y(i: u32) -> f32 {
    let word = y[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn load_weight(i: u32) -> f32 {
    let word = expert_weights[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn store_out(i: u32, v: f32) {
    let at = i >> 1u;
    let b = pie_f32_to_bf16(v);
    if ((i & 1u) == 1u) {
        atomicAnd(&out_[at], 0x0000ffffu);
        atomicOr(&out_[at], b << 16u);
    } else {
        atomicAnd(&out_[at], 0xffff0000u);
        atomicOr(&out_[at], b);
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let row = gid.y;
    if (c >= params.width) {
        return;
    }
    var acc = 0.0;
    for (var e = 0u; e < params.experts_per_token; e = e + 1u) {
        // `inv` is the sort's inverse: where THIS (row, slot) pair ended up in
        // the sorted order, or -1 if it was never placed. Reading `y` by
        // `row * k + e` instead would read the sorted tensor at an unsorted
        // index and blend a different row's expert output in.
        let slot = row * params.experts_per_token + e;
        let at = inv[slot];
        if (at >= 0) {
            acc = acc + load_weight(slot) * load_y(u32(at) * params.width + c);
        }
    }
    let pitch = select(params.out_pitch, params.width, params.out_pitch == 0u);
    let h = row * pitch + c;
    // The row count is not in the block -- the grid carries it -- so the tail
    // guard is the buffer's own length, which is the descriptor range the shell
    // already bound. An overshot row is then a no-op instead of a write past
    // the tensor.
    if ((h >> 1u) >= arrayLength(&out_)) {
        return;
    }
    store_out(h, acc);
}

//#else
@group(0) @binding(0) var<storage, read_write> routed: array<u32>;
@group(0) @binding(1) var<storage, read_write> shared_: array<u32>;
@group(0) @binding(2) var<storage, read_write> gate: array<u32>;
// May ALIAS `routed`, which is fine and is why `routed[at]` is read before
// `out_[at]` is written, by the same invocation, in program order. The atomic
// is for the neighbouring HALF of the word -- see the other arms -- and not for
// the alias: the alias is exact, one address, and each element has one writer.
@group(0) @binding(3) var<storage, read_write> out_: array<atomic<u32>>;

struct Params {
    width: u32,
//#if defined(PIE_STRIDED)
    row_pitch: i32,
//#endif
}
@group(1) @binding(0) var<uniform> params: Params;

fn load_routed(i: u32) -> f32 {
    let word = routed[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn load_shared(i: u32) -> f32 {
    let word = shared_[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn load_gate(i: u32) -> f32 {
    let word = gate[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn store_out(i: u32, v: f32) {
    let at = i >> 1u;
    let b = pie_f32_to_bf16(v);
    if ((i & 1u) == 1u) {
        atomicAnd(&out_[at], 0x0000ffffu);
        atomicOr(&out_[at], b << 16u);
    } else {
        atomicAnd(&out_[at], 0xffff0000u);
        atomicOr(&out_[at], b);
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let r = gid.y;
    if (c >= params.width) {
        return;
    }
    // A row's DATA BASE and a row's GATE INDEX are two different numbers, and
    // the Vulkan port collapsed them into one variable: it read `gate[r*width]`
    // where Metal reads `gate[r]`. The gate is ONE value per row, so for a
    // `rows`-long allocation the wrong index leaves the buffer almost
    // immediately and blends every row but row 0 with a garbage weight. The
    // test that should have caught it allocated `rows * width` and made the
    // wrong index representable.
    //
    // The strided variant is NOT a copy of this and the difference is the whole
    // point: there the gate really does stride by the pitch, because
    // `qmv_out_size` answers 1 for the shared gate projection and so its single
    // output column is written a full pitch apart like every other projection's.
    // Both halves are stated in `route.metal`; the port collapsed the one where
    // they differ.
//#if defined(PIE_STRIDED)
    let base = r * u32(params.row_pitch);
    let gate_at = base;
//#else
    let base = r * params.width;
    let gate_at = r;
//#endif
    let g = 1.0 / (1.0 + exp(-load_gate(gate_at)));
    let at = base + c;
    if ((at >> 1u) >= arrayLength(&out_)) {
        return;
    }
    store_out(at, load_routed(at) + g * load_shared(at));
}
//#endif

// pie:instantiate combine_sorted PIE_COMBINE_SORTED=1
// pie:instantiate route_gather PIE_ROUTE_GATHER=1
// pie:instantiate route_sort PIE_ROUTE_SORT=1
// pie:instantiate router_topk_bfloat16 PIE_ROUTER_TOPK=1
// pie:instantiate router_topk_scaled_bfloat16 PIE_ROUTER_TOPK=1 PIE_SCALED=1
// pie:instantiate shared_expert_combine
// pie:instantiate shared_expert_combine_strided PIE_STRIDED=1
