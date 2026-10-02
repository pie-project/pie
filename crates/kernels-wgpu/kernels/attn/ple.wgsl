//#include "common/u64.inc.wgsl"

const PIE_MAX_NGRAM = 4;
const PIE_MAX_HEADS = 32;

fn ple_mask_window(window: ptr<function, array<i32, 4>>, ngram: i32, eos: i32) {
    var crossed = false;
    for (var p = 1; p < ngram; p = p + 1) {
        if (crossed) {
            (*window)[p] = eos;
        }
        if ((*window)[p] == eos) {
            crossed = true;
        }
    }
}

fn hash_at(i: u32) -> U64 {
    return U64(hash[2u * i], hash[2u * i + 1u]);
}

fn ple_hash_row(window: ptr<function, array<i32, 4>>, ngram: i32, heads: i32, heads_per_ngram: i32) -> array<i32, 32> {
    var out_: array<i32, 32>;
    for (var k = 0; k < PIE_MAX_HEADS; k = k + 1) {
        out_[k] = 0;
    }
    for (var order = 2; order <= ngram; order = order + 1) {
        var mixed = u64_mul(u64_from_i32((*window)[0]), hash_at(0u));
        for (var p = 1; p < order; p = p + 1) {
            mixed = u64_xor(mixed, u64_mul(u64_from_i32((*window)[p]), hash_at(u32(p))));
        }
        let base = (order - 2) * heads_per_ngram;
        for (var k = 0; k < heads_per_ngram; k = k + 1) {
            let head = base + k;
            let prime = hash_at(u32(ngram + head));
            let offset = hash_at(u32(ngram + heads + head));
            out_[head] = bitcast<i32>(u64_add(u64_mod(mixed, prime), offset).lo);
        }
    }
    return out_;
}

fn ple_cell(cell: i32, eos: i32) -> i32 {
    if (cell == 0) {
        return eos;
    }
    return cell - 1;
}

//#if defined(PIE_PLE_UPDATE)

@group(0) @binding(0) var<storage, read> ids: array<i32>;
@group(0) @binding(1) var<storage, read_write> state: array<i32>;
@group(0) @binding(2) var<storage, read> slots: array<u32>;
@group(0) @binding(3) var<storage, read> hash: array<u32>;
@group(0) @binding(4) var<storage, read_write> ngram_ids: array<i32>;
struct Params {
    ngram: i32,
    heads: i32,
    heads_per_ngram: i32,
    eos: i32,
    rows: i32,
}
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= u32(params.rows)) {
        return;
    }
    let span = params.ngram - 1;
    let slab = slots[r] * u32(span);
    var window: array<i32, 4>;
    for (var p = 0; p < PIE_MAX_NGRAM; p = p + 1) {
        window[p] = 0;
    }
    let fresh = ids[r];
    window[0] = fresh;
    for (var p = 1; p <= span; p = p + 1) {
        window[p] = ple_cell(state[slab + u32(span - p)], params.eos);
    }
    ple_mask_window(&window, params.ngram, params.eos);
    let out_ = ple_hash_row(&window, params.ngram, params.heads, params.heads_per_ngram);
    for (var k = 0; k < params.heads; k = k + 1) {
        ngram_ids[r * u32(params.heads) + u32(k)] = out_[k];
    }
    for (var p = 0; p + 1 < span; p = p + 1) {
        state[slab + u32(p)] = state[slab + u32(p + 1)];
    }
    state[slab + u32(span - 1)] = fresh + 1;
}

//#elif defined(PIE_PLE_CHUNKED)

@group(0) @binding(0) var<storage, read> ids: array<i32>;
@group(0) @binding(1) var<storage, read> indptr: array<i32>;
@group(0) @binding(2) var<storage, read_write> state: array<i32>;
@group(0) @binding(3) var<storage, read> slots: array<u32>;
@group(0) @binding(4) var<storage, read> hash: array<u32>;
@group(0) @binding(5) var<storage, read_write> ngram_ids: array<i32>;
struct Params {
    ngram: i32,
    heads: i32,
    heads_per_ngram: i32,
    eos: i32,
    lanes: i32,
}
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= u32(params.lanes)) {
        return;
    }
    let begin = indptr[r];
    let end = indptr[r + 1u];
    if (end <= begin) {
        return;
    }
    let rows = end - begin;
    let span = params.ngram - 1;
    let slab = slots[u32(begin)] * u32(span);
    for (var t = 0; t < rows; t = t + 1) {
        var window: array<i32, 4>;
        for (var p = 0; p < PIE_MAX_NGRAM; p = p + 1) {
            window[p] = 0;
        }
        window[0] = ids[u32(begin + t)];
        for (var p = 1; p <= span; p = p + 1) {
            if (t - p >= 0) {
                window[p] = ids[u32(begin + t - p)];
            } else {
                window[p] = ple_cell(state[slab + u32(span - (p - t))], params.eos);
            }
        }
        ple_mask_window(&window, params.ngram, params.eos);
        let out_ = ple_hash_row(&window, params.ngram, params.heads, params.heads_per_ngram);
        for (var k = 0; k < params.heads; k = k + 1) {
            ngram_ids[u32(begin + t) * u32(params.heads) + u32(k)] = out_[k];
        }
    }
    var next: array<i32, 4>;
    for (var p = 0; p < PIE_MAX_NGRAM; p = p + 1) {
        next[p] = 0;
    }
    for (var p = 0; p < span; p = p + 1) {
        let src = rows - span + p;
        if (src >= 0) {
            next[p] = ids[u32(begin + src)] + 1;
        } else {
            next[p] = state[slab + u32(p + rows)];
        }
    }
    for (var p = 0; p < span; p = p + 1) {
        state[slab + u32(p)] = next[p];
    }
}

//#else

@group(0) @binding(0) var<storage, read> ids: array<i32>;
@group(0) @binding(1) var<storage, read> indptr: array<i32>;
@group(0) @binding(2) var<storage, read> replay: array<i32>;
@group(0) @binding(3) var<storage, read> commit: array<i32>;
@group(0) @binding(4) var<storage, read> slots: array<i32>;
@group(0) @binding(5) var<storage, read_write> state: array<i32>;
@group(0) @binding(6) var<storage, read> hash: array<u32>;
@group(0) @binding(7) var<storage, read_write> ngram_ids: array<i32>;
struct Params {
    lane0: i32,
    ngram: i32,
    heads: i32,
    heads_per_ngram: i32,
    eos: i32,
    lanes: i32,
}
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= u32(params.lanes)) {
        return;
    }
    var begin = indptr[r];
    for (var j = 0u; j < r; j = j + 1u) {
        begin = begin + replay[u32(params.lane0) + j];
    }
    let replayed = replay[u32(params.lane0) + r];
    let rows = (indptr[r + 1u] - indptr[r]) + replayed;
    if (rows <= 0) {
        return;
    }
    let slot = slots[u32(params.lane0) + r];
    if (slot < 0) {
        return;
    }
    let span = params.ngram - 1;
    let slab = u32(slot) * u32(span);
    for (var t = 0; t < rows; t = t + 1) {
        var window: array<i32, 4>;
        for (var p = 0; p < PIE_MAX_NGRAM; p = p + 1) {
            window[p] = 0;
        }
        window[0] = ids[u32(begin + t)];
        for (var p = 1; p <= span; p = p + 1) {
            if (t - p >= 0) {
                window[p] = ids[u32(begin + t - p)];
            } else {
                window[p] = ple_cell(state[slab + u32(span - (p - t))], params.eos);
            }
        }
        ple_mask_window(&window, params.ngram, params.eos);
        if (t < replayed) {
            continue;
        }
        let out_ = ple_hash_row(&window, params.ngram, params.heads, params.heads_per_ngram);
        let own = u32(indptr[r] + (t - replayed));
        for (var k = 0; k < params.heads; k = k + 1) {
            ngram_ids[own * u32(params.heads) + u32(k)] = out_[k];
        }
    }
    var keep = commit[u32(params.lane0) + r];
    if (keep > rows) {
        keep = rows;
    }
    if (keep <= 0) {
        return;
    }
    var next: array<i32, 4>;
    for (var p = 0; p < PIE_MAX_NGRAM; p = p + 1) {
        next[p] = 0;
    }
    for (var p = 0; p < span; p = p + 1) {
        let src = keep - span + p;
        if (src >= 0) {
            next[p] = ids[u32(begin + src)] + 1;
        } else {
            next[p] = state[slab + u32(p + keep)];
        }
    }
    for (var p = 0; p < span; p = p + 1) {
        state[slab + u32(p)] = next[p];
    }
}
//#endif

// pie:instantiate ple_ngram_ids_update PIE_PLE_UPDATE=1 PIE_GROUP_X=64
// pie:instantiate ple_ngram_ids_chunked PIE_PLE_CHUNKED=1 PIE_GROUP_X=64
// pie:instantiate ple_ngram_ids_committed PIE_GROUP_X=64
