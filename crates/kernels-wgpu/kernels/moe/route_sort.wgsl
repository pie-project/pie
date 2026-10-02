//#include "common/bf16.inc.wgsl"

const ROUTER_MAX_EXPERTS = 1024u;

//#if defined(PIE_ROUTE_SORT)
@group(0) @binding(0) var<storage, read> expert_ids: array<i32>;
@group(0) @binding(1) var<storage, read_write> perm: array<i32>;
@group(0) @binding(2) var<storage, read_write> row_expert: array<i32>;
@group(0) @binding(3) var<storage, read_write> tile_expert: array<i32>;
@group(0) @binding(4) var<storage, read_write> inv: array<i32>;

struct Params {
    n: u32,
    n_experts: u32,
    experts_per_token: u32,
    tile_rows: u32,
    padded: u32,
    width: u32,
    x_pitch: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> counts: array<atomic<u32>, ROUTER_MAX_EXPERTS>;
var<workgroup> base_: array<u32, ROUTER_MAX_EXPERTS>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
    let lid = local.x;
    let lanes = 256u;
    let n_experts = min(params.n_experts, ROUTER_MAX_EXPERTS);
    let tile = max(params.tile_rows, 1u);
    let tiles = params.padded / tile;

    for (var e = lid; e < n_experts; e = e + lanes) {
        atomicStore(&counts[e], 0u);
    }
    for (var i = lid; i < params.padded; i = i + lanes) {
        perm[i] = -1;
        row_expert[i] = 0;
    }
    for (var t = lid; t < tiles; t = t + lanes) {
        tile_expert[t] = -1;
    }
    for (var i = lid; i < params.n; i = i + lanes) {
        inv[i] = -1;
    }
    workgroupBarrier();

    for (var i = lid; i < params.n; i = i + lanes) {
        let e = expert_ids[i];
        if (e >= 0 && u32(e) < n_experts) {
            atomicAdd(&counts[u32(e)], 1u);
        }
    }
    workgroupBarrier();

    if (lid == 0u) {
        var at = 0u;
        for (var e = 0u; e < n_experts; e = e + 1u) {
            let c = atomicLoad(&counts[e]);
            var span = 0u;
            if (c > 0u) {
                span = ((c + tile - 1u) / tile) * tile;
            }
            base_[e] = at;
            var t = at / tile;
            let end = (at + span) / tile;
            while (t < end && t < tiles) {
                tile_expert[t] = i32(e);
                t = t + 1u;
            }
            atomicStore(&counts[e], 0u);
            at = at + span;
        }
    }
    workgroupBarrier();

    for (var i = lid; i < params.n; i = i + lanes) {
        let e = expert_ids[i];
        if (e < 0 || u32(e) >= n_experts) {
            continue;
        }
        let previous = atomicAdd(&counts[u32(e)], 1u);
        let at = base_[u32(e)] + previous;
        if (at < params.padded) {
            perm[at] = i32(i);
            row_expert[at] = e;
            inv[i] = i32(at);
        }
    }
}

//#elif defined(PIE_ROUTE_GATHER)
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;
@group(0) @binding(2) var<storage, read> perm: array<i32>;

struct Params {
    n: u32,
    n_experts: u32,
    experts_per_token: u32,
    tile_rows: u32,
    padded: u32,
    width: u32,
    x_pitch: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word = gid.x;
    let half_width = params.width >> 1u;
    if (half_width == 0u) {
        return;
    }
    let r = word / half_width;
    if (r >= params.padded) {
        return;
    }
    let c = (word - r * half_width) << 1u;
    let sel = perm[r];
    if (sel < 0) {
        out_[word] = 0u;
        return;
    }
    let k = max(params.experts_per_token, 1u);
    var pitch = params.x_pitch;
    if (pitch == 0u) {
        pitch = params.width;
    }
    let src = (u32(sel) / k) * pitch + c;
    let lo = pie_f32_to_bf16(pie_bf16_at(x[src >> 1u], src));
    let hi = pie_f32_to_bf16(pie_bf16_at(x[(src + 1u) >> 1u], src + 1u));
    out_[word] = lo | (hi << 16u);
}

//#else
@group(0) @binding(0) var<storage, read> sorted: array<u32>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;
@group(0) @binding(2) var<storage, read> inv: array<i32>;

struct Params {
    n: u32,
    width: u32,
    zero: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word = gid.x;
    let half_width = params.width >> 1u;
    if (half_width == 0u) {
        return;
    }
    let r = word / half_width;
    if (r >= params.n) {
        return;
    }
    let c2 = word - r * half_width;
    let at = inv[r];
    if (at < 0) {
        y[word] = 0u;
        return;
    }
    y[word] = sorted[u32(at) * half_width + c2];
}
//#endif

// pie:instantiate route_sort PIE_ROUTE_SORT=1
// pie:instantiate route_gather PIE_ROUTE_GATHER=1
// pie:instantiate route_scatter PIE_ROUTE_SCATTER=1
