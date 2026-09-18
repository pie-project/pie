//#include "common/bf16.inc.wgsl"

const PIE_CONV_HIST_MAX = 64;

//#if defined(PIE_COMMITTED)
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> indptr: array<i32>;
@group(0) @binding(2) var<storage, read> replay: array<i32>;
@group(0) @binding(3) var<storage, read> commit: array<i32>;
@group(0) @binding(4) var<storage, read> slots: array<i32>;
@group(0) @binding(5) var<storage, read> weight: array<u32>;
@group(0) @binding(6) var<storage, read_write> state: array<f32>;
@group(0) @binding(7) var<storage, read_write> y: array<atomic<u32>>;

struct Params {
    lane0: i32,
    channels: i32,
    conv_width: i32,
    dilation: i32,
}
@group(0) @binding(8) var<uniform> params: Params;
//#elif defined(PIE_CHUNKED)
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> indptr: array<i32>;
@group(0) @binding(2) var<storage, read> weight: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read> slots: array<u32>;
@group(0) @binding(5) var<storage, read_write> y: array<atomic<u32>>;

struct Params {
    channels: i32,
    conv_width: i32,
    dilation: i32,
}
@group(0) @binding(6) var<uniform> params: Params;
//#else
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
@group(0) @binding(3) var<storage, read> slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<atomic<u32>>;

struct Params {
    channels: i32,
    conv_width: i32,
    dilation: i32,
}
@group(0) @binding(5) var<uniform> params: Params;
//#endif

fn silu(z: f32) -> f32 {
    return z / (1.0 + exp(-z));
}

fn load_x(i: u32) -> f32 {
    return pie_bf16_at(x[i >> 1u], i);
}

fn load_w(i: u32) -> f32 {
    return pie_bf16_at(weight[i >> 1u], i);
}

fn store_y(i: u32, v: f32) {
    let at = i >> 1u;
    let b = pie_f32_to_bf16(v);
    if ((i & 1u) == 1u) {
        atomicAnd(&y[at], 0x0000ffffu);
        atomicOr(&y[at], b << 16u);
    } else {
        atomicAnd(&y[at], 0xffff0000u);
        atomicOr(&y[at], b);
    }
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let r = gid.y;
    let chans = u32(params.channels);
    if (c >= chans) {
        return;
    }
    let width = params.conv_width;
    let dil = params.dilation;
    let hist = (width - 1) * dil + 1;
    let tap0 = c * u32(width);
    var past: array<f32, PIE_CONV_HIST_MAX>;

//#if defined(PIE_COMMITTED)
    let lane0 = u32(params.lane0);
    var begin = indptr[r];
    for (var j = 0u; j < r; j = j + 1u) {
        begin = begin + replay[lane0 + j];
    }
    let span = (indptr[r + 1u] - indptr[r]) + replay[lane0 + r];
    if (span <= 0) {
        return;
    }
    let slot = slots[lane0 + r];
    if (slot < 0) {
        return;
    }
    let slab = u32(slot) * u32(hist) * chans;

    for (var s = 0; s < hist; s = s + 1) {
        past[s] = state[slab + u32(s) * chans + c];
    }
    for (var t = 0; t < span; t = t + 1) {
        var acc = 0.0;
        for (var k = 0; k < width; k = k + 1) {
            let src = t - (width - 1 - k) * dil;
            var tap = 0.0;
            if (src < 0) {
                tap = past[hist + src];
            } else {
                tap = load_x(u32(begin + src) * chans + c);
            }
            acc = acc + tap * load_w(tap0 + u32(k));
        }
        store_y(u32(begin + t) * chans + c, silu(acc));
    }
    let keep = min(commit[lane0 + r], span);
    if (keep <= 0) {
        return;
    }

    for (var s = 0; s < hist; s = s + 1) {
        let src = keep - hist + s;
        var v = 0.0;
        if (src < 0) {
            v = past[hist + src];
        } else {
            v = load_x(u32(begin + src) * chans + c);
        }
        state[slab + u32(s) * chans + c] = v;
    }
//#elif defined(PIE_CHUNKED)
    let begin = indptr[r];
    let end = indptr[r + 1u];
    if (end <= begin) {
        return;
    }
    let span = end - begin;
    let slab = slots[u32(begin)] * u32(hist) * chans;

    for (var s = 0; s < hist; s = s + 1) {
        past[s] = state[slab + u32(s) * chans + c];
    }
    for (var t = 0; t < span; t = t + 1) {
        var acc = 0.0;
        for (var k = 0; k < width; k = k + 1) {
            let src = t - (width - 1 - k) * dil;
            var tap = 0.0;
            if (src < 0) {
                tap = past[hist + src];
            } else {
                tap = load_x(u32(begin + src) * chans + c);
            }
            acc = acc + tap * load_w(tap0 + u32(k));
        }
        store_y(u32(begin + t) * chans + c, silu(acc));
    }
    for (var s = 0; s < hist; s = s + 1) {
        let src = span - hist + s;
        var v = 0.0;
        if (src < 0) {
            v = past[hist + src];
        } else {
            v = load_x(u32(begin + src) * chans + c);
        }
        state[slab + u32(s) * chans + c] = v;
    }
//#else
    let slab = slots[r] * u32(hist) * chans;
    for (var s = 0; s < hist; s = s + 1) {
        past[s] = state[slab + u32(s) * chans + c];
    }
    let fresh = load_x(r * chans + c);
    var acc = 0.0;
    for (var k = 0; k + 1 < width; k = k + 1) {
        acc = acc + past[k * dil + 1] * load_w(tap0 + u32(k));
    }
    acc = acc + fresh * load_w(tap0 + u32(width - 1));
    store_y(r * chans + c, silu(acc));
    for (var s = 0; s + 1 < hist; s = s + 1) {
        state[slab + u32(s) * chans + c] = past[s + 1];
    }
    state[slab + u32(hist - 1) * chans + c] = fresh;
//#endif
}

// pie:instantiate causal_conv1d_bf16 PIE_GROUP_X=256
// pie:instantiate causal_conv1d_chunked_bf16 PIE_GROUP_X=256 PIE_CHUNKED=1
// pie:instantiate causal_conv1d_committed_bf16 PIE_GROUP_X=256 PIE_COMMITTED=1
