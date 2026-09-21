//#include "common/bf16.inc.wgsl"

const PIE_LANES = 32u;
const PIE_SLICES = 2u;
const PIE_ROWS_PER_SLICE = 2u;
const PIE_ROWS = PIE_SLICES * PIE_ROWS_PER_SLICE;

const PIE_SUPER = 256u;

@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> y: array<u32>;

struct Params {
    out_vec_size: i32,
    in_vec_size: i32,
    vecs: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partials: array<f32, PIE_ROWS * PIE_LANES>;

fn wbyte(at: u32) -> u32 {
    return (w[at >> 2u] >> ((at & 3u) * 8u)) & 0xffu;
}

fn gguf_f16(base: u32) -> f32 {
    return unpack2x16float(wbyte(base) | (wbyte(base + 1u) << 8u)).x;
}

fn q4k_scale_min(base: u32, sub: u32) -> vec2<f32> {
    if (sub < 4u) {
        return vec2<f32>(
            f32(wbyte(base + sub) & 63u),
            f32(wbyte(base + sub + 4u) & 63u),
        );
    }
    let a = wbyte(base + sub + 4u);
    let b = wbyte(base + sub - 4u);
    let c = wbyte(base + sub);
    return vec2<f32>(
        f32((a & 0x0fu) | ((b >> 6u) << 4u)),
        f32((a >> 4u) | ((c >> 6u) << 4u)),
    );
}

fn q3k_scale(base: u32, sub: u32) -> i32 {
    let grp = sub >> 2u;
    let j = sub & 3u;
    let src = wbyte(base + select(0u, 4u, (grp & 1u) != 0u) + j);
    let low = select(src >> 4u, src & 0x0fu, grp < 2u);
    let top = (wbyte(base + 8u + j) >> (2u * grp)) & 3u;
    return i32(low | (top << 4u));
}

@compute @workgroup_size(32, 2, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let lid = local.x;
    let ly = local.y;
    let token = group.y;
    let n = u32(max(params.out_vec_size, 0));
    let k = u32(max(params.in_vec_size, 0));
    let blocks = k / PIE_SUPER;
    let row_bytes = blocks * u32(PIE_BLOCK_BYTES);
    let out0 = group.x * PIE_ROWS + ly * PIE_ROWS_PER_SLICE;

    var acc = array<f32, 2>(0.0, 0.0);

    for (var g = lid; g < blocks; g = g + PIE_LANES) {
        let xb = token * k + g * PIE_SUPER;
        for (var r = 0u; r < PIE_ROWS_PER_SLICE; r = r + 1u) {
            let row = out0 + r;
            if (row >= n) {
                continue;
            }
            let base = row * row_bytes + g * u32(PIE_BLOCK_BYTES);
            var sum = 0.0;

//#if PIE_SCHEME == 2
            let d = gguf_f16(base + 80u);
            let dmin = gguf_f16(base + 82u);
            for (var b = 0u; b < 16u; b = b + 1u) {
                let shift = 2u * ((b >> 1u) & 3u);
                let at = 16u + (b >> 3u) * 32u + (b & 1u) * 16u;
                var part = 0.0;
                var xsum = 0.0;
                for (var l = 0u; l < 16u; l = l + 1u) {
                    let e = xb + b * 16u + l;
                    let xv = pie_bf16_at(x[e >> 1u], e);
                    xsum = xsum + xv;
                    part = part + f32((wbyte(base + at + l) >> shift) & 3u) * xv;
                }
                let packed = wbyte(base + b);
                sum = sum + d * f32(packed & 0x0fu) * part
                          - dmin * f32(packed >> 4u) * xsum;
            }
//#endif
//#if PIE_SCHEME == 3
            let d = gguf_f16(base + 108u);
            for (var b = 0u; b < 16u; b = b + 1u) {
                let step = (b >> 1u) & 3u;
                let shift = 2u * step;
                let selector = 1u << ((b >> 3u) * 4u + step);
                let at = 32u + (b >> 3u) * 32u + (b & 1u) * 16u;
                let mask_at = (b & 1u) * 16u;
                var part = 0.0;
                for (var l = 0u; l < 16u; l = l + 1u) {
                    let e = xb + b * 16u + l;
                    let xv = pie_bf16_at(x[e >> 1u], e);
                    let code = i32((wbyte(base + at + l) >> shift) & 3u);
                    let borrow = select(4, 0, (wbyte(base + mask_at + l) & selector) != 0u);
                    part = part + f32(code - borrow) * xv;
                }
                sum = sum + d * f32(q3k_scale(base + 96u, b) - 32) * part;
            }
//#endif
//#if PIE_SCHEME == 4
            let d = gguf_f16(base);
            let dmin = gguf_f16(base + 2u);
            for (var b = 0u; b < 8u; b = b + 1u) {
                let pair = b >> 1u;
                let high = (b & 1u) != 0u;
                var part = 0.0;
                var xsum = 0.0;
                for (var i = 0u; i < 32u; i = i + 1u) {
                    let e = xb + b * 32u + i;
                    let xv = pie_bf16_at(x[e >> 1u], e);
                    xsum = xsum + xv;
                    let byte_ = wbyte(base + 16u + pair * 32u + i);
                    part = part + f32(select(byte_ & 0x0fu, byte_ >> 4u, high)) * xv;
                }
                let sm = q4k_scale_min(base + 4u, b);
                sum = sum + d * sm.x * part - dmin * sm.y * xsum;
            }
//#endif
//#if PIE_SCHEME == 5
            let d = gguf_f16(base);
            let dmin = gguf_f16(base + 2u);
            for (var b = 0u; b < 8u; b = b + 1u) {
                let pair = b >> 1u;
                let high = (b & 1u) != 0u;
                var part = 0.0;
                var xsum = 0.0;
                for (var i = 0u; i < 32u; i = i + 1u) {
                    let e = xb + b * 32u + i;
                    let xv = pie_bf16_at(x[e >> 1u], e);
                    xsum = xsum + xv;
                    let byte_ = wbyte(base + 48u + pair * 32u + i);
                    let low = select(byte_ & 0x0fu, byte_ >> 4u, high);
                    let fifth = (wbyte(base + 16u + i) >> b) & 1u;
                    part = part + f32(low | (fifth << 4u)) * xv;
                }
                let sm = q4k_scale_min(base + 4u, b);
                sum = sum + d * sm.x * part - dmin * sm.y * xsum;
            }
//#endif
//#if PIE_SCHEME == 6
            let d = gguf_f16(base + 208u);
            for (var half_ = 0u; half_ < 2u; half_ = half_ + 1u) {
                for (var quarter = 0u; quarter < 4u; quarter = quarter + 1u) {
                    for (var sub = 0u; sub < 2u; sub = sub + 1u) {
                        var part = 0.0;
                        for (var t = 0u; t < 16u; t = t + 1u) {
                            let i = sub * 16u + t;
                            let e = xb + half_ * 128u + quarter * 32u + i;
                            let xv = pie_bf16_at(x[e >> 1u], e);
                            let byte_ = wbyte(base + half_ * 64u + i + 32u * (quarter & 1u));
                            let low = select(byte_ >> 4u, byte_ & 0x0fu, quarter < 2u);
                            let top = (wbyte(base + 128u + half_ * 32u + i) >> (2u * quarter)) & 3u;
                            part = part + f32(i32(low | (top << 4u)) - 32) * xv;
                        }
                        let raw = i32(wbyte(base + 192u + half_ * 8u + sub + 2u * quarter));
                        sum = sum + d * f32(select(raw, raw - 256, raw > 127)) * part;
                    }
                }
            }
//#endif
            acc[r] = acc[r] + sum;
        }
    }

    for (var r = 0u; r < PIE_ROWS_PER_SLICE; r = r + 1u) {
        partials[(ly * PIE_ROWS_PER_SLICE + r) * PIE_LANES + lid] = acc[r];
    }
    workgroupBarrier();
    for (var s = PIE_LANES / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            for (var r = 0u; r < PIE_ROWS_PER_SLICE; r = r + 1u) {
                let at = (ly * PIE_ROWS_PER_SLICE + r) * PIE_LANES;
                partials[at + lid] = partials[at + lid] + partials[at + lid + s];
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let row = out0;
        if (row < n) {
            let lo = partials[(ly * PIE_ROWS_PER_SLICE) * PIE_LANES];
            let hi = partials[(ly * PIE_ROWS_PER_SLICE + 1u) * PIE_LANES];
            y[(token * n + row) >> 1u] = pie_pack_bf16(lo, hi);
        }
    }
}

// pie:instantiate kquant_q2k_bf16 PIE_SCHEME=2 PIE_BLOCK_BYTES=84
// pie:instantiate kquant_q3k_bf16 PIE_SCHEME=3 PIE_BLOCK_BYTES=110
// pie:instantiate kquant_q4k_bf16 PIE_SCHEME=4 PIE_BLOCK_BYTES=144
// pie:instantiate kquant_q5k_bf16 PIE_SCHEME=5 PIE_BLOCK_BYTES=176
// pie:instantiate kquant_q6k_bf16 PIE_SCHEME=6 PIE_BLOCK_BYTES=210
