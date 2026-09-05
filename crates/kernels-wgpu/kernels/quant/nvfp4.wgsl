//#include "common/bf16.inc.wgsl"

const PIE_LANES = 32u;
const PIE_SLICES = 2u;
const PIE_ROWS_PER_SLICE = 2u;
const PIE_ROWS = PIE_SLICES * PIE_ROWS_PER_SLICE;

const PIE_NVFP4_GROUP = 16u;

@group(0) @binding(0) var<storage, read> codes: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<u32>;

struct Params {
    out_vec_size: i32,
    in_vec_size: i32,
    vecs: i32,
    tensor_scale: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partials: array<f32, PIE_ROWS * PIE_LANES>;

const kNvfp4Lut = array<f32, 16>(
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);

fn code_byte(at: u32) -> u32 {
    return (codes[at >> 2u] >> ((at & 3u) * 8u)) & 0xffu;
}

fn scale_byte(at: u32) -> u32 {
    return (scales[at >> 2u] >> ((at & 3u) * 8u)) & 0xffu;
}

fn e4m3_to_f32(byte_: u32) -> f32 {
    let exp = i32((byte_ >> 3u) & 0xfu);
    let mant = i32(byte_ & 0x7u);
    var mag = 0.0;
    if (exp == 0) {
        mag = f32(mant) * 0.001953125;
    } else if (exp == 0xf && mant == 0x7) {
        mag = bitcast<f32>(0x7fc00000u);
    } else {
        mag = (1.0 + f32(mant) * 0.125) * bitcast<f32>(u32((exp - 7 + 127) << 23u));
    }
    return select(mag, -mag, (byte_ & 0x80u) != 0u);
}

@compute @workgroup_size(32, 2, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let lid = local.x;
    let ly = local.y;
    let token = group.y;
    let n = u32(max(params.out_vec_size, 0));
    let k = u32(max(params.in_vec_size, 0));
    let groups = k / PIE_NVFP4_GROUP;
    let code_bytes = k / 2u;
    let out0 = group.x * PIE_ROWS + ly * PIE_ROWS_PER_SLICE;

    var acc = array<f32, 2>(0.0, 0.0);

    for (var g = lid; g < groups; g = g + PIE_LANES) {
        let xb = token * k + g * PIE_NVFP4_GROUP;
        for (var r = 0u; r < PIE_ROWS_PER_SLICE; r = r + 1u) {
            let row = out0 + r;
            if (row >= n) {
                continue;
            }
            let sc = e4m3_to_f32(scale_byte(row * groups + g));
            let base = row * code_bytes + g * (PIE_NVFP4_GROUP / 2u);
            var part = 0.0;
            for (var j = 0u; j < PIE_NVFP4_GROUP / 2u; j = j + 1u) {
                let byte_ = code_byte(base + j);
                let lo = xb + 2u * j;
                part = part + kNvfp4Lut[byte_ & 0x0fu] * pie_bf16_at(x[lo >> 1u], lo);
                part = part + kNvfp4Lut[byte_ >> 4u] * pie_bf16_at(x[(lo + 1u) >> 1u], lo + 1u);
            }
            acc[r] = acc[r] + sc * part;
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
            let lo = params.tensor_scale * partials[(ly * PIE_ROWS_PER_SLICE) * PIE_LANES];
            let hi = params.tensor_scale * partials[(ly * PIE_ROWS_PER_SLICE + 1u) * PIE_LANES];
            y[(token * n + row) >> 1u] = pie_pack_bf16(lo, hi);
        }
    }
}

// pie:instantiate nvfp4_qmv_bf16
