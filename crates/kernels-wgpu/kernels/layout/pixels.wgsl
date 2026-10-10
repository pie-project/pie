//#include "common/bf16.inc.wgsl"

// Raw RGB patch bytes (HWC) to normalized bf16 rows, channel-major (order 0)
// or pixel-major (order 1), the frame repeated width / (3 patch^2) times.
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;
struct Params {
    side_px: i32,
    width: i32,
    order: i32,
    mean0: f32,
    mean1: f32,
    mean2: f32,
    sd0: f32,
    sd1: f32,
    sd2: f32,
    rows: i32,
}
@group(0) @binding(2) var<uniform> params: Params;

fn pixel(n: u32, c_out: u32) -> f32 {
    let pixels = u32(params.side_px * params.side_px);
    let in_width = 3u * pixels;
    let temporal = u32(params.width) / in_width;
    var ch = 0u;
    var at = 0u;
    if (params.order == 0) {
        ch = c_out / (temporal * pixels);
        at = (c_out % pixels) * 3u + ch;
    } else {
        let k = c_out % in_width;
        ch = k % 3u;
        at = k;
    }
    var mean = params.mean2;
    var sd = params.sd2;
    if (ch == 0u) {
        mean = params.mean0;
        sd = params.sd0;
    } else if (ch == 1u) {
        mean = params.mean1;
        sd = params.sd1;
    }
    let at_byte = n * in_width + at;
    let v = f32((x[at_byte >> 2u] >> ((at_byte & 3u) * 8u)) & 0xffu) / 255.0;
    return (v - mean) / sd;
}

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x * 2u;
    let n = gid.y;
    if (c >= u32(params.width) || n >= u32(params.rows)) {
        return;
    }
    y[(n * u32(params.width) + c) >> 1u] = pie_pack_bf16(pixel(n, c), pixel(n, c + 1u));
}

// pie:instantiate pixels_bf16 PIE_GROUP_X=256
