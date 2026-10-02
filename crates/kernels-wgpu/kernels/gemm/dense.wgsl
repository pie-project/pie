//#include "common/bf16.inc.wgsl"
//#if defined(PIE_SUBGROUP)
//#include "common/subgroup.inc.wgsl"
//#endif

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;

struct Params {
    m: i32,
    n: i32,
    k: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

fn load_x(row: u32, kk: u32) -> f32 {
    let i = row * u32(params.k) + kk;
    return pie_bf16_at(x[i >> 1u], i);
}

fn load_w(col: u32, kk: u32) -> f32 {
    let i = col * u32(params.k) + kk;
    return pie_bf16_at(w[i >> 1u], i);
}

fn write_pair(row: u32, col: u32, lo: f32, hi: f32) {
    if (row >= u32(params.m) || col >= u32(params.n)) {
        return;
    }
    out_[(row * u32(params.n) + col) >> 1u] = pie_pack_bf16(lo, hi);
}

//#if defined(PIE_GEMV)
const PIE_KLANES = 32u;
const PIE_NLANES = 8u;

const PIE_WORDS = 4u;
const PIE_WIDE = PIE_KLANES * PIE_WORDS;
var<workgroup> partial: array<f32, 256>;

fn pair_dot(xw: u32, ww: u32) -> f32 {
    return pie_bf16_to_f32(xw & 0xffffu) * pie_bf16_to_f32(ww & 0xffffu)
         + pie_bf16_to_f32(xw >> 16u) * pie_bf16_to_f32(ww >> 16u);
}

@compute @workgroup_size(32, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let klane = local.x;
    let clane = local.y;
    let col = gid.y;
    let row = gid.z;
    var acc = 0.0;
    if (col < u32(params.n) && row < u32(params.m)) {
        let k = u32(params.k);

        let words = k >> 1u;
        var wide = 0u;
//#if !defined(PIE_NARROW)
        if ((k & 1u) == 0u) {
            wide = (words / PIE_WIDE) * PIE_WIDE;
        }
//#endif
        let xbase = row * words;
        let wbase = col * words;
        var wi = klane * PIE_WORDS;
        for (; wi < wide; wi = wi + PIE_WIDE) {
            let xa = xbase + wi;
            let wa = wbase + wi;
            acc = acc + pair_dot(x[xa], w[wa]);
            acc = acc + pair_dot(x[xa + 1u], w[wa + 1u]);
            acc = acc + pair_dot(x[xa + 2u], w[wa + 2u]);
            acc = acc + pair_dot(x[xa + 3u], w[wa + 3u]);
        }

        var kk = wide * 2u + klane;
        for (; kk < k; kk = kk + PIE_KLANES) {
            acc = acc + load_x(row, kk) * load_w(col, kk);
        }
    }
    let slot = clane * PIE_KLANES + klane;
//#if defined(PIE_SUBGROUP)

    let folded = pie_subgroup_sum32(acc);
    if (klane == 0u) {
        partial[clane] = folded;
    }
    workgroupBarrier();
    if (klane == 0u && (clane & 1u) == 0u) {
        write_pair(row, col, partial[clane], partial[clane + 1u]);
    }
//#else
    partial[slot] = acc;
    workgroupBarrier();
    for (var step = PIE_KLANES / 2u; step > 0u; step = step >> 1u) {
        if (klane < step) {
            partial[slot] = partial[slot] + partial[slot + step];
        }
        workgroupBarrier();
    }
    if (klane == 0u && (clane & 1u) == 0u) {
        write_pair(row, col, partial[clane * PIE_KLANES], partial[(clane + 1u) * PIE_KLANES]);
    }
//#endif
}
//#else

const PIE_BN = 64u;
const PIE_BK = 16u;
const PIE_BN4 = 16u;
const PIE_LANES = 256u;
const PIE_RPL = PIE_BM / 16u;

var<workgroup> xs: array<vec4<f32>, PIE_BK * (PIE_BM / 4u)>;

var<workgroup> ws: array<vec4<f32>, PIE_BK * PIE_BN4>;

fn load_x_word(row: u32, kk: u32) -> u32 {
    return x[(row * u32(params.k) + kk) >> 1u];
}

fn load_w_word(col: u32, kk: u32) -> u32 {
    return w[(col * u32(params.k) + kk) >> 1u];
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let tx = local.x;
    let ty = local.y;
    let flat = tx + 16u * ty;
    let tile_row = group.y * PIE_BM;
    let tile_col = group.x * PIE_BN;
    let m = u32(params.m);
    let n = u32(params.n);
    let k = u32(params.k);
    let k_even = (k & 1u) == 0u;
    var acc: array<vec4<f32>, PIE_RPL>;
    for (var i = 0u; i < PIE_RPL; i++) {
        acc[i] = vec4<f32>(0.0);
    }
    for (var kb = 0u; kb < k; kb = kb + PIE_BK) {
        workgroupBarrier();

        for (var e = flat; e < PIE_BM * 8u; e = e + PIE_LANES) {
            let r = e / 8u;
            let kk = (e - r * 8u) * 2u;
            let row = tile_row + r;
            var lo = 0.0;
            var hi = 0.0;
            if (row < m && kb + kk < k) {
                if (k_even) {
                    let word = load_x_word(row, kb + kk);
                    lo = pie_bf16_to_f32(word & 0xffffu);
                    if (kb + kk + 1u < k) {
                        hi = pie_bf16_to_f32(word >> 16u);
                    }
                } else {
                    let i0 = row * k + kb + kk;
                    lo = pie_bf16_at(x[i0 >> 1u], i0);
                    if (kb + kk + 1u < k) {
                        hi = pie_bf16_at(x[(i0 + 1u) >> 1u], i0 + 1u);
                    }
                }
            }
            xs[kk * (PIE_BM / 4u) + r / 4u][r & 3u] = lo;
            xs[(kk + 1u) * (PIE_BM / 4u) + r / 4u][r & 3u] = hi;
        }

        for (var e = flat; e < PIE_BN * 8u; e = e + PIE_LANES) {
            let c = e / 8u;
            let kk = (e - c * 8u) * 2u;
            let col = tile_col + c;
            var lo = 0.0;
            var hi = 0.0;
            if (col < n && kb + kk < k) {
                if (k_even) {
                    let word = load_w_word(col, kb + kk);
                    lo = pie_bf16_to_f32(word & 0xffffu);
                    if (kb + kk + 1u < k) {
                        hi = pie_bf16_to_f32(word >> 16u);
                    }
                } else {
                    let i0 = col * k + kb + kk;
                    lo = pie_bf16_at(w[i0 >> 1u], i0);
                    if (kb + kk + 1u < k) {
                        hi = pie_bf16_at(w[(i0 + 1u) >> 1u], i0 + 1u);
                    }
                }
            }
            ws[kk * PIE_BN4 + c / 4u][c & 3u] = lo;
            ws[(kk + 1u) * PIE_BN4 + c / 4u][c & 3u] = hi;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < PIE_BK; kk = kk + 1u) {
            let wv = ws[kk * PIE_BN4 + tx];
//#if PIE_BM == 64
            let xv = xs[kk * (PIE_BM / 4u) + ty];
            acc[0] = acc[0] + xv.x * wv;
            acc[1] = acc[1] + xv.y * wv;
            acc[2] = acc[2] + xv.z * wv;
            acc[3] = acc[3] + xv.w * wv;
//#else
            let xv = xs[kk * (PIE_BM / 4u) + ty / 4u][ty & 3u];
            acc[0] = acc[0] + xv * wv;
//#endif
        }
    }
    let col = tile_col + tx * 4u;
    for (var i = 0u; i < PIE_RPL; i++) {
        let row = tile_row + ty * PIE_RPL + i;
        write_pair(row, col, acc[i].x, acc[i].y);
        write_pair(row, col + 2u, acc[i].z, acc[i].w);
    }
}
//#endif

// pie:instantiate dense_gemv_t_bf16 PIE_GEMV=1
// pie:instantiate dense_gemm_t_bf16 PIE_BM=64
// pie:instantiate dense_gemm_t16_bf16 PIE_BM=16
// pie:instantiate dense_gemv_t_bf16 @subgroup PIE_GEMV=1

// pie:instantiate dense_gemv_narrow_bf16 PIE_GEMV=1 PIE_NARROW=1
// pie:instantiate dense_gemv_narrow_bf16 @subgroup PIE_GEMV=1 PIE_NARROW=1
