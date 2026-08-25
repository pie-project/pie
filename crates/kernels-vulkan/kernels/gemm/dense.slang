// The dense bf16 projection: `y[M, N] = act[M, K] @ w[N, K]^T`.
//
// The point the whole plane was waiting on. `scripts/planes.sh` said of this
// backend that "what is left is a lane, not a kernel: no catalog row binds
// here, because `gemm` is unclaimed on this plane", and this file is the
// kernel half of closing that.
//
// ## THE LAYOUT, WHICH IS THE ONE THING TO GET RIGHT
//
// `w` is row-major `[N, K]` with K CONTIGUOUS -- output-major, so element
// `w[n][k]` lives at `w + n * K + k` and the contraction runs down each weight
// ROW. That is not a choice this file makes, it is what the point declares:
// `#[shape(y = [act.rows, w.axis(0)])]` says axis 0 of the weight is the output
// width. `kernels-cuda` states the same thing as a cuBLAS call --
// `cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, .., w, lda = k, act,
// ldb = k, .., y, ldc = n)`, the standard column-major idiom for a row-major
// `C = A * B^T` -- and both sibling shader planes index `w[col * k + kk]`.
//
// A reader's instinct is that a `[K, N]` weight would let the inner loop walk
// contiguously in `n`. It would, and it is not what the checkpoints hold. The
// transpose is FREE HERE because the tile is staged: the K-major weight slab
// below is read once per tile and transposed on the way into shared memory, so
// the inner product reads `ws[kk][c]` contiguously in `c` anyway.
//
// ## THE ACCUMULATE IS fp32 AND THE STORAGE IS bf16
//
// Both operands widen to `float` on load, the running sum is `float`, and the
// single narrowing is on the way out. This matches CUDA's `COMPUTE` type on
// the `cublasGemmEx` above and both siblings. A bf16 accumulate over a 5120-
// long contraction is not a smaller version of this kernel, it is a different
// answer.
//
// ## TWO ARMS, AND THE SEAM BETWEEN THEM IS M = 32
//
// `dense_gemm_t_bfloat16_bm_32_bn_32` -- the staged BM=BN=BK=32 tile over 128
// invocations, for M at or above the row tile.
//
// `dense_gemv_t_bfloat16` -- 32 K-lanes by 8 columns, for M below it. At
// decode M is 1 and the tile arm would stage 32 activation rows to use one,
// doing BM times the multiplies the answer needs. This is the same seam
// `kernels-cuda` puts at `m == 1 && beta == 0.0` for its `gemv_bf16` kernel,
// widened to the tile height because a staged tile wastes proportionally at
// every M below BM and not only at one.
//
// The suffix on the tile entrypoint is LOAD-BEARING and not decoration:
// `driver_vulkan::geometry::Tile::from_entrypoint` parses `_bm_32_bn_32` back
// out of the name, on the reading that "the name is where the truth is".
// `quant/qmm_t.slang`'s entrypoint matrix is spelled the same way.
//
// ## THE COLUMN OVERHANG IS THE DANGEROUS ONE
//
// A tile is BM x BN whatever the matrix is, so the last tile in each direction
// runs off the end of the real output. The row overhang writes past the buffer
// and `robustBufferAccess` -- which `driver-vulkan` REQUIRES of every device it
// opens -- discards it. The column overhang does not: the output is row-major,
// so a lane at `col >= n` does not write past the end, it writes over
// `(row + 1, col - n)`, a LIVE element of the next row, computed from a weight
// fetch that was itself out of range.
//
// `quant/qmm_t.slang` records the GPU sweep that caught this in this tree --
// every row after the first beginning with a zero, at `n = 47`, invisible at
// every tile-aligned shape the earlier tests used. That file could only guard
// the column half, because no entrypoint of its push block carries `m`.
//
// THIS ONE CARRIES `m`, so both halves are guarded and the tile-aligned-output
// contract `qmm_t` puts on its caller does not exist here. Three words in a
// push block against a real requirement on every caller is not a trade this
// file needed to make twice.
//
// ## THE STAGED TILE IS ZERO-FILLED, WHICH IS WHAT KEEPS THE BARRIERS UNIFORM
//
// Out-of-range rows, columns and K past the tail are staged as 0.0 rather than
// skipped, so the inner product runs the full BK with no lane-dependent bound
// and every `GroupMemoryBarrierWithGroupSync` is reached by all 128 invocations
// or by none. There is no `return`, no `continue` and no divergent loop trip
// between the first barrier and the epilogue. A zero staged into either slab
// contributes a zero term, so the fill is exact and not an approximation.
//
// ## THE ACCUMULATORS ARE NAMED, NEVER AN ARRAY
//
// `acc0` and `acc1`, two `float4`s, and not `float4 acc[2]`. This is
// `quant/qmm_t.wgsl`'s measured finding carried across: a dynamically indexed
// local is thread-local MEMORY rather than registers, and the same sweep that
// found it measured 1.68 -> 1.40 TFLOP/s for the array spelling. The loop over
// them is unrolled by hand for that reason alone.

#include "common/bf16.slang"

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, w);
PIE_BUFFER_RW(2, out_);

// The three extents, in the order the claim body passes them: `m` then `n`
// then `k`, which is rows, columns, contraction. A body that swapped `n` and
// `k` would index a rectangle of the right size the wrong way round and report
// nothing.
struct Push {
    int m;
    int n;
    int k;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

float load_x(uint row, uint kk) {
    return PIE_LOAD(x[row * uint(pc.k) + kk]);
}

float load_w(uint col, uint kk) {
    return PIE_LOAD(w[col * uint(pc.k) + kk]);
}

// Both guards, for the reason the header gives at length.
void write_out(uint row, uint col, float acc) {
    if (row >= uint(pc.m) || col >= uint(pc.n)) {
        return;
    }
    out_[row * uint(pc.n) + col] = PIE_STORE(acc);
}

#if defined(PIE_GEMV)

// One workgroup per (8 columns, 1 row); 32 lanes fold the contraction.
#define PIE_KLANES 32u
#define PIE_NLANES 8u

groupshared float partial[PIE_KLANES * PIE_NLANES];

[numthreads(32, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 local : SV_GroupThreadID) {
    uint klane = local.x;
    uint clane = local.y;
    uint col = gid.y;
    uint row = gid.z;

    // THE GUARD IS ON THE ACCUMULATION, NOT ON THE INVOCATION. A lane whose
    // column is past `n` still walks into the fold below and still reaches
    // both barriers; it just contributes zero. Returning early here would put
    // a barrier behind a lane-dependent branch.
    float acc = 0.0;
    if (col < uint(pc.n) && row < uint(pc.m)) {
        for (uint kk = klane; kk < uint(pc.k); kk += PIE_KLANES) {
            acc += load_x(row, kk) * load_w(col, kk);
        }
    }

    uint slot = clane * PIE_KLANES + klane;
    partial[slot] = acc;
    GroupMemoryBarrierWithGroupSync();

    for (uint step = PIE_KLANES / 2u; step > 0u; step >>= 1u) {
        if (klane < step) {
            partial[slot] += partial[slot + step];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (klane == 0u) {
        write_out(row, col, partial[clane * PIE_KLANES]);
    }
}

#else

#define PIE_BM 32u
#define PIE_BN 32u
#define PIE_BK 32u
// The column QUADS a `float4` accumulator covers, and the lane count.
#define PIE_BN4 (PIE_BN / 4u)
#define PIE_LANES 128u
// 128 lanes over 8 column-quads is 16 row starts, so each lane owns rows
// `r0` and `r0 + 16` -- the two named accumulators.
#define PIE_RSTEP (PIE_LANES / PIE_BN4)

// 4096 bytes each, 8192 together, against the 16384 every Vulkan
// implementation guarantees.
groupshared float xs[PIE_BM * PIE_BK];
groupshared float4 ws[PIE_BK * PIE_BN4];

[numthreads(32, 2, 2)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint flat = local.x + 32u * local.y + 64u * local.z;
    uint c4 = flat % PIE_BN4;
    uint r0 = flat / PIE_BN4;

    uint tile_row = group.y * PIE_BM;
    uint tile_col = group.x * PIE_BN;

    float4 acc0 = float4(0.0, 0.0, 0.0, 0.0);
    float4 acc1 = float4(0.0, 0.0, 0.0, 0.0);

    for (uint kb = 0u; kb < uint(pc.k); kb += PIE_BK) {
        uint kn = min(PIE_BK, uint(pc.k) - kb);
        GroupMemoryBarrierWithGroupSync();

        for (uint e = flat; e < PIE_BM * PIE_BK; e += PIE_LANES) {
            uint r = e / PIE_BK;
            uint kk = e - r * PIE_BK;
            bool live = kk < kn && (tile_row + r) < uint(pc.m);
            xs[e] = live ? load_x(tile_row + r, kb + kk) : 0.0;
        }
        for (uint e = flat; e < PIE_BK * PIE_BN4; e += PIE_LANES) {
            uint kk = e / PIE_BN4;
            uint q = e - kk * PIE_BN4;
            float4 v = float4(0.0, 0.0, 0.0, 0.0);
            if (kk < kn) {
                uint col = tile_col + q * 4u;
                v.x = (col + 0u) < uint(pc.n) ? load_w(col + 0u, kb + kk) : 0.0;
                v.y = (col + 1u) < uint(pc.n) ? load_w(col + 1u, kb + kk) : 0.0;
                v.z = (col + 2u) < uint(pc.n) ? load_w(col + 2u, kb + kk) : 0.0;
                v.w = (col + 3u) < uint(pc.n) ? load_w(col + 3u, kb + kk) : 0.0;
            }
            ws[e] = v;
        }
        GroupMemoryBarrierWithGroupSync();

        for (uint kk = 0u; kk < PIE_BK; kk++) {
            float4 wv = ws[kk * PIE_BN4 + c4];
            acc0 += xs[r0 * PIE_BK + kk] * wv;
            acc1 += xs[(r0 + PIE_RSTEP) * PIE_BK + kk] * wv;
        }
    }

    uint col = tile_col + c4 * 4u;
    write_out(tile_row + r0, col + 0u, acc0.x);
    write_out(tile_row + r0, col + 1u, acc0.y);
    write_out(tile_row + r0, col + 2u, acc0.z);
    write_out(tile_row + r0, col + 3u, acc0.w);
    write_out(tile_row + r0 + PIE_RSTEP, col + 0u, acc1.x);
    write_out(tile_row + r0 + PIE_RSTEP, col + 1u, acc1.y);
    write_out(tile_row + r0 + PIE_RSTEP, col + 2u, acc1.z);
    write_out(tile_row + r0 + PIE_RSTEP, col + 3u, acc1.w);
}

#endif

// pie:instantiate dense_gemm_t_bfloat16_bm_32_bn_32
// pie:instantiate dense_gemv_t_bfloat16 PIE_GEMV=1
