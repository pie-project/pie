#include "common/bf16.slang"

#if !defined(PIE_GEMV_V)
PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, w);
#endif
PIE_BUFFER_RW(2, out_);

struct Push {
    int m;
    int n;
    int k;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

#if !defined(PIE_GEMV_V)
float load_x(uint row, uint kk) {
    return PIE_LOAD(x[row * uint(pc.k) + kk]);
}

float load_w(uint col, uint kk) {
    return PIE_LOAD(w[col * uint(pc.k) + kk]);
}
#endif

void write_out(uint row, uint col, float acc) {
    if (row >= uint(pc.m) || col >= uint(pc.n)) {
        return;
    }
    out_[row * uint(pc.n) + col] = PIE_STORE(acc);
}

#if defined(PIE_GEMV)

#define PIE_KLANES 32u
#define PIE_NLANES 8u

groupshared float partial[PIE_KLANES * PIE_NLANES];

[numthreads(32, 8, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 local : SV_GroupThreadID) {
    uint klane = local.x;
    uint clane = local.y;
    uint col = gid.y;
    uint row = gid.z;
    float acc = 0.0;
    if (col < uint(pc.n) && row < uint(pc.m)) {
        uint k = uint(pc.k);
        uint kk = klane;
        for (; kk + 3u * PIE_KLANES < k; kk += 4u * PIE_KLANES) {
            acc += load_x(row, kk) * load_w(col, kk);
            acc += load_x(row, kk + PIE_KLANES) * load_w(col, kk + PIE_KLANES);
            acc += load_x(row, kk + 2u * PIE_KLANES) * load_w(col, kk + 2u * PIE_KLANES);
            acc += load_x(row, kk + 3u * PIE_KLANES) * load_w(col, kk + 3u * PIE_KLANES);
        }
        for (; kk < k; kk += PIE_KLANES) {
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

#elif defined(PIE_GEMV_V)

[[vk::binding(0)]] StructuredBuffer<uint4> x4;
[[vk::binding(1)]] StructuredBuffer<uint4> w4;

#ifndef PIE_KLANES
#define PIE_KLANES 32u
#endif
#ifndef PIE_NLANES
#define PIE_NLANES 8u
#endif
#define PIE_VROWS 4u

groupshared float partial[PIE_NLANES * PIE_VROWS * PIE_KLANES];

float dot8(uint4 a, uint4 b) {
    float acc = 0.0;
    acc += asfloat(a.x << 16u) * asfloat(b.x << 16u);
    acc += asfloat(a.x & 0xffff0000u) * asfloat(b.x & 0xffff0000u);
    acc += asfloat(a.y << 16u) * asfloat(b.y << 16u);
    acc += asfloat(a.y & 0xffff0000u) * asfloat(b.y & 0xffff0000u);
    acc += asfloat(a.z << 16u) * asfloat(b.z << 16u);
    acc += asfloat(a.z & 0xffff0000u) * asfloat(b.z & 0xffff0000u);
    acc += asfloat(a.w << 16u) * asfloat(b.w << 16u);
    acc += asfloat(a.w & 0xffff0000u) * asfloat(b.w & 0xffff0000u);
    return acc;
}

[numthreads(PIE_KLANES, PIE_NLANES, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint klane = local.x;
    uint clane = local.y;
    uint col = group.x * PIE_NLANES + clane;
    uint m = uint(pc.m);
    uint n = uint(pc.n);
    uint k8 = uint(pc.k) / 8u;
    float acc[PIE_VROWS];
    [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) acc[r] = 0.0;
    if (col < n) {
        uint wbase = col * k8;
        uint c = klane;

        for (; c + 3u * PIE_KLANES < k8; c += 4u * PIE_KLANES) {
            uint4 w0 = w4[wbase + c];
            uint4 w1 = w4[wbase + c + PIE_KLANES];
            uint4 w2 = w4[wbase + c + 2u * PIE_KLANES];
            uint4 w3 = w4[wbase + c + 3u * PIE_KLANES];
            [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
                if (r < m) {
                    uint xb = r * k8 + c;
                    acc[r] += dot8(x4[xb], w0) + dot8(x4[xb + PIE_KLANES], w1)
                        + dot8(x4[xb + 2u * PIE_KLANES], w2) + dot8(x4[xb + 3u * PIE_KLANES], w3);
                }
            }
        }
        for (; c < k8; c += PIE_KLANES) {
            uint4 wv = w4[wbase + c];
            [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
                if (r < m) acc[r] += dot8(x4[r * k8 + c], wv);
            }
        }
    }

    [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
        if (r < m) partial[(clane * PIE_VROWS + r) * PIE_KLANES + klane] = acc[r];
    }
    GroupMemoryBarrierWithGroupSync();
    [ForceUnroll] for (uint step = PIE_KLANES / 2u; step > 0u; step >>= 1u) {
        if (klane < step) {
            [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
                if (r < m) {
                    uint at = (clane * PIE_VROWS + r) * PIE_KLANES;
                    partial[at + klane] += partial[at + klane + step];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (klane == 0u && col < n) {
        [ForceUnroll] for (uint r = 0u; r < PIE_VROWS; r++) {
            if (r < m) out_[r * n + col] = PIE_STORE(partial[(clane * PIE_VROWS + r) * PIE_KLANES]);
        }
    }
}

#elif defined(PIE_COOPMAT)

using namespace linalg;
typealias MatA = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>;
typealias MatB = CoopMat<half, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>;
typealias MatAcc = CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator>;

#define CM_BM 64u
#define CM_BN 64u
#define CM_BK 32u
#define CM_THREADS 128u
#define CM_SUBGROUPS 4u

groupshared half sa[CM_BM * CM_BK];
groupshared half sb[CM_BN * CM_BK];
groupshared float sc[CM_SUBGROUPS * 256u];
groupshared float row_scale[CM_BM];
groupshared float row_max[CM_BM * 2u];

[numthreads(32, 4, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lane = local.x;
    uint sg = local.y;
    uint flat = lane + 32u * sg;
    uint tile_row = group.y * CM_BM;
    uint tile_col = group.x * CM_BN;
    uint m = uint(pc.m);
    uint n = uint(pc.n);
    uint k = uint(pc.k);

    {
        uint r = flat >> 1u;
        uint half_ = flat & 1u;
        float amax = 0.0;
        uint row = tile_row + r;
        if (row < m) {
            for (uint kk = half_; kk < k; kk += 2u) {
                amax = max(amax, abs(load_x(row, kk)));
            }
        }
        row_max[flat] = amax;
    }
    GroupMemoryBarrierWithGroupSync();
    if (flat < CM_BM) {
        float amax = max(row_max[flat * 2u], row_max[flat * 2u + 1u]);
        float scale = 1.0;
        if (amax > 32768.0 && !isinf(amax) && !isnan(amax)) {
            scale = exp2(-ceil(log2(amax / 32768.0)));
        }
        row_scale[flat] = scale;
    }
    GroupMemoryBarrierWithGroupSync();

    MatAcc acc[4];
    [ForceUnroll] for (uint ni = 0u; ni < 4u; ni++) acc[ni] = MatAcc(0.0);

    for (uint kb = 0u; kb < k; kb += CM_BK) {
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint i = 0u; i < (CM_BM * CM_BK) / CM_THREADS; i++) {
            uint e = flat + i * CM_THREADS;
            uint r = e / CM_BK;
            uint kk = e - r * CM_BK;
            uint row = tile_row + r;
            bool live = row < m && (kb + kk) < k;
            sa[e] = half(live ? load_x(row, kb + kk) * row_scale[r] : 0.0);
        }
        [ForceUnroll] for (uint i = 0u; i < (CM_BN * CM_BK) / CM_THREADS; i++) {
            uint e = flat + i * CM_THREADS;
            uint c = e / CM_BK;
            uint kk = e - c * CM_BK;
            uint col = tile_col + c;
            bool live = col < n && (kb + kk) < k;
            sb[e] = half(live ? load_w(col, kb + kk) : 0.0);
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint kk = 0u; kk < CM_BK; kk += 16u) {
            MatA a = MatA.Load<CoopMatMatrixLayout.RowMajor>(sa, sg * 16u * CM_BK + kk, CM_BK);
            [ForceUnroll] for (uint ni = 0u; ni < 4u; ni++) {
                MatB b = MatB.Load<CoopMatMatrixLayout.ColumnMajor>(sb, ni * 16u * CM_BK + kk, CM_BK);
                acc[ni] = coopMatMulAdd<float, false>(a, b, acc[ni]);
            }
        }
    }

    [ForceUnroll] for (uint ni = 0u; ni < 4u; ni++) {
        GroupMemoryBarrierWithGroupSync();
        acc[ni].Store<CoopMatMatrixLayout.RowMajor>(sc, sg * 256u, 16u);
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint i = 0u; i < 8u; i++) {
            uint e = lane + i * 32u;
            uint r = e >> 4u;
            uint c = e & 15u;
            uint row = tile_row + sg * 16u + r;
            uint col = tile_col + ni * 16u + c;
            write_out(row, col, sc[sg * 256u + e] / row_scale[sg * 16u + r]);
        }
    }
}

#else

#define PIE_BM 32u
#define PIE_BN 32u
#define PIE_BK 32u
#define PIE_BN4 (PIE_BN / 4u)
#define PIE_LANES 128u
#define PIE_RSTEP (PIE_LANES / PIE_BN4)

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

// pie:instantiate dense_gemv_t_bf16 PIE_GEMV=1
// pie:instantiate dense_gemv_t_v_bf16 PIE_GEMV_V=1
// pie:instantiate dense_gemv_t_v16_bf16 PIE_GEMV_V=1 PIE_KLANES=16u PIE_NLANES=16u
// pie:instantiate dense_gemv_t_v64_bf16 PIE_GEMV_V=1 PIE_KLANES=64u PIE_NLANES=4u
// pie:instantiate dense_gemv_t_v128_bf16 PIE_GEMV_V=1 PIE_KLANES=128u PIE_NLANES=2u
// pie:instantiate dense_gemv_t_v256_bf16 PIE_GEMV_V=1 PIE_KLANES=256u PIE_NLANES=1u
// pie:instantiate dense_gemm_t_bf16
// pie:instantiate dense_gemm_t_cm_bf16 @coopmat PIE_COOPMAT=1
