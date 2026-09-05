#include "common/bf16.slang"
#include "common/reduce.slang"

void index_rope_pair(RWStructuredBuffer<PIE_ACT> row, uint base, int i, int rope_dim, int pos, float theta) {
    float freq = pow(theta, -2.0 * float(i) / float(rope_dim));
    float ang = float(pos) * freq;
    float c = cos(ang);
    float s = sin(ang);
    float a = PIE_LOAD(row[base + uint(2 * i)]);
    float b = PIE_LOAD(row[base + uint(2 * i + 1)]);
    row[base + uint(2 * i)] = PIE_STORE(a * c - b * s);
    row[base + uint(2 * i + 1)] = PIE_STORE(b * c + a * s);
}

#if defined(PIE_INDEX_KNORM)

#define PIE_GROUP_X 256
PIE_BUFFER_RW(0, idx_k);
PIE_BUFFER_RO(1, w);
PIE_BUFFER_RO(2, b);
[[vk::binding(3)]] StructuredBuffer<int> positions;
struct Push { int head_dim; int rope_dim; float theta; float eps; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = group.x;
    uint tid = local.x;
    uint head_dim = uint(pc.head_dim);
    uint base = n * head_dim;

    float s = 0.0;
    for (uint d = tid; d < head_dim; d += PIE_GROUP_X) s += PIE_LOAD(idx_k[base + d]);
    float mean = pie_workgroup_sum(s) / float(head_dim);

    float vv = 0.0;
    for (uint d = tid; d < head_dim; d += PIE_GROUP_X) {
        float x = PIE_LOAD(idx_k[base + d]) - mean;
        vv += x * x;
    }
    float inv = rsqrt(pie_workgroup_sum(vv) / float(head_dim) + pc.eps);

    for (uint d = tid; d < head_dim; d += PIE_GROUP_X) {
        float x = (PIE_LOAD(idx_k[base + d]) - mean) * inv;
        idx_k[base + d] = PIE_STORE(x * PIE_LOAD(w[d]) + PIE_LOAD(b[d]));
    }

    AllMemoryBarrierWithGroupSync();

    int pos = positions[n];
    int pairs = pc.rope_dim / 2;
    for (int i = int(tid); i < pairs; i += PIE_GROUP_X) {
        index_rope_pair(idx_k, base, i, pc.rope_dim, pos, pc.theta);
    }
}

#elif defined(PIE_INDEX_Q_ROPE)

#define PIE_GROUP_X 32
PIE_BUFFER_RW(0, idx_q);
[[vk::binding(1)]] StructuredBuffer<int> positions;
struct Push { int n_heads; int head_dim; int rope_dim; float theta; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int h = int(gid.x);
    if (h >= pc.n_heads) return;
    uint n = gid.y;
    uint base = (n * uint(pc.n_heads) + uint(h)) * uint(pc.head_dim);
    int pos = positions[n];
    int pairs = pc.rope_dim / 2;
    for (int i = 0; i < pairs; i++) {
        index_rope_pair(idx_q, base, i, pc.rope_dim, pos, pc.theta);
    }
}

#else

#define PIE_GROUP_X 256
#define PIE_LANES_PER_KEY 64
#define PIE_KEYS_PER_PASS (PIE_GROUP_X / PIE_LANES_PER_KEY)
PIE_BUFFER_RO(0, idx_q);
PIE_BUFFER_RO(1, idx_w);
PIE_BUFFER_RO(2, key_pages);
[[vk::binding(3)]] StructuredBuffer<int> positions;
[[vk::binding(4)]] StructuredBuffer<int> req_of_token;
[[vk::binding(5)]] StructuredBuffer<uint> kv_page_indices;
[[vk::binding(6)]] StructuredBuffer<uint> kv_page_indptr;
[[vk::binding(7)]] RWStructuredBuffer<float> scores;
[[vk::binding(8)]] RWStructuredBuffer<int> selection;
struct Push { int H; int D; int page_size; int score_stride; int topk; int ratio; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float pie_key_acc[PIE_GROUP_X];

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint t = group.x;
    int tid = int(local.x);
    uint srow = t * uint(pc.topk);
    int topk = pc.topk;

    int r = req_of_token[t];
    uint pages_first = kv_page_indptr[uint(r)];
    int stride = pc.ratio > 0 ? pc.ratio : 1;
    int total = positions[t] + 1;
    int npools = total > 0 ? total / stride : 0;
    int tail = total > 0 ? total - npools * stride : 0;
    if (tail > topk) tail = topk;
    int pool_budget = (topk - tail) / stride;
    if (npools > pc.score_stride) npools = pc.score_stride;

    uint frow = t * uint(pc.score_stride);
    uint H = uint(pc.H);
    uint D = uint(pc.D);
    uint qi = t * H * D;
    uint wi = t * H;

    int slot = tid / PIE_LANES_PER_KEY;
    int hh = tid % PIE_LANES_PER_KEY;
    for (int j0 = 0; j0 < npools; j0 += PIE_KEYS_PER_PASS) {
        int j = j0 + slot;
        float acc = 0.0;
        if (j < npools) {
            int cell = (j + 1) * stride - 1;
            uint page = kv_page_indices[pages_first + uint(cell / pc.page_size)];
            uint kj = (page * uint(pc.page_size) + uint(cell % pc.page_size)) * D;
            for (uint h = uint(hh); h < H; h += PIE_LANES_PER_KEY) {
                uint qh = qi + h * D;
                float dot = 0.0;
                for (uint d = 0u; d < D; d++) dot += PIE_LOAD(idx_q[qh + d]) * PIE_LOAD(key_pages[kj + d]);
                acc += max(dot, 0.0) * PIE_LOAD(idx_w[wi + h]);
            }
        }
        pie_key_acc[tid] = acc;
        GroupMemoryBarrierWithGroupSync();
        if (tid < PIE_KEYS_PER_PASS && j0 + tid < npools) {
            float sum = 0.0;
            for (int g = 0; g < PIE_LANES_PER_KEY; g++) sum += pie_key_acc[tid * PIE_LANES_PER_KEY + g];
            scores[frow + uint(j0 + tid)] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    AllMemoryBarrierWithGroupSync();

    for (int i = tid; i < tail; i += PIE_GROUP_X) selection[srow + uint(i)] = npools * stride + i;

    if (npools <= pool_budget) {
        for (int n = tid; n < topk - tail; n += PIE_GROUP_X) {
            int j = n / stride;
            selection[srow + uint(tail + n)] = (j < npools) ? j * stride + (n % stride) : -1;
        }
        return;
    }

    float lo_l = 3.0e38;
    float hi_l = -3.0e38;
    for (int j = tid; j < npools; j += PIE_GROUP_X) {
        float s = scores[frow + uint(j)];
        lo_l = min(lo_l, s);
        hi_l = max(hi_l, s);
    }
    float lo = -pie_workgroup_max(-lo_l);
    float hi = pie_workgroup_max(hi_l);

    float thr = hi;
    for (int it = 0; it < 40; it++) {
        float mid = 0.5 * (lo + hi);
        float c = 0.0;
        for (int j = tid; j < npools; j += PIE_GROUP_X) {
            if (scores[frow + uint(j)] >= mid) c += 1.0;
        }
        int cnt = int(pie_workgroup_sum(c));
        if (cnt > pool_budget) lo = mid; else hi = mid;
        thr = hi;
    }

    if (tid == 0) {
        int n = tail;
        for (int j = 0; j < npools && n + stride <= topk; j++) {
            if (scores[frow + uint(j)] >= thr) {
                for (int i = 0; i < stride; i++) selection[srow + uint(n++)] = j * stride + i;
            }
        }
        for (; n < topk; n++) selection[srow + uint(n)] = -1;
    }
}
#endif

// pie:instantiate index_knorm_rope_bf16 PIE_INDEX_KNORM=1
// pie:instantiate index_q_rope_bf16 PIE_INDEX_Q_ROPE=1
// pie:instantiate index_topk_paged_bf16
