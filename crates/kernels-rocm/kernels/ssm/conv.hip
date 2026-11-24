// The causal depthwise conv in front of the mixers: decode, carried rows,
// chunked prefill, the state fold and commit.

#include "ssm/common.hip"

namespace pie::attn {

// ------------------------------------------------------------------ conv --

// Decode: each row r convolves its new input with its slot's window, then
// (when it folds) shifts the window by one. A thread owns one channel, so a
// wavefront's reads of x, of each window row and of the taps are adjacent,
// and every read of the step is issued before the first is used (KMAX taps
// at most; a wider conv takes the loop).
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_conv1d_update(const T* __restrict__ x, const T* __restrict__ weight,
                                  const T* __restrict__ bias, StateRows rows,
                                  T* __restrict__ y, int R, int C, int K, int dil, bool write_state,
                                  const u8* __restrict__ mask, const u32* __restrict__ win) {
    constexpr int KMAX = 8;
    const int r = blockIdx.y;
    if (r >= R || (win != nullptr && r >= static_cast<int>(win[0]))) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int cs = rows.atoms;
    const int ci = c % cs;
    T* state = rows.piece<T>(rid, c / cs);
    const int span = (K - 1) * dil;
    const bool fold = folds(write_state, mask, rl);
    const float xn = to_f32(x[row * C + c]);
    float acc = bias != nullptr ? to_f32(bias[c]) : 0.f;
    if (K <= KMAX && dil == 1) {
        float wk[KMAX], hist[KMAX];
#pragma unroll
        for (int k = 0; k < KMAX; ++k) {
            wk[k] = k < K ? to_f32(weight[c * K + k]) : 0.f;
            hist[k] = k + 1 <= span ? to_f32(state[(k + 1) * cs + ci]) : 0.f;
        }
#pragma unroll
        for (int k = 0; k < KMAX - 1; ++k)
            if (k < K - 1) acc += wk[k] * hist[k];
        acc += wk[K - 1 < KMAX ? K - 1 : 0] * xn;
        y[row * C + c] = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, xn));
        if (fold) {
            // The window shifted by one: row k takes row k + 1's input.
#pragma unroll
            for (int k = 0; k < KMAX; ++k)
                if (k < span) state[k * cs + ci] = from_f32<T>(hist[k]);
            state[span * cs + ci] = from_f32<T>(xn);
        }
        return;
    }
    for (int k = 0; k < K - 1; ++k) acc += to_f32(weight[c * K + k]) * to_f32(state[(k * dil + 1) * cs + ci]);
    acc += to_f32(weight[c * K + K - 1]) * xn;
    y[row * C + c] = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, xn));
    if (fold) {
        for (int k = 0; k < span; ++k) state[k * cs + ci] = state[(k + 1) * cs + ci];
        state[span * cs + ci] = from_f32<T>(xn);
    }
}

// A decode step over a lane's carried rows then its own (`Carry`). Row t of
// the lane's run is a carried row (t < n, its input read from the lane's rs
// buffer pages, plane 0) or its own (t = n); positions below 0 are the
// committed window's cells. The own row's output sums its window as
// `ssm_conv1d_update` does, from at most `K - 1` carried rows; where a
// recurrence re-steps the carried rows (`carry.scratch`), their outputs land
// there too. The window is stored as it stands after `commit` rows of the
// run, never past a row the lane has not committed. The own row's input
// lands in its buffer (plane 0), and the channels `[k_off, k_off + k_len)` of
// its output in plane 1 (a gated-delta layer's keys). A thread a channel.
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_conv1d_carried(const T* __restrict__ x, const T* __restrict__ weight,
                                   const T* __restrict__ bias, StateRows rows, T* __restrict__ y, int R,
                                   int C, int K, int dil, bool write_state, const u8* __restrict__ mask,
                                   Carry carry, int k_off, int k_len, const u32* __restrict__ win) {
    constexpr int KMAX = 8;
    const int r = blockIdx.y;
    if (r >= R || (win != nullptr && r >= static_cast<int>(win[0]))) return;
    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const CarryLane lane_ = carry.lane(rl);
    const T x_own = x[row * C + c];
    if (T* dst = carry.own_row<T>(lane_, 0, 0)) dst[c] = x_own;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const int cs = rows.atoms;
    const int ci = c % cs;
    T* state = rows.piece<T>(rid, c / cs);
    const int n = carry.carried(lane_);
    const int span = (K - 1) * dil + 1;
    auto at = [&](int t) -> float {
        if (t < 0) return to_f32(state[(span + t) * cs + ci]);
        if (t < n) return to_f32(carry.carried_row<T>(lane_, t, 0)[c]);
        return to_f32(x_own);
    };
    const float bias_v = bias != nullptr ? to_f32(bias[c]) : 0.f;
    float wk[KMAX];
#pragma unroll
    for (int k = 0; k < KMAX; ++k) wk[k] = k < K ? to_f32(weight[c * K + k]) : 0.f;
    float acc = bias_v;
#pragma unroll
    for (int k = 0; k < KMAX; ++k) {
        if (k >= K) break;
        acc += wk[k] * at(n - (K - 1 - k) * dil);
    }
    const T out = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, at(n)));
    y[row * C + c] = out;
    if (c >= k_off && c < k_off + k_len) {
        if (T* dst = carry.own_row<T>(lane_, 0, 1)) dst[c - k_off] = out;
    }
    const int commit = carry.commit(rl, n + 1);
    if (!folds(write_state, mask, rl) || commit <= 0) return;
    // Ascending, each cell reads one above it or a row: the old window is
    // read before it is overwritten.
    for (int s = 0; s < span; ++s) state[s * cs + ci] = from_f32<T>(at(commit - span + s));
}

// A beam group's fold of its conv window (its rows below the common ancestor): the window
// stored as it stands after the group's fold rows (the committed cells, then
// the rows' inputs, plane 0, through the representative lane's pages), so the
// committed point moves with the set's recurrent state. A thread a channel;
// grid ⌈C / block⌉; a block folds each group in turn.
template <class T>
__global__ void ssm_conv1d_fold(StateRows rows, int C, int K, int dil, Carry carry,
                                const GroupFold* __restrict__ folds, int groups) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    // The groups that fold lead the table; the first that folds none ends it.
    for (int g = 0; g < groups; ++g) {
        const GroupFold f = folds[g];
        const int n = static_cast<int>(f.rows);
        if (n <= 0) break;
        const int rl = static_cast<int>(f.lane);
        const i32* rid = rows.row(rl);
        if (rid[0] < 0) continue;
        const CarryLane cl = carry.lane(rl);
        const int cs = rows.atoms;
        const int ci = c % cs;
        T* state = rows.piece<T>(rid, c / cs);
        const int span = (K - 1) * dil + 1;
        auto at = [&](int t) -> T {
            if (t < 0) return state[(span + t) * cs + ci];
            u64 slot;
            u32 pos;
            carry.fold_at(cl, f.fold_off, t, slot, pos);
            return carry.plane_at<const T>(slot, pos, 0)[c];
        };
        // In place, ascending: cell k takes run position n - span + k, which
        // is a committed cell only as the old cell n + k > k (n >= 1), not
        // yet written; so any span folds without a copy of the window.
        for (int k = 0; k < span; ++k) state[k * cs + ci] = at(n - span + k);
    }
}

constexpr int kConvRows = 32;

// Prefill: grid (⌈C / block⌉, ⌈rows / 32⌉); a block convolves 32 rows of its
// channels, whichever requests they belong to. Inputs before a request's
// first row come from its slot's window. The window is not written here;
// `ssm_conv1d_state_commit` folds it after.
template <class T, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_conv1d_chunked(const T* __restrict__ x, const T* __restrict__ weight,
                                   const T* __restrict__ bias, T* __restrict__ y, StateRows rows,
                                   const u32* __restrict__ indptr, int C,
                                   int K, int dil, bool write_state, const u8* __restrict__ mask,
                                   const int* __restrict__ commit_len, const int* __restrict__ begin_at,
                                   const u32* __restrict__ win, int lanes_stated) {
    (void)write_state;
    (void)mask;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(indptr[lanes]);
    const int tile0 = blockIdx.y * kConvRows;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + kConvRows, total);
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const int window = (K - 1) * dil + 1;
    const float b = bias != nullptr ? to_f32(bias[c]) : 0.f;

    // The first request whose rows reach into this tile.
    int lo = 0, hi = lanes - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (static_cast<int>(indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
    }
    for (int r = lo; r < lanes; ++r) {
        if (static_cast<int>(indptr[r]) >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const Span s = span_of(indptr, r, rl, commit_len, begin_at);
        if (s.n <= 0) continue;
        const i32* rid = rows.row(rl);
        if (rid[0] < 0) continue;
        const T* xr = x + (s.t0 + row0) * C;
        T* yr = y + (s.t0 + row0) * C;
        const int cs = rows.atoms;
        const int ci = c % cs;
        const T* state = rows.piece<const T>(rid, c / cs);
        for (int t = max(tile0, s.t0) - s.t0; t < min(tile_end, s.t0 + s.n) - s.t0; ++t) {
            float acc = b;
            for (int k = 0; k < K; ++k) {
                const int src = t - (K - 1 - k) * dil;
                const float xv = src < 0 ? to_f32(state[(window + src) * cs + ci])
                                         : to_f32(xr[static_cast<long long>(src) * C + c]);
                acc += to_f32(weight[c * K + k]) * xv;
            }
            yr[static_cast<long long>(t) * C + c] = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, to_f32(xr[static_cast<long long>(t) * C + c])));
        }
    }
}

// The prefill conv at dilation 1 and K taps, eight channels a thread: each
// input row is read once as one 16-byte load and slides through a window of
// K rows in registers, where the per-channel walk above reads every input K
// times, two bytes a lane. The same products in the same order, so the same
// bits. Grid (⌈C / (8·block)⌉, ⌈rows / 32⌉); C a multiple of 8.
template <class T, int K, bool SILU = true, bool RESIDUAL = false>
__global__ void ssm_conv1d_chunked_v8(const T* __restrict__ x, const T* __restrict__ weight,
                                      const T* __restrict__ bias, T* __restrict__ y, StateRows rows,
                                      const u32* __restrict__ indptr, int C,
                                      const int* __restrict__ commit_len, const int* __restrict__ begin_at,
                                      const u32* __restrict__ win, int lanes_stated) {
    constexpr int V = 8;
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * V;
    if (c0 >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    if (lanes <= 0) return;
    const int total = static_cast<int>(indptr[lanes]);
    const int tile0 = blockIdx.y * kConvRows;
    if (tile0 >= total) return;
    const int tile_end = min(tile0 + kConvRows, total);
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    float w[V][K], b[V];
#pragma unroll
    for (int v = 0; v < V; ++v) {
        b[v] = bias != nullptr ? to_f32(bias[c0 + v]) : 0.f;
#pragma unroll
        for (int k = 0; k < K; ++k) w[v][k] = to_f32(weight[(c0 + v) * K + k]);
    }
    auto load = [](const T* at, float (&out)[V]) {
        const Pack8<T> p = *reinterpret_cast<const Pack8<T>*>(at);
#pragma unroll
        for (int v = 0; v < V; ++v) out[v] = to_f32(p.v[v]);
    };

    int lo = 0, hi = lanes - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (static_cast<int>(indptr[mid]) <= tile0) lo = mid; else hi = mid - 1;
    }
    for (int r = lo; r < lanes; ++r) {
        if (static_cast<int>(indptr[r]) >= tile_end) break;
        const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
        const Span s = span_of(indptr, r, rl, commit_len, begin_at);
        if (s.n <= 0) continue;
        const i32* rid = rows.row(rl);
        if (rid[0] < 0) continue;
        const T* xr = x + (s.t0 + row0) * C + c0;
        T* yr = y + (s.t0 + row0) * C + c0;
        // Eight channels never straddle a piece: a piece holds whole eights
        // of them (the launcher checks).
        const int cs = rows.atoms;
        const T* state = rows.piece<const T>(rid, c0 / cs) + c0 % cs;
        const int t_begin = max(tile0, s.t0) - s.t0, t_end = min(tile_end, s.t0 + s.n) - s.t0;
        if (t_begin >= t_end) continue;
        // win_[j] holds input t - (K - 1) + j.
        float xw[K][V];
#pragma unroll
        for (int j = 0; j < K - 1; ++j) {
            const int src = t_begin - (K - 1) + j;
            load(src < 0 ? state + static_cast<long long>(K + src) * cs : xr + static_cast<long long>(src) * C, xw[j]);
        }
        for (int t = t_begin; t < t_end; ++t) {
            load(xr + static_cast<long long>(t) * C, xw[K - 1]);
            Pack8<T> out;
#pragma unroll
            for (int v = 0; v < V; ++v) {
                float acc = b[v];
#pragma unroll
                for (int k = 0; k < K; ++k) acc += w[v][k] * xw[k][v];
                out.v[v] = from_f32<T>(conv_out<SILU, RESIDUAL>(acc, xw[K - 1][v]));
            }
            *reinterpret_cast<Pack8<T>*>(yr + static_cast<long long>(t) * C) = out;
#pragma unroll
            for (int j = 0; j < K - 1; ++j)
#pragma unroll
                for (int v = 0; v < V; ++v) xw[j][v] = xw[j + 1][v];
        }
    }
}

// The window after a request's rows: its last (K−1)·dil + 1 inputs, the
// older ones from the window it started with. Grid (⌈C / block⌉, lanes).
__global__ void ssm_conv1d_state_commit(const bf16* __restrict__ x, StateRows rows,
                                        const u32* __restrict__ indptr, int C,
                                        int K, int dil, const u8* __restrict__ mask,
                                        const int* __restrict__ commit_len,
                                        const int* __restrict__ begin_at, const int* __restrict__ snap,
                                        const int* __restrict__ snap_rows, const i32* __restrict__ snap_slots,
                                        const u32* __restrict__ win, int lanes_stated) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int lanes = win != nullptr ? static_cast<int>(win[2]) : lanes_stated;
    const int r = blockIdx.y;
    if (r >= lanes) return;
    const int rl = win != nullptr ? r + static_cast<int>(win[3]) : r;
    if (mask != nullptr && mask[rl] == 0) return;
    const i32* rid = rows.row(rl);
    if (rid[0] < 0) return;
    const Span s = span_of(indptr, r, rl, commit_len, begin_at);
    if (s.n <= 0) return;
    const long long row0 = win != nullptr ? static_cast<long long>(win[1]) : 0;
    const int window = (K - 1) * dil + 1;
    const bf16* xr = x + (s.t0 + row0) * C;
    const int cs = rows.atoms;
    const int ci = c % cs;
    bf16* state = rows.piece<bf16>(rid, c / cs);
    // Each snapshot's window, from the rows and the window the lane started
    // with, before the lane's own overwrites it.
    const Snaps sn = snaps_of(snap, rl);
    for (int e = sn.e0; e < sn.e1 && snap_rows[e] < s.n; ++e) {
        bf16* at = rows.piece<bf16>(snap_slots + static_cast<long long>(e) * rows.pieces, c / cs);
        for (int i = 0; i < window; ++i) {
            const int src = snap_rows[e] - window + i;
            at[i * cs + ci] = src < 0 ? state[(window + src) * cs + ci] : xr[static_cast<long long>(src) * C + c];
        }
    }
    // Ascending s reads only rows at or after the one it overwrites.
    for (int i = 0; i < window; ++i) {
        const int src = s.n - window + i;
        state[i * cs + ci] = src < 0 ? state[(window + src) * cs + ci] : xr[static_cast<long long>(src) * C + c];
    }
}

}  // namespace pie::attn
