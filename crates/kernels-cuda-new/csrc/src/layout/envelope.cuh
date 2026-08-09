//===-- envelope.cuh - the KV min/max envelope kernels ---------------===//
//
// Seven `__global__`s, three `__device__` helpers and the fused form's token
// cap. `envelope.cu` includes this file and keeps its five launchers, so
// exactly ONE definition of each kernel exists in the tree -- a split and not
// a copy. `norm/altup_aux` shipped a release with two definitions of six
// kernels; they agreed the day they were written and each stayed right for
// whichever half of the tests exercised it.
//
// The launchers are called from `attn/kv_paged.cu`, which is another
// family's file. Nothing here changes that: this migration moves device text
// and adds rows, it does not delete launchers, so every caller keeps
// compiling against the same `.hpp` it always did.
//
// # Why none of the seven is a row, kernel by kernel
//
// The head axis is no longer the reason. `LaunchRule::GatedRms` is
// `[rows, kv_heads]` at 256 threads and `PerHeadElementwise` is
// `[rows, q_heads]` at `clamp(head_dim, 32, 128)`, so the six two-dimensional
// launchers here — `dim3(pages_or_tokens, num_kv_heads)` with a block sized
// on `head_dim` — finally have a rule that spells their grid. Each is
// refused for a reason of its own, and each reason survives the rule
// landing:
//
//   • `merge_written_fused` and `merge_written` narrow with
//     `f32_to_bf16_rd` and `f32_to_bf16_ru` — DIRECTED rounding, toward
//     -inf for a minimum and +inf for a maximum, which is what keeps a
//     stored envelope a true bound. `Elem<T>::from_f32` is
//     round-to-nearest-even and the trait has no directed member, so
//     retyping these two through it would round a bound back INSIDE the
//     range it must contain — the pruning fault this file's own arithmetic
//     exists to prevent, and one that shows up as a slightly wrong answer
//     and never as a failure. `merge_written` also routes every update
//     through `atomic_min`/`atomic_max`, whose CAS loops are written on
//     `bf16_as_u16` and the bf16 sign-magnitude order. Templating either
//     means rewriting a body, which is a rewrite and not a retype.
//   • `reset_started_pages` is clean — no rounding, no atomics — and still
//     cannot be rowed, because a row is one symbol firing one launch and
//     its symbol fires two. `launch_envelope_merge_written_bf16` picks on
//     `num_tokens <= kEnvelopeFuseMaxTokens`: at or under the cap it fires
//     `merge_written_fused` alone, above it fires `reset_started_pages`
//     and then `merge_written`. A row for either half states half a
//     contract, which is the refusal `causal_conv1d_prefill_batched`
//     already carries.
//   • `recompute`, `update_appended` and `dot<BLOCK>` are reached the same
//     way, from launchers `attn/kv_paged.cu` calls while composing a wave.
//   • `seed_empty` is `<<<ceil(n/256), 256>>>` with its own bound check,
//     which is `LaunchRule::Elementwise` exactly. It still has no row,
//     because no model text states it — the driver seeds envelopes while
//     composing a wave, so there is no fire whose operands a `Source` could
//     name. A row written anyway would be a contract with nothing on the
//     other end.
//
// Five of the six stay untemplated, and that is deliberate rather than
// pending: a template is the price of admission for a row and buys nothing
// else, and a template with one instantiation and no row is a rename.
//
// `update_appended` is the exception and it was NOT templated for a row —
// it still has none, and the rectangle below still refuses it. It was
// templated for LINKAGE, which is a fact about the archive's C++ and not
// about the table. A non-template `__global__` defined in a header may be
// included by exactly ONE translation unit: the second includer is a hard
// link error on the function and on its host stub, measured this session as
// `multiple definition of __device_stub__Z...`, and it fires even when the
// second includer never launches it. `envelope.cu` is already that one
// includer, so `attn/kv_paged.cu` — the single consumer of
// `launch_envelope_update_appended_bf16`, at `:728` — could not include this
// header to fire the kernel directly and had to keep going through the host
// launcher. A template instantiates with INTERNAL linkage (`nm -C` says `t`,
// not even a weak `W`), so each includer gets a private copy and there is
// nothing to collide. Templating it is what lets that launcher's consumer
// set empty and the launcher become deletable under §10.10.
//
// # What `T` is, and what it deliberately is not
//
// `T` is the KV CACHE's element type — the thing `reduce_page` reads. The
// ENVELOPE's is not a parameter and must not become one. `reduce_page`
// narrows with `f32_to_bf16_rd` and `f32_to_bf16_ru`, directed rounding in
// the two directions that make a stored envelope a true bound;
// `Elem<T>::from_f32` is round-to-nearest-even and the trait has no directed
// member. Parameterising the write side would round each bound INSIDE the
// range it exists to contain — a wrong answer that never faults, on a
// structure whose whole job is to be conservative. So the read widens
// through `Elem<T>::to_f32`, which at `T = bf16` is `bf16_to_f32` verbatim
// (`pie_device.cuh:307`), and the write stays `bf16` and directed. The same
// rounding is why `merge_written` and `merge_written_fused` are not
// templated at all, and it is why this one is templated only halfway.
//
// Every launch site names `T` explicitly — `update_appended<device::bf16>`.
// `T` appears in the parameter list, so an un-edited call site would DEDUCE
// it rather than take a default, and this kernel's host interface spells the
// cache `device::u16*` and only becomes `bf16*` through a `reinterpret_cast`
// inside the launcher. A future call site that dropped the cast would deduce
// `T = u16` and instantiate against an `Elem<u16>` that does not exist.
// Naming the argument at the launch costs one token and removes the question.
//
// # `Tu`, and why templating `update_appended` alone was not enough
//
// The linkage rule above is about the HEADER, not about one kernel in it.
// Templating `update_appended` and stopping there was measured this session
// and does not work: two translation units that both include this file still
// fail to link, on all five of the `__global__`s that were still plain —
// `recompute`, `seed_empty`, `merge_written_fused`, `reset_started_pages`
// and `merge_written` — each reported twice, once for the function and once
// for its `__device_stub__`. The same link named NEITHER `update_appended`
// NOR `dot`, which is the positive control: in one measurement the two
// templates were silent and the five non-templates all collided. One plain
// `__global__` anywhere in a header is enough to cap it at one includer.
//
// So the five carry `template <int Tu = 0>`. `Tu` is not used, is not
// deducible from any parameter, and has a default — so every existing
// `<<<>>>` in `envelope.cu` fires un-edited and instantiates `Tu = 0`, and
// the emitted device code is what it was. This is deliberately NOT the `T`
// treatment: `T` parameterises an element type and would have to answer the
// directed-rounding question above, which for `merge_written` and
// `merge_written_fused` has no answer. `Tu` parameterises nothing. It buys
// the one property that is wanted — a template's internal linkage, a private
// copy per includer — and changes no arithmetic, which is why it is
// available to kernels whose element type is not negotiable.
//
// It buys no row either. A defaulted non-type parameter is not an element
// axis and none of the five became nameable; `elem` would have to spell `0`,
// and the rectangle refuses them for their grids regardless. This is a
// linkage change and only a linkage change.
//
// # NVRTC's constraints, which are why this file looks the way it does
//
// There is no include path: NVRTC resolves the two directives below out of a
// header set carried in the Rust binary, by name. So nothing here may reach
// for the C++ standard library -- the `stdlib_probe` measured 0 of 31
// standard headers answering -- and `<cstdint>` stays behind in `envelope.cu`
// where the host needs it. `device::bf16` and the fixed-width integer names
// are `pie_device.cuh`'s.
//
// The kernels also had to leave the anonymous namespace: an instantiation is
// named to `nvrtcAddNameExpression` as a STRING, and internal linkage has no
// such name.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

// `envelope_dot_thread_partial`, shared with the fused PTIR runtime so the
// scoring loop is byte-identical on both paths.
#include "layout/envelope_device.cuh"

namespace pie_cuda_driver::kernels::layout::device {

// The scalar layer is the PRELUDE's, not this family's. Named here so the
// kernels below read as they always did, so a row may keep spelling its
// element type `device::bf16`, and so the launchers in the enclosing
// namespace -- which write `device::` meaning the prelude's -- go on
// resolving to the same types through these declarations.
using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::bf16_as_u16;
using ::pie_cuda_driver::kernels::device::bf16_to_f32;
using ::pie_cuda_driver::kernels::device::f32_to_bf16;
using ::pie_cuda_driver::kernels::device::f32_to_bf16_rd;
using ::pie_cuda_driver::kernels::device::f32_to_bf16_ru;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::neg_inf;
using ::pie_cuda_driver::kernels::device::pos_inf;
using ::pie_cuda_driver::kernels::device::u16;
using ::pie_cuda_driver::kernels::device::u16_as_bf16;
using ::pie_cuda_driver::kernels::device::u32;
using ::pie_cuda_driver::kernels::device::u8;
using ::pie_cuda_driver::kernels::device::usize;

// The per-(page, kv_head) min/max reduction, shared by full recompute and the
// page-list update so both paths are literally the same numerics (the full
// recompute is what `test_envelope_dot` parity-checks).
// `T` is the CACHE's element type only; the envelope stays `bf16` because the
// narrowing below is directed and `Elem<T>::from_f32` is not. See the header.
template <class T>
__device__ inline void reduce_page(
    const T* __restrict__ k_pages,
    int page,
    int kh,
    int live,
    int page_size,
    int num_kv_heads,
    int head_dim,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max)
{
    const long token_stride = static_cast<long>(num_kv_heads) * head_dim;
    const long page_base = static_cast<long>(page) * page_size * token_stride +
                           static_cast<long>(kh) * head_dim;
    const long env_base =
        (static_cast<long>(page) * num_kv_heads + kh) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float mn = pos_inf();
        float mx = -pos_inf();
        for (int t = 0; t < live; ++t) {
            const float v = Elem<T>::to_f32(
                k_pages[page_base + static_cast<long>(t) * token_stride + d]);
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        env_min[env_base + d] = f32_to_bf16_rd(mn);
        env_max[env_base + d] = f32_to_bf16_ru(mx);
    }
}

// One block per (page, kv_head); threads stride over head_dim, each reducing its
// dims' min/max across the page's live tokens. Streaming reads of the NHD layout.
template <int Tu = 0>
__global__ void recompute(
    const bf16* __restrict__ k_pages,
    const i32* __restrict__ page_live_lens,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int page_size,
    int num_kv_heads,
    int head_dim)
{
    const int page = blockIdx.x;
    const int kh = blockIdx.y;
    reduce_page(k_pages, page, kh, page_live_lens[page], page_size,
                         num_kv_heads, head_dim, env_min, env_max);
}

// Refresh exactly the pages this fire appended to, deriving that set on-device
// from the same CSR arithmetic `write_kv_kernel` uses. A request's post-append
// length is `(pages-1)*page_size + last_page_len`, so its new tokens occupy
// `[total_after - qo_len, total_after)` and the touched pages are that span
// divided by `page_size`. Rescanning the whole page list instead would cost a
// full KV read per layer -- as much as attention itself.
//
// One block per (touched slot, kv_head), where the grid's x extent is the host's
// worst-case bound `ceil(total_tokens/page_size) + num_requests`; blocks past
// the true count exit. Pages are append-only, so recomputing a touched page in
// full gives the same answer an incremental merge would.
template <class T>
__global__ void update_appended(
    const T* __restrict__ k_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim)
{
    const int slot = blockIdx.x;
    const int kh = blockIdx.y;

    // Walk requests accumulating their touched-page counts until `slot` lands
    // inside one. R is the batch size, so a linear scan beats the divergence a
    // binary search would add.
    int seen = 0;
    for (int r = 0; r < num_requests; ++r) {
        const int pages_first = static_cast<int>(kv_page_indptr[r]);
        const int pages_last = static_cast<int>(kv_page_indptr[r + 1]);
        const int num_pages_r = pages_last - pages_first;
        if (num_pages_r <= 0) continue;

        const int qo_len =
            static_cast<int>(qo_indptr[r + 1]) - static_cast<int>(qo_indptr[r]);
        if (qo_len <= 0) continue;

        const int total_after =
            (num_pages_r - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
        const int pre_len = total_after - qo_len;
        if (total_after <= 0) continue;

        const int first_page = pre_len / page_size;
        const int last_page = (total_after - 1) / page_size;
        const int touched = last_page - first_page + 1;

        if (slot < seen + touched) {
            const int page_in_req = first_page + (slot - seen);
            if (page_in_req >= num_pages_r) return;
            const int live = (page_in_req == last_page)
                ? static_cast<int>(kv_last_page_lens[r])
                : page_size;
            if (live <= 0) return;
            reduce_page(
                k_pages,
                static_cast<int>(kv_page_indices[pages_first + page_in_req]),
                kh, live, page_size, num_kv_heads, head_dim, env_min, env_max);
            return;
        }
        seen += touched;
    }
}

// One block per (kv_head, page); threads reduce over the group·head_dim terms of
// `Σ max(q·min, q·max)`. Pages beyond `live_pages` are `-inf`.
template <int BLOCK>
__global__ void dot(
    const float* __restrict__ q,
    const bf16* __restrict__ env_min,
    const bf16* __restrict__ env_max,
    float* __restrict__ score,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int p_max,
    int live_pages)
{
    const int kh = blockIdx.y;
    const int p = blockIdx.x;
    float* out = &score[static_cast<long>(kh) * p_max + p];

    if (p >= live_pages) {
        if (threadIdx.x == 0) *out = -pos_inf();
        return;
    }

    const int group = num_q_heads / num_kv_heads;
    const long env_base =
        (static_cast<long>(p) * num_kv_heads + kh) * head_dim;

    const float local = envelope_dot_thread_partial(
        q, env_min, env_max, env_base, kh * group, group, head_dim,
        static_cast<int>(threadIdx.x), BLOCK);

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = buf[0];
}

// One thread per (page, kv_head, dim) triple of the empty envelope.
template <int Tu = 0>
__global__ void seed_empty(
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    usize n)
{
    const usize i =
        static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // `pos_inf()`/`neg_inf()` and not `INFINITY`: the macro is `<math.h>`'s,
    // nvcc pre-includes that header into a `.cu` and NVRTC includes nothing
    // at all -- the unit probe measured this file failing on the identifier
    // the first time it was compiled with no include path. The prelude states
    // both as bit patterns, so the value is exactly the one the C library
    // names rather than a decimal that has to round back to it.
    env_min[i] = f32_to_bf16(pos_inf());
    env_max[i] = f32_to_bf16(neg_inf());
}

// Single-launch form of the reset+merge pair below, for fires narrow enough
// that the launch overhead of the two-kernel version dominates -- which is
// every decode fire, and therefore the overwhelming majority of calls. Two
// launches cost ~4.6 us per layer on an L40S regardless of how few tokens they
// carry, so at 28 layers the pair is ~129 us of pure dispatch on the critical
// path of every step; this halves that.
//
// It removes the atomics too. Blocks elect ONE writer per (page, kv_head) --
// the first valid token naming that page -- so that block owns the page's
// envelope outright and can gather its own fire's keys into registers and
// store once. Sole ownership is also what makes the reset safe to fold in:
// the race the two-kernel split exists to avoid (a reset erasing a key merged
// by another token of the same fire) cannot arise when the same block does
// both, in order, for every token on the page.
//
// Thread 0 would serialise the O(num_tokens) scans, which shows up as soon as
// a fire carries more than a handful of tokens, so the scan is strided across
// the block and the gathered list is built with shared-memory atomics. Order
// within the list is irrelevant -- it feeds a min/max.
constexpr int kEnvelopeFuseMaxTokens = 128;

template <int Tu = 0>
__global__ void merge_written_fused(
    const bf16* __restrict__ k_curr,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    __shared__ int s_mine[kEnvelopeFuseMaxTokens];
    __shared__ int s_count;
    __shared__ int s_started;
    __shared__ int s_taken;

    const int token = blockIdx.x;
    const int kh = blockIdx.y;
    if (token >= num_tokens) return;

    if (threadIdx.x == 0) {
        s_count = 0;
        s_started = 0;
        s_taken = 0;
    }
    __syncthreads();

    // Uniform across the block: it depends only on blockIdx.
    if (row_valid != nullptr && row_valid[token] == 0) return;
    const u32 page = w_page[token];

    for (int t = threadIdx.x; t < num_tokens; t += blockDim.x) {
        if (row_valid != nullptr && row_valid[t] == 0) continue;
        if (w_page[t] != page) continue;
        if (t < token) {
            atomicOr(&s_taken, 1);  // an earlier token owns this page
            continue;
        }
        s_mine[atomicAdd(&s_count, 1)] = t;
        if (w_off[t] == 0u) atomicOr(&s_started, 1);
    }
    __syncthreads();
    if (s_taken != 0) return;

    const long env_base =
        (static_cast<long>(page) * num_kv_heads + kh) * head_dim;
    const int count = s_count;
    const bool started = s_started != 0;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float lo = pos_inf();
        float hi = -pos_inf();
        for (int i = 0; i < count; ++i) {
            const long src =
                (static_cast<long>(s_mine[i]) * num_kv_heads + kh) * head_dim;
            const float v = bf16_to_f32(k_curr[src + d]);
            lo = fminf(lo, v);
            hi = fmaxf(hi, v);
        }
        // Started in this fire => the page was recycled, so the previous
        // tenant's bound is dead and must be replaced, not widened.
        if (!started) {
            lo = fminf(lo, bf16_to_f32(env_min[env_base + d]));
            hi = fmaxf(hi, bf16_to_f32(env_max[env_base + d]));
        }
        env_min[env_base + d] = f32_to_bf16_rd(lo);
        env_max[env_base + d] = f32_to_bf16_ru(hi);
    }
}

// bf16 min/max via CAS on the bit pattern. bf16 IS the top 16 bits of an
// IEEE-754 float, so the ordering argument is the float one: the values in play
// are real keys plus the `+inf`/`-inf` seed, the ordinary comparison inside the
// loop is what decides, and the CAS only serialises the update.
//
// The stored value is rounded AWAY from the incoming key -- down for the
// minimum, up for the maximum -- so a widening can never round back inside the
// true range. The keys are themselves bf16, so in practice both roundings are
// exact and this costs nothing; it is here so that the envelope stays a valid
// bound if a caller ever merges a value that is not already a bf16.
__device__ inline void atomic_min(bf16* addr, float value) {
    unsigned short* as_u16 = reinterpret_cast<unsigned short*>(addr);
    const unsigned short want = bf16_as_u16(f32_to_bf16_rd(value));
    unsigned short old = *as_u16;
    unsigned short assumed;
    do {
        if (bf16_to_f32(u16_as_bf16(old)) <= value) return;
        assumed = old;
        old = atomicCAS(as_u16, assumed, want);
    } while (assumed != old);
}

__device__ inline void atomic_max(bf16* addr, float value) {
    unsigned short* as_u16 = reinterpret_cast<unsigned short*>(addr);
    const unsigned short want = bf16_as_u16(f32_to_bf16_ru(value));
    unsigned short old = *as_u16;
    unsigned short assumed;
    do {
        if (bf16_to_f32(u16_as_bf16(old)) >= value) return;
        assumed = old;
        old = atomicCAS(as_u16, assumed, want);
    } while (assumed != old);
}

// Companion to the merge below: a page whose FIRST cell is being written in
// this fire is being started from scratch, so whatever its envelope holds
// belongs to content that is now dead (the page was recycled through the
// pool). Reset it to empty before the merge widens it, or the bound would
// accumulate every request that ever used the page and converge on "keep
// everything". Pages entered at a non-zero offset are being continued and must
// keep what they have.
//
// A separate launch, not a branch inside the merge: two tokens of the same
// fire can land on one page, and a reset racing a merge would erase a key.
template <int Tu = 0>
__global__ void reset_started_pages(
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    const int token = blockIdx.x;
    const int kh = blockIdx.y;
    if (token >= num_tokens) return;
    if (row_valid != nullptr && row_valid[token] == 0) return;
    if (w_off[token] != 0u) return;

    const long env_base =
        (static_cast<long>(w_page[token]) * num_kv_heads + kh) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        env_min[env_base + d] = f32_to_bf16(pos_inf());
        env_max[env_base + d] = f32_to_bf16(-pos_inf());
    }
}

// Envelope maintenance for the EXPLICIT-descriptor KV write, where the program
// names the (physical page, offset) for every token instead of letting the
// kernel derive it from the CSR. There is no page list to walk and no live
// length to recompute from, so this MERGES each written key into the page's
// existing envelope rather than recomputing the page.
//
// Merging is the right operation here, not a shortcut:
//   * for the append-only case it is exactly equal to a full recompute,
//     because the seed is the empty envelope (+inf, -inf), the reset pass
//     above restores that seed whenever a page is started, and every key that
//     has ever been written to the page since then has passed through here;
//   * for the beam fork/freeze case -- the reason the explicit path exists at
//     all -- a cell can be REWRITTEN mid-page, and a recompute keyed on the
//     descriptor's offset would shrink the envelope to a prefix and could drop
//     a page that still holds a live key. Merging only ever widens, so the
//     bound stays an upper bound, which is the direction Quest must fail in.
//
// One block per (token, kv_head); threads stride over head_dim.
template <int Tu = 0>
__global__ void merge_written(
    const bf16* __restrict__ k_curr,
    const u32* __restrict__ w_page,
    const u8* __restrict__ row_valid,
    bf16* __restrict__ env_min,
    bf16* __restrict__ env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    const int token = blockIdx.x;
    const int kh = blockIdx.y;
    if (token >= num_tokens) return;
    if (row_valid != nullptr && row_valid[token] == 0) return;

    const long src_base =
        (static_cast<long>(token) * num_kv_heads + kh) * head_dim;
    const long env_base =
        (static_cast<long>(w_page[token]) * num_kv_heads + kh) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = bf16_to_f32(k_curr[src_base + d]);
        atomic_min(&env_min[env_base + d], v);
        atomic_max(&env_max[env_base + d], v);
    }
}

}  // namespace pie_cuda_driver::kernels::layout::device
