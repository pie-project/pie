// The scalar layer and the fixed-width integer names, out of the prelude:
// NVRTC has no CUDA device headers, and this file is meant to compile
// under both it and nvcc.
#include "pie_device.cuh"
#include "attn/attention_xqa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>


#include "cuda_check.hpp"

// THIS FILE HELD THE LAST `__global__` IN THIS ARCHIVE, AND IT IS GONE.
//
// `build_xqa_metadata_kernel` -- the per-request dense page table and
// sequence-length build -- is now
// `kernels-cuda-new/csrc/src/attn/attention_xqa.cuh`, compiled by NVRTC as the
// `attn/attention_xqa` unit's one row (`families::attn`'s `ATTN_XQA_SIGS`),
// and its launcher `prepare_attention_xqa_decode_bf16` is
// `driver-cuda/src/fire/xqa.rs`. Unlike `attention_flashinfer.cuh`'s partial
// split, nothing was left behind here: this file kept no copy of the kernel
// and holds no `<<<>>>` at all.
//
// It could be a TOTAL split because the launcher's whole consumer set was one
// obligation rather than one call. `attn::prepare_attention_xqa_decode_bf16`
// is in no `kernels::table`, so `abi::emit_c_shim` emitted no entry for it and
// the driver could not reach it; no `.cu`, `.cuh`, `.cpp` or `.hpp` in the
// tree called it; and no hand-written `ffi::pie_k_*` in `driver-cuda/src`
// named it. What held it was `new-horizon.md` §44.5's argument -- the live row
// `attn::attention_xqa_decode_bf16_prepared` states `needs = Prepare::FireWide`
// and this was the only text in the tree that wrote the page table and the
// sequence lengths at the offsets `attention_xqa_decode_bf16_prepared` below
// reads back. The Rust writes them now, at the same offsets, and the argument
// transfers with the implementation.
//
// WHAT STAYS, AND WHY IT CANNOT LEAVE ON ITS OWN. Everything below ends in
// `launchMHAFlashInfer_xqa_gqa5_bf16_p32_h128`, an upstream FlashInfer HOST
// function reached by `#include <xqa/mha.cu>` into this translation unit under
// a renamed symbol. There is no device text of ours in it -- §50.1's
// measurement, on this file -- so the §48 split is degenerate: this becomes
// Rust in its entirety or it does not move at all. `new-horizon.md` §50.9 is
// the gap that decides when.
//
// THE CARVE IS DUPLICATED FIVE WAYS AND NOTHING CHECKS IT. The page table and
// sequence lengths are read back out of `workspace.float_buffer` by
// `attention_xqa_decode_bf16_prepared` below AND by each of the four
// `detail::launch_attention_xqa_decode_bf16_gqa*_prepared` bodies, each
// recomputing the same offsets. `driver-cuda/src/fire/xqa.rs`' `carve` is the
// statement of that layout in one place, and a port of any of the five should
// take it rather than writing a sixth copy.

// Build the GQA=5 FlashInfer XQA specialization into the native driver. Other
// ratios live in separate translation units because the XQA csrc exposes
// non-templated launch entry points.
#define NDEBUG 1
#define BEAM_WIDTH 1
#define USE_INPUT_KV 0
#define USE_CUSTOM_BARRIER 1
#define INPUT_FP16 0
#define DTYPE device::bf16
#define CACHE_ELEM_ENUM 0
#define TOKENS_PER_PAGE 32
#define HEAD_ELEMS 128
#define HEAD_GRP_SIZE 5
#define SLIDING_WINDOW 0
#define LOW_PREC_OUTPUT 0
#define SPEC_DEC 0
#define MLA_WRAPPER 0
#define USE_SM90_MHA 0
#define launchMHA launchMHA_xqa_gqa5_bf16_p32_h128
#define launchMHAFlashInfer launchMHAFlashInfer_xqa_gqa5_bf16_p32_h128

#include <xqa/mha.cu>

#undef launchMHA
#undef launchMHAFlashInfer

namespace pie_cuda_driver::kernels::attn {

namespace {

constexpr int kXqaPageSize = TOKENS_PER_PAGE;
constexpr int kXqaHeadDim = HEAD_ELEMS;
constexpr int kXqaHeadGroupRatio = HEAD_GRP_SIZE;
constexpr std::size_t kSemaphoreAlignment = 256;

std::uintptr_t align_up_ptr(std::uintptr_t p, std::size_t a) {
    return (p + a - 1) / a * a;
}

int current_device_major() {
    thread_local int cached_device = -1;
    thread_local int cached_major = 0;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev != cached_device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        cached_device = dev;
        cached_major = prop.major;
    }
    return cached_major;
}

int current_device_sm_count() {
    thread_local int cached_device = -1;
    thread_local int cached_sms = 0;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev != cached_device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        cached_device = dev;
        cached_sms = prop.multiProcessorCount;
    }
    return cached_sms;
}

bool xqa_ratio_supported(int ratio) {
    return ratio == 2 || ratio == 4 || ratio == 5 || ratio == 8;
}

bool xqa_gqa2_page16_enabled() { return false; }

}  // namespace

namespace detail {

// EIGHT DECLARATIONS STOOD HERE AND NAMED NOTHING. Deleted, and worth a line
// because they made this file read as if the XQA wrapper had sixteen entry
// points when it has eight.
//
//   * `launch_attention_xqa_decode_bf16_gqa2`, `_gqa4` and `_gqa8` -- the
//     UNPREPARED forms, taking the page CSR directly. No sibling defines any
//     of them: `attention_xqa_gqa{2,4,8}.cu` define only the `_prepared`
//     forms, and `new-horizon.md` §44.4 deleted the unprepared launchers a
//     pass ago without removing the declarations that named them. A
//     declaration with no definition and no call site links fine and says
//     something false, which is the worst combination a declaration has.
//   * `xqa_decode_bf16_gqa2_warmup_current_device`, `_gqa2_p16_`, `_gqa4_`,
//     `_gqa8_` and `_gqa8_sm90_` -- the per-device max-dynamic-smem setters
//     the header used to describe. §44.4 deleted the two warmups that had
//     bodies (`xqa_decode_bf16_warmup_current_device` and its gqa5 half);
//     these five never had one in this tree at all. The `.hpp`'s paragraph
//     about calling them after `cudaSetDevice` on each rank under TP>1
//     describes a call nothing makes and a function nothing defines.
//
// `attention_xqa_gqa8.cu:82` still carries the same dead shape --
// `launch_attention_xqa_decode_bf16_gqa8_sm90`, unprepared, defined by
// nothing, including by `attention_xqa_gqa8_sm90.cu`. Left alone here because
// that file is not this pass's.

void launch_attention_xqa_decode_bf16_gqa2_prepared(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float sm_scale);

void launch_attention_xqa_decode_bf16_gqa2_p16_prepared(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float sm_scale);

void launch_attention_xqa_decode_bf16_gqa4_prepared(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float sm_scale);

void launch_attention_xqa_decode_bf16_gqa8_prepared(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float sm_scale);

}  // namespace detail

bool xqa_decode_bf16_supported(int num_q_heads,
                               int num_kv_heads,
                               int head_dim,
                               int page_size,
                               int window_left,
                               float logits_soft_cap,
                               float sm_scale)
{
    if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) return false;
    const int ratio = num_q_heads / num_kv_heads;
    if (!xqa_ratio_supported(ratio)) return false;
    const bool page_supported =
        page_size == kXqaPageSize ||
        (ratio == 2 && page_size == 16 && xqa_gqa2_page16_enabled());
    if (head_dim != kXqaHeadDim || !page_supported) return false;
    if (window_left >= 0 || logits_soft_cap > 0.f) return false;
    const float default_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (sm_scale > 0.f && std::abs(sm_scale - default_scale) > 1.0e-6f) {
        return false;
    }
    // FlashInfer's public XQA wrapper only enables this path on SM90+.
    // The Ampere/Ada csrc instantiations compile, but local SM89 TP2
    // serving runs can spin indefinitely after graph capture, so keep
    // those devices on the regular FlashInfer decode path.
    return current_device_major() >= 9;
}
int xqa_decode_page_bucket(int max_pages_per_seq) {
    int bucket = 1;
    const int pages = std::max(1, max_pages_per_seq);
    while (bucket < pages && bucket < 4096) bucket <<= 1;
    return bucket;
}

// `prepare_attention_xqa_decode_bf16` STOOD HERE, and with it this archive's
// last `<<<>>>`. It is `driver-cuda/src/fire/xqa.rs::prepare_decode` now; the
// header comment argues why the move could take the whole of it.
//
// Two things it did are NOT in the Rust and are not lost:
//
//   * `xqa_decode_page_bucket` above stays, because it is C++'s answer to a
//     question the row cannot ask -- the launch passed `page_bucket` where the
//     kernel's parameter is named `max_pages_per_seq`, and every reader of the
//     buffer below has to round the same way. `fire/xqa.rs::page_bucket`
//     transcribes it. If one changes, both change, and nothing in the tree
//     will say so.
//   * `CUDA_CHECK(cudaPeekAtLastError())` after the launch. `KernelModule::fire`
//     checks `cuLaunchKernel`'s return, which catches a bad configuration but
//     not a sticky error left by an earlier launch, so the Rust is very
//     slightly less eager to blame itself for someone else's fault. Recorded
//     rather than reproduced: a peek-and-throw in the driver's fire path would
//     attribute an unrelated async fault to the next kernel that runs.

void attention_xqa_decode_bf16_prepared(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float sm_scale)
{
    if (!xqa_decode_bf16_supported(
            num_q_heads, num_kv_heads, head_dim, page_size,
            /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale)) {
        throw std::runtime_error("xqa decode: unsupported shape");
    }
    if (num_requests <= 0) return;

    const int head_group_ratio = num_q_heads / num_kv_heads;
    if (head_group_ratio == 2) {
        if (page_size == 16) {
            detail::launch_attention_xqa_decode_bf16_gqa2_p16_prepared(
                q,
                k_pages,
                v_pages,
                o,
                num_requests,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                max_pages_per_seq,
                workspace,
                stream,
                sm_scale);
            return;
        }
        detail::launch_attention_xqa_decode_bf16_gqa2_prepared(
            q,
            k_pages,
            v_pages,
            o,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_pages_per_seq,
            workspace,
            stream,
            sm_scale);
        return;
    }
    if (head_group_ratio == 4) {
        detail::launch_attention_xqa_decode_bf16_gqa4_prepared(
            q,
            k_pages,
            v_pages,
            o,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_pages_per_seq,
            workspace,
            stream,
            sm_scale);
        return;
    }
    if (head_group_ratio == 8) {
        detail::launch_attention_xqa_decode_bf16_gqa8_prepared(
            q,
            k_pages,
            v_pages,
            o,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_pages_per_seq,
            workspace,
            stream,
            sm_scale);
        return;
    }
    if (head_group_ratio != kXqaHeadGroupRatio) {
        throw std::runtime_error("xqa decode: unsupported GQA ratio");
    }

    const int page_bucket = xqa_decode_page_bucket(max_pages_per_seq);
    const std::size_t page_table_bytes =
        static_cast<std::size_t>(num_requests) * page_bucket *
        sizeof(device::i32);
    const std::size_t seq_lens_bytes =
        static_cast<std::size_t>(num_requests) * sizeof(device::u32);
    std::uintptr_t base =
        reinterpret_cast<std::uintptr_t>(workspace.float_buffer);
    std::uintptr_t p_page_table = align_up_ptr(base, alignof(device::i32));
    std::uintptr_t p_seq_lens =
        align_up_ptr(p_page_table + page_table_bytes, alignof(device::u32));
    std::uintptr_t p_scratch =
        align_up_ptr(p_seq_lens + seq_lens_bytes, kSemaphoreAlignment);
    const std::uintptr_t end =
        reinterpret_cast<std::uintptr_t>(workspace.float_buffer) +
        workspace.float_bytes;
    if (p_scratch >= end) {
        throw std::runtime_error("xqa decode: attention workspace too small");
    }

    auto* page_table = reinterpret_cast<device::i32*>(p_page_table);
    auto* seq_lens = reinterpret_cast<device::u32*>(p_seq_lens);
    void* scratch = reinterpret_cast<void*>(p_scratch);

    const int semaphore_count = num_requests * num_kv_heads;
    if (static_cast<std::size_t>(semaphore_count) * sizeof(device::u32) >
        workspace.int_bytes) {
        throw std::runtime_error("xqa decode: semaphore workspace too small");
    }
    auto* semaphores =
        reinterpret_cast<device::u32*>(workspace.int_buffer);
    CUDA_CHECK(cudaMemsetAsync(
        semaphores, 0,
        static_cast<std::size_t>(semaphore_count) * sizeof(device::u32),
        stream));

    const float q_scale = 1.0f;
    const float kv_scale = 1.0f;
    const device::u64 kv_stride_head =
        static_cast<device::u64>(head_dim);
    const device::u64 kv_stride_token =
        static_cast<device::u64>(num_kv_heads) * head_dim;
    const device::u64 kv_stride_page =
        static_cast<device::u64>(page_size) * num_kv_heads * head_dim;

    launchMHAFlashInfer_xqa_gqa5_bf16_p32_h128(
        static_cast<device::u32>(current_device_sm_count()),
        static_cast<device::u32>(num_kv_heads),
        /*slidingWinSize=*/0,
        q_scale,
        /*qScalePtr=*/nullptr,
        reinterpret_cast<OutputHead*>(o),
        reinterpret_cast<InputHead const*>(q),
        /*attentionSinks=*/nullptr,
        reinterpret_cast<GMemCacheHead*>(k_pages),
        reinterpret_cast<GMemCacheHead*>(v_pages),
        reinterpret_cast<KVCachePageIndex const*>(page_table),
        static_cast<device::u32>(page_bucket * page_size),
        seq_lens,
        static_cast<device::u32>(num_requests),
        kv_scale,
        /*kvScalePtr=*/nullptr,
        semaphores,
        scratch,
        current_device_major() >= 9,
        kv_stride_page,
        kv_stride_token,
        kv_stride_head,
        stream);
}

}  // namespace pie_cuda_driver::kernels::attn
