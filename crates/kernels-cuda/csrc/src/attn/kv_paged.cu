// The host half of the paged KV cache, and nothing else.
//
// The fourteen `__global__`s this file used to hold -- six write forms, four
// dequantisers, a cell move, a device-window write and the two page-view
// builders -- and the four `__device__` helpers they call live in
// `attn/kv_paged.cuh`, which is what the include below reads. ONE text, read
// by nvcc here and by NVRTC from the same bytes at run time.
//
// It was two. This file kept its own copy of all fourteen while the header
// carried fourteen more with the same bodies, and every gate was green
// because a split RENAMES -- `write_kv_kernel` became `write_kv`, in another
// namespace -- so the name-comparing gate could not see them
// (`new-horizon.md` §21.7). `a_split_file_uses_the_header_it_was_split_into`
// asks the question that needs no names and is what closed it.
//
// # The launches are unchanged, and that is the claim
//
// Twenty-two `<<<>>>` at the split; EIGHTEEN today, and every one of the
// four that went is recorded at its own tombstone below with the evidence
// that took it. Each surviving launch keeps the grid, block, shared-memory
// size and stream it had. Nine of the header's fourteen are
// templates; the argument every launch below passes is the one that
// reproduces the archive's kernel, measured rather than inferred:
//
// * `HND_LAYOUT` and `UseFp8` were already the archive's own template
//   parameters and each call site already chose an arm -- those carried over
//   spelling for spelling.
// * The three `template <class T>` dequantisers were PLAIN here, writing
//   `__nv_bfloat16`, so `T` is `device::bf16` and nothing else. It is stated
//   explicitly at each launch rather than defaulted: a wrong arm compiles,
//   runs, and is numerically plausible (§18.4 measured one at 99.83% of the
//   right answer), so the argument is written where a reader can check it
//   against the cast beside it.
//
// All fourteen were compared on an L40S against the archive's bodies taken
// verbatim from git, same input, `memcmp` on the destination: zero bytes
// differ on twenty instantiations, and every `bool` and `class T` parameter
// has a negative control on the other arm that DOES differ.
//
// `fp8_kind` stays a runtime `__nv_fp8_interpretation_t` argument on the two
// kernels that take one, for the reason the header gives: as a template
// parameter with a default, an `__NV_E5M2` page would decode as `__NV_E4M3`
// and be wrong plausibly. Both interpretations are in the parity set.
//
// # This file is the header's only includer in the archive, and must stay so
//
// Five of the fourteen are not templates -- `write_kv_fp8_per_tensor`,
// `write_kv_fp4_block`, `dequant_fp8_pages_active`, `build_window_page_view`
// and `build_full_split_view`. A `.cuh` holding a non-template `__global__`
// can be included by exactly one translation unit: the host stub and the
// function both take external linkage, so a second includer is a hard
// `multiple definition` at link EVEN IF IT NEVER LAUNCHES IT (§21.6). A
// second consumer means templating those five first, which is a body change
// and needs its own parity evidence.
#include "attn/kv_paged.cuh"
#include "attn/kv_paged.hpp"

#include <cuda_fp8.h>
#include <stdexcept>

#include "cuda_check.hpp"
// `layout/envelope.hpp` WAS included here. Both call sites below are gone
// and so is the header: `driver-cuda/src/fire/envelope.rs` is the envelope
// tier now.

namespace pie_cuda_driver::kernels::attn {

// ── `write_kv_to_pages_bf16` AND `write_kv_to_pages` WERE HERE ────────────
//
// Both are DELETED, and they are Rust:
// `driver-cuda/src/fire/kv_paged.rs::write_kv_to_pages_bf16` and
// `::write_kv_to_pages`, with `::max_touched_pages` for the bound the
// envelope refresh is launched over.
//
// `write_kv_to_pages_bf16` had no `table` row of its own -- it was a C++
// helper the dispatcher called -- so it had no shim entry to drop and its
// two `<<<>>>` closed by deletion alone. `attn::write_kv_to_pages` is a
// `table::attn` row and a live one: `model-compiler/src/dsl.rs:7408` wraps
// it and every fire calls it once per layer. It closed through
// `execution::RUST_SERVED`, which is what makes `abi::emit_c_shim` stop
// emitting `pie_k_attn_write_kv_to_pages`; the row is still fully sourced,
// `emit_rust_dispatch` still writes its arm, and the arm now lands in
// `bind::service::attn_write_kv_to_pages`. It was classified
// `Execution::Walk` -- `Control::Switch { on: "layer.scheme" }` -- before it
// was taken over, which `every_taken_over_row_was_classified_first`
// requires.
//
// WHAT WENT WITH THEM. The envelope call at the old `:145`,
// `kernels::layout::launch_envelope_update_appended_bf16`, was one of this
// file's two reasons to include `layout/envelope.hpp`. Removing both is what
// let `layout/envelope.cu` be deleted whole.

// `write_kv_to_pages_at_positions_bf16` WAS HERE, and it is deleted.
//
// It held two `<<<>>>` -- one per cache half -- and `launch_abi.rs` recorded
// it as `NoRow::KernelsInternal`. The audit measured the sibling that called
// it and found none reachable. The KERNEL is untouched: `attn/kv_paged.cuh`
// still carries `write_kv_at_positions` and `families::attn::KV_PAGED`'s
// twenty rows still compile it, so a caller that wants explicit positions has
// a row to fire rather than eleven arguments to assemble.

// ── `write_kv_explicit_bf16_devwin` WAS HERE, AND IT IS DELETED ──────────
//
// It is `driver-cuda/src/fire/kv_paged.rs::write_kv_explicit_bf16_devwin`,
// reached through `bind::service::attn_write_kv_explicit_bf16_devwin`.
//
// The paragraph that stood here for two passes said the block was a caller
// and not a classification -- §58's reading, that a `Specialisation` IS the
// walk, so this symbol needed neither `Walk`, `RUST_SERVED` nor a
// `bind::service` shim. That reading was right about the mechanism and wrong
// about the consequence: the only thing that CAN call the Rust is a generated
// dispatch arm, and the emitter writes one only for `device::JIT_DISPATCHED`
// or `execution::RUST_SERVED`. "It needs a caller" and "it needs a
// classification" were the same sentence.
//
// §60.6 dissolves it. The DEVICE rows are now
// `attn::write_kv_explicit_bf16_devwin_dev` and `WRITE_KV_EXPLICIT_DEVWIN`'s
// `base` moved with them, so `unit_of("attn::write_kv_explicit_bf16_devwin")`
// is `None`, the `Walk` is legal, and the `Specialisation` still resolves
// against a base that exists. The SIBLING `attn::write_kv_explicit_bf16` was
// already in exactly this arrangement -- see `families/attn.rs`'s
// `write_kv_explicit` comment -- which is the evidence that it is an
// arrangement and not a dodge.
//
// The two `<<<n_max, 256, 0, stream>>>` are quoted line by line in the Rust,
// and both `throw std::runtime_error` are `assert!` there with the same two
// messages: a refusal is never a fallback, and neither of those conditions
// is one the launch may decline. `.wiki/driver/new-horizon.md` §56.1 and §59
// carry the account they were written for; §58's correction is above.

// ── `write_kv_explicit_bf16` WAS HERE, AND IT IS DELETED ─────────────────
//
// It is `driver-cuda/src/fire/kv_paged.rs::write_kv_explicit_bf16`, reached
// through `bind::service::attn_write_kv_explicit_bf16`.
//
// THE ROW IS LIVE AND STAYED LIVE. `table::attn`'s `write_kv_explicit` is
// fully sourced and `model-compiler/src/dsl.rs:7393` wraps it, so a model
// trace reaches it exactly as before; `execution::RUST_SERVED` naming the
// symbol is what stops `abi::emit_c_shim` emitting
// `pie_k_attn_write_kv_explicit_bf16`, and `emit_rust_dispatch`'s arm lands
// in Rust instead. It was classified `Execution::Walk` first.
//
// THE SYMBOL SPLIT THAT MADE THAT LEGAL, because it is the interesting part.
// `execution::tests::a_walk_is_only_a_walk` requires a `WALKED` symbol to
// satisfy `unit_of(sym).is_none()` (§52.11, *a walk may drive a JIT'd
// kernel; it may not be one*), and this symbol had a DEVICE row of the same
// name -- the base of `SPECIALISATIONS`' `WRITE_KV_EXPLICIT`. §60.6 fixes
// the direction: the ahead-of-time row's name is the one a trace records, so
// the DEVICE rows moved. They are `attn::write_kv_explicit_bf16_dev` and its
// `#hnd`/`#nhd` arms now, and the Rust fires those two directly.
//
// This launcher's second reason to exist was the envelope merge at the old
// `:344`. That call and `write_kv_to_pages`' are the two that held
// `layout/envelope.hpp` in this file; both are gone, and so is
// `layout/envelope.cu`.

// `copy_kv_cells_bf16` IS GONE, and it is the first `kv_paged.cu` launcher
// deleted rather than staged. Two `<<<>>>`: `copy_kv_cells<true>` at the old
// `:367` and `copy_kv_cells<false>` at `:373`, both `<<<N, 256, 0, stream>>>`.
//
// Consumer set, swept on all five channels and empty on four:
//
//   * C++ across `.cu`/`.cuh`/`.cpp`/`.hpp`: nothing but the declaration in
//     `kv_paged.hpp`, deleted with it.
//   * `crates/model/src`: neither `attn::copy_kv_cells_bf16` nor a DSL
//     wrapper name. There is no wrapper -- `model-compiler/src/dsl.rs` does
//     not state it, which is why its row was `driver_internal` and not a
//     `table/attn.rs` row.
//   * `lower.rs::semantic()`: no reading picks it.
//   * The hand-written `ffi::pie_k_*` arms in `driver-cuda/src`: ONE, at
//     `serve/transfer.rs:321`, and it is Rust. That made the move a
//     Rust-to-Rust edit rather than a port -- it now calls
//     `fire::kv_paged::copy_kv_cells_bf16` directly.
//
// The `table/driver_internal.rs` row went in the SAME edit, and had to:
// a row that states `operands` and is in neither `device::JIT_DISPATCHED`
// nor `execution::RUST_SERVED` still gets an `emit_c_shim` forwarder, and a
// forwarder onto a deleted launcher is the one failure that stops the whole
// workspace compiling. Routing instead was not available -- a
// `driver_internal` row is not in `table::TABLES`, so `table::sig` cannot
// resolve it and `every_taken_over_row_is_stated` refuses `RUST_SERVED`.
// `driver-cuda/tests/launch_abi.rs`'s `("copy_kv_cells_bf16",
// NoRow::DriverInternal)` went too, for the reason that list already records
// for `set_decode_plan_int_base`: `mentions_word` reads all of
// `driver-cuda/src`, and a ported name left standing there fails the
// `Orphaned` arm for a launcher that is not orphaned.
//
// The two DEVICE rows stay: `families/attn.rs:3293`/`:3301` are what the
// Rust fires, and `SPECIALISATIONS`' `COPY_KV_CELLS` still resolves its base.

void dequant_kv_cache_layer_to_bf16_active(
    KvCacheLayerView layer,
    const std::uint32_t* kv_page_indices,
    int num_pages_in_batch,
    cudaStream_t stream)
{
    if (layer.is_native_bf16() || num_pages_in_batch <= 0) return;
    constexpr int BLOCK = 256;
    const int page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
    const long long logical_n =
        static_cast<long long>(num_pages_in_batch) * page_elems;
    const auto blocks = static_cast<unsigned>((logical_n + BLOCK - 1) / BLOCK);

    switch (layer.scheme) {
        case KvCacheScheme::Fp8PerTensor: {
            const auto fp8_kind = layer.storage_dtype == DType::FP8_E5M2
                ? __NV_E5M2
                : __NV_E4M3;
            device::dequant_fp8_pages_active<<<blocks, BLOCK, 0, stream>>>(
                static_cast<const __nv_fp8_storage_t*>(layer.k_pages),
                static_cast<const __nv_fp8_storage_t*>(layer.v_pages),
                static_cast<device::bf16*>(layer.k_bf16_pages),
                static_cast<device::bf16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, page_elems, fp8_kind);
            break;
        }
        case KvCacheScheme::Fp8PerTokenHead:
            device::dequant_fp8_per_token_head_pages_active<device::bf16>
                <<<blocks, BLOCK, 0, stream>>>(
                static_cast<const __nv_fp8_storage_t*>(layer.k_pages),
                static_cast<const __nv_fp8_storage_t*>(layer.v_pages),
                static_cast<const float*>(layer.k_scales),
                static_cast<const float*>(layer.v_scales),
                static_cast<device::bf16*>(layer.k_bf16_pages),
                static_cast<device::bf16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, layer.page_size, layer.num_kv_heads,
                layer.head_dim);
            break;
        case KvCacheScheme::Int8PerTokenHead:
            device::dequant_int8_per_token_head_pages_active<device::bf16>
                <<<blocks, BLOCK, 0, stream>>>(
                static_cast<const std::int8_t*>(layer.k_pages),
                static_cast<const std::int8_t*>(layer.v_pages),
                static_cast<const float*>(layer.k_scales),
                static_cast<const float*>(layer.v_scales),
                static_cast<device::bf16*>(layer.k_bf16_pages),
                static_cast<device::bf16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, layer.page_size, layer.num_kv_heads,
                layer.head_dim);
            break;
        case KvCacheScheme::Fp4Block: {
            const int block_size = layer.block_size > 0
                ? layer.block_size
                : 16;
            device::dequant_fp4_pages_active<device::bf16>
                <<<blocks, BLOCK, 0, stream>>>(
                static_cast<const std::uint8_t*>(layer.k_pages),
                static_cast<const std::uint8_t*>(layer.v_pages),
                static_cast<const float*>(layer.k_scales),
                static_cast<const float*>(layer.v_scales),
                static_cast<device::bf16*>(layer.k_bf16_pages),
                static_cast<device::bf16*>(layer.v_bf16_pages),
                kv_page_indices, logical_n, layer.page_size, layer.num_kv_heads,
                layer.head_dim, block_size);
            break;
        }
        case KvCacheScheme::Native:
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════
// `build_window_page_view` AND `build_full_split_view` WERE HERE AND ARE
// DELETED
// ═════════════════════════════════════════════════════════════════════════
//
// Two launches, `<<<1, 256>>>` and `<<<1, 32>>>`, both now Rust:
// `driver-cuda/src/fire/kv_paged.rs::build_window_page_view` and
// `::build_full_split_view`. The `<<<>>>` lines are quoted verbatim in the
// doc comment above each, and the reason `build_full_split_view` is one warp
// — `kv_paged.cuh:842` is `if (threadIdx.x != 0) return;`, so every thread
// but one exits and 32 is the smallest thing the hardware schedules —
// travelled with it rather than being consumed by the port.
//
// Their two `table::attn` rows and their two `dsl::cuda` wrappers went in the
// same change. Both rows were unsourced on every operand, so neither ever
// generated a dispatch, and `crates/model/src` named neither the symbols nor
// the wrappers. The DEVICE rows stay.
//
// Neither Rust function carries an `Execution` classification. §58: a single
// launch with no choice and no loop needs none.

}  // namespace pie_cuda_driver::kernels::attn
