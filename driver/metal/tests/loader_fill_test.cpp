// `StorageInstr::Fill` on a real Metal heap.
//
// The loader emits `Fill` when an expression pads: the padded region is memory
// no source covers, and the compiler prices it as one fill rather than one copy
// per band. `heap_bind.cpp` implements it as a host-side `memset` over the slot,
// on two assumptions that were reasoned about and never executed:
//
//   * the weights region is host-visible, so `SlotHandle::contents()` is a real
//     CPU pointer; and
//   * program order is enough, because Metal's writes are synchronous host
//     stores — unlike CUDA's, which are stream-async and need a flush before
//     the fill.
//
// No shipping contract pads, so nothing reached the arm. This file reaches it:
// it writes a checkpoint, authors a contract that pads, compiles it with the
// real loader, and stages the result into a real Metal heap.
//
// One thing measured here decides how the checks below are written. A placement
// buffer cut from a fresh `MTLHeap` arrives **zeroed** — always, on this
// platform, even when the heap's memory was dirtied and released moments
// earlier. So on Metal the *zeroing* half of `Fill` is unobservable end to end,
// and the half that can actually break is the ordering: a `Fill` that ran after
// the copies it precedes would erase live weights. `test_the_heap_arrives_zeroed`
// pins the platform fact so this reasoning is not silently invalidated, and
// `heap_storage_is_host_visible_and_slices_are_exact` covers the zeroing itself
// against memory the test dirtied on purpose.
//
// Apple-only, because staging needs `RawMetalContext`. The plan-shape half
// would run anywhere, but splitting it across two binaries to say so would
// leave the interesting half still needing a Mac.


#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "pie_loader.h"
#include "pie_loader/checkpoint_source.hpp"
#include "pie_loader/model_contract.hpp"
#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"
#include "pie_loader/source_checkpoint.hpp"

#include "heap_bind_metal.hpp"
#include "loader/load_plan.hpp"
#include "loader/transcode.hpp"
#include "model/qwen3_5/geometry.hpp"
#include "mtl4_context.hpp"

using pie::metal::RawMetalContext;
using pie::metal::SlotHandle;

namespace {

int g_pass = 0, g_fail = 0;

bool expect(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
        std::printf("  PASS  %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", what.c_str());
    }
    return ok;
}

// -- part 1: what the arm assumes about heap storage -------------------------

// `Fill` is the only arm that dereferences `contents()` without first checking a
// bound, so if the heap were ever Private it would be the one that faults rather
// than the one that reports. It also writes through a *sliced* handle, and the
// slice arithmetic (`heap_bind.cpp::slice_slot`) is what keeps it inside its own
// buffer: a fill that used the parent's size would erase the tensor next door.
bool heap_storage_is_host_visible_and_slices_are_exact() {
    auto ctx = RawMetalContext::create(4u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) return false;

    constexpr std::size_t kRegion = 4096;
    const SlotHandle region = ctx->heap_alloc(kRegion);
    if (!expect(region.valid(), "the weights region allocates")) return false;
    if (!expect(region.contents() != nullptr,
                "the weights region is host-visible (Shared, not Private)")) {
        return false;
    }

    // Dirty the whole region so "it is zero afterwards" cannot be an accident of
    // a freshly faulted page.
    auto* base = static_cast<std::uint8_t*>(region.contents());
    std::memset(base, 0xAB, kRegion);

    // Two adjacent slots, exactly as `Allocate` carves them out of the region.
    const std::size_t split = 1024;
    SlotHandle target = region;
    target.contents_ptr = base;
    target.size = split;
    SlotHandle neighbour = region;
    neighbour.contents_ptr = base + split;
    neighbour.size = kRegion - split;

    // The arm, verbatim.
    std::memset(target.contents(), 0, target.size);

    bool filled = true, untouched = true;
    for (std::size_t i = 0; i < kRegion; ++i) {
        if (i < split) {
            filled = filled && base[i] == 0x00;
        } else {
            untouched = untouched && base[i] == 0xAB;
        }
    }
    expect(filled, "Fill zeroes every byte of its own slot");
    expect(untouched, "Fill touches no byte outside its slot");
    return true;
}

// -- part 2: what a fresh heap already gives you ------------------------------

// The reason the end-to-end zero check below is corroborating rather than
// decisive. Dirty a heap, drop it, take a new one: the bytes come back zero.
// If this ever stops holding, the staging check downstream becomes the real
// test of `Fill` and this one names the day it changed.
bool the_heap_arrives_zeroed() {
    constexpr std::size_t kHeap = 8u << 20;
    constexpr std::size_t kProbe = 1u << 20;
    {
        auto dirty = RawMetalContext::create(kHeap);
        if (!expect(dirty != nullptr, "RawMetalContext::create succeeds")) return false;
        const SlotHandle whole = dirty->heap_alloc(kHeap - kProbe);
        if (!expect(whole.valid() && whole.contents() != nullptr,
                    "a heap-wide slot allocates")) {
            return false;
        }
        std::memset(whole.contents(), 0xCD, whole.size);
    }

    auto fresh = RawMetalContext::create(kHeap);
    if (!expect(fresh != nullptr, "a second RawMetalContext creates")) return false;
    const SlotHandle probe = fresh->heap_alloc(kProbe);
    if (!expect(probe.valid() && probe.contents() != nullptr, "the probe slot allocates")) {
        return false;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(probe.contents());
    bool zeroed = true;
    for (std::size_t i = 0; i < probe.size; ++i) zeroed = zeroed && bytes[i] == 0;
    expect(zeroed, "a placement buffer from a fresh heap arrives zeroed");
    return true;
}

// -- part 3: a contract that pads, end to end --------------------------------

/// A one-tensor safetensors checkpoint whose bytes are recognisable.
///
/// BF16 `[kRows, kCols]`, element `(r, c)` = `0x0100 + r * kCols + c`. Any byte
/// the plan moves is therefore identifiable by position, which is what lets the
/// check below tell "copied to the right place" from "copied at all".
constexpr std::int64_t kRows = 4;
constexpr std::int64_t kCols = 8;
constexpr std::int64_t kPadBefore = 2;
constexpr std::int64_t kPadAfter = 2;

std::uint16_t source_element(std::int64_t r, std::int64_t c) {
    return static_cast<std::uint16_t>(0x0100 + r * kCols + c);
}

std::filesystem::path write_checkpoint() {
    const auto dir = std::filesystem::temp_directory_path() /
                     ("pie_metal_fill_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);

    std::vector<std::uint16_t> data(static_cast<std::size_t>(kRows * kCols));
    for (std::int64_t r = 0; r < kRows; ++r) {
        for (std::int64_t c = 0; c < kCols; ++c) {
            data[static_cast<std::size_t>(r * kCols + c)] = source_element(r, c);
        }
    }
    const std::size_t payload = data.size() * sizeof(std::uint16_t);
    const std::string header = "{\"w\":{\"dtype\":\"BF16\",\"shape\":[" + std::to_string(kRows) +
                               "," + std::to_string(kCols) + "],\"data_offsets\":[0," +
                               std::to_string(payload) + "]}}";

    const auto path = dir / "model.safetensors";
    std::ofstream out(path, std::ios::binary);
    const std::uint64_t len = header.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(payload));
    out.close();
    return dir;
}

bool a_padded_contract_stages_zeros_where_no_source_reaches() {
    const auto dir = write_checkpoint();

    std::string open_error;
    pie_loader::Checkpoint checkpoint = pie_loader::Checkpoint::open(dir.string(), &open_error);
    if (!expect(static_cast<bool>(checkpoint), "the fixture checkpoint opens: " + open_error)) {
        return false;
    }

    // The contract no shipping family authors yet: rows of zeros above and below
    // the tensor that is actually on disk.
    const auto target = pie::metal::metal_device_target();
    pie_loader::ModelContract contract;
    contract.align(target.preferred_alignment);
    const auto bf16 = pie_loader::raw(pie_loader::PieLoaderDType::BF16);
    contract.define("padded",
                    contract.concat({contract.fill(0.0f, {kPadBefore, kCols}, bf16),
                                     contract.src("w"),
                                     contract.fill(0.0f, {kPadAfter, kCols}, bf16)},
                                    0),
                    bf16)
        .expect({kRows + kPadBefore + kPadAfter, kCols});

    const auto request =
        pie_loader::build_contract_request(checkpoint, target, contract.view());
    pie_loader::LoadPlan plan;
    try {
        plan = pie_loader::LoadPlan::compile(request);
        plan.verify(request);
    } catch (const std::exception& error) {
        expect(false, std::string("compiling the padded contract: ") + error.what());
        return false;
    }

    const auto view = plan.view();
    std::size_t fills = 0, fill_step = 0, first_write_step = view.schedule.len;
    for (std::size_t step = 0; step < view.schedule.len; ++step) {
        const auto& instr = view.instrs.ptr[view.schedule.ptr[step]];
        using Tag = pie_loader::PieLoaderStorageOp::Tag;
        if (instr.op.tag == Tag::Fill) {
            ++fills;
            fill_step = step;
        }
        if ((instr.op.tag == Tag::ExtentWrite ||
             instr.op.tag == Tag::BulkExtentWrite) &&
            step < first_write_step) {
            first_write_step = step;
        }
    }
    if (!expect(fills == 1, "a padding contract compiles to exactly one Fill")) return false;
    // The Rust side calls this `validate-fill-order`. `heap_bind.cpp` relies on
    // it rather than re-deriving it, so this is where the reliance is checked.
    expect(fill_step < first_write_step, "the Fill precedes every write to the buffer");

    // Stage it. `n_layers = 0` keeps the KV/GDN loops empty — the point here is
    // the storage schedule, not the model.
    pie::metal::DecodeGeometry geometry;
    geometry.n_layers = 0;
    geometry.vocab = 64;
    geometry.max_tokens = 1;
    geometry.paged_kv_enabled = false;

    pie::metal::HeapPlan heap_plan;
    heap_plan.weights_bytes = view.memory.persistent_bytes;
    heap_plan.scratch_slot_bytes = 4096;

    auto ctx = RawMetalContext::create(16u << 20, 64u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) return false;

    auto source = std::make_shared<pie_loader::CheckpointSource>(view);
    pie::metal::BoundDecode bound;
    try {
        bound = pie::metal::stage_decode_storage(*ctx, std::move(source), plan, geometry,
                                                 heap_plan);
    } catch (const std::exception& error) {
        expect(false, std::string("staging the padded plan: ") + error.what());
        return false;
    }

    const auto found = bound.weights.find("padded");
    if (!expect(found != bound.weights.end(), "the staged tensor is published under its name")) {
        return false;
    }
    const SlotHandle& slot = found->second;
    const std::int64_t rows = kRows + kPadBefore + kPadAfter;
    if (!expect(slot.contents() != nullptr && slot.size >= static_cast<std::size_t>(rows * kCols * 2),
                "the staged tensor is host-readable and large enough")) {
        return false;
    }

    const auto* got = static_cast<const std::uint16_t*>(slot.contents());
    bool pad_is_zero = true, body_survived = true;
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < kCols; ++c) {
            const std::uint16_t value = got[r * kCols + c];
            if (r < kPadBefore || r >= kPadBefore + kRows) {
                pad_is_zero = pad_is_zero && value == 0;
            } else {
                body_survived =
                    body_survived && value == source_element(r - kPadBefore, c);
            }
        }
    }
    expect(pad_is_zero, "the padded rows come back zero");
    // The decisive one. A `Fill` that ran after the copies — the hazard CUDA
    // answers with `copy_engine_.flush()` and this driver answers with nothing
    // but program order — would leave zeros here too, and every other check in
    // this file would still pass.
    expect(body_survived, "the copied rows survive the Fill");

    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    return true;
}

/// The encoder either reproduces `mx.quantize` or it does not.
///
/// This is the only test in the tree with a hardcoded oracle, and it earns it.
/// `push_encoded_affine` exists so a checkpoint MLX has not quantized can be
/// read by kernels that only speak MLX affine U4 -- gpt-oss's openai-native
/// `_blocks`/`_scales` experts and its unquantized BF16 attention. No
/// checkpoint on disk takes that path, so no end-to-end gate covers it, and the
/// encoder drifted 8.2% away from `mx.quantize` without anything going red.
///
/// Two of the three ways it had drifted are pinned below; the third, `round`
/// rather than `rint` on the endpoint snap, is not. `edge / scale` is within a
/// hair of +/-15 by construction and never lands on a half-integer for any
/// input tried, so the two spellings cannot be told apart there. It is written
/// MLX's way anyway, because the reason to write it either way is fidelity.
///
/// The values below are `mx.quantize(x, group_size=64, bits=4)` on the input
/// generated below, with the scales and biases rounded to BF16 as the encoder
/// stores them.
const std::uint32_t kOracleCodes[] = {
    0xd8859cb8u, 0x64c53cf7u, 0x98fba0f9u, 0xa05f8484u, 0x8d88c9b9u, 0x998884fbu, 0xb683b698u, 0xd9dacf7du,
    0x5c8eb59eu, 0x95ebfb6bu, 0xbc97759bu, 0xcb86b579u, 0x3ba9bb65u, 0x8b0ecb90u, 0x8850b9e5u, 0x558e90b5u,
    0xd37df964u, 0x4d470997u, 0x8b45c56au, 0x848379c9u, 0x8da84897u, 0x76a7f7f6u, 0x76fbb838u, 0x77a488cbu,
    0x9b4d8044u, 0xc6947f85u, 0x99c34bf6u, 0xca530aa8u, 0xd756bf8cu, 0x789a8f88u, 0x88663477u, 0xc457479du,
    0x98667459u, 0x087a4d49u, 0xf84cc984u, 0x5989647du, 0xa74975d9u, 0xdcf09aa7u, 0xaa87d456u, 0x69ab373du,
    0xbaf6fcf3u, 0x5d675608u, 0x69899b79u, 0xdd0348f3u, 0xd7b7b986u, 0x8f8f9894u, 0x4737d898u, 0x7d07a999u,
    0x978fb9fcu, 0x8d83777bu, 0x8ad00378u, 0xcca540a3u, 0xbb588c08u, 0x70668533u, 0x9d496998u, 0x93c498ccu,
    0xd08f0d79u, 0x7b73950du, 0x804f0d88u, 0x4b3a74f3u, 0xc8c7b0cbu, 0xbf89f667u, 0xd8963f7au, 0x0a5f77bfu,
    0x9b777098u, 0x40470c07u, 0x8067afd8u, 0x9d89ff0du, 0xd77898fau, 0x5a654a09u, 0x7a73ca69u, 0x777b58bcu,
    0x6339d8ffu, 0xd8769a7bu, 0x697c8780u, 0x0ca933c9u, 0x39774dc7u, 0xf68b970bu, 0x475a3707u, 0x877fdcd8u,
    0x798bfb98u, 0x965f0b74u, 0xdd795858u, 0xb78b9c8au, 0x685dcbc7u, 0x86989966u, 0x4679887bu, 0x47858a8bu,
    0x998ad7bdu, 0x933d7f9cu, 0x95f9c867u, 0xd8997cfdu, 0xfa7c3330u, 0x99484d50u, 0xa687a6bbu, 0x9473b070u,
    0xad7b5c7fu, 0x96a089d7u, 0xd8937c99u, 0xc5c86a3au, 0xfb583ac7u, 0x6d64c309u, 0x99778977u, 0x96ffa866u,
    0x6b6a3b8fu, 0x8d590957u, 0x77075090u, 0xfc754594u, 0x099d4db7u, 0xa5b459a4u, 0xdf966d68u, 0xa846a085u,
    0x9f88335fu, 0x69cd8596u, 0x9790b8c7u, 0x789058bbu, 0x88839739u, 0x947778d5u, 0xa58849a9u, 0x747ff4acu,
    0x008e8b8bu, 0x080eeeeeu, 0xbb8800bbu, 0xbebe88bbu, 0xbe8e80b8u, 0x8bbee08eu, 0x8088eeeeu, 0xb0e0e800u,
};
const std::uint16_t kOracleScales[] = {
    0xbe40u, 0xc02bu, 0xbf40u, 0xbf40u, 0xc040u, 0xbf40u, 0xbf40u, 0xbfc0u,
    0xc040u, 0xc040u, 0xbf40u, 0xbfc0u, 0xbec0u, 0xbf40u, 0xbec0u, 0x3e4du,
};
const std::uint16_t kOracleBiases[] = {
    0x3fc0u, 0x41c0u, 0x40c0u, 0x40c0u, 0x41c0u, 0x40c0u, 0x40c0u, 0x4140u,
    0x41c0u, 0x41c0u, 0x40c0u, 0x4140u, 0x4040u, 0x40c0u, 0x4040u, 0xc040u,
};

void the_encoder_reproduces_mlx_quantize_exactly() {
    std::printf("[affine encoding]\n");
    using namespace pie::metal::transcode;
    constexpr std::int64_t kRows = 8;
    constexpr std::int64_t kCols = 128;  // two groups of 64

    // MXFP4-derived values: eight E2M1 levels times a per-group power of two,
    // which is exactly what `push_encoded_affine`'s live caller feeds it -- a
    // gpt-oss expert bank dequantized from `_blocks`/`_scales`. It matters that
    // the input is this and not uniform noise: on a lattice, `(w - bias)/scale`
    // lands on an exact half-integer often, and a half-integer is the only
    // place a rounding mode is visible at all. The seed is chosen, not
    // arbitrary: most seeds put every tie at 7.5, where half-away-from-zero and
    // half-to-even both answer 8 and the choice cannot be seen. This one puts
    // 22 of them elsewhere.
    static const float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    std::vector<float> values(static_cast<std::size_t>(kRows * kCols));
    std::uint32_t seed = 0x17u;
    auto next = [&seed] {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 16;
    };
    for (std::int64_t r = 0; r < kRows; ++r) {
        for (std::int64_t g = 0; g < 2; ++g) {
            const float sc = std::ldexp(1.0f, -2 + int(next() % 5u));
            // The last group is forced entirely negative. It is the only way the
            // zero-initialised `w_max` becomes visible: everywhere else the
            // group maximum is above zero and the initial value is overwritten.
            const bool all_negative = (r == kRows - 1 && g == 1);
            for (std::int64_t i = 0; i < 64; ++i) {
                const std::uint32_t t = next();
                float v = kE2M1[all_negative ? ((t & 7u) | 1u) : (t & 7u)] * sc;
                if (all_negative || (t & 8u)) v = -v;
                values[static_cast<std::size_t>(r * kCols + g * 64 + i)] = v;
            }
        }
    }

    std::vector<std::uint32_t> codes(static_cast<std::size_t>(kRows * kCols / 8));
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(kRows * 2));
    std::vector<std::uint16_t> biases(scales.size());
    encode_mlx_affine_u4(
        reinterpret_cast<const std::uint8_t*>(values.data()), /*input_element_bytes=*/4, kRows,
        kCols,
        Region{reinterpret_cast<std::uint8_t*>(codes.data()), codes.size() * 4},
        Region{reinterpret_cast<std::uint8_t*>(scales.data()), scales.size() * 2},
        Region{reinterpret_cast<std::uint8_t*>(biases.data()), biases.size() * 2});

    static_assert(sizeof(kOracleCodes) / sizeof(kOracleCodes[0]) == kRows * kCols / 8, "");
    int wrong_codes = 0;
    for (std::size_t i = 0; i < codes.size(); ++i) {
        for (int k = 0; k < 8; ++k) {
            const auto mine = int((codes[i] >> (4 * k)) & 0xf);
            const auto want = int((kOracleCodes[i] >> (4 * k)) & 0xf);
            if (mine != want) ++wrong_codes;
        }
    }
    expect(wrong_codes == 0, "every code is the one mx.quantize picked (" +
                                 std::to_string(wrong_codes) + " of " +
                                 std::to_string(kRows * kCols) + " are not)");

    int wrong_params = 0;
    for (std::size_t i = 0; i < scales.size(); ++i) {
        if (scales[i] != kOracleScales[i]) ++wrong_params;
        if (biases[i] != kOracleBiases[i]) ++wrong_params;
    }
    expect(wrong_params == 0, "and every scale and bias is bit-identical to MLX's, at BF16 (" +
                                  std::to_string(wrong_params) + " are not)");

    // The host encoder above is the FALLBACK. `heap_bind.cpp` reaches for
    // `transcode.metal` first and only falls back when the kernels will not
    // compile, so the same oracle has to be held against the GPU arm or the
    // path that actually runs is the untested one. The two implementations are
    // separate transcriptions of the same MLX kernel and have already drifted
    // apart once.
    auto owned = RawMetalContext::create(4u << 20);
    if (!expect(owned != nullptr, "RawMetalContext::create succeeds")) return;
    RawMetalContext& ctx = *owned;
    const char* env = std::getenv("PIE_METAL_KERNELS_DIR");
    std::string dir = env != nullptr ? std::string(env) : std::string(PIE_METAL_KERNELS_DIR_DEFAULT);
    if (!dir.empty() && dir.back() != '/') dir += '/';
    std::string error;
    const auto pso =
        ctx.compile_precise_pso_from_file(dir + "transcode.metal", "affine_encode_u4_f32", &error);
    if (!expect(pso.valid(), "affine_encode_u4_f32 compiles: " + error)) return;

    const std::uint32_t groups = std::uint32_t(kRows * kCols / 64);
    SlotHandle in = ctx.create_standalone_buffer(values.size() * 4);
    SlotHandle gc = ctx.create_standalone_buffer(codes.size() * 4);
    SlotHandle gs = ctx.create_standalone_buffer(scales.size() * 2);
    SlotHandle gb = ctx.create_standalone_buffer(biases.size() * 2);
    SlotHandle params = ctx.create_standalone_buffer(256);
    if (!expect(in.valid() && gc.valid() && gs.valid() && gb.valid() && params.valid(),
                "the encode buffers allocate")) {
        return;
    }
    std::memcpy(in.contents(), values.data(), values.size() * 4);
    const std::uint32_t args[2] = {groups, 64u};
    std::memcpy(params.contents(), args, sizeof(args));
    constexpr int kOrdinal = 30001;
    ctx.arg_bind_ordinal(kOrdinal, 0, in);
    ctx.arg_bind_ordinal(kOrdinal, 1, gc);
    ctx.arg_bind_ordinal(kOrdinal, 2, gs);
    ctx.arg_bind_ordinal(kOrdinal, 3, gb);
    ctx.arg_bind_ordinal(kOrdinal, 4, params);
    ctx.make_resident();
    const std::uint32_t width = std::max(1u, std::min({ctx.pso_max_threads(pso), 256u, groups}));
    const auto timing = ctx.run_step([&](pie::metal::StepEncoder& encoder) {
        encoder.set_pso(pso);
        encoder.set_argtable_ordinal(kOrdinal);
        encoder.dispatch(pie::metal::Grid{groups, 1, 1}, pie::metal::Threadgroup{width, 1, 1});
    });
    if (!expect(!timing.timed_out, "the encode dispatch completes")) return;

    int gpu_wrong = 0;
    for (std::size_t i = 0; i < codes.size(); ++i) {
        const auto word = static_cast<const std::uint32_t*>(gc.contents())[i];
        for (int k = 0; k < 8; ++k) {
            if (((word >> (4 * k)) & 0xf) != ((kOracleCodes[i] >> (4 * k)) & 0xf)) ++gpu_wrong;
        }
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
        if (static_cast<const std::uint16_t*>(gs.contents())[i] != kOracleScales[i]) ++gpu_wrong;
        if (static_cast<const std::uint16_t*>(gb.contents())[i] != kOracleBiases[i]) ++gpu_wrong;
    }
    expect(gpu_wrong == 0, "and the Metal encoder -- the one that actually runs -- agrees with "
                           "the same oracle (" + std::to_string(gpu_wrong) + " do not)");

    for (const SlotHandle& h : {in, gc, gs, gb, params}) ctx.release_standalone_buffer(h);
    ctx.release_argtable_ordinal(kOrdinal);
}

}  // namespace

int main() {
    std::printf("[loader Fill on Metal]\n");
    the_encoder_reproduces_mlx_quantize_exactly();
    heap_storage_is_host_visible_and_slices_are_exact();
    the_heap_arrives_zeroed();
    a_padded_contract_stages_zeros_where_no_source_reaches();
    std::printf("\n==== loader_fill_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
