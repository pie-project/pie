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
#include <cstring>
#include <filesystem>
#include <fstream>
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
    contract.define("padded",
                    contract.pad(contract.src("w"), 0, kPadBefore, kPadAfter),
                    pie_loader::raw(pie_loader::PieLoaderDType::BF16))
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
        const auto kind = static_cast<pie_loader::PieLoaderStorageInstrKind>(instr.kind);
        if (kind == pie_loader::PieLoaderStorageInstrKind::Fill) {
            ++fills;
            fill_step = step;
        }
        if ((kind == pie_loader::PieLoaderStorageInstrKind::ExtentWrite ||
             kind == pie_loader::PieLoaderStorageInstrKind::BulkExtentWrite ||
             kind == pie_loader::PieLoaderStorageInstrKind::SlabScatter) &&
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

    pie_loader::CheckpointSource source(view);
    pie::metal::BoundDecode bound;
    try {
        bound = pie::metal::stage_decode_storage(*ctx, source, plan, geometry, heap_plan);
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

}  // namespace

int main() {
    std::printf("[loader Fill on Metal]\n");
    heap_storage_is_host_visible_and_slices_are_exact();
    the_heap_arrives_zeroed();
    a_padded_contract_stages_zeros_where_no_source_reaches();
    std::printf("\n==== loader_fill_test: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
