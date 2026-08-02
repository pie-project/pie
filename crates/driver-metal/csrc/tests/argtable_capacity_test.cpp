// Argument table capacity matches what MSL can actually reference.
//
// `MTL4ArgumentTable` fixes its `maxBufferBindCount` at creation, and the driver
// creates every table with 31 slots. That number is not arbitrary: MSL rejects
// `[[buffer(n)]]` for n > 30, so 31 bindings is the hard ceiling a kernel can
// reference and a larger table would only waste space. This test pins that
// contract down, because the driver's bind index is a `uint8_t` and nothing else
// in the type system stops a caller from passing 31 or more -- an argument table
// ignores an out-of-range index silently, which would surface as a kernel
// reading an unbound address rather than as an error.
//
// Runs on-device: it fills every one of the 31 slots and reads them all back.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "mtl4_context.hpp"

using pie::metal::Grid;
using pie::metal::RawMetalContext;
using pie::metal::SlotHandle;
using pie::metal::StepEncoder;
using pie::metal::StepTiming;
using pie::metal::Threadgroup;

namespace {
int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

// Reads one uint from each of 40 separate bindings -- well past the old 31-slot
// ceiling -- and sums them, so a dropped bind shows up as a wrong total.
constexpr const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

#define B(n) constant uint& b##n [[buffer(n)]]

// Buffer 30 is the highest index MSL accepts, so this signature uses every
// binding an argument table can ever be asked for.
kernel void sum31(device uint* out [[buffer(0)]],
                  B(1),  B(2),  B(3),  B(4),  B(5),  B(6),  B(7),  B(8),  B(9),
                  B(10), B(11), B(12), B(13), B(14), B(15), B(16), B(17), B(18),
                  B(19), B(20), B(21), B(22), B(23), B(24), B(25), B(26), B(27),
                  B(28), B(29), B(30)) {
    uint s = 0;
    s += b1;  s += b2;  s += b3;  s += b4;  s += b5;  s += b6;  s += b7;
    s += b8;  s += b9;  s += b10; s += b11; s += b12; s += b13; s += b14;
    s += b15; s += b16; s += b17; s += b18; s += b19; s += b20; s += b21;
    s += b22; s += b23; s += b24; s += b25; s += b26; s += b27; s += b28;
    s += b29; s += b30;
    out[0] = s;
}
)MSL";

int finish(const char* name) {
    std::printf("\n==== %s: %d passed, %d failed ====\n", name, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
}  // namespace

int main() {
    std::printf("[Metal 4 argument tables: full 31-slot capacity + out-of-range rejection]\n");

    auto ctx = RawMetalContext::create(16u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) {
        return finish("argtable_capacity_test");
    }

    std::string err;
    auto pso = ctx->compile_pso(kSrc, "sum31", &err);
    if (!expect(pso.valid(), "31-binding kernel (the MSL maximum) compiles under MSL 4.0 (" + err + ")")) {
        return finish("argtable_capacity_test");
    }

    constexpr std::uint8_t kBinds = 31;  // indices 0..30, the MSL maximum
    constexpr int kOrdinal = 7;

    auto out = ctx->create_standalone_buffer(sizeof(std::uint32_t));
    if (!expect(out.contents_ptr != nullptr, "output buffer allocated")) {
        return finish("argtable_capacity_test");
    }
    *static_cast<std::uint32_t*>(out.contents_ptr) = 0u;

    std::vector<SlotHandle> args;
    std::uint32_t want = 0;
    for (std::uint8_t i = 1; i < kBinds; ++i) {
        auto s = ctx->create_standalone_buffer(sizeof(std::uint32_t));
        if (s.contents_ptr == nullptr) {
            expect(false, "argument buffer allocated");
            return finish("argtable_capacity_test");
        }
        // Distinct powers-ish values so a dropped or duplicated bind changes the
        // sum rather than cancelling out.
        const std::uint32_t v = 1u + std::uint32_t(i) * 37u;
        *static_cast<std::uint32_t*>(s.contents_ptr) = v;
        want += v;
        args.push_back(s);
    }
    ctx->make_resident();

    ctx->arg_bind_ordinal(kOrdinal, 0, out);
    for (std::uint8_t i = 1; i < kBinds; ++i) {
        ctx->arg_bind_ordinal(kOrdinal, i, args[i - 1]);
    }

    for (std::uint8_t i = 0; i < kBinds; ++i) {
        if (!ctx->arg_slot_is_bound(kOrdinal, i)) {
            expect(false, "bind index " + std::to_string(i) + " is tracked");
            return finish("argtable_capacity_test");
        }
    }
    expect(true, "all 31 bind indices are tracked");

    const StepTiming timing = ctx->run_step([&](StepEncoder& se) {
        se.set_pso(pso);
        se.set_argtable_ordinal(kOrdinal);
        se.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
    });
    expect(timing.completed && !timing.timed_out, "dispatch completes");
    expect(!timing.gpu_error, "no GPU error");

    const std::uint32_t got = *static_cast<const std::uint32_t*>(out.contents_ptr);
    expect(got == want,
           "kernel read all 30 argument bindings, including index 30 (got " +
               std::to_string(got) + ", want " + std::to_string(want) + ")");

    // Releasing the ordinal must drop the grown capacity too, so a later reuse
    // of the same ordinal starts from a clean table rather than a stale one.
    // Past the ceiling the bind must be rejected rather than recorded, so the
    // caller cannot end up believing a slot is live when the GPU never sees it.
    std::printf("  (one diagnostic line below is expected)\n");
    ctx->arg_bind_ordinal(kOrdinal, 31, out);
    expect(!ctx->arg_slot_is_bound(kOrdinal, 31),
           "a bind past the MSL ceiling is rejected, not silently dropped");

    ctx->release_argtable_ordinal(kOrdinal);
    expect(!ctx->arg_slot_is_bound(kOrdinal, 30),
           "release_argtable_ordinal clears the table's binds");

    return finish("argtable_capacity_test");
}
