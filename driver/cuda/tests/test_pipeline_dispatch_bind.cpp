#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "pipeline/dispatch.hpp"
#include "support/host_kernels.hpp"

using pie_cuda_driver::pipeline::Dispatch;

namespace {

bool expect(bool cond, const char* msg) {
    if (!cond) std::fprintf(stderr, "FAIL: %s\n", msg);
    return cond;
}

}  // namespace

int main() {
    // Only a registry key: the driver no longer checks a program hash, because
    // it no longer sees the bytes one would be taken over. Deliberately not a
    // recognisable constant -- `rng_magic_is_owned_by_the_contract` reads this
    // tree for transcribed RNG words and cannot tell an arbitrary key from one.
    constexpr std::uint64_t program_hash = 0xdeadbeef00000001ULL;

    // The launch package the engine would have handed the driver. Kept beside
    // the trace by `cargo test -p pie-compiler-tests --test cuda_golden
    // emit_driver_test_kernel`; the driver derives none of it any more.
    const std::string fixture = "golden-ptir-kernels/greedy_argmax";
    pie_cuda_driver::tests::HostKernelFixture host_kernels;
    pie_cuda_driver::tests::HostLaunchFixture host_launch;
    pie_cuda_driver::tests::HostRegionFixture host_regions;
    std::string fixture_error;
    // Sequence the loads BEFORE the message: C++ call arguments are
    // unsequenced, so passing `fixture_error.c_str()` alongside the call that
    // fills it reports an empty reason for every real failure.
    const bool fixtures_ok =
        host_kernels.load(fixture + ".kernels", &fixture_error) &&
        host_launch.load(fixture + ".launch", &fixture_error) &&
        host_regions.load(fixture + ".regions", &fixture_error);
    if (!expect(fixtures_ok, fixture_error.c_str())) return 1;

    Dispatch dispatch;
    std::string err;
    const int rc = dispatch.register_program(
        program_hash,
        host_launch.package(),
        host_kernels.slice(),
        host_regions.slice(),
        &err);
    if (!expect(rc == PIE_STATUS_OK, err.c_str())) return 1;

    std::vector<std::uint64_t> channel_ids(2);
    const PieLaunchChannelSlice declared = host_launch.package().channels;
    if (!expect(declared.len == channel_ids.size(),
                "golden channel count")) return 1;
    std::vector<PieChannelEndpointBinding> endpoints(channel_ids.size());
    for (std::size_t i = 0; i < channel_ids.size(); ++i) {
        channel_ids[i] = 1000 + i;
        const PieChannelDesc desc = pie_cuda_driver::tests::channel_desc_from(
            declared.ptr[i], channel_ids[i], 2000 + i, 3000 + i);
        if (!expect(
                dispatch.register_channel(desc, &endpoints[i], &err) ==
                    PIE_STATUS_OK,
                err.c_str())) return 1;
    }

    PieInstanceBinding binding{};
    const std::uint8_t bad_seed_bytes[8] = {};
    const PieChannelValueDesc bad_seed{
        .channel_id = channel_ids[0],
        .bytes = {.ptr = bad_seed_bytes, .len = sizeof(bad_seed_bytes)},
    };
    if (!expect(dispatch.bind_instance(
                    /*instance_id=*/76,
                    program_hash,
                    PIE_GEOMETRY_CLASS_HOST,
                    /*pacing_wait_id=*/1234,
                    channel_ids,
                    {bad_seed},
                    &binding,
                    &err) == PIE_STATUS_INVALID_ARGUMENT,
                "reject oversized seed")) return 1;
    std::vector<std::uint64_t> duplicate_ids = channel_ids;
    duplicate_ids[1] = duplicate_ids[0];
    if (!expect(dispatch.bind_instance(
                    /*instance_id=*/76,
                    program_hash,
                    PIE_GEOMETRY_CLASS_HOST,
                    /*pacing_wait_id=*/1234,
                    duplicate_ids,
                    {},
                    &binding,
                    &err) == PIE_STATUS_INVALID_ARGUMENT,
                "reject duplicate channel ids")) return 1;
    if (!expect(dispatch.bind_instance(
                    /*instance_id=*/76,
                    program_hash,
                    PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
                    /*pacing_wait_id=*/1234,
                    channel_ids,
                    {},
                    &binding,
                    &err) == PIE_STATUS_INVALID_ARGUMENT,
                "reject mismatched geometry classification")) return 1;
    if (!expect(
            err.find("cannot execute as a decode envelope") != std::string::npos,
            "geometry mismatch diagnostic")) return 1;
    const int bind_rc = dispatch.bind_instance(
        /*instance_id=*/77, program_hash, PIE_GEOMETRY_CLASS_HOST,
        /*pacing_wait_id=*/1234,
        channel_ids, {}, &binding, &err);
    if (!expect(bind_rc == PIE_STATUS_OK, err.c_str())) return 1;
    if (!expect(binding.instance_id == 77, "instance id")) return 1;
    if (!expect(
            binding.geometry_class == PIE_GEOMETRY_CLASS_HOST,
            "geometry class ack")) return 1;
    for (std::size_t i = 0; i < endpoints.size(); ++i) {
        const auto& endpoint = endpoints[i];
        if (!expect(endpoint.channel_id == channel_ids[i], "stable channel id")) return 1;
        if (!expect(endpoint.mirror_base != 0 && endpoint.word_base != 0,
                    "endpoint storage")) return 1;
        if (!expect(endpoint.head_word_index == 0 &&
                        endpoint.tail_word_index == 1 &&
                        endpoint.poison_word_index == 2 &&
                        endpoint.closed_word_index == 3,
                    "endpoint word layout")) return 1;
        if (!expect(endpoint.word_bytes == 4 * sizeof(std::uint64_t),
                    "endpoint word bytes")) return 1;
        // A close against a live endpoint used to be rejected outright. It is
        // now *deferred*: the registry refcounts attachments and retires the
        // slot at zero. The hard failure was a real bug -- a close racing an
        // instance teardown was dropped by the engine and the slot leaked --
        // so this assertion follows the driver rather than the other way
        // round (`dispatch.cu::close_channel`).
        if (!expect(dispatch.close_channel(channel_ids[i], &err) ==
                        PIE_STATUS_OK,
                    "live endpoint close defers")) return 1;
    }

    dispatch.close_instance(binding.instance_id);
    for (std::uint64_t channel_id : channel_ids) {
        // Already pending from the deferred close above; the slot retires when
        // the instance that held it goes away, so this one finds nothing left.
        const int rc = dispatch.close_channel(channel_id, &err);
        if (!expect(rc == PIE_STATUS_OK || rc == PIE_STATUS_CLOSED,
                    err.c_str())) return 1;
    }
    std::puts("test_ptir_dispatch_bind: OK");
    return 0;
}
