#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "ptir/ptir_dispatch.hpp"
#include "ptir/container.hpp"

using pie_cuda_driver::ptir::PtirDispatch;

namespace {

bool expect(bool cond, const char* msg) {
    if (!cond) std::fprintf(stderr, "FAIL: %s\n", msg);
    return cond;
}

std::string trim(const std::string& s) {
    const std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<std::uint8_t> out;
    for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
        out.push_back(static_cast<std::uint8_t>(
            std::stoul(hex.substr(i, 2), nullptr, 16)));
    }
    return out;
}

std::vector<std::uint8_t> load_golden_container(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (trim(line.substr(0, colon)) != "container") continue;
        return hex_to_bytes(trim(line.substr(colon + 1)));
    }
    return {};
}

std::vector<std::uint8_t> load_golden_sidecar(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (trim(line.substr(0, colon)) != "sidecar") continue;
        return hex_to_bytes(trim(line.substr(colon + 1)));
    }
    return {};
}

std::uint64_t load_golden_hash(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        if (trim(line.substr(0, colon)) != "hash") continue;
        return std::stoull(trim(line.substr(colon + 1)), nullptr, 16);
    }
    return 0;
}

}  // namespace

int main() {
    const std::string golden = "../tests/golden-ptir/greedy_argmax.txt";
    const auto bytes = load_golden_container(golden);
    if (!expect(!bytes.empty(), "load golden PTIR")) return 1;
    const auto sidecar = load_golden_sidecar(golden);
    if (!expect(!sidecar.empty(), "load golden PTIB")) return 1;
    const std::uint64_t program_hash = load_golden_hash(golden);
    if (!expect(program_hash != 0, "load golden PTIR hash")) return 1;

    PtirDispatch dispatch;
    std::string err;
    const int rc = dispatch.register_program(
        program_hash,
        pie_native::ByteSlice{bytes.data(), bytes.size()},
        pie_native::ByteSlice{sidecar.data(), sidecar.size()},
        &err);
    if (!expect(rc == PIE_STATUS_OK, err.c_str())) return 1;

    std::vector<std::uint64_t> channel_ids(2);
    pie_cuda_driver::ptir::container::Container container;
    pie_cuda_driver::ptir::container::DecodeError decode_error;
    if (!expect(
            pie_cuda_driver::ptir::container::decode(
                bytes.data(), bytes.size(), container, &decode_error),
            decode_error.detail.c_str())) return 1;
    if (!expect(container.channels.size() == channel_ids.size(),
                "golden channel count")) return 1;
    std::vector<PieChannelEndpointBinding> endpoints(channel_ids.size());
    for (std::size_t i = 0; i < channel_ids.size(); ++i) {
        channel_ids[i] = 1000 + i;
        const auto& source = container.channels[i];
        PieChannelDesc desc{};
        desc.abi_version = PIE_DRIVER_ABI_VERSION;
        desc.channel_id = channel_ids[i];
        desc.shape = {.ptr = source.shape.dims, .len = source.shape.rank};
        desc.dtype = source.dtype;
        desc.host_role = source.host_role;
        desc.seeded = source.seeded;
        desc.extern_dir = source.extern_dir < 0
            ? PIE_CHANNEL_EXTERN_NONE
            : static_cast<std::uint8_t>(source.extern_dir + 1);
        desc.capacity = source.capacity;
        desc.reader_wait_id = 2000 + i;
        desc.writer_wait_id = 3000 + i;
        desc.extern_name = {
            .ptr = reinterpret_cast<const std::uint8_t*>(
                source.extern_name.data()),
            .len = source.extern_name.size(),
        };
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
                    /*pacing_wait_id=*/1234,
                    duplicate_ids,
                    {},
                    &binding,
                    &err) == PIE_STATUS_INVALID_ARGUMENT,
                "reject duplicate channel ids")) return 1;
    const int bind_rc = dispatch.bind_instance(
        /*instance_id=*/77, program_hash, /*pacing_wait_id=*/1234,
        channel_ids, {}, &binding, &err);
    if (!expect(bind_rc == PIE_STATUS_OK, err.c_str())) return 1;
    if (!expect(binding.instance_id == 77, "instance id")) return 1;
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
        if (!expect(dispatch.close_channel(channel_ids[i], &err) ==
                        PIE_STATUS_INVALID_ARGUMENT,
                    "live endpoint close rejected")) return 1;
    }

    dispatch.close_instance(binding.instance_id);
    for (std::uint64_t channel_id : channel_ids) {
        if (!expect(dispatch.close_channel(channel_id, &err) == PIE_STATUS_OK,
                    err.c_str())) return 1;
    }
    std::puts("test_ptir_dispatch_bind: OK");
    return 0;
}
