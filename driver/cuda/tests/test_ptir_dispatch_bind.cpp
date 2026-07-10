#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "ptir/ptir_dispatch.hpp"
#include "sampling_ir/frame_carrier.hpp"

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
    std::vector<PieChannelWait> waits(2);
    for (std::size_t i = 0; i < channel_ids.size(); ++i) {
        channel_ids[i] = 1000 + i;
        waits[i].reader_wait_id = 2000 + i;
        waits[i].writer_wait_id = 3000 + i;
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
                    waits,
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
                    waits,
                    {},
                    &binding,
                    &err) == PIE_STATUS_INVALID_ARGUMENT,
                "reject duplicate channel ids")) return 1;
    const int bind_rc = dispatch.bind_instance(
        /*instance_id=*/77, program_hash, /*pacing_wait_id=*/1234,
        channel_ids, waits, {}, &binding, &err);
    if (!expect(bind_rc == PIE_STATUS_OK, err.c_str())) return 1;
    if (!expect(binding.instance_id == 77, "instance id")) return 1;
    if (!expect(binding.channels.ptr != nullptr, "channels ptr")) return 1;
    if (!expect(binding.channels.len == binding.channel_count, "channel slice len")) return 1;
    if (!expect(binding.word_count ==
                    pie_cuda_driver::sampling_ir::WordLayout::words(binding.channel_count),
                "word count")) return 1;
    if (!expect(binding.word_bytes ==
                    static_cast<std::uint64_t>(binding.word_count) * sizeof(std::uint64_t),
                "word bytes")) return 1;

    std::uint64_t prev_mirror_end = 0;
    for (std::size_t i = 0; i < binding.channels.len; ++i) {
        const PieChannelBinding& channel = binding.channels.ptr[i];
        bool found_channel_id = false;
        for (std::uint64_t candidate : channel_ids) {
            if (candidate == channel.channel_id) {
                found_channel_id = true;
                break;
            }
        }
        if (!expect(found_channel_id, "stable channel id")) return 1;
        if (!expect(channel.head_word_index ==
                        pie_cuda_driver::sampling_ir::WordLayout::head(i),
                    "head index")) return 1;
        if (!expect(channel.tail_word_index ==
                        pie_cuda_driver::sampling_ir::WordLayout::tail(i),
                    "tail index")) return 1;
        if (!expect(channel.poison_word_index ==
                        pie_cuda_driver::sampling_ir::WordLayout::poison(i),
                    "poison index")) return 1;
        if (!expect(channel.mirror_offset >= prev_mirror_end, "mirror offsets monotonic"))
            return 1;
        prev_mirror_end =
            channel.mirror_offset +
            static_cast<std::uint64_t>(channel.cell_bytes) * (channel.capacity + 1ull);
    }
    if (!expect(prev_mirror_end <= binding.mirror_bytes, "mirror bytes cover bindings")) return 1;

    dispatch.close_instance(binding.instance_id);
    std::puts("test_ptir_dispatch_bind: OK");
    return 0;
}
