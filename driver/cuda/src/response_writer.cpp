#include "response_writer.hpp"

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace pie_cuda_driver::response {

std::size_t flat_response_size(std::span<const std::uint32_t> per_request_counts) {
    std::uint64_t total = 0;
    for (auto c : per_request_counts) total += c;
    return HEADER_SIZE
         + 4 * per_request_counts.size()
         + 4 * total;
}

std::size_t write_flat_response(
    std::span<std::uint8_t> buf,
    std::span<const std::uint32_t> per_request_counts,
    std::span<const std::uint32_t> tokens)
{
    const auto need = flat_response_size(per_request_counts);
    if (buf.size() < need) {
        throw std::runtime_error("BPIS response buffer too small: have " +
                                 std::to_string(buf.size()) + ", need " +
                                 std::to_string(need));
    }
    const std::uint64_t expected_tokens =
        std::accumulate(per_request_counts.begin(), per_request_counts.end(), 0ULL);
    if (tokens.size() != expected_tokens) {
        throw std::runtime_error("BPIS response: tokens.size() != sum(per_request_counts)");
    }

    std::uint32_t header[4] = {
        MAGIC,
        MODE_FLAT,
        static_cast<std::uint32_t>(per_request_counts.size()),
        static_cast<std::uint32_t>(tokens.size()),
    };
    std::memcpy(buf.data(), header, HEADER_SIZE);

    std::size_t off = HEADER_SIZE;
    std::memcpy(buf.data() + off,
                per_request_counts.data(),
                per_request_counts.size() * sizeof(std::uint32_t));
    off += per_request_counts.size() * sizeof(std::uint32_t);

    std::memcpy(buf.data() + off,
                tokens.data(),
                tokens.size() * sizeof(std::uint32_t));
    off += tokens.size() * sizeof(std::uint32_t);

    return off;
}

}  // namespace pie_cuda_driver::response
