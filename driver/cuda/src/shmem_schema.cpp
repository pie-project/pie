#include "shmem_schema.hpp"

#include <cstring>
#include <stdexcept>

namespace pie_cuda_driver::schema {

DecodedRequest decode_request(std::span<const std::uint8_t> buf) {
    if (buf.size() < HEADER_SIZE) {
        throw std::runtime_error("shmem request smaller than header");
    }

    std::uint32_t magic{}, version{}, device_id{}, flags{}, num_arrays{};
    std::memcpy(&magic,       buf.data() + 0,  4);
    std::memcpy(&version,     buf.data() + 4,  4);
    std::memcpy(&device_id,   buf.data() + 8,  4);
    std::memcpy(&flags,       buf.data() + 12, 4);
    std::memcpy(&num_arrays,  buf.data() + 16, 4);

    if (magic != MAGIC) {
        throw std::runtime_error("shmem schema magic mismatch");
    }
    if (version != SCHEMA_VERSION) {
        throw std::runtime_error("shmem schema version mismatch: expected " +
                                 std::to_string(SCHEMA_VERSION) + ", got " +
                                 std::to_string(version));
    }
    if (num_arrays != NUM_ARRAYS) {
        throw std::runtime_error("shmem schema array count mismatch: expected " +
                                 std::to_string(NUM_ARRAYS) + ", got " +
                                 std::to_string(num_arrays));
    }

    DecodedRequest out;
    out.device_id = device_id;
    out.single_token_mode = (flags & 0x1u) != 0;

    for (std::size_t i = 0; i < NUM_ARRAYS; ++i) {
        std::uint32_t off = 0, len = 0;
        std::memcpy(&off, buf.data() + FIXED_HEADER + i * 8 + 0, 4);
        std::memcpy(&len, buf.data() + FIXED_HEADER + i * 8 + 4, 4);

        const std::size_t nbytes = static_cast<std::size_t>(len) * ELEM_SIZE[i];
        if (off + nbytes > buf.size()) {
            throw std::runtime_error("shmem array " + std::to_string(i) +
                                     " out of bounds");
        }
        out.arrays[i] = std::span<const std::uint8_t>(buf.data() + off, nbytes);
    }

    return out;
}

}  // namespace pie_cuda_driver::schema
