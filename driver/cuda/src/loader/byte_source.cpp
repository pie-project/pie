#include "loader/byte_source.hpp"

#include <cstdint>
#include <stdexcept>

namespace pie_cuda_driver {

void MmapByteSource::write_to_device(
    const ByteRangeWrite& write,
    void* dst_base)
{
    if (dst_base == nullptr) {
        throw std::runtime_error(
            "byte source: null destination for '" + write.output_name + "'");
    }
    auto* dst = static_cast<std::uint8_t*>(dst_base) + write.dst_offset_bytes;
    loader_.copy_strided_to_device(
        write.raw_name,
        write.slices,
        dst,
        write.dst_shape);
}

}  // namespace pie_cuda_driver
