#include "loader/gguf_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pie_cuda_driver {

namespace {

enum class GgufValueType : std::uint32_t {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
};

enum class GgmlType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    I8 = 24,
    I32 = 26,
    I64 = 27,
    BF16 = 30,
};

struct GgufTensorType {
    DType dtype = DType::BF16;
    std::uint64_t bytes_per_element = 0;
    std::uint32_t block_elements = 0;
    std::uint32_t block_bytes = 0;
    const char* encoding = "dense";
};

class Cursor {
public:
    explicit Cursor(const std::vector<std::uint8_t>& data) noexcept
        : data_(data) {}

    std::uint64_t position() const noexcept { return pos_; }

    void require(std::uint64_t bytes, const char* context) const
    {
        if (bytes > data_.size() || pos_ > data_.size() - bytes) {
            throw std::runtime_error(
                "gguf source: truncated file while reading " +
                std::string(context));
        }
    }

    std::uint8_t read_u8(const char* context)
    {
        require(1, context);
        return data_[static_cast<std::size_t>(pos_++)];
    }

    std::uint32_t read_u32(const char* context)
    {
        require(4, context);
        const auto* p = data_.data() + pos_;
        pos_ += 4;
        return static_cast<std::uint32_t>(p[0]) |
               (static_cast<std::uint32_t>(p[1]) << 8) |
               (static_cast<std::uint32_t>(p[2]) << 16) |
               (static_cast<std::uint32_t>(p[3]) << 24);
    }

    std::uint64_t read_u64(const char* context)
    {
        require(8, context);
        const auto* p = data_.data() + pos_;
        pos_ += 8;
        std::uint64_t out = 0;
        for (int i = 0; i < 8; ++i) {
            out |= static_cast<std::uint64_t>(p[i]) << (8 * i);
        }
        return out;
    }

    std::string read_string(const char* context)
    {
        const std::uint64_t len = read_u64(context);
        if (len > static_cast<std::uint64_t>(
                      std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(
                "gguf source: string is too large while reading " +
                std::string(context));
        }
        require(len, context);
        const auto begin = data_.data() + pos_;
        pos_ += len;
        return std::string(
            reinterpret_cast<const char*>(begin),
            static_cast<std::size_t>(len));
    }

    void skip(std::uint64_t bytes, const char* context)
    {
        require(bytes, context);
        pos_ += bytes;
    }

private:
    const std::vector<std::uint8_t>& data_;
    std::uint64_t pos_ = 0;
};

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment)
{
    if (alignment == 0) {
        throw std::runtime_error("gguf source: alignment must be non-zero");
    }
    const std::uint64_t rem = value % alignment;
    return rem == 0 ? value : value + (alignment - rem);
}

std::uint64_t checked_mul(
    std::uint64_t a,
    std::uint64_t b,
    const std::string& tensor_name)
{
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::runtime_error(
            "gguf source: tensor byte size overflows for '" +
            tensor_name + "'");
    }
    return a * b;
}

std::uint64_t numel_for_shape(
    const std::vector<std::int64_t>& shape,
    const std::string& tensor_name)
{
    std::uint64_t out = 1;
    for (const auto dim : shape) {
        if (dim < 0) {
            throw std::runtime_error(
                "gguf source: negative dimension for '" + tensor_name + "'");
        }
        out = checked_mul(out, static_cast<std::uint64_t>(dim), tensor_name);
    }
    return out;
}

GgufTensorType map_tensor_type(
    std::uint32_t raw_type,
    const std::string& tensor_name)
{
    switch (static_cast<GgmlType>(raw_type)) {
    case GgmlType::F32:
        return GgufTensorType{DType::FP32, 4};
    case GgmlType::F16:
        return GgufTensorType{DType::FP16, 2};
    case GgmlType::Q4_0:
        return GgufTensorType{
            .dtype = DType::UINT8,
            .bytes_per_element = 0,
            .block_elements = 32,
            .block_bytes = 18,
            .encoding = "gguf.q4_0",
        };
    case GgmlType::I8:
        return GgufTensorType{DType::INT8, 1};
    case GgmlType::I32:
        return GgufTensorType{DType::INT32, 4};
    case GgmlType::I64:
        return GgufTensorType{DType::INT64, 8};
    case GgmlType::BF16:
        return GgufTensorType{DType::BF16, 2};
    default:
        throw std::runtime_error(
            "gguf source: tensor '" + tensor_name +
            "' uses unsupported GGUF/GGML type id " +
            std::to_string(raw_type) +
            ". Add a GGUF quant dialect adapter before loading this type.");
    }
}

float half_to_float(std::uint16_t half) noexcept
{
    const std::uint32_t sign = (half >> 15) & 0x1u;
    const std::uint32_t exp = (half >> 10) & 0x1fu;
    const std::uint32_t frac = half & 0x3ffu;
    if (exp == 0) {
        if (frac == 0) return sign ? -0.0f : 0.0f;
        float value = static_cast<float>(frac) / 1024.0f;
        value = std::ldexp(value, -14);
        return sign ? -value : value;
    }
    if (exp == 31) {
        return frac == 0
            ? (sign ? -std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::infinity())
            : std::numeric_limits<float>::quiet_NaN();
    }
    float value = 1.0f + static_cast<float>(frac) / 1024.0f;
    value = std::ldexp(value, static_cast<int>(exp) - 15);
    return sign ? -value : value;
}

void skip_value(Cursor& cursor, GgufValueType type);

void skip_array(Cursor& cursor)
{
    const auto item_type =
        static_cast<GgufValueType>(cursor.read_u32("metadata array type"));
    const std::uint64_t count = cursor.read_u64("metadata array length");
    for (std::uint64_t i = 0; i < count; ++i) {
        skip_value(cursor, item_type);
    }
}

void skip_value(Cursor& cursor, GgufValueType type)
{
    switch (type) {
    case GgufValueType::Uint8:
    case GgufValueType::Int8:
    case GgufValueType::Bool:
        cursor.skip(1, "metadata scalar");
        return;
    case GgufValueType::Uint16:
    case GgufValueType::Int16:
        cursor.skip(2, "metadata scalar");
        return;
    case GgufValueType::Uint32:
    case GgufValueType::Int32:
    case GgufValueType::Float32:
        cursor.skip(4, "metadata scalar");
        return;
    case GgufValueType::Uint64:
    case GgufValueType::Int64:
    case GgufValueType::Float64:
        cursor.skip(8, "metadata scalar");
        return;
    case GgufValueType::String:
        (void)cursor.read_string("metadata string");
        return;
    case GgufValueType::Array:
        skip_array(cursor);
        return;
    }
    throw std::runtime_error("gguf source: unknown metadata value type");
}

std::uint64_t read_alignment_metadata(
    Cursor& cursor,
    GgufValueType type)
{
    switch (type) {
    case GgufValueType::Uint32:
        return cursor.read_u32("general.alignment");
    case GgufValueType::Uint64:
        return cursor.read_u64("general.alignment");
    default:
        throw std::runtime_error(
            "gguf source: general.alignment must be uint32 or uint64");
    }
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error(
            "gguf source: failed to open '" + path.string() + "'");
    }
    const auto size = in.tellg();
    if (size < 0) {
        throw std::runtime_error(
            "gguf source: failed to stat '" + path.string() + "'");
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
    }
    if (!in) {
        throw std::runtime_error(
            "gguf source: failed to read '" + path.string() + "'");
    }
    return data;
}

}  // namespace

GgufCheckpointSource GgufCheckpointSource::open(
    const std::filesystem::path& path)
{
    const auto data = read_file(path);
    Cursor cursor(data);
    cursor.require(4, "magic");
    if (std::memcmp(data.data(), "GGUF", 4) != 0) {
        throw std::runtime_error(
            "gguf source: '" + path.string() + "' is not a GGUF file");
    }
    cursor.skip(4, "magic");

    GgufCheckpointSource source;
    source.path_ = path;
    source.version_ = cursor.read_u32("version");
    if (source.version_ != 2 && source.version_ != 3) {
        throw std::runtime_error(
            "gguf source: unsupported GGUF version " +
            std::to_string(source.version_));
    }

    const std::uint64_t tensor_count = cursor.read_u64("tensor count");
    const std::uint64_t metadata_count = cursor.read_u64("metadata count");
    if (tensor_count > static_cast<std::uint64_t>(
                           std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("gguf source: tensor count is too large");
    }

    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        const std::string key = cursor.read_string("metadata key");
        const auto type =
            static_cast<GgufValueType>(cursor.read_u32("metadata type"));
        if (key == "general.alignment") {
            source.alignment_ = read_alignment_metadata(cursor, type);
        } else {
            skip_value(cursor, type);
        }
    }

    struct PendingTensor {
        std::string name;
        TensorInfo info;
        std::uint64_t relative_offset = 0;
    };
    std::vector<PendingTensor> pending;
    pending.reserve(static_cast<std::size_t>(tensor_count));

    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        PendingTensor tensor;
        tensor.name = cursor.read_string("tensor name");
        const std::uint32_t dim_count = cursor.read_u32("tensor dimension count");
        if (dim_count > 16) {
            throw std::runtime_error(
                "gguf source: tensor '" + tensor.name +
                "' has unreasonable rank " + std::to_string(dim_count));
        }
        tensor.info.shape.reserve(dim_count);
        for (std::uint32_t d = 0; d < dim_count; ++d) {
            const std::uint64_t dim = cursor.read_u64("tensor dimension");
            if (dim > static_cast<std::uint64_t>(
                          std::numeric_limits<std::int64_t>::max())) {
                throw std::runtime_error(
                    "gguf source: dimension is too large for '" +
                    tensor.name + "'");
            }
            tensor.info.shape.push_back(static_cast<std::int64_t>(dim));
        }
        const std::uint32_t raw_type = cursor.read_u32("tensor type");
        const GgufTensorType type = map_tensor_type(raw_type, tensor.name);
        tensor.relative_offset = cursor.read_u64("tensor data offset");
        tensor.info.dtype = type.dtype;
        tensor.info.encoding = type.encoding;
        tensor.info.block_elements = type.block_elements;
        tensor.info.block_bytes = type.block_bytes;
        tensor.info.data_offset = tensor.relative_offset;
        tensor.info.shard_id = 0;
        const std::uint64_t logical_elements =
            numel_for_shape(tensor.info.shape, tensor.name);
        if (type.block_elements != 0) {
            if (logical_elements % type.block_elements != 0) {
                throw std::runtime_error(
                    "gguf source: quantized tensor '" + tensor.name +
                    "' element count is not divisible by block size " +
                    std::to_string(type.block_elements));
            }
            tensor.info.nbytes = checked_mul(
                logical_elements / type.block_elements,
                type.block_bytes,
                tensor.name);
        } else {
            tensor.info.nbytes = checked_mul(
                logical_elements,
                type.bytes_per_element,
                tensor.name);
        }
        pending.push_back(std::move(tensor));
    }

    const std::uint64_t data_base = align_up(cursor.position(), source.alignment_);
    if (data_base > data.size()) {
        throw std::runtime_error("gguf source: tensor data section is missing");
    }

    for (auto& tensor : pending) {
        const std::uint64_t absolute_offset =
            data_base + tensor.relative_offset;
        if (absolute_offset > data.size() ||
            tensor.info.nbytes > data.size() - absolute_offset) {
            throw std::runtime_error(
                "gguf source: tensor '" + tensor.name +
                "' points outside the file");
        }
        source.names_.push_back(tensor.name);
        source.storage_.emplace(
            tensor.name,
            TensorStorageInfo{
                .path = path,
                .file_offset = absolute_offset,
                .nbytes = tensor.info.nbytes,
                .shard_id = 0,
            });
        auto [_, inserted] = source.tensors_.emplace(
            std::move(tensor.name), std::move(tensor.info));
        if (!inserted) {
            throw std::runtime_error(
                "gguf source: duplicate tensor name in '" + path.string() + "'");
        }
    }
    std::sort(source.names_.begin(), source.names_.end());
    return source;
}

const TensorInfo& GgufCheckpointSource::info(const std::string& name) const
{
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("gguf source: missing tensor '" + name + "'");
    }
    return it->second;
}

TensorStorageInfo GgufCheckpointSource::storage_info(
    const std::string& name) const
{
    const auto it = storage_.find(name);
    if (it == storage_.end()) {
        throw std::runtime_error("gguf source: missing tensor '" + name + "'");
    }
    return it->second;
}

std::vector<float> decode_gguf_q4_0_block(
    const std::uint8_t* block,
    std::size_t bytes)
{
    if (block == nullptr || bytes != 18) {
        throw std::runtime_error(
            "gguf source: Q4_0 block decode expects exactly 18 bytes");
    }
    const std::uint16_t scale_half =
        static_cast<std::uint16_t>(block[0]) |
        (static_cast<std::uint16_t>(block[1]) << 8);
    const float scale = half_to_float(scale_half);
    std::vector<float> values(32);
    for (std::size_t i = 0; i < 16; ++i) {
        const std::uint8_t packed = block[2 + i];
        const int lo = static_cast<int>(packed & 0x0fu) - 8;
        const int hi = static_cast<int>((packed >> 4) & 0x0fu) - 8;
        values[i] = scale * static_cast<float>(lo);
        values[i + 16] = scale * static_cast<float>(hi);
    }
    return values;
}

}  // namespace pie_cuda_driver
