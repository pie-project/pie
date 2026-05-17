#include "loader/safetensors.hpp"

#include <cerrno>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "cuda_check.hpp"
#include <pie_driver_common/safetensors_manifest.hpp>
#include <pie_driver_common/shard_plan.hpp>

namespace pie_cuda_driver {

namespace {

std::vector<std::int64_t> normalize_slices(
    const TensorInfo& ti,
    const std::vector<TensorSlice>& slices,
    const std::string& name,
    std::vector<std::int64_t>& start)
{
    const auto rank = static_cast<int>(ti.shape.size());
    start.assign(ti.shape.size(), 0);
    std::vector<std::int64_t> shape = ti.shape;
    for (const auto& slice : slices) {
        if (slice.axis < 0 || slice.axis >= rank) {
            throw std::runtime_error(
                "safetensors: slice axis out of range for '" + name + "'");
        }
        if (slice.start < 0 || slice.length <= 0 ||
            slice.start + slice.length >
                ti.shape[static_cast<std::size_t>(slice.axis)]) {
            throw std::runtime_error(
                "safetensors: slice range out of bounds for '" + name + "'");
        }
        start[static_cast<std::size_t>(slice.axis)] = slice.start;
        shape[static_cast<std::size_t>(slice.axis)] = slice.length;
    }
    return shape;
}

}  // namespace

// Parse a single shard's JSON header into TensorInfo entries. Mutates the
// caller's `index_` and `total_bytes_`.
void SafetensorsLoader::parse_shard_header_(
    Shard& s,
    std::uint32_t shard_id,
    std::unordered_map<std::string, TensorInfo>& index,
    std::uint64_t& total_bytes)
{
    std::ifstream in(s.path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + s.path.string());

    std::uint64_t header_size = 0;
    in.read(reinterpret_cast<char*>(&header_size), 8);
    if (!in) throw std::runtime_error("safetensors: short header in " + s.path.string());

    std::string header_bytes(static_cast<std::size_t>(header_size), '\0');
    in.read(header_bytes.data(), header_bytes.size());
    if (!in) throw std::runtime_error("safetensors: header read failed in " + s.path.string());

    s.data_section_offset = 8 + header_size;

    auto j = nlohmann::json::parse(header_bytes);
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "__metadata__") continue;
        const auto& v = it.value();

        TensorInfo ti;
        ti.dtype = dtype_from_safetensors(v.at("dtype").get<std::string>());
        ti.shape = v.at("shape").get<std::vector<std::int64_t>>();
        const auto& off = v.at("data_offsets");
        if (!off.is_array() || off.size() != 2) {
            throw std::runtime_error("safetensors: bad data_offsets for " + it.key());
        }
        ti.data_offset = off[0].get<std::uint64_t>();
        ti.nbytes      = off[1].get<std::uint64_t>() - ti.data_offset;
        ti.shard_id    = shard_id;

        // Sanity: nbytes vs shape × dtype.
        std::uint64_t expected = dtype_bytes(ti.dtype);
        for (auto d : ti.shape) expected *= static_cast<std::uint64_t>(d);
        if (expected != ti.nbytes) {
            throw std::runtime_error(
                "safetensors: nbytes mismatch for '" + it.key() + "': declared "
                + std::to_string(ti.nbytes) + ", expected " + std::to_string(expected));
        }

        total_bytes += ti.nbytes;
        index.emplace(it.key(), std::move(ti));
    }
}

SafetensorsLoader SafetensorsLoader::open(const std::filesystem::path& snapshot_dir) {
    SafetensorsLoader loader;
    const auto manifest =
        pie_driver_common::discover_safetensors_manifest(
            snapshot_dir,
            pie_driver_common::SafetensorsLayoutPreference::SingleFile);
    const auto& shard_paths = manifest.shard_paths;

    loader.shards_.reserve(shard_paths.size());
    for (auto& p : shard_paths) {
        Shard s;
        s.path = p;
        loader.shards_.push_back(std::move(s));
    }

    for (std::uint32_t i = 0; i < loader.shards_.size(); ++i) {
        parse_shard_header_(loader.shards_[i], i, loader.index_, loader.total_bytes_);
    }
    return loader;
}

SafetensorsLoader::~SafetensorsLoader() {
    for (auto& s : shards_) {
        if (s.data && s.mapped_size) {
            munmap(const_cast<std::uint8_t*>(s.data), s.mapped_size);
        }
        if (s.fd >= 0) {
            ::close(s.fd);
        }
    }
}

void SafetensorsLoader::open_shard_(Shard& s) const {
    if (s.data) return;
    auto& m = const_cast<Shard&>(s);
    m.fd = ::open(s.path.c_str(), O_RDONLY);
    if (m.fd < 0) {
        throw std::runtime_error("open(" + s.path.string() + ") failed: " + std::strerror(errno));
    }
    struct stat st{};
    if (::fstat(m.fd, &st) != 0) {
        ::close(m.fd);
        m.fd = -1;
        throw std::runtime_error("fstat(" + s.path.string() + ") failed");
    }
    m.mapped_size = static_cast<std::size_t>(st.st_size);
    void* p = ::mmap(nullptr, m.mapped_size, PROT_READ, MAP_SHARED, m.fd, 0);
    if (p == MAP_FAILED) {
        ::close(m.fd);
        m.fd = -1;
        throw std::runtime_error(std::string("mmap failed: ") + std::strerror(errno));
    }
    m.data = static_cast<const std::uint8_t*>(p);
    // Hint the kernel: we'll stream sequentially through this mmap once.
    ::madvise(p, m.mapped_size, MADV_SEQUENTIAL);
}

std::vector<std::string> SafetensorsLoader::tensor_names() const {
    std::vector<std::string> out;
    out.reserve(index_.size());
    for (const auto& [k, _] : index_) out.push_back(k);
    std::sort(out.begin(), out.end());
    return out;
}

const TensorInfo& SafetensorsLoader::info(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("tensor not found in safetensors: " + name);
    }
    return it->second;
}

void SafetensorsLoader::copy_to_device(
    const std::string& name,
    void* dst,
    const std::vector<std::int64_t>& dst_shape)
{
    const auto& ti = info(name);
    if (dst == nullptr) {
        throw std::runtime_error(
            "safetensors: null destination for tensor '" + name + "'");
    }
    if (dst_shape != ti.shape) {
        throw std::runtime_error(
            "safetensors: destination shape mismatch for tensor '" + name + "'");
    }
    auto& shard = shards_[ti.shard_id];
    if (!shard.data) open_shard_(shard);

    const auto* host_src = shard.data + shard.data_section_offset + ti.data_offset;
    CUDA_CHECK(cudaMemcpy(dst, host_src, ti.nbytes, cudaMemcpyHostToDevice));
}

void SafetensorsLoader::copy_shard_to_device(
    const std::string& name,
    int axis,
    int rank,
    int world_size,
    void* dst,
    const std::vector<std::int64_t>& dst_shape)
{
    if (world_size <= 1 || axis < 0) {
        copy_to_device(name, dst, dst_shape);
        return;
    }

    const auto& ti = info(name);
    const auto plan = pie_driver_common::plan_axis_shard(
        ti.shape, axis, rank, world_size, "copy_shard_to_device: " + name);
    if (plan.output_shape != dst_shape) {
        throw std::runtime_error(
            "safetensors: destination shape mismatch for sharded tensor '" +
            name + "'");
    }
    const std::int64_t shard_dim = plan.shard_dim;

    auto& shard = shards_[ti.shard_id];
    if (!shard.data) open_shard_(shard);
    const auto* host_base = shard.data + shard.data_section_offset + ti.data_offset;
    const std::size_t elem = dtype_bytes(ti.dtype);

    if (axis == 0) {
        // Contiguous slice along the leading dim: rank `r` owns bytes
        // [r*per_rank, (r+1)*per_rank).
        std::int64_t inner = 1;
        for (std::size_t i = 1; i < ti.shape.size(); ++i) {
            inner *= ti.shape[i];
        }
        const std::size_t bytes =
            static_cast<std::size_t>(shard_dim) *
            static_cast<std::size_t>(inner) * elem;
        const auto* host_src =
            host_base + static_cast<std::size_t>(plan.offset) *
                            static_cast<std::size_t>(inner) * elem;
        CUDA_CHECK(cudaMemcpy(dst, host_src, bytes, cudaMemcpyHostToDevice));
    } else {
        copy_strided_to_device(
            name,
            {TensorSlice{axis, plan.offset, shard_dim}},
            dst,
            dst_shape);
    }
}

void SafetensorsLoader::copy_slice_to_device(
    const std::string& name,
    int axis,
    std::int64_t start,
    std::int64_t length,
    void* dst,
    const std::vector<std::int64_t>& dst_shape)
{
    copy_strided_to_device(
        name,
        {TensorSlice{axis, start, length}},
        dst,
        dst_shape);
}

void SafetensorsLoader::copy_strided_to_device(
    const std::string& name,
    const std::vector<TensorSlice>& slices,
    void* dst,
    const std::vector<std::int64_t>& dst_shape)
{
    const auto& ti = info(name);
    if (dst == nullptr) {
        throw std::runtime_error(
            "safetensors: null destination for sliced tensor '" + name + "'");
    }
    if (slices.empty()) {
        copy_to_device(name, dst, dst_shape);
        return;
    }

    std::vector<std::int64_t> start;
    const auto shape = normalize_slices(ti, slices, name, start);
    if (shape != dst_shape) {
        throw std::runtime_error(
            "safetensors: destination shape mismatch for sliced tensor '" +
            name + "'");
    }

    auto& shard = shards_[ti.shard_id];
    if (!shard.data) open_shard_(shard);
    const auto* host_base = shard.data + shard.data_section_offset + ti.data_offset;
    const std::size_t elem = dtype_bytes(ti.dtype);
    const auto rank = static_cast<int>(ti.shape.size());

    std::vector<std::int64_t> source_strides(ti.shape.size(), 1);
    for (int axis = rank - 2; axis >= 0; --axis) {
        source_strides[static_cast<std::size_t>(axis)] =
            source_strides[static_cast<std::size_t>(axis + 1)] *
            ti.shape[static_cast<std::size_t>(axis + 1)];
    }

    auto* dst_base = static_cast<std::uint8_t*>(dst);
    const std::int64_t inner = shape.empty() ? 1 : shape.back();
    const std::size_t contiguous_bytes =
        static_cast<std::size_t>(inner) * elem;

    if (rank == 0) {
        CUDA_CHECK(cudaMemcpy(
            dst_base, host_base, elem, cudaMemcpyHostToDevice));
        return;
    }

    std::vector<std::int64_t> index(ti.shape.size(), 0);
    std::size_t dst_offset = 0;
    const int outer_rank = rank - 1;
    bool done = false;
    while (!done) {
        std::int64_t source_linear =
            start.back() * source_strides.back();
        for (int axis = 0; axis < outer_rank; ++axis) {
            source_linear +=
                (start[static_cast<std::size_t>(axis)] +
                 index[static_cast<std::size_t>(axis)]) *
                source_strides[static_cast<std::size_t>(axis)];
        }
        CUDA_CHECK(cudaMemcpy(
            dst_base + dst_offset,
            host_base + static_cast<std::size_t>(source_linear) * elem,
            contiguous_bytes,
            cudaMemcpyHostToDevice));
        dst_offset += contiguous_bytes;

        for (int axis = outer_rank - 1; axis >= 0; --axis) {
            auto& v = index[static_cast<std::size_t>(axis)];
            ++v;
            if (v < shape[static_cast<std::size_t>(axis)]) {
                break;
            }
            v = 0;
            if (axis == 0) done = true;
        }
        if (outer_rank == 0) done = true;
    }
}

}  // namespace pie_cuda_driver
