#include "loader/safetensors.hpp"

#include <cerrno>
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

namespace pie_cuda_driver {

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

    const auto single = snapshot_dir / "model.safetensors";
    const auto index_json = snapshot_dir / "model.safetensors.index.json";

    std::vector<std::filesystem::path> shard_paths;
    if (std::filesystem::exists(single)) {
        shard_paths.push_back(single);
    } else if (std::filesystem::exists(index_json)) {
        std::ifstream in(index_json);
        nlohmann::json j;
        in >> j;
        // weight_map: { tensor_name -> shard_filename }
        std::vector<std::string> uniq;
        for (auto& [_, v] : j.at("weight_map").items()) {
            const auto fn = v.get<std::string>();
            if (std::find(uniq.begin(), uniq.end(), fn) == uniq.end()) uniq.push_back(fn);
        }
        std::sort(uniq.begin(), uniq.end());
        for (auto& fn : uniq) shard_paths.push_back(snapshot_dir / fn);
    } else {
        throw std::runtime_error(
            "no safetensors at " + snapshot_dir.string() +
            " (looked for model.safetensors and model.safetensors.index.json)");
    }

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

DeviceTensor SafetensorsLoader::load_to_device(const std::string& name) {
    const auto& ti = info(name);
    auto& shard = shards_[ti.shard_id];
    if (!shard.data) open_shard_(shard);

    auto t = DeviceTensor::allocate(ti.dtype, ti.shape);

    const auto* host_src = shard.data + shard.data_section_offset + ti.data_offset;
    CUDA_CHECK(cudaMemcpy(t.data(), host_src, ti.nbytes, cudaMemcpyHostToDevice));
    return t;
}

DeviceTensor SafetensorsLoader::load_to_device_sharded(
    const std::string& name, int axis, int rank, int world_size)
{
    if (world_size <= 1 || axis < 0) {
        return load_to_device(name);
    }

    const auto& ti = info(name);
    if (rank < 0 || rank >= world_size) {
        throw std::runtime_error("load_to_device_sharded: rank " +
                                 std::to_string(rank) + " out of range for world " +
                                 std::to_string(world_size));
    }
    if (ti.shape.empty() || ti.shape.size() > 2) {
        throw std::runtime_error("load_to_device_sharded: " + name +
                                 " has unsupported rank " +
                                 std::to_string(ti.shape.size()));
    }
    if (axis >= static_cast<int>(ti.shape.size())) {
        throw std::runtime_error("load_to_device_sharded: axis " +
                                 std::to_string(axis) + " out of range for " + name);
    }
    const std::int64_t orig_dim = ti.shape[axis];
    if (orig_dim % world_size != 0) {
        throw std::runtime_error("load_to_device_sharded: " + name +
                                 " dim " + std::to_string(orig_dim) +
                                 " not divisible by world_size " +
                                 std::to_string(world_size));
    }
    const std::int64_t shard_dim = orig_dim / world_size;

    auto& shard = shards_[ti.shard_id];
    if (!shard.data) open_shard_(shard);
    const auto* host_base = shard.data + shard.data_section_offset + ti.data_offset;
    const std::size_t elem = dtype_bytes(ti.dtype);

    std::vector<std::int64_t> out_shape = ti.shape;
    out_shape[axis] = shard_dim;
    auto t = DeviceTensor::allocate(ti.dtype, out_shape);

    if (axis == 0) {
        // Contiguous slice along the leading dim: rank `r` owns bytes
        // [r*per_rank, (r+1)*per_rank).
        const std::size_t per_rank = ti.nbytes / world_size;
        const auto* host_src = host_base + per_rank * rank;
        CUDA_CHECK(cudaMemcpy(t.data(), host_src, per_rank,
                              cudaMemcpyHostToDevice));
    } else {
        // 2-D row-parallel: keep all rows (d0) but only this rank's column
        // band on the inner dim (d1). Use a 2-D pitched copy: each row is
        // (d1/world_size) elements wide, source pitch is d1 elements.
        const std::int64_t d0 = ti.shape[0];
        const std::int64_t d1 = ti.shape[1];
        const std::size_t row_bytes = static_cast<std::size_t>(shard_dim) * elem;
        const std::size_t src_pitch = static_cast<std::size_t>(d1) * elem;
        const std::size_t dst_pitch = row_bytes;
        const auto* host_src = host_base + (row_bytes * rank);
        CUDA_CHECK(cudaMemcpy2D(
            t.data(), dst_pitch,
            host_src, src_pitch,
            row_bytes, static_cast<std::size_t>(d0),
            cudaMemcpyHostToDevice));
        (void)d1;
    }
    return t;
}

DeviceTensor SafetensorsLoader::load_to_device_row_range_sharded(
    const std::string& name,
    std::int64_t row_offset, std::int64_t rows,
    int rank, int world_size)
{
    const auto& ti = info(name);
    if (ti.shape.size() != 2) {
        throw std::runtime_error("load_to_device_row_range_sharded: " + name +
                                 " must be 2-D, got rank " +
                                 std::to_string(ti.shape.size()));
    }
    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        throw std::runtime_error("load_to_device_row_range_sharded: bad rank/world");
    }
    if (rows <= 0 || rows % world_size != 0) {
        throw std::runtime_error("load_to_device_row_range_sharded: row range " +
                                 std::to_string(rows) +
                                 " not divisible by world_size " +
                                 std::to_string(world_size));
    }
    const std::int64_t total_rows = ti.shape[0];
    if (row_offset < 0 || row_offset + rows > total_rows) {
        throw std::runtime_error(
            "load_to_device_row_range_sharded: range [" +
            std::to_string(row_offset) + ", " +
            std::to_string(row_offset + rows) +
            ") out of bounds for " + name +
            " (rows=" + std::to_string(total_rows) + ")");
    }
    const std::int64_t cols = ti.shape[1];
    const std::int64_t shard_rows = rows / world_size;

    auto& shard = shards_[ti.shard_id];
    if (!shard.data) open_shard_(shard);
    const auto* host_base = shard.data + shard.data_section_offset + ti.data_offset;
    const std::size_t elem = dtype_bytes(ti.dtype);

    // Per-rank start row within the file. Each rank gets a contiguous
    // slab of `shard_rows` rows from the requested range.
    const std::int64_t my_row_start = row_offset + static_cast<std::int64_t>(rank) * shard_rows;
    const std::size_t bytes = static_cast<std::size_t>(shard_rows) *
                              static_cast<std::size_t>(cols) * elem;
    const auto* host_src = host_base + static_cast<std::size_t>(my_row_start) *
                                       static_cast<std::size_t>(cols) * elem;

    auto t = DeviceTensor::allocate(ti.dtype, {shard_rows, cols});
    CUDA_CHECK(cudaMemcpy(t.data(), host_src, bytes, cudaMemcpyHostToDevice));
    return t;
}

DeviceTensor SafetensorsLoader::load_to_device_moe_gate_up_sharded(
    const std::string& name, int rank, int world_size)
{
    const auto& ti = info(name);
    if (ti.shape.size() != 3) {
        throw std::runtime_error("load_to_device_moe_gate_up_sharded: " + name +
                                 " must be 3-D [E, 2*Im, H]");
    }
    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        throw std::runtime_error("load_to_device_moe_gate_up_sharded: bad rank/world");
    }
    const std::int64_t E       = ti.shape[0];
    const std::int64_t two_Im  = ti.shape[1];
    const std::int64_t H       = ti.shape[2];
    if (two_Im % 2 != 0 || (two_Im / 2) % world_size != 0) {
        throw std::runtime_error(
            "load_to_device_moe_gate_up_sharded: 2*Im=" + std::to_string(two_Im) +
            " must be even and Im divisible by world_size=" + std::to_string(world_size));
    }
    const std::int64_t Im       = two_Im / 2;
    const std::int64_t Im_local = Im / world_size;

    auto& shard = shards_[ti.shard_id];
    if (!shard.data) open_shard_(shard);
    const auto* host_base = shard.data + shard.data_section_offset + ti.data_offset;
    const std::size_t elem      = dtype_bytes(ti.dtype);
    const std::size_t row_bytes = static_cast<std::size_t>(H) * elem;
    const std::size_t local_block_rows = static_cast<std::size_t>(Im_local);

    auto t = DeviceTensor::allocate(ti.dtype, {E, 2 * Im_local, H});
    auto* dst = static_cast<std::uint8_t*>(t.data());

    for (std::int64_t e = 0; e < E; ++e) {
        // gate block: file rows [e*two_Im + rank*Im_local, +Im_local)
        const std::size_t src_gate_off = static_cast<std::size_t>(
            e * two_Im + static_cast<std::int64_t>(rank) * Im_local) * row_bytes;
        const std::size_t dst_gate_off = static_cast<std::size_t>(
            e * 2 * Im_local) * row_bytes;
        CUDA_CHECK(cudaMemcpy(
            dst + dst_gate_off, host_base + src_gate_off,
            local_block_rows * row_bytes, cudaMemcpyHostToDevice));
        // up block: file rows [e*two_Im + Im + rank*Im_local, +Im_local)
        const std::size_t src_up_off = static_cast<std::size_t>(
            e * two_Im + Im + static_cast<std::int64_t>(rank) * Im_local) * row_bytes;
        const std::size_t dst_up_off = static_cast<std::size_t>(
            e * 2 * Im_local + Im_local) * row_bytes;
        CUDA_CHECK(cudaMemcpy(
            dst + dst_up_off, host_base + src_up_off,
            local_block_rows * row_bytes, cudaMemcpyHostToDevice));
    }
    return t;
}

DeviceTensor SafetensorsLoader::load_to_device_moe_down_sharded(
    const std::string& name, int rank, int world_size)
{
    const auto& ti = info(name);
    if (ti.shape.size() != 3) {
        throw std::runtime_error("load_to_device_moe_down_sharded: " + name +
                                 " must be 3-D [E, H, Im]");
    }
    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        throw std::runtime_error("load_to_device_moe_down_sharded: bad rank/world");
    }
    const std::int64_t E   = ti.shape[0];
    const std::int64_t H   = ti.shape[1];
    const std::int64_t Im  = ti.shape[2];
    if (Im % world_size != 0) {
        throw std::runtime_error(
            "load_to_device_moe_down_sharded: Im=" + std::to_string(Im) +
            " not divisible by world_size=" + std::to_string(world_size));
    }
    const std::int64_t Im_local = Im / world_size;

    auto& shard = shards_[ti.shard_id];
    if (!shard.data) open_shard_(shard);
    const auto* host_base = shard.data + shard.data_section_offset + ti.data_offset;
    const std::size_t elem = dtype_bytes(ti.dtype);

    auto t = DeviceTensor::allocate(ti.dtype, {E, H, Im_local});
    auto* dst = static_cast<std::uint8_t*>(t.data());

    // Per-expert 2-D pitched copy: H rows × Im_local cols (this rank's
    // band starts at column rank*Im_local), file pitch = Im, dst pitch = Im_local.
    const std::size_t local_pitch = static_cast<std::size_t>(Im_local) * elem;
    const std::size_t file_pitch  = static_cast<std::size_t>(Im) * elem;
    for (std::int64_t e = 0; e < E; ++e) {
        const std::size_t src_off = static_cast<std::size_t>(
            e * H * Im + static_cast<std::int64_t>(rank) * Im_local) * elem;
        const std::size_t dst_off = static_cast<std::size_t>(
            e * H * Im_local) * elem;
        CUDA_CHECK(cudaMemcpy2D(
            dst + dst_off, local_pitch,
            host_base + src_off, file_pitch,
            local_pitch, static_cast<std::size_t>(H),
            cudaMemcpyHostToDevice));
    }
    return t;
}

}  // namespace pie_cuda_driver
