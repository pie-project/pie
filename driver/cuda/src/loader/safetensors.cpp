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

}  // namespace pie_cuda_driver
