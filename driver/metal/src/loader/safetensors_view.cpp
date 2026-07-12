// safetensors_view.cpp — mmap + header-parse impl. Pure C++/POSIX + nlohmann/json,
// NO MLX, NO Metal. Safetensors layout: [u64 LE header_len][JSON header][raw data];
// each tensor's bytes = data_base + data_offsets[begin..end].

#include "safetensors_view.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace pie::metal {

namespace fs = std::filesystem;

namespace {

std::vector<fs::path> resolve_shards(const fs::path& dir) {
    const fs::path single = dir / "model.safetensors";
    if (fs::exists(single)) return {single};
    const fs::path index = dir / "model.safetensors.index.json";
    if (fs::exists(index)) {
        std::ifstream in(index);
        if (!in) throw std::runtime_error("cannot open " + index.string());
        nlohmann::json j;
        in >> j;
        std::set<std::string> files;
        if (j.contains("weight_map") && j["weight_map"].is_object()) {
            for (const auto& [_, v] : j["weight_map"].items())
                files.insert(v.get<std::string>());
        }
        std::vector<fs::path> shards;
        for (const auto& f : files) shards.push_back(dir / f);
        if (!shards.empty()) return shards;
    }
    std::vector<fs::path> shards;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".safetensors") shards.push_back(e.path());
    if (shards.empty())
        throw std::runtime_error("no .safetensors weights found under " + dir.string());
    return shards;
}

struct Mapping {
    void*  base = MAP_FAILED;
    size_t len  = 0;
};

Mapping mmap_file(const fs::path& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("open failed: " + path.string());
    struct stat st {};
    if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("fstat: " + path.string()); }
    size_t len = static_cast<size_t>(st.st_size);
    void* base = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);  // mapping survives close()
    if (base == MAP_FAILED) throw std::runtime_error("mmap failed: " + path.string());
    return {base, len};
}

}  // namespace

struct SafetensorsView::Impl {
    std::vector<Mapping> maps;
    struct Entry { const uint8_t* data; size_t nbytes; std::string dtype; std::vector<int64_t> shape; };
    std::unordered_map<std::string, Entry> tensors;
    bool indexed = false;
};

void SafetensorsView::ensure_index() const {
    if (impl_->indexed) return;
    for (const Mapping& mapping : impl_->maps) {
        const auto* base = static_cast<const std::uint8_t*>(mapping.base);
        if (mapping.len < 8) throw std::runtime_error("safetensors shard is too small");
        std::uint64_t header_len = 0;
        std::memcpy(&header_len, base, 8);
        if (8 + header_len > mapping.len) {
            throw std::runtime_error("safetensors header length is invalid");
        }
        const std::uint8_t* data_base = base + 8 + header_len;
        const auto json = nlohmann::json::parse(base + 8, base + 8 + header_len);
        for (auto it = json.begin(); it != json.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            const auto& value = it.value();
            const auto& offsets = value.at("data_offsets");
            const std::uint64_t begin = offsets[0].get<std::uint64_t>();
            const std::uint64_t end = offsets[1].get<std::uint64_t>();
            SafetensorsView::Impl::Entry entry;
            entry.data = data_base + begin;
            entry.nbytes = static_cast<std::size_t>(end - begin);
            entry.dtype = value.at("dtype").get<std::string>();
            for (const auto& dim : value.at("shape")) {
                entry.shape.push_back(dim.get<std::int64_t>());
            }
            impl_->tensors.emplace(it.key(), std::move(entry));
        }
    }
    impl_->indexed = true;
}

SafetensorsView::SafetensorsView(const std::string& hf_path) : impl_(new Impl()) {
    try {
        for (const auto& shard : resolve_shards(fs::path(hf_path))) {
            Mapping m = mmap_file(shard);
            impl_->maps.push_back(m);
        }
        if (impl_->maps.empty()) {
            throw std::runtime_error(
                "safetensors under " + hf_path + " contained no files");
        }
    } catch (...) {
        for (auto& m : impl_->maps) if (m.base != MAP_FAILED) ::munmap(m.base, m.len);
        delete impl_;
        throw;
    }
}

SafetensorsView::~SafetensorsView() {
    for (auto& m : impl_->maps) if (m.base != MAP_FAILED) ::munmap(m.base, m.len);
    delete impl_;
}

std::optional<RawTensor> SafetensorsView::try_get(const std::string& name) const {
    ensure_index();
    auto it = impl_->tensors.find(name);
    if (it == impl_->tensors.end()) return std::nullopt;
    RawTensor rt;
    rt.data = it->second.data;
    rt.nbytes = it->second.nbytes;
    rt.dtype = it->second.dtype;
    rt.shape = it->second.shape;
    return rt;
}

RawTensor SafetensorsView::get(const std::string& name) const {
    auto rt = try_get(name);
    if (!rt) throw std::runtime_error("weight not found in safetensors: " + name);
    return *rt;
}

bool SafetensorsView::has(const std::string& name) const {
    ensure_index();
    return impl_->tensors.find(name) != impl_->tensors.end();
}

size_t SafetensorsView::size() const {
    ensure_index();
    return impl_->tensors.size();
}

std::vector<std::string> SafetensorsView::names() const {
    ensure_index();
    std::vector<std::string> out;
    out.reserve(impl_->tensors.size());
    for (const auto& [k, _] : impl_->tensors) out.push_back(k);
    return out;
}

void SafetensorsView::copy_storage_bytes(
    std::uint32_t file_id,
    std::uint64_t file_offset,
    std::uint64_t bytes,
    void* destination,
    std::uint64_t max_tile_bytes) const {
    if (file_id >= impl_->maps.size()) {
        throw std::runtime_error("storage program file id is out of range");
    }
    const Mapping& mapping = impl_->maps[file_id];
    if (file_offset > mapping.len || bytes > mapping.len - file_offset) {
        throw std::runtime_error("storage program file extent is out of range");
    }
    if (bytes != 0 && destination == nullptr) {
        throw std::runtime_error("storage program destination is null");
    }
    const std::uint64_t tile =
        max_tile_bytes == 0 ? std::max<std::uint64_t>(1, bytes)
                            : std::max<std::uint64_t>(1, max_tile_bytes);
    const auto* source =
        static_cast<const std::uint8_t*>(mapping.base) + file_offset;
    auto* dest = static_cast<std::uint8_t*>(destination);
    for (std::uint64_t copied = 0; copied < bytes;) {
        const std::uint64_t chunk = std::min(tile, bytes - copied);
        std::memcpy(dest + copied, source + copied, static_cast<std::size_t>(chunk));
        copied += chunk;
    }
    if (bytes != 0) {
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size > 0) {
            const std::uintptr_t start =
                reinterpret_cast<std::uintptr_t>(source) &
                ~static_cast<std::uintptr_t>(page_size - 1);
            const std::uintptr_t end =
                (reinterpret_cast<std::uintptr_t>(source + bytes) +
                 static_cast<std::uintptr_t>(page_size - 1)) &
                ~static_cast<std::uintptr_t>(page_size - 1);
            (void)::madvise(
                reinterpret_cast<void*>(start),
                static_cast<std::size_t>(end - start),
                MADV_DONTNEED);
        }
    }
}

}  // namespace pie::metal
