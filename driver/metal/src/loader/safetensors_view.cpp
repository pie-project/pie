// safetensors_view.cpp — mmap + header-parse impl. Pure C++/POSIX + nlohmann/json,
// NO MLX, NO Metal. Safetensors layout: [u64 LE header_len][JSON header][raw data];
// each tensor's bytes = data_base + data_offsets[begin..end].

#include "safetensors_view.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
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
    const fs::path single = dir / "model.safetensors";
    if (fs::exists(single)) return {single};
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
};

SafetensorsView::SafetensorsView(const std::string& hf_path) : impl_(new Impl()) {
    try {
        for (const auto& shard : resolve_shards(fs::path(hf_path))) {
            Mapping m = mmap_file(shard);
            impl_->maps.push_back(m);
            const uint8_t* base = static_cast<const uint8_t*>(m.base);
            if (m.len < 8) throw std::runtime_error("shard too small: " + shard.string());
            uint64_t header_len = 0;
            std::memcpy(&header_len, base, 8);  // u64 LE (assumes LE host — Apple Silicon)
            if (8 + header_len > m.len) throw std::runtime_error("bad header len: " + shard.string());
            const uint8_t* data_base = base + 8 + header_len;
            auto j = nlohmann::json::parse(base + 8, base + 8 + header_len);
            for (auto it = j.begin(); it != j.end(); ++it) {
                if (it.key() == "__metadata__") continue;
                const auto& v = it.value();
                const auto& off = v.at("data_offsets");
                uint64_t b = off[0].get<uint64_t>(), e = off[1].get<uint64_t>();
                Impl::Entry ent;
                ent.data   = data_base + b;
                ent.nbytes = static_cast<size_t>(e - b);
                ent.dtype  = v.at("dtype").get<std::string>();
                for (const auto& d : v.at("shape")) ent.shape.push_back(d.get<int64_t>());
                impl_->tensors.emplace(it.key(), std::move(ent));
            }
        }
        if (impl_->tensors.empty())
            throw std::runtime_error("safetensors under " + hf_path + " contained no tensors");
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
    return impl_->tensors.find(name) != impl_->tensors.end();
}

size_t SafetensorsView::size() const { return impl_->tensors.size(); }

std::vector<std::string> SafetensorsView::names() const {
    std::vector<std::string> out;
    out.reserve(impl_->tensors.size());
    for (const auto& [k, _] : impl_->tensors) out.push_back(k);
    return out;
}

}  // namespace pie::metal
