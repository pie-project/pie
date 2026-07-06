#pragma once
//
// Minimal safetensors (BF16) reader + Qwen3-0.6B weight loader, shared by the
// real-weight forward (qwen3_forward) and the autoregressive generator
// (qwen3_generate). mmap + flat-header lookup; bf16 -> f32 via `bits << 16`.
// HF layout is [out, in] = matmul_xwt's W layout (no transpose).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "reference.hpp"

namespace qwen3 {

struct SafeTensors {
    int fd = -1;
    const std::uint8_t* base = nullptr;
    std::size_t file_size = 0;
    std::string header;
    std::size_t data_base = 0;

    bool open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0) return false;
        file_size = st.st_size;
        base = static_cast<const std::uint8_t*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (base == MAP_FAILED) { base = nullptr; return false; }
        std::uint64_t hlen;
        std::memcpy(&hlen, base, 8);
        header.assign(reinterpret_cast<const char*>(base + 8), hlen);
        data_base = 8 + hlen;
        return true;
    }
    ~SafeTensors() {
        if (base) munmap(const_cast<std::uint8_t*>(base), file_size);
        if (fd >= 0) ::close(fd);
    }
    bool offsets(const std::string& name, std::size_t& start, std::size_t& end) const {
        std::string key = "\"" + name + "\":";
        std::size_t k = header.find(key);
        if (k == std::string::npos) return false;
        std::size_t o = header.find("\"data_offsets\":[", k);
        if (o == std::string::npos) return false;
        o += std::strlen("\"data_offsets\":[");
        start = std::strtoull(header.c_str() + o, nullptr, 10);
        std::size_t comma = header.find(',', o);
        end = std::strtoull(header.c_str() + comma + 1, nullptr, 10);
        return true;
    }
    bool load_f32(const std::string& name, std::vector<float>& out) const {
        std::size_t start, end;
        if (!offsets(name, start, end)) { std::fprintf(stderr, "missing tensor: %s\n", name.c_str()); return false; }
        std::size_t n = (end - start) / 2;  // BF16
        out.resize(n);
        const std::uint16_t* src = reinterpret_cast<const std::uint16_t*>(base + data_base + start);
        for (std::size_t i = 0; i < n; ++i) {
            std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
            std::memcpy(&out[i], &bits, 4);
        }
        return true;
    }
};

inline bool load_layer(const SafeTensors& st, int l, ref::LayerWeights& w) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    return st.load_f32(p + "input_layernorm.weight", w.input_ln) &&
           st.load_f32(p + "self_attn.q_proj.weight", w.wq) &&
           st.load_f32(p + "self_attn.k_proj.weight", w.wk) &&
           st.load_f32(p + "self_attn.v_proj.weight", w.wv) &&
           st.load_f32(p + "self_attn.q_norm.weight", w.q_norm) &&
           st.load_f32(p + "self_attn.k_norm.weight", w.k_norm) &&
           st.load_f32(p + "self_attn.o_proj.weight", w.wo) &&
           st.load_f32(p + "post_attention_layernorm.weight", w.post_ln) &&
           st.load_f32(p + "mlp.gate_proj.weight", w.wgate) &&
           st.load_f32(p + "mlp.up_proj.weight", w.wup) &&
           st.load_f32(p + "mlp.down_proj.weight", w.wdown);
}

}  // namespace qwen3
