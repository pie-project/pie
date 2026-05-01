#include "kv_cache.hpp"

#include <stdexcept>
#include <string>

namespace pie_portable_driver {

KvCachePaged::KvCachePaged(ggml_backend_t backend,
                           std::int32_t n_layers,
                           std::int32_t n_kv_heads,
                           std::int32_t head_dim,
                           std::int32_t total_pages,
                           std::int32_t page_size,
                           ggml_type    dtype)
    : backend_(backend),
      n_layers_(n_layers),
      n_kv_heads_(n_kv_heads),
      head_dim_(head_dim),
      total_pages_(total_pages),
      page_size_(page_size) {
    if (total_pages <= 0 || page_size <= 0) {
        throw std::runtime_error("kv_cache: total_pages / page_size must be > 0");
    }

    constexpr std::size_t MAX_TENSORS = 4096;
    ggml_init_params ip{
        /*.mem_size   =*/ ggml_tensor_overhead() * MAX_TENSORS,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_ = ggml_init(ip);
    if (!ctx_) throw std::runtime_error("kv_cache: ggml_init failed");

    const std::int64_t n_embd_gqa = static_cast<std::int64_t>(n_kv_heads) * head_dim;
    const std::int64_t n_rows = static_cast<std::int64_t>(total_pages) * page_size;

    k_layers_.reserve(n_layers);
    v_layers_.reserve(n_layers);
    for (std::int32_t il = 0; il < n_layers; ++il) {
        auto* k = ggml_new_tensor_2d(ctx_, dtype, n_embd_gqa, n_rows);
        auto* v = ggml_new_tensor_2d(ctx_, dtype, n_embd_gqa, n_rows);
        ggml_set_name(k, ("kv.k." + std::to_string(il)).c_str());
        ggml_set_name(v, ("kv.v." + std::to_string(il)).c_str());
        k_layers_.push_back(k);
        v_layers_.push_back(v);
    }

    buf_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
    if (!buf_) {
        ggml_free(ctx_);
        ctx_ = nullptr;
        throw std::runtime_error(
            "kv_cache: ggml_backend_alloc_ctx_tensors failed (out of memory?)");
    }
}

KvCachePaged::~KvCachePaged() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

std::size_t KvCachePaged::buffer_size() const noexcept {
    return buf_ ? ggml_backend_buffer_get_size(buf_) : 0;
}

}  // namespace pie_portable_driver
