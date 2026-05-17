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
                           ggml_type    dtype,
                           KvCacheQuantFormat quant_format)
    : backend_(backend),
      n_layers_(n_layers),
      n_kv_heads_(n_kv_heads),
      per_layer_head_dim_(n_layers, head_dim),
      total_pages_(total_pages),
      page_size_(page_size),
      dtype_(dtype),
      quant_format_(std::move(quant_format)) {
    allocate_();
}

KvCachePaged::KvCachePaged(ggml_backend_t backend,
                           std::int32_t n_kv_heads,
                           std::vector<std::int32_t> per_layer_head_dim,
                           std::int32_t total_pages,
                           std::int32_t page_size,
                           ggml_type    dtype,
                           KvCacheQuantFormat quant_format)
    : backend_(backend),
      n_layers_(static_cast<std::int32_t>(per_layer_head_dim.size())),
      n_kv_heads_(n_kv_heads),
      per_layer_head_dim_(std::move(per_layer_head_dim)),
      total_pages_(total_pages),
      page_size_(page_size),
      dtype_(dtype),
      quant_format_(std::move(quant_format)) {
    allocate_();
}

KvCachePaged::KvCachePaged(ggml_backend_t backend,
                           std::vector<std::int32_t> per_layer_kv_heads,
                           std::vector<std::int32_t> per_layer_head_dim,
                           std::int32_t total_pages,
                           std::int32_t page_size,
                           ggml_type    dtype,
                           KvCacheQuantFormat quant_format)
    : backend_(backend),
      n_layers_(static_cast<std::int32_t>(per_layer_head_dim.size())),
      n_kv_heads_(per_layer_kv_heads.empty() ? 0 : per_layer_kv_heads[0]),
      per_layer_head_dim_(std::move(per_layer_head_dim)),
      per_layer_kv_heads_(std::move(per_layer_kv_heads)),
      total_pages_(total_pages),
      page_size_(page_size),
      dtype_(dtype),
      quant_format_(std::move(quant_format)) {
    if (per_layer_kv_heads_.size() != per_layer_head_dim_.size()) {
        throw std::runtime_error(
            "kv_cache: per_layer_kv_heads / per_layer_head_dim size mismatch");
    }
    allocate_();
}

void KvCachePaged::allocate_() {
    if (total_pages_ <= 0 || page_size_ <= 0) {
        throw std::runtime_error("kv_cache: total_pages / page_size must be > 0");
    }
    if (n_layers_ <= 0 || per_layer_head_dim_.empty()) {
        throw std::runtime_error("kv_cache: n_layers must be > 0");
    }

    constexpr std::size_t MAX_TENSORS = 4096;
    ggml_init_params ip{
        /*.mem_size   =*/ ggml_tensor_overhead() * MAX_TENSORS,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx_ = ggml_init(ip);
    if (!ctx_) throw std::runtime_error("kv_cache: ggml_init failed");

    const std::int64_t n_rows =
        static_cast<std::int64_t>(total_pages_) * page_size_;

    k_layers_.reserve(n_layers_);
    v_layers_.reserve(n_layers_);
    for (std::int32_t il = 0; il < n_layers_; ++il) {
        const std::int64_t kv_heads_il = per_layer_kv_heads_.empty()
            ? n_kv_heads_
            : per_layer_kv_heads_[il];
        const std::int64_t n_embd_gqa =
            kv_heads_il * per_layer_head_dim_[il];
        auto* k = ggml_new_tensor_2d(ctx_, dtype_, n_embd_gqa, n_rows);
        auto* v = ggml_new_tensor_2d(ctx_, dtype_, n_embd_gqa, n_rows);
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

ggml_tensor* KvCachePaged::qdq_for_append(ggml_context* ctx,
                                          std::int32_t layer,
                                          ggml_tensor* tensor) const {
    return qdq_tensor_for_append(ctx, tensor, quant_format_,
                                 n_kv_heads_at(layer), head_dim_at(layer));
}

}  // namespace pie_portable_driver
