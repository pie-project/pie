#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "tensor.hpp"
#include "profiler.hpp"
#include "flashinfer_ops.cuh"
#include "stack_allocator.cuh"
// Forward declarations

typedef struct cublasLtContext* cublasLtHandle_t;
typedef struct CUstream_st* cudaStream_t;


/**
 * @brief Base class for all model components (modules).
 */
template <typename T>
class Module {
public:
    virtual ~Module() = default;
    virtual std::map<std::string, Tensor<T>*> get_parameters() = 0;
};



/**
 * @brief Consolidates all workspace memory, intermediate computations, and index
 * pointers for a single forward pass. This object manages the lifecycle of the
 * temporary workspace buffer required for the model execution.
 */
template <typename T>
class L4maBuffer {
public:
    // --- Device-side pointers, valid after `plan()` is called ---
    Tensor<int32_t> input_ids;
    Tensor<int32_t> position_ids;
    Tensor<int32_t> kv_page_indices;
    Tensor<int32_t> kv_page_indptr;
    Tensor<int32_t> kv_last_page_lens;
    Tensor<int32_t> qo_indptr;
    Tensor<uint8_t> custom_mask;
    Tensor<int32_t> mask_indptr;
    Tensor<int32_t> kv_batch_indices;
    Tensor<int32_t> kv_positions;
    Tensor<int32_t> output_indices_src;

    // --- Shape, Config, and Execution Context ---
    const L4maConfig& config;
    const int32_t page_size;
    const int32_t dist_size;
    size_t num_tokens;
    size_t batch_size;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;

    // --- Handlers ---
    flashinfer::BatchPrefillHandler prefill_handler;

    // --- Constructor / Destructor ---
    L4maBuffer(const L4maConfig& cfg, int32_t page_size, int32_t dist_size, size_t workspace_size);
    ~L4maBuffer();

    // --- Deleted Functions ---
    L4maBuffer(const L4maBuffer&) = delete;
    L4maBuffer& operator=(const L4maBuffer&) = delete;
    L4maBuffer(L4maBuffer&&) = delete;
    L4maBuffer& operator=(L4maBuffer&&) = delete;

    // --- Static Method for Workspace Calculation ---
    static size_t get_workspace_size(
        const L4maConfig& config,
        size_t max_num_tokens,
        size_t max_batch_size,
        size_t max_kv_seqlens,
        size_t dist_size
    );

    // --- Public Methods ---
    void plan(
        cudaStream_t strm,
         std::vector<int32_t>& input_ids_host,
         std::vector<int32_t>& position_ids_host,
         std::vector<int32_t>& kv_page_indices_host,
         std::vector<int32_t>& kv_page_indptr_host,
         std::vector<int32_t>& kv_last_page_lens_host,
         std::vector<int32_t>& qo_indptr_host,
         std::vector<bool>& packed_custom_mask_host,
         std::vector<int32_t>& mask_indptr_host,
         std::vector<int32_t>& kv_batch_indices_host,
         std::vector<int32_t>& kv_positions_host,
         std::vector<int32_t>& output_indices_src_host
    );

    // --- Allocator Wrappers ---
    template <typename U> Tensor<U> allocate(size_t count);
    Tensor<uint8_t> allocate_rest();
    template <typename U> void deallocate(Tensor<U>& tensor);

private:
    size_t buffer_size_;
    std::unique_ptr<StackAllocator> allocator_;
};




/**
 * @brief Manages the paged Key-Value cache memory for all layers.
 * This class allocates a single contiguous block of device memory and provides
 * accessors to layer-specific cache pointers.
 */
template <typename T>
class L4maKVCache {
public:
    /**
     * @brief Calculates the total device memory in bytes required for the KV cache.
     * @param config Model configuration.
     * @param num_kv_pages Total number of pages available for the cache.
     * @param page_size The number of tokens each page can hold.
     * @return The required memory size in bytes.
     */
    static size_t get_workspace_size(const L4maConfig& config, int32_t num_kv_pages, int32_t page_size);

    /**
     * @brief Allocates the KV cache memory.
     * @param config Model configuration.
     * @param num_kv_pages Total number of pages for the cache.
     * @param page_size Size of each page in tokens.
     */
    L4maKVCache(const L4maConfig& config, int32_t num_kv_pages, int32_t page_size);

    /**
     * @brief Gets the pointers to the key and value cache for a specific layer.
     * @param layer_idx The index of the decoder layer.
     * @return A pair of device pointers (key_cache_ptr, value_cache_ptr).
     */
    std::pair<T*, T*> get_layer_pointers(size_t layer_idx);

private:
    const L4maConfig& config_;
    int32_t num_kv_pages_;
    int32_t page_size_;
    Tensor<T> kv_cache_; // Single tensor for both K and V caches
};




/**
 * @brief RMS Normalization layer.
 */
template <typename T>
class RMSNorm : public Module<T> {
public:
    explicit RMSNorm(const L4maConfig& config);

    // Unchanged: This layer does not use a workspace buffer.
    void forward(T* output,
                 const T* input,
                 int num_tokens,
                 cudaStream_t stream);

    std::map<std::string, Tensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    Tensor<T> weight_;
};

// ---

/**
 * @brief The MLP block of the L4MA model.
 */
template <typename T>
class L4maMlp : public Module<T> {
public:
    explicit L4maMlp(const L4maConfig& config);

    // REFACTORED: Now accepts a StackAllocator.
    void forward(ProfileScope profiler,
                 L4maBuffer<T>& buffer,
                 T* output,
                 const T* x);

    std::map<std::string, Tensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    Tensor<T> gate_proj_weights_;
    Tensor<T> up_proj_weights_;
    Tensor<T> down_proj_weights_;
};

// ---

/**
 * @brief The attention block of the L4MA model.
 */
template <typename T>
class L4maAttention : public Module<T> {
public:
    explicit L4maAttention(const L4maConfig& config);

    void forward(ProfileScope profiler,
                 L4maBuffer<T>& buffer,
                 T* attn_output,
                 const T* hidden_states,
                 T* kv_cache_k,
                 T* kv_cache_v
                );

    std::map<std::string, Tensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    Tensor<T> q_proj_weights_;
    Tensor<T> k_proj_weights_;
    Tensor<T> v_proj_weights_;
    Tensor<T> o_proj_weights_;
    // Tensor<T> q_proj_bias_;
    // Tensor<T> k_proj_bias_;
    // Tensor<T> v_proj_bias_;
};

// ---

/**
 * @brief A single decoder layer of the L4MA model.
 */
template <typename T>
class L4maDecoderLayer : public Module<T> {
public:
    explicit L4maDecoderLayer(const L4maConfig& config);

    void forward(ProfileScope profiler,
                 L4maBuffer<T>& buffer,
                 T* hidden_states,
                 T* kv_cache_k,
                 T* kv_cache_v);

    std::map<std::string, Tensor<T>*> get_parameters() override;

private:
    L4maConfig config_;
    L4maAttention<T> self_attn_;
    L4maMlp<T> mlp_;
    RMSNorm<T> input_layernorm_;
    RMSNorm<T> post_attention_layernorm_;
};

// ---

/**
 * @brief The main body of the L4MA model.
 */
template <typename T>
class L4maModel : public Module<T> {
public:
    explicit L4maModel(const L4maConfig& config);

    void forward(ProfileScope profiler,
                 L4maBuffer<T>& buffer,
                 L4maKVCache<T>& kv_cache,
                 T* final_norm_output);

    std::map<std::string, Tensor<T>*> get_parameters() override;

    Tensor<T>& get_embed_tokens_weight() { return embed_tokens_weight_; }

private:
    L4maConfig config_;
    Tensor<T> embed_tokens_weight_;
    std::vector<L4maDecoderLayer<T>> layers_;
    RMSNorm<T> norm_;
};

// ---

/**
 * @brief The L4MA model with a causal language model head.
 */
template <typename T>
class L4maForCausalLM : public Module<T> {
public:
    explicit L4maForCausalLM(const L4maConfig& config);


    std::pair<std::vector<float>, std::vector<int32_t>> forward(ProfileScope profiler,
                 L4maBuffer<T>& buffer, L4maKVCache<T>& kv_cache);

    std::map<std::string, Tensor<T>*> get_parameters() override;

    L4maConfig& get_config() { return config_; }

private:
    L4maConfig config_;
    L4maModel<T> model_;

};