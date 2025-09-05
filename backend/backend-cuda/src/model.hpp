#pragma once

#include <memory>
#include <vector>
#include <cstdint>
#include <string>

#include "config.hpp" // Contains ModelMetadata

// No protobuf headers here!

class Model {
public:
    // --- Public nested types for the model's interface ---

    // Corresponds to l4m.ObjectKind enum
    enum class ObjectKind {
        UNSPECIFIED = 0,
        KV_BLOCK = 1,
        EMB = 2,
        DIST = 3,
    };

    // Corresponds to l4m.Allocate
    struct AllocateCommand {
        ObjectKind kind;
        uint32_t object_id_offset;
        uint32_t count;
    };

    // Corresponds to l4m.Deallocate
    struct DeallocateCommand {
        ObjectKind kind;
        uint32_t object_id_offset;
        uint32_t count;
    };

    // Corresponds to l4m.EmbedText
    struct EmbedTextCommand {
        uint32_t embedding_id;
        uint32_t token_id;
        uint32_t position_id;
    };

    // Corresponds to l4m.FillBlock
    struct FillBlockCommand {
        uint32_t last_block_len;
        std::vector<uint32_t> context_block_ids;
        std::vector<uint32_t> input_embedding_ids;
        std::vector<uint32_t> output_embedding_ids;
    };

    // Corresponds to l4m.MaskBlock
    struct MaskBlockCommand {
        uint32_t block_id;
        std::vector<bool> mask;
    };

    // Corresponds to l4m.CopyBlock
    struct CopyBlockCommand {
        uint32_t source_block_id;
        uint32_t destination_block_id;
        uint32_t source_start;
        uint32_t destination_start;
        uint32_t length;
    };

    // Corresponds to l4m.DecodeTokenDistribution
    struct DecodeTokenDistributionCommand {
      uint32_t embedding_id;
      uint32_t distribution_id;
    };

    // Corresponds to l4m.SampleTopKRequest
    struct SampleTopKCommand {
        uint32_t distribution_id;
        uint32_t k;
    };

    // Corresponds to l4m.SampleTopKResponse
    struct SampleTopKResult {
        std::vector<uint32_t> token_ids;
        std::vector<float> probabilities;
    };

    // Corresponds to l4m.ForwardText
    struct ForwardTextCommand {
        uint32_t kv_page_last_len;
        std::vector<uint32_t> kv_page_ids;
        std::vector<uint32_t> token_ids;
        std::vector<uint32_t> position_ids;
        std::vector<std::vector<uint32_t>> brle_masks; // raw BRLE buffers per token
        std::vector<uint32_t> output_indices; // indices within token_ids to produce distributions for
    };

    struct Distribution {
        std::vector<uint32_t> token_ids;
        std::vector<float> probabilities;
    };

    // ForwardText handler: returns a vector of items, each item containing a vector of distributions
    std::vector<std::vector<Distribution>> handle_forward_text(const std::vector<ForwardTextCommand>& commands);

    // --- Core Class Methods ---

    Model(const AppConfig& config, const ModelMetadata& out_metadata);
    ~Model();
    void run();

    // --- New L4M Handler Methods ---

    void handle_allocate(const std::vector<AllocateCommand>& commands);
    void handle_deallocate(const std::vector<DeallocateCommand>& commands);
    void handle_embed_text(const std::vector<EmbedTextCommand>& commands);
    void handle_fill_block(const std::vector<FillBlockCommand>& commands);
    void handle_mask_block(const std::vector<MaskBlockCommand>& commands);
    void handle_copy_block(const std::vector<CopyBlockCommand>& commands);
    void handle_decode_token_distribution(const std::vector<DecodeTokenDistributionCommand>& commands);

    std::vector<SampleTopKResult> handle_sample_top_k(const std::vector<SampleTopKCommand>& commands);


private:
    struct ModelImpl;
    std::unique_ptr<ModelImpl> pimpl;
};