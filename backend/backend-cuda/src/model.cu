#include "model.hpp"

// All implementation-specific headers are safely included here
#include "l4ma.cuh"
#include "ztensor.hpp"
#include "common.cuh"
#include "stack_allocator.cuh"
#include <iostream>
#include <set>
#include <memory>
#include <stdexcept>
#include <cuda_runtime.h>
#include <vector>
#include <map>



// --- Internal data structures ---

// Represents a block in the KV cache on the CPU.
struct Block {
    std::vector<uint32_t> position_ids;
    std::vector<bool> occupancy;

    Block() = default; // Default constructor
    Block(int32_t kv_page_size)
        : position_ids(kv_page_size, 0), occupancy(kv_page_size, false) {}
};

// Represents a text embedding on the CPU.
struct TextEmbed {
    uint32_t token_id;
    uint32_t position_id;
};

// Represents a token distribution (probabilities and corresponding token IDs).
struct Dist {
    std::vector<float> probabilities;
    std::vector<int32_t> token_ids;
};

// The actual implementation of the Server is hidden in this struct.
struct Model::ModelImpl {

    std::unique_ptr<L4maForCausalLM<__nv_bfloat16>> model;
    std::unique_ptr<L4maBuffer<__nv_bfloat16>> buffer;
    std::unique_ptr<L4maKVCache<__nv_bfloat16>> kv_cache;


    // --- State Management ---
    std::map<uint32_t, Block> blocks;
    std::map<uint32_t, TextEmbed> embeds;
    std::map<uint32_t, Dist> dists;

    // Configuration
    int32_t kv_page_size;
    int32_t dist_size;

    // stream
    cudaStream_t stream;

    // --- Handler method declarations added to ModelImpl ---
    // These methods contain the core logic and have access to the model pointer.
    void handle_allocate(const std::vector<Model::AllocateCommand>& commands);
    void handle_deallocate(const std::vector<Model::DeallocateCommand>& commands);
    void handle_embed_text(const std::vector<Model::EmbedTextCommand>& commands);
    void handle_fill_block(const std::vector<Model::FillBlockCommand>& commands);
    void handle_mask_block(const std::vector<Model::MaskBlockCommand>& commands);
    void handle_copy_block(const std::vector<Model::CopyBlockCommand>& commands);
    void handle_decode_token_distribution(const std::vector<Model::DecodeTokenDistributionCommand>& commands);
    std::vector<Model::SampleTopKResult> handle_sample_top_k(const std::vector<Model::SampleTopKCommand>& commands);
};

namespace {

template<typename T>
std::unique_ptr<L4maForCausalLM<T>> load_model_internal(const AppConfig& config, const ModelMetadata& metadata) {
    std::cout << "Instantiating model structure on device..." << std::endl;

    auto model_ptr = std::make_unique<L4maForCausalLM<T>>(metadata.architecture);

    auto params_map = model_ptr->get_parameters();
    std::cout << "Found " << params_map.size() << " parameter tensors in the model structure." << std::endl;

    const auto model_dir = config.cache_dir / "models" / config.model_name;
    std::set<std::string> loaded_keys;

    for (const auto& param_file : metadata.parameters) {
        std::filesystem::path weights_path = model_dir / param_file;
        std::cout << "Reading weights from: " << weights_path.string() << std::endl;

        try {
            ztensor::zTensorReader reader(weights_path.string());
            for (const auto& name : reader.list_tensors()) {
                if (params_map.count(name) && !loaded_keys.count(name)) {
                    const auto& info = reader.get_tensor_info(name);
                    auto& target_tensor_ptr = params_map[name];

                    if (target_tensor_ptr->size() != info.num_elements()) {
                        std::cerr << "    Warning: Shape mismatch for tensor '" << name << "'. ZT: " << info.num_elements() << ", Model: " << target_tensor_ptr->size() << ". Skipping." << std::endl;
                        continue;
                    }

                    const T* host_ptr = static_cast<const T*>(reader.get_raw_tensor_pointer(name));
                    if (host_ptr) {
                        target_tensor_ptr->from_pointer(host_ptr, info.num_elements());

                        //cudaMemcpy(thrust::raw_pointer_cast(target_tensor_ptr->data()), host_ptr, info.size, cudaMemcpyHostToDevice);
                        loaded_keys.insert(name);
                    }
                }
            }
        } catch (const std::runtime_error& e) {
            std::cerr << "Warning: Could not read file " << weights_path.string() << ". Error: " << e.what() << std::endl;
        }
    }

    if (loaded_keys.size() != params_map.size()) {
        std::cout << "\nWarning: Mismatch between loaded and expected parameter counts." << std::endl;
        std::cout << "Missing parameters:" << std::endl;
        for (const auto& pair : params_map) {
            if (loaded_keys.find(pair.first) == loaded_keys.end()) {
                std::cout << "  - " << pair.first << std::endl;
            }
        }
    }

    std::cout << "\nSuccessfully loaded " << loaded_keys.size() << " expected weights." << std::endl;

    return model_ptr;
}

} // anonymous namespace

// --- New: Placeholder implementations for handler methods ---
// These are the actual implementations within the ModelImpl struct.

void Model::ModelImpl::handle_allocate(const std::vector<Model::AllocateCommand>& commands) {
    //std::cout << "  [ModelImpl] handle_allocate called with " << commands.size() << " items." << std::endl;
    for (const auto& cmd : commands) {
        if (cmd.kind == Model::ObjectKind::KV_BLOCK) {
            for (uint32_t i = 0; i < cmd.count; ++i) {
                uint32_t block_id = cmd.object_id_offset + i;
                blocks[block_id] = Block(kv_page_size);
            }
        }
    }
}

void Model::ModelImpl::handle_deallocate(const std::vector<Model::DeallocateCommand>& commands) {
    //std::cout << "  [ModelImpl] handle_deallocate called with " << commands.size() << " items." << std::endl;
    // Currently a no-op, as in the Python implementation.
    // Blocks are cleared implicitly when the model is destroyed.
}

void Model::ModelImpl::handle_embed_text(const std::vector<Model::EmbedTextCommand>& commands) {
    //std::cout << "  [ModelImpl] handle_embed_text called with " << commands.size() << " items." << std::endl;
    for (const auto& cmd : commands) {
        embeds[cmd.embedding_id] = {cmd.token_id, cmd.position_id};
    }
}
// Assuming other necessary includes and the Model::ModelImpl class definition exist above

void Model::ModelImpl::handle_fill_block(const std::vector<Model::FillBlockCommand>& commands) {

    Profiler profiler(false);
    ProfileScope scope = profiler.scope("fill", stream);

    // --- Host-side vector preparations ---
    std::vector<int32_t> kv_page_indices_host;
    std::vector<int32_t> kv_page_indptr_host = {0};
    std::vector<int32_t> kv_last_page_lens_host;
    std::vector<int32_t> qo_indptr_host = {0};
    std::vector<bool> custom_masks_host;
    std::vector<int32_t> mask_indptr_host = {0};
    std::vector<int32_t> kv_batch_indices_host;
    std::vector<int32_t> kv_positions_host;
    std::vector<int32_t> new_token_ids_host;
    std::vector<int32_t> new_position_ids_host;

    struct OutputEmbedPostproc {
        size_t logit_row_idx;
        uint32_t dest_embed_id;
    };
    std::vector<int32_t> output_indices_src_host;
    std::vector<int32_t> output_indices_dest_host;
    std::vector<OutputEmbedPostproc> output_embed_postproc;


    int batch_idx = 0;
    for (const auto& cmd : commands) {
        kv_page_indices_host.insert(kv_page_indices_host.end(), cmd.context_block_ids.begin(), cmd.context_block_ids.end());
        kv_page_indptr_host.push_back(kv_page_indices_host.size());
        kv_last_page_lens_host.push_back(cmd.last_block_len);

        int32_t num_new_tokens = cmd.input_embedding_ids.size();
        qo_indptr_host.push_back(qo_indptr_host.back() + num_new_tokens);

        size_t total_ctx_tokens = (cmd.context_block_ids.empty()) ? 0 :
                                kv_page_size * (cmd.context_block_ids.size() - 1) + cmd.last_block_len;

        mask_indptr_host.push_back(mask_indptr_host.back() + (num_new_tokens * total_ctx_tokens));

        for (int32_t i = 0; i < num_new_tokens; ++i) {
            kv_batch_indices_host.push_back(batch_idx);
            // When there's no prior context, positions should start from 0 for new tokens.
            int32_t pos = (total_ctx_tokens > 0) ? (static_cast<int32_t>(total_ctx_tokens) - num_new_tokens + i) : i;
            kv_positions_host.push_back(pos);
        }

        std::vector<uint32_t> inp_pos_ids_for_mask;

        for (size_t i = 0; i < cmd.input_embedding_ids.size(); ++i) {
            uint32_t embed_id = cmd.input_embedding_ids[i];
            auto it = embeds.find(embed_id);
            if (it != embeds.end()) {
                const auto& embed = it->second;
                new_token_ids_host.push_back(embed.token_id);
                new_position_ids_host.push_back(embed.position_id);
                inp_pos_ids_for_mask.push_back(embed.position_id);

                size_t token_abs_pos = total_ctx_tokens - num_new_tokens + i;
                uint32_t tgt_block_idx = token_abs_pos / kv_page_size;
                uint32_t tgt_block_offset = token_abs_pos % kv_page_size;

                // // print tgt_block_idx and tgt_block_offset for debugging
                // std::cout << "Processing token: " << embed.token_id
                //           << ", position: " << embed.position_id
                //           << ", token_abs_pos: " << token_abs_pos
                //           << ", target block index: " << tgt_block_idx
                //           << ", target block offset: " << tgt_block_offset << std::endl;

                if (tgt_block_idx < cmd.context_block_ids.size()) {
                    uint32_t tgt_block_id = cmd.context_block_ids[tgt_block_idx];
                    auto block_it = blocks.find(tgt_block_id);
                    if (block_it != blocks.end()) {
                        block_it->second.occupancy[tgt_block_offset] = true;
                        block_it->second.position_ids[tgt_block_offset] = embed.position_id;
                    }
                }
            }
        }

        for (size_t i = 0; i < cmd.output_embedding_ids.size(); ++i) {
            size_t logit_row = new_token_ids_host.size() - cmd.output_embedding_ids.size() + i;
            output_embed_postproc.push_back({logit_row, cmd.output_embedding_ids[i]});
            output_indices_src_host.push_back(logit_row);
            output_indices_dest_host.push_back(cmd.output_embedding_ids[i]);
        }

        if (total_ctx_tokens > 0) {
            std::vector<uint32_t> ctx_pos_ids;
            std::vector<bool> ctx_occupancy;
            ctx_pos_ids.reserve(total_ctx_tokens);
            ctx_occupancy.reserve(total_ctx_tokens);

            for (size_t i = 0; i < cmd.context_block_ids.size(); ++i) {
                uint32_t block_id = cmd.context_block_ids[i];
                const auto& block = blocks.at(block_id);
                size_t len_to_copy = (i == cmd.context_block_ids.size() - 1) ? cmd.last_block_len : kv_page_size;
                ctx_pos_ids.insert(ctx_pos_ids.end(), block.position_ids.begin(), block.position_ids.begin() + len_to_copy);
                ctx_occupancy.insert(ctx_occupancy.end(), block.occupancy.begin(), block.occupancy.begin() + len_to_copy);
            }

            // // print all ctx_pos_ids and ctx_occupancy for debugging
            // std::cout << "ctx_pos_ids: ";
            // for (const auto& pos_id : ctx_pos_ids) {
            //     std::cout << pos_id << " ";
            // }
            // std::cout << "\nctx_occupancy: ";
            // for (const auto& occ : ctx_occupancy) {
            //     std::cout << (occ ? 1 : 0) << " ";
            // }
            // std::cout << std::endl;


            for (uint32_t inp_pos_id : inp_pos_ids_for_mask) {
                for (size_t j = 0; j < total_ctx_tokens; ++j) {
                    bool causal_mask = ctx_pos_ids[j] <= inp_pos_id;
                    bool valid_mask = ctx_occupancy[j];
                    custom_masks_host.push_back(causal_mask && valid_mask);
                }
            }
        }
        batch_idx++;
    }

    scope.record("preproc");


    // // print all host vectors for debugging
    // std::cout << "kv_page_indices_host: ";
    // for (const auto& idx : kv_page_indices_host) {
    //     std::cout << idx << " ";
    // }
    // std::cout << "\nkv_page_indptr_host: ";
    // for (const auto& idx : kv_page_indptr_host) {
    //     std::cout << idx << " ";
    // }
    // std::cout << "\nkv_last_page_lens_host: ";
    // for (const auto& len : kv_last_page_lens_host) {
    //     std::cout << len << " ";
    // }
    // std::cout << "\nqo_indptr_host: ";
    // for (const auto& idx : qo_indptr_host) {
    //     std::cout << idx << " ";
    // }
    // std::cout << "\ncustom_masks_host: ";
    // for (const auto& mask : custom_masks_host) {
    //     std::cout << static_cast<int>(mask) << " ";
    // }
    // std::cout << "\nmask_indptr_host: ";
    // for (const auto& idx : mask_indptr_host) {
    //     std::cout << idx << " ";
    // }
    // std::cout << "\nnew_token_ids_host: ";
    // for (const auto& token_id : new_token_ids_host) {
    //     std::cout << token_id << " ";
    // }
    // std::cout << "\nnew_position_ids_host: ";
    // for (const auto& pos_id : new_position_ids_host) {
    //     std::cout << pos_id << " ";
    // }
    // std::cout << "\nkv_batch_indices_host: ";
    // for (const auto& batch_idx : kv_batch_indices_host) {
    //     std::cout << batch_idx << " ";
    // }
    // std::cout << "\nkv_positions_host: ";
    // for (const auto& pos : kv_positions_host) {
    //     std::cout << pos << " ";
    // }
    // std::cout << std::endl;

    // --- 2. Allocate and Plan L4maBuffer ---
    size_t num_total_new_tokens = new_token_ids_host.size();


    // Empty vectors are now allowed by StackAllocator::allocate_and_copy_async (returning zero-sized tensors).

    buffer->plan(
        stream, new_token_ids_host, new_position_ids_host,
        kv_page_indices_host, kv_page_indptr_host, kv_last_page_lens_host,
        qo_indptr_host, custom_masks_host, mask_indptr_host,
        kv_batch_indices_host, kv_positions_host, output_indices_src_host
    );
    scope.record("plan_buffer");

    auto [logits_vals, logits_indices] = model->forward(
        scope.scope("forward_pass"),
        *buffer, *kv_cache);

    // Store the top-k distributions in the map
    for (size_t i = 0; i < output_embed_postproc.size(); ++i) {
        const auto& postproc_info = output_embed_postproc[i];
        uint32_t dest_embed_id = postproc_info.dest_embed_id;

        // The returned logits_vals is a compact tensor for the requested output tokens.
        // We must use an index `i` from 0 to N-1, where N is the number of requested distributions.
        size_t src_output_row_idx = i;
        size_t src_offset = src_output_row_idx * dist_size;

        // Create a new distribution object or get the existing one
        Dist& dist = dists[dest_embed_id];
        dist.probabilities.resize(dist_size);
        dist.token_ids.resize(dist_size);

        // Copy probability values from the model's output
        std::copy(
            logits_vals.begin() + src_offset,
            logits_vals.begin() + src_offset + dist_size,
            dist.probabilities.begin()
        );
        // Copy token indices from the model's output
        std::copy(
            logits_indices.begin() + src_offset,
            logits_indices.begin() + src_offset + dist_size,
            dist.token_ids.begin()
        );
    }
    scope.record("postproc");

    //cudaStreamSynchronize(stream);
    profiler.print_report();

}

void Model::ModelImpl::handle_mask_block(const std::vector<Model::MaskBlockCommand>& commands) {
    //std::cout << "  [ModelImpl] handle_mask_block called with " << commands.size() << " items." << std::endl;
    for (const auto& cmd : commands) {
        auto it = blocks.find(cmd.block_id);
        if (it != blocks.end()) {
            Block& block = it->second;
            if (block.occupancy.size() == cmd.mask.size()) {
                block.occupancy = cmd.mask;
            } else {
                std::cerr << "Warning: Mask size mismatch for block " << cmd.block_id << std::endl;
            }
        } else {
            std::cerr << "Warning: Block not found for masking: " << cmd.block_id << std::endl;
        }
    }
}

void Model::ModelImpl::handle_copy_block(const std::vector<Model::CopyBlockCommand>& commands) {
    for (const auto& cmd : commands) {
        // Find source and destination blocks
        auto src_it = blocks.find(cmd.source_block_id);
        auto dst_it = blocks.find(cmd.destination_block_id);

        if (src_it == blocks.end()) {
            std::cerr << "Warning: Source block not found: " << cmd.source_block_id << std::endl;
            continue;
        }

        if (dst_it == blocks.end()) {
            std::cerr << "Warning: Destination block not found: " << cmd.destination_block_id << std::endl;
            continue;
        }

        Block& src_block = src_it->second;
        Block& dst_block = dst_it->second;

        // Validate bounds
        if (cmd.source_start + cmd.length > src_block.occupancy.size() ||
            cmd.destination_start + cmd.length > dst_block.occupancy.size()) {
            std::cerr << "Warning: Copy operation out of bounds for blocks "
                      << cmd.source_block_id << " -> " << cmd.destination_block_id << std::endl;
            continue;
        }

        // Copy occupancy and position_ids (CPU-side arrays)
        std::copy(
            src_block.occupancy.begin() + cmd.source_start,
            src_block.occupancy.begin() + cmd.source_start + cmd.length,
            dst_block.occupancy.begin() + cmd.destination_start
        );

        std::copy(
            src_block.position_ids.begin() + cmd.source_start,
            src_block.position_ids.begin() + cmd.source_start + cmd.length,
            dst_block.position_ids.begin() + cmd.destination_start
        );

        // Copy KV cache data (GPU-side) for each layer
        size_t num_layers = model->get_config().num_layers;
        size_t num_kv_heads = model->get_config().num_key_value_heads;
        size_t head_size = model->get_config().head_size;

        // Calculate the number of elements per token position in the KV cache
        size_t elements_per_token = num_kv_heads * head_size;
        size_t bytes_per_token = elements_per_token * sizeof(__nv_bfloat16);

        for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto [k_cache_ptr, v_cache_ptr] = kv_cache->get_layer_pointers(layer_idx);

            // Calculate source and destination offsets within the layer
            // Each block contains kv_page_size tokens, and we need to copy a range within blocks
            size_t src_token_offset = cmd.source_block_id * kv_page_size + cmd.source_start;
            size_t dst_token_offset = cmd.destination_block_id * kv_page_size + cmd.destination_start;

            size_t src_element_offset = src_token_offset * elements_per_token;
            size_t dst_element_offset = dst_token_offset * elements_per_token;

            size_t copy_size_bytes = cmd.length * bytes_per_token;

            // Copy K cache
            cudaMemcpy(
                k_cache_ptr + dst_element_offset,
                k_cache_ptr + src_element_offset,
                copy_size_bytes,
                cudaMemcpyDeviceToDevice
            );

            // Copy V cache
            cudaMemcpy(
                v_cache_ptr + dst_element_offset,
                v_cache_ptr + src_element_offset,
                copy_size_bytes,
                cudaMemcpyDeviceToDevice
            );
        }
    }
}

void Model::ModelImpl::handle_decode_token_distribution(const std::vector<Model::DecodeTokenDistributionCommand>& commands) {
    //std::cout << "  [ModelImpl] handle_decode_token_distribution called with " << commands.size() << " items." << std::endl;
    // This is a no-op in the provided python implementation.
    // The logic is integrated into fill_block where top-k results are computed and stored directly.
}

std::vector<Model::SampleTopKResult> Model::ModelImpl::handle_sample_top_k(const std::vector<Model::SampleTopKCommand>& commands) {
    //std::cout << "  [ModelImpl] handle_sample_top_k called with " << commands.size() << " items." << std::endl;
    std::vector<Model::SampleTopKResult> results;
    results.reserve(commands.size());

    for (const auto& cmd : commands) {
        Model::SampleTopKResult res;

        auto it = dists.find(cmd.distribution_id);
        if (it == dists.end()) {
            std::cerr << "Warning: sample_top_k requested invalid distribution_id " << cmd.distribution_id << std::endl;
            results.push_back(res); // Return empty result
            continue;
        }

        const auto& dist = it->second;

        // Create pairs for sorting from the entire stored distribution.
        std::vector<std::pair<float, int32_t>> sorted_pairs;
        sorted_pairs.reserve(dist.probabilities.size());
        for (size_t i = 0; i < dist.probabilities.size(); ++i) {
            sorted_pairs.push_back({
                dist.probabilities[i],
                dist.token_ids[i]
            });
        }

        // Sort pairs in descending order based on probability.
        std::sort(sorted_pairs.begin(), sorted_pairs.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        // Determine how many results to return (k).
        uint32_t k = (cmd.k > 0 && cmd.k < sorted_pairs.size()) ? cmd.k : sorted_pairs.size();

        // Populate result from the top k sorted pairs.
        res.probabilities.reserve(k);
        res.token_ids.reserve(k);
        for (uint32_t i = 0; i < k; ++i) {
            res.probabilities.push_back(sorted_pairs[i].first);
            res.token_ids.push_back(sorted_pairs[i].second);
        }

        results.push_back(res);
    }
    return results;
}

// ForwardText (batched implementation aggregating all items into single embed/fill/sample passes)
std::vector<std::vector<Model::Distribution>> Model::handle_forward_text(const std::vector<ForwardTextCommand>& commands) {
    std::vector<std::vector<Model::Distribution>> results(commands.size());
    if (commands.empty()) return results;

    // 1. Allocate embedding IDs for every token across all items (single static counter)
    static uint32_t next_embed_id = 2'000'000; // reserved ephemeral id space
    std::vector<EmbedTextCommand> all_embed_cmds;
    all_embed_cmds.reserve([&](){ size_t s=0; for (auto& c:commands) s+=c.token_ids.size(); return s;}());

    // Per-item token embedding ids for mapping output indices -> embedding ids
    std::vector<std::vector<uint32_t>> per_item_embed_ids(commands.size());
    for (size_t item_idx = 0; item_idx < commands.size(); ++item_idx) {
        const auto& cmd = commands[item_idx];
        per_item_embed_ids[item_idx].reserve(cmd.token_ids.size());
        for (size_t t = 0; t < cmd.token_ids.size(); ++t) {
            uint32_t emb_id = next_embed_id++;
            per_item_embed_ids[item_idx].push_back(emb_id);
            all_embed_cmds.push_back({emb_id, cmd.token_ids[t], cmd.position_ids[t]});
        }
    }
    pimpl->handle_embed_text(all_embed_cmds);

    // 2. Build all FillBlockCommands (one per ForwardText item) and gather output embedding ids
    std::vector<FillBlockCommand> fill_cmds;
    fill_cmds.reserve(commands.size());
    std::vector<std::vector<uint32_t>> per_item_output_embed_ids(commands.size());
    size_t total_requested_dists = 0;
    for (size_t item_idx = 0; item_idx < commands.size(); ++item_idx) {
        const auto& cmd = commands[item_idx];
        const auto& embed_ids = per_item_embed_ids[item_idx];
        auto& out_embed_ids = per_item_output_embed_ids[item_idx];
        out_embed_ids.reserve(cmd.output_indices.size());
        for (auto out_idx : cmd.output_indices) {
            if (out_idx < embed_ids.size()) out_embed_ids.push_back(embed_ids[out_idx]);
        }
        total_requested_dists += out_embed_ids.size();
        fill_cmds.push_back(FillBlockCommand{cmd.kv_page_last_len, cmd.kv_page_ids, embed_ids, out_embed_ids});
    }

    // NOTE: BRLE masks (cmd.brle_masks) are currently ignored; existing fill_block builds causal masks internally.
    // A future enhancement would decode and integrate them into custom_masks_host inside handle_fill_block.

    // 3. Single batched fill_block
    if (!fill_cmds.empty()) {
        pimpl->handle_fill_block(fill_cmds);
    }

    // 4. Collect sample commands for all requested output embeddings
    std::vector<SampleTopKCommand> sample_cmds;
    sample_cmds.reserve(total_requested_dists);
    for (auto& out_ids : per_item_output_embed_ids) {
        for (auto eid : out_ids) sample_cmds.push_back({eid, (uint32_t)pimpl->dist_size});
    }
    auto sample_results = pimpl->handle_sample_top_k(sample_cmds);

    // 5. Reconstruct per-item distributions from flattened sample_results
    size_t cursor = 0;
    for (size_t item_idx = 0; item_idx < commands.size(); ++item_idx) {
        auto& out_ids = per_item_output_embed_ids[item_idx];
        auto& item_out = results[item_idx];
        item_out.reserve(out_ids.size());
        for (size_t j = 0; j < out_ids.size(); ++j) {
            if (cursor >= sample_results.size()) break; // safety
            auto& sr = sample_results[cursor++];
            item_out.push_back(Model::Distribution{sr.token_ids, sr.probabilities});
        }
    }
    return results;
}

// --- Public Interface Implementation ---

Model::Model(const AppConfig& config,const ModelMetadata& out_metadata)
    : pimpl(std::make_unique<ModelImpl>()) {

    std::cout << "Starting service..." << std::endl;
    // Load the model and store it in the implementation object
    pimpl->model = load_model_internal<__nv_bfloat16>(config, out_metadata);
    std::cout << "Model loaded successfully and is resident on the GPU." << std::endl;

    // Initialize the L4maBuffer with the model's configuration
    size_t buffer_workspace_size = L4maBuffer<__nv_bfloat16>::get_workspace_size(
        pimpl->model->get_config(),
        4096, // number of new tokens
        4096, // batch size
        4096, // number of old tokens
        config.dist_size
    );

    size_t kv_cache_workspace_size = L4maKVCache<__nv_bfloat16>::get_workspace_size(
        pimpl->model->get_config(),
        config.max_num_kv_pages,
        config.kv_page_size
    );

    // print out the workspace sizes (in MB)
    std::cout << "Buffer workspace size: " << (buffer_workspace_size / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "KV Cache workspace size: " << (kv_cache_workspace_size / (1024 * 1024)) << " MB" << std::endl;

    pimpl->buffer = std::make_unique<L4maBuffer<__nv_bfloat16>>(pimpl->model->get_config(), config.kv_page_size, config.dist_size, buffer_workspace_size);
    pimpl->kv_cache = std::make_unique<L4maKVCache<__nv_bfloat16>>(pimpl->model->get_config(), config.max_num_kv_pages, config.kv_page_size);
    // Initialize the CUDA stream
    cudaStreamCreate(&pimpl->stream);


    // initialize kv cache
    //pimpl->model->create_kv_device_vectors(config.max_num_kv_pages * config.kv_page_size);

    // Initialize state
    pimpl->kv_page_size = config.kv_page_size;
    pimpl->dist_size = config.dist_size;

}

Model::~Model() = default;

void Model::run() {
    // This function is now used as a test routine for handle_fill_block.

    std::cout << "\n--- [START] Running Test Routine for handle_fill_block ---" << std::endl;

    // 1. Define test parameters: a random sequence of tokens and IDs.
    const std::vector<uint32_t> token_ids = {3513, 5331, 533, 11};
    const uint32_t block_id = 101; // A unique ID for our KV block
    const uint32_t embed_id_offset = 201; // Starting ID for our input embeddings
    const uint32_t dist_id = 301;         // ID for the output distribution object

    // Ensure the tokens fit within a single page.
    if (token_ids.size() > static_cast<size_t>(pimpl->kv_page_size)) {
        std::cerr << "Test Error: Number of tokens exceeds kv_page_size." << std::endl;
        return;
    }

    // 2. Call handle_allocate to allocate a page for the KV cache.
    std::cout << "\n[Step 1] Allocating KV Block..." << std::endl;
    Model::AllocateCommand alloc_cmd;
    alloc_cmd.kind = Model::ObjectKind::KV_BLOCK;
    alloc_cmd.object_id_offset = block_id;
    alloc_cmd.count = 1;
    handle_allocate({alloc_cmd});
    std::cout << "Allocated block with ID: " << block_id << std::endl;

    // 3. Call handle_embed_texts to create mappings for token and position IDs.
    std::cout << "\n[Step 2] Creating Text Embeddings..." << std::endl;
    std::vector<Model::EmbedTextCommand> embed_cmds;
    std::vector<uint32_t> input_embed_ids;
    for (size_t i = 0; i < token_ids.size(); ++i) {
        uint32_t current_embed_id = embed_id_offset + i;
        input_embed_ids.push_back(current_embed_id);

        Model::EmbedTextCommand embed_cmd;
        embed_cmd.embedding_id = current_embed_id;
        embed_cmd.token_id = token_ids[i];
        embed_cmd.position_id = i; // Simple sequential positions 0, 1, 2, ...
        embed_cmds.push_back(embed_cmd);
    }
    handle_embed_text(embed_cmds);
    std::cout << "Created " << embed_cmds.size() << " embeddings." << std::endl;

    // 4. Call handle_fill_block to do a single forward pass.
    std::cout << "\n[Step 3] Calling handle_fill_block for a forward pass..." << std::endl;
    Model::FillBlockCommand fill_cmd;
    fill_cmd.last_block_len = token_ids.size(); // No previous context in the block
    fill_cmd.context_block_ids = {block_id}; // The block to fill with new KV data
    fill_cmd.input_embedding_ids = input_embed_ids;
    fill_cmd.output_embedding_ids = {dist_id}; // Store logits for the last token in this distribution
    handle_fill_block({fill_cmd});
    std::cout << "handle_fill_block completed." << std::endl;

    // 5. Verify the output by sampling the resulting distribution.
    std::cout << "\n[Step 4] Verifying output with handle_sample_top_k..." << std::endl;
    Model::SampleTopKCommand sample_cmd;
    sample_cmd.distribution_id = dist_id;
    sample_cmd.k = 5; // Get top 5 predictions
    auto results = handle_sample_top_k({sample_cmd});

    if (!results.empty()) {
        const auto& result = results[0];
        std::cout << "Successfully retrieved Top-" << result.token_ids.size() << " predicted next tokens:" << std::endl;
        for (size_t i = 0; i < result.token_ids.size(); ++i) {
            std::cout << "  - Token ID: " << result.token_ids[i]
                      << ", Probability: " << result.probabilities[i] << std::endl;
        }
    } else {
        std::cerr << "Test Error: Failed to get sampling results." << std::endl;
    }

    std::cout << "\n--- [END] Test Routine Finished ---\n" << std::endl;
}

// --- New: Public handler methods delegating to PIMPL ---
// These methods are the public API of your Model class. They simply
// forward the calls to the actual implementation in ModelImpl.

void Model::handle_allocate(const std::vector<AllocateCommand>& commands) {
    pimpl->handle_allocate(commands);
}

void Model::handle_deallocate(const std::vector<DeallocateCommand>& commands) {
    pimpl->handle_deallocate(commands);
}

void Model::handle_embed_text(const std::vector<EmbedTextCommand>& commands) {
    pimpl->handle_embed_text(commands);
}

void Model::handle_fill_block(const std::vector<FillBlockCommand>& commands) {
    pimpl->handle_fill_block(commands);
}

void Model::handle_mask_block(const std::vector<MaskBlockCommand>& commands) {
    pimpl->handle_mask_block(commands);
}

void Model::handle_copy_block(const std::vector<CopyBlockCommand>& commands) {
    pimpl->handle_copy_block(commands);
}

void Model::handle_decode_token_distribution(const std::vector<DecodeTokenDistributionCommand>& commands) {
    pimpl->handle_decode_token_distribution(commands);
}

std::vector<Model::SampleTopKResult> Model::handle_sample_top_k(const std::vector<SampleTopKCommand>& commands) {
    return pimpl->handle_sample_top_k(commands);
}