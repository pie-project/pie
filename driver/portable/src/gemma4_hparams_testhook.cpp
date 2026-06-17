// Test-only entry points exercising the gemma4 GGUF load-path hardening
// guards in isolation — no model file, no backend, no compute.
//
// Linked into `pie_driver_portable_lib` and called from `#[cfg(test)]` Rust
// tests in pie-server (`cargo test -p pie-server --bin pie`, which runs in CI;
// the driver/portable ctest targets do not). In a release build no Rust code
// references these symbols, so the linker garbage-collects this object out of
// the final binary.
//
// Why this exists (#712): the gemma4 GGUF load path is otherwise guarded only
// by a real 12B boot (#705). Two facts it derives from GGUF metadata, if wrong,
// previously degraded generation SILENTLY rather than failing:
//
//   1. `parse_gguf_hparams` reduces the per-layer `attention.head_count_kv`
//      array to the two scalars the attention path uses (sliding + global).
//      A length/within-type mismatch must now throw, not be skipped.
//   2. `validate_gemma4_rope_freqs` requires the root `rope_freqs.weight`
//      tensor whenever the model has a full-attention layer (the GGUF carries
//      the proportional-RoPE factors only there). Absent => throw, not a
//      silent full-rotation fallback.

#include <cstdint>
#include <string>
#include <vector>

#include "gguf_archive.hpp"   // GgufMeta
#include "gguf_hparams.hpp"   // parse_gguf_hparams, validate_gemma4_rope_freqs
#include "hf_config.hpp"      // Hparams

namespace {

using pie_portable_driver::GgufMeta;

void add_kv(GgufMeta& m, const std::string& key, double num) {
    GgufMeta::KV kv;
    kv.key = key;
    kv.type = 0;
    kv.num_value = num;
    m.kv.emplace(key, std::move(kv));
}

void add_int_arr(GgufMeta& m, const std::string& key,
                 std::vector<std::int64_t> arr) {
    GgufMeta::KV kv;
    kv.key = key;
    kv.type = 0;
    kv.arr_int = std::move(arr);
    m.kv.emplace(key, std::move(kv));
}

// Synthetic dense gemma4 metadata: 6 layers, sliding_window_pattern
// [1,1,1,1,1,0] -> layer_types "sssssg" (1=sliding, 0=full/global). The
// caller supplies the per-layer head_count_kv array so each guard case can be
// driven independently.
GgufMeta make_gemma4_meta(std::vector<std::int64_t> head_count_kv) {
    GgufMeta m;
    m.general_architecture = "gemma4";
    m.has_output_weight = true;
    add_kv(m, "gemma4.block_count", 6);
    add_kv(m, "gemma4.attention.head_count", 16);
    add_kv(m, "gemma4.embedding_length", 3840);
    add_kv(m, "gemma4.feed_forward_length", 15360);
    add_kv(m, "gemma4.attention.key_length", 512);
    add_kv(m, "gemma4.attention.key_length_swa", 256);
    add_kv(m, "gemma4.rope.freq_base", 1000000.0);
    add_kv(m, "gemma4.rope.freq_base_swa", 10000.0);
    add_int_arr(m, "gemma4.attention.sliding_window_pattern",
                {1, 1, 1, 1, 1, 0});
    add_int_arr(m, "gemma4.attention.head_count_kv", std::move(head_count_kv));
    return m;
}

// 0 = parsed without throwing, 1 = threw (mismatch caught). Selects the
// head_count_kv array shape by `mode`:
//   0: valid       [8,8,8,8,8,1] (consistent per attention type)
//   1: wrong length [8,8,8,8,8]  (5 entries for 6 layers)
//   2: inconsistent [8,8,8,4,8,1] (sliding layers disagree: 8 vs 4)
int head_count_kv_status(int mode) {
    std::vector<std::int64_t> kvh;
    switch (mode) {
        case 0: kvh = {8, 8, 8, 8, 8, 1}; break;
        case 1: kvh = {8, 8, 8, 8, 8};    break;
        case 2: kvh = {8, 8, 8, 4, 8, 1}; break;
        default: return -1;
    }
    try {
        (void)pie_portable_driver::parse_gguf_hparams(
            make_gemma4_meta(std::move(kvh)));
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

}  // namespace

extern "C" {

// See head_count_kv_status: 0=ok, 1=threw, -1=bad mode.
int pie_portable_test_gemma4_head_count_kv_status(int mode) {
    return head_count_kv_status(mode);
}

// For the valid array, returns the scalar the reduction produced:
//   0 -> num_key_value_heads (sliding)   1 -> num_global_key_value_heads
int pie_portable_test_gemma4_parsed_kv_heads(int which) {
    const auto h = pie_portable_driver::parse_gguf_hparams(
        make_gemma4_meta({8, 8, 8, 8, 8, 1}));
    switch (which) {
        case 0: return h.num_key_value_heads;
        case 1: return h.num_global_key_value_heads;
        default: return -1;
    }
}

// validate_gemma4_rope_freqs: returns 1 if it threw, 0 if it accepted.
//   has_full_layer != 0 -> layer_types contains a 'g'
//   present       != 0 -> rope_freqs.weight tensor is present
int pie_portable_test_gemma4_rope_freqs_status(int has_full_layer,
                                               int present) {
    std::vector<char> layer_types = has_full_layer
        ? std::vector<char>{'s', 's', 'g'}
        : std::vector<char>{'s', 's', 's'};
    try {
        pie_portable_driver::validate_gemma4_rope_freqs(layer_types,
                                                        present != 0);
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}

}  // extern "C"
