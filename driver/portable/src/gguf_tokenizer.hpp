#pragma once

// Emit a HuggingFace `tokenizers`-compatible `tokenizer.json` (and
// `tokenizer_config.json`) from a GGUF file's `tokenizer.ggml.*` KV
// metadata. Lets pie's server-side Tokenizer::from_file stay unchanged
// for `.gguf` snapshot paths — the driver mints the JSON to a temp dir
// and reports that dir as the snapshot_dir in its capabilities.

#include <filesystem>
#include <string>

namespace pie_portable_driver {

// The SentencePiece (gemma) normalizer node as a serialized `tokenizers`
// JSON string. `add_space_prefix` (SentencePiece add_dummy_prefix /
// GGUF tokenizer.ggml.add_space_prefix) selects the leading-metaspace
// Prepend. Exposed for the bin-level normalizer-shape regression test.
std::string spm_normalizer_json(bool add_space_prefix);

// Read the tokenizer KVs from `gguf_file` and write `tokenizer.json` and
// `tokenizer_config.json` into `target_dir` (must exist). Throws on:
//   * unsupported `tokenizer.ggml.model` (e.g. "bert" / "t5" / "rwkv")
//   * unknown `tokenizer.ggml.pre` value (we hardcode llama.cpp's table)
//   * missing required arrays (`tokens`, `merges` for BPE)
// The thrown message names the offending key and what we support.
void emit_tokenizer_files(const std::filesystem::path& gguf_file,
                          const std::filesystem::path& target_dir);

}  // namespace pie_portable_driver
