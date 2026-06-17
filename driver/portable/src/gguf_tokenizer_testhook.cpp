// Test-only entry point exposing the SentencePiece (gemma) normalizer node
// built by the GGUF tokenizer minter, so a `#[cfg(test)]` Rust test in
// pie-server (`cargo test -p pie-server --bin pie`, which runs in CI; the
// driver/portable ctest targets do not) can assert that the leading
// metaspace Prepend is emitted exactly when tokenizer.ggml.add_space_prefix
// is set — the F3 fidelity gap (a non-space-leading input must keep its
// leading ▁ to tokenize to the same IDs as the reference gemma tokenizer).
//
// In a release build no Rust code references this symbol, so the linker
// garbage-collects this object out of the final binary.

#include <cstring>
#include <string>

#include "gguf_tokenizer.hpp"

// C linkage so the Rust side can declare it without name mangling. Writes
// the serialized normalizer JSON for the given add_space_prefix flag into
// `out` (capacity `cap`) and returns its length, or -1 if it would not fit.
extern "C" int pie_portable_test_spm_normalizer_json(int add_space_prefix,
                                                     char* out,
                                                     int cap) {
    const std::string s =
        pie_portable_driver::spm_normalizer_json(add_space_prefix != 0);
    if (out == nullptr || cap <= 0 ||
        s.size() + 1 > static_cast<std::size_t>(cap)) {
        return -1;
    }
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return static_cast<int>(s.size());
}
