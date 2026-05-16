#include "gguf_tokenizer.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gguf.h>
#include <nlohmann/json.hpp>

namespace pie_portable_driver {

namespace {

using json = nlohmann::json;

// Wrap a gguf_context with RAII for the local read.
struct GgufHandle {
    gguf_context* ctx = nullptr;
    explicit GgufHandle(const std::filesystem::path& p) {
        gguf_init_params ip{};
        ip.no_alloc = true;
        ip.ctx      = nullptr;
        ctx = gguf_init_from_file(p.string().c_str(), ip);
        if (!ctx) {
            throw std::runtime_error(
                "gguf_tokenizer: gguf_init_from_file failed for " + p.string());
        }
    }
    ~GgufHandle() { if (ctx) gguf_free(ctx); }
    GgufHandle(const GgufHandle&) = delete;
    GgufHandle& operator=(const GgufHandle&) = delete;
};

bool has_key(gguf_context* ctx, const char* k) {
    return gguf_find_key(ctx, k) >= 0;
}

std::string get_str(gguf_context* ctx, const char* k) {
    const std::int64_t id = gguf_find_key(ctx, k);
    if (id < 0) {
        throw std::runtime_error(
            std::string("gguf_tokenizer: missing string key '") + k + "'");
    }
    return std::string(gguf_get_val_str(ctx, id));
}

std::string get_str_or(gguf_context* ctx, const char* k, std::string fb) {
    const std::int64_t id = gguf_find_key(ctx, k);
    if (id < 0) return fb;
    return std::string(gguf_get_val_str(ctx, id));
}

// Read a uint32-ish scalar that may be encoded as u32/u64/i32 in GGUF.
std::int64_t get_int_or(gguf_context* ctx, const char* k, std::int64_t fb) {
    const std::int64_t id = gguf_find_key(ctx, k);
    if (id < 0) return fb;
    const auto t = gguf_get_kv_type(ctx, id);
    switch (t) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8 (ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8 (ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return static_cast<std::int64_t>(gguf_get_val_u64(ctx, id));
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default: return fb;
    }
}

std::vector<std::string> get_str_array(gguf_context* ctx, const char* k) {
    const std::int64_t id = gguf_find_key(ctx, k);
    if (id < 0) {
        throw std::runtime_error(
            std::string("gguf_tokenizer: missing array key '") + k + "'");
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(
            std::string("gguf_tokenizer: key '") + k + "' is not an array");
    }
    if (gguf_get_arr_type(ctx, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(
            std::string("gguf_tokenizer: array '") + k +
            "' has non-string element type");
    }
    const std::size_t n = gguf_get_arr_n(ctx, id);
    std::vector<std::string> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(ctx, id, i));
    }
    return out;
}

std::vector<std::int32_t> get_i32_array_or_empty(gguf_context* ctx, const char* k) {
    const std::int64_t id = gguf_find_key(ctx, k);
    if (id < 0) return {};
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return {};
    const auto et = gguf_get_arr_type(ctx, id);
    if (et != GGUF_TYPE_INT32 && et != GGUF_TYPE_UINT32) return {};
    const std::size_t n = gguf_get_arr_n(ctx, id);
    const void* raw = gguf_get_arr_data(ctx, id);
    std::vector<std::int32_t> out(n);
    if (et == GGUF_TYPE_INT32) {
        const auto* p = static_cast<const std::int32_t*>(raw);
        for (std::size_t i = 0; i < n; ++i) out[i] = p[i];
    } else {
        const auto* p = static_cast<const std::uint32_t*>(raw);
        for (std::size_t i = 0; i < n; ++i) out[i] = static_cast<std::int32_t>(p[i]);
    }
    return out;
}

// ---------------------------------------------------------------------
// Pre-tokenizer regex table.
//
// Mirrors llama.cpp's `llm_tokenizer_bpe` switch in
// `src/llama-vocab.cpp`. Where llama.cpp keeps a comment with the
// "original regex from tokenizer.json", that's the one we emit — it is
// the canonical form HF's `tokenizers` crate accepts (the "adapted"
// llama.cpp form rewrites `(?i:...)` into `(?:...)` etc. for its
// internal regex engine; HF's onig-based Split supports both, but the
// originals match what shipped tokenizer.json files actually contain).
// ---------------------------------------------------------------------

struct PreSpec {
    std::vector<std::string> regex_exprs;  // in order
    bool needs_nfc_normalizer = false;     // emit a top-level NFC normalizer
};

// Returns a non-null PreSpec for known `pre` values, or throws.
// Coverage notes (mirroring the `if-else` ladder in llama-vocab.cpp):
//   * Most BPE families share one of a handful of regexes; we
//     deduplicate at construction.
//   * We default `needs_nfc_normalizer` to false. The Qwen-family
//     reference tokenizer.json includes NFC normalization upfront;
//     Llama-3 family does not. Other families overwhelmingly follow
//     the Llama convention (no normalizer); we add NFC only where the
//     upstream tokenizer.json is known to.
PreSpec resolve_pre_spec(const std::string& pre) {
    // ---- shared regex strings (matches llama.cpp's grouping) -----------
    static const std::string kLlama3 =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
        "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
        "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static const std::string kQwen2 =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
        "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
        "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static const std::string kQwen35 =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|"
        "\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
        "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static const std::string kGpt2 =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
    static const std::string kStarcoder =
        // STARCODER / REFACT / COMMAND_R / SMOLLM / CODESHELL / EXAONE / MINERVA
        // share: { "\\p{N}", kGpt2 }
        "";  // placeholder, actual builder below
    static const std::string kJais2 =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|"
        "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|"
        "\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|"
        "\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|"
        "\\s{1,2}(?!\\S)|\\s{1}";
    static const std::string kFalcon0 =
        "[\\p{P}\\$\\+<=>\\^~\\|`]+";
    static const std::string kFalcon1 =
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
    static const std::string kFalcon2 = "[0-9][0-9][0-9]";
    static const std::string kPoro    = " ?[^(\\s|.,!?…。，、।۔،)]+";
    static const std::string kChatGLM4 =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
        "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static const std::string kTekken =
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
        "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static const std::string kGpt4o =
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
        "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
        "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static const std::string kSeedCoder =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}|"
        " ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    static const std::string kBailingMoe =
        "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}|"
        " ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";
    static const std::string kExaoneMoe =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        "[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|"
        "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+";

    static const std::string kDefault0 = "[\\p{P}\\$\\+<=>\\^~\\|]+";
    static const std::string kDefault1 = kFalcon1;  // same regex
    static const std::string kDefault2 = "\\p{N}+";
    static const std::string kDefault3 = "[0-9][0-9][0-9]";

    PreSpec spec;

    // ---- llama3 family -----------------------------------------------
    if (pre == "llama3" || pre == "llama-v3" || pre == "llama-bpe" ||
        pre == "falcon3" || pre == "falcon-h1" || pre == "pixtral" ||
        pre == "midm-2.0" || pre == "lfm2" || pre == "jina-v5-nano" ||
        pre == "dbrx" || pre == "smaug-bpe") {
        spec.regex_exprs = {kLlama3};
        return spec;
    }
    if (pre == "jais-2") {
        spec.regex_exprs = {kJais2};
        return spec;
    }

    // ---- DeepSeek family ---------------------------------------------
    if (pre == "deepseek-llm") {
        spec.regex_exprs = {
            "[\\r\\n]",
            "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐳲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
            "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
            "\\s+$",
            "[一-龥ࠀ-一가-퟿]+",
            "\\p{N}+",
        };
        return spec;
    }
    if (pre == "deepseek-coder") {
        spec.regex_exprs = {
            "[\\r\\n]",
            "\\s?\\p{L}+",
            "\\s?\\p{P}+",
            "[一-龥ࠀ-一가-퟿]+",
            "\\p{N}",
        };
        return spec;
    }
    if (pre == "deepseek-v3" || pre == "hunyuan-dense" || pre == "joyai-llm") {
        spec.regex_exprs = {
            "\\p{N}{1,3}",
            "[一-龥぀-ゟ゠-ヿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|"
            "[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        return spec;
    }
    if (pre == "youtu") {
        spec.regex_exprs = {
            "[가-힣ㄱ-ㆎ]+|[！…“”‘’—：；，、-〿︰-﹏]+|[ㄅ-ㄯ]+|[一-龥぀-ゟ゠-ヿ]+",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+"
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*"
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
            "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        return spec;
    }

    // ---- Falcon ------------------------------------------------------
    if (pre == "falcon") {
        spec.regex_exprs = {kFalcon0, kFalcon1, kFalcon2};
        return spec;
    }

    // ---- MPT / Starcoder family (single GPT-2-style regex) -----------
    if (pre == "mpt") { spec.regex_exprs = {kGpt2}; return spec; }
    if (pre == "starcoder" || pre == "refact" || pre == "command-r" ||
        pre == "smollm" || pre == "codeshell" || pre == "exaone" ||
        pre == "minerva-7b") {
        spec.regex_exprs = {"\\p{N}", kGpt2};
        return spec;
    }

    // ---- gpt-2 lookalikes (single regex, GPT-2 family) ---------------
    if (pre == "gpt-2" || pre == "phi-2" || pre == "jina-es" ||
        pre == "jina-de" || pre == "gigachat" || pre == "jina-v2-es" ||
        pre == "jina-v2-de" || pre == "a.x-4.0" || pre == "mellum" ||
        pre == "modern-bert" ||
        pre == "mpt-redpajama" /* historical alias */ ||
        pre == "olmo" || pre == "jais" || pre == "trillion" ||
        pre == "granite-docling" ||
        pre == "jina-v1-en" || pre == "jina-v2-code" || pre == "roberta-bpe") {
        spec.regex_exprs = {kGpt2};
        return spec;
    }

    // ---- Qwen family (NFC normalizer + qwen2-style regex) ------------
    if (pre == "qwen2" || pre == "deepseek-r1-qwen" ||
        pre == "kormo" || pre == "f2llmv2" ||
        pre == "stablelm2" || pre == "hunyuan" || pre == "solar-open") {
        spec.regex_exprs = {kQwen2};
        spec.needs_nfc_normalizer = true;
        return spec;
    }
    if (pre == "qwen35") {
        spec.regex_exprs = {kQwen35};
        spec.needs_nfc_normalizer = true;
        return spec;
    }

    // ---- Poro / Bloom / GPT3-Finnish (single token regex) ------------
    if (pre == "poro-chat" || pre == "bloom" || pre == "gpt3-finnish") {
        spec.regex_exprs = {kPoro};
        return spec;
    }
    if (pre == "viking") {
        spec.regex_exprs = {kPoro, "\\p{N}"};
        return spec;
    }

    if (pre == "chatglm-bpe") {  // CHATGLM4
        spec.regex_exprs = {kChatGLM4};
        return spec;
    }
    if (pre == "tekken") {
        spec.regex_exprs = {kTekken};
        return spec;
    }
    if (pre == "chameleon") {
        spec.regex_exprs = {
            "<sentinel:[0-9]+>",
            "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",
            "([\\t\\n]|    |  )",
            "\\p{N}",
            "[\\p{P}!-/:-@\\[-`{-~]",
            kGpt2,
        };
        return spec;
    }
    if (pre == "gpt-4o" || pre == "minimax-m2") {
        spec.regex_exprs = {kGpt4o};
        return spec;
    }
    if (pre == "tiny-aya") {
        spec.regex_exprs = {
            "\\d{1,3}(?=(?:\\d{3})*\\b)",
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
            "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
            "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
            "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
            "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        return spec;
    }
    if (pre == "kimi-k2") {
        spec.regex_exprs = {"\\p{Han}+"};
        return spec;
    }
    if (pre == "superbpe") {
        spec.regex_exprs = {"\\p{N}+", "(?=(\\d{3})+(?!\\d))"};
        return spec;
    }
    if (pre == "bailingmoe") {
        spec.regex_exprs = {kBailingMoe};
        return spec;
    }
    if (pre == "seed-coder") {
        spec.regex_exprs = {kSeedCoder};
        return spec;
    }
    if (pre == "grok-2") {
        spec.regex_exprs = {kQwen2};  // identical to QWEN2 case
        return spec;
    }
    if (pre == "afmoe") {
        spec.regex_exprs = {
            "\\p{AFMoE_digits}",
            "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+",
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|"
            "[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|"
            "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
        return spec;
    }
    if (pre == "exaone-moe") {
        spec.regex_exprs = {kExaoneMoe};
        return spec;
    }
    if (pre == "gemma4") {
        // SPM-style BPE: tokenize after metaspace replacement; only split
        // on newlines. We'd handle the byte-encoding off by setting
        // `use_regex=false` on ByteLevel and using a Metaspace decoder
        // instead — but pie's Gemma path isn't on GGUF yet, so emit the
        // regex and let the user file an issue if it tokenizes wrong.
        spec.regex_exprs = {"[^\\n]+|[\\n]+"};
        return spec;
    }
    if (pre == "default") {
        spec.regex_exprs = {kDefault0, kDefault1, kDefault2, kDefault3};
        return spec;
    }

    throw std::runtime_error(
        "gguf_tokenizer: unsupported tokenizer.ggml.pre='" + pre +
        "'. Update the regex table in driver/portable/src/gguf_tokenizer.cpp.");
}

// Split a "left right" merge entry into [left, right] using the LAST
// whitespace as the separator. Some token strings contain spaces (e.g.
// "Ġ Ġ" — two space-prefixed tokens), so splitting on the FIRST space
// would produce the wrong pair; in BPE serialization the convention is
// that the rightmost space is the separator. (HF's own writer joins via
// `" "` between two byte-encoded tokens, neither of which can contain
// a literal space in the encoded form unless the encoding produced one
// — so split on the single space between them.)
//
// In practice GGUF stores merges as `left + " " + right` exactly; the
// last-space rule keeps us safe for the corner case where ByteLevel
// encoding produces tokens with a literal space inside (which doesn't
// happen for the GPT-2 byte mapping — space becomes `Ġ`).
std::pair<std::string, std::string> split_merge(const std::string& m) {
    const auto pos = m.rfind(' ');
    if (pos == std::string::npos) {
        throw std::runtime_error(
            "gguf_tokenizer: merge entry has no space separator: '" + m + "'");
    }
    return {m.substr(0, pos), m.substr(pos + 1)};
}

}  // namespace

void emit_tokenizer_files(const std::filesystem::path& gguf_file,
                          const std::filesystem::path& target_dir) {
    if (!std::filesystem::is_directory(target_dir)) {
        throw std::runtime_error(
            "gguf_tokenizer: target_dir must exist and be a directory: " +
            target_dir.string());
    }

    GgufHandle gg(gguf_file);
    gguf_context* ctx = gg.ctx;

    // -- Required scalars ------------------------------------------------
    const std::string model_type = get_str(ctx, "tokenizer.ggml.model");
    if (model_type != "gpt2") {
        // SPM ("llama"), WordPiece ("bert"), Unigram ("t5"), and RWKV
        // are not yet supported. Surface clearly — the user is hitting
        // a GGUF whose tokenizer family pie can't synthesize yet.
        throw std::runtime_error(
            "gguf_tokenizer: tokenizer.ggml.model='" + model_type +
            "' is not yet supported (need: 'gpt2'). "
            "SPM/WordPiece/Unigram conversion is a separate follow-up.");
    }
    const std::string pre = get_str_or(ctx, "tokenizer.ggml.pre", "default");
    const PreSpec pre_spec = resolve_pre_spec(pre);

    const std::vector<std::string> tokens =
        get_str_array(ctx, "tokenizer.ggml.tokens");
    const std::vector<std::string> merges =
        get_str_array(ctx, "tokenizer.ggml.merges");
    const std::vector<std::int32_t> token_types =
        get_i32_array_or_empty(ctx, "tokenizer.ggml.token_type");

    const std::int64_t bos_id = get_int_or(ctx, "tokenizer.ggml.bos_token_id", -1);
    const std::int64_t eos_id = get_int_or(ctx, "tokenizer.ggml.eos_token_id", -1);
    const std::int64_t pad_id = get_int_or(ctx, "tokenizer.ggml.padding_token_id", -1);
    const std::int64_t unk_id = get_int_or(ctx, "tokenizer.ggml.unknown_token_id", -1);
    const std::int64_t sep_id = get_int_or(ctx, "tokenizer.ggml.seperator_token_id", -1);

    const std::string chat_template =
        get_str_or(ctx, "tokenizer.chat_template", "");

    // -- Build vocab map --------------------------------------------------
    json vocab = json::object();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        vocab[tokens[i]] = static_cast<std::int64_t>(i);
    }

    // -- Build merges array (pair form) -----------------------------------
    json merges_arr = json::array();
    for (const auto& m : merges) {
        const auto [a, b] = split_merge(m);
        merges_arr.push_back(json::array({a, b}));
    }

    // -- Build added_tokens -----------------------------------------------
    // GGUF token_type values (from gguf-py):
    //   1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 5=UNUSED, 6=BYTE
    // Anything not NORMAL/UNUSED/BYTE should be exposed as a special added
    // token so HF Tokenizer treats it atomically.
    std::unordered_set<std::int64_t> added_ids;
    json added_tokens = json::array();
    auto push_added = [&](std::int64_t id, bool special) {
        if (id < 0 || static_cast<std::size_t>(id) >= tokens.size()) return;
        if (added_ids.count(id)) return;
        added_ids.insert(id);
        added_tokens.push_back({
            {"id", id},
            {"content", tokens[static_cast<std::size_t>(id)]},
            {"single_word", false},
            {"lstrip", false},
            {"rstrip", false},
            {"normalized", false},
            {"special", special},
        });
    };
    if (!token_types.empty() && token_types.size() == tokens.size()) {
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto t = token_types[i];
            if (t == 3 /*CONTROL*/ || t == 4 /*USER_DEFINED*/ || t == 2 /*UNKNOWN*/) {
                push_added(static_cast<std::int64_t>(i), /*special=*/true);
            }
        }
    }
    // Always-include canonical special IDs even when token_type didn't flag them.
    push_added(bos_id, true);
    push_added(eos_id, true);
    push_added(pad_id, true);
    push_added(unk_id, true);
    push_added(sep_id, true);

    // -- Pre-tokenizer ----------------------------------------------------
    json pre_tokenizers = json::array();
    for (const auto& re : pre_spec.regex_exprs) {
        pre_tokenizers.push_back({
            {"type",     "Split"},
            {"pattern",  {{"Regex", re}}},
            {"behavior", "Isolated"},
            {"invert",   false},
        });
    }
    pre_tokenizers.push_back({
        {"type",            "ByteLevel"},
        {"add_prefix_space", false},
        {"trim_offsets",     false},
        {"use_regex",        false},
    });

    json pre_tokenizer = {
        {"type",          "Sequence"},
        {"pretokenizers", pre_tokenizers},
    };

    // -- Normalizer -------------------------------------------------------
    json normalizer = nullptr;
    if (pre_spec.needs_nfc_normalizer) {
        normalizer = {{"type", "NFC"}};
    }

    // -- Post-processor + decoder (ByteLevel pass-through) ---------------
    json byte_level_pp = {
        {"type",             "ByteLevel"},
        {"add_prefix_space", false},
        {"trim_offsets",     false},
        {"use_regex",        false},
    };
    json decoder = byte_level_pp;
    json post_processor = byte_level_pp;

    // -- BPE model body ---------------------------------------------------
    json model_body = {
        {"type",                      "BPE"},
        {"dropout",                   nullptr},
        {"unk_token",                 nullptr},
        {"continuing_subword_prefix", nullptr},
        {"end_of_word_suffix",        nullptr},
        {"fuse_unk",                  false},
        {"byte_fallback",             false},
        {"ignore_merges",             true},
        {"vocab",                     vocab},
        {"merges",                    merges_arr},
    };

    // -- tokenizer.json ---------------------------------------------------
    json tok = {
        {"version",        "1.0"},
        {"truncation",     nullptr},
        {"padding",        nullptr},
        {"added_tokens",   added_tokens},
        {"normalizer",     normalizer},
        {"pre_tokenizer",  pre_tokenizer},
        {"post_processor", post_processor},
        {"decoder",        decoder},
        {"model",          model_body},
    };

    {
        std::ofstream f(target_dir / "tokenizer.json");
        if (!f) {
            throw std::runtime_error(
                "gguf_tokenizer: cannot open tokenizer.json for write in " +
                target_dir.string());
        }
        f << tok.dump(/*indent=*/-1);
    }

    // -- tokenizer_config.json (chat template, special tokens) -----------
    json tok_cfg = json::object();
    if (!chat_template.empty()) {
        tok_cfg["chat_template"] = chat_template;
    }
    auto add_special = [&](const char* k, std::int64_t id) {
        if (id < 0 || static_cast<std::size_t>(id) >= tokens.size()) return;
        tok_cfg[k] = tokens[static_cast<std::size_t>(id)];
    };
    add_special("bos_token", bos_id);
    add_special("eos_token", eos_id);
    add_special("pad_token", pad_id);
    add_special("unk_token", unk_id);
    {
        std::ofstream f(target_dir / "tokenizer_config.json");
        if (!f) {
            throw std::runtime_error(
                "gguf_tokenizer: cannot open tokenizer_config.json for write in " +
                target_dir.string());
        }
        f << tok_cfg.dump(/*indent=*/2);
    }
}

}  // namespace pie_portable_driver
