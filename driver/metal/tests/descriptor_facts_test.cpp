// Does an artifact's descriptor say the same thing its `config.json` did?
//
// Two paths reach `ModelFacts`: `read_model_facts` parses a snapshot's
// `config.json`, and `read_model_facts_from_descriptor` reads the `pie.model/1`
// object an imported `.zt` carries. They serve the same boot, so a field one
// reads and the other does not is not a smaller answer -- it is a different
// one, and silently, because both paths return a fully-formed `ModelFacts`.
//
// The affine width is the case that bites. Every geometry defaults its quant to
// `{4, 64}` and overrides only when the config says otherwise, so a field left
// at zero is not "unknown", it is "four bits" -- and an 8-bit checkpoint served
// as an artifact would be decoded by 4-bit kernels with nothing to say so.
#include <cstdio>
#include <string>

#include "model_facts.hpp"

namespace {

int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}

}  // namespace

int main() {
    std::printf("descriptor facts\n");

    // The shape `pie model import` writes: flat, normalized, versioned. Only
    // the fields under test plus the version, since absence is the default.
    const std::string eight_bit = R"({
      "version": "pie.model/1",
      "model_type": "llama",
      "vocab_size": 128256,
      "num_hidden_layers": 16,
      "quant_bits": 8,
      "quant_group_size": 64
    })";
    const auto facts = pie::metal::read_model_facts_from_descriptor(
        eight_bit);
    expect(facts.has_value(), "a pie.model/1 descriptor is accepted");
    if (facts) {
        expect(facts->model_type == "llama", "model_type crosses");
        expect(facts->vocab_size == 128256, "vocab_size crosses");
        expect(facts->quant_bits == 8,
               "an 8-bit checkpoint stays 8-bit (it would default to 4)");
        expect(facts->quant_group_size == 64, "the affine group crosses");
    }

    // A descriptor that says nothing about quantization leaves the fields at
    // zero, which is what lets each geometry keep its own default -- the
    // override is `!= 0`, so zero has to mean "not stated".
    const std::string unquantized = R"({
      "version": "pie.model/1",
      "model_type": "llama",
      "num_hidden_layers": 16
    })";
    const auto plain = pie::metal::read_model_facts_from_descriptor(
        unquantized);
    expect(plain.has_value(), "a descriptor without a quant block is accepted");
    if (plain) {
        expect(plain->quant_bits == 0 && plain->quant_group_size == 0,
               "an unstated affine format stays unstated");
    }

    // Not a descriptor at all: a raw `config.json` has no `version`, and
    // reading one here would report HF's field names as pie's.
    const std::string foreign = R"({"model_type": "llama", "quant_bits": 8})";
    expect(!pie::metal::read_model_facts_from_descriptor(foreign).has_value(),
           "a JSON object without pie.model/1 is refused");

    // `arch_name` is the key BOTH registries use -- this driver's family
    // switch and the runtime's `instruct::create`. The descriptor carries
    // `architectures[0]` verbatim, so the reduction to a stem happens here,
    // and a suffix this misses is not a refusal: the name simply fails to
    // match any row and takes the fallback, which for `instruct::create` is
    // ChatML. A gemma-4 served that way loads correct weights, runs correct
    // kernels and answers a conversation nobody had.
    std::printf("-- architectures[0] reduces to the stem the registries key on --\n");
    struct { const char* in; const char* want; } stems[] = {
        {"Qwen3ForCausalLM", "qwen3"},
        {"Gemma4ForConditionalGeneration", "gemma4"},
        {"Gemma3ForConditionalGeneration", "gemma3"},
        // The reason the suffix list is explicit: the first `for` in this one
        // is inside the stem, so cutting there would leave `re`.
        {"ReformerForCausalLM", "reformer"},
        // Nothing to strip stays whole rather than becoming empty.
        {"Llama", "llama"},
    };
    for (const auto& c : stems) {
        const std::string got = pie::metal::arch_stem(c.in);
        expect(got == c.want,
               std::string(c.in) + " -> " + got + " (want " + c.want + ")");
    }

    // And the two paths that reach `ModelFacts` must agree on it, since a
    // snapshot boot and an artifact boot serve the same registry.
    const std::string mm_descriptor = R"({
      "version": "pie.model/1",
      "model_type": "gemma4_text",
      "arch_name": "Gemma4ForConditionalGeneration",
      "num_hidden_layers": 60
    })";
    const auto mm = pie::metal::read_model_facts_from_descriptor(mm_descriptor);
    expect(mm.has_value() && mm->arch_name == "gemma4",
           "a descriptor's ForConditionalGeneration reduces the same way");

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
