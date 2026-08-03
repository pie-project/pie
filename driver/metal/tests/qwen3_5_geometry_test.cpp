// What the Qwen3.5 / Qwen3-Next family will and will not run.
//
// This family's `DecodeGeometry` used to be default-constructed and its
// defaults were one preview checkpoint's dimensions: hidden 1024, 24 layers,
// 16 linear heads of 128, a 3584-wide FFN. Nothing in the path ever compared
// those numbers against the checkpoint being loaded, and nothing could have --
// the loader binds by NAME, and a name carries no dimension to disagree with.
// A different member of the same family therefore loaded, ran, and produced
// tokens at the wrong shape.
//
// So the test is not "the geometry is right for one config". It is that the
// geometry is DERIVED, and that everything not derivable is refused with the
// key that was missing. A driver that guesses here is a driver that is fluent
// and wrong, which is the failure this whole family's schema is arranged to
// avoid elsewhere.

#include <cstdio>
#include <string>

#include "batch/forward.hpp"
#include "model/qwen3_5/geometry_facts.hpp"

using pie::metal::DecodeGeometry;
using pie::metal::geometry_from_facts;
using Facts = pie::metal::batch::SetupConfig::Qwen35Facts;

namespace {

int g_pass = 0;
int g_fail = 0;

void expect(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
        std::printf("  PASS  %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

bool contains(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

/// Qwen3-Next-80B-A3B's shape, which is what this family exists to run.
Facts qwen3_next() {
    Facts f;
    f.n_layers = 48;
    f.hidden = 2048;
    f.vocab = 151936;
    f.n_q_heads = 16;
    f.n_kv_heads = 2;
    f.head_dim = 256;
    f.intermediate = 5120;
    f.gdn_k_heads = 16;
    f.gdn_v_heads = 32;
    f.gdn_k_dim = 128;
    f.gdn_v_dim = 128;
    f.gdn_conv_k = 4;
    f.full_attn_interval = 4;
    f.eps = 1e-6f;
    return f;
}

/// The refusal a config produces, or the empty string when it is accepted.
std::string refusal(const Facts& f) {
    DecodeGeometry g{};
    std::string err;
    if (geometry_from_facts(f, g, &err)) return {};
    return err.empty() ? "(refused with no reason)" : err;
}

void check_the_shape_comes_from_the_config() {
    DecodeGeometry g{};
    std::string err;
    const Facts f = qwen3_next();
    expect(geometry_from_facts(f, g, &err), "a complete config builds: " + err);
    expect(g.hidden == 2048 && g.n_layers == 48 && g.vocab == 151936,
           "the decoder's dimensions are the config's, not the struct's defaults");
    expect(g.head_dim == 256 && g.n_q_heads == 16 && g.n_kv_heads == 2,
           "so are the attention heads");
    expect(g.full_attn_interval == 4, "and the full-attention interval");

    // The two GDN widths are DERIVED. A config cannot state them
    // inconsistently with its head counts, so neither can this driver: the
    // convolution runs over q, k and v concatenated, and the value total is
    // what the output projection consumes.
    expect(g.gdn_v_total == 32 * 128, "the value total is v_heads x v_dim");
    expect(g.gdn_conv_dim == 2 * 16 * 128 + 32 * 128,
           "the convolution spans q and k at the key heads plus v at the value heads");
    // Which is exactly what the state strides are read off.
    expect(g.gdn_conv_stride_bytes() == std::size_t(g.gdn_conv_dim) * 4u * 4u,
           "the conv state stride follows the derived width");

    // Asymmetric head counts are the case the old defaults could not express:
    // they had 16 value heads, and Qwen3-Next has 32. A driver that kept the
    // default would stride one head's recurrent state into another's.
    Facts sym = f;
    sym.gdn_v_heads = 16;
    DecodeGeometry gs{};
    expect(geometry_from_facts(sym, gs, &err) && gs.gdn_conv_dim != g.gdn_conv_dim,
           "changing the value-head count changes the derived widths");
}

void check_head_dim_is_derived_only_when_it_can_be() {
    Facts f = qwen3_next();
    f.head_dim = 0;
    DecodeGeometry g{};
    std::string err;
    expect(geometry_from_facts(f, g, &err) && g.head_dim == 2048 / 16,
           "an absent head_dim falls back to hidden/n_q_heads");
}

void check_what_is_refused() {
    // Nothing at all.
    expect(contains(refusal(Facts{}), "no decoder shape"),
           "an empty config is refused rather than defaulted");

    // The linear-attention block. These decide the conv and recurrent state
    // strides, and a wrong stride reads one head's state as another's -- not a
    // crash, a fluent model with the wrong memory.
    for (const char* which : {"k_heads", "v_heads", "k_dim", "v_dim"}) {
        Facts f = qwen3_next();
        const std::string w = which;
        if (w == "k_heads") f.gdn_k_heads = 0;
        if (w == "v_heads") f.gdn_v_heads = 0;
        if (w == "k_dim") f.gdn_k_dim = 0;
        if (w == "v_dim") f.gdn_v_dim = 0;
        expect(contains(refusal(f), "linear-attention block needs"),
               "a missing linear " + w + " is refused");
    }
    {
        Facts f = qwen3_next();
        f.gdn_conv_k = 0;
        expect(contains(refusal(f), "linear_conv_kernel_dim"),
               "a missing convolution width is refused");
    }

    // Which layers are linear cannot be guessed, and an irregular pattern is
    // not rounded to a regular one -- that would put full attention on layers
    // that are linear and vice versa.
    {
        Facts f = qwen3_next();
        f.full_attn_interval = 0;
        expect(contains(refusal(f), "full_attention_interval"),
               "no stated layer pattern is refused");
        f.full_attn_interval = -1;
        expect(contains(refusal(f), "irregular"), "an irregular pattern is refused");
    }

    // GQA, and the dense width.
    {
        Facts f = qwen3_next();
        f.n_kv_heads = 5;
        expect(contains(refusal(f), "GQA"), "a head count GQA cannot divide is refused");
    }
    {
        Facts f = qwen3_next();
        f.intermediate = 0;
        expect(contains(refusal(f), "dense FFN needs"),
               "a dense model with no intermediate_size is refused");
    }
}

void check_the_routed_ffn_is_bounded_by_what_the_kernels_do() {
    const auto routed = [] {
        Facts f = qwen3_next();
        f.n_experts = 512;
        f.experts_per_token = 10;
        f.moe_intermediate = 512;
        return f;
    };
    {
        DecodeGeometry g{};
        std::string err;
        expect(geometry_from_facts(routed(), g, &err) && g.is_moe(),
               "a routed config builds: " + err);
        expect(g.ffn_width() == 512, "and ffn_width is the per-expert width");
    }
    {
        Facts f = routed();
        f.experts_per_token = 0;
        expect(contains(refusal(f), "num_experts_per_tok"),
               "experts with no top-k is refused");
    }
    {
        Facts f = routed();
        f.experts_per_token = 600;
        expect(contains(refusal(f), "exceeds num_experts"),
               "a top-k wider than the bank is refused");
    }
    {
        Facts f = routed();
        f.experts_per_token = 17;
        expect(contains(refusal(f), "top-k limit"),
               "a top-k past what the router kernel holds is refused, not clamped");
    }
    {
        Facts f = routed();
        f.n_experts = 2048;
        f.experts_per_token = 10;
        expect(contains(refusal(f), "threadgroup can rank"),
               "an expert bank past what one threadgroup ranks is refused, not clamped");
    }
    {
        Facts f = routed();
        f.moe_intermediate = 0;
        expect(contains(refusal(f), "moe_intermediate_size"),
               "a routed FFN with no expert width is refused");
    }
    {
        Facts f = routed();
        f.norm_topk_prob = false;
        expect(contains(refusal(f), "norm_topk_prob"),
               "weights normalized over all experts are refused, not approximated");
    }
    // A shared expert runs beside the routed bank on every token, and this
    // driver has no such block. Running the mixture without it is the
    // checkpoint's weights producing a different model: fluent, wrong, and
    // invisible. Refused from the CONFIG so the diagnosis arrives before the
    // load rather than as a missing tensor during it.
    {
        Facts f = routed();
        f.shared_expert_intermediate = 512;
        expect(contains(refusal(f), "shared expert"),
               "a config implying a shared expert is refused");
    }
    // Routing only some layers is a third FFN shape in the same stack, and
    // this family's DAG emits one shape per model.
    {
        Facts f = routed();
        f.decoder_sparse_step = 2;
        expect(contains(refusal(f), "decoder_sparse_step"),
               "routing every other layer is refused");
    }
    {
        Facts f = routed();
        f.mlp_only_layer_count = 3;
        expect(contains(refusal(f), "mlp_only_layers"),
               "exempting some layers from routing is refused");
    }
}

}  // namespace

int main() {
    std::printf("==== qwen3_5_geometry_test ====\n");
    check_the_shape_comes_from_the_config();
    check_head_dim_is_derived_only_when_it_can_be();
    check_what_is_refused();
    check_the_routed_ffn_is_bounded_by_what_the_kernels_do();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
