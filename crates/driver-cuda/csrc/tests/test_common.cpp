#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include "model/hf_config_json.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                   \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n",                 \
                         __FILE__, __LINE__, #cond);                  \
            ++g_failures;                                             \
        }                                                             \
    } while (0)

#define CHECK_EQ(a, b)                                                \
    do {                                                              \
        const auto _a = (a);                                          \
        const auto _b = (b);                                          \
        if (!(_a == _b)) {                                            \
            std::fprintf(stderr, "FAIL: %s:%d: %s == %s\n",           \
                         __FILE__, __LINE__, #a, #b);                 \
            ++g_failures;                                             \
        }                                                             \
    } while (0)

template <typename Fn>
void expect_throws(Fn&& fn, const char* expr) {
    try {
        fn();
        std::fprintf(stderr, "FAIL: expected throw: %s\n", expr);
        ++g_failures;
    } catch (const std::exception&) {
    }
}

void test_hf_config_json() {
    namespace common = pie_driver_common;
    const auto wrapped = nlohmann::json::parse(R"json({
        "model_type": "gemma4",
        "text_config": {"model_type": "gemma4_text"},
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0},
            "sliding_attention": {"rope_theta": 10000.0}
        }
    })json");
    auto view = common::hf_config_json_view(wrapped);
    CHECK_EQ(view.outer_model_type, std::string("gemma4"));
    CHECK_EQ(view.text_model_type, std::string("gemma4_text"));
    CHECK_EQ(view.text_or_outer_model_type(), std::string("gemma4_text"));
    CHECK_EQ(view.outer_or_text_model_type(), std::string("gemma4"));
    CHECK(common::flat_rope_parameters_view(wrapped) == nullptr);

    const auto flat = nlohmann::json::parse(R"json({
        "model_type": "qwen3_5",
        "rope_parameters": {
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.5
        }
    })json");
    CHECK(common::flat_rope_parameters_view(flat) != nullptr);
    CHECK(common::flat_rope_config_view(flat) != nullptr);

    const auto llm_wrapped = nlohmann::json::parse(R"json({
        "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
        "llm_config": {"model_type": "nemotron_h"}
    })json");
    auto llm_view = common::hf_config_json_view(llm_wrapped);
    CHECK_EQ(llm_view.outer_model_type,
             std::string("NemotronH_Nano_Omni_Reasoning_V3"));
    CHECK_EQ(llm_view.text_model_type, std::string("nemotron_h"));
    CHECK_EQ(llm_view.text_or_outer_model_type(), std::string("nemotron_h"));

    const auto scaling = nlohmann::json::parse(R"json({
        "rope_scaling": {"rope_type": "llama3", "factor": 8.0},
        "rope_parameters": {"rope_theta": 10000.0}
    })json");
    const auto* rope = common::flat_rope_config_view(scaling);
    CHECK(rope != nullptr);
    CHECK_EQ((*rope)["rope_type"].get<std::string>(), std::string("llama3"));
}

}  // namespace

int main() {
    test_hf_config_json();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d common driver test failure(s)\n", g_failures);
        return 1;
    }
    return 0;
}
