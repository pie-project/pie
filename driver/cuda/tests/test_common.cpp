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
#include "loader/safetensors_manifest.hpp"
#include "loader/shard_plan.hpp"

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

struct TempDir {
    std::filesystem::path path;

    TempDir() {
        const auto now =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
               ("pie-driver-common-test-" + std::to_string(now));
        std::filesystem::create_directories(path);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

void write_text(const std::filesystem::path& path, std::string_view text) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("open failed: " + path.string());
    out << text;
}

void touch(const std::filesystem::path& path) {
    write_text(path, "");
}

void test_safetensors_manifest_single_and_index_preferences() {
    namespace common = pie_driver_common;
    TempDir tmp;
    const auto single = tmp.path / "model.safetensors";
    const auto index = tmp.path / "model.safetensors.index.json";
    touch(single);
    write_text(index, R"json({
        "metadata": {"total_size": 4},
        "weight_map": {
            "b": "model-00002-of-00002.safetensors",
            "a": "model-00001-of-00002.safetensors",
            "a_dup": "model-00001-of-00002.safetensors"
        }
    })json");

    auto prefer_index = common::discover_safetensors_manifest(tmp.path);
    CHECK(prefer_index.sharded);
    CHECK_EQ(prefer_index.source_path.filename().string(),
             std::string("model.safetensors.index.json"));
    CHECK_EQ(prefer_index.shard_paths.size(), std::size_t{2});
    CHECK_EQ(prefer_index.shard_paths[0].filename().string(),
             std::string("model-00001-of-00002.safetensors"));
    CHECK_EQ(prefer_index.shard_paths[1].filename().string(),
             std::string("model-00002-of-00002.safetensors"));

    auto prefer_single = common::discover_safetensors_manifest(
        tmp.path, common::SafetensorsLayoutPreference::SingleFile);
    CHECK(!prefer_single.sharded);
    CHECK_EQ(prefer_single.source_path.filename().string(),
             std::string("model.safetensors"));
    CHECK_EQ(prefer_single.shard_paths.size(), std::size_t{1});
}

void test_safetensors_manifest_errors() {
    namespace common = pie_driver_common;
    {
        TempDir tmp;
        expect_throws(
            [&] { (void)common::discover_safetensors_manifest(tmp.path); },
            "missing safetensors files");
    }
    {
        TempDir tmp;
        write_text(tmp.path / "model.safetensors.index.json",
                   R"json({"metadata": {}})json");
        expect_throws(
            [&] { (void)common::discover_safetensors_manifest(tmp.path); },
            "index without weight_map");
    }
}

void test_shard_plan() {
    namespace common = pie_driver_common;
    auto axis0 = common::plan_axis_shard({8, 4}, 0, 1, 2, "axis0");
    CHECK_EQ(axis0.shard_dim, std::int64_t{4});
    CHECK_EQ(axis0.offset, std::int64_t{4});
    CHECK((axis0.output_shape == std::vector<std::int64_t>{4, 4}));

    auto axis1 = common::plan_axis_shard({8, 4}, 1, 1, 2, "axis1");
    CHECK_EQ(axis1.shard_dim, std::int64_t{2});
    CHECK_EQ(axis1.offset, std::int64_t{2});
    CHECK((axis1.output_shape == std::vector<std::int64_t>{8, 2}));

    auto rows = common::plan_row_range_shard(10, 2, 6, 1, 2, "rows");
    CHECK_EQ(rows.row_start, std::int64_t{5});
    CHECK_EQ(rows.rows, std::int64_t{3});

    expect_throws(
        [] { (void)common::plan_axis_shard({7, 4}, 0, 0, 2, "bad"); },
        "non-divisible axis shard");
    expect_throws(
        [] { (void)common::plan_axis_shard({8, 4}, 1, 2, 2, "rank"); },
        "invalid rank");
    expect_throws(
        [] { (void)common::plan_row_range_shard(4, 2, 4, 0, 2, "range"); },
        "row range out of bounds");
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
    test_safetensors_manifest_single_and_index_preferences();
    test_safetensors_manifest_errors();
    test_shard_plan();
    test_hf_config_json();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d common driver test failure(s)\n", g_failures);
        return 1;
    }
    return 0;
}
