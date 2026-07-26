#include "planner_profile_cache.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "../model/config.hpp"
#include "kv_cache_format.hpp"

namespace pie_cuda_driver {
namespace {

nlohmann::json key_to_json(const PlannerProfileKey& k) {
    return nlohmann::json{
        {"gpu_name", k.gpu_name},
        {"compute_major", k.compute_major},
        {"compute_minor", k.compute_minor},
        {"sm_count", k.sm_count},
        {"kv_cache_dtype", k.kv_cache_dtype},
        {"tp_size", k.tp_size},
        {"model_type", k.model_type},
        {"hidden_size", k.hidden_size},
        {"num_hidden_layers", k.num_hidden_layers},
        {"num_attention_heads", k.num_attention_heads},
        {"num_key_value_heads", k.num_key_value_heads},
        {"head_dim", k.head_dim},
    };
}

bool field_eq(const nlohmann::json& key, const char* name,
              const std::string& expected) {
    const auto it = key.find(name);
    return it != key.end() && it->is_string() &&
           it->get<std::string>() == expected;
}

bool field_eq(const nlohmann::json& key, const char* name, int expected) {
    const auto it = key.find(name);
    return it != key.end() && it->is_number_integer() &&
           it->get<int>() == expected;
}

bool key_matches(const nlohmann::json& stored, const PlannerProfileKey& k) {
    return field_eq(stored, "gpu_name", k.gpu_name) &&
           field_eq(stored, "compute_major", k.compute_major) &&
           field_eq(stored, "compute_minor", k.compute_minor) &&
           field_eq(stored, "sm_count", k.sm_count) &&
           field_eq(stored, "kv_cache_dtype", k.kv_cache_dtype) &&
           field_eq(stored, "tp_size", k.tp_size) &&
           field_eq(stored, "model_type", k.model_type) &&
           field_eq(stored, "hidden_size", k.hidden_size) &&
           field_eq(stored, "num_hidden_layers", k.num_hidden_layers) &&
           field_eq(stored, "num_attention_heads", k.num_attention_heads) &&
           field_eq(stored, "num_key_value_heads", k.num_key_value_heads) &&
           field_eq(stored, "head_dim", k.head_dim);
}

// Reads the whole document, tolerating every failure mode. A cache that cannot
// be parsed must not take the process down: it is an optimisation record, and
// the planner's analytic score is a complete fallback.
nlohmann::json read_document(const std::filesystem::path& path,
                             std::string* error) {
    nlohmann::json root = nlohmann::json::object();
    if (path.empty()) return root;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) return root;
    try {
        std::ifstream in(path);
        if (!in) return root;
        root = nlohmann::json::parse(in);
    } catch (const std::exception& e) {
        if (error != nullptr) *error = e.what();
        return nlohmann::json::object();
    }
    if (!root.is_object()) return nlohmann::json::object();
    return root;
}

}  // namespace

std::filesystem::path planner_profile_cache_path() {
    if (const char* home = std::getenv("HOME")) {
        if (home[0] != '\0') {
            return std::filesystem::path(home) / ".cache" / "pie" /
                   "cuda_memory_profiles.json";
        }
    }
    return {};
}

PlannerProfileKey make_planner_profile_key(const cudaDeviceProp& prop,
                                           const HfConfig& hf,
                                           int tp_size,
                                           const KvCacheFormat& kv_format) {
    PlannerProfileKey k;
    k.gpu_name = prop.name;
    k.compute_major = prop.major;
    k.compute_minor = prop.minor;
    k.sm_count = prop.multiProcessorCount;
    k.kv_cache_dtype = kv_format.name;
    k.tp_size = tp_size;
    k.model_type = hf.model_type;
    k.hidden_size = hf.hidden_size;
    k.num_hidden_layers = hf.num_hidden_layers;
    k.num_attention_heads = hf.num_attention_heads;
    k.num_key_value_heads = hf.num_key_value_heads;
    k.head_dim = hf.head_dim_kernel;
    return k;
}

std::optional<PlannerProfileShape> planner_profile_cache_lookup(
    const PlannerProfileKey& key,
    std::string* error) {
    const nlohmann::json root = read_document(planner_profile_cache_path(),
                                              error);
    const auto entries = root.find("entries");
    if (entries == root.end() || !entries->is_array()) return std::nullopt;
    for (const auto& entry : *entries) {
        if (!entry.is_object()) continue;
        const auto stored_key = entry.find("key");
        const auto stored_plan = entry.find("plan");
        if (stored_key == entry.end() || stored_plan == entry.end()) continue;
        if (!stored_key->is_object() || !stored_plan->is_object()) continue;
        if (!key_matches(*stored_key, key)) continue;
        PlannerProfileShape shape;
        shape.policy_profile =
            stored_plan->value("policy_profile", std::string{});
        shape.kv_page_size = stored_plan->value("kv_page_size", 0);
        shape.max_forward_tokens =
            stored_plan->value("max_forward_tokens", 0);
        shape.max_forward_requests =
            stored_plan->value("max_forward_requests", 0);
        return shape;
    }
    return std::nullopt;
}

bool planner_profile_cache_store(
    const PlannerProfileKey& key,
    const PlannerProfileShape& shape,
    const std::vector<PlannerShapeSample>& samples,
    std::string* error) {
    const auto path = planner_profile_cache_path();
    if (path.empty()) {
        if (error != nullptr) *error = "HOME is unset";
        return false;
    }

    nlohmann::json root = read_document(path, nullptr);
    if (!root.contains("entries") || !root["entries"].is_array()) {
        root["entries"] = nlohmann::json::array();
    }

    nlohmann::json plan = nlohmann::json::object();
    if (!shape.policy_profile.empty()) {
        plan["policy_profile"] = shape.policy_profile;
    }
    if (shape.kv_page_size > 0) plan["kv_page_size"] = shape.kv_page_size;
    if (shape.max_forward_tokens > 0) {
        plan["max_forward_tokens"] = shape.max_forward_tokens;
    }
    if (shape.max_forward_requests > 0) {
        plan["max_forward_requests"] = shape.max_forward_requests;
    }

    nlohmann::json measured = nlohmann::json::array();
    for (const auto& s : samples) {
        measured.push_back(nlohmann::json{
            {"max_forward_tokens", s.max_forward_tokens},
            {"max_forward_requests", s.max_forward_requests},
            {"tokens_per_request", s.tokens_per_request},
            {"step_ms", s.step_ms},
            {"step_ms_stddev", s.step_ms_stddev},
            {"tokens_per_s", s.tokens_per_s},
        });
    }

    nlohmann::json entry = nlohmann::json::object();
    entry["key"] = key_to_json(key);
    entry["plan"] = plan;
    entry["measured"] = measured;
    entry["measured_at"] = static_cast<std::int64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());

    auto& entries = root["entries"];
    bool replaced = false;
    for (auto& existing : entries) {
        if (!existing.is_object()) continue;
        const auto stored_key = existing.find("key");
        if (stored_key == existing.end() || !stored_key->is_object()) continue;
        if (!key_matches(*stored_key, key)) continue;
        existing = entry;
        replaced = true;
        break;
    }
    if (!replaced) entries.push_back(entry);

    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
        if (error != nullptr) *error = ec.message();
        return false;
    }

    // Rename is atomic within a directory, so a reader racing the write sees
    // one whole document or the other — never a half-serialised one.
    const auto tmp = path.parent_path() /
                     (path.filename().string() + ".tmp." +
                      std::to_string(static_cast<long>(::getpid())));
    try {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) {
            if (error != nullptr) *error = "cannot open " + tmp.string();
            return false;
        }
        out << root.dump(2) << "\n";
        out.flush();
        if (!out) {
            if (error != nullptr) *error = "write failed: " + tmp.string();
            return false;
        }
    } catch (const std::exception& e) {
        if (error != nullptr) *error = e.what();
        std::filesystem::remove(tmp, ec);
        return false;
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        if (error != nullptr) *error = ec.message();
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

}  // namespace pie_cuda_driver
