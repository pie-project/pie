// Differential oracle for driver-cuda-new's store::profile_cache.
//
// Unlike the store oracle this extracts nothing: it compiles the REAL
// `store/planner_profile_cache.cpp` verbatim and drives it. Only its two
// external inputs are stood in for -- `cache_dir()` (an engine config global)
// and the two structs `make_planner_profile_key` copies fields out of. The
// logic under test is untouched.
//
// The transcript is line-oriented and deterministic. Every case prints its
// own name so a diff names the failing behaviour rather than a line number.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.hpp"
#include "model/config.hpp"
#include "store/kv_cache_format.hpp"
#include "store/planner_profile_cache.hpp"

using namespace pie_cuda_driver;

namespace {

std::filesystem::path g_root;

void put(const std::string& text) {
    const auto p = planner_profile_cache_path();
    std::filesystem::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::trunc);
    out << text;
}

void clear() {
    const auto p = planner_profile_cache_path();
    std::error_code ec;
    std::filesystem::remove(p, ec);
}

// Print a lookup's full outcome, including the exception the header promises
// cannot happen. Catching here is what makes the C++'s real behaviour
// observable instead of a crash.
void show_lookup(const char* label, const PlannerProfileKey& key) {
    std::string err;
    try {
        const auto shape = planner_profile_cache_lookup(key, &err);
        if (shape.has_value()) {
            std::printf("%s|hit|%s|%d|%d|%d|%zu|err=%s\n", label,
                        shape->policy_profile.c_str(), shape->kv_page_size,
                        shape->max_forward_tokens, shape->max_forward_requests,
                        shape->budget_bytes, err.c_str());
        } else {
            std::printf("%s|miss|err=%s\n", label, err.c_str());
        }
    } catch (const std::exception& e) {
        std::printf("%s|THROWS|%s\n", label, e.what());
    }
}

PlannerProfileKey key_a() {
    PlannerProfileKey k;
    k.gpu_name = "NVIDIA L40S";
    k.compute_major = 8;
    k.compute_minor = 9;
    k.sm_count = 142;
    k.kv_cache_dtype = "bf16";
    k.tp_size = 1;
    k.model_type = "llama";
    k.hidden_size = 4096;
    k.num_hidden_layers = 32;
    k.num_attention_heads = 32;
    k.num_key_value_heads = 8;
    k.head_dim = 128;
    return k;
}

std::string doc_for(const PlannerProfileKey& k, const std::string& plan,
                    const std::string& version = "2") {
    nlohmann::json key = {
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
    return "{\"version\":" + version + ",\"entries\":[{\"key\":" + key.dump() +
           ",\"plan\":" + plan + "}]}";
}

}  // namespace

int main() {
    g_root = std::filesystem::temp_directory_path() / "pie_pc_oracle";
    std::error_code ec;
    std::filesystem::remove_all(g_root, ec);
    std::filesystem::create_directories(g_root);

    // --- path derivation ---------------------------------------------------
    // The three sources in priority order, plus the empty-string cases that
    // decide whether an exported-but-empty variable counts as set.
    ::setenv("XDG_CACHE_HOME", "/xdg", 1);
    ::setenv("HOME", "/home/u", 1);
    mutable_cache_dir() = "/cfg";
    std::printf("path|cfg|%s\n", planner_profile_cache_path().c_str());
    mutable_cache_dir().clear();
    std::printf("path|xdg|%s\n", planner_profile_cache_path().c_str());
    ::setenv("XDG_CACHE_HOME", "", 1);
    std::printf("path|xdg_empty|%s\n", planner_profile_cache_path().c_str());
    ::unsetenv("XDG_CACHE_HOME");
    std::printf("path|home|%s\n", planner_profile_cache_path().c_str());
    ::setenv("HOME", "", 1);
    std::printf("path|home_empty|%s\n", planner_profile_cache_path().c_str());
    ::unsetenv("HOME");
    std::printf("path|none|%s|empty=%d\n", planner_profile_cache_path().c_str(),
                static_cast<int>(planner_profile_cache_path().empty()));

    // --- make_planner_profile_key -----------------------------------------
    {
        cudaDeviceProp prop{};
        std::snprintf(prop.name, sizeof prop.name, "NVIDIA L40S");
        prop.major = 8;
        prop.minor = 9;
        prop.multiProcessorCount = 142;
        HfConfig hf;
        hf.model_type = "llama";
        hf.hidden_size = 4096;
        hf.num_hidden_layers = 32;
        hf.num_attention_heads = 32;
        hf.num_key_value_heads = 8;
        hf.head_dim_kernel = 128;
        KvCacheFormat fmt;
        fmt.name = "fp8_e4m3";
        for (int tp : {1, 2, 8}) {
            const auto k = make_planner_profile_key(prop, hf, tp, fmt);
            std::printf("key|%s|%d|%d|%d|%s|%d|%s|%d|%d|%d|%d|%d\n",
                        k.gpu_name.c_str(), k.compute_major, k.compute_minor,
                        k.sm_count, k.kv_cache_dtype.c_str(), k.tp_size,
                        k.model_type.c_str(), k.hidden_size,
                        k.num_hidden_layers, k.num_attention_heads,
                        k.num_key_value_heads, k.head_dim);
        }
    }

    // --- the budget publish/read pair -------------------------------------
    std::printf("budget|initial|%zu\n", planner_budget_bytes());
    for (std::size_t b : {std::size_t{0}, std::size_t{1},
                          std::size_t{42} * 1024 * 1024 * 1024}) {
        set_planner_budget_bytes(b);
        std::printf("budget|set|%zu|%zu\n", b, planner_budget_bytes());
    }

    // Everything below writes into a real directory.
    mutable_cache_dir() = g_root.string();

    // --- lookup over crafted documents ------------------------------------
    const auto k = key_a();

    clear();
    show_lookup("lookup|absent", k);

    put("");
    show_lookup("lookup|empty_file", k);
    put("not json at all");
    show_lookup("lookup|garbage", k);
    put("[1,2,3]");
    show_lookup("lookup|array_root", k);
    put("{}");
    show_lookup("lookup|no_entries", k);
    put("{\"entries\":{}}");
    show_lookup("lookup|entries_not_array", k);
    put("{\"entries\":[]}");
    show_lookup("lookup|entries_empty_no_version", k);
    put("{\"version\":1,\"entries\":[]}");
    show_lookup("lookup|entries_empty_wrong_version", k);
    put(doc_for(k, "{}", "1"));
    show_lookup("lookup|wrong_version_with_entry", k);
    put("{\"version\":\"2\",\"entries\":[]}");
    show_lookup("lookup|version_string", k);
    put("{\"version\":null,\"entries\":[]}");
    show_lookup("lookup|version_null", k);
    put(doc_for(k, "{}", "2.0"));
    show_lookup("lookup|version_float", k);
    put(doc_for(k, "{}", "true"));
    show_lookup("lookup|version_bool", k);
    put(doc_for(k, "{}", "2.9"));
    show_lookup("lookup|version_float_truncates", k);

    // Plan-field types. The header says a corrupt cache degrades to "no
    // measurement"; several of these do not.
    const char* plans[] = {
        "{}",
        R"({"policy_profile":"throughput","kv_page_size":16,"max_forward_tokens":8192,"max_forward_requests":256,"budget_bytes":42949672960})",
        R"({"policy_profile":7})",
        R"({"policy_profile":null})",
        R"({"kv_page_size":16.9})",
        R"({"kv_page_size":"16"})",
        R"({"kv_page_size":true})",
        R"({"kv_page_size":null})",
        R"({"kv_page_size":-5})",
        R"({"budget_bytes":-1})",
        R"({"budget_bytes":1.5})",
        R"({"budget_bytes":null})",
        R"({"budget_bytes":true})",
        R"({"policy_profile":true})",
        R"({"max_forward_requests":false})",
        R"({"budget_bytes":18446744073709551615})",
        R"({"max_forward_tokens":2147483648})",
    };
    for (const char* plan : plans) {
        put(doc_for(k, plan));
        show_lookup((std::string("lookup|plan|") + plan).c_str(), k);
    }

    // Key matching: every field wrong one at a time, plus the type-strictness
    // cases that make a stored 132.0 fail to match 132.
    {
        const char* mutations[] = {
            R"("gpu_name":"NVIDIA L40")",  R"("compute_major":9)",
            R"("compute_minor":0)",        R"("sm_count":141)",
            R"("sm_count":142.0)",         R"("sm_count":"142")",
            R"("kv_cache_dtype":"fp8")",   R"("tp_size":2)",
            R"("model_type":"qwen3")",     R"("hidden_size":4097)",
            R"("num_hidden_layers":33)",   R"("num_attention_heads":31)",
            R"("num_key_value_heads":4)",  R"("head_dim":64)",
        };
        for (const char* mut : mutations) {
            std::string doc = doc_for(k, R"({"kv_page_size":16})");
            // Splice the mutation in ahead of the original field so it wins
            // (nlohmann keeps the last occurrence).
            const auto pos = doc.find("\"key\":{") + 7;
            doc.insert(pos, std::string(mut) + ",");
            // The mutated field appears twice; nlohmann's map keeps the LAST,
            // so append instead.
            doc = doc_for(k, R"({"kv_page_size":16})");
            const auto end = doc.find("},\"plan\"");
            doc.insert(end, std::string(",") + mut);
            put(doc);
            show_lookup((std::string("lookup|key|") + mut).c_str(), k);
        }
        // A missing key field is a mismatch, not a wildcard.
        std::string doc = doc_for(k, R"({"kv_page_size":16})");
        const auto p = doc.find(R"("sm_count":142,)");
        doc.erase(p, std::strlen(R"("sm_count":142,)"));
        put(doc);
        show_lookup("lookup|key|sm_count_missing", k);
    }

    // Entry shape: things that must be skipped rather than matched.
    put(R"({"version":2,"entries":[1,2,"x",null,{},{"key":{}},{"plan":{}},{"key":[],"plan":{}}]})");
    show_lookup("lookup|entries_junk", k);

    // First match wins even when a later entry also matches.
    {
        std::string a = doc_for(k, R"({"kv_page_size":16})");
        std::string b = doc_for(k, R"({"kv_page_size":32})");
        const auto entry_b = b.substr(b.find("[{") + 1, b.rfind("}]") - b.find("[{"));
        a.insert(a.rfind("}]") + 1, "," + entry_b);
        put(a);
        show_lookup("lookup|two_matches", k);
    }

    // --- store -------------------------------------------------------------
    // The exact bytes are the contract: two implementations read-merge-rewrite
    // this file, so a difference in key order or float formatting would make
    // every write a spurious whole-file diff.
    // `measured_at` is a wall-clock stamp taken inside the function under
    // test, so it can never agree with the Rust's injected one. It is zeroed
    // HERE rather than in a post-processing pass so that the `bytes=` count
    // below still means something -- a 10-digit stamp versus a 1-digit one
    // would otherwise make every byte count differ for an uninteresting
    // reason, and the count is what catches stray whitespace.
    auto zero_measured_at = [](std::string t) {
        const std::string k = "\"measured_at\":";
        for (std::size_t at = t.find(k); at != std::string::npos;
             at = t.find(k, at + k.size() + 1)) {
            std::size_t b = at + k.size();
            while (b < t.size() && (t[b] == ' ' || t[b] == '-')) ++b;
            std::size_t e = b;
            while (e < t.size() && std::isdigit(static_cast<unsigned char>(t[e]))) ++e;
            t.replace(at, e - at, k + "0");
        }
        return t;
    };

    auto dump_file = [&](const char* label) {
        std::ifstream in(planner_profile_cache_path());
        std::string all((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
        all = zero_measured_at(std::move(all));
        std::printf("store|%s|bytes=%zu\n%s<<END>>\n", label, all.size(),
                    all.c_str());
    };

    clear();
    {
        PlannerProfileShape s;
        s.policy_profile = "throughput";
        s.kv_page_size = 16;
        s.max_forward_tokens = 8192;
        s.max_forward_requests = 256;
        s.budget_bytes = 42949672960ULL;
        std::vector<PlannerShapeSample> samples;
        for (int i = 0; i < 3; ++i) {
            PlannerShapeSample smp;
            smp.max_forward_tokens = 1024 << i;
            smp.max_forward_requests = 32 * (i + 1);
            smp.tokens_per_request = 7 + i;
            // Values chosen to exercise the float formatter, including one
            // where Grisu2 and shortest-round-trip disagree.
            smp.step_ms = i == 0 ? 46934.815584012416 : (1.0 / (i + 3));
            smp.step_ms_stddev = i == 1 ? 0.0 : 1e-7 * (i + 1);
            smp.tokens_per_s = i == 2 ? 1e21 : 12345.0 * (i + 1);
            samples.push_back(smp);
        }
        std::string err;
        const bool ok = planner_profile_cache_store(k, s, samples, &err);
        std::printf("store|first|ok=%d|err=%s\n", static_cast<int>(ok),
                    err.c_str());
        dump_file("first");
    }

    // A second key must be appended, not replace the first.
    {
        auto k2 = key_a();
        k2.tp_size = 2;
        PlannerProfileShape s;
        s.policy_profile = "latency";
        s.max_forward_tokens = 4096;
        std::string err;
        planner_profile_cache_store(k2, s, {}, &err);
        dump_file("second_key");
    }

    // The same key must replace in place.
    {
        PlannerProfileShape s;
        s.kv_page_size = 32;
        std::string err;
        planner_profile_cache_store(k, s, {}, &err);
        dump_file("replace_first");
    }

    // Every zero/empty field is omitted, so a partial sweep pins nothing else.
    clear();
    {
        PlannerProfileShape s;
        std::string err;
        planner_profile_cache_store(k, s, {}, &err);
        dump_file("all_defaults");
    }

    // A corrupt document is discarded, not preserved.
    put("{ this is not json");
    {
        PlannerProfileShape s;
        s.kv_page_size = 8;
        std::string err;
        const bool ok = planner_profile_cache_store(k, s, {}, &err);
        std::printf("store|over_corrupt|ok=%d|err=%s\n", static_cast<int>(ok),
                    err.c_str());
        dump_file("over_corrupt");
    }

    // Unknown fields and unrelated entries survive a merge.
    put(R"({"version":2,"note":"kept","entries":[{"key":{"gpu_name":"other"},"plan":{"kv_page_size":4},"extra":[1,2.5,"x"]}]})");
    {
        PlannerProfileShape s;
        s.kv_page_size = 64;
        std::string err;
        planner_profile_cache_store(k, s, {}, &err);
        dump_file("merge_preserves");
    }

    // Strings that need escaping, in a field the vendor controls.
    {
        auto k3 = key_a();
        // The literal is SPLIT deliberately. C++'s \x consumes hex digits
        // greedily, so "\x01E" is the single character 0x1E, not 0x01
        // followed by 'E' -- which is not what this case means to test.
        k3.gpu_name = "A\"B\\C\tD\x01" "E/F\xc3\xa9";
        PlannerProfileShape s;
        s.policy_profile = "p";
        std::string err;
        planner_profile_cache_store(k3, s, {}, &err);
        dump_file("escapes");
    }

    std::filesystem::remove_all(g_root, ec);
    return 0;
}
