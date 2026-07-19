#include "model/qwen3_5/qwen3_5_config.hpp"

#include <algorithm>
#include <cstdlib>

namespace pie_cuda_driver::model {

int qwen35_small_spec_graph_tokens() {
    static const int tokens = [] {
        const char* v = std::getenv("PIE_QWEN35_SPEC_VERIFY_GRAPH_N");
        if (v == nullptr || v[0] == '\0') return 17;
        return std::clamp(std::atoi(v), 0, 64);
    }();
    return tokens;
}

bool qwen35_forward_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_FORWARD_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

int qwen35_mtp_draft_position_offset() {
    static const int offset = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_POSITION_OFFSET");
        if (v == nullptr || v[0] == '\0') return 0;
        return std::clamp(std::atoi(v), 0, 2);
    }();
    return offset;
}

bool qwen35_mtp_fused_gemv_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_FUSED_GEMV");
        if (v == nullptr || v[0] == '\0') return false;
        return v[0] != '0';
    }();
    return enabled;
}

bool qwen35_mtp_prefix_global_cache() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_QWEN35_MTP_PREFIX_GLOBAL");
        if (v == nullptr || v[0] == '\0') return true;
        return v[0] != '0';
    }();
    return enabled;
}

}  // namespace pie_cuda_driver::model
