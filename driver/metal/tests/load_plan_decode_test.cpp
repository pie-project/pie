#include <cstdio>
#include <string>
#include <vector>

#include "pie_native/load_plan.hpp"

int main() {
    const std::string json = R"JSON({
      "version": 5,
      "compiler_version": 1,
      "target": {
        "backend": "Metal",
        "tp_rank": 0,
        "tp_size": 1,
        "max_tile_bytes": 67108864,
        "tile_map_mask": 0,
        "preferred_alignment": 256,
        "mxfp4_moe": "RoutedDecode",
        "native_mxfp4_moe": false
      },
      "optimizer": {"passes": []},
      "sources": [{
        "id": 0,
        "name": "lm_head.weight",
        "file_id": 0,
        "file_offset": 128,
        "span_bytes": 16,
        "shape": [1, 4],
        "encoding": {"Raw": "U32"}
      }],
      "tensors": [],
      "buffers": [],
      "instrs": [],
      "schedule": [],
      "memory": {
        "persistent_bytes": 0,
        "temporary_peak_bytes": 0,
        "transform_scratch_peak_bytes": 0,
        "checkpoint_read_bytes": 0,
        "device_write_bytes": 0
      }
    })JSON";
    const auto* data = reinterpret_cast<const std::uint8_t*>(json.data());
    auto plan = pie_load_planner::LoadPlan::deserialize(
        std::span<const std::uint8_t>(data, json.size()), 1);
    const auto view = plan.view();
    if (plan.backend() != pie_load_planner::PieLoaderBackendKind::Metal ||
        plan.tile_map_mask() != pie_load_planner::kMetalTileMapMask ||
        plan.preferred_alignment() != 256 ||
        view.sources.len != 1 ||
        pie_load_planner::cpp::bytes_to_string(view.sources.ptr[0].name) !=
            "lm_head.weight") {
        std::fprintf(stderr, "decoded LoadPlan fields do not match\n");
        return 1;
    }
    try {
        (void)pie_load_planner::LoadPlan::deserialize(
            std::span<const std::uint8_t>(data, json.size()), 2);
        std::fprintf(stderr, "compiler version mismatch was accepted\n");
        return 1;
    } catch (const std::exception&) {
    }

    auto must_reject = [&](std::string malformed, const char* message) {
        try {
            const auto* malformed_data =
                reinterpret_cast<const std::uint8_t*>(malformed.data());
            (void)pie_load_planner::LoadPlan::deserialize(
                std::span<const std::uint8_t>(
                    malformed_data, malformed.size()),
                1);
            std::fprintf(stderr, "%s was accepted\n", message);
            return false;
        } catch (const std::exception&) {
            return true;
        }
    };
    std::string unknown_backend = json;
    unknown_backend.replace(
        unknown_backend.find("\"Metal\""), 7, "\"FutureGpu\"");
    if (!must_reject(unknown_backend, "unknown backend")) return 1;
    std::string unknown_policy = json;
    unknown_policy.replace(
        unknown_policy.find("\"RoutedDecode\""), 14, "\"FuturePolicy\"");
    if (!must_reject(unknown_policy, "unknown MXFP4 policy")) return 1;
    std::string negative_id = json;
    negative_id.replace(negative_id.find("\"id\": 0"), 7, "\"id\": -1");
    if (!must_reject(negative_id, "negative id")) return 1;
    std::string oversized_id = json;
    oversized_id.replace(
        oversized_id.find("\"id\": 0"), 7, "\"id\": 4294967296");
    if (!must_reject(oversized_id, "oversized id")) return 1;

    std::string bad = json;
    bad.replace(bad.find("\"version\": 5"), 12, "\"version\": 3");
    try {
        const auto* bad_data =
            reinterpret_cast<const std::uint8_t*>(bad.data());
        (void)pie_load_planner::LoadPlan::deserialize(
            std::span<const std::uint8_t>(bad_data, bad.size()), 1);
        std::fprintf(stderr, "version mismatch was accepted\n");
        return 1;
    } catch (const std::exception&) {
    }
    return 0;
}
