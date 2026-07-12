#include <cstdio>
#include <string>
#include <vector>

#include "pie_native/storage_program.hpp"

int main() {
    const std::string json = R"JSON({
      "version": 4,
      "compiler_version": 1,
      "target": {
        "backend": "Metal",
        "tp_rank": 0,
        "tp_size": 1,
        "max_tile_bytes": 67108864,
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
    auto program = pie_weight_loader::StorageProgram::deserialize(
        std::span<const std::uint8_t>(data, json.size()), 1);
    const auto view = program.view();
    if (program.backend() != pie_weight_loader::PieLoaderBackendKind::Metal ||
        program.preferred_alignment() != 256 ||
        view.sources.len != 1 ||
        pie_weight_loader::cpp::bytes_to_string(view.sources.ptr[0].name) !=
            "lm_head.weight") {
        std::fprintf(stderr, "decoded StorageProgram fields do not match\n");
        return 1;
    }
    try {
        (void)pie_weight_loader::StorageProgram::deserialize(
            std::span<const std::uint8_t>(data, json.size()), 2);
        std::fprintf(stderr, "compiler version mismatch was accepted\n");
        return 1;
    } catch (const std::exception&) {
    }

    std::string bad = json;
    bad.replace(bad.find("\"version\": 4"), 12, "\"version\": 3");
    try {
        const auto* bad_data =
            reinterpret_cast<const std::uint8_t*>(bad.data());
        (void)pie_weight_loader::StorageProgram::deserialize(
            std::span<const std::uint8_t>(bad_data, bad.size()), 1);
        std::fprintf(stderr, "version mismatch was accepted\n");
        return 1;
    } catch (const std::exception&) {
    }
    return 0;
}
