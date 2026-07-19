#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pie_driver_abi.h>

#include "pipeline/m1_runtime.hpp"
#include "pipeline/m1_codegen.hpp"
#include "pipeline/descriptor_resolve.hpp"
#include "observability.hpp"
#include "support/ptib_v2_plan.hpp"

using namespace pie::metal;
using namespace pie::metal::batch;
using namespace pie::metal::pipeline;

namespace {

int g_pass = 0;
int g_fail = 0;

void expect(bool condition, const std::string& message) {
    if (condition) {
        ++g_pass;
        std::printf("  PASS  %s\n", message.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", message.c_str());
    }
}

std::vector<std::uint8_t> hex_bytes(const std::string& text) {
    std::vector<std::uint8_t> bytes;
    for (std::size_t index = 0; index + 1 < text.size(); index += 2) {
        bytes.push_back(static_cast<std::uint8_t>(
            std::stoul(text.substr(index, 2), nullptr, 16)));
    }
    return bytes;
}

std::uint32_t read_u32(
        const std::vector<std::uint8_t>& bytes,
        std::size_t offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    }

    void write_u32(
        std::vector<std::uint8_t>& bytes,
        std::size_t offset,
        std::uint32_t value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }

    std::size_t find_ptrp(const std::vector<std::uint8_t>& sidecar) {
        for (std::size_t offset = 0; offset + 4 <= sidecar.size(); ++offset) {
            if (std::memcmp(sidecar.data() + offset, "PTRP", 4) == 0) {
                return offset;
            }
        }
        return sidecar.size();
    }

    enum class RegionIndexKind {
        Node,
        Input,
        Output,
        SinkChannel,
        SinkValue,
    };

struct RegionIndexMutation {
    bool changed = false;
    std::size_t offset = 0;
    std::uint32_t original = 0;
    std::uint32_t upper_bound = 0;
};

struct PtrpCursor {
    std::vector<std::uint8_t>& bytes;
    std::size_t offset = 0;

    bool skip(std::size_t count) {
        if (count > bytes.size() - std::min(offset, bytes.size())) {
            return false;
        }
        offset += count;
        return true;
    }

    bool u8(std::uint8_t& value) {
        if (offset >= bytes.size()) return false;
        value = bytes[offset++];
        return true;
    }

    bool u16(std::uint16_t& value) {
        if (bytes.size() - std::min(offset, bytes.size()) < 2) {
            return false;
        }
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        offset += 2;
        return true;
    }

    bool u32(std::uint32_t& value) {
        if (bytes.size() - std::min(offset, bytes.size()) < 4) {
            return false;
        }
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        offset += 4;
        return true;
    }

    bool skip_records(std::uint32_t count, std::size_t width) {
        if (width == 0 ||
            count >
                (bytes.size() -
                 std::min(offset, bytes.size())) /
                    width) {
            return false;
        }
        return skip(static_cast<std::size_t>(count) * width);
    }
};

RegionIndexMutation corrupt_region_index(
    std::vector<std::uint8_t>& sidecar,
    RegionIndexKind target) {
    const std::size_t plan = find_ptrp(sidecar);
    RegionIndexMutation mutation;
    if (plan == sidecar.size()) return mutation;
    PtrpCursor cursor{sidecar, plan};
    std::uint16_t plan_version = 0;
    std::uint16_t compiler_version = 0;
    std::uint8_t stage = 0;
    std::uint32_t signature_len = 0;
    if (!cursor.skip(4) || !cursor.u16(plan_version) ||
        !cursor.u16(compiler_version) || !cursor.u8(stage) ||
        !cursor.skip(8) || !cursor.u32(signature_len) ||
        plan_version != PTIR_REGION_PLAN_VERSION ||
        compiler_version != PTIR_COMPILER_VERSION ||
        stage > PTIR_STAGE_EPILOGUE ||
        !cursor.skip(signature_len)) {
        return mutation;
    }
    std::uint32_t channel_count = 0;
    if (!cursor.u32(channel_count) ||
        !cursor.skip_records(channel_count, 4)) {
        return mutation;
    }
    std::uint32_t name_count = 0;
    if (!cursor.u32(name_count)) return mutation;
    for (std::uint32_t name = 0; name < name_count; ++name) {
        std::uint16_t length = 0;
        if (!cursor.u16(length) || !cursor.skip(length)) {
            return mutation;
        }
    }
    std::uint32_t op_count = 0;
    if (!cursor.u32(op_count)) return mutation;
    for (std::uint32_t op = 0; op < op_count; ++op) {
        std::uint32_t op_len = 0;
        std::uint32_t sources = 0;
        if (!cursor.u32(op_len) || !cursor.skip(op_len) ||
            !cursor.u32(sources) ||
            !cursor.skip_records(sources, 4)) {
            return mutation;
        }
    }
    std::uint32_t value_count = 0;
    if (!cursor.u32(value_count)) return mutation;
    for (std::uint32_t value = 0; value < value_count; ++value) {
        std::uint8_t dtype = 0;
        std::uint8_t rank = 0;
        if (!cursor.u8(dtype) || dtype > PTIR_DT_BOOL ||
            !cursor.u8(rank) || rank > 4) {
            return mutation;
        }
        for (std::uint8_t dimension = 0; dimension < rank; ++dimension) {
            std::uint8_t tag = 0;
            if (!cursor.u8(tag) ||
                !cursor.skip(tag == 0 ? 4 : tag == 1 ? 1 : 0) ||
                tag > 1) {
                return mutation;
            }
        }
        std::uint8_t domain = 0;
        if (!cursor.u8(domain) || domain > 7) return mutation;
    }
    for (int partition = 0; partition < 2; ++partition) {
        std::uint8_t kind = 0;
        std::uint8_t fallback = 0;
        std::uint32_t region_count = 0;
        if (!cursor.u8(kind) || !cursor.u8(fallback) ||
            !cursor.u32(region_count) || kind != partition ||
            fallback > 1) {
            return mutation;
        }
        for (std::uint32_t region = 0; region < region_count; ++region) {
            std::uint8_t region_kind = 0;
            std::uint8_t library = 0;
            std::uint8_t schedule = 0;
            if (!cursor.u8(region_kind) || !cursor.u8(library) ||
                !cursor.u8(schedule) || region_kind > 1 ||
                (region_kind == 1 &&
                 library > PTIR_LIBRARY_SECOND_PARTY) ||
                schedule > PTIR_SCHEDULE_LIBRARY) {
                return mutation;
            }
            for (RegionIndexKind vector_kind : {
                     RegionIndexKind::Node,
                     RegionIndexKind::Input,
                     RegionIndexKind::Output,
                 }) {
                std::uint32_t count = 0;
                if (!cursor.u32(count)) return mutation;
                for (std::uint32_t index = 0; index < count; ++index) {
                    const std::size_t index_offset = cursor.offset;
                    std::uint32_t original = 0;
                    if (!cursor.u32(original)) return mutation;
                    if (vector_kind == target) {
                        const std::uint32_t upper_bound =
                            vector_kind == RegionIndexKind::Node
                                ? op_count
                                : value_count;
                        write_u32(
                            sidecar, index_offset, upper_bound);
                        return {
                            .changed = true,
                            .offset = index_offset,
                            .original = original,
                            .upper_bound = upper_bound,
                        };
                    }
                }
            }
            std::uint32_t sinks = 0;
            if (!cursor.u32(sinks)) return mutation;
            for (std::uint32_t sink = 0; sink < sinks; ++sink) {
                const std::size_t channel_offset = cursor.offset;
                std::uint32_t channel = 0;
                if (!cursor.u32(channel)) return mutation;
                if (target == RegionIndexKind::SinkChannel) {
                    write_u32(
                        sidecar, channel_offset, channel_count);
                    return {
                        .changed = true,
                        .offset = channel_offset,
                        .original = channel,
                        .upper_bound = channel_count,
                    };
                }
                const std::size_t value_offset = cursor.offset;
                std::uint32_t value = 0;
                if (!cursor.u32(value)) return mutation;
                if (target == RegionIndexKind::SinkValue) {
                    write_u32(
                        sidecar, value_offset, value_count);
                    return {
                        .changed = true,
                        .offset = value_offset,
                        .original = value,
                        .upper_bound = value_count,
                    };
                }
            }
        }
    }
    return mutation;
}

bool load_golden(
    const std::string& name,
    std::vector<std::uint8_t>& container,
    std::vector<std::uint8_t>& sidecar) {
    std::ifstream input(
        std::filesystem::path(PIE_PTIR_GOLDEN_DIR) / (name + ".txt"));
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("container: ", 0) == 0 && container.empty()) {
            container = hex_bytes(line.substr(11));
        } else if (line.rfind("sidecar: ", 0) == 0 && sidecar.empty()) {
            sidecar = hex_bytes(line.substr(9));
        }
    }
    return !container.empty() && !sidecar.empty();
}

std::uint16_t bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

std::uint32_t float_bits(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

std::vector<std::uint8_t> last_put_container() {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION);
    put_u16(out, 0);
    put_u32(out, 0);
    put_u32(out, 1);
    put_u32(out, 0);
    put_u32(out, 1);
    out.push_back(PTIR_DT_U32);
    out.push_back(1);
    put_u32(out, 1);
    put_u32(out, 1);
    out.push_back(PTIR_HOST_NONE);
    out.push_back(1);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 5);
    out.push_back(PTIR_OP_CHAN_TAKE);
    put_u32(out, 0);
    out.push_back(PTIR_OP_CONST);
    out.push_back(PTIR_DT_U32);
    put_u32(out, 1);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 0);
    put_u32(out, 1);
    out.push_back(PTIR_OP_CONST);
    out.push_back(PTIR_DT_U32);
    put_u32(out, 2);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 0);
    put_u32(out, 2);
    return out;
}

std::vector<std::uint8_t> last_put_sidecar(
    const std::vector<std::uint8_t>& container) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    put_u16(out, 1);
    put_u16(out, 0);
    put_u64(
        out,
        pie_native::ptir::container::fnv1a64(
            container.data(), container.size()));
    put_u32(out, 1);
    out.push_back(PTIR_CHAN_FULL_RING);
    put_u32(out, 1);
    put_u32(out, 0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    out.push_back(PTIR_NEEDS_FULL);
    put_u32(out, 1);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 3);
    for (int value = 0; value < 3; ++value) {
        out.push_back(PTIR_DT_U32);
        out.push_back(value == 1 || value == 2 ? 0 : 1);
        if (value == 0) put_u32(out, 1);
    }
    return upgrade_ptib_v1(container, std::move(out));
}

std::vector<std::uint8_t> put_then_take_container() {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION);
    put_u16(out, 0);
    put_u32(out, 0);
    put_u32(out, 2);
    put_u32(out, 0);
    put_u32(out, 1);
    for (int channel = 0; channel < 2; ++channel) {
        out.push_back(PTIR_DT_U32);
        out.push_back(0);
        put_u32(out, 1);
        out.push_back(channel == 0 ? PTIR_HOST_NONE : PTIR_HOST_READER);
        out.push_back(0);
    }
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 4);
    out.push_back(PTIR_OP_CONST);
    out.push_back(PTIR_DT_U32);
    put_u32(out, 7);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 0);
    put_u32(out, 0);
    out.push_back(PTIR_OP_CHAN_TAKE);
    put_u32(out, 0);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 1);
    put_u32(out, 1);
    return out;
}

std::vector<std::uint8_t> put_then_take_sidecar(
    const std::vector<std::uint8_t>& container) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    put_u16(out, 1);
    put_u16(out, 0);
    put_u64(
        out,
        pie_native::ptir::container::fnv1a64(
            container.data(), container.size()));
    put_u32(out, 2);
    out.insert(out.end(), 2, PTIR_CHAN_FULL_RING);
    put_u32(out, 2);
    for (std::uint32_t channel = 0; channel < 2; ++channel) {
        put_u32(out, channel);
        out.push_back(PTIR_STAGE_EPILOGUE);
        out.push_back(PTIR_NEEDS_EMPTY);
    }
    put_u32(out, 1);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 2);
    for (int value = 0; value < 2; ++value) {
        out.push_back(PTIR_DT_U32);
        out.push_back(0);
    }
    return upgrade_ptib_v1(container, std::move(out));
}

std::vector<std::uint8_t> pre_post_container(
    std::uint32_t prologue_value = 5,
    std::uint32_t channel_count = 2) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION);
    put_u16(out, 0);
    put_u32(out, 0);
    put_u32(out, channel_count);
    put_u32(out, 0);
    put_u32(out, 2);
    for (std::uint32_t channel = 0;
         channel < channel_count;
         ++channel) {
        out.push_back(PTIR_DT_U32);
        out.push_back(0);
        put_u32(out, 1);
        out.push_back(channel == 1 ? PTIR_HOST_READER : PTIR_HOST_NONE);
        out.push_back(0);
    }
    out.push_back(PTIR_STAGE_PROLOGUE);
    put_u32(out, 2);
    out.push_back(PTIR_OP_CONST);
    out.push_back(PTIR_DT_U32);
    put_u32(out, prologue_value);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 0);
    put_u32(out, 0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 3);
    out.push_back(PTIR_OP_INTRINSIC_VAL);
    put_u16(out, PTIR_INTR_LOGITS);
    out.push_back(PTIR_DT_F32);
    out.push_back(1);
    put_u32(out, 1);
    out.push_back(PTIR_OP_CHAN_TAKE);
    put_u32(out, 0);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 1);
    put_u32(out, 1);
    return out;
}

std::vector<std::uint8_t> pre_post_sidecar(
    const std::vector<std::uint8_t>& container,
    std::uint32_t channel_count = 2) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    put_u16(out, 1);
    put_u16(out, 0);
    put_u64(
        out,
        pie_native::ptir::container::fnv1a64(
            container.data(), container.size()));
    put_u32(out, channel_count);
    out.insert(
        out.end(), channel_count, PTIR_CHAN_FULL_RING);
    put_u32(out, 2);
    for (std::uint32_t channel = 0; channel < 2; ++channel) {
        put_u32(out, channel);
        out.push_back(
            channel == 0 ? PTIR_STAGE_PROLOGUE : PTIR_STAGE_EPILOGUE);
        out.push_back(PTIR_NEEDS_EMPTY);
    }
    put_u32(out, 2);
    out.push_back(PTIR_STAGE_PROLOGUE);
    put_u32(out, 1);
    out.push_back(PTIR_DT_U32);
    out.push_back(0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 2);
    out.push_back(PTIR_DT_F32);
    out.push_back(1);
    put_u32(out, 1);
    out.push_back(PTIR_DT_U32);
    out.push_back(0);
    return upgrade_ptib_v1(container, std::move(out));
}

std::vector<std::uint8_t> multistage_sink_base_container() {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION);
    put_u16(out, 0);
    put_u32(out, 0);
    put_u32(out, 2);
    put_u32(out, 0);
    put_u32(out, 2);
    for (int channel = 0; channel < 2; ++channel) {
        out.push_back(PTIR_DT_U32);
        out.push_back(0);
        put_u32(out, 1);
        out.push_back(
            channel == 0 ? PTIR_HOST_NONE : PTIR_HOST_READER);
        out.push_back(0);
    }
    out.push_back(PTIR_STAGE_PROLOGUE);
    put_u32(out, 2);
    out.push_back(PTIR_OP_CONST);
    out.push_back(PTIR_DT_U32);
    put_u32(out, 5);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 0);
    put_u32(out, 0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 3);
    out.push_back(PTIR_OP_INTRINSIC_VAL);
    put_u16(out, PTIR_INTR_LOGITS);
    out.push_back(PTIR_DT_F32);
    out.push_back(1);
    put_u32(out, 16);
    out.push_back(PTIR_OP_CHAN_TAKE);
    put_u32(out, 0);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 1);
    put_u32(out, 1);
    return out;
}

std::vector<std::uint8_t> multistage_sink_base_sidecar(
    const std::vector<std::uint8_t>& container) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    put_u16(out, 1);
    put_u16(out, 0);
    put_u64(
        out,
        pie_native::ptir::container::fnv1a64(
            container.data(), container.size()));
    put_u32(out, 2);
    out.insert(out.end(), 2, PTIR_CHAN_FULL_RING);
    put_u32(out, 2);
    for (std::uint32_t channel = 0; channel < 2; ++channel) {
        put_u32(out, channel);
        out.push_back(
            channel == 0 ? PTIR_STAGE_PROLOGUE : PTIR_STAGE_EPILOGUE);
        out.push_back(PTIR_NEEDS_EMPTY);
    }
    put_u32(out, 2);
    out.push_back(PTIR_STAGE_PROLOGUE);
    put_u32(out, 1);
    out.push_back(PTIR_DT_U32);
    out.push_back(0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 2);
    out.push_back(PTIR_DT_F32);
    out.push_back(1);
    put_u32(out, 16);
    out.push_back(PTIR_DT_U32);
    out.push_back(0);
    return upgrade_ptib_v1(container, std::move(out));
}

std::vector<std::uint8_t> mtp_drafts_container() {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION);
    put_u16(out, 0);
    put_u32(out, 0);
    put_u32(out, 1);
    put_u32(out, 0);
    put_u32(out, 1);
    out.push_back(PTIR_DT_I32);
    out.push_back(1);
    put_u32(out, 2);
    put_u32(out, 1);
    out.push_back(PTIR_HOST_READER);
    out.push_back(0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 2);
    out.push_back(PTIR_OP_INTRINSIC_VAL);
    put_u16(out, PTIR_INTR_MTP_DRAFTS);
    out.push_back(PTIR_DT_I32);
    out.push_back(1);
    put_u32(out, 2);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 0);
    put_u32(out, 0);
    return out;
}

std::vector<std::uint8_t> mtp_drafts_sidecar(
    const std::vector<std::uint8_t>& container) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    put_u16(out, 1);
    put_u16(out, 0);
    put_u64(
        out,
        pie_native::ptir::container::fnv1a64(
            container.data(), container.size()));
    put_u32(out, 1);
    out.push_back(PTIR_CHAN_FULL_RING);
    put_u32(out, 1);
    put_u32(out, 0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    out.push_back(PTIR_NEEDS_EMPTY);
    put_u32(out, 1);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 1);
    out.push_back(PTIR_DT_I32);
    out.push_back(1);
    put_u32(out, 2);
    return upgrade_ptib_v1(container, std::move(out));
}

std::vector<std::uint8_t> semantic_boundary_container(
    bool known = true) {
    using namespace pie::metal::tests;
    const std::string kernel =
        known ? "metal.identity" : "unknown.kernel";
    const std::string sink =
        known ? "metal.discard" : "unknown.sink";
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION);
    put_u16(out, 0);
    put_u32(out, 2);
    put_u32(out, 2);
    put_u32(out, 0);
    put_u32(out, 1);
    put_u16(out, static_cast<std::uint16_t>(kernel.size()));
    out.insert(out.end(), kernel.begin(), kernel.end());
    put_u16(out, static_cast<std::uint16_t>(sink.size()));
    out.insert(out.end(), sink.begin(), sink.end());
    for (int channel = 0; channel < 2; ++channel) {
        out.push_back(PTIR_DT_U32);
        out.push_back(1);
        put_u32(out, 2);
        put_u32(out, 1);
        out.push_back(
            channel == 0 ? PTIR_HOST_NONE : PTIR_HOST_READER);
        out.push_back(channel == 0 ? 1 : 0);
    }
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 4);
    out.push_back(PTIR_OP_CHAN_TAKE);
    put_u32(out, 0);
    out.push_back(PTIR_OP_KERNEL_CALL);
    put_u16(out, 0);
    out.push_back(PTIR_DT_U32);
    out.push_back(1);
    put_u32(out, 2);
    out.push_back(1);
    put_u32(out, 0);
    out.push_back(PTIR_OP_SINK_CALL);
    put_u16(out, 1);
    out.push_back(1);
    put_u32(out, 1);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 1);
    put_u32(out, 1);
    return out;
}

std::vector<std::uint8_t> semantic_boundary_sidecar(
    const std::vector<std::uint8_t>& container) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    put_u16(out, 1);
    put_u16(out, 0);
    put_u64(
        out,
        pie_native::ptir::container::fnv1a64(
            container.data(), container.size()));
    put_u32(out, 2);
    out.insert(out.end(), 2, PTIR_CHAN_FULL_RING);
    put_u32(out, 2);
    for (std::uint32_t channel = 0; channel < 2; ++channel) {
        put_u32(out, channel);
        out.push_back(PTIR_STAGE_EPILOGUE);
        out.push_back(
            channel == 0 ? PTIR_NEEDS_FULL : PTIR_NEEDS_EMPTY);
    }
    put_u32(out, 1);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 2);
    for (int value = 0; value < 2; ++value) {
        out.push_back(PTIR_DT_U32);
        out.push_back(1);
        put_u32(out, 2);
    }
    return upgrade_ptib_v1(container, std::move(out));
}

std::vector<std::uint8_t> signed_zero_container() {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION);
    put_u16(out, 0);
    put_u32(out, 0);
    put_u32(out, 3);
    put_u32(out, 0);
    put_u32(out, 1);
    for (int channel = 0; channel < 3; ++channel) {
        out.push_back(PTIR_DT_F32);
        out.push_back(channel == 0 ? 1 : 0);
        if (channel == 0) put_u32(out, 2);
        put_u32(out, 1);
        out.push_back(channel == 0 ? PTIR_HOST_NONE : PTIR_HOST_READER);
        out.push_back(channel == 0 ? 1 : 0);
    }
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 5);
    out.push_back(PTIR_OP_CHAN_TAKE);
    put_u32(out, 0);
    out.push_back(PTIR_OP_REDUCE_MAX);
    put_u32(out, 0);
    out.push_back(PTIR_OP_REDUCE_MIN);
    put_u32(out, 0);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 1);
    put_u32(out, 1);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 2);
    put_u32(out, 2);
    return out;
}

std::vector<std::uint8_t> signed_zero_sidecar(
    const std::vector<std::uint8_t>& container) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    put_u16(out, 1);
    put_u16(out, 0);
    put_u64(
        out,
        pie_native::ptir::container::fnv1a64(
            container.data(), container.size()));
    put_u32(out, 3);
    out.insert(out.end(), 3, PTIR_CHAN_FULL_RING);
    put_u32(out, 3);
    for (std::uint32_t channel = 0; channel < 3; ++channel) {
        put_u32(out, channel);
        out.push_back(PTIR_STAGE_EPILOGUE);
        out.push_back(
            channel == 0 ? PTIR_NEEDS_FULL : PTIR_NEEDS_EMPTY);
    }
    put_u32(out, 1);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 3);
    out.push_back(PTIR_DT_F32);
    out.push_back(1);
    put_u32(out, 2);
    for (int value = 1; value < 3; ++value) {
        out.push_back(PTIR_DT_F32);
        out.push_back(0);
    }
    return upgrade_ptib_v1(container, std::move(out));
}

std::vector<std::uint8_t> rank3_argmax_container() {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION);
    put_u16(out, 0);
    put_u32(out, 0);
    put_u32(out, 4);
    put_u32(out, 0);
    put_u32(out, 1);
    const std::uint8_t dtypes[] = {
        PTIR_DT_F32, PTIR_DT_F32, PTIR_DT_I32, PTIR_DT_I32};
    for (int channel = 0; channel < 4; ++channel) {
        out.push_back(dtypes[channel]);
        if (channel == 0) {
            out.push_back(3);
            put_u32(out, 2);
            put_u32(out, 2);
            put_u32(out, 2);
        } else if (channel == 1) {
            out.push_back(2);
            put_u32(out, 2);
            put_u32(out, 2);
        } else if (channel == 2) {
            out.push_back(1);
            put_u32(out, 2);
        } else {
            out.push_back(0);
        }
        put_u32(out, 1);
        out.push_back(
            channel == 1 || channel == 3 ? PTIR_HOST_READER
                                         : PTIR_HOST_NONE);
        out.push_back(channel == 0 || channel == 2 ? 1 : 0);
    }
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 6);
    out.push_back(PTIR_OP_CHAN_TAKE);
    put_u32(out, 0);
    out.push_back(PTIR_OP_REDUCE_MAX);
    put_u32(out, 0);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 1);
    put_u32(out, 1);
    out.push_back(PTIR_OP_CHAN_TAKE);
    put_u32(out, 2);
    out.push_back(PTIR_OP_REDUCE_ARGMAX);
    put_u32(out, 2);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 3);
    put_u32(out, 3);
    return out;
}

std::vector<std::uint8_t> rank3_argmax_sidecar(
    const std::vector<std::uint8_t>& container) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    put_u16(out, 1);
    put_u16(out, 0);
    put_u64(
        out,
        pie_native::ptir::container::fnv1a64(
            container.data(), container.size()));
    put_u32(out, 4);
    out.insert(out.end(), 4, PTIR_CHAN_FULL_RING);
    put_u32(out, 4);
    for (std::uint32_t channel = 0; channel < 4; ++channel) {
        put_u32(out, channel);
        out.push_back(PTIR_STAGE_EPILOGUE);
        out.push_back(
            channel == 0 || channel == 2 ? PTIR_NEEDS_FULL
                                         : PTIR_NEEDS_EMPTY);
    }
    put_u32(out, 1);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 4);
    out.push_back(PTIR_DT_F32);
    out.push_back(3);
    put_u32(out, 2);
    put_u32(out, 2);
    put_u32(out, 2);
    out.push_back(PTIR_DT_F32);
    out.push_back(2);
    put_u32(out, 2);
    put_u32(out, 2);
    out.push_back(PTIR_DT_I32);
    out.push_back(1);
    put_u32(out, 2);
    out.push_back(PTIR_DT_I32);
    out.push_back(0);
    return upgrade_ptib_v1(container, std::move(out));
}

std::vector<std::uint8_t> ragged_container() {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION);
    put_u16(out, 0);
    put_u32(out, 0);
    put_u32(out, 1);
    put_u32(out, 0);
    put_u32(out, 1);
    out.push_back(PTIR_DT_I32);
    out.push_back(1);
    put_u32(out, 4);
    put_u32(out, 1);
    out.push_back(PTIR_HOST_READER);
    out.push_back(0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 3);
    out.push_back(PTIR_OP_INTRINSIC_VAL);
    put_u16(out, PTIR_INTR_LOGITS);
    out.push_back(PTIR_DT_F32);
    out.push_back(2);
    put_u32(out, 4);
    put_u32(out, 4);
    out.push_back(PTIR_OP_REDUCE_ARGMAX);
    put_u32(out, 0);
    out.push_back(PTIR_OP_CHAN_PUT);
    put_u32(out, 0);
    put_u32(out, 1);
    return out;
}

std::vector<std::uint8_t> ragged_sidecar(
    const std::vector<std::uint8_t>& container) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'B'};
    put_u16(out, 1);
    put_u16(out, 0);
    put_u64(
        out,
        pie_native::ptir::container::fnv1a64(
            container.data(), container.size()));
    put_u32(out, 1);
    out.push_back(PTIR_CHAN_FULL_RING);
    put_u32(out, 1);
    put_u32(out, 0);
    out.push_back(PTIR_STAGE_EPILOGUE);
    out.push_back(PTIR_NEEDS_EMPTY);
    put_u32(out, 1);
    out.push_back(PTIR_STAGE_EPILOGUE);
    put_u32(out, 2);
    out.push_back(PTIR_DT_F32);
    out.push_back(2);
    put_u32(out, 4);
    put_u32(out, 4);
    out.push_back(PTIR_DT_I32);
    out.push_back(1);
    put_u32(out, 4);
    return upgrade_ptib_v1(container, std::move(out));
}

std::vector<std::uint8_t> topk_container(
    std::uint32_t vocab = 4,
    std::uint32_t k = 2) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P', 'T', 'I', 'R'};
    put_u16(out, PTIR_VERSION); put_u16(out, 0);
    put_u32(out, 0); put_u32(out, 3); put_u32(out, 0); put_u32(out, 1);
    const std::uint8_t dtypes[] = {PTIR_DT_F32, PTIR_DT_F32, PTIR_DT_U32};
    const std::uint32_t lengths[] = {vocab, k, k};
    for (int channel = 0; channel < 3; ++channel) {
        out.push_back(dtypes[channel]); out.push_back(1); put_u32(out, lengths[channel]);
        put_u32(out, 1);
        out.push_back(channel == 0 ? PTIR_HOST_NONE : PTIR_HOST_READER);
        out.push_back(channel == 0 ? 1 : 0);
    }
    out.push_back(PTIR_STAGE_EPILOGUE); put_u32(out, 4);
    out.push_back(PTIR_OP_CHAN_TAKE); put_u32(out, 0);
    out.push_back(PTIR_OP_TOP_K); put_u32(out, 0); put_u32(out, k);
    out.push_back(PTIR_OP_CHAN_PUT); put_u32(out, 1); put_u32(out, 1);
    out.push_back(PTIR_OP_CHAN_PUT); put_u32(out, 2); put_u32(out, 2);
    return out;
}

std::vector<std::uint8_t> topk_sidecar(
    const std::vector<std::uint8_t>& container,
    std::uint32_t vocab = 4,
    std::uint32_t k = 2) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P','T','I','B'};
    put_u16(out,1); put_u16(out,0);
    put_u64(out,pie_native::ptir::container::fnv1a64(container.data(),container.size()));
    put_u32(out,3); out.insert(out.end(),3,PTIR_CHAN_FULL_RING);
    put_u32(out,3);
    for(std::uint32_t c=0;c<3;++c){put_u32(out,c);out.push_back(PTIR_STAGE_EPILOGUE);out.push_back(c==0?PTIR_NEEDS_FULL:PTIR_NEEDS_EMPTY);}
    put_u32(out,1);out.push_back(PTIR_STAGE_EPILOGUE);put_u32(out,3);
    out.push_back(PTIR_DT_F32);out.push_back(1);put_u32(out,vocab);
    out.push_back(PTIR_DT_F32);out.push_back(1);put_u32(out,k);
    out.push_back(PTIR_DT_U32);out.push_back(1);put_u32(out,k);
    return upgrade_ptib_v1(container,std::move(out));
}

std::vector<std::uint8_t> generic_beam_container(
    std::uint32_t beams = 2,
    std::uint32_t vocab = 3,
    std::uint32_t k = 2) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P','T','I','R'};
    put_u16(out,PTIR_VERSION);put_u16(out,0);
    put_u32(out,0);put_u32(out,4);put_u32(out,0);put_u32(out,1);
    const std::uint8_t dtypes[]={
        PTIR_DT_F32,PTIR_DT_U32,PTIR_DT_U32,PTIR_DT_U32};
    for(int c=0;c<4;++c){
        out.push_back(dtypes[c]);
        out.push_back(c==0?2:1);
        put_u32(out,c==0?beams:(c==1?vocab:k));
        if(c==0)put_u32(out,vocab);
        put_u32(out,1);
        out.push_back(c<2?PTIR_HOST_NONE:PTIR_HOST_READER);
        out.push_back(c<2?1:0);
    }
    out.push_back(PTIR_STAGE_EPILOGUE);put_u32(out,10);
    out.push_back(PTIR_OP_CHAN_TAKE);put_u32(out,0);
    out.push_back(PTIR_OP_CHAN_TAKE);put_u32(out,1);
    out.push_back(PTIR_OP_RESHAPE);put_u32(out,0);out.push_back(1);put_u32(out,beams*vocab);
    out.push_back(PTIR_OP_TOP_K);put_u32(out,2);put_u32(out,k);
    out.push_back(PTIR_OP_CONST);out.push_back(PTIR_DT_U32);put_u32(out,vocab);
    out.push_back(PTIR_OP_DIV);put_u32(out,4);put_u32(out,5);
    out.push_back(PTIR_OP_REM);put_u32(out,4);put_u32(out,5);
    out.push_back(PTIR_OP_GATHER);put_u32(out,1);put_u32(out,7);
    out.push_back(PTIR_OP_CHAN_PUT);put_u32(out,2);put_u32(out,8);
    out.push_back(PTIR_OP_CHAN_PUT);put_u32(out,3);put_u32(out,6);
    return out;
}

std::vector<std::uint8_t> generic_beam_sidecar(
    const std::vector<std::uint8_t>& container,
    std::uint32_t beams = 2,
    std::uint32_t vocab = 3,
    std::uint32_t k = 2){
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P','T','I','B'};
    put_u16(out,1);put_u16(out,0);put_u64(out,pie_native::ptir::container::fnv1a64(container.data(),container.size()));
    put_u32(out,4);out.insert(out.end(),4,PTIR_CHAN_FULL_RING);
    put_u32(out,4);for(std::uint32_t c=0;c<4;++c){put_u32(out,c);out.push_back(PTIR_STAGE_EPILOGUE);out.push_back(c<2?PTIR_NEEDS_FULL:PTIR_NEEDS_EMPTY);}
    put_u32(out,1);out.push_back(PTIR_STAGE_EPILOGUE);put_u32(out,9);
    out.push_back(PTIR_DT_F32);out.push_back(2);put_u32(out,beams);put_u32(out,vocab);
    out.push_back(PTIR_DT_U32);out.push_back(1);put_u32(out,vocab);
    out.push_back(PTIR_DT_F32);out.push_back(1);put_u32(out,beams*vocab);
    out.push_back(PTIR_DT_F32);out.push_back(1);put_u32(out,k);
    out.push_back(PTIR_DT_U32);out.push_back(1);put_u32(out,k);
    out.push_back(PTIR_DT_U32);out.push_back(0);
    out.push_back(PTIR_DT_U32);out.push_back(1);put_u32(out,k);
    out.push_back(PTIR_DT_U32);out.push_back(1);put_u32(out,k);
    out.push_back(PTIR_DT_U32);out.push_back(1);put_u32(out,k);
    return upgrade_ptib_v1(container,std::move(out));
}

std::vector<std::uint8_t> ssa_nucleus_container() {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P','T','I','R'};
    put_u16(out,PTIR_VERSION);put_u16(out,0);
    put_u32(out,0);put_u32(out,4);put_u32(out,0);put_u32(out,1);
    const std::uint8_t dtypes[]={PTIR_DT_F32,PTIR_DT_F32,PTIR_DT_U32,PTIR_DT_I32};
    const std::uint32_t lengths[]={4,1,2,1};
    for(int c=0;c<4;++c){
        out.push_back(dtypes[c]);out.push_back(lengths[c]==1?0:1);if(lengths[c]!=1)put_u32(out,lengths[c]);
        put_u32(out,1);out.push_back(c==3?PTIR_HOST_READER:PTIR_HOST_NONE);out.push_back(c<3?1:0);
    }
    out.push_back(PTIR_STAGE_EPILOGUE);put_u32(out,17);
    out.push_back(PTIR_OP_CHAN_TAKE);put_u32(out,0);
    out.push_back(PTIR_OP_CHAN_TAKE);put_u32(out,1);
    out.push_back(PTIR_OP_CHAN_TAKE);put_u32(out,2);
    out.push_back(PTIR_OP_REDUCE_MAX);put_u32(out,0);
    out.push_back(PTIR_OP_BROADCAST);put_u32(out,3);out.push_back(1);put_u32(out,4);
    out.push_back(PTIR_OP_SUB);put_u32(out,0);put_u32(out,4);
    out.push_back(PTIR_OP_EXP);put_u32(out,5);
    out.push_back(PTIR_OP_REDUCE_SUM);put_u32(out,6);
    out.push_back(PTIR_OP_BROADCAST);put_u32(out,7);out.push_back(1);put_u32(out,4);
    out.push_back(PTIR_OP_DIV);put_u32(out,6);put_u32(out,8);
    out.push_back(PTIR_OP_PIVOT_THRESHOLD);put_u32(out,9);out.push_back(1);put_u32(out,1);
    out.push_back(PTIR_OP_CONST);out.push_back(PTIR_DT_F32);put_u32(out,0xff800000u);
    out.push_back(PTIR_OP_SELECT);put_u32(out,10);put_u32(out,0);put_u32(out,11);
    out.push_back(PTIR_OP_RNG_KEYED);put_u32(out,2);out.push_back(1);put_u32(out,4);out.push_back(1);
    out.push_back(PTIR_OP_ADD);put_u32(out,12);put_u32(out,13);
    out.push_back(PTIR_OP_REDUCE_ARGMAX);put_u32(out,14);
    out.push_back(PTIR_OP_CHAN_PUT);put_u32(out,3);put_u32(out,15);
    return out;
}

std::vector<std::uint8_t> ssa_nucleus_sidecar(
    const std::vector<std::uint8_t>& container) {
    using namespace pie::metal::tests;
    std::vector<std::uint8_t> out{'P','T','I','B'};
    put_u16(out,1);put_u16(out,0);put_u64(out,pie_native::ptir::container::fnv1a64(container.data(),container.size()));
    put_u32(out,4);out.insert(out.end(),4,PTIR_CHAN_FULL_RING);
    put_u32(out,4);for(std::uint32_t c=0;c<4;++c){put_u32(out,c);out.push_back(PTIR_STAGE_EPILOGUE);out.push_back(c<3?PTIR_NEEDS_FULL:PTIR_NEEDS_EMPTY);}
    put_u32(out,1);out.push_back(PTIR_STAGE_EPILOGUE);put_u32(out,16);
    auto put_type = [&](std::uint8_t dtype, std::uint32_t len) {
        out.push_back(dtype);
        out.push_back(len == 1 ? 0 : 1);
        if (len != 1) put_u32(out, len);
    };
    put_type(PTIR_DT_F32,4);
    put_type(PTIR_DT_F32,1);
    put_type(PTIR_DT_U32,2);
    put_type(PTIR_DT_F32,1);
    put_type(PTIR_DT_F32,4);
    put_type(PTIR_DT_F32,4);
    put_type(PTIR_DT_F32,4);
    put_type(PTIR_DT_F32,1);
    put_type(PTIR_DT_F32,4);
    put_type(PTIR_DT_F32,4);
    put_type(PTIR_DT_BOOL,4);
    put_type(PTIR_DT_F32,1);
    put_type(PTIR_DT_F32,4);
    put_type(PTIR_DT_F32,4);
    put_type(PTIR_DT_F32,4);
    put_type(PTIR_DT_I32,1);
    ExplicitLibraryRegion nucleus;
    nucleus.library_op = PTIR_LIBRARY_NUCLEUS_SAMPLE;
    for (std::uint32_t node = 3; node <= 15; ++node) {
        nucleus.nodes.push_back(node);
    }
    nucleus.inputs = {0,1,2};
    nucleus.outputs = {15};
    return upgrade_ptib_v1(
        container,
        std::move(out),
        {{std::move(nucleus)}});
}

std::vector<ChannelTicket> tickets_for(
    const ExecPlan& plan,
    const std::vector<std::shared_ptr<ChannelState>>& channels) {
    std::vector<ChannelTicket> tickets;
    for (std::size_t dense = 0; dense < channels.size(); ++dense) {
        tickets.push_back({
            .channel_id = dense + 1,
            .dense = dense,
            .expected_head =
                plan.takes_channel(static_cast<std::uint32_t>(dense))
                    ? channels[dense]->head()
                    : kNoChannelTicket,
            .expected_tail =
                plan.puts_channel(static_cast<std::uint32_t>(dense))
                    ? channels[dense]->tail()
                    : kNoChannelTicket,
            .requires_input = plan.requires_channel_input(
                static_cast<std::uint32_t>(dense)),
        });
    }
    return tickets;
}

M1ExecuteOutcome execute_generated(
    M1Runtime& runtime,
    const std::shared_ptr<M1ProgramExecutable>& executable,
    const ExecPlan& plan,
    const std::vector<std::shared_ptr<ChannelState>>& channels,
    const std::vector<float>& logits,
    std::uint32_t rows,
    std::uint32_t vocab,
    std::string& error,
    int mtp_draft_row = -1,
    M1ExecutionMode mode = M1ExecutionMode::Singleton) {
    std::shared_ptr<M1PreparedFire> fire;
    const M1PrepareOutcome prepared = runtime.prepare(
        executable,
        channels,
        tickets_for(plan, channels),
        fire,
        error);
    if (prepared != M1PrepareOutcome::Ready) {
        return M1ExecuteOutcome::Failed;
    }
    SlotHandle logits_buffer;
    if (!logits.empty()) {
        logits_buffer = runtime.context().create_standalone_buffer(
            logits.size() * sizeof(std::uint16_t));
        auto* encoded =
            static_cast<std::uint16_t*>(logits_buffer.contents());
        for (std::size_t index = 0; index < logits.size(); ++index) {
            encoded[index] = bf16(logits[index]);
        }
    }
    M1DeviceInputs inputs;
    inputs.logits_bf16 = logits_buffer;
    inputs.logits_row_count = rows;
    inputs.vocab = vocab;
    inputs.mtp_draft_row = mtp_draft_row;
    inputs.extents.sampled_rows = std::max(rows, 1u);
    inputs.extents.row_count = std::max(rows, 1u);
    const M1ExecuteOutcome outcome =
        runtime.execute(fire, inputs, error, mode);
    if (logits_buffer.valid()) {
        runtime.context().release_standalone_buffer(logits_buffer);
    }
    runtime.release(fire);
    return outcome;
}

struct RunResult {
    M1PrepareOutcome prepare = M1PrepareOutcome::Failed;
    M1ExecuteOutcome execute = M1ExecuteOutcome::Failed;
    std::int32_t token = -1;
    std::string error;
};

RunResult run_greedy(
    M1Runtime& runtime,
    const std::shared_ptr<M1ProgramExecutable>& executable,
    const ExecPlan& plan,
    const std::vector<float>& logits,
    bool output_full = false,
    M1ExecutionMode mode = M1ExecutionMode::Singleton) {
    RunResult result;
    std::vector<std::shared_ptr<ChannelState>> channels;
    for (const auto& channel : plan.trace.channels) {
        channels.push_back(make_platform_channel_state(
            channel.type.dtype,
            channel.type.shape.numel(),
            channel.capacity));
    }
    (void)channels[0]->push(Value::i32({1}));
    if (output_full) (void)channels[1]->push(Value::i32({99}));

    const std::vector<ChannelTicket> tickets =
        tickets_for(plan, channels);
    std::shared_ptr<M1PreparedFire> fire;
    result.prepare = runtime.prepare(
        executable, channels, tickets, fire, result.error);
    if (result.prepare != M1PrepareOutcome::Ready) return result;

    SlotHandle logits_buffer = runtime.context().create_standalone_buffer(
        logits.size() * sizeof(std::uint16_t));
    auto* encoded = static_cast<std::uint16_t*>(logits_buffer.contents());
    for (std::size_t index = 0; index < logits.size(); ++index) {
        encoded[index] = bf16(logits[index]);
    }
    M1DeviceInputs inputs;
    inputs.logits_bf16 = logits_buffer;
    inputs.logits_row_count = 1;
    inputs.vocab = static_cast<std::uint32_t>(logits.size());
    inputs.extents.sampled_rows = 1;
    inputs.extents.row_count = 1;
    result.execute = runtime.execute(
        fire, inputs, result.error, mode);
    if (result.execute == M1ExecuteOutcome::Committed) {
        result.token = channels[1]->front().i[0];
    }
    runtime.context().release_standalone_buffer(logits_buffer);
    runtime.release(fire);
    return result;
}

}  // namespace

int main() {
#define M1_TAG_ENTRY(name, tag, arity, results) std::uint8_t{tag},
    const std::vector<std::uint8_t> all_tags = {
        PTIR_OP_LIST(M1_TAG_ENTRY)
    };
#undef M1_TAG_ENTRY
    bool every_tag_emits = true;
    for (std::uint8_t tag : all_tags) {
        every_tag_emits =
            every_tag_emits &&
            !emit_singleton_region_msl(
                 "", "coverage", tag)
                 .empty();
    }
    expect(
        every_tag_emits,
        "singleton emitter covers every first-party op and explicit "
        "semantic boundary tag");

    M1RuntimeExtents divergent{
        .kv_len = 2,
        .page_count = 3,
        .row_count = 5,
        .token_count = 7,
        .sampled_rows = 11,
        .query_len = 13,
        .key_len = 17,
    };
    const std::array<std::uint32_t, 7> expected_extents{
        2, 3, 5, 7, 11, 13, 17};
    bool independent_extents = true;
    for (std::uint32_t role = 0; role < expected_extents.size(); ++role) {
        pie_native::ptir::plan::ValueType type;
        type.dtype = PTIR_DT_U32;
        type.dims.push_back({.symbolic = true, .value = role});
        independent_extents =
            independent_extents &&
            resolve_m1_shape_for_test(type, divergent).len ==
                expected_extents[role];
    }
    expect(
        independent_extents,
        "lane symbolic extents remain independent under divergent canaries");
    MemberForwardDesc extent_desc;
    extent_desc.kv_len = 2;
    extent_desc.page_count = 3;
    extent_desc.row_count = 5;
    extent_desc.token_count = 7;
    extent_desc.sampled_rows = 11;
    extent_desc.query_len = 13;
    extent_desc.key_len = 17;
    extent_desc.readout_local_indices = {0, 2, 4};
    const M1RuntimeExtents mapped_extents =
        m1_extents_from_forward_desc(extent_desc, 11);
    expect(
        mapped_extents.kv_len == 2 &&
            mapped_extents.page_count == 3 &&
            mapped_extents.row_count == 5 &&
            mapped_extents.token_count == 7 &&
            mapped_extents.sampled_rows == 11 &&
            mapped_extents.query_len == 13 &&
            mapped_extents.key_len == 17,
        "forward geometry populates every lane extent independently");
    expect(
        m3_extents_from_forward_desc(extent_desc).sampled_rows == 3,
        "Context M3 extents derive sampled_rows from the actual row map");
    pie_native::ptir::plan::ValueType rank3_type;
    rank3_type.dtype = PTIR_DT_F32;
    rank3_type.dims = {
        {.symbolic = false, .value = 2},
        {.symbolic = false, .value = 3},
        {.symbolic = false, .value = 4},
    };
    const M1ResolvedShape rank3_shape =
        resolve_m1_shape_for_test(rank3_type, divergent);
    expect(
        rank3_shape.rows == 6 && rank3_shape.row_len == 4,
        "rank-3 reductions use the product of every leading dimension");

    std::vector<std::uint8_t> container;
    std::vector<std::uint8_t> sidecar;
    expect(
        load_golden("greedy_argmax", container, sidecar),
        "loaded evolving PTIB v2 greedy golden");

    {
        std::vector<std::uint8_t> bad_signature = sidecar;
        const std::size_t plan_offset = find_ptrp(bad_signature);
        const std::uint32_t signature_len =
            read_u32(bad_signature, plan_offset + 17);
        if (signature_len != 0) bad_signature[plan_offset + 21] ^= 0x01;
        ExecPlan rejected;
        std::string rejection;
        expect(
            signature_len != 0 &&
                !build_exec_plan(
                    container.data(),
                    container.size(),
                    bad_signature.data(),
                    bad_signature.size(),
                    rejected,
                    &rejection) &&
                rejection.find("signature hash mismatch") !=
                    std::string::npos,
            "updated PTRP decoder validates signature bytes against signature_hash");
    }
    for (const auto& [kind, expected] : std::vector<
             std::pair<RegionIndexKind, std::string>>{
             {RegionIndexKind::Node, "region node out of range"},
             {RegionIndexKind::Input, "region input out of range"},
             {RegionIndexKind::Output, "region output out of range"},
             {RegionIndexKind::SinkChannel, "region sink out of range"},
             {RegionIndexKind::SinkValue, "region sink out of range"},
         }) {
        std::vector<std::uint8_t> corrupted = sidecar;
        ExecPlan rejected;
        std::string rejection;
        const RegionIndexMutation mutation =
            corrupt_region_index(corrupted, kind);
        const bool exact_boundary_mutation =
            mutation.changed &&
            mutation.original < mutation.upper_bound &&
            read_u32(corrupted, mutation.offset) ==
                mutation.upper_bound;
        expect(
            exact_boundary_mutation &&
                !build_exec_plan(
                    container.data(),
                    container.size(),
                    corrupted.data(),
                    corrupted.size(),
                    rejected,
                    &rejection) &&
                rejection.find("invalid region partition") !=
                    std::string::npos,
            "updated PTRP decoder rejects " + expected +
                " (changed=" +
                std::to_string(mutation.changed) +
                ", original=" +
                std::to_string(mutation.original) +
                ", upper_bound=" +
                std::to_string(mutation.upper_bound) +
                ", rejection=" + rejection + ")");
    }

    ExecPlan plan;
    std::string error;
    expect(
        build_exec_plan(
            container.data(),
            container.size(),
            sidecar.data(),
            sidecar.size(),
            plan,
            &error),
        "decoded PTIB v2 with PTRP v4/compiler v3 plans (" + error + ")");
    expect(
        plan.bound.version == 2 && !plan.region_plans.empty(),
        "golden carries compiler-owned singleton partitions");

    {
        const M1CacheIdentityVersions zero{};
        const std::string baseline =
            encode_m1_cache_identity(0x1122334455667788ULL, 7, zero);
        std::array<M1CacheIdentityVersions, 4> version_16{zero, zero, zero, zero};
        version_16[0].compiler = 16;
        version_16[1].region_plan = 16;
        version_16[2].lane_table = 16;
        version_16[3].emitter = 16;
        const bool full_width = std::all_of(
            version_16.begin(),
            version_16.end(),
            [&](M1CacheIdentityVersions versions) {
                return encode_m1_cache_identity(
                           0x1122334455667788ULL, 7, versions) !=
                       baseline;
            });
        expect(
            full_width,
            "disk-cache identity keeps full compiler/PTRP/lane/emitter "
            "versions (0 and 16 never alias)");
    }

    {
        auto pool = RawMetalContext::create(1u << 20);
        pool->set_transient_buffer_pool_limit_for_test(1024);
        SlotHandle first = pool->acquire_transient_buffer(300);
        if (first.valid()) {
            static_cast<std::uint8_t*>(first.contents())[0] = 0x5a;
        }
        pool->recycle_transient_buffer(first);
        SlotHandle reused = pool->acquire_transient_buffer(400);
        const bool preserved =
            reused.valid() && reused.buffer == first.buffer &&
            static_cast<const std::uint8_t*>(reused.contents())[0] == 0x5a;
        const SlotHandle over_limit =
            pool->acquire_transient_buffer(700);
        pool->recycle_transient_buffer(reused);
        SlotHandle largest = pool->acquire_transient_buffer(700);
        pool->recycle_transient_buffer(largest);
        const TransientBufferPoolStats stats =
            pool->transient_buffer_pool_stats();
        expect(
            preserved && !over_limit.valid() && largest.valid() &&
                stats.allocations == 2 && stats.reuse_hits == 1 &&
                stats.allocation_failures == 1 &&
                stats.in_use_buffers == 0 &&
                stats.resident_bytes <= stats.capacity_bytes &&
                stats.cached_buffers == 1,
            "resident size-class pool reuses without CPU clearing and enforces "
            "its byte bound");
    }

    std::string kernels_dir = PIE_METAL_KERNELS_DIR_DEFAULT;
    const std::filesystem::path cache = "m1-generated-test-cache";
    std::error_code ec;
    std::filesystem::remove_all(cache, ec);
    auto runtime = M1Runtime::create(kernels_dir, cache.string(), error);
    expect(runtime != nullptr, "created M1 runtime (" + error + ")");
    if (!runtime) return 1;

    for (const std::string& golden : {
             "beam_epilogue",
             "counter_pingpong",
             "dfa_ingraph",
             "matrix_select_mask",
             "nucleus_sample",
             "structured_masks",
         }) {
        std::vector<std::uint8_t> golden_container;
        std::vector<std::uint8_t> golden_sidecar;
        ExecPlan golden_plan;
        error.clear();
        const bool decoded =
            load_golden(
                golden,
                golden_container,
                golden_sidecar) &&
            build_exec_plan(
                golden_container.data(),
                golden_container.size(),
                golden_sidecar.data(),
                golden_sidecar.size(),
                golden_plan,
                &error);
        const std::uint64_t hash =
            decoded
                ? pie_native::ptir::container::fnv1a64(
                      golden_container.data(), golden_container.size())
                : 0;
        const auto compiled =
            decoded
                ? runtime->compile_program(hash, golden_plan, error)
                : nullptr;
        if (golden == "beam_epilogue") {
            auto const_port_is = [&](std::uint8_t port,
                                     std::vector<std::uint32_t> expected) {
                const auto found = std::find_if(
                    golden_plan.const_ports.begin(),
                    golden_plan.const_ports.end(),
                    [&](const auto& value) {
                        return value.port == port;
                    });
                return found != golden_plan.const_ports.end() &&
                       found->value.dtype == DType::U32 &&
                       found->value.u == expected;
            };
            const bool generic_beam =
                decoded &&
                const_port_is(
                    PTIR_PORT_EMBED_INDPTR, {0, 1, 2}) &&
                const_port_is(
                    PTIR_PORT_PAGE_INDPTR, {0, 3, 6}) &&
                std::any_of(
                    golden_plan.region_plans.begin(),
                    golden_plan.region_plans.end(),
                    [](const auto& stage) {
                        auto has_tag = [&](std::uint8_t tag) {
                            return std::any_of(
                                stage.ops.begin(),
                                stage.ops.end(),
                                [&](const auto& op) {
                                    return op.op.tag == tag;
                                });
                        };
                        const bool topk_region = std::any_of(
                            stage.fused.regions.begin(),
                            stage.fused.regions.end(),
                            [&](const auto& region) {
                                return region.library &&
                                    region.library_op ==
                                        PTIR_LIBRARY_TOP_K &&
                                    region.nodes.size() == 1 &&
                                    stage.ops[region.nodes[0]].op.tag ==
                                        PTIR_OP_TOP_K;
                            });
                        return topk_region &&
                               has_tag(PTIR_OP_DIV) &&
                               has_tag(PTIR_OP_REM) &&
                               has_tag(PTIR_OP_GATHER);
                    });
            expect(
                generic_beam,
                "shared beam golden preserves const request CSRs and uses "
                "opcode-derived TopK plus generic DIV/REM/Gather SSA");
        }
        if (golden == "nucleus_sample") {
            const bool nucleus_region_abi =
                decoded &&
                std::any_of(
                    golden_plan.region_plans.begin(),
                    golden_plan.region_plans.end(),
                    [](const auto& stage) {
                        return std::any_of(
                            stage.fused.regions.begin(),
                            stage.fused.regions.end(),
                            [&](const auto& region) {
                                return region.library &&
                                    region.library_op ==
                                        PTIR_LIBRARY_NUCLEUS_SAMPLE &&
                                    region.nodes.size() == 13 &&
                                    region.inputs.size() == 3 &&
                                    region.outputs.size() == 1 &&
                                    region.sinks.empty() &&
                                    stage.value_types[region.inputs[0]].dtype ==
                                        PTIR_DT_F32 &&
                                    stage.value_types[region.inputs[1]].dtype ==
                                        PTIR_DT_F32 &&
                                    stage.value_types[region.inputs[2]].dtype ==
                                        PTIR_DT_U32 &&
                                    stage.value_types[region.outputs[0]].dtype ==
                                        PTIR_DT_I32;
                            });
                    });
            expect(
                nucleus_region_abi,
                "shared nucleus golden carries the PTRP v4 role-ordered "
                "13-node library region ABI");
            if (decoded && !golden_plan.region_plans.empty()) {
                auto malformed = golden_plan.region_plans.front();
                const auto region = std::find_if(
                    malformed.fused.regions.begin(),
                    malformed.fused.regions.end(),
                    [](const auto& candidate) {
                        return candidate.library &&
                            candidate.library_op ==
                                PTIR_LIBRARY_NUCLEUS_SAMPLE;
                    });
                std::vector<M1OpMeta> operations;
                std::string malformed_error;
                bool rejected = false;
                if (region != malformed.fused.regions.end() &&
                    region->inputs.size() == 3) {
                    std::swap(region->inputs[1], region->inputs[2]);
                    rejected = !validate_singleton_plan(
                        malformed, operations, malformed_error);
                }
                expect(
                    rejected,
                    "Metal validates nucleus input roles from the Rust region "
                    "ABI without matching the SSA graph");
            }
        }
        expect(
            compiled != nullptr,
            "registered generated golden " + golden + " (" + error + ")");
    }
    {
        std::vector<std::uint8_t> phased_container;
        std::vector<std::uint8_t> phased_sidecar;
        ExecPlan phased_plan;
        error.clear();
        const bool decoded =
            load_golden(
                "pivot_predicates_multistage",
                phased_container,
                phased_sidecar) &&
            build_exec_plan(
                phased_container.data(),
                phased_container.size(),
                phased_sidecar.data(),
                phased_sidecar.size(),
                phased_plan,
                &error);
        const auto compiled =
            decoded
                ? runtime->compile_program(
                      pie_native::ptir::container::fnv1a64(
                          phased_container.data(),
                          phased_container.size()),
                      phased_plan,
                      error)
                : nullptr;
        expect(
            compiled != nullptr,
            "fresh multistage golden compiles with M2 pre/post placement (" +
                error + ")");
    }

    auto executable =
        runtime->compile_program(
            0xff694395428759feULL, plan, error, container);
    expect(executable != nullptr, "compiled every greedy singleton region (" + error + ")");
    auto rekey_plan = [](ExecPlan& candidate, std::uint8_t marker) {
        candidate.region_plans[0].signature.push_back(marker);
        candidate.region_plans[0].signature_hash =
            pie_native::ptir::container::fnv1a64(
                candidate.region_plans[0].signature.data(),
                candidate.region_plans[0].signature.size());
    };
    {
        ExecPlan malformed = plan;
        auto op = std::find_if(
            malformed.region_plans[0].ops.begin(),
            malformed.region_plans[0].ops.end(),
            [](const auto& value) {
                return !value.op.args.empty();
            });
        op->op.args[0] =
            static_cast<std::uint32_t>(
                malformed.region_plans[0].value_types.size());
        rekey_plan(malformed, 0x91);
        error.clear();
        expect(
            runtime->compile_program(
                0xa100000000000001ULL, malformed, error) == nullptr &&
                error.find("SSA operand") != std::string::npos,
            "registration rejects an out-of-range normalized SSA operand");
    }
    {
        ExecPlan malformed = plan;
        auto op = std::find_if(
            malformed.region_plans[0].ops.begin(),
            malformed.region_plans[0].ops.end(),
            [](const auto& value) {
                return value.op.tag == PTIR_OP_CHAN_PUT;
            });
        op->op.chan = static_cast<std::int32_t>(
            malformed.region_plans[0].channel_bindings.size());
        rekey_plan(malformed, 0x92);
        error.clear();
        expect(
            runtime->compile_program(
                0xa100000000000002ULL, malformed, error) == nullptr &&
                error.find("channel slot") != std::string::npos,
            "registration rejects an out-of-range normalized channel slot");
    }
    {
        ExecPlan malformed = plan;
        malformed.region_plans[0].value_types[0].dims = {
            {.symbolic = false,
             .value = std::numeric_limits<std::uint32_t>::max()},
            {.symbolic = false, .value = 2},
        };
        rekey_plan(malformed, 0x93);
        error.clear();
        expect(
            runtime->compile_program(
                0xa100000000000003ULL, malformed, error) == nullptr &&
                error.find("shape product") != std::string::npos,
            "registration rejects an overflowing normalized shape product");
    }
    if (!executable) return 1;
    std::vector<std::uint8_t> colliding_program = container;
    colliding_program.back() ^= 0x01;
    error.clear();
    expect(
        runtime->compile_program(
            0xff694395428759feULL,
            plan,
            error,
            colliding_program) == nullptr &&
            error.find("program hash collision") != std::string::npos,
        "M1 program cache rejects same-hash nonidentical canonical bytes");
    expect(
        runtime->context().last_ptir_compile_disabled_fast_math(),
        "all generated executables use strict math");

    const RunResult ordinary = run_greedy(
        *runtime,
        executable,
        plan,
        {0, 1, 9, 2, 0, 0, 0, 3});
    expect(
        ordinary.prepare == M1PrepareOutcome::Ready &&
            ordinary.execute == M1ExecuteOutcome::Committed &&
            ordinary.token == 2,
        "generated greedy matches the Rust golden token (prepare=" +
            std::to_string(static_cast<int>(ordinary.prepare)) +
            " execute=" +
            std::to_string(static_cast<int>(ordinary.execute)) +
            " token=" + std::to_string(ordinary.token) +
            " error=" + ordinary.error + ")");
    const RunResult fused_greedy = run_greedy(
        *runtime,
        executable,
        plan,
        {0, 1, 9, 2, 0, 0, 0, 3},
        false,
        M1ExecutionMode::Fused);
    expect(
        fused_greedy.execute == M1ExecuteOutcome::Committed &&
            fused_greedy.token == ordinary.token,
        "fused and singleton partitions match bitwise for greedy argmax (" +
            fused_greedy.error + ")");

    {
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : plan.trace.channels) {
            channels.push_back(make_platform_channel_state(
                channel.type.dtype,
                channel.type.shape.numel(),
                channel.capacity));
        }
        (void)channels[0]->push(Value::i32({1}));
        std::shared_ptr<M1PreparedFire> fire;
        error.clear();
        const M1PrepareOutcome prepared = runtime->prepare(
            executable,
            channels,
            tickets_for(plan, channels),
            fire,
            error);
        auto forward_context = RawMetalContext::create(4u << 20);
        SlotHandle logits =
            forward_context->create_standalone_buffer(8 * sizeof(std::uint16_t));
        const std::string writer_source = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void write_forward_logits(device bfloat* output [[buffer(0)]],
                                  uint gid [[thread_position_in_grid]]) {
  const float values[8] = {0, 1, 9, 2, 0, 0, 0, 3};
  output[gid] = bfloat(values[gid]);
}
)MSL";
        Pso writer = forward_context->compile_ptir_pso(
            writer_source, "write_forward_logits", &error);
        forward_context->arg_bind_ordinal(92000, 0, logits);
        forward_context->make_resident();
        M1DeviceInputs inputs;
        inputs.logits_bf16 = logits;
        inputs.logits_row_count = 1;
        inputs.vocab = 8;
        inputs.extents.sampled_rows = 1;
        inputs.extents.row_count = 1;
        std::shared_ptr<M2CommandPlan> command;
        const bool command_ready =
            prepared == M1PrepareOutcome::Ready &&
            runtime->prepare_m2_command(
                fire, inputs, *forward_context, command, error);
        if (command_ready) {
            forward_context->run_step([&](StepEncoder& encoder) {
                runtime->encode_m2_pre(command, encoder);
                encoder.set_pso(writer);
                encoder.set_argtable_ordinal(92000);
                encoder.dispatch(Grid{8, 1, 1}, Threadgroup{8, 1, 1});
                encoder.barrier(BarrierVisibility::Device);
                runtime->encode_m2_post(command, encoder);
            });
        }
        const M1ExecuteOutcome command_outcome =
            command_ready
                ? runtime->finish_m2_command(command, error)
                : M1ExecuteOutcome::Failed;
        expect(
            command_outcome == M1ExecuteOutcome::Committed &&
                channels[1]->front().i[0] == ordinary.token,
            "fused epilogue encodes in the forward command-buffer context (" +
                error + ")");
        expect(
            forward_context->external_buffer_count() == 0,
            "M2 releases per-command external residency references");
        forward_context->release_standalone_buffer(logits);
        runtime->release(fire);
    }

    {
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits =
            target->create_standalone_buffer(16 * sizeof(std::uint16_t));
        const std::vector<float> rows = {
            0, 1, 9, 2, 0, 0, 0, 3,
            0, 1, 2, 10, 0, 0, 0, 3,
        };
        auto* encoded = static_cast<std::uint16_t*>(logits.contents());
        for (std::size_t index = 0; index < rows.size(); ++index)
            encoded[index] = bf16(rows[index]);
        target->make_resident();

        std::array<std::vector<std::shared_ptr<ChannelState>>, 2> channels;
        std::array<std::shared_ptr<M1PreparedFire>, 2> fires;
        std::array<std::shared_ptr<M2CommandPlan>, 2> commands;
        bool ready = true;
        for (std::size_t member = 0; member < 2; ++member) {
            for (const auto& channel : plan.trace.channels) {
                channels[member].push_back(make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
            }
            (void)channels[member][0]->push(Value::i32({1}));
            ready =
                ready &&
                runtime->prepare(
                    executable,
                    channels[member],
                    tickets_for(plan, channels[member]),
                    fires[member],
                    error) == M1PrepareOutcome::Ready;
            M1DeviceInputs inputs;
            inputs.logits_bf16 = logits;
            inputs.logits_row_offset =
                static_cast<std::uint32_t>(member);
            inputs.logits_row_count = 1;
            inputs.vocab = 8;
            inputs.extents.sampled_rows = 1;
            inputs.extents.row_count = 1;
            ready =
                ready &&
                runtime->prepare_m2_command(
                    fires[member],
                    inputs,
                    *target,
                    commands[member],
                    error);
        }
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m2_pre(commands[0], encoder);
                runtime->encode_m2_pre(commands[1], encoder);
                runtime->encode_m2_post(commands[0], encoder);
                runtime->encode_m2_post(commands[1], encoder);
            });
        }
        const auto first =
            ready ? runtime->finish_m2_command(commands[0], error)
                  : M1ExecuteOutcome::Failed;
        const auto second =
            ready ? runtime->finish_m2_command(commands[1], error)
                  : M1ExecuteOutcome::Failed;
        expect(
            first == M1ExecuteOutcome::Committed &&
                second == M1ExecuteOutcome::Committed &&
                channels[0][1]->front().i[0] == 2 &&
                channels[1][1]->front().i[0] == 3 &&
                target->external_buffer_count() == 0,
            "two in-flight members retain distinct argument-table bindings");
        runtime->release(fires[0]);
        runtime->release(fires[1]);
        target->release_standalone_buffer(logits);
    }

    auto cross_program = runtime->compile_program(
        0x1234432112344321ULL, plan, error, container);
    std::vector<std::uint64_t> grouped_launches;
    std::vector<std::uint64_t> readiness_launches;
    std::vector<std::uint64_t> commit_launches;
    std::vector<double> m3_gpu_ms;
    bool m3_pool_bounded = true;
    for (const std::size_t lane_count : {1u, 2u, 4u, 8u}) {
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits = target->create_standalone_buffer(
            lane_count * 8 * sizeof(std::uint16_t));
        auto* encoded = static_cast<std::uint16_t*>(logits.contents());
        std::vector<std::uint32_t> expected;
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            const std::uint32_t token =
                static_cast<std::uint32_t>((lane + 2) % 8);
            expected.push_back(token);
            for (std::size_t column = 0; column < 8; ++column) {
                encoded[lane * 8 + column] =
                    bf16(column == token ? 10.0f : 0.0f);
            }
        }
        target->make_resident();
        std::vector<std::vector<std::shared_ptr<ChannelState>>> lanes(
            lane_count);
        std::vector<std::shared_ptr<M1PreparedFire>> fires(lane_count);
        std::vector<M3LaneCandidate> candidates;
        bool ready = true;
        for (std::size_t lane = 0; lane < lane_count; ++lane) {
            for (const auto& channel : plan.trace.channels) {
                lanes[lane].push_back(make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
            }
            (void)lanes[lane][0]->push(Value::i32({1}));
            const auto& lane_program =
                lane % 2 == 0 ? executable : cross_program;
            ready =
                ready &&
                runtime->prepare(
                    lane_program,
                    lanes[lane],
                    tickets_for(plan, lanes[lane]),
                    fires[lane],
                    error) == M1PrepareOutcome::Ready;
            M1DeviceInputs inputs;
            inputs.logits_bf16 = logits;
            inputs.logits_row_offset =
                static_cast<std::uint32_t>(lane);
            inputs.logits_row_count = 1;
            inputs.vocab = 8;
            inputs.extents.sampled_rows = 1;
            inputs.extents.row_count = 1;
            candidates.push_back({
                .fire = fires[lane],
                .inputs = inputs,
                .retry_ineligible = false,
            });
        }
        const M3GroupStats before = runtime->m3_stats();
        std::shared_ptr<M3GroupCommand> group;
        ready = ready && runtime->prepare_m3_group(
                             candidates, *target, group, error);
        StepTiming timing;
        if (ready) {
            timing = target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        m3_gpu_ms.push_back(timing.gpu_exec_ms);
        const auto outcomes =
            ready ? runtime->finish_m3_group(group, error)
                  : std::vector<M1ExecuteOutcome>{};
        const M3GroupStats after = runtime->m3_stats();
        const TransientBufferPoolStats m3_pool =
            target->transient_buffer_pool_stats();
        m3_pool_bounded =
            m3_pool_bounded && m3_pool.recycles > 0 &&
            m3_pool.in_use_buffers == 0 &&
            m3_pool.resident_bytes <= m3_pool.capacity_bytes;
        grouped_launches.push_back(
            after.body_launches - before.body_launches);
        readiness_launches.push_back(
            after.readiness_launches - before.readiness_launches);
        commit_launches.push_back(
            after.commit_launches - before.commit_launches);
        bool attributed = outcomes.size() == lane_count;
        for (std::size_t lane = 0; lane < outcomes.size(); ++lane) {
            attributed =
                attributed &&
                outcomes[lane] == M1ExecuteOutcome::Committed &&
                lanes[lane][1]->front().i[0] ==
                    static_cast<std::int32_t>(expected[lane]);
        }
        expect(
            attributed,
            "grouped N=" + std::to_string(lane_count) +
                " preserves lane attribution across programs (" + error + ")");
        for (auto& fire : fires) runtime->release(fire);
        target->release_standalone_buffer(logits);
    }
    expect(
        grouped_launches.size() == 4 &&
            grouped_launches[0] == grouped_launches[1] &&
            grouped_launches[1] == grouped_launches[2] &&
            grouped_launches[2] == grouped_launches[3],
        "N=1/2/4/8 grouped body launch count is lane-count invariant");
    expect(
        readiness_launches == std::vector<std::uint64_t>({1, 1, 1, 1}) &&
            commit_launches ==
                std::vector<std::uint64_t>({1, 1, 1, 1}),
        "N=1/2/4/8 grouped readiness and commit launches are lane-count invariant");
    expect(
        runtime->m3_stats().post_forward_critical_ns > 0,
        "grouped post-forward critical path is timestamped");

    {
        const std::size_t singleton_body =
            plan.region_plans[0].singleton.regions.size();
        const std::size_t fused_body =
            plan.region_plans[0].fused.regions.size();
        std::size_t perf_index = 0;
        bool m2_pool_bounded = true;
        for (const std::size_t lanes : {1u, 2u, 4u, 8u}) {
            SlotHandle m1_logits =
                runtime->context().create_standalone_buffer(
                    lanes * 8 * sizeof(std::uint16_t));
            auto* encoded =
                static_cast<std::uint16_t*>(m1_logits.contents());
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                for (std::size_t column = 0; column < 8; ++column) {
                    encoded[lane * 8 + column] =
                        bf16(column == (lane + 2) % 8 ? 10.0f : 0.0f);
                }
            }
            std::vector<std::vector<std::shared_ptr<ChannelState>>>
                m1_channels(lanes);
            std::vector<std::shared_ptr<M1PreparedFire>> m1_fires(lanes);
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                for (const auto& channel : plan.trace.channels) {
                    m1_channels[lane].push_back(
                        make_platform_channel_state(
                            channel.type.dtype,
                            channel.type.shape.numel(),
                            channel.capacity));
                }
                (void)m1_channels[lane][0]->push(Value::i32({1}));
                (void)runtime->prepare(
                    executable,
                    m1_channels[lane],
                    tickets_for(plan, m1_channels[lane]),
                    m1_fires[lane],
                    error);
            }
            const auto m1_begin = std::chrono::steady_clock::now();
            bool m1_ok = true;
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                M1DeviceInputs inputs;
                inputs.logits_bf16 = m1_logits;
                inputs.logits_row_offset =
                    static_cast<std::uint32_t>(lane);
                inputs.logits_row_count = 1;
                inputs.vocab = 8;
                inputs.extents.sampled_rows = 1;
                inputs.extents.row_count = 1;
                m1_ok =
                    m1_ok &&
                    runtime->execute(
                        m1_fires[lane],
                        inputs,
                        error,
                        M1ExecutionMode::Singleton) ==
                        M1ExecuteOutcome::Committed;
            }
            const double m1_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - m1_begin)
                    .count();
            for (auto& fire : m1_fires) runtime->release(fire);

            auto m2_target = RawMetalContext::create(4u << 20);
            SlotHandle m2_logits =
                m2_target->create_standalone_buffer(
                    lanes * 8 * sizeof(std::uint16_t));
            std::memcpy(
                m2_logits.contents(),
                encoded,
                lanes * 8 * sizeof(std::uint16_t));
            runtime->context().release_standalone_buffer(m1_logits);
            m2_target->make_resident();
            std::vector<std::vector<std::shared_ptr<ChannelState>>>
                m2_channels(lanes);
            std::vector<std::shared_ptr<M1PreparedFire>> m2_fires(lanes);
            std::vector<std::shared_ptr<M2CommandPlan>> commands(lanes);
            bool m2_ok = true;
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                for (const auto& channel : plan.trace.channels) {
                    m2_channels[lane].push_back(
                        make_platform_channel_state(
                            channel.type.dtype,
                            channel.type.shape.numel(),
                            channel.capacity));
                }
                (void)m2_channels[lane][0]->push(Value::i32({1}));
                m2_ok =
                    m2_ok &&
                    runtime->prepare(
                        executable,
                        m2_channels[lane],
                        tickets_for(plan, m2_channels[lane]),
                        m2_fires[lane],
                        error) == M1PrepareOutcome::Ready;
                M1DeviceInputs inputs;
                inputs.logits_bf16 = m2_logits;
                inputs.logits_row_offset =
                    static_cast<std::uint32_t>(lane);
                inputs.logits_row_count = 1;
                inputs.vocab = 8;
                inputs.extents.sampled_rows = 1;
                inputs.extents.row_count = 1;
                m2_ok =
                    m2_ok &&
                    runtime->prepare_m2_command(
                        m2_fires[lane],
                        inputs,
                        *m2_target,
                        commands[lane],
                        error);
            }
            StepTiming m2_timing;
            if (m2_ok) {
                m2_timing = m2_target->run_step(
                    [&](StepEncoder& encoder) {
                        for (const auto& command : commands)
                            runtime->encode_m2_pre(command, encoder);
                        for (const auto& command : commands)
                            runtime->encode_m2_post(command, encoder);
                    });
            }
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                m2_ok =
                    m2_ok &&
                    runtime->finish_m2_command(
                        commands[lane], error) ==
                        M1ExecuteOutcome::Committed;
                runtime->release(m2_fires[lane]);
            }
            const TransientBufferPoolStats m2_pool =
                m2_target->transient_buffer_pool_stats();
            m2_pool_bounded =
                m2_pool_bounded && m2_pool.recycles > 0 &&
                m2_pool.in_use_buffers == 0 &&
                m2_pool.resident_bytes <= m2_pool.capacity_bytes;
            m2_target->release_standalone_buffer(m2_logits);
            expect(
                m1_ok && m2_ok,
                "M1/M2 performance probe commits B=" +
                    std::to_string(lanes));
            std::printf(
                "  PERF M1 B=%zu dispatches=%zu wall_ms=%.3f\n",
                lanes,
                lanes * (singleton_body + 2),
                m1_ms);
            std::printf(
                "  PERF M2 B=%zu dispatches=%zu gpu_ms=%.3f\n",
                lanes,
                lanes * (fused_body + 2),
                m2_timing.gpu_exec_ms);
            std::printf(
                "  PERF M3 B=%zu dispatches=%llu gpu_ms=%.3f\n",
                lanes,
                static_cast<unsigned long long>(
                    grouped_launches[perf_index] + 2),
                m3_gpu_ms[perf_index]);
            ++perf_index;
        }
        expect(
            m2_pool_bounded && m3_pool_bounded,
            "M2/M3 command scratch returns to bounded resident pools after "
            "every B=1/2/4/8 completion fence");
    }

    {
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits =
            target->create_standalone_buffer(16 * sizeof(std::uint16_t));
        auto* encoded = static_cast<std::uint16_t*>(logits.contents());
        for (std::size_t lane = 0; lane < 2; ++lane)
            for (std::size_t column = 0; column < 8; ++column)
                encoded[lane * 8 + column] =
                    bf16(column == lane + 2 ? 10.0f : 0.0f);
        target->make_resident();
        std::array<std::vector<std::shared_ptr<ChannelState>>, 2> channels;
        std::array<std::shared_ptr<M1PreparedFire>, 2> fires;
        std::vector<M3LaneCandidate> candidates;
        for (std::size_t lane = 0; lane < 2; ++lane) {
            for (const auto& channel : plan.trace.channels)
                channels[lane].push_back(make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
            (void)channels[lane][0]->push(Value::i32({1}));
            (void)runtime->prepare(
                executable,
                channels[lane],
                tickets_for(plan, channels[lane]),
                fires[lane],
                error);
            M1DeviceInputs inputs;
            inputs.logits_bf16 = logits;
            inputs.logits_row_offset =
                static_cast<std::uint32_t>(lane);
            inputs.logits_row_count = 1;
            inputs.vocab = 8;
            inputs.extents.sampled_rows = 1;
            inputs.extents.row_count = 1;
            candidates.push_back({.fire = fires[lane], .inputs = inputs});
        }
        {
            std::shared_ptr<M3GroupCommand> rejected;
            const std::vector<M3LaneCandidate> aliased{
                candidates[0], candidates[0]};
            std::string alias_error;
            expect(
                !runtime->prepare_m3_group(
                    aliased, *target, rejected, alias_error) &&
                    alias_error.find("shared-channel alias") !=
                        std::string::npos,
                "shared-channel aliases reject grouped execution");
        }
        std::shared_ptr<M3GroupCommand> group;
        const bool ready = runtime->prepare_m3_group(
            candidates, *target, group, error);
        // Late backpressure affects lane 0 only.
        (void)channels[0][1]->push(Value::i32({99}));
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        const auto outcomes =
            ready ? runtime->finish_m3_group(group, error)
                  : std::vector<M1ExecuteOutcome>{};
        expect(
            outcomes.size() == 2 &&
                outcomes[0] == M1ExecuteOutcome::Retry &&
                outcomes[1] == M1ExecuteOutcome::Committed &&
                channels[1][1]->front().i[0] == 3,
            "partial readiness retries one lane without grouped fate");
        for (auto& fire : fires) runtime->release(fire);
        target->release_standalone_buffer(logits);
    }

    {
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits =
            target->create_standalone_buffer(16 * sizeof(std::uint16_t));
        auto* encoded = static_cast<std::uint16_t*>(logits.contents());
        for (std::size_t lane = 0; lane < 2; ++lane)
            for (std::size_t column = 0; column < 8; ++column)
                encoded[lane * 8 + column] =
                    bf16(column == lane + 4 ? 10.0f : 0.0f);
        target->make_resident();
        std::array<std::vector<std::shared_ptr<ChannelState>>, 2>
            channels;
        std::array<std::shared_ptr<M1PreparedFire>, 2> fires;
        std::vector<M3LaneCandidate> candidates;
        bool prepared = true;
        for (std::size_t lane = 0; lane < 2; ++lane) {
            for (const auto& channel : plan.trace.channels) {
                channels[lane].push_back(
                    make_platform_channel_state(
                        channel.type.dtype,
                        channel.type.shape.numel(),
                        channel.capacity));
            }
            (void)channels[lane][0]->push(Value::i32({1}));
            prepared =
                prepared &&
                runtime->prepare(
                    executable,
                    channels[lane],
                    tickets_for(plan, channels[lane]),
                    fires[lane],
                    error) == M1PrepareOutcome::Ready;
            M1DeviceInputs inputs;
            inputs.logits_bf16 = logits;
            inputs.logits_row_offset =
                static_cast<std::uint32_t>(lane);
            inputs.logits_row_count = 1;
            inputs.vocab = 8;
            inputs.extents.sampled_rows = 1;
            inputs.extents.row_count = 1;
            candidates.push_back({
                .fire = fires[lane],
                .inputs = inputs,
                .retry_ineligible = true,
            });
        }
        std::shared_ptr<M3GroupCommand> group;
        const bool ready =
            prepared &&
            runtime->prepare_m3_group(
                candidates, *target, group, error);
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        const auto outcomes =
            ready
                ? runtime->finish_m3_group(group, error)
                : std::vector<M1ExecuteOutcome>{};
        expect(
            outcomes ==
                std::vector<M1ExecuteOutcome>({
                    M1ExecuteOutcome::Committed,
                    M1ExecuteOutcome::Committed}),
            "definitively ready RS/prologue-style retry-ineligible lanes "
            "are admitted to production M3 grouping (" +
                error + ")");
        for (auto& fire : fires) runtime->release(fire);
        target->release_standalone_buffer(logits);
    }

    {
        auto target = RawMetalContext::create(4u << 20);
        target->make_resident();
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : plan.trace.channels) {
            channels.push_back(
                make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
        }
        (void)channels[0]->push(Value::i32({1}));
        std::shared_ptr<M1PreparedFire> fire;
        const bool prepared =
            runtime->prepare(
                executable,
                channels,
                tickets_for(plan, channels),
                fire,
                error) == M1PrepareOutcome::Ready;
        (void)channels[1]->push(Value::i32({99}));
        const M3GroupStats before = runtime->m3_stats();
        std::shared_ptr<M3GroupCommand> aborted;
        const bool accepted =
            prepared &&
            runtime->prepare_m3_group(
                {{.fire = fire}},
                *target,
                aborted,
                error);
        const M3GroupStats after = runtime->m3_stats();
        expect(
            !accepted && aborted == nullptr &&
                after.readiness_launches ==
                    before.readiness_launches &&
                after.body_launches == before.body_launches &&
                after.commit_launches == before.commit_launches,
            "definitive pre-launch abort records zero readiness/body/commit "
            "dispatches");
        runtime->release(fire);
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const RunResult adversarial = run_greedy(
        *runtime,
        executable,
        plan,
        {nan, -INFINITY, 4, 4, nan, 0, 0, 0});
    expect(
        adversarial.execute == M1ExecuteOutcome::Committed &&
            adversarial.token == 2,
        "generated argmax ignores NaNs and keeps the lower tie index (execute=" +
            std::to_string(static_cast<int>(adversarial.execute)) +
            " token=" + std::to_string(adversarial.token) +
            " error=" + adversarial.error + ")");

    const RunResult retry = run_greedy(
        *runtime,
        executable,
        plan,
        {0, 1, 9, 2, 0, 0, 0, 3},
        true);
    expect(
        retry.prepare == M1PrepareOutcome::Retry,
        "device readiness returns retry without executing");

    {
        const std::size_t external_before =
            runtime->context().external_buffer_count();
        const TransientBufferPoolStats pool_before =
            runtime->context().transient_buffer_pool_stats();
        const M0TimingSnapshot timing_before =
            m0_timing_counters().snapshot();
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : plan.trace.channels) {
            channels.push_back(
                make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
        }
        (void)channels[0]->push(Value::i32({1}));
        std::shared_ptr<M1PreparedFire> fire;
        const bool prepared =
            runtime->prepare(
                executable,
                channels,
                tickets_for(plan, channels),
                fire,
                error) == M1PrepareOutcome::Ready;
        SlotHandle logits =
            runtime->context().create_standalone_buffer(
                8 * sizeof(std::uint16_t));
        const std::array<float, 8> values{0, 1, 9, 2, 0, 0, 0, 3};
        auto* encoded =
            static_cast<std::uint16_t*>(logits.contents());
        for (std::size_t index = 0; index < values.size(); ++index) {
            encoded[index] = bf16(values[index]);
        }
        M1DeviceInputs inputs;
        inputs.logits_bf16 = logits;
        inputs.logits_row_count = 1;
        inputs.vocab = 8;
        inputs.extents.sampled_rows = 1;
        inputs.extents.row_count = 1;
        runtime->context().force_next_wait_timeout_for_test();
        const M1ExecuteOutcome outcome =
            prepared
                ? runtime->execute(fire, inputs, error)
                : M1ExecuteOutcome::Failed;
        runtime->context().release_standalone_buffer(logits);
        runtime->release(fire);
        const M0TimingSnapshot timing_after =
            m0_timing_counters().snapshot();
        const TransientBufferPoolStats pool_after =
            runtime->context().transient_buffer_pool_stats();
        expect(
            outcome == M1ExecuteOutcome::Committed &&
                channels[0]->head() == 1 &&
                channels[1]->tail() == 1 &&
                channels[1]->front().i[0] == 2 &&
                channels[0]->poison() == 0 &&
                channels[1]->poison() == 0 &&
                timing_after.forward_wait_timeouts ==
                    timing_before.forward_wait_timeouts + 1 &&
                runtime->context().external_buffer_count() ==
                    external_before &&
                pool_after.recycles > pool_before.recycles &&
                pool_after.in_use_buffers == 0 &&
                pool_after.resident_bytes <= pool_after.capacity_bytes,
            "an initial wait timeout drains to the fence, reports telemetry, "
            "preserves channel epochs, and recycles scratch only afterward");
    }

    std::vector<std::uint8_t> gumbel_container;
    std::vector<std::uint8_t> gumbel_sidecar;
    ExecPlan gumbel_plan;
    expect(
        load_golden(
            "section3_masked_gumbel",
            gumbel_container,
            gumbel_sidecar) &&
            build_exec_plan(
                gumbel_container.data(),
                gumbel_container.size(),
                gumbel_sidecar.data(),
                gumbel_sidecar.size(),
                gumbel_plan,
                &error),
        "decoded masked keyed-RNG PTIB v2 golden");
    auto gumbel_executable = runtime->compile_program(
        0xf97fec496c3a74edULL, gumbel_plan, error);
    expect(
        gumbel_executable != nullptr,
        "compiled masked keyed-RNG singleton regions (" + error + ")");
    {
        std::vector<std::uint8_t> pivot_container;
        std::vector<std::uint8_t> pivot_sidecar;
        ExecPlan malformed;
        const bool decoded =
            load_golden(
                "pivot_predicates_multistage",
                pivot_container,
                pivot_sidecar) &&
            build_exec_plan(
                pivot_container.data(),
                pivot_container.size(),
                pivot_sidecar.data(),
                pivot_sidecar.size(),
                malformed,
                &error);
        bool corrupted = false;
        for (auto& stage : malformed.region_plans) {
            auto pivot = std::find_if(
                stage.ops.begin(),
                stage.ops.end(),
                [](const auto& value) {
                    return value.op.tag ==
                        PTIR_OP_PIVOT_THRESHOLD;
                });
            if (pivot == stage.ops.end()) continue;
            pivot->op.pred_payload =
                static_cast<std::uint32_t>(
                    stage.value_types.size());
            stage.signature.push_back(0x94);
            stage.signature_hash =
                pie_native::ptir::container::fnv1a64(
                    stage.signature.data(),
                    stage.signature.size());
            corrupted = true;
            break;
        }
        error.clear();
        expect(
            decoded && corrupted &&
                runtime->compile_program(
                0xa100000000000004ULL, malformed, error) == nullptr &&
                error.find("predicate payload") != std::string::npos,
            "registration rejects an out-of-range predicate payload (" +
                error + ")");
    }
    M1RuntimeExtents signature_extents;
    expect(
        runtime->m3_stage_group_key(
            executable, PTIR_STAGE_EPILOGUE, signature_extents) !=
            runtime->m3_stage_group_key(
                gumbel_executable,
                PTIR_STAGE_EPILOGUE,
                signature_extents),
        "mixed canonical signatures form separate ready groups");
    if (gumbel_executable) {
        std::vector<std::shared_ptr<ChannelState>> device_channels;
        for (const auto& channel : gumbel_plan.trace.channels) {
            device_channels.push_back(make_platform_channel_state(
                channel.type.dtype,
                channel.type.shape.numel(),
                channel.capacity));
        }
        (void)device_channels[0]->push(Value::i32({1}));
        (void)device_channels[2]->push(
            Value::boolean(std::vector<std::uint8_t>(32, 1)));
        (void)device_channels[3]->push(Value::u32({1}));
        (void)device_channels[4]->push(Value::u32({1234, 0}));
        const auto gumbel_tickets =
            tickets_for(gumbel_plan, device_channels);
        std::shared_ptr<M1PreparedFire> fire;
        const M1PrepareOutcome prepared = runtime->prepare(
            gumbel_executable,
            device_channels,
            gumbel_tickets,
            fire,
            error);
        std::vector<float> gumbel_logits(32, 0.0f);
        gumbel_logits[7] = 100.0f;
        SlotHandle logits_buffer =
            runtime->context().create_standalone_buffer(
                gumbel_logits.size() * sizeof(std::uint16_t));
        auto* encoded =
            static_cast<std::uint16_t*>(logits_buffer.contents());
        for (std::size_t index = 0; index < gumbel_logits.size(); ++index) {
            encoded[index] = bf16(gumbel_logits[index]);
        }
        M1DeviceInputs inputs;
        inputs.logits_bf16 = logits_buffer;
        inputs.logits_row_count = 1;
        inputs.vocab = 32;
        inputs.extents.sampled_rows = 1;
        inputs.extents.row_count = 1;
        const M1ExecuteOutcome generated =
            prepared == M1PrepareOutcome::Ready
                ? runtime->execute(fire, inputs, error)
                : M1ExecuteOutcome::Failed;

        std::map<std::uint32_t, std::shared_ptr<ChannelState>> externs;
        std::map<std::uint32_t, Value> seeds{
            {0, Value::i32({1})},
            {3, Value::u32({1})},
            {4, Value::u32({1234, 0})},
        };
        InterpInstance reference =
            make_instance(gumbel_plan, externs, seeds);
        (void)host_put(
            reference,
            gumbel_plan,
            2,
            Value::boolean(std::vector<std::uint8_t>(32, 1)));
        PassInputs reference_inputs{
            .logits = gumbel_logits.data(),
            .rows = 1,
            .vocab = 32,
            .mtp_draft_row = -1,
        };
        const StepResult reference_result =
            step(reference, gumbel_plan, reference_inputs);
        bool same_channels =
            generated == M1ExecuteOutcome::Committed &&
            reference_result.ok && reference_result.committed;
        for (std::size_t channel = 0;
             channel < device_channels.size() && same_channels;
             ++channel) {
            const Value device_value = device_channels[channel]->current();
            const Value reference_value =
                reference.channels[channel]->current();
            std::vector<std::uint8_t> device_wire(
                wire_cell_bytes(device_value.dtype, device_value.len()));
            std::vector<std::uint8_t> reference_wire(
                wire_cell_bytes(
                    reference_value.dtype, reference_value.len()));
            encode_wire(device_value, device_wire.data());
            encode_wire(reference_value, reference_wire.data());
            same_channels =
                device_channels[channel]->head() ==
                    reference.channels[channel]->head() &&
                device_channels[channel]->tail() ==
                    reference.channels[channel]->tail() &&
                device_wire == reference_wire;
        }
        expect(
            same_channels && device_channels[1]->front().i[0] == 7,
            "generated mask/RNG/recurrent-channel state matches Rust oracle");
        runtime->context().release_standalone_buffer(logits_buffer);
        runtime->release(fire);
    }

    std::vector<std::uint8_t> matrix_container;
    std::vector<std::uint8_t> matrix_sidecar;
    ExecPlan matrix_plan;
    error.clear();
    const bool matrix_decoded =
        load_golden(
            "matrix_mask_apply_packed",
            matrix_container,
            matrix_sidecar) &&
        build_exec_plan(
            matrix_container.data(),
            matrix_container.size(),
            matrix_sidecar.data(),
            matrix_sidecar.size(),
            matrix_plan,
            &error);
    auto matrix_executable =
        matrix_decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      matrix_container.data(), matrix_container.size()),
                  matrix_plan,
                  error)
            : nullptr;
    auto matrix_output = make_platform_channel_state(DType::I32, 2, 1);
    const std::vector<float> matrix_logits = {
        0, 0, 9, 1, 0, 2, 0, 0,
        0, 0, 0, 4, 0, 3, 0, 9,
    };
    const M1ExecuteOutcome matrix_outcome =
        matrix_executable
            ? execute_generated(
                  *runtime,
                  matrix_executable,
                  matrix_plan,
                  {matrix_output},
                  matrix_logits,
                  2,
                  8,
                  error)
            : M1ExecuteOutcome::Failed;
    const Value matrix_value = matrix_output->current();
    expect(
        matrix_outcome == M1ExecuteOutcome::Committed &&
            matrix_value.i == std::vector<std::int32_t>({5, 3}),
        "generated matrix mask/argmax matches Rust golden");

    {
        std::vector<std::uint8_t> mask_container;
        std::vector<std::uint8_t> mask_sidecar;
        ExecPlan mask_plan;
        error.clear();
        const bool decoded =
            load_golden(
                "structured_masks",
                mask_container,
                mask_sidecar) &&
            build_exec_plan(
                mask_container.data(),
                mask_container.size(),
                mask_sidecar.data(),
                mask_sidecar.size(),
                mask_plan,
                &error);
        const std::array<std::uint8_t, 10> row_membership_tags{
            PTIR_OP_RESHAPE,
            PTIR_OP_IOTA,
            PTIR_OP_DIV,
            PTIR_OP_REM,
            PTIR_OP_MUL,
            PTIR_OP_ADD,
            PTIR_OP_GATHER,
            PTIR_OP_EQ,
            PTIR_OP_CAST,
            PTIR_OP_REDUCE_MAX,
        };
        const bool general_row_membership =
            decoded &&
            std::all_of(
                row_membership_tags.begin(),
                row_membership_tags.end(),
                [&](std::uint8_t tag) {
                    return std::any_of(
                        mask_plan.region_plans[0].ops.begin(),
                        mask_plan.region_plans[0].ops.end(),
                        [&](const auto& op) {
                            return op.op.tag == tag;
                        });
                }) &&
            std::none_of(
                mask_plan.region_plans[0].fused.regions.begin(),
                mask_plan.region_plans[0].fused.regions.end(),
                [](const auto& region) {
                    return region.library;
                });
        auto executable = decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      mask_container.data(),
                      mask_container.size()),
                  mask_plan,
                  error)
            : nullptr;
        const std::array<std::vector<std::uint8_t>, 4> expected{{
            {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1},
            {0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1},
            {1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1},
            {1, 1, 1, 0, 0, 1, 1, 1},
        }};
        auto make_channels = [&] {
            std::vector<std::shared_ptr<ChannelState>> result;
            for (const auto& channel : mask_plan.trace.channels) {
                result.push_back(
                    make_platform_channel_state(
                        channel.type.dtype,
                        channel.type.shape.numel(),
                        channel.capacity));
            }
            if (result.size() == 7) {
                (void)result[0]->push(Value::u32({3, 5}));
                (void)result[1]->push(
                    Value::u32({0, 1, 2, 1, 2, 3}));
                (void)result[2]->push(
                    Value::u32({0, 1, 2, 3}));
            }
            return result;
        };
        auto outputs_match = [&](const auto& channels) {
            if (channels.size() != 7) return false;
            for (std::size_t output = 0;
                 output < expected.size();
                 ++output) {
                if (channels[output + 3]->front().b !=
                    expected[output]) {
                    return false;
                }
            }
            return true;
        };
        auto solo_channels = make_channels();
        const M1ExecuteOutcome solo =
            executable
                ? execute_generated(
                      *runtime,
                      executable,
                      mask_plan,
                      solo_channels,
                      {},
                      0,
                      0,
                      error,
                      -1,
                      M1ExecutionMode::Singleton)
                : M1ExecuteOutcome::Failed;
        auto generated_channels = make_channels();
        const M1ExecuteOutcome generated =
            executable
                ? execute_generated(
                      *runtime,
                      executable,
                      mask_plan,
                      generated_channels,
                      {},
                      0,
                      0,
                      error,
                      -1,
                      M1ExecutionMode::Fused)
                : M1ExecuteOutcome::Failed;
        std::array<std::vector<std::shared_ptr<ChannelState>>, 2>
            grouped_channels{
                make_channels(),
                make_channels(),
            };
        std::array<std::shared_ptr<M1PreparedFire>, 2> fires;
        std::vector<M3LaneCandidate> candidates;
        bool prepared = executable != nullptr;
        for (std::size_t lane = 0; lane < grouped_channels.size();
             ++lane) {
            prepared =
                prepared &&
                runtime->prepare(
                    executable,
                    grouped_channels[lane],
                    tickets_for(
                        mask_plan, grouped_channels[lane]),
                    fires[lane],
                    error) == M1PrepareOutcome::Ready;
            candidates.push_back({.fire = fires[lane]});
        }
        auto target = RawMetalContext::create(4u << 20);
        target->make_resident();
        std::shared_ptr<M3GroupCommand> group;
        const bool ready =
            prepared &&
            runtime->prepare_m3_group(
                candidates,
                *target,
                group,
                error);
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        const auto outcomes =
            ready
                ? runtime->finish_m3_group(group, error)
                : std::vector<M1ExecuteOutcome>{};
        bool exact =
            general_row_membership &&
            solo == M1ExecuteOutcome::Committed &&
            generated == M1ExecuteOutcome::Committed &&
            outputs_match(solo_channels) &&
            outputs_match(generated_channels) &&
            outcomes ==
                std::vector<M1ExecuteOutcome>{
                    M1ExecuteOutcome::Committed,
                    M1ExecuteOutcome::Committed};
        for (const auto& channels : grouped_channels) {
            exact =
                exact &&
                outputs_match(channels);
        }
        expect(
            exact,
            "row_membership general SSA and semantic masks match "
            "authoritative structured_masks.txt in singleton, fused, and "
            "N=2 grouped execution (" +
                error + ")");
        for (auto& fire : fires) runtime->release(fire);
    }

    std::vector<std::uint8_t> mtp_container;
    std::vector<std::uint8_t> mtp_sidecar;
    ExecPlan mtp_plan;
    error.clear();
    const bool mtp_decoded =
        load_golden("mtp_verify_tail", mtp_container, mtp_sidecar) &&
        build_exec_plan(
            mtp_container.data(),
            mtp_container.size(),
            mtp_sidecar.data(),
            mtp_sidecar.size(),
            mtp_plan,
            &error);
    auto mtp_executable =
        mtp_decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      mtp_container.data(), mtp_container.size()),
                  mtp_plan,
                  error)
            : nullptr;
    std::vector<std::shared_ptr<ChannelState>> mtp_channels;
    for (const auto& channel : mtp_plan.trace.channels) {
        mtp_channels.push_back(
            make_platform_channel_state(
                channel.type.dtype,
                channel.type.shape.numel(),
                channel.capacity));
    }
    if (mtp_channels.size() == 4) {
        (void)mtp_channels[0]->push(Value::i32({3, 5, 6}));
        (void)mtp_channels[1]->push(Value::boolean({
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 1, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        }));
    }
    const std::vector<float> mtp_logits{
        0, 0, 0, 9, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 9, 0, 0,
        0, 0, 1, 0, 0, 0, 9, 0,
        0, 0, 0, 0, 9, 0, 0, 0,
        0, 7, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 7, 0, 0, 0,
        7, 0, 0, 0, 0, 0, 0, 0,
    };
    const M1ExecuteOutcome mtp_outcome =
        mtp_executable
            ? execute_generated(
                  *runtime,
                  mtp_executable,
                  mtp_plan,
                  mtp_channels,
                  mtp_logits,
                  7,
                  8,
                  error,
                  4,
                  M1ExecutionMode::Singleton)
            : M1ExecuteOutcome::Failed;
    std::vector<std::shared_ptr<ChannelState>> mtp_fused_channels;
    for (const auto& channel : mtp_plan.trace.channels) {
        mtp_fused_channels.push_back(
            make_platform_channel_state(
                channel.type.dtype,
                channel.type.shape.numel(),
                channel.capacity));
    }
    if (mtp_fused_channels.size() == 4) {
        (void)mtp_fused_channels[0]->push(
            Value::i32({3, 5, 6}));
        (void)mtp_fused_channels[1]->push(Value::boolean({
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 1, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        }));
    }
    const M1ExecuteOutcome mtp_fused_outcome =
        mtp_executable
            ? execute_generated(
                  *runtime,
                  mtp_executable,
                  mtp_plan,
                  mtp_fused_channels,
                  mtp_logits,
                  7,
                  8,
                  error,
                  4,
                  M1ExecutionMode::Fused)
            : M1ExecuteOutcome::Failed;
    expect(
        mtp_outcome == M1ExecuteOutcome::Committed &&
            mtp_fused_outcome ==
                M1ExecuteOutcome::Committed &&
            mtp_channels[2]->front().i ==
                std::vector<std::int32_t>({3, 5, 2, -1}) &&
            mtp_channels[3]->front().i ==
                std::vector<std::int32_t>({1, 4, 0}) &&
            mtp_fused_channels[2]->front().i ==
                mtp_channels[2]->front().i &&
            mtp_fused_channels[3]->front().i ==
                mtp_channels[3]->front().i,
        "bounded generated MtpLogits rows preserve row base 4 in singleton "
        "and fused execution (" +
            error + "; outcome=" +
            std::to_string(static_cast<int>(mtp_outcome)) +
            "; out2=" +
            (mtp_channels.size() > 2 &&
                     !mtp_channels[2]->front().i.empty()
                 ? (std::to_string(mtp_channels[2]->front().i[0]) + "," +
                    std::to_string(mtp_channels[2]->front().i[1]) + "," +
                    std::to_string(mtp_channels[2]->front().i[2]) + "," +
                    std::to_string(mtp_channels[2]->front().i[3]) + "; out3=" +
                    std::to_string(mtp_channels[3]->front().i[0]) + "," +
                    std::to_string(mtp_channels[3]->front().i[1]) + "," +
                    std::to_string(mtp_channels[3]->front().i[2]))
                 : "empty") +
            ")");
    {
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : mtp_plan.trace.channels) {
            channels.push_back(
                make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
        }
        (void)channels[0]->push(Value::i32({3, 5, 6}));
        (void)channels[1]->push(Value::boolean({
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 1, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        }));
        std::shared_ptr<M1PreparedFire> fire;
        const bool prepared =
            runtime->prepare(
                mtp_executable,
                channels,
                tickets_for(mtp_plan, channels),
                fire,
                error) == M1PrepareOutcome::Ready;
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits = target->create_standalone_buffer(
            mtp_logits.size() * sizeof(std::uint16_t));
        auto* encoded =
            static_cast<std::uint16_t*>(logits.contents());
        for (std::size_t index = 0; index < mtp_logits.size();
             ++index) {
            encoded[index] = bf16(mtp_logits[index]);
        }
        target->make_resident();
        M1DeviceInputs inputs;
        inputs.logits_bf16 = logits;
        inputs.logits_row_count = 7;
        inputs.vocab = 8;
        inputs.mtp_draft_row = 4;
        inputs.extents.sampled_rows = 7;
        inputs.extents.row_count = 7;
        std::shared_ptr<M2CommandPlan> command;
        const bool ready =
            prepared &&
            runtime->prepare_m2_command(
                fire, inputs, *target, command, error);
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m2_pre(command, encoder);
                runtime->encode_m2_post(command, encoder);
            });
        }
        const M1ExecuteOutcome outcome =
            ready
                ? runtime->finish_m2_command(command, error)
                : M1ExecuteOutcome::Failed;
        expect(
            outcome == M1ExecuteOutcome::Committed &&
                channels[2]->front().i ==
                    std::vector<std::int32_t>({3, 5, 2, -1}) &&
                channels[3]->front().i ==
                    std::vector<std::int32_t>({1, 4, 0}),
            "M2 fused intrinsics apply MtpLogits row base 4 without "
            "aliasing Logits row zero (" +
                error + ")");
        runtime->release(fire);
        target->release_standalone_buffer(logits);
    }
    {
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : mtp_plan.trace.channels) {
            channels.push_back(
                make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
        }
        (void)channels[0]->push(Value::i32({3, 5, 6}));
        (void)channels[1]->push(Value::boolean({
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 1, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        }));
        std::shared_ptr<M1PreparedFire> fire;
        const bool prepared =
            runtime->prepare(
                mtp_executable,
                channels,
                tickets_for(mtp_plan, channels),
                fire,
                error) == M1PrepareOutcome::Ready;
        SlotHandle logits =
            runtime->context().create_standalone_buffer(
                mtp_logits.size() * sizeof(std::uint16_t));
        auto* encoded =
            static_cast<std::uint16_t*>(logits.contents());
        for (std::size_t index = 0; index < mtp_logits.size(); ++index) {
            encoded[index] = bf16(mtp_logits[index]);
        }
        LogitsOut output;
        output.rows = 7;
        output.vocab = 8;
        output.device_buffer = logits.buffer;
        output.device_contents = logits.contents();
        output.device_gpu_address = logits.gpu_address;
        output.device_bytes = logits.size;
        MemberForwardDesc desc;
        desc.readout_local_indices = {0, 1, 2, 3, 4, 5, 6};
        M1DeviceInputs fallback_inputs =
            m1_singleton_fallback_inputs(output, desc, 4);
        const M1ExecuteOutcome outcome =
            prepared
                ? runtime->execute(
                      fire,
                      fallback_inputs,
                      error,
                      M1ExecutionMode::Singleton)
                : M1ExecuteOutcome::Failed;
        expect(
            fallback_inputs.mtp_draft_row == 4 &&
                outcome == M1ExecuteOutcome::Committed &&
                channels[3]->front().i ==
                    std::vector<std::int32_t>({1, 4, 0}),
            "Context singleton fallback preserves later MTP draft row base 4 "
            "(" +
                error + ")");
        runtime->context().release_standalone_buffer(logits);
        runtime->release(fire);
    }
    {
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : mtp_plan.trace.channels) {
            channels.push_back(
                make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
        }
        (void)channels[0]->push(Value::i32({3, 5, 6}));
        (void)channels[1]->push(Value::boolean({
            1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 1, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1,
        }));
        std::shared_ptr<M1PreparedFire> fire;
        const bool prepared =
            runtime->prepare(
                mtp_executable,
                channels,
                tickets_for(mtp_plan, channels),
                fire,
                error) == M1PrepareOutcome::Ready;
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits = target->create_standalone_buffer(
            mtp_logits.size() * sizeof(std::uint16_t));
        auto* encoded =
            static_cast<std::uint16_t*>(logits.contents());
        for (std::size_t index = 0; index < mtp_logits.size(); ++index) {
            encoded[index] = bf16(mtp_logits[index]);
        }
        target->make_resident();
        M1DeviceInputs inputs;
        inputs.logits_bf16 = logits;
        inputs.logits_row_count = 7;
        inputs.logits_rows = {0, 1, 2, 3, 4, 5, 6};
        inputs.vocab = 8;
        inputs.mtp_draft_row = 4;
        inputs.extents.sampled_rows = 7;
        inputs.extents.row_count = 7;
        std::shared_ptr<M3GroupCommand> group;
        const bool ready =
            prepared &&
            runtime->prepare_m3_group(
                {{.fire = fire, .inputs = inputs}},
                *target,
                group,
                error);
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        const auto outcomes =
            ready
                ? runtime->finish_m3_group(group, error)
                : std::vector<M1ExecuteOutcome>{};
        expect(
            outcomes ==
                    std::vector<M1ExecuteOutcome>{
                        M1ExecuteOutcome::Committed} &&
                channels[2]->front().i ==
                    std::vector<std::int32_t>({3, 5, 2, -1}) &&
                channels[3]->front().i ==
                    std::vector<std::int32_t>({1, 4, 0}),
            "Context-supplied MTP row base keeps grouped Logits rows 0..3 "
            "and MtpLogits rows 4..6 distinct (" +
                error + ")");
        runtime->release(fire);
        target->release_standalone_buffer(logits);
    }
    const auto drafts_container = mtp_drafts_container();
    const auto drafts_sidecar = mtp_drafts_sidecar(drafts_container);
    ExecPlan drafts_plan;
    error.clear();
    const bool drafts_decoded =
        build_exec_plan(
            drafts_container.data(),
            drafts_container.size(),
            drafts_sidecar.data(),
            drafts_sidecar.size(),
            drafts_plan,
            &error);
    auto drafts_executable = drafts_decoded
        ? runtime->compile_program(
              pie_native::ptir::container::fnv1a64(
                  drafts_container.data(), drafts_container.size()),
              drafts_plan,
              error)
        : nullptr;
    std::vector<std::shared_ptr<ChannelState>> draft_channels{
        make_platform_channel_state(DType::I32, 2, 1)};
    const M1ExecuteOutcome drafts_outcome =
        drafts_executable
            ? execute_generated(
                  *runtime,
                  drafts_executable,
                  drafts_plan,
                  draft_channels,
                  {
                      9, 0, 0, 0,
                      0, 9, 0, 0,
                      0, 1, 9, 2,
                      0, 7, 1, 7,
                  },
                  4,
                  4,
                  error,
                  2,
                  M1ExecutionMode::Fused)
            : M1ExecuteOutcome::Failed;
    expect(
        drafts_outcome == M1ExecuteOutcome::Committed &&
            draft_channels[0]->front().i ==
                std::vector<std::int32_t>({2, 1}),
        "fused MtpDrafts applies later row base 2 without CPU vocab work (" +
            error + ")");

    {
        const auto boundary_container =
            semantic_boundary_container();
        const auto boundary_sidecar =
            semantic_boundary_sidecar(boundary_container);
        ExecPlan boundary_plan;
        error.clear();
        const bool decoded = build_exec_plan(
            boundary_container.data(),
            boundary_container.size(),
            boundary_sidecar.data(),
            boundary_sidecar.size(),
            boundary_plan,
            &error);
        auto boundary_program = decoded
            ? runtime->compile_program(
                  0xa200000000000001ULL,
                  boundary_plan,
                  error)
            : nullptr;
        std::vector<std::shared_ptr<ChannelState>> channels{
            make_platform_channel_state(DType::U32, 2, 1),
            make_platform_channel_state(DType::U32, 2, 1),
        };
        (void)channels[0]->push(Value::u32({7, 9}));
        const M1ExecuteOutcome outcome =
            boundary_program
                ? execute_generated(
                      *runtime,
                      boundary_program,
                      boundary_plan,
                      channels,
                      {},
                      0,
                      0,
                      error,
                      -1,
                      M1ExecutionMode::Fused)
                : M1ExecuteOutcome::Failed;
        const auto unknown_container =
            semantic_boundary_container(false);
        const auto unknown_sidecar =
            semantic_boundary_sidecar(unknown_container);
        ExecPlan unknown_plan;
        const bool unknown_decoded = build_exec_plan(
            unknown_container.data(),
            unknown_container.size(),
            unknown_sidecar.data(),
            unknown_sidecar.size(),
            unknown_plan,
            &error);
        expect(
            outcome == M1ExecuteOutcome::Committed &&
                channels[1]->front().u ==
                    std::vector<std::uint32_t>({7, 9}) &&
                unknown_decoded && !unknown_plan.executable &&
                unknown_plan.reject_reason.find(
                    "semantic") !=
                    std::string::npos,
            "explicit metal.identity/metal.discard semantic boundaries "
            "execute, while unknown second-party boundaries reject "
            "(decoded=" + std::to_string(decoded) +
            ", program=" + std::to_string(boundary_program != nullptr) +
            ", outcome=" +
            std::to_string(static_cast<int>(outcome)) +
            ", unknown_decoded=" +
            std::to_string(unknown_decoded) +
            ", unknown_executable=" +
            std::to_string(unknown_plan.executable) +
            ", output=" +
            (channels[1]->front().u.size() == 2
                 ? std::to_string(channels[1]->front().u[0]) + "," +
                       std::to_string(channels[1]->front().u[1])
                 : "empty") +
            ", error=" + error + ")");
    }

    const std::vector<std::uint8_t> last_container =
        last_put_container();
    const std::vector<std::uint8_t> last_sidecar =
        last_put_sidecar(last_container);
    ExecPlan last_plan;
    expect(
        build_exec_plan(
            last_container.data(),
            last_container.size(),
            last_sidecar.data(),
            last_sidecar.size(),
            last_plan,
            &error),
        "decoded repeated-put singleton plan");
    const std::uint64_t last_hash =
        pie_native::ptir::container::fnv1a64(
            last_container.data(), last_container.size());
    auto last_executable =
        runtime->compile_program(last_hash, last_plan, error);
    auto last_channel = make_platform_channel_state(DType::U32, 1, 1);
    (void)last_channel->push(Value::u32({9}));
    std::vector<std::shared_ptr<ChannelState>> last_channels{last_channel};
    const auto last_tickets = tickets_for(last_plan, last_channels);
    std::shared_ptr<M1PreparedFire> last_fire;
    const M1PrepareOutcome last_prepared = runtime->prepare(
        last_executable,
        last_channels,
        last_tickets,
        last_fire,
        error);
    const M1ExecuteOutcome last_executed =
        last_prepared == M1PrepareOutcome::Ready
            ? runtime->execute(last_fire, M1DeviceInputs{}, error)
            : M1ExecuteOutcome::Failed;
    expect(
        last_executed == M1ExecuteOutcome::Committed &&
            last_channel->head() == 1 && last_channel->tail() == 2 &&
            last_channel->front().u[0] == 2,
        "generated repeated puts preserve last-put-wins and one atomic commit");
    runtime->release(last_fire);

    const std::vector<std::uint8_t> register_container =
        put_then_take_container();
    const std::vector<std::uint8_t> register_sidecar =
        put_then_take_sidecar(register_container);
    ExecPlan register_plan;
    error.clear();
    const bool register_decoded = build_exec_plan(
        register_container.data(),
        register_container.size(),
        register_sidecar.data(),
        register_sidecar.size(),
        register_plan,
        &error);
    auto register_executable =
        register_decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      register_container.data(), register_container.size()),
                  register_plan,
                  error)
            : nullptr;
    std::vector<std::shared_ptr<ChannelState>> register_channels{
        make_platform_channel_state(DType::U32, 1, 1),
        make_platform_channel_state(DType::U32, 1, 1),
    };
    const M1ExecuteOutcome register_outcome =
        register_executable
            ? execute_generated(
                  *runtime,
                  register_executable,
                  register_plan,
                  register_channels,
                  {},
                  0,
                  0,
                  error)
            : M1ExecuteOutcome::Failed;
    expect(
        register_outcome == M1ExecuteOutcome::Committed &&
            register_channels[0]->head() == 0 &&
            register_channels[0]->tail() == 1 &&
            register_channels[0]->front().u[0] == 7 &&
            register_channels[1]->front().u[0] == 7,
        "generated same-stage put-then-take observes pending register value");
    std::vector<std::shared_ptr<ChannelState>> fused_register_channels{
        make_platform_channel_state(DType::U32, 1, 1),
        make_platform_channel_state(DType::U32, 1, 1),
    };
    const M1ExecuteOutcome fused_register_outcome =
        execute_generated(
            *runtime,
            register_executable,
            register_plan,
            fused_register_channels,
            {},
            0,
            0,
            error,
            -1,
            M1ExecutionMode::Fused);
    expect(
        fused_register_outcome == M1ExecuteOutcome::Committed &&
            fused_register_channels[0]->head() == 0 &&
            fused_register_channels[0]->tail() == 1 &&
            fused_register_channels[0]->front().u[0] == 7 &&
            fused_register_channels[1]->front().u[0] == 7,
        "fused same-stage put-then-take preserves register semantics");
    std::vector<std::shared_ptr<ChannelState>> full_register_channels{
        make_platform_channel_state(DType::U32, 1, 1),
        make_platform_channel_state(DType::U32, 1, 1),
    };
    (void)full_register_channels[0]->push(Value::u32({9}));
    std::shared_ptr<M1PreparedFire> full_register_fire;
    const M1PrepareOutcome full_register_prepare = runtime->prepare(
        register_executable,
        full_register_channels,
        tickets_for(register_plan, full_register_channels),
        full_register_fire,
        error);
    expect(
        full_register_prepare == M1PrepareOutcome::Retry &&
            full_register_channels[0]->head() == 0 &&
            full_register_channels[0]->tail() == 1 &&
            full_register_channels[0]->front().u[0] == 9,
        "full leading-put channel retries without consuming committed state");
    runtime->release(full_register_fire);

    const std::vector<std::uint8_t> phased_container =
        pre_post_container();
    const std::vector<std::uint8_t> phased_sidecar =
        pre_post_sidecar(phased_container);
    ExecPlan phased_plan;
    error.clear();
    const bool phased_decoded = build_exec_plan(
        phased_container.data(),
        phased_container.size(),
        phased_sidecar.data(),
        phased_sidecar.size(),
        phased_plan,
        &error);
    auto phased_executable =
        phased_decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      phased_container.data(), phased_container.size()),
                  phased_plan,
                  error)
            : nullptr;
    std::vector<std::shared_ptr<ChannelState>> phased_channels{
        make_platform_channel_state(DType::U32, 1, 1),
        make_platform_channel_state(DType::U32, 1, 1),
    };
    std::shared_ptr<M1PreparedFire> phased_fire;
    const M1PrepareOutcome phased_prepared =
        phased_executable
            ? runtime->prepare(
                  phased_executable,
                  phased_channels,
                  tickets_for(phased_plan, phased_channels),
                  phased_fire,
                  error)
            : M1PrepareOutcome::Failed;
    auto phased_context = RawMetalContext::create(4u << 20);
    SlotHandle phased_logits =
        phased_context->create_standalone_buffer(sizeof(std::uint16_t));
    phased_context->make_resident();
    M1DeviceInputs phased_inputs;
    phased_inputs.logits_bf16 = phased_logits;
    phased_inputs.logits_row_count = 1;
    phased_inputs.vocab = 1;
    phased_inputs.extents.sampled_rows = 1;
    phased_inputs.extents.row_count = 1;
    std::shared_ptr<M2CommandPlan> phased_command;
    const bool phased_ready =
        phased_prepared == M1PrepareOutcome::Ready &&
        runtime->prepare_m2_command(
            phased_fire,
            phased_inputs,
            *phased_context,
            phased_command,
            error);
    if (phased_ready) {
        phased_context->run_step([&](StepEncoder& encoder) {
            runtime->encode_m2_pre(phased_command, encoder);
            runtime->encode_m2_post(phased_command, encoder);
        });
    }
    const M1ExecuteOutcome phased_outcome =
        phased_ready
            ? runtime->finish_m2_command(phased_command, error)
            : M1ExecuteOutcome::Failed;
    expect(
        phased_outcome == M1ExecuteOutcome::Committed &&
            phased_channels[0]->head() == 0 &&
            phased_channels[0]->tail() == 1 &&
            phased_channels[0]->front().u[0] == 5 &&
            phased_channels[1]->front().u[0] == 5,
        "M2 places prologue before and epilogue after the forward boundary");
    phased_context->release_standalone_buffer(phased_logits);
    runtime->release(phased_fire);

    {
        const auto bytes = multistage_sink_base_container();
        const auto sidecar = multistage_sink_base_sidecar(bytes);
        ExecPlan sink_plan;
        error.clear();
        const bool decoded = build_exec_plan(
            bytes.data(), bytes.size(),
            sidecar.data(), sidecar.size(),
            sink_plan, &error);
        auto program = decoded
            ? runtime->compile_program(
                  0xa300000000000001ULL, sink_plan, error)
            : nullptr;
        std::vector<std::shared_ptr<ChannelState>> channels{
            make_platform_channel_state(DType::U32, 1, 1),
            make_platform_channel_state(DType::U32, 1, 1),
        };
        const M1ExecuteOutcome outcome =
            program
                ? execute_generated(
                      *runtime,
                      program,
                      sink_plan,
                      channels,
                      std::vector<float>(16, 0.0f),
                      1,
                      16,
                      error,
                      -1,
                      M1ExecutionMode::Singleton)
                : M1ExecuteOutcome::Failed;
        expect(
            outcome == M1ExecuteOutcome::Committed &&
                channels[1]->front().u[0] == 5,
            "multi-stage singleton sink bounds use stage_value_base (" +
                error + ")");
    }

    {
        const auto first_container = pre_post_container(5, 2);
        const auto second_container = pre_post_container(9, 3);
        const auto first_sidecar =
            pre_post_sidecar(first_container, 2);
        const auto second_sidecar =
            pre_post_sidecar(second_container, 3);
        ExecPlan first_plan;
        ExecPlan second_plan;
        error.clear();
        const bool decoded =
            build_exec_plan(
                first_container.data(),
                first_container.size(),
                first_sidecar.data(),
                first_sidecar.size(),
                first_plan,
                &error) &&
            build_exec_plan(
                second_container.data(),
                second_container.size(),
                second_sidecar.data(),
                second_sidecar.size(),
                second_plan,
                &error);
        auto first_program = decoded
            ? runtime->compile_program(
                  0x5a5a000000000001ULL,
                  first_plan,
                  error,
                  first_container)
            : nullptr;
        auto second_program = decoded
            ? runtime->compile_program(
                  0x5a5a000000000002ULL,
                  second_plan,
                  error,
                  second_container)
            : nullptr;
        std::array<std::vector<std::shared_ptr<ChannelState>>, 2>
            channels;
        std::array<std::shared_ptr<M1PreparedFire>, 2> fires;
        const std::array<const ExecPlan*, 2> plans{
            &first_plan, &second_plan};
        const std::array<std::shared_ptr<M1ProgramExecutable>, 2>
            programs{first_program, second_program};
        bool prepared = first_program && second_program;
        for (std::size_t lane = 0; lane < 2; ++lane) {
            for (const auto& channel : plans[lane]->trace.channels) {
                channels[lane].push_back(
                    make_platform_channel_state(
                        channel.type.dtype,
                        channel.type.shape.numel(),
                        channel.capacity));
            }
            prepared =
                prepared &&
                runtime->prepare(
                    programs[lane],
                    channels[lane],
                    tickets_for(*plans[lane], channels[lane]),
                    fires[lane],
                    error) == M1PrepareOutcome::Ready;
        }
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits =
            target->create_standalone_buffer(2 * sizeof(std::uint16_t));
        target->make_resident();
        std::vector<M3LaneCandidate> candidates;
        for (std::size_t lane = 0; lane < 2; ++lane) {
            M1DeviceInputs inputs;
            inputs.logits_bf16 = logits;
            inputs.logits_row_offset =
                static_cast<std::uint32_t>(lane);
            inputs.logits_row_count = 1;
            inputs.vocab = 1;
            inputs.extents.sampled_rows = 1;
            inputs.extents.row_count = 1;
            candidates.push_back({
                .fire = fires[lane],
                .inputs = inputs,
                .retry_ineligible = true,
            });
        }
        M1RuntimeExtents extents;
        const bool independently_keyed =
            first_program && second_program &&
            runtime->m3_stage_group_key(
                first_program, PTIR_STAGE_EPILOGUE, extents) ==
                runtime->m3_stage_group_key(
                    second_program, PTIR_STAGE_EPILOGUE, extents) &&
            runtime->m3_stage_group_key(
                first_program, PTIR_STAGE_PROLOGUE, extents) !=
                runtime->m3_stage_group_key(
                    second_program, PTIR_STAGE_PROLOGUE, extents);
        std::shared_ptr<M3GroupCommand> group;
        const bool grouped =
            prepared && independently_keyed &&
            runtime->prepare_m3_group(
                candidates, *target, group, error);
        if (grouped) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        const auto outcomes =
            grouped
                ? runtime->finish_m3_group(group, error)
                : std::vector<M1ExecuteOutcome>{};
        expect(
            outcomes ==
                    std::vector<M1ExecuteOutcome>({
                        M1ExecuteOutcome::Committed,
                        M1ExecuteOutcome::Committed}) &&
                channels[0][1]->front().u[0] == 5 &&
                channels[1][1]->front().u[0] == 9,
            "different prologues and unrelated channels share one canonical "
            "epilogue group while each prologue executes (" +
                error + ")");
        for (auto& fire : fires) runtime->release(fire);
        target->release_standalone_buffer(logits);
    }

    {
        const auto bytes = pre_post_container(5, 2);
        const auto sidecar = pre_post_sidecar(bytes, 2);
        ExecPlan first_plan;
        ExecPlan second_plan;
        error.clear();
        const bool decoded =
            build_exec_plan(
                bytes.data(), bytes.size(),
                sidecar.data(), sidecar.size(),
                first_plan, &error);
        second_plan = first_plan;
        if (decoded) {
            auto epilogue = std::find_if(
                second_plan.region_plans.begin(),
                second_plan.region_plans.end(),
                [](const auto& stage) {
                    return stage.stage == PTIR_STAGE_EPILOGUE;
                });
            epilogue->signature.push_back(0xe1);
            epilogue->signature_hash =
                pie_native::ptir::container::fnv1a64(
                    epilogue->signature.data(),
                    epilogue->signature.size());
        }
        auto first = decoded
            ? runtime->compile_program(
                  0xa400000000000001ULL,
                  first_plan,
                  error)
            : nullptr;
        auto second = decoded
            ? runtime->compile_program(
                  0xa400000000000002ULL,
                  second_plan,
                  error)
            : nullptr;
        std::array<std::vector<std::shared_ptr<ChannelState>>, 2>
            channels;
        std::array<std::shared_ptr<M1PreparedFire>, 2> fires;
        std::vector<M3LaneCandidate> candidates;
        const std::array<std::shared_ptr<M1ProgramExecutable>, 2>
            programs{first, second};
        const std::array<const ExecPlan*, 2> plans{
            &first_plan, &second_plan};
        bool prepared = first && second;
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits =
            target->create_standalone_buffer(2 * sizeof(std::uint16_t));
        target->make_resident();
        for (std::size_t lane = 0; lane < 2; ++lane) {
            for (const auto& channel : plans[lane]->trace.channels) {
                channels[lane].push_back(
                    make_platform_channel_state(
                        channel.type.dtype,
                        channel.type.shape.numel(),
                        channel.capacity));
            }
            prepared =
                prepared &&
                runtime->prepare(
                    programs[lane],
                    channels[lane],
                    tickets_for(*plans[lane], channels[lane]),
                    fires[lane],
                    error) == M1PrepareOutcome::Ready;
            M1DeviceInputs inputs;
            inputs.logits_bf16 = logits;
            inputs.logits_row_offset =
                static_cast<std::uint32_t>(lane);
            inputs.logits_row_count = 1;
            inputs.vocab = 1;
            inputs.extents.sampled_rows = 1;
            inputs.extents.row_count = 1;
            candidates.push_back({
                .fire = fires[lane],
                .inputs = inputs,
                .retry_ineligible = true,
            });
        }
        M1RuntimeExtents extents;
        const bool independent =
            runtime->m3_stage_group_key(
                first, PTIR_STAGE_PROLOGUE, extents) ==
                runtime->m3_stage_group_key(
                    second, PTIR_STAGE_PROLOGUE, extents) &&
            runtime->m3_stage_group_key(
                first, PTIR_STAGE_EPILOGUE, extents) !=
                runtime->m3_stage_group_key(
                    second, PTIR_STAGE_EPILOGUE, extents);
        std::shared_ptr<M3GroupCommand> group;
        const bool ready =
            prepared && independent &&
            runtime->prepare_m3_group(
                candidates, *target, group, error);
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        const auto outcomes =
            ready
                ? runtime->finish_m3_group(group, error)
                : std::vector<M1ExecuteOutcome>{};
        expect(
            outcomes ==
                    std::vector<M1ExecuteOutcome>({
                        M1ExecuteOutcome::Committed,
                        M1ExecuteOutcome::Committed}) &&
                channels[0][1]->front().u[0] == 5 &&
                channels[1][1]->front().u[0] == 5,
            "shared prologue lanes group independently from distinct "
            "epilogue signatures (" +
                error + ")");
        for (auto& fire : fires) runtime->release(fire);
        target->release_standalone_buffer(logits);
    }

    const std::vector<std::uint8_t> zero_container =
        signed_zero_container();
    const std::vector<std::uint8_t> zero_sidecar =
        signed_zero_sidecar(zero_container);
    ExecPlan zero_plan;
    error.clear();
    const bool zero_decoded = build_exec_plan(
        zero_container.data(),
        zero_container.size(),
        zero_sidecar.data(),
        zero_sidecar.size(),
        zero_plan,
        &error);
    auto zero_executable =
        zero_decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      zero_container.data(), zero_container.size()),
                  zero_plan,
                  error)
            : nullptr;
    std::vector<std::shared_ptr<ChannelState>> zero_channels{
        make_platform_channel_state(DType::F32, 2, 1),
        make_platform_channel_state(DType::F32, 1, 1),
        make_platform_channel_state(DType::F32, 1, 1),
    };
    (void)zero_channels[0]->push(Value::f32({-0.0f, 0.0f}));
    const M1ExecuteOutcome zero_outcome =
        zero_executable
            ? execute_generated(
                  *runtime,
                  zero_executable,
                  zero_plan,
                  zero_channels,
                  {},
                  0,
                  0,
                  error)
            : M1ExecuteOutcome::Failed;
    expect(
        zero_outcome == M1ExecuteOutcome::Committed &&
            float_bits(zero_channels[1]->front().f[0]) ==
                float_bits(0.0f) &&
            float_bits(zero_channels[2]->front().f[0]) ==
                float_bits(-0.0f),
        "generated reduce max/min match Rust signed-zero bits");

    std::vector<std::shared_ptr<ChannelState>> reversed_zero_channels{
        make_platform_channel_state(DType::F32, 2, 1),
        make_platform_channel_state(DType::F32, 1, 1),
        make_platform_channel_state(DType::F32, 1, 1),
    };
    (void)reversed_zero_channels[0]->push(
        Value::f32({0.0f, -0.0f}));
    const M1ExecuteOutcome reversed_zero_outcome =
        execute_generated(
                *runtime,
                zero_executable,
                zero_plan,
                reversed_zero_channels,
                {},
                0,
                0,
                error,
                -1,
                M1ExecutionMode::Fused);
    expect(
        reversed_zero_outcome == M1ExecuteOutcome::Committed &&
                float_bits(reversed_zero_channels[1]->front().f[0]) ==
                    float_bits(0.0f) &&
                float_bits(reversed_zero_channels[2]->front().f[0]) ==
                    float_bits(-0.0f),
        "generated signed-zero max/min are operand-order independent");

    const auto rank_container = rank3_argmax_container();
    const auto rank_sidecar = rank3_argmax_sidecar(rank_container);
    ExecPlan rank_plan;
    error.clear();
    const bool rank_decoded = build_exec_plan(
        rank_container.data(),
        rank_container.size(),
        rank_sidecar.data(),
        rank_sidecar.size(),
        rank_plan,
        &error);
    auto rank_executable =
        rank_decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      rank_container.data(), rank_container.size()),
                  rank_plan,
                  error)
            : nullptr;
    std::vector<std::shared_ptr<ChannelState>> rank_channels{
        make_platform_channel_state(DType::F32, 8, 1),
        make_platform_channel_state(DType::F32, 4, 1),
        make_platform_channel_state(DType::I32, 2, 1),
        make_platform_channel_state(DType::I32, 1, 1),
    };
    (void)rank_channels[0]->push(
        Value::f32({1, 2, 4, 3, -1, -2, 8, 8}));
    (void)rank_channels[2]->push(Value::i32({-2, -1}));
    const M1ExecuteOutcome rank_outcome =
        rank_executable
            ? execute_generated(
                  *runtime,
                  rank_executable,
                  rank_plan,
                  rank_channels,
                  {},
                  0,
                  0,
                  error,
                  -1,
                  M1ExecutionMode::Fused)
            : M1ExecuteOutcome::Failed;
    expect(
        rank_outcome == M1ExecuteOutcome::Committed &&
            rank_channels[1]->front().f ==
                std::vector<float>({2, 4, -1, 8}) &&
            rank_channels[3]->front().i[0] == 1,
        "rank-3 row reduction and signed I32 argmax match the oracle");

    const auto ragged_bytes = ragged_container();
    const auto ragged_plan_bytes = ragged_sidecar(ragged_bytes);
    ExecPlan ragged_plan;
    error.clear();
    const bool ragged_decoded = build_exec_plan(
        ragged_bytes.data(),
        ragged_bytes.size(),
        ragged_plan_bytes.data(),
        ragged_plan_bytes.size(),
        ragged_plan,
        &error);
    if (ragged_decoded) {
        auto& stage = ragged_plan.region_plans[0];
        stage.value_types[0].dims[0] = {
            .symbolic = true,
            .value = PTIR_EXTENT_SAMPLED_ROWS,
        };
        stage.value_types[1].dims[0] = {
            .symbolic = true,
            .value = PTIR_EXTENT_SAMPLED_ROWS,
        };
    }
    auto ragged_executable =
        ragged_decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      ragged_bytes.data(), ragged_bytes.size()),
                  ragged_plan,
                  error)
            : nullptr;
    auto ragged_target = RawMetalContext::create(4u << 20);
    SlotHandle ragged_logits = ragged_target->create_standalone_buffer(
        7 * 4 * sizeof(std::uint16_t));
    auto* ragged_encoded =
        static_cast<std::uint16_t*>(ragged_logits.contents());
    const std::uint32_t ragged_tokens[] = {0, 1, 2, 3, 2, 1, 0};
    for (std::size_t row = 0; row < 7; ++row)
        for (std::size_t column = 0; column < 4; ++column)
            ragged_encoded[row * 4 + column] =
                bf16(column == ragged_tokens[row] ? 10.0f : 0.0f);
    ragged_target->make_resident();
    std::array<std::vector<std::shared_ptr<ChannelState>>, 2> ragged_channels{
        std::vector<std::shared_ptr<ChannelState>>{
            make_platform_channel_state(DType::I32, 4, 1)},
        std::vector<std::shared_ptr<ChannelState>>{
            make_platform_channel_state(DType::I32, 4, 1)},
    };
    auto prime_reused_ring = [](const std::shared_ptr<ChannelState>& channel) {
        Value discarded;
        return channel->push(Value::i32({91, 92, 93, 94})) &&
               channel->pop(discarded) &&
               channel->push(Value::i32({81, 82, 83, 84})) &&
               channel->pop(discarded) && channel->head() == 2 &&
               channel->tail() == 2;
    };
    const bool ragged_ring_primed =
        prime_reused_ring(ragged_channels[0][0]);
    std::array<std::shared_ptr<M1PreparedFire>, 2> ragged_fires;
    std::vector<M3LaneCandidate> ragged_candidates;
    for (std::size_t lane = 0; lane < 2; ++lane) {
        (void)runtime->prepare(
            ragged_executable,
            ragged_channels[lane],
            tickets_for(ragged_plan, ragged_channels[lane]),
            ragged_fires[lane],
            error);
        M1DeviceInputs inputs;
        inputs.logits_bf16 = ragged_logits;
        inputs.logits_row_offset = lane == 0 ? 0 : 3;
        inputs.logits_row_count = lane == 0 ? 3 : 4;
        inputs.logits_rows =
            lane == 0
                ? std::vector<std::uint32_t>{0, 1, 2}
                : std::vector<std::uint32_t>{6, 4, 3, 5};
        inputs.vocab = 4;
        inputs.extents.sampled_rows = inputs.logits_row_count;
        inputs.extents.row_count = inputs.logits_row_count;
        ragged_candidates.push_back({
            .fire = ragged_fires[lane],
            .inputs = inputs,
        });
    }
    std::shared_ptr<M3GroupCommand> ragged_group;
    const bool ragged_ready =
        ragged_executable &&
        runtime->prepare_m3_group(
            ragged_candidates,
            *ragged_target,
            ragged_group,
            error);
    const auto active_masks =
        runtime->m3_active_masks_for_test(ragged_group);
    if (ragged_ready) {
        ragged_target->run_step([&](StepEncoder& encoder) {
            runtime->encode_m3_pre(ragged_group, encoder);
            runtime->encode_m3_post(ragged_group, encoder);
        });
    }
    const auto ragged_outcomes =
        ragged_ready
            ? runtime->finish_m3_group(ragged_group, error)
            : std::vector<M1ExecuteOutcome>{};
    const Value ragged_first = ragged_channels[0][0]->front();
    const Value ragged_second = ragged_channels[1][0]->front();
    expect(
        ragged_outcomes.size() == 2 &&
            ragged_outcomes[0] == M1ExecuteOutcome::Committed &&
            ragged_outcomes[1] == M1ExecuteOutcome::Committed &&
            ragged_ring_primed &&
            active_masks == std::vector<std::uint64_t>({0x7, 0xf}) &&
            ragged_first.i ==
                std::vector<std::int32_t>({0, 1, 2, 0}) &&
            ragged_second.i ==
                std::vector<std::int32_t>({0, 2, 3, 1}),
        "ragged M3 ChanPut clears the stale fixed-cell tail after real ring "
        "reuse while preserving active masks and instance attribution");
    for (auto& fire : ragged_fires) runtime->release(fire);
    ragged_target->release_standalone_buffer(ragged_logits);

    std::vector<float> short_ragged_logits(3 * 4, 0.0f);
    for (std::size_t row = 0; row < 3; ++row) {
        short_ragged_logits[row * 4 + row] = 10.0f;
    }
    bool m1_tail_clear = true;
    for (const M1ExecutionMode mode : {
             M1ExecutionMode::Singleton,
             M1ExecutionMode::Fused,
         }) {
        auto channel =
            make_platform_channel_state(DType::I32, 4, 1);
        const bool primed = prime_reused_ring(channel);
        const M1ExecuteOutcome outcome =
            execute_generated(
                *runtime,
                ragged_executable,
                ragged_plan,
                {channel},
                short_ragged_logits,
                3,
                4,
                error,
                -1,
                mode);
        m1_tail_clear =
            m1_tail_clear && primed &&
            outcome == M1ExecuteOutcome::Committed &&
            channel->front().i ==
                std::vector<std::int32_t>({0, 1, 2, 0});
    }

    auto m2_channel =
        make_platform_channel_state(DType::I32, 4, 1);
    const bool m2_primed = prime_reused_ring(m2_channel);
    std::shared_ptr<M1PreparedFire> m2_ragged_fire;
    const bool m2_prepared =
        ragged_executable &&
        runtime->prepare(
            ragged_executable,
            {m2_channel},
            tickets_for(ragged_plan, {m2_channel}),
            m2_ragged_fire,
            error) == M1PrepareOutcome::Ready;
    auto m2_ragged_target = RawMetalContext::create(4u << 20);
    SlotHandle m2_ragged_logits =
        m2_ragged_target->create_standalone_buffer(
            short_ragged_logits.size() * sizeof(std::uint16_t));
    auto* m2_ragged_encoded =
        static_cast<std::uint16_t*>(m2_ragged_logits.contents());
    for (std::size_t index = 0;
         index < short_ragged_logits.size();
         ++index) {
        m2_ragged_encoded[index] = bf16(short_ragged_logits[index]);
    }
    m2_ragged_target->make_resident();
    M1DeviceInputs m2_ragged_inputs;
    m2_ragged_inputs.logits_bf16 = m2_ragged_logits;
    m2_ragged_inputs.logits_row_count = 3;
    m2_ragged_inputs.vocab = 4;
    m2_ragged_inputs.extents.sampled_rows = 3;
    m2_ragged_inputs.extents.row_count = 3;
    std::shared_ptr<M2CommandPlan> m2_ragged_command;
    const bool m2_command_ready =
        m2_prepared &&
        runtime->prepare_m2_command(
            m2_ragged_fire,
            m2_ragged_inputs,
            *m2_ragged_target,
            m2_ragged_command,
            error);
    StepTiming m2_ragged_timing;
    if (m2_command_ready) {
        m2_ragged_timing =
            m2_ragged_target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m2_pre(m2_ragged_command, encoder);
                runtime->encode_m2_post(m2_ragged_command, encoder);
            });
    }
    const M1ExecuteOutcome m2_ragged_outcome =
        m2_command_ready
            ? runtime->finish_m2_command(m2_ragged_command, error)
            : M1ExecuteOutcome::Failed;
    expect(
        m1_tail_clear && m2_primed && m2_ragged_timing.succeeded() &&
            m2_ragged_outcome == M1ExecuteOutcome::Committed &&
            m2_channel->front().i ==
                std::vector<std::int32_t>({0, 1, 2, 0}),
        "M1 singleton/fused and M2 ChanPut zero stale fixed-cell tails on GPU "
        "without clearing vocabulary scratch");
    runtime->release(m2_ragged_fire);
    m2_ragged_target->release_standalone_buffer(m2_ragged_logits);

    const auto topk_bytes = topk_container();
    const auto topk_plan_bytes = topk_sidecar(topk_bytes);
    ExecPlan topk_plan;
    error.clear();
    const bool topk_decoded = build_exec_plan(
        topk_bytes.data(), topk_bytes.size(),
        topk_plan_bytes.data(), topk_plan_bytes.size(),
        topk_plan, &error);
    if (topk_decoded) {
        auto& stage = topk_plan.region_plans[0];
        stage.fused.regions = {
            {.nodes = {0}},
            {
                .library = true,
                .library_op = PTIR_LIBRARY_TOP_K,
                .schedule = PTIR_SCHEDULE_LIBRARY,
                .nodes = {1},
            },
            {.nodes = {2, 3}},
        };
        stage.signature.push_back(0x51);
        stage.signature_hash =
            pie_native::ptir::container::fnv1a64(
                stage.signature.data(), stage.signature.size());
    }
    auto topk_executable =
        topk_decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      topk_bytes.data(), topk_bytes.size()),
                  topk_plan, error)
            : nullptr;
    auto topk_target = RawMetalContext::create(4u << 20);
    topk_target->make_resident();
    std::array<std::vector<std::shared_ptr<ChannelState>>, 2> topk_channels;
    std::array<std::shared_ptr<M1PreparedFire>, 2> topk_fires;
    std::vector<M3LaneCandidate> topk_candidates;
    const std::array<std::array<float,4>,2> topk_inputs{{
        {NAN,5,5,4}, {-INFINITY,NAN,-INFINITY,0}
    }};
    for (std::size_t lane=0; lane<2; ++lane) {
        topk_channels[lane] = {
            make_platform_channel_state(DType::F32,4,1),
            make_platform_channel_state(DType::F32,2,1),
            make_platform_channel_state(DType::U32,2,1)};
        (void)topk_channels[lane][0]->push(Value::f32(
            std::vector<float>(topk_inputs[lane].begin(),topk_inputs[lane].end())));
        (void)runtime->prepare(
            topk_executable, topk_channels[lane],
            tickets_for(topk_plan,topk_channels[lane]),topk_fires[lane],error);
        topk_candidates.push_back({.fire=topk_fires[lane],.inputs={}});
    }
    const M3GroupStats topk_before = runtime->m3_stats();
    std::shared_ptr<M3GroupCommand> topk_group;
    const bool topk_ready = topk_executable &&
        runtime->prepare_m3_group(topk_candidates,*topk_target,topk_group,error);
    if(topk_ready) topk_target->run_step([&](StepEncoder& encoder){
        runtime->encode_m3_pre(topk_group,encoder);
        runtime->encode_m3_post(topk_group,encoder);
    });
    const auto topk_outcomes = topk_ready
        ? runtime->finish_m3_group(topk_group,error)
        : std::vector<M1ExecuteOutcome>{};
    const M3GroupStats topk_after = runtime->m3_stats();
    expect(
        topk_outcomes.size()==2 &&
        topk_channels[0][2]->front().u==std::vector<std::uint32_t>({1,2}) &&
        topk_channels[1][2]->front().u==std::vector<std::uint32_t>({3,0}) &&
        topk_after.library_launches>topk_before.library_launches,
        "opcode-derived TopK library preserves NaN/tie semantics and lane attribution "
        "(ready=" + std::to_string(topk_ready) +
        ", outcomes=" + std::to_string(topk_outcomes.size()) +
        ", lane0=" +
        (topk_channels[0][2]->front().u.empty()
             ? "empty"
             : (std::to_string(topk_channels[0][2]->front().u[0]) + "," +
                std::to_string(topk_channels[0][2]->front().u[1]))) +
        ", lane1=" +
        (topk_channels[1][2]->front().u.empty()
             ? "empty"
             : (std::to_string(topk_channels[1][2]->front().u[0]) + "," +
                std::to_string(topk_channels[1][2]->front().u[1]))) +
        ", library_delta=" +
        std::to_string(
            topk_after.library_launches -
            topk_before.library_launches) +
        ", fallback_delta=" +
        std::to_string(
            topk_after.singleton_fallback_launches -
            topk_before.singleton_fallback_launches) +
        ", error=" + error + ")");
    for(auto& fire:topk_fires) runtime->release(fire);

    {
        constexpr std::uint32_t production_vocab = 248320;
        constexpr std::uint32_t production_k = 64;
        const auto bytes =
            topk_container(production_vocab, production_k);
        const auto sidecar =
            topk_sidecar(bytes, production_vocab, production_k);
        ExecPlan production_plan;
        error.clear();
        const bool decoded = build_exec_plan(
            bytes.data(),
            bytes.size(),
            sidecar.data(),
            sidecar.size(),
            production_plan,
            &error);
        if (decoded) {
            auto& stage = production_plan.region_plans[0];
            stage.fused.regions = {
                {.nodes = {0}},
                {
                    .library = true,
                    .library_op = PTIR_LIBRARY_TOP_K,
                    .schedule = PTIR_SCHEDULE_LIBRARY,
                    .nodes = {1},
                },
                {.nodes = {2, 3}},
            };
            stage.signature.push_back(0x53);
            stage.signature_hash =
                pie_native::ptir::container::fnv1a64(
                    stage.signature.data(), stage.signature.size());
        }
        auto program = decoded
            ? runtime->compile_program(
                  0x5300000000000001ULL,
                  production_plan,
                  error)
            : nullptr;
        std::vector<std::shared_ptr<ChannelState>> channels{
            make_platform_channel_state(
                DType::F32, production_vocab, 1),
            make_platform_channel_state(
                DType::F32, production_k, 1),
            make_platform_channel_state(
                DType::U32, production_k, 1),
        };
        std::vector<float> values(production_vocab, -1000.0f);
        values[0] = 10.0f;
        values[1] = 10.0f;
        values[2] = NAN;
        values[3] = -0.0f;
        values[4] = 0.0f;
        (void)channels[0]->push(Value::f32(std::move(values)));
        std::shared_ptr<M1PreparedFire> fire;
        const bool prepared =
            program &&
            runtime->prepare(
                program,
                channels,
                tickets_for(production_plan, channels),
                fire,
                error) == M1PrepareOutcome::Ready;
        auto target = RawMetalContext::create(4u << 20);
        target->make_resident();
        const M3GroupStats before = runtime->m3_stats();
        std::shared_ptr<M3GroupCommand> group;
        const bool ready =
            prepared &&
            runtime->prepare_m3_group(
                {{.fire = fire}}, *target, group, error);
        const auto begin = std::chrono::steady_clock::now();
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        const auto outcomes =
            ready
                ? runtime->finish_m3_group(group, error)
                : std::vector<M1ExecuteOutcome>{};
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - begin)
                .count();
        const M3GroupStats after = runtime->m3_stats();
        std::vector<std::uint32_t> expected{0, 1, 3, 4};
        for (std::uint32_t index = 5;
             expected.size() < production_k;
             ++index) {
            expected.push_back(index);
        }
        const Value output_values = channels[1]->front();
        expect(
            outcomes ==
                    std::vector<M1ExecuteOutcome>{
                        M1ExecuteOutcome::Committed} &&
                channels[2]->front().u == expected &&
                output_values.f.size() == production_k &&
                float_bits(output_values.f[2]) ==
                    float_bits(-0.0f) &&
                float_bits(output_values.f[3]) ==
                    float_bits(0.0f) &&
                after.parallel_selection_launches ==
                    before.parallel_selection_launches + 1 &&
                elapsed_ms < 500,
            "parallel exact TopK preserves NaN/tie/signed-zero ordering at "
            "V=248320,K=64 within 500 ms (" +
                std::to_string(elapsed_ms) + " ms, " + error + ")");
        runtime->release(fire);
    }

    const auto beam_bytes = generic_beam_container();
    const auto beam_plan_bytes = generic_beam_sidecar(beam_bytes);
    ExecPlan beam_plan;
    error.clear();
    const bool beam_decoded = build_exec_plan(
        beam_bytes.data(), beam_bytes.size(),
        beam_plan_bytes.data(), beam_plan_bytes.size(),
        beam_plan, &error);
    auto beam_executable = beam_decoded
        ? runtime->compile_program(
              pie_native::ptir::container::fnv1a64(
                  beam_bytes.data(), beam_bytes.size()),
              beam_plan,
              error)
        : nullptr;
    auto beam_target = RawMetalContext::create(4u << 20);
    beam_target->make_resident();
    std::array<std::shared_ptr<M1PreparedFire>,2> beam_fires;
    const std::array<std::array<float,6>,2> beam_inputs{{
        {5,5,NAN,4,6,6},{9,1,8,9,0,0}
    }};
    auto make_beam_channels = [&](std::size_t lane) {
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : beam_plan.trace.channels) {
            channels.push_back(make_platform_channel_state(
                channel.type.dtype,
                channel.type.shape.numel(),
                channel.capacity));
        }
        (void)channels[0]->push(Value::f32(
            std::vector<float>(
                beam_inputs[lane].begin(), beam_inputs[lane].end())));
        (void)channels[1]->push(Value::u32({10, 20, 30}));
        return channels;
    };
    auto beam_singleton_channels = make_beam_channels(0);
    const M1ExecuteOutcome beam_singleton =
        beam_executable
            ? execute_generated(
                  *runtime,
                  beam_executable,
                  beam_plan,
                  beam_singleton_channels,
                  {},
                  0,
                  0,
                  error,
                  -1,
                  M1ExecutionMode::Singleton)
            : M1ExecuteOutcome::Failed;
    auto beam_fused_channels = make_beam_channels(0);
    const M1ExecuteOutcome beam_fused =
        beam_executable
            ? execute_generated(
                  *runtime,
                  beam_executable,
                  beam_plan,
                  beam_fused_channels,
                  {},
                  0,
                  0,
                  error,
                  -1,
                  M1ExecutionMode::Fused)
            : M1ExecuteOutcome::Failed;
    std::array<std::vector<std::shared_ptr<ChannelState>>,2> beam_channels{
        make_beam_channels(0),
        make_beam_channels(1),
    };
    std::vector<M3LaneCandidate> beam_candidates;
    for(std::size_t lane=0;lane<2;++lane){
        (void)runtime->prepare(beam_executable,beam_channels[lane],tickets_for(beam_plan,beam_channels[lane]),beam_fires[lane],error);
        beam_candidates.push_back({.fire=beam_fires[lane],.inputs={}});
    }
    const M3GroupStats beam_before = runtime->m3_stats();
    std::shared_ptr<M3GroupCommand> beam_group;
    const bool beam_ready=beam_executable&&runtime->prepare_m3_group(beam_candidates,*beam_target,beam_group,error);
    if(beam_ready)beam_target->run_step([&](StepEncoder& encoder){
        runtime->encode_m3_pre(beam_group,encoder);runtime->encode_m3_post(beam_group,encoder);
    });
    const auto beam_outcomes=beam_ready?runtime->finish_m3_group(beam_group,error):std::vector<M1ExecuteOutcome>{};
    const M3GroupStats beam_after = runtime->m3_stats();
    expect(
        beam_singleton == M1ExecuteOutcome::Committed &&
        beam_fused == M1ExecuteOutcome::Committed &&
        beam_singleton_channels[2]->front().u ==
            std::vector<std::uint32_t>({20,30}) &&
        beam_singleton_channels[3]->front().u ==
            std::vector<std::uint32_t>({1,1}) &&
        beam_fused_channels[2]->front().u ==
            beam_singleton_channels[2]->front().u &&
        beam_fused_channels[3]->front().u ==
            beam_singleton_channels[3]->front().u &&
        beam_outcomes ==
            std::vector<M1ExecuteOutcome>({
                M1ExecuteOutcome::Committed,
                M1ExecuteOutcome::Committed}) &&
        beam_channels[0][2]->front().u ==
            std::vector<std::uint32_t>({20,30}) &&
        beam_channels[0][3]->front().u ==
            std::vector<std::uint32_t>({1,1}) &&
        beam_channels[1][2]->front().u ==
            std::vector<std::uint32_t>({10,10}) &&
        beam_channels[1][3]->front().u ==
            std::vector<std::uint32_t>({0,1}) &&
        beam_after.parallel_selection_launches ==
            beam_before.parallel_selection_launches + 1,
        "beam parity uses ordinary TopK plus generic DIV/REM/Gather in "
        "singleton, fused, and grouped execution");
    for(auto& fire:beam_fires)runtime->release(fire);

    {
        std::vector<std::uint8_t> shared_container;
        std::vector<std::uint8_t> shared_sidecar;
        ExecPlan shared_plan;
        error.clear();
        const bool decoded =
            load_golden(
                "beam_epilogue",
                shared_container,
                shared_sidecar) &&
            build_exec_plan(
                shared_container.data(),
                shared_container.size(),
                shared_sidecar.data(),
                shared_sidecar.size(),
                shared_plan,
                &error);
        auto program = decoded
            ? runtime->compile_program(
                  pie_native::ptir::container::fnv1a64(
                      shared_container.data(),
                      shared_container.size()),
                  shared_plan,
                  error)
            : nullptr;
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : shared_plan.trace.channels) {
            channels.push_back(
                make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
        }
        if (channels.size() == 16) {
            (void)channels[0]->push(
                Value::u32({5, 6, 0, 5, 6, 0}));
            (void)channels[1]->push(
                Value::u32({4, 2, 0, 4, 2, 0}));
            (void)channels[2]->push(Value::u32({6, 6}));
            (void)channels[3]->push(Value::boolean({
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            }));
            (void)channels[4]->push(Value::u32({6, 6}));
            (void)channels[5]->push(Value::u32({2, 2}));
            (void)channels[6]->push(Value::u32({6, 6}));
            (void)channels[7]->push(Value::u32({2, 2}));
            (void)channels[8]->push(Value::u32({6, 6}));
            (void)channels[9]->push(Value::u32({2, 2}));
            (void)channels[10]->push(Value::i32({1, 2}));
            (void)channels[11]->push(Value::f32({0.0f, 0.0f}));
            (void)channels[12]->push(Value::u32({7, 8}));
        }
        InterpInstance geometry_instance;
        geometry_instance.channels = channels;
        pie_native::ptir::FireGeometry beam_geometry;
        std::string geometry_error;
        const auto geometry_resolution =
            resolve_fire_geometry_typed(
                shared_plan,
                geometry_instance,
                4,
                beam_geometry,
                &geometry_error);
        MemberForwardDesc beam_desc;
        const std::uint32_t beam_folded_slots[] = {0, 1};
        const std::uint8_t beam_reset[] = {
            PIE_RS_FLAG_RESET,
            PIE_RS_FLAG_RESET,
        };
        const std::uint32_t beam_buffered_activation_slots[] = {3};
        const std::uint32_t beam_request_slot_indptr[] = {0, 1};
        pie_native::LaunchView beam_launch;
        beam_launch.rs_slot_ids =
            pie_native::slice_from_u32(beam_folded_slots, 2);
        beam_launch.rs_slot_flags =
            pie_native::slice_from_u8(beam_reset, 2);
        beam_launch.rs_buffer_slot_ids =
            pie_native::slice_from_u32(
                beam_buffered_activation_slots, 1);
        beam_launch.rs_buffer_slot_indptr =
            pie_native::slice_from_u32(
                beam_request_slot_indptr, 2);
        const bool canonical_beam_geometry =
            geometry_resolution.status ==
                GeometryResolveStatus::Ready &&
            build_member_forward_desc(
                beam_launch,
                0,
                1,
                true,
                4,
                &beam_geometry,
                beam_desc,
                geometry_error) &&
            beam_desc.qo_indptr ==
                std::vector<std::uint32_t>({0, 1, 2}) &&
            beam_desc.kv_page_indptr ==
                std::vector<std::uint32_t>({0, 3, 6}) &&
            beam_desc.kv_last_page_lens ==
                std::vector<std::uint32_t>({2, 2}) &&
            beam_desc.sampling_indptr ==
                std::vector<std::uint32_t>({0, 1, 2}) &&
            beam_desc.request_rs_slot_ids ==
                std::vector<std::uint32_t>({0, 1}) &&
            beam_desc.request_rs_reset ==
                std::vector<std::uint8_t>({1, 1}) &&
            beam_desc.request_rs_read ==
                std::vector<std::uint8_t>({0, 0}) &&
            beam_desc.request_rs_write ==
                std::vector<std::uint8_t>({1, 1}) &&
            validate_request_local_positions(
                beam_desc, &geometry_error);
        std::shared_ptr<M1PreparedFire> fire;
        const bool prepared =
            program &&
            runtime->prepare(
                program,
                channels,
                tickets_for(shared_plan, channels),
                fire,
                error) == M1PrepareOutcome::Ready;
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits =
            target->create_standalone_buffer(
                16 * sizeof(std::uint16_t));
        const std::array<float, 16> values{
            0, 0, 0, 8, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 7, 0, 0,
        };
        auto* encoded =
            static_cast<std::uint16_t*>(logits.contents());
        for (std::size_t index = 0; index < values.size(); ++index) {
            encoded[index] = bf16(values[index]);
        }
        target->make_resident();
        M1DeviceInputs inputs;
        inputs.logits_bf16 = logits;
        inputs.logits_row_count = 2;
        inputs.logits_rows = {0, 1};
        inputs.vocab = 8;
        inputs.extents.sampled_rows = 2;
        inputs.extents.row_count = 2;
        const M3GroupStats before = runtime->m3_stats();
        std::shared_ptr<M3GroupCommand> group;
        const bool ready =
            prepared &&
            runtime->prepare_m3_group(
                {{.fire = fire, .inputs = inputs}},
                *target,
                group,
                error);
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        const auto outcomes =
            ready
                ? runtime->finish_m3_group(group, error)
                : std::vector<M1ExecuteOutcome>{};
        const M3GroupStats after = runtime->m3_stats();
        expect(
            canonical_beam_geometry &&
                outcomes ==
                    std::vector<M1ExecuteOutcome>{
                        M1ExecuteOutcome::Committed} &&
                channels[13]->front().i ==
                    std::vector<std::int32_t>({3, 5}) &&
                channels[14]->front().u ==
                    std::vector<std::uint32_t>({0, 1}) &&
                std::abs(channels[15]->front().f[0] -
                         -0.0023454318f) <
                    1e-6f &&
                std::abs(channels[15]->front().f[1] -
                         -0.0063628945f) <
                    1e-6f &&
                after.parallel_selection_launches >
                    before.parallel_selection_launches,
            "opcode-derived parallel TopK and generic beam SSA match "
            "authoritative multi-request geometry and outputs exactly (" +
                error + geometry_error + ")");
        runtime->release(fire);
        target->release_standalone_buffer(logits);
    }

    const auto nucleus_bytes = ssa_nucleus_container();
    const auto nucleus_plan_bytes = ssa_nucleus_sidecar(nucleus_bytes);
    ExecPlan nucleus_plan;
    error.clear();
    const bool nucleus_decoded = build_exec_plan(
        nucleus_bytes.data(), nucleus_bytes.size(),
        nucleus_plan_bytes.data(), nucleus_plan_bytes.size(),
        nucleus_plan, &error);
    const bool ssa_nucleus_region =
        nucleus_decoded &&
        nucleus_plan.region_plans[0].fused.regions.size() == 3 &&
        nucleus_plan.region_plans[0].fused.regions[1].library &&
        nucleus_plan.region_plans[0].fused.regions[1].library_op ==
            PTIR_LIBRARY_NUCLEUS_SAMPLE &&
        nucleus_plan.region_plans[0].fused.regions[1].nodes ==
            std::vector<std::uint32_t>({
                3,4,5,6,7,8,9,10,11,12,13,14,15}) &&
        nucleus_plan.region_plans[0].fused.regions[1].inputs ==
            std::vector<std::uint32_t>({0,1,2}) &&
        nucleus_plan.region_plans[0].fused.regions[1].outputs ==
            std::vector<std::uint32_t>({15});
    std::string nucleus_parallel_source;
    std::string nucleus_emit_error;
    auto source_count = [](const std::string& source, std::string_view needle) {
        std::size_t count = 0;
        for (std::size_t position = 0;
             (position = source.find(needle, position)) != std::string::npos;
             position += needle.size()) {
            ++count;
        }
        return count;
    };
    const bool nucleus_ping_pong_reduction =
        ssa_nucleus_region &&
        emit_grouped_nucleus_msl(
            "",
            "nucleus_ping_pong_regression",
            nucleus_plan.region_plans[0],
            nucleus_plan.region_plans[0].fused.regions[1],
            nucleus_parallel_source,
            nucleus_emit_error) &&
        nucleus_parallel_source.find("reduction_input") !=
            std::string::npos &&
        nucleus_parallel_source.find("reduction_output") !=
            std::string::npos &&
        nucleus_parallel_source.find(
            "device float* reduction_a = probabilities + len;") !=
            std::string::npos &&
        nucleus_parallel_source.find("device float* reduction_b =") !=
            std::string::npos &&
        nucleus_parallel_source.find(
            "reinterpret_cast<device float*>(order_a);") !=
            std::string::npos &&
        source_count(
            nucleus_parallel_source,
            "device float* swap = reduction_input;") == 2 &&
        source_count(
            nucleus_parallel_source,
            "reduction_output[chunk] = values[0];") == 2 &&
        nucleus_parallel_source.find("reduction[chunk]") ==
            std::string::npos;
    auto nucleus_executable = nucleus_decoded
        ? runtime->compile_program(
              pie_native::ptir::container::fnv1a64(
                  nucleus_bytes.data(), nucleus_bytes.size()),
              nucleus_plan,
              error)
        : nullptr;
    auto make_nucleus_channels = [&](std::size_t lane) {
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : nucleus_plan.trace.channels) {
            channels.push_back(make_platform_channel_state(
                channel.type.dtype,
                channel.type.shape.numel(),
                channel.capacity));
        }
        (void)channels[0]->push(Value::f32(
            lane == 0
                ? std::vector<float>({0.5f,0.3f,0.2f,0.0f})
                : std::vector<float>({0.4f,0.35f,0.25f,NAN})));
        (void)channels[1]->push(
            Value::f32({lane == 0 ? 0.5f : 0.4f}));
        (void)channels[2]->push(Value::u32(
            {1234u + static_cast<std::uint32_t>(lane), 0}));
        return channels;
    };
    auto nucleus_singleton_channels = make_nucleus_channels(0);
    const M1ExecuteOutcome nucleus_singleton =
        nucleus_executable
            ? execute_generated(
                  *runtime,
                  nucleus_executable,
                  nucleus_plan,
                  nucleus_singleton_channels,
                  {},
                  0,
                  0,
                  error,
                  -1,
                  M1ExecutionMode::Singleton)
            : M1ExecuteOutcome::Failed;
    auto nucleus_fused_channels = make_nucleus_channels(0);
    const M1ExecuteOutcome nucleus_fused =
        nucleus_executable
            ? execute_generated(
                  *runtime,
                  nucleus_executable,
                  nucleus_plan,
                  nucleus_fused_channels,
                  {},
                  0,
                  0,
                  error,
                  -1,
                  M1ExecutionMode::Fused)
            : M1ExecuteOutcome::Failed;
    auto nucleus_target=RawMetalContext::create(4u<<20);
    nucleus_target->make_resident();
    std::array<std::vector<std::shared_ptr<ChannelState>>,2> nucleus_channels{
        make_nucleus_channels(0),
        make_nucleus_channels(1),
    };
    std::array<std::shared_ptr<M1PreparedFire>,2> nucleus_fires;
    std::vector<M3LaneCandidate> nucleus_candidates;
    for(std::size_t lane=0;lane<2;++lane){
        (void)runtime->prepare(nucleus_executable,nucleus_channels[lane],tickets_for(nucleus_plan,nucleus_channels[lane]),nucleus_fires[lane],error);
        nucleus_candidates.push_back({.fire=nucleus_fires[lane],.inputs={}});
    }
    const M3GroupStats nucleus_before=runtime->m3_stats();
    std::shared_ptr<M3GroupCommand> nucleus_group;
    const bool nucleus_ready=nucleus_executable&&runtime->prepare_m3_group(nucleus_candidates,*nucleus_target,nucleus_group,error);
    if(nucleus_ready)nucleus_target->run_step([&](StepEncoder& encoder){
        runtime->encode_m3_pre(nucleus_group,encoder);runtime->encode_m3_post(nucleus_group,encoder);
    });
    const auto nucleus_outcomes=nucleus_ready?runtime->finish_m3_group(nucleus_group,error):std::vector<M1ExecuteOutcome>{};
    const M3GroupStats nucleus_after=runtime->m3_stats();
    expect(
        ssa_nucleus_region &&
        nucleus_ping_pong_reduction &&
        nucleus_singleton == M1ExecuteOutcome::Committed &&
        nucleus_fused == M1ExecuteOutcome::Committed &&
        nucleus_singleton_channels[3]->front().i[0] == 0 &&
        nucleus_fused_channels[3]->front().i[0] == 0 &&
        nucleus_outcomes ==
            std::vector<M1ExecuteOutcome>({
                M1ExecuteOutcome::Committed,
                M1ExecuteOutcome::Committed}) &&
        nucleus_channels[0][3]->front().i[0]==0 &&
        nucleus_channels[1][3]->front().i[0]==0 &&
        nucleus_after.library_launches ==
            nucleus_before.library_launches + 1 &&
        nucleus_after.parallel_selection_launches ==
            nucleus_before.parallel_selection_launches + 1,
        "PTRP-authored nucleus library uses race-free canonical ping-pong "
        "reductions while materialized SSA logits execute through singleton, "
        "fused, grouped, and parallel paths");
    for(auto& fire:nucleus_fires)runtime->release(fire);

    std::vector<std::uint8_t> nucleus_golden_container;
    std::vector<std::uint8_t> nucleus_golden_sidecar;
    ExecPlan nucleus_golden_plan;
    error.clear();
    const bool nucleus_golden_decoded =
        load_golden(
            "nucleus_sample",
            nucleus_golden_container,
            nucleus_golden_sidecar) &&
        build_exec_plan(
            nucleus_golden_container.data(),
            nucleus_golden_container.size(),
            nucleus_golden_sidecar.data(),
            nucleus_golden_sidecar.size(),
            nucleus_golden_plan,
            &error);
    auto nucleus_golden_executable = nucleus_golden_decoded
        ? runtime->compile_program(
              pie_native::ptir::container::fnv1a64(
                  nucleus_golden_container.data(),
                  nucleus_golden_container.size()),
              nucleus_golden_plan,
              error)
        : nullptr;
    const std::array<float, 8> nucleus_golden_values{
        4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f, NAN};
    auto make_nucleus_golden_channels = [&](std::size_t lane) {
        std::vector<std::shared_ptr<ChannelState>> channels;
        for (const auto& channel : nucleus_golden_plan.trace.channels) {
            channels.push_back(
                make_platform_channel_state(
                    channel.type.dtype,
                    channel.type.shape.numel(),
                    channel.capacity));
        }
        (void)channels[0]->push(
            Value::u32({1234u, static_cast<std::uint32_t>(lane)}));
        (void)channels[1]->push(
            Value::f32({lane == 0 ? 0.5f : 1.0f}));
        return channels;
    };
    const std::vector<float> raw_nucleus_logits(
        nucleus_golden_values.begin(), nucleus_golden_values.end());
    auto nucleus_raw_singleton_channels =
        make_nucleus_golden_channels(0);
    const M1ExecuteOutcome nucleus_raw_singleton =
        nucleus_golden_executable
            ? execute_generated(
                  *runtime,
                  nucleus_golden_executable,
                  nucleus_golden_plan,
                  nucleus_raw_singleton_channels,
                  raw_nucleus_logits,
                  1,
                  8,
                  error,
                  -1,
                  M1ExecutionMode::Singleton)
            : M1ExecuteOutcome::Failed;
    auto nucleus_raw_fused_channels =
        make_nucleus_golden_channels(0);
    const M1ExecuteOutcome nucleus_raw_fused =
        nucleus_golden_executable
            ? execute_generated(
                  *runtime,
                  nucleus_golden_executable,
                  nucleus_golden_plan,
                  nucleus_raw_fused_channels,
                  raw_nucleus_logits,
                  1,
                  8,
                  error,
                  -1,
                  M1ExecutionMode::Fused)
            : M1ExecuteOutcome::Failed;
    auto nucleus_golden_target = RawMetalContext::create(4u << 20);
    SlotHandle nucleus_golden_logits =
        nucleus_golden_target->create_standalone_buffer(16 * sizeof(std::uint16_t));
    auto* nucleus_golden_encoded =
        static_cast<std::uint16_t*>(nucleus_golden_logits.contents());
    for (std::size_t lane = 0; lane < 2; ++lane) {
        for (std::size_t column = 0; column < nucleus_golden_values.size(); ++column) {
            nucleus_golden_encoded[lane * 8 + column] =
                bf16(nucleus_golden_values[column]);
        }
    }
    nucleus_golden_target->make_resident();
    auto nucleus_raw_m2_channels =
        make_nucleus_golden_channels(0);
    std::shared_ptr<M1PreparedFire> nucleus_raw_m2_fire;
    const bool nucleus_raw_m2_prepared =
        nucleus_golden_executable &&
        runtime->prepare(
            nucleus_golden_executable,
            nucleus_raw_m2_channels,
            tickets_for(
                nucleus_golden_plan,
                nucleus_raw_m2_channels),
            nucleus_raw_m2_fire,
            error) == M1PrepareOutcome::Ready;
    M1DeviceInputs nucleus_raw_m2_inputs;
    nucleus_raw_m2_inputs.logits_bf16 = nucleus_golden_logits;
    nucleus_raw_m2_inputs.logits_row_count = 1;
    nucleus_raw_m2_inputs.vocab = 8;
    nucleus_raw_m2_inputs.extents.sampled_rows = 1;
    nucleus_raw_m2_inputs.extents.row_count = 1;
    std::shared_ptr<M2CommandPlan> nucleus_raw_m2_command;
    const bool nucleus_raw_m2_ready =
        nucleus_raw_m2_prepared &&
        runtime->prepare_m2_command(
            nucleus_raw_m2_fire,
            nucleus_raw_m2_inputs,
            *nucleus_golden_target,
            nucleus_raw_m2_command,
            error);
    StepTiming nucleus_raw_m2_timing;
    if (nucleus_raw_m2_ready) {
        nucleus_raw_m2_timing =
            nucleus_golden_target->run_step(
                [&](StepEncoder& encoder) {
                    runtime->encode_m2_pre(
                        nucleus_raw_m2_command, encoder);
                    runtime->encode_m2_post(
                        nucleus_raw_m2_command, encoder);
                });
    }
    const M1ExecuteOutcome nucleus_raw_m2 =
        nucleus_raw_m2_ready
            ? runtime->finish_m2_command(
                  nucleus_raw_m2_command, error)
            : M1ExecuteOutcome::Failed;
    std::array<std::vector<std::shared_ptr<ChannelState>>, 2>
        nucleus_golden_channels;
    std::array<std::shared_ptr<M1PreparedFire>, 2> nucleus_golden_fires;
    std::vector<M3LaneCandidate> nucleus_golden_candidates;
    for (std::size_t lane = 0; lane < 2; ++lane) {
        nucleus_golden_channels[lane] =
            make_nucleus_golden_channels(lane);
        (void)runtime->prepare(
            nucleus_golden_executable,
            nucleus_golden_channels[lane],
            tickets_for(nucleus_golden_plan, nucleus_golden_channels[lane]),
            nucleus_golden_fires[lane],
            error);
        M1DeviceInputs inputs;
        inputs.logits_bf16 = nucleus_golden_logits;
        inputs.logits_row_offset = static_cast<std::uint32_t>(lane);
        inputs.logits_row_count = 1;
        inputs.vocab = 8;
        inputs.extents.sampled_rows = 1;
        inputs.extents.row_count = 1;
        nucleus_golden_candidates.push_back({
            .fire = nucleus_golden_fires[lane],
            .inputs = inputs,
        });
    }
    const M3GroupStats nucleus_golden_before = runtime->m3_stats();
    std::shared_ptr<M3GroupCommand> nucleus_golden_group;
    const bool nucleus_golden_ready =
        nucleus_golden_executable &&
        runtime->prepare_m3_group(
            nucleus_golden_candidates,
            *nucleus_golden_target,
            nucleus_golden_group,
            error);
    if (nucleus_golden_ready) {
        nucleus_golden_target->run_step([&](StepEncoder& encoder) {
            runtime->encode_m3_pre(nucleus_golden_group, encoder);
            runtime->encode_m3_post(nucleus_golden_group, encoder);
        });
    }
    const auto nucleus_golden_outcomes = nucleus_golden_ready
        ? runtime->finish_m3_group(nucleus_golden_group, error)
        : std::vector<M1ExecuteOutcome>{};
    const M3GroupStats nucleus_golden_after = runtime->m3_stats();
    expect(
        nucleus_raw_singleton == M1ExecuteOutcome::Committed &&
            nucleus_raw_fused == M1ExecuteOutcome::Committed &&
            nucleus_raw_m2 == M1ExecuteOutcome::Committed &&
            nucleus_raw_m2_timing.succeeded() &&
            nucleus_raw_singleton_channels[2]->front().i[0] == 0 &&
            nucleus_raw_fused_channels[2]->front().i[0] == 0 &&
            nucleus_raw_m2_channels[2]->front().i[0] == 0 &&
            nucleus_golden_outcomes.size() == 2 &&
            nucleus_golden_channels[0][2]->front().i[0] == 0 &&
            nucleus_golden_channels[1][2]->front().i[0] == 0 &&
            nucleus_golden_after.library_launches >
                nucleus_golden_before.library_launches,
        "shared nucleus golden materializes raw BF16 logits correctly through "
        "generic singleton/fused/M2 fallback and the M3 stock library "
        "(ready=" + std::to_string(nucleus_golden_ready) +
        " outcomes=" + std::to_string(nucleus_golden_outcomes.size()) +
        " tokens=" +
        std::to_string(nucleus_golden_channels[0][2]->front().i[0]) + "," +
        std::to_string(nucleus_golden_channels[1][2]->front().i[0]) +
        " error=" + error + ")");
    for (auto& fire : nucleus_golden_fires) runtime->release(fire);
    runtime->release(nucleus_raw_m2_fire);
    nucleus_golden_target->release_standalone_buffer(nucleus_golden_logits);

    {
        constexpr std::uint32_t production_vocab = 248320;
        ExecPlan production_plan = nucleus_golden_plan;
        for (auto& stage : production_plan.region_plans) {
            for (auto& type : stage.value_types) {
                for (auto& dimension : type.dims) {
                    if (!dimension.symbolic && dimension.value == 8) {
                        dimension.value = production_vocab;
                    }
                }
            }
        }
        auto production_program = runtime->compile_program(
            0xf43945e596bc81f3ULL,
            production_plan,
            error);
        auto target = RawMetalContext::create(4u << 20);
        SlotHandle logits = target->create_standalone_buffer(
            2ull * production_vocab * sizeof(std::uint16_t));
        auto* encoded =
            static_cast<std::uint16_t*>(logits.contents());
        const std::uint16_t negative_infinity = bf16(-INFINITY);
        std::fill(
            encoded,
            encoded + 2ull * production_vocab,
            negative_infinity);
        for (std::size_t lane = 0; lane < 2; ++lane) {
            for (std::size_t column = 0;
                 column < nucleus_golden_values.size();
                 ++column) {
                encoded[lane * production_vocab + column] =
                    bf16(nucleus_golden_values[column]);
            }
        }
        target->make_resident();
        std::array<std::vector<std::shared_ptr<ChannelState>>, 2>
            channels;
        std::array<std::shared_ptr<M1PreparedFire>, 2> fires;
        std::vector<M3LaneCandidate> candidates;
        bool prepared = production_program != nullptr;
        for (std::size_t lane = 0; lane < 2; ++lane) {
            for (const auto& channel :
                 production_plan.trace.channels) {
                channels[lane].push_back(
                    make_platform_channel_state(
                        channel.type.dtype,
                        channel.type.shape.numel(),
                        channel.capacity));
            }
            (void)channels[lane][0]->push(
                Value::u32(
                    {1234u, static_cast<std::uint32_t>(lane)}));
            (void)channels[lane][1]->push(
                Value::f32({lane == 0 ? 0.5f : 1.0f}));
            prepared =
                prepared &&
                runtime->prepare(
                    production_program,
                    channels[lane],
                    tickets_for(production_plan, channels[lane]),
                    fires[lane],
                    error) == M1PrepareOutcome::Ready;
            M1DeviceInputs inputs;
            inputs.logits_bf16 = logits;
            inputs.logits_row_offset =
                static_cast<std::uint32_t>(lane);
            inputs.logits_row_count = 1;
            inputs.vocab = production_vocab;
            inputs.extents.sampled_rows = 1;
            inputs.extents.row_count = 1;
            candidates.push_back({
                .fire = fires[lane],
                .inputs = inputs,
            });
        }
        const M3GroupStats before = runtime->m3_stats();
        std::shared_ptr<M3GroupCommand> group;
        const bool ready =
            prepared &&
            runtime->prepare_m3_group(
                candidates, *target, group, error);
        const auto begin = std::chrono::steady_clock::now();
        if (ready) {
            target->run_step([&](StepEncoder& encoder) {
                runtime->encode_m3_pre(group, encoder);
                runtime->encode_m3_post(group, encoder);
            });
        }
        const auto outcomes =
            ready
                ? runtime->finish_m3_group(group, error)
                : std::vector<M1ExecuteOutcome>{};
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - begin)
                .count();
        const M3GroupStats after = runtime->m3_stats();
        expect(
            outcomes ==
                    std::vector<M1ExecuteOutcome>({
                        M1ExecuteOutcome::Committed,
                        M1ExecuteOutcome::Committed}) &&
                channels[0][2]->front().i[0] == 0 &&
                channels[1][2]->front().i[0] == 0 &&
                after.parallel_selection_launches ==
                    before.parallel_selection_launches + 1 &&
                elapsed_ms < 500,
            "parallel exact nucleus library preserves both shared-golden "
            "tokens at vocab=248320 with bounded latency (" +
                std::to_string(elapsed_ms) + " ms, " + error + ")");
        for (auto& fire : fires) runtime->release(fire);
        target->release_standalone_buffer(logits);
    }

    auto memory_hit =
        runtime->compile_program(0xff694395428759feULL, plan, error);
    expect(
        memory_hit == executable &&
            runtime->cache_stats().memory_hits > 0,
        "bounded in-memory cache reuses the executable");

    ExecPlan unsupported = plan;
    unsupported.region_plans[0].singleton.whole_stage_fallback = true;
    error.clear();
    M1CompileFailureKind deterministic_failure =
        M1CompileFailureKind::None;
    expect(
        runtime->compile_program(
            0x1111222233334444ULL,
            unsupported,
            error,
            {},
            &deterministic_failure) == nullptr &&
            deterministic_failure ==
                M1CompileFailureKind::Deterministic &&
            error.find("fallback") != std::string::npos,
        "deterministic unsupported plan rejects at registration");
    error.clear();
    (void)runtime->compile_program(
        0x1111222233334444ULL, unsupported, error);
    expect(
        runtime->cache_stats().negative_hits > 0,
        "negative cache avoids recompiling deterministic failures");

    {
        std::string retry_error;
        auto retry_runtime = M1Runtime::create(
            kernels_dir,
            (cache / "compiler-retry").string(),
            retry_error);
        retry_runtime->inject_compile_failure_for_test(
            "ptir_m1_",
            "fault injection: transient Metal compiler unavailable");
        M1CompileFailureKind first_kind = M1CompileFailureKind::None;
        const auto first = retry_runtime->compile_program(
            0xc011a11e00000001ULL,
            plan,
            retry_error,
            container,
            &first_kind);
        const M1CacheStats after_first = retry_runtime->cache_stats();
        retry_error.clear();
        M1CompileFailureKind second_kind = M1CompileFailureKind::Retryable;
        const auto second = retry_runtime->compile_program(
            0xc011a11e00000001ULL,
            plan,
            retry_error,
            container,
            &second_kind);
        expect(
            first == nullptr &&
                first_kind == M1CompileFailureKind::Retryable &&
                after_first.negative_entries == 0 &&
                after_first.program_entries == 0 && second != nullptr &&
                second_kind == M1CompileFailureKind::None,
            "fail-once Metal compiler error retries without negative-cache "
            "poisoning (" + retry_error + ")");
    }

    {
        std::string retry_error;
        auto retry_runtime = M1Runtime::create(
            kernels_dir,
            (cache / "late-compile-rollback").string(),
            retry_error);
        constexpr std::uint32_t retry_count = 6;
        const std::size_t retained_before =
            retry_runtime->context().retained_pso_count();
        retry_runtime->inject_compile_failure_for_test(
            "_commit",
            "fault injection: fail after stage and readiness PSOs",
            retry_count);
        bool bounded = true;
        for (std::uint32_t attempt = 0; attempt < retry_count; ++attempt) {
            retry_error.clear();
            M1CompileFailureKind kind = M1CompileFailureKind::None;
            const auto failed = retry_runtime->compile_program(
                0xfa11a7e000000001ULL,
                plan,
                retry_error,
                container,
                &kind);
            const M1CacheStats stats = retry_runtime->cache_stats();
            bounded =
                bounded && failed == nullptr &&
                kind == M1CompileFailureKind::Retryable &&
                stats.stage_entries == 0 && stats.program_entries == 0 &&
                stats.negative_entries == 0 &&
                retry_runtime->context().retained_pso_count() ==
                    retained_before;
        }
        retry_error.clear();
        const auto recovered = retry_runtime->compile_program(
            0xfa11a7e000000001ULL,
            plan,
            retry_error,
            container);
        const std::size_t retained_after =
            retry_runtime->context().retained_pso_count();
        const auto memory_hit = retry_runtime->compile_program(
            0xfa11a7e000000001ULL,
            plan,
            retry_error,
            container);
        expect(
            bounded && recovered != nullptr && memory_hit == recovered &&
                retained_after > retained_before &&
                retry_runtime->context().retained_pso_count() ==
                    retained_after,
            "repeated fail-late compile retries roll back every uncommitted "
            "PSO and recover with bounded retained state (" +
                retry_error + ")");
    }

    {
        std::string rollback_error;
        auto rollback_runtime = M1Runtime::create(
            kernels_dir,
            (cache / "late-deterministic-rollback").string(),
            rollback_error);
        ExecPlan late_deterministic = ragged_plan;
        auto& stage = late_deterministic.region_plans[0];
        stage.stage = PTIR_STAGE_PROLOGUE;
        stage.channel_bindings.resize(kMetalM2MaxFusedChannels + 1, 0);
        stage.signature.push_back(0xd7);
        stage.signature_hash =
            pie_native::ptir::container::fnv1a64(
                stage.signature.data(), stage.signature.size());
        const std::size_t retained_before =
            rollback_runtime->context().retained_pso_count();
        M1CompileFailureKind kind = M1CompileFailureKind::None;
        const auto rejected = rollback_runtime->compile_program(
            0xde7e000000000001ULL,
            late_deterministic,
            rollback_error,
            ragged_bytes,
            &kind);
        const M1CacheStats stats = rollback_runtime->cache_stats();
        expect(
            rejected == nullptr &&
                kind == M1CompileFailureKind::Deterministic &&
                rollback_error.find("no fused executable") !=
                    std::string::npos &&
                stats.stage_entries == 0 && stats.program_entries == 0 &&
                stats.negative_entries == 1 && stats.compilations > 0 &&
                rollback_runtime->context().retained_pso_count() ==
                    retained_before,
            "late deterministic rejection rolls back compiled stage PSOs "
            "before entering the negative cache (kind=" +
                std::to_string(static_cast<int>(kind)) +
                " stages=" + std::to_string(stats.stage_entries) +
                " programs=" + std::to_string(stats.program_entries) +
                " negatives=" + std::to_string(stats.negative_entries) +
                " retained=" +
                std::to_string(
                    rollback_runtime->context().retained_pso_count()) +
                "/" + std::to_string(retained_before) +
                " error=" + rollback_error + ")");
    }

    {
        const std::filesystem::path blocked_cache =
            cache / "io-retry-blocked";
        std::filesystem::remove_all(blocked_cache, ec);
        {
            std::ofstream blocked(blocked_cache);
            blocked << "not a directory";
        }
        std::string retry_error;
        auto retry_runtime = M1Runtime::create(
            kernels_dir, blocked_cache.string(), retry_error);
        M1CompileFailureKind first_kind = M1CompileFailureKind::None;
        const auto first = retry_runtime->compile_program(
            0x10fau,
            plan,
            retry_error,
            container,
            &first_kind);
        const bool io_reported =
            retry_error.find("cache directory") != std::string::npos;
        std::filesystem::remove(blocked_cache, ec);
        retry_error.clear();
        M1CompileFailureKind second_kind = M1CompileFailureKind::Retryable;
        const auto second = retry_runtime->compile_program(
            0x10fau,
            plan,
            retry_error,
            container,
            &second_kind);
        expect(
            first == nullptr && io_reported &&
                first_kind == M1CompileFailureKind::Retryable &&
                second != nullptr &&
                second_kind == M1CompileFailureKind::None &&
                retry_runtime->cache_stats().negative_entries == 0,
            "cache-directory IO failure remains retryable after storage "
            "recovers (" + retry_error + ")");
    }

    {
        bool recovered_all = true;
        std::string last_error;
        const std::array<std::pair<const char*, std::uint64_t>, 2> faults{{
            {"ptir_m3_generic_ready_v", 0xeffec70000000001ULL},
            {"ptir_m3_generic_commit_v", 0xeffec70000000002ULL},
        }};
        for (const auto& [function, hash] : faults) {
            std::string retry_error;
            auto retry_runtime = M1Runtime::create(
                kernels_dir,
                (cache / ("grouped-effect-retry-" +
                          std::to_string(hash)))
                    .string(),
                retry_error);
            retry_runtime->inject_compile_failure_for_test(
                function,
                "fault injection: ENOSPC flushing grouped effect archive");
            M1CompileFailureKind first_kind =
                M1CompileFailureKind::None;
            const auto first = retry_runtime->compile_program(
                hash,
                plan,
                retry_error,
                container,
                &first_kind);
            const M1CacheStats after_first =
                retry_runtime->cache_stats();
            const bool effect_reported =
                retry_error.find("grouped") != std::string::npos;
            retry_error.clear();
            M1CompileFailureKind second_kind =
                M1CompileFailureKind::Retryable;
            const auto recovered = retry_runtime->compile_program(
                hash,
                plan,
                retry_error,
                container,
                &second_kind);
            recovered_all =
                recovered_all && first == nullptr && effect_reported &&
                first_kind == M1CompileFailureKind::Retryable &&
                after_first.program_entries == 0 &&
                after_first.negative_entries == 0 &&
                recovered != nullptr &&
                second_kind == M1CompileFailureKind::None &&
                retry_runtime->cache_stats().program_entries == 1;
            last_error = retry_error;
        }
        expect(
            recovered_all,
            "grouped readiness/commit PSO fail-once recovery never caches a "
            "degraded positive executable (" + last_error + ")");
    }

    {
        std::string retry_error;
        auto bounded_runtime = M1Runtime::create(
            kernels_dir,
            (cache / "capacity-retry").string(),
            retry_error);
        bounded_runtime->set_program_cache_capacity_for_test(1);
        const auto first = bounded_runtime->compile_program(
            0xca9e000000000001ULL, plan, retry_error, container);
        M1CompileFailureKind full_kind = M1CompileFailureKind::None;
        const auto full = bounded_runtime->compile_program(
            0xca9e000000000002ULL,
            plan,
            retry_error,
            container,
            &full_kind);
        expect(
            first != nullptr && full == nullptr &&
                full_kind == M1CompileFailureKind::Retryable &&
                bounded_runtime->cache_stats().negative_entries == 0,
            "program-cache capacity failure is retryable and never "
            "negative-cached");
    }

    expect(
        runtime->context().external_buffer_count() == 0,
        "M1 releases channel/logits residency after the last command");
    const TransientBufferPoolStats runtime_pool =
        runtime->context().transient_buffer_pool_stats();
    expect(
        runtime_pool.reuse_hits > 0 &&
            runtime_pool.in_use_buffers == 0 &&
            runtime_pool.resident_bytes <= runtime_pool.capacity_bytes,
        "M1 scratch pool reports hot reuse with zero buffers still in flight");
    std::printf(
        "  POOL M1 allocations=%llu reuse_hits=%llu resident_bytes=%zu "
        "capacity_bytes=%zu\n",
        static_cast<unsigned long long>(runtime_pool.allocations),
        static_cast<unsigned long long>(runtime_pool.reuse_hits),
        runtime_pool.resident_bytes,
        runtime_pool.capacity_bytes);

    ExecPlan misplaced = plan;
    misplaced.region_plans[0].stage = PTIR_STAGE_PROLOGUE;
    error.clear();
    expect(
        runtime->compile_program(
            0x5555666677778888ULL, misplaced, error) == nullptr &&
            error.find("host-resolved descriptor") != std::string::npos,
        "registration rejects unsafe prologue-to-descriptor pending flow");

    std::string collision_error;
    auto collision_runtime = M1Runtime::create(
        kernels_dir, (cache / "collision").string(), collision_error);
    std::vector<std::uint8_t> colliding_bytes =
        plan.region_plans[0].signature;
    colliding_bytes.push_back(0xff);
    if (collision_runtime) {
        collision_runtime->inject_stage_cache_entry_for_test(
            plan.region_plans[0].signature_hash,
            std::move(colliding_bytes));
    }
    const auto collision_result =
        collision_runtime
            ? collision_runtime->compile_program(
                  0x9999888877776666ULL, plan, collision_error)
            : nullptr;
    expect(
        collision_result == nullptr &&
            collision_error.find("signature hash collision") !=
                std::string::npos,
        "stage cache rejects a forced canonical-signature collision");
    collision_runtime.reset();

    runtime.reset();
    error.clear();
    auto restarted = M1Runtime::create(
        kernels_dir, cache.string(), error);
    auto persisted = restarted
                         ? restarted->compile_program(
                               0xff694395428759feULL, plan, error)
                         : nullptr;
    expect(
        persisted != nullptr &&
            restarted->cache_stats().persistent_hits > 0,
        "persistent MTL4 archives survive runtime restart (" + error + ")");
    restarted.reset();

    {
        const std::string config_path =
            "m1-register-retry.generated.toml";
        {
            std::ofstream config(config_path, std::ios::trunc);
            config << "[model]\nbackend = \"metal:0\"\n"
                   << "[batching]\nkv_page_size = 32\n"
                   << "total_pages = 128\n";
        }
        const std::filesystem::path register_cache =
            cache / "context-register-retry";
        std::filesystem::remove_all(register_cache, ec);
        ::setenv(
            "PIE_METAL_PTIR_CACHE_DIR",
            register_cache.c_str(),
            1);
        ::setenv(
            "PIE_METAL_PTIR_TEST_FAIL_COMPILE_ONCE",
            "fault injection: transient driver registration compiler error",
            1);

        PieDriverCreateDesc create{};
        create.abi_version = PIE_DRIVER_ABI_VERSION;
        create.config_bytes = {
            .ptr = reinterpret_cast<const std::uint8_t*>(
                config_path.data()),
            .len = config_path.size(),
        };
        create.runtime.abi_version = PIE_DRIVER_ABI_VERSION;
        create.runtime.notify =
            +[](void*, std::uint64_t, std::uint64_t) {};
        PieDriverCaps caps{};
        PieDriver* driver = pie_metal_create(&create, &caps);
        PieProgramDesc program{};
        program.abi_version = PIE_DRIVER_ABI_VERSION;
        program.program_hash = 0xd21a000000000001ULL;
        program.canonical_bytes = {
            .ptr = container.data(),
            .len = container.size(),
        };
        program.sidecar_bytes = {
            .ptr = sidecar.data(),
            .len = sidecar.size(),
        };
        std::uint64_t program_id = 0;
        const int first_status =
            driver == nullptr
                ? PIE_STATUS_DRIVER_ERROR
                : pie_metal_register_program(
                      driver, &program, &program_id);
        ::unsetenv("PIE_METAL_PTIR_TEST_FAIL_COMPILE_ONCE");
        const int retry_status =
            driver == nullptr
                ? PIE_STATUS_DRIVER_ERROR
                : pie_metal_register_program(
                      driver, &program, &program_id);
        expect(
            driver != nullptr &&
                first_status == PIE_STATUS_DRIVER_ERROR &&
                retry_status == PIE_STATUS_OK && program_id != 0,
            "driver registration leaves record.m1_error clear after a "
            "retryable compiler failure");
        if (driver != nullptr) pie_metal_destroy(driver);
        ::unsetenv("PIE_METAL_PTIR_CACHE_DIR");
        std::remove(config_path.c_str());
    }

    std::filesystem::remove_all(cache, ec);
    std::printf(
        "\n==== m1_generated_test: %d passed, %d failed ====\n",
        g_pass,
        g_fail);
    return g_fail == 0 ? 0 : 1;
}
