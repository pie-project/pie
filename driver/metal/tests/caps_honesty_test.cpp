#include <cstdio>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>
#include <pie_driver_abi.h>

namespace {

void notify_cb(void*, std::uint64_t, std::uint64_t) {}

bool expect(bool condition, const char* message) {
    std::printf("  %s  %s\n", condition ? "PASS" : "FAIL", message);
    return condition;
}

}  // namespace

int main() {
    const std::string config_path = "caps_honesty.generated.toml";
    {
        std::ofstream config(config_path, std::ios::trunc);
        config << "[model]\nbackend = \"metal:0\"\n"
               << "[batching]\nkv_page_size = 32\ntotal_pages = 128\n";
    }

    PieDriverCreateDesc create{};
    create.abi_version = PIE_DRIVER_ABI_VERSION;
    create.config_bytes.ptr =
        reinterpret_cast<const std::uint8_t*>(config_path.data());
    create.config_bytes.len = config_path.size();
    create.runtime.abi_version = PIE_DRIVER_ABI_VERSION;
    create.runtime.notify = notify_cb;
    PieDriverCaps payload{};
    PieDriver* driver = pie_metal_create(&create, &payload);
    std::remove(config_path.c_str());
    if (!expect(driver != nullptr, "create succeeds without reading a checkpoint")) {
        return 1;
    }

    const auto facts = nlohmann::json::parse(
        payload.json_bytes, payload.json_bytes + payload.json_len);
    int failures = 0;
    failures += !expect(
        facts.at("abi_version").get<std::uint32_t>() ==
            PIE_DRIVER_ABI_VERSION,
        "device facts carry the direct ABI version");
    failures += !expect(
        facts.at("backend") == "metal",
        "device facts identify the Metal backend");
    failures += !expect(
        facts.at("unified_memory").get<bool>(),
        "Metal reports unified memory");
    failures += !expect(
        facts.at("storage_alignment").get<std::uint32_t>() > 0,
        "storage alignment is device-derived and nonzero");
    failures += !expect(
        facts.at("page_size").get<std::uint32_t>() > 0,
        "host/GPU page size is reported");
    failures += !expect(
        !facts.contains("arch_name") && !facts.contains("total_pages"),
        "create payload contains no model-derived capabilities");

    pie_metal_destroy(driver);
    return failures == 0 ? 0 : 1;
}
