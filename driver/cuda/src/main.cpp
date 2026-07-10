#include <iostream>
#include <string>

#include <pie_driver_abi.h>

namespace {

void noop_notify(void*, std::uint64_t, std::uint64_t) {}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path = "dev.toml";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        } else {
            std::cerr << "usage: pie_driver_cuda [-c CONFIG]" << std::endl;
            return 1;
        }
    }

    PieDriverCreateDesc desc{};
    desc.abi_version = PIE_DRIVER_ABI_VERSION;
    desc.reserved0 = 0;
    desc.runtime.abi_version = PIE_DRIVER_ABI_VERSION;
    desc.runtime.reserved0 = 0;
    desc.runtime.notify = noop_notify;
    desc.config_bytes.ptr = reinterpret_cast<const std::uint8_t*>(config_path.data());
    desc.config_bytes.len = config_path.size();
    PieDriverCaps caps{};
    PieDriver* driver = pie_cuda_create(&desc, &caps);
    if (driver == nullptr) return 1;
    std::cout << "READY "
              << std::string(
                     reinterpret_cast<const char*>(caps.json_bytes), caps.json_len)
              << std::endl;
    pie_cuda_destroy(driver);
    return 0;
}
