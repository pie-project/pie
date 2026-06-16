#include "driver_startup.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <pie_bridge/inproc_server.hpp>

#include "config.hpp"

namespace pie_cuda_driver {

namespace {

std::mutex g_servers_mu;
std::vector<pie_driver::InProcServer*> g_servers;
std::atomic<pie_driver::InProcServer*> g_signal_server{nullptr};

}  // namespace

std::size_t custom_all_reduce_max_bytes() {
    const char* v = std::getenv("PIE_CUDA_CUSTOM_ALL_REDUCE_MAX_MIB");
    if (v == nullptr || v[0] == '\0') return 64ull * 1024ull * 1024ull;
    const unsigned long long mib = std::strtoull(v, nullptr, 10);
    if (mib == 0ull) return 64ull * 1024ull * 1024ull;
    return static_cast<std::size_t>(mib) * 1024ull * 1024ull;
}

void register_server(pie_driver::InProcServer* server) {
    std::lock_guard<std::mutex> lk(g_servers_mu);
    g_servers.push_back(server);
    g_signal_server.store(server);
}

void unregister_server(pie_driver::InProcServer* server) {
    std::lock_guard<std::mutex> lk(g_servers_mu);
    g_servers.erase(
        std::remove(g_servers.begin(), g_servers.end(), server),
        g_servers.end());
    if (g_signal_server.load() == server) {
        g_signal_server.store(g_servers.empty() ? nullptr : g_servers.back());
    }
}

void stop_servers() {
    std::vector<pie_driver::InProcServer*> servers;
    {
        std::lock_guard<std::mutex> lk(g_servers_mu);
        servers = g_servers;
    }
    for (auto* server : servers) {
        if (server != nullptr) server->stop();
    }
}

void on_signal(int) {
    if (auto* server = g_signal_server.load()) server->stop();
}

void tp_startup_cpu_barrier(const Config& cfg) {
    if (cfg.distributed.tp_size <= 1) return;

    const std::string& key = cfg.distributed.nccl_unique_id_hex;
    if (key.empty()) return;

    static std::mutex registry_mu;
    static std::unordered_map<std::string, std::shared_ptr<std::barrier<>>>
        registry;

    std::shared_ptr<std::barrier<>> b;
    {
        std::lock_guard<std::mutex> lk(registry_mu);
        auto& entry = registry[key];
        if (!entry) {
            entry = std::make_shared<std::barrier<>>(cfg.distributed.tp_size);
        }
        b = entry;
    }
    b->arrive_and_wait();
}

}  // namespace pie_cuda_driver
