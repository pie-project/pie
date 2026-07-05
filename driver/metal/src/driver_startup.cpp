#include "driver_startup.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

#include <pie_ipc/inproc_server.hpp>

namespace pie_metal_driver {

namespace {

std::mutex g_servers_mu;
std::vector<pie_driver::InProcServer*> g_servers;
std::atomic<pie_driver::InProcServer*> g_signal_server{nullptr};

}  // namespace

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

}  // namespace pie_metal_driver
