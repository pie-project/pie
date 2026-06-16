#pragma once

// Cross-server-instance helpers used during driver startup/shutdown:
//   - The in-process server registry (so a SIGINT/SIGTERM can fan out to
//     every loaded server in the address space).
//   - The shared `custom_all_reduce` payload sizing knob.
//   - The TP-rank CPU rendezvous barrier — every TP rank is a thread in
//     the same pie-server process, and they must reach the post-load
//     point together before any rank publishes READY.

#include <cstddef>

namespace pie_driver { class InProcServer; }

namespace pie_cuda_driver {

struct Config;

std::size_t custom_all_reduce_max_bytes();

// Registry of all live in-proc servers. SIGINT/SIGTERM's handler walks
// this list to stop every loaded server (not just the last-registered).
void register_server(pie_driver::InProcServer* server);
void unregister_server(pie_driver::InProcServer* server);
void stop_servers();

// Signal handler suitable for std::signal — stops the most recently
// registered server. Does nothing when no server is registered.
void on_signal(int signum);

// CPU rendezvous keyed by `cfg.distributed.nccl_unique_id_hex`. No-op
// when tp_size <= 1 or no key is configured. Used to gate post-load
// readiness so all TP ranks in one DP group reach the executor together.
void tp_startup_cpu_barrier(const Config& cfg);

}  // namespace pie_cuda_driver
