#pragma once

// Server registry + signal handling for the in-process Metal driver.
// Mirrors driver/cuda/src/driver_startup.{hpp,cpp}, minus the CUDA/NCCL
// tensor-parallel barrier (Metal is single-device for now).

namespace pie_driver {
class InProcServer;
}  // namespace pie_driver

namespace pie_metal_driver {

// Track live in-process servers so a SIGINT/SIGTERM (standalone) or a
// `pie_driver_metal_request_stop()` call from the host can stop them all.
void register_server(pie_driver::InProcServer* server);
void unregister_server(pie_driver::InProcServer* server);
void stop_servers();

// Installed as the SIGINT/SIGTERM handler when the standalone executable
// asks for signal handling. Library callers own their own signals.
void on_signal(int signum);

}  // namespace pie_metal_driver
