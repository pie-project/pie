#pragma once

// Aux IPC server thread. Listens on a unix socket; for each accepted
// connection, reads one (or many) `AuxCmdHeader` + `AuxPagePair[]`
// commands and dispatches page-copy operations against the binary's
// paged KV cache + host swap pool.

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "adapter.hpp"
#include "aux_ipc.hpp"
#include "hf_config.hpp"
#include "kv_cache.hpp"

namespace pie_ggml_driver {

// Host-side swap pool: per-layer K and V buffers of `cpu_pages × page_size`
// rows. Lives in plain process memory; the binary owns it.
class HostSwapPool {
public:
    HostSwapPool(std::int32_t n_layers,
                 std::int32_t n_kv_heads,
                 std::int32_t head_dim,
                 std::int32_t cpu_pages,
                 std::int32_t page_size,
                 std::size_t  dtype_size);

    std::int32_t  cpu_pages()  const noexcept { return cpu_pages_; }
    std::int32_t  page_size()  const noexcept { return page_size_; }
    std::size_t   page_bytes() const noexcept { return page_bytes_; }
    std::int32_t  n_layers()   const noexcept { return n_layers_; }

    // Pointer to the start of the page-th `page_size`-row block of layer
    // `layer`'s K (or V) buffer.
    std::uint8_t* k_slot(std::int32_t layer, std::int32_t page) noexcept;
    std::uint8_t* v_slot(std::int32_t layer, std::int32_t page) noexcept;

private:
    std::int32_t  n_layers_;
    std::int32_t  cpu_pages_;
    std::int32_t  page_size_;
    std::size_t   page_bytes_;  // n_kv_heads * head_dim * page_size * dtype_size
    // Two flat host buffers (K and V) per layer, sized for cpu_pages.
    std::vector<std::vector<std::uint8_t>> k_buffers_;
    std::vector<std::vector<std::uint8_t>> v_buffers_;
};

// Thread that owns the unix socket. Construct with a path; it starts
// listening immediately. Joins on destruction.
class AuxServer {
public:
    AuxServer(const std::string& socket_path,
              KvCachePaged& kv,
              HostSwapPool* swap /* nullable; nullptr → swap ops fail */,
              AdapterPool*  adapters /* nullable; nullptr → load_adapter fails */,
              ggml_backend_t adapter_backend = nullptr,
              const Hparams* hparams = nullptr);
    ~AuxServer();

    AuxServer(const AuxServer&) = delete;
    AuxServer& operator=(const AuxServer&) = delete;

    // Stop the listener. Safe to call multiple times.
    void stop();

private:
    void run_();
    aux::Status handle_command_(aux::Method method,
                                const std::vector<aux::AuxPagePair>& pages);

    KvCachePaged&     kv_;
    HostSwapPool*     swap_;
    AdapterPool*      adapters_       = nullptr;
    ggml_backend_t    adapter_backend_ = nullptr;
    const Hparams*    hparams_         = nullptr;
    std::string       socket_path_;
    int               listen_fd_ = -1;
    std::thread       thread_;
    std::atomic<bool> stop_{false};
};

}  // namespace pie_ggml_driver
