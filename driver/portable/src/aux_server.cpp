#include "aux_server.hpp"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifndef _WIN32
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include <ggml-backend.h>

namespace pie_portable_driver {

// =============================================================================
// HostSwapPool
// =============================================================================

HostSwapPool::HostSwapPool(std::int32_t n_layers,
                           std::int32_t n_kv_heads,
                           std::int32_t head_dim,
                           std::int32_t cpu_pages,
                           std::int32_t page_size,
                           std::size_t  dtype_size)
    : n_layers_(n_layers),
      cpu_pages_(cpu_pages),
      page_size_(page_size),
      page_bytes_(static_cast<std::size_t>(n_kv_heads) * head_dim
                  * page_size * dtype_size) {
    if (cpu_pages <= 0) return;  // empty pool — caller should treat as absent
    const std::size_t per_buf = static_cast<std::size_t>(cpu_pages) * page_bytes_;
    k_buffers_.resize(n_layers);
    v_buffers_.resize(n_layers);
    for (std::int32_t il = 0; il < n_layers; ++il) {
        k_buffers_[il].assign(per_buf, 0);
        v_buffers_[il].assign(per_buf, 0);
    }
}

std::uint8_t* HostSwapPool::k_slot(std::int32_t layer, std::int32_t page) noexcept {
    return k_buffers_[layer].data() + static_cast<std::size_t>(page) * page_bytes_;
}
std::uint8_t* HostSwapPool::v_slot(std::int32_t layer, std::int32_t page) noexcept {
    return v_buffers_[layer].data() + static_cast<std::size_t>(page) * page_bytes_;
}

// =============================================================================
// AuxServer
// =============================================================================

#ifdef _WIN32

AuxServer::AuxServer(const std::string& socket_path,
                     KvCachePaged& kv,
                     HostSwapPool* swap,
                     AdapterPool* adapters,
                     ggml_backend_t adapter_backend,
                     const Hparams* hparams)
    : kv_(kv), swap_(swap),
      adapters_(adapters),
      adapter_backend_(adapter_backend),
      hparams_(hparams),
      socket_path_(socket_path) {
    throw std::runtime_error("aux IPC is not supported on Windows yet");
}

AuxServer::~AuxServer() = default;
void AuxServer::stop() {}
void AuxServer::run_() {}
aux::Status AuxServer::handle_command_(
        aux::Method,
        const std::vector<aux::AuxPagePair>&) {
    return aux::Status::BackendError;
}

#else

namespace {

bool read_full(int fd, void* buf, std::size_t n) {
    auto* p = static_cast<std::uint8_t*>(buf);
    std::size_t got = 0;
    while (got < n) {
        const ssize_t r = ::read(fd, p + got, n - got);
        if (r > 0) { got += static_cast<std::size_t>(r); continue; }
        if (r == 0) return false;             // EOF
        if (errno == EINTR) continue;
        return false;
    }
    return true;
}

bool write_full(int fd, const void* buf, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(buf);
    std::size_t sent = 0;
    while (sent < n) {
        const ssize_t r = ::write(fd, p + sent, n - sent);
        if (r > 0) { sent += static_cast<std::size_t>(r); continue; }
        if (errno == EINTR) continue;
        return false;
    }
    return true;
}

}  // namespace

AuxServer::AuxServer(const std::string& socket_path,
                     KvCachePaged& kv,
                     HostSwapPool* swap,
                     AdapterPool*  adapters,
                     ggml_backend_t adapter_backend,
                     const Hparams* hparams)
    : kv_(kv), swap_(swap),
      adapters_(adapters),
      adapter_backend_(adapter_backend),
      hparams_(hparams),
      socket_path_(socket_path) {
    listen_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        throw std::runtime_error("aux_server: socket() failed: " +
                                 std::string(std::strerror(errno)));
    }

    // Best-effort cleanup of any stale socket from a prior run.
    ::unlink(socket_path_.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socket_path_.size() + 1 > sizeof(addr.sun_path)) {
        ::close(listen_fd_);
        throw std::runtime_error("aux_server: socket path too long: " + socket_path_);
    }
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(listen_fd_);
        throw std::runtime_error("aux_server: bind() failed: " +
                                 std::string(std::strerror(errno)));
    }
    if (::listen(listen_fd_, /*backlog=*/ 4) != 0) {
        ::close(listen_fd_);
        ::unlink(socket_path_.c_str());
        throw std::runtime_error("aux_server: listen() failed: " +
                                 std::string(std::strerror(errno)));
    }

    int p[2];
    if (::pipe(p) != 0) {
        ::close(listen_fd_);
        ::unlink(socket_path_.c_str());
        throw std::runtime_error("aux_server: pipe() failed: " +
                                 std::string(std::strerror(errno)));
    }
    ::fcntl(p[0], F_SETFD, FD_CLOEXEC);
    ::fcntl(p[1], F_SETFD, FD_CLOEXEC);
    ::fcntl(p[0], F_SETFL, O_NONBLOCK);
    wakeup_read_  = p[0];
    wakeup_write_ = p[1];

    thread_ = std::thread([this] { run_(); });
}

AuxServer::~AuxServer() {
    stop();
    if (thread_.joinable()) thread_.join();
    if (wakeup_read_  >= 0) ::close(wakeup_read_);
    if (wakeup_write_ >= 0) ::close(wakeup_write_);
    if (listen_fd_ >= 0) ::close(listen_fd_);
    if (!socket_path_.empty()) ::unlink(socket_path_.c_str());
}

void AuxServer::stop() {
    stop_.store(true, std::memory_order_relaxed);
    // Wake the listener thread out of `poll()` (or `accept()` if it
    // somehow bypassed the poll). Best-effort: a full pipe means a
    // wakeup is already pending, which is fine.
    if (wakeup_write_ >= 0) {
        const std::uint8_t b = 1;
        ssize_t r = ::write(wakeup_write_, &b, 1);
        (void)r;
    }
}

void AuxServer::run_() {
    while (!stop_.load(std::memory_order_relaxed)) {
        struct pollfd pfds[2] = {
            { listen_fd_,    POLLIN, 0 },
            { wakeup_read_,  POLLIN, 0 },
        };
        const int prc = ::poll(pfds, 2, /*timeout=*/-1);
        if (prc < 0) {
            if (errno == EINTR) continue;
            std::cerr << "[pie-driver-portable] aux poll() failed: "
                      << std::strerror(errno) << "\n";
            break;
        }
        if (pfds[1].revents & (POLLIN | POLLHUP | POLLERR)) break;
        if (!(pfds[0].revents & POLLIN)) continue;

        const int conn = ::accept(listen_fd_, nullptr, nullptr);
        if (conn < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            if (stop_.load(std::memory_order_relaxed)) break;
            std::cerr << "[pie-driver-portable] aux accept() failed: "
                      << std::strerror(errno) << "\n";
            break;
        }

        // Process commands on this connection until the peer closes it.
        while (!stop_.load(std::memory_order_relaxed)) {
            aux::AuxCmdHeader hdr{};
            if (!read_full(conn, &hdr, sizeof(hdr))) break;

            aux::Status status = aux::Status::Ok;
            std::vector<aux::AuxPagePair> pages;

            if (hdr.magic != aux::MAGIC) {
                status = aux::Status::BadMagic;
            } else if (hdr.method == static_cast<std::uint32_t>(aux::Method::LoadAdapter)) {
                // Variable-length payload: AuxLoadAdapterPayload + path bytes.
                aux::AuxLoadAdapterPayload p{};
                if (!read_full(conn, &p, sizeof(p))) break;
                std::string path(p.path_len, '\0');
                if (p.path_len > 0 && !read_full(conn, path.data(), p.path_len)) {
                    break;
                }
                try {
                    if (!adapters_) {
                        status = aux::Status::NoSwapPool;  // reuse for "no pool"
                    } else if (!adapter_backend_ || !hparams_) {
                        status = aux::Status::BackendError;
                    } else {
                        // Standard PEFT scaling: alpha = 2*rank by default
                        // unless adapter_config overrides; the wrapper
                        // doesn't currently parse adapter_config and the
                        // file alone doesn't carry alpha. Use scale=1 for
                        // v1; the wrapper can override later via metadata
                        // RPCs (post-v1).
                        const std::int32_t guessed_rank = 0;  // resolved
                        // by reading shapes inside Adapter — but we need
                        // a number here for the API. Use 0 as sentinel
                        // ("auto") and let Adapter ignore it for now.
                        const float scale = 1.0f;
                        auto adapter = std::make_unique<Adapter>(
                            adapter_backend_,
                            hparams_->num_hidden_layers,
                            guessed_rank,
                            scale,
                            std::filesystem::path(path),
                            *hparams_);
                        adapters_->insert(p.adapter_id, std::move(adapter));
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[pie-driver-portable] LoadAdapter error: "
                              << e.what() << "\n";
                    status = aux::Status::BackendError;
                }
            } else if (hdr.n_pages > 0) {
                pages.resize(hdr.n_pages);
                if (!read_full(conn, pages.data(),
                               sizeof(aux::AuxPagePair) * hdr.n_pages)) {
                    break;  // connection broken mid-command
                }
            }

            if (status == aux::Status::Ok
                && hdr.method != static_cast<std::uint32_t>(aux::Method::LoadAdapter)) {
                try {
                    status = handle_command_(static_cast<aux::Method>(hdr.method),
                                             pages);
                } catch (const std::exception& e) {
                    std::cerr << "[pie-driver-portable] aux handler error: "
                              << e.what() << "\n";
                    status = aux::Status::BackendError;
                }
            }

            aux::AuxAck ack{};
            ack.magic  = aux::ACK_MAGIC;
            ack.status = static_cast<std::uint32_t>(status);
            ack.req_id = hdr.req_id;
            if (!write_full(conn, &ack, sizeof(ack))) break;
        }

        ::close(conn);
    }
}

aux::Status AuxServer::handle_command_(
        aux::Method method,
        const std::vector<aux::AuxPagePair>& pages) {
    const std::int32_t total_pages_dev = kv_.total_pages();
    const std::int32_t cpu_pages = swap_ ? swap_->cpu_pages() : 0;
    const std::size_t  page_bytes =
        static_cast<std::size_t>(kv_.n_embd_gqa()) * kv_.page_size()
        * ggml_type_size(kv_.k(0)->type);

    auto check_dev = [&](std::uint32_t p) -> bool {
        return p < static_cast<std::uint32_t>(total_pages_dev);
    };
    auto check_host = [&](std::uint32_t p) -> bool {
        return p < static_cast<std::uint32_t>(cpu_pages);
    };

    switch (method) {
        case aux::Method::CopyD2H: {
            if (!swap_) return aux::Status::NoSwapPool;
            for (auto pair : pages) {
                if (!check_dev(pair.src) || !check_host(pair.dst))
                    return aux::Status::OutOfBounds;
                const std::size_t offset =
                    static_cast<std::size_t>(pair.src) * page_bytes;
                for (std::int32_t il = 0; il < kv_.n_layers(); ++il) {
                    ggml_backend_tensor_get(kv_.k(il), swap_->k_slot(il, pair.dst),
                                            offset, page_bytes);
                    ggml_backend_tensor_get(kv_.v(il), swap_->v_slot(il, pair.dst),
                                            offset, page_bytes);
                }
            }
            return aux::Status::Ok;
        }
        case aux::Method::CopyH2D: {
            if (!swap_) return aux::Status::NoSwapPool;
            for (auto pair : pages) {
                if (!check_host(pair.src) || !check_dev(pair.dst))
                    return aux::Status::OutOfBounds;
                const std::size_t offset =
                    static_cast<std::size_t>(pair.dst) * page_bytes;
                for (std::int32_t il = 0; il < kv_.n_layers(); ++il) {
                    ggml_backend_tensor_set(kv_.k(il), swap_->k_slot(il, pair.src),
                                            offset, page_bytes);
                    ggml_backend_tensor_set(kv_.v(il), swap_->v_slot(il, pair.src),
                                            offset, page_bytes);
                }
            }
            return aux::Status::Ok;
        }
        case aux::Method::CopyD2D: {
            // Device-to-device. ggml has no "partial copy within tensor"
            // primitive that's universally supported across backends, so
            // we round-trip through a small host buffer per page.
            std::vector<std::uint8_t> tmp(page_bytes);
            for (auto pair : pages) {
                if (!check_dev(pair.src) || !check_dev(pair.dst))
                    return aux::Status::OutOfBounds;
                const std::size_t src_off =
                    static_cast<std::size_t>(pair.src) * page_bytes;
                const std::size_t dst_off =
                    static_cast<std::size_t>(pair.dst) * page_bytes;
                for (std::int32_t il = 0; il < kv_.n_layers(); ++il) {
                    ggml_backend_tensor_get(kv_.k(il), tmp.data(), src_off, page_bytes);
                    ggml_backend_tensor_set(kv_.k(il), tmp.data(), dst_off, page_bytes);
                    ggml_backend_tensor_get(kv_.v(il), tmp.data(), src_off, page_bytes);
                    ggml_backend_tensor_set(kv_.v(il), tmp.data(), dst_off, page_bytes);
                }
            }
            return aux::Status::Ok;
        }
        case aux::Method::CopyH2H: {
            if (!swap_) return aux::Status::NoSwapPool;
            for (auto pair : pages) {
                if (!check_host(pair.src) || !check_host(pair.dst))
                    return aux::Status::OutOfBounds;
                for (std::int32_t il = 0; il < kv_.n_layers(); ++il) {
                    std::memcpy(swap_->k_slot(il, pair.dst),
                                swap_->k_slot(il, pair.src), page_bytes);
                    std::memcpy(swap_->v_slot(il, pair.dst),
                                swap_->v_slot(il, pair.src), page_bytes);
                }
            }
            return aux::Status::Ok;
        }
        default:
            return aux::Status::BadMethod;
    }
}

#endif

}  // namespace pie_portable_driver
