#include "control_socket.hpp"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

namespace pie_cuda_driver {

void serve_control_socket(int fd, const CtrlHandler& handler) {
    // Generous buffer — header (16 B) + up to 64 K pairs (512 KiB).
    std::vector<std::uint8_t> buf(1 << 20);

    while (true) {
        const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
        if (n == 0) return;  // peer closed
        if (n < 0) {
            if (errno == EINTR) continue;
            std::cerr << "[pie-driver-cuda] control socket recv failed: "
                      << std::strerror(errno) << "\n";
            return;
        }
        if (static_cast<std::size_t>(n) < 16) {
            std::cerr << "[pie-driver-cuda] control socket short read (" << n
                      << " bytes)\n";
            std::uint32_t status = 1;
            ::send(fd, &status, sizeof(status), 0);
            continue;
        }

        std::uint32_t header[4];
        std::memcpy(header, buf.data(), sizeof(header));
        const auto* pairs =
            reinterpret_cast<const std::uint32_t*>(buf.data() + 16);
        const std::size_t expected =
            16 + static_cast<std::size_t>(header[2]) * 8;
        if (static_cast<std::size_t>(n) < expected) {
            std::cerr << "[pie-driver-cuda] control socket truncated message: "
                      << "have=" << n << " expected=" << expected << "\n";
            std::uint32_t status = 1;
            ::send(fd, &status, sizeof(status), 0);
            continue;
        }

        CtrlRequest req{
            .method = header[0],
            .layer  = header[1],
            .num_pairs = header[2],
            .src_dst_pairs = pairs,
        };

        std::uint32_t status = 1;
        try {
            status = handler(req);
        } catch (const std::exception& e) {
            std::cerr << "[pie-driver-cuda] control handler threw: "
                      << e.what() << "\n";
            status = 2;
        }

        const ssize_t sent = ::send(fd, &status, sizeof(status), 0);
        if (sent < 0) {
            std::cerr << "[pie-driver-cuda] control socket send failed: "
                      << std::strerror(errno) << "\n";
            return;
        }
    }
}

}  // namespace pie_cuda_driver
