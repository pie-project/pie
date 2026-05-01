// Smoke harness for `response::write_msgpack_response` + `write_flat_response`.
//
// Writes deterministic fixture payloads to two output files (paths
// passed via argv); a sibling Python test reads them via the `msgpack`
// library and validates the structure round-trips. Splitting C++ build
// from Python verification means we don't link msgpack-c into the
// driver just to test it.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "response_writer.hpp"

namespace {

void write_file(const std::string& path,
                const std::uint8_t* data, std::size_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "open %s: %s\n", path.c_str(), std::strerror(errno));
        std::exit(2);
    }
    if (std::fwrite(data, 1, n, f) != n) {
        std::fprintf(stderr, "write %s short\n", path.c_str());
        std::exit(2);
    }
    std::fclose(f);
}

// Fixture A: flat response. 3 requests, counts {2, 0, 1}, tokens
// concatenated as [11, 12, 99].
void emit_flat(const std::string& out_path) {
    namespace R = pie_cuda_driver::response;
    std::vector<std::uint32_t> counts = {2u, 0u, 1u};
    std::vector<std::uint32_t> tokens = {11u, 12u, 99u};
    std::vector<std::uint8_t> buf(R::flat_response_size(counts) + 16);
    const std::size_t n = R::write_flat_response(
        std::span<std::uint8_t>(buf), counts, tokens);
    write_file(out_path, buf.data(), n);
}

// Fixture B: msgpack response covering all populated fields:
//   req 0: 1 token + 1 dist (top-3) + 1 logits payload (8 f32 bytes).
//   req 1: 2 logprobs entries (lengths 1 and 3) + 1 entropy.
//   req 2: empty (every field zero-length).
void emit_msgpack(const std::string& out_path) {
    namespace R = pie_cuda_driver::response;

    std::vector<std::uint32_t> tokens_r0 = {7u};
    std::vector<std::uint32_t> tokens_r1 = {};
    std::vector<std::uint32_t> tokens_r2 = {};

    std::vector<std::uint8_t> logits_payload(8u * sizeof(float), 0u);
    // Stuff a recognizable f32 pattern: 1.0 at slot 0, 2.0 at slot 1.
    auto* lp = reinterpret_cast<float*>(logits_payload.data());
    lp[0] = 1.0f;
    lp[1] = 2.0f;

    std::vector<R::PerRequestMsgpack> per_req(3);
    per_req[0].tokens = std::span<const std::uint32_t>(tokens_r0);
    per_req[0].logits.push_back(logits_payload);
    per_req[0].dists.emplace_back(
        std::vector<std::uint32_t>{42u, 13u, 7u},
        std::vector<float>{0.5f, 0.3f, 0.2f});

    per_req[1].tokens = std::span<const std::uint32_t>(tokens_r1);
    per_req[1].logprobs.push_back({-1.5f});
    per_req[1].logprobs.push_back({-0.1f, -2.2f, -3.3f});
    per_req[1].entropies.push_back(2.5f);

    per_req[2].tokens = std::span<const std::uint32_t>(tokens_r2);

    std::vector<std::uint8_t> buf(64 * 1024, 0);
    const std::size_t n = R::write_msgpack_response(
        std::span<std::uint8_t>(buf), per_req);
    write_file(out_path, buf.data(), n);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
            "usage: %s <flat-out> <msgpack-out>\n", argv[0]);
        return 2;
    }
    emit_flat(argv[1]);
    emit_msgpack(argv[2]);
    return 0;
}
