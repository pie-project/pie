// Oracle for tests/dtoa_parity.rs: nlohmann's own double formatter.
//
// Unlike the store oracle this extracts nothing from the driver sources --
// the thing under test is `nlohmann::json::dump()` itself, so the library IS
// the oracle. What has to match on both sides is the corpus, which is why it
// is generated from an explicit LCG with the constants written out rather
// than from any language's random facility.
//
// Keep every line of `emit` and the two generators identical to
// tests/dtoa_parity.rs.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <nlohmann/json.hpp>

namespace {

// Knuth's MMIX LCG. Reproduced verbatim in the Rust side.
std::uint64_t next(std::uint64_t& x) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    return x;
}

void emit(double v) {
    nlohmann::json j = v;
    // The bit pattern travels with the text so a mismatch names the input
    // exactly, instead of leaving a decimal string to be reverse engineered.
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof bits);
    std::printf("%016llx\t%s\n", static_cast<unsigned long long>(bits),
                j.dump().c_str());
}

}  // namespace

int main() {
    // --- edge cases, spelled out -------------------------------------------
    const double edges[] = {
        0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 0.1, 100.0, 12345.0,
        1e-4, 1e-5, 1e-6, 1e-7, 1e14, 1e15, 1e16, 1e17, 1e21, 1e-21,
        1e100, 1e-100, 1.0 / 3.0, 123456789.123456, 3.0e30, 2.5e-30,
        5e-324, 1.7976931348623157e308, 2.2250738585072014e-308,
        // Boundary of the fixed-point window from both sides.
        999999999999999.0, 1000000000000000.0, 0.00009999, 0.0001,
        // Powers of two: the `lower_boundary_is_closer` branch.
        2.0, 4.0, 1024.0, 4503599627370496.0, 2.0e-308,
        // The four Grisu2-vs-shortest disagreements found by sweeping.
        46934.815584012416, 72972.67707126706, 27453.918300648482,
        3.4110366750178187e-295,
    };
    for (double v : edges) emit(v);

    // --- the range these fields actually hold: ms and tokens/s -------------
    std::uint64_t x = 88172645463325252ULL;
    for (int i = 0; i < 100000; ++i) {
        const std::uint64_t r = next(x);
        // 53 bits of mantissa scaled into [0, 100000).
        const double unit =
            static_cast<double>(r >> 11) / 9007199254740992.0;  // 2^53
        emit(unit * 100000.0);
    }

    // --- arbitrary finite bit patterns -------------------------------------
    std::uint64_t y = 1234567890123456789ULL;
    for (int i = 0; i < 100000; ++i) {
        double v;
        std::uint64_t bits;
        do {
            bits = next(y);
            std::memcpy(&v, &bits, sizeof v);
        } while (!std::isfinite(v));
        emit(v);
    }
    return 0;
}
