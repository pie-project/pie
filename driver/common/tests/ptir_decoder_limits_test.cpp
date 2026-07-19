#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "pie_native/ptir/bound.hpp"
#include "pie_native/ptir/plan.hpp"

static_assert(sizeof(PtirLaneTableHeader) == 16);
static_assert(sizeof(PtirLaneRecord) == 96);
static_assert(sizeof(PtirLaneChannelSlot) == 32);

namespace {
std::atomic<std::size_t> allocation_count{0};
thread_local bool track_allocations = false;
}

void* operator new(std::size_t size) {
    if (track_allocations) ++allocation_count;
    if (void* value = std::malloc(size == 0 ? 1 : size)) return value;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* value) noexcept {
    std::free(value);
}

void operator delete[](void* value) noexcept {
    std::free(value);
}

void operator delete(void* value, std::size_t) noexcept {
    std::free(value);
}

void operator delete[](void* value, std::size_t) noexcept {
    std::free(value);
}

namespace {

struct Case {
    std::string kind;
    bool accept = false;
    std::string name;
    std::vector<std::uint8_t> bytes;
};

std::vector<std::uint8_t> decode_hex(const std::string& hex) {
    if (hex.size() % 2 != 0) throw std::runtime_error("odd corpus hex length");
    std::vector<std::uint8_t> bytes;
    bytes.reserve(hex.size() / 2);
    for (std::size_t offset = 0; offset < hex.size(); offset += 2) {
        bytes.push_back(static_cast<std::uint8_t>(
            std::stoul(hex.substr(offset, 2), nullptr, 16)));
    }
    return bytes;
}

std::vector<Case> read_corpus(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("unable to open corpus: " + path);
    std::vector<Case> cases;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::istringstream fields(line);
        Case test;
        std::string expected;
        std::string hex;
        if (!(fields >> test.kind >> expected >> test.name >> hex)) {
            throw std::runtime_error("invalid corpus row");
        }
        test.accept = expected == "accept";
        test.bytes = decode_hex(hex);
        cases.push_back(std::move(test));
    }
    return cases;
}

bool accepted(const Case& test) {
    if (test.kind == "PTIB") {
        pie_native::ptir::bound::Bound decoded;
        return pie_native::ptir::bound::parse_sidecar(
            test.bytes.data(), test.bytes.size(), decoded, nullptr);
    }
    if (test.kind == "PTRP") {
        pie_native::ptir::plan::StagePlan decoded;
        return pie_native::ptir::plan::decode(
            test.bytes.data(), test.bytes.size(), decoded, nullptr);
    }
    throw std::runtime_error("unknown corpus kind: " + test.kind);
}

bool validate_golden_sidecar(const std::string& path) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line)) {
        constexpr const char* prefix = "sidecar:";
        if (line.rfind(prefix, 0) != 0) continue;
        const auto first = line.find_first_not_of(" \t", std::strlen(prefix));
        const std::vector<std::uint8_t> bytes =
            decode_hex(first == std::string::npos ? "" : line.substr(first));
        pie_native::ptir::bound::Bound bound;
        if (!pie_native::ptir::bound::parse_sidecar(
                bytes.data(), bytes.size(), bound, nullptr)) {
            return false;
        }
        for (const auto& encoded : bound.plans) {
            pie_native::ptir::plan::StagePlan plan;
            if (!pie_native::ptir::plan::decode(
                    encoded.bytes.data(),
                    encoded.bytes.size(),
                    plan,
                    nullptr)) {
                return false;
            }
        }
        return true;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1
        ? argv[1]
        : "interface/ptir/tests/malformed_wire_corpus.txt";
    const std::vector<Case> cases = read_corpus(path);
    std::size_t short_bombs = 0;
    for (const Case& test : cases) {
        const bool got = accepted(test);
        if (got != test.accept) {
            std::cerr << test.kind << ' ' << test.name
                      << " acceptance mismatch\n";
            return 1;
        }
        if (!test.accept && test.bytes.size() >= 20 &&
            test.bytes.size() <= 40) {
            ++short_bombs;
            allocation_count = 0;
            track_allocations = true;
            const bool accepted_while_tracking = accepted(test);
            track_allocations = false;
            if (accepted_while_tracking || allocation_count != 0) {
                std::cerr << test.name
                          << " allocated while rejecting a short count bomb\n";
                return 1;
            }
        }
    }
    if (short_bombs < 12) {
        std::cerr << "short adversarial corpus coverage regressed\n";
        return 1;
    }

    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < 4096; ++iteration) {
        for (const Case& test : cases) {
            if (!test.accept && accepted(test)) {
                std::cerr << test.name << " unexpectedly accepted\n";
                return 1;
            }
        }
    }
    if (std::chrono::steady_clock::now() - start >
        std::chrono::seconds(2)) {
        std::cerr << "malformed corpus rejection exceeded CPU budget\n";
        return 1;
    }
    for (int index = 2; index < argc; ++index) {
        if (!validate_golden_sidecar(argv[index])) {
            std::cerr << "canonical sidecar rejected: " << argv[index] << '\n';
            return 1;
        }
    }
    std::cout << "PTIB/PTRP decoder corpus: " << cases.size()
              << " cases and " << (argc > 2 ? argc - 2 : 0)
              << " canonical sidecars passed\n";
    return 0;
}
