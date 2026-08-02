// Loads a relocatable launch-package image (see `interface/driver/src/image.rs`)
// back into a `PieLaunchPackage`.
//
// Test fixtures only. The engine builds the package in memory and hands the
// driver a pointer; nothing in production reads a file. This exists so the
// native driver tests can register the same program the engine would without
// carrying a PTIR decoder to build it from.
//
// Soundness rests on Rust and C++ agreeing byte-for-byte on every `PieLaunch*`
// struct, which `interface/driver/tests/header_layout.rs` gates.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <pie_driver_abi.h>

#include "pie/driver/validate.hpp"

namespace pie::driver::testing {

class PackageImage {
  public:
    // `PIELPKG\0`
    static constexpr std::uint64_t kMagic = 0x0047'4b50'4c45'4950ULL;

    bool load(const std::string& path, std::string* error) {
        std::FILE* file = std::fopen(path.c_str(), "rb");
        if (file == nullptr) {
            if (error != nullptr) *error = "cannot open launch image " + path;
            return false;
        }
        std::vector<std::uint8_t> raw;
        std::uint8_t chunk[65536];
        while (const std::size_t got = std::fread(chunk, 1, sizeof(chunk), file)) {
            raw.insert(raw.end(), chunk, chunk + got);
        }
        std::fclose(file);
        return adopt(raw, error);
    }

    bool adopt(const std::vector<std::uint8_t>& raw, std::string* error) {
        auto fail = [&](const char* message) {
            if (error != nullptr) *error = message;
            return false;
        };
        constexpr std::size_t kHeader = 5 * sizeof(std::uint64_t);
        if (raw.size() < kHeader) return fail("launch image is truncated");
        std::uint64_t header[5];
        std::memcpy(header, raw.data(), kHeader);
        if (header[0] != kMagic) return fail("launch image has a bad magic");
        if (header[1] != PIE_DRIVER_ABI_VERSION) {
            return fail("launch image was written for a different driver ABI");
        }
        const std::uint64_t blob_len = header[2];
        const std::uint64_t reloc_count = header[3];
        const std::uint64_t root = header[4];
        if (raw.size() != kHeader + blob_len + reloc_count * sizeof(std::uint64_t)) {
            return fail("launch image length disagrees with its header");
        }
        if (root + sizeof(PieLaunchPackage) > blob_len) {
            return fail("launch image root is outside the blob");
        }

        // `std::uint64_t` storage so the blob is 8-aligned, which every record
        // in it requires.
        blob_.assign((blob_len + 7) / 8, 0);
        std::memcpy(blob_.data(), raw.data() + kHeader, blob_len);
        auto* const base = reinterpret_cast<std::uint8_t*>(blob_.data());

        const std::uint8_t* relocs = raw.data() + kHeader + blob_len;
        for (std::uint64_t i = 0; i < reloc_count; ++i) {
            std::uint64_t at = 0;
            std::memcpy(&at, relocs + i * sizeof(std::uint64_t), sizeof(at));
            if (at + sizeof(std::uintptr_t) > blob_len) {
                return fail("launch image relocation is outside the blob");
            }
            std::uintptr_t offset = 0;
            std::memcpy(&offset, base + at, sizeof(offset));
            if (offset >= blob_len) {
                return fail("launch image relocation targets outside the blob");
            }
            const auto absolute =
                reinterpret_cast<std::uintptr_t>(base) + offset;
            std::memcpy(base + at, &absolute, sizeof(absolute));
        }
        std::memcpy(&package_, base + root, sizeof(package_));
        // The same validator the driver runs on a wire package. A relocation
        // the encoder failed to record shows up here as a null pointer with a
        // nonzero length, instead of as a fault deep inside `adopt`.
        if (validate::launch_package(package_) != PIE_STATUS_OK) {
            package_ = PieLaunchPackage{};
            return fail("launch image does not satisfy the launch-package ABI");
        }
        return true;
    }

    const PieLaunchPackage& package() const { return package_; }

  private:
    std::vector<std::uint64_t> blob_;
    PieLaunchPackage package_{};
};

}  // namespace pie::driver::testing
