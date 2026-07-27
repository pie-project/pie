#pragma once

// Load the host-emitted kernel table a driver test needs to register a program.
//
// The driver stopped generating kernels (ptir-refactor.md phase 2'), so
// registration now requires the table the engine would have handed it. A C++
// test cannot run the Rust emitter, so `pie-codegen` writes the table beside
// the traces as a fixture -- regenerate with
//
//   cargo test -p pie-compiler-tests --test cuda_golden emit_driver_test_kernel
//
// Format, one record per kernel:
//   `kernel <kind> <stage> <region> <entry-or-dash> <source-byte-count>\n`
//   then exactly that many bytes of source, then a newline.
//
// Byte counts rather than delimiters because the sources are CUDA and contain
// every delimiter one might pick.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "pie_driver_abi.h"

namespace pie_cuda_driver::tests {

class HostKernelFixture {
  public:
    static const std::uint8_t* as_bytes(const std::string& text) {
        return reinterpret_cast<const std::uint8_t*>(text.data());
    }

    // Returns false when the fixture is missing or malformed; `err` says which.
    bool load(const std::string& path, std::string* err) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            if (err) *err = "cannot open host kernel fixture: " + path;
            return false;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream header(line);
            std::string tag, entry;
            std::uint32_t kind = 0, stage = 0, region = 0;
            std::size_t length = 0;
            if (!(header >> tag >> kind >> stage >> region >> entry >> length) ||
                tag != "kernel") {
                if (err) *err = "malformed record in " + path + ": " + line;
                return false;
            }
            std::string source(length, '\0');
            if (length > 0) in.read(source.data(), static_cast<std::streamsize>(length));
            in.get();  // the newline the writer appends after the body
            records_.push_back(
                {kind, stage, region, entry == "-" ? std::string{} : entry,
                 std::move(source)});
        }
        // The ABI slice borrows, so the stable storage has to be built once the
        // vector has stopped reallocating.
        views_.reserve(records_.size());
        for (const Record& record : records_) {
            views_.push_back(PieEmittedKernel{
                .kind = record.kind,
                .stage_index = record.stage,
                .region_index = record.region,
                .reserved0 = 0,
                .entry_name = {as_bytes(record.entry), record.entry.size()},
                .source = {as_bytes(record.source), record.source.size()},
                .error = {nullptr, 0},
            });
        }
        return true;
    }

    PieEmittedKernelSlice slice() const {
        return PieEmittedKernelSlice{views_.data(), views_.size()};
    }

    std::size_t size() const { return views_.size(); }

  private:
    struct Record {
        std::uint32_t kind;
        std::uint32_t stage;
        std::uint32_t region;
        std::string entry;
        std::string source;
    };
    std::vector<Record> records_;
    std::vector<PieEmittedKernel> views_;
};

}  // namespace pie_cuda_driver::tests
