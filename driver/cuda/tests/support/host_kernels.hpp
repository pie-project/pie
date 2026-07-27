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

// The stage identities and the per-region analysis travel the same way and for
// the same reason: registration needs what the host decided, and a C++ test
// cannot run the host planner. Both are written beside the kernels by the same
// `emit_driver_test_kernel_fixtures`.
//
//   `<name>.identities` -- one 16-hex stage identity per line.
//   `<name>.regions`    -- `region <stage> <region> <flags> <argmax-count> [skipped...]`
//                          followed by `argmax <node> <source_value> <intrinsic> <single_row>`.
class HostIdentityFixture {
  public:
    bool load(const std::string& path, std::string* err) {
        std::ifstream in(path);
        if (!in) {
            if (err) *err = "cannot open host identity fixture: " + path;
            return false;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            identities_.push_back(std::stoull(line, nullptr, 16));
        }
        return true;
    }

    PieU64Slice slice() const {
        return PieU64Slice{identities_.data(), identities_.size()};
    }

  private:
    std::vector<std::uint64_t> identities_;
};

class HostRegionFixture {
  public:
    bool load(const std::string& path, std::string* err) {
        std::ifstream in(path);
        if (!in) {
            if (err) *err = "cannot open host region fixture: " + path;
            return false;
        }
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream fields(line);
            std::string kind;
            if (!(fields >> kind)) continue;
            if (kind == "region") {
                PieRegionAnalysis region{};
                std::uint32_t argmax_count = 0;
                fields >> region.stage_index >> region.region_index >>
                    region.flags >> argmax_count;
                std::vector<std::uint32_t> skipped;
                std::uint32_t node = 0;
                while (fields >> node) skipped.push_back(node);
                skipped_.push_back(std::move(skipped));
                argmax_.emplace_back();
                argmax_.back().reserve(argmax_count);
                regions_.push_back(region);
            } else if (kind == "argmax") {
                if (argmax_.empty()) {
                    if (err) {
                        *err = "argmax record before any region in " + path;
                    }
                    return false;
                }
                PieDirectArgmax record{};
                std::uint32_t intrinsic = 0, single_row = 0;
                fields >> record.node >> record.source_value >> intrinsic >>
                    single_row;
                record.intrinsic = static_cast<std::uint16_t>(intrinsic);
                record.requires_single_row =
                    static_cast<std::uint8_t>(single_row);
                argmax_.back().push_back(record);
            }
        }
        // Same rule as the kernel views: the nested slices are taken only once
        // every vector has stopped growing.
        for (std::size_t i = 0; i < regions_.size(); ++i) {
            regions_[i].direct_argmax =
                PieDirectArgmaxSlice{argmax_[i].data(), argmax_[i].size()};
            regions_[i].skipped =
                PieU32Slice{skipped_[i].data(), skipped_[i].size()};
        }
        return true;
    }

    PieRegionAnalysisSlice slice() const {
        return PieRegionAnalysisSlice{regions_.data(), regions_.size()};
    }

  private:
    std::vector<PieRegionAnalysis> regions_;
    std::vector<std::vector<PieDirectArgmax>> argmax_;
    std::vector<std::vector<std::uint32_t>> skipped_;
};

}  // namespace pie_cuda_driver::tests
