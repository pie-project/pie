#pragma once

// CUDA-FREE geometry POD for the PTIR pre-forward descriptor resolver (W1.1).
// Split out of descriptor_resolve.hpp so the host-side dispatch façade
// (ptir_dispatch.hpp, included by CUDA-free .cpp TUs) can name it without
// pulling device headers. The resolver (descriptor_resolve.hpp, nvcc TU) fills
// it; the executor feeds it into the standard batch assembly.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace pie_cuda_driver::ptir {

// The forward geometry a device-geometry PTIR fire contributes, resolved from
// its descriptor-port channels. Mirrors the runtime `ReqGeometry` fields plus
// the explicit-write descriptor (`w_page`/`w_off`) and the dense attention mask.
struct FireGeometry {
    std::vector<std::uint32_t> token_ids;         // embed_tokens
    std::vector<std::uint32_t> position_ids;      // positions (else 0..nnz)
    std::vector<std::uint32_t> qo_indptr;         // embed_indptr (else [0,nnz])
    std::vector<std::uint32_t> kv_page_indices;   // pages (CSR-prefix trimmed)
    std::vector<std::uint32_t> kv_page_indptr;    // page_indptr
    std::vector<std::uint32_t> kv_last_page_lens; // from kv_len
    std::vector<std::uint32_t> sampling_indices;  // readout (else last of each lane)
    std::vector<std::uint32_t> sampling_indptr;
    std::vector<std::uint32_t> w_page;            // w_slot (PHYSICAL page id per lane)
    std::vector<std::uint32_t> w_off;             // w_off (offset-in-page per lane)
    std::vector<std::uint8_t>  mask;              // attn_mask, dense [lanes, stride] bytes
    bool has_kv_family = false;                   // pages/page_indptr present
    bool has_write_desc = false;                  // w_slot/w_off present
    bool has_mask = false;                        // attn_mask present
};

inline bool validate_fire_geometry(const FireGeometry& geometry,
                                   std::uint32_t device_pages,
                                   std::uint32_t page_size,
                                   std::string* error = nullptr) {
    auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (page_size == 0 ||
        geometry.token_ids.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        geometry.position_ids.size() != geometry.token_ids.size() ||
        geometry.qo_indptr.size() < 2 ||
        geometry.qo_indptr.front() != 0 ||
        geometry.qo_indptr.back() != geometry.token_ids.size()) {
        return fail("ptir: invalid resolved token geometry");
    }
    const std::size_t requests = geometry.qo_indptr.size() - 1;
    for (std::size_t request = 0; request < requests; ++request) {
        if (geometry.qo_indptr[request] >
            geometry.qo_indptr[request + 1]) {
            return fail("ptir: resolved qo_indptr is not monotonic");
        }
    }

    if (!geometry.token_ids.empty()) {
        if (!geometry.has_kv_family ||
            geometry.kv_page_indptr.size() != requests + 1 ||
            geometry.kv_page_indptr.front() != 0 ||
            geometry.kv_page_indptr.back() !=
                geometry.kv_page_indices.size() ||
            geometry.kv_last_page_lens.size() != requests) {
            return fail("ptir: invalid resolved KV geometry");
        }
        for (std::uint32_t page : geometry.kv_page_indices) {
            if (page >= device_pages) {
                return fail("ptir: resolved KV page is out of range");
            }
        }
        for (std::size_t request = 0; request < requests; ++request) {
            const std::uint32_t page_begin =
                geometry.kv_page_indptr[request];
            const std::uint32_t page_end =
                geometry.kv_page_indptr[request + 1];
            if (page_begin > page_end) {
                return fail("ptir: resolved kv_page_indptr is not monotonic");
            }
            const std::uint32_t query_rows =
                geometry.qo_indptr[request + 1] -
                geometry.qo_indptr[request];
            const std::uint32_t page_count = page_end - page_begin;
            const std::uint32_t last_len =
                geometry.kv_last_page_lens[request];
            if (page_count == 0 || last_len == 0 ||
                last_len > page_size) {
                return fail("ptir: invalid resolved KV extent");
            }
            const std::uint64_t kv_len =
                static_cast<std::uint64_t>(page_count - 1) * page_size +
                last_len;
            if (kv_len < query_rows) {
                return fail("ptir: resolved KV extent is shorter than query span");
            }
        }
    }

    if (geometry.sampling_indptr.size() != requests + 1 ||
        geometry.sampling_indptr.front() != 0 ||
        geometry.sampling_indptr.back() !=
            geometry.sampling_indices.size()) {
        return fail("ptir: invalid resolved sampling CSR");
    }
    for (std::size_t request = 0; request < requests; ++request) {
        if (geometry.sampling_indptr[request] >
            geometry.sampling_indptr[request + 1]) {
            return fail("ptir: resolved sampling CSR is not monotonic");
        }
        const std::uint32_t query_rows =
            geometry.qo_indptr[request + 1] -
            geometry.qo_indptr[request];
        for (std::uint32_t index = geometry.sampling_indptr[request];
             index < geometry.sampling_indptr[request + 1];
             ++index) {
            if (geometry.sampling_indices[index] >= query_rows) {
                return fail("ptir: resolved sampling row is out of range");
            }
        }
    }

    if (geometry.has_write_desc ||
        !geometry.w_page.empty() ||
        !geometry.w_off.empty()) {
        if (!geometry.has_write_desc ||
            geometry.w_page.size() != geometry.token_ids.size() ||
            geometry.w_off.size() != geometry.token_ids.size()) {
            return fail("ptir: invalid resolved write descriptor shape");
        }
        for (std::size_t i = 0; i < geometry.w_page.size(); ++i) {
            if (geometry.w_page[i] >= device_pages ||
                geometry.w_off[i] >= page_size) {
                return fail("ptir: resolved write descriptor is out of range");
            }
        }
    }

    if (geometry.has_mask && !geometry.mask.empty()) {
        if (geometry.token_ids.empty() ||
            geometry.mask.size() % geometry.token_ids.size() != 0) {
            return fail("ptir: invalid resolved attention mask shape");
        }
        const std::size_t stride =
            geometry.mask.size() / geometry.token_ids.size();
        for (std::size_t request = 0; request < requests; ++request) {
            const std::uint32_t page_count =
                geometry.kv_page_indptr[request + 1] -
                geometry.kv_page_indptr[request];
            const std::uint64_t kv_len =
                static_cast<std::uint64_t>(page_count - 1) * page_size +
                geometry.kv_last_page_lens[request];
            if (stride < kv_len) {
                return fail("ptir: resolved attention mask is shorter than KV");
            }
        }
    }
    return true;
}

}  // namespace pie_cuda_driver::ptir
