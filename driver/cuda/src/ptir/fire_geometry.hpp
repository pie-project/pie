#pragma once

// CUDA-FREE geometry POD for the PTIR pre-forward descriptor resolver (W1.1).
// Split out of descriptor_resolve.hpp so the host-side dispatch façade
// (ptir_dispatch.hpp, included by CUDA-free .cpp TUs) can name it without
// pulling device headers. The resolver (descriptor_resolve.hpp, nvcc TU) fills
// it; the executor feeds it into the standard batch assembly.

#include <cstdint>
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

}  // namespace pie_cuda_driver::ptir
