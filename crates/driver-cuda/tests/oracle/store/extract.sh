#!/bin/bash
# Lift the pure logic out of the real sources. Every .inc here is VERBATIM
# source text, so the oracle cannot drift from what it is testing.
set -e
SRC="$1"

# --- mla_cache: page_buffers ------------------------------------------------
awk '/^std::vector<MlaCache::PageBuffer> MlaCache::page_buffers/{f=1}
     f{print; if($0=="}") exit}' "$SRC/store/mla_cache.cpp" > mla.inc

# --- dsv4: compressor_coff + bytes_per_token --------------------------------
awk '/^int compressor_coff/{print; exit}' "$SRC/store/dsv4_compress_cache.cpp" > dsv4.inc
awk '/^std::size_t dsv4_compress_bytes_per_token/{f=1}
     f{print; if($0=="}") exit}' "$SRC/store/dsv4_compress_cache.cpp" >> dsv4.inc

# --- recurrent_state_cache: strides (hpp) + offsets (cpp) -------------------
awk '/^    std::size_t conv_slot_stride_bytes/{f=1}
     f{print; if($0=="    }"){n++; if(n==3) exit}}' \
    "$SRC/store/recurrent_state_cache.hpp" > rec_strides.inc
awk '/^void\* RecurrentStateCache::conv_state/{f=1}
     f{print; if($0=="}") exit}' "$SRC/store/recurrent_state_cache.cpp" > rec_addr.inc
awk '/^void\* RecurrentStateCache::recurrent_state_raw/{f=1}
     f{print; if($0=="}") exit}' "$SRC/store/recurrent_state_cache.cpp" >> rec_addr.inc
awk '/^void\* RecurrentStateCache::mtp_pending_hidden/{f=1}
     f{print; if($0=="}") exit}' "$SRC/store/recurrent_state_cache.cpp" >> rec_addr.inc

# --- swap_pool: check_pairs, page_addr, submit_batch, four copy routines -----
awk '/^void check_pairs/{f=1} f{print; if($0=="}") exit}' "$SRC/store/swap_pool.cpp" > swap_helpers.inc
awk '/^inline void\* page_addr/{f=1} f{print; if($0=="}") exit}' "$SRC/store/swap_pool.cpp" >> swap_helpers.inc
awk '/^inline void submit_batch/{f=1} f{print; if($0=="}") exit}' "$SRC/store/swap_pool.cpp" >> swap_helpers.inc
for fn in copy_d2h_async copy_h2d_async copy_d2d_async copy_h2h_async; do
  awk -v fn="$fn" '$0 ~ ("^void SwapPool::" fn "\\(") {f=1} f{print; if($0=="}") exit}' \
      "$SRC/store/swap_pool.cpp"
done > swap_copy.inc

# --- planner_profile_cache: key_to_json, field_eq x2, key_matches -----------
awk '/^nlohmann::json key_to_json/{f=1} f{print; if($0=="}") exit}' \
    "$SRC/store/planner_profile_cache.cpp" > profile.inc
awk '/^bool field_eq\(const nlohmann::json& key, const char\* name,$/{f=1}
     f{print; if($0=="}"){n++; if(n==1) exit}}' \
    "$SRC/store/planner_profile_cache.cpp" >> profile.inc
awk '/^bool field_eq\(const nlohmann::json& key, const char\* name, int expected\) \{/{f=1}
     f{print; if($0=="}") exit}' "$SRC/store/planner_profile_cache.cpp" >> profile.inc
awk '/^bool key_matches/{f=1} f{print; if($0=="}") exit}' \
    "$SRC/store/planner_profile_cache.cpp" >> profile.inc

wc -l *.inc
