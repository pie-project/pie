// The Qwen3.5 linear-attention workspace oracle — gate-linear-attn-ws.
//
// Compiles the REAL `qwen3_5_forward.cpp` — the whole 2.4k-line TU, over
// the same stub tree the llama_like oracles use, with `--gc-sections`
// discarding the forward body — and drives the one function this gate is
// about: `Qwen3_5LinearAttnWorkspace::allocate`, twenty-seven device
// buffers whose sizes are seven dims multiplied in ways that transpose
// silently (`k_h`/`v_h`, `k_d`/`v_d` — the header's own warning).
//
// The replaced implementation is `allocate_device_memory`, which records
// ordinal and BYTES. After each allocation sweep the oracle prints the
// member → buffer mapping, so a reordering of two same-size members is a
// transcript change even though the byte sequence would not move. The
// two small structs the family adds (`Qwen3_5ForwardCfg`,
// `Qwen3_5PlanState`) get default rows, as the plan-state gate's slice A
// did for the llama-like pair.

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>

#include "model/qwen3_5/qwen3_5_forward.hpp"

using pie_cuda_driver::model::Qwen3_5ForwardCfg;
using pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace;
using pie_cuda_driver::model::Qwen3_5PlanState;

namespace {

constexpr char SEP = '\x1f';
std::string g_case;
std::map<const void*, std::string> g_bufs;
int g_next = 0;

void note(const std::string& body) {
    std::printf("%s%c%s\n", g_case.c_str(), SEP, body.c_str());
}

std::string buf_of(const void* p) {
    if (p == nullptr) return "null";
    auto it = g_bufs.find(p);
    return it == g_bufs.end() ? "unknown" : it->second;
}

}  // namespace

// The plan caches are opaque in the real header so their definition can
// live GPU-side; `Qwen3_5PlanState`'s destructor still needs the deleters.
namespace pie_cuda_driver::kernels::attn {
struct DecodePlanCache {};
struct PrefillPlanCache {};
void DecodePlanCacheDeleter::operator()(DecodePlanCache* p) const noexcept {
    delete p;
}
void PrefillPlanCacheDeleter::operator()(PrefillPlanCache* p) const noexcept {
    delete p;
}
}  // namespace pie_cuda_driver::kernels::attn

namespace pie_cuda_driver {

DeviceMemoryBlock allocate_device_memory(std::size_t bytes, std::size_t) {
    if (bytes == 0) return DeviceMemoryBlock{nullptr, false};
    void* p = std::malloc(bytes);
    const std::string name = "buf" + std::to_string(g_next++);
    g_bufs[p] = name;
    note("alloc " + name + " bytes=" + std::to_string(bytes));
    return DeviceMemoryBlock{p, false};
}

void free_device_memory(DeviceMemoryBlock block) noexcept {
    if (block.ptr != nullptr) {
        g_bufs.erase(block.ptr);
        std::free(block.ptr);
    }
}

}  // namespace pie_cuda_driver

namespace {

void members_row(Qwen3_5LinearAttnWorkspace& ws) {
    note("members mixed_qkv=" + buf_of(ws.mixed_qkv.data()) +
         " mixed_qkvz=" + buf_of(ws.mixed_qkvz.data()) +
         " ba=" + buf_of(ws.ba.data()) +
         " z=" + buf_of(ws.z.data()) +
         " a=" + buf_of(ws.a.data()) +
         " b=" + buf_of(ws.b.data()) +
         " mixed_qkv_post=" + buf_of(ws.mixed_qkv_post.data()) +
         " q_norm=" + buf_of(ws.q_norm.data()) +
         " k_norm=" + buf_of(ws.k_norm.data()) +
         " v_fp32=" + buf_of(ws.v_fp32.data()) +
         " g_log=" + buf_of(ws.g_log.data()) +
         " beta=" + buf_of(ws.beta.data()) +
         " core_out=" + buf_of(ws.core_out.data()) +
         " core_out_bf16=" + buf_of(ws.core_out_bf16.data()) +
         " q_raw=" + buf_of(ws.q_raw.data()) +
         " k_raw=" + buf_of(ws.k_raw.data()) +
         " v_raw=" + buf_of(ws.v_raw.data()) +
         " q_pre=" + buf_of(ws.q_pre.data()) +
         " k_pre=" + buf_of(ws.k_pre.data()) +
         " fa_qg_packed=" + buf_of(ws.fa_qg_packed.data()) +
         " fa_gate=" + buf_of(ws.fa_gate.data()) +
         " qo_ext=" + buf_of(ws.qo_ext.data()) +
         " rs_write_state_mask=" + buf_of(ws.rs_write_state_mask.data()) +
         " qo_split=" + buf_of(ws.qo_split.data()) +
         " split_slot_head=" + buf_of(ws.split_slot_head.data()) +
         " split_slot_tail=" + buf_of(ws.split_slot_tail.data()) +
         " split_mask_head=" + buf_of(ws.split_mask_head.data()) +
         " max_tokens=" + std::to_string(ws.max_tokens));
}

void run_case(const std::string& name, int max_tokens, int conv_dim, int v_h,
              int k_h, int k_d, int v_d, int hq) {
    g_case = name;
    g_bufs.clear();
    g_next = 0;
    note("case-begin N=" + std::to_string(max_tokens) + " conv=" +
         std::to_string(conv_dim) + " vh=" + std::to_string(v_h) + " kh=" +
         std::to_string(k_h) + " kd=" + std::to_string(k_d) + " vd=" +
         std::to_string(v_d) + " hq=" + std::to_string(hq));
    auto ws = Qwen3_5LinearAttnWorkspace::allocate(max_tokens, conv_dim, v_h,
                                                   k_h, k_d, v_d, hq);
    members_row(ws);
}

}  // namespace

int main() {
    // Pairwise-coprime dims so any transposition moves at least one size.
    run_case("a-asym", 64, 97, 5, 3, 7, 11, 13);
    // The Qwen3.5-4B shape from the header's comments.
    run_case("b-4b", 128, 4096, 32, 16, 128, 128, 32);
    // Ones: the smallest live workspace.
    run_case("c-ones", 1, 1, 1, 1, 1, 1, 1);
    // Zero tokens: every N-scaled buffer is empty (null data), the
    // CSR-shaped ones keep their +1.
    run_case("d-zero-tokens", 0, 96, 4, 2, 8, 16, 8);

    g_case = "e-cfg-defaults";
    {
        const Qwen3_5ForwardCfg c;
        note("force_prefill_path=" +
             std::to_string(c.force_prefill_path ? 1 : 0));
        note("small_prefill_naive_attention_max_tokens=" +
             std::to_string(c.small_prefill_naive_attention_max_tokens));
        note("tp_size=" + std::to_string(c.tp_size));
        note("tp_comm_null=" + std::to_string(c.tp_comm == nullptr ? 1 : 0));
        note("mtp_global_cache_uses_prefix_position=" +
             std::to_string(c.mtp_global_cache_uses_prefix_position ? 1 : 0));
        const Qwen3_5PlanState s;
        note("decode_plan_null=" +
             std::to_string(s.decode_plan == nullptr ? 1 : 0));
        note("prefill_plan_null=" +
             std::to_string(s.prefill_plan == nullptr ? 1 : 0));
        note("use_prefill_plan=" +
             std::to_string(s.use_prefill_plan ? 1 : 0));
    }
    return 0;
}
