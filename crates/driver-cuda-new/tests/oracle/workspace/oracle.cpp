// Drives the real `Workspace::allocate_full` over a grid of model shapes and
// reports, for each, the exact sequence of tensor allocations it makes —
// alongside what `workspace_bytes` claims the same shape costs.
//
// Both, because they are supposed to be the same number and the header says
// so: *"this function is the planner's arena figure (`memory_planner.cpp`) and
// every byte missing from it is a byte handed to the KV pool instead."* One
// function allocates and the other budgets, they are written out separately by
// hand, and nothing in the shipping build compares them. A transcript that
// carries only one of the two cannot see them disagree.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "model/workspace.hpp"

namespace pie_cuda_driver {
extern std::vector<std::string> g_alloc_log;
void reset_alloc_log();
}  // namespace pie_cuda_driver

using namespace pie_cuda_driver;

namespace {

constexpr char kSep = '\x1f';

struct Case {
    const char* label;
    int hidden;
    int intermediate;
    int vocab;
    int head_dim;
    int head_dim_kernel;
    int q_heads;
    int kv_heads;
    int max_tokens;
    int output_rows;
    int mtp_draft_rows;
};

// The grid. Every row is a shape the shipping driver actually meets, and the
// last three exist to reach the branches the first four never do.
const Case kCases[] = {
    // label                H     I       V        hd   hdk  qh  kvh   N     O    D
    {"qwen3_0_6b",       1024,  3072,  151936,  128, 128, 16,  8,  2048,  64,   0},
    {"llama3_8b",        4096, 14336,  128256,  128, 128, 32,  8,  4096, 128,   0},
    {"olmo2_1b",         2048,  8192,  100352,  128, 128, 16, 16,  1024,  32,   0},
    {"qwen3_32b",        5120, 25600,  151936,  128, 128, 64,  8,  8192, 256,   0},
    // head_dim != head_dim_kernel: the padded q/k/v/attn_out branch. Phi-3
    // ships 96 and flashinfer's TC kernel rounds it to 128.
    {"phi3_mini",        3072,  8192,   32064,   96, 128, 32, 32,  4096, 128,   0},
    // MTP draft rows: `logits` grows past `max_tokens` and three more
    // tensors follow its row count.
    {"qwen3_6_mtp",      2048,  8192,  248320,  128, 128, 16,  8,  8192,  64, 192},
    // Both at once, and an output row count above the token count.
    {"padded_with_mtp",  3072,  8192,   32064,   96, 128, 32, 32,  2048, 256,  32},
};

void run(const Case& c) {
    HfConfig cfg;
    cfg.hidden_size = c.hidden;
    cfg.intermediate_size = c.intermediate;
    cfg.vocab_size = c.vocab;
    cfg.head_dim = c.head_dim;
    cfg.head_dim_kernel = c.head_dim_kernel;
    cfg.num_attention_heads = c.q_heads;
    cfg.num_key_value_heads = c.kv_heads;

    const int max_Hq = c.q_heads * c.head_dim;
    const int max_Hk = c.kv_heads * c.head_dim;

    reset_alloc_log();
    model::Workspace ws = model::Workspace::allocate_full(
        cfg, c.max_tokens, c.intermediate, max_Hq, max_Hk, c.output_rows,
        c.mtp_draft_rows);

    // What was actually allocated, in order.
    std::uint64_t allocated = 0;
    for (const std::string& row : g_alloc_log) {
        std::cout << c.label << kSep << "alloc" << kSep << row << "\n";
        // `row` is `dtype[d0,d1]=nbytes`; see tensor_recorder.cpp.
        allocated += std::stoull(row.substr(row.rfind('=') + 1));
    }

    // The two scalars `allocate_full` sets that no allocation records.
    std::cout << c.label << kSep << "mtp_draft_row_base" << kSep
              << ws.mtp_draft_row_base << "\n";
    std::cout << c.label << kSep << "mtp_draft_row_capacity" << kSep
              << ws.mtp_draft_row_capacity << "\n";

    // And what the planner is told the same shape costs.
    const std::uint64_t budgeted = model::workspace_bytes(
        cfg, c.max_tokens, c.output_rows, c.intermediate, max_Hq, max_Hk,
        c.mtp_draft_rows);

    std::cout << c.label << kSep << "allocated_bytes" << kSep << allocated << "\n";
    std::cout << c.label << kSep << "budgeted_bytes" << kSep << budgeted << "\n";
    std::cout << c.label << kSep << "shortfall_bytes" << kSep
              << (allocated - budgeted) << "\n";
}

}  // namespace

int main() {
    for (const Case& c : kCases) run(c);
    return 0;
}
