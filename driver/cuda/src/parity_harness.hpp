#pragma once

// Numeric-parity harness: runs a one-shot forward pass on a binary file
// of i32 token ids and dumps the last token's logits (bf16, [vocab]) to
// `logits_out`. Invoked only by the standalone `pie_driver_cuda_run`
// entry point — the production serve loop never touches it.

#include <string>

namespace pie_cuda_driver {

struct Config;
class NcclComm;

int run_parity(const Config& cfg,
               const std::string& tokens_in,
               const std::string& logits_out,
               bool paged,
               bool decode_after_prefill = false,
               NcclComm* tp_comm = nullptr);

}  // namespace pie_cuda_driver
