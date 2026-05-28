#include "custom_all_reduce.hpp"

#include <stdexcept>
#include <utility>

#include "distributed.hpp"

namespace vllm {
class CustomAllreduce {};
struct Signal {};
}  // namespace vllm

namespace pie_cuda_driver {

CustomAllReduce::CustomAllReduce() = default;

CustomAllReduce::CustomAllReduce(NcclComm& comm,
                                 bool /*same_process*/,
                                 std::size_t /*max_bytes*/,
                                 std::size_t /*rank_data_bytes*/,
                                 int /*fusion_max_tokens*/,
                                 int /*fusion_hidden*/) {
    rank_ = comm.rank();
    world_size_ = comm.world_size();
}

CustomAllReduce::~CustomAllReduce() = default;

CustomAllReduce::CustomAllReduce(CustomAllReduce&& o) noexcept
    : rank_(o.rank_),
      world_size_(o.world_size_),
      fully_connected_(o.fully_connected_),
      same_process_(o.same_process_),
      max_bytes_(o.max_bytes_),
      signal_self_(o.signal_self_),
      signal_peers_(std::move(o.signal_peers_)),
      rank_data_(o.rank_data_),
      rank_data_bytes_(o.rank_data_bytes_),
      impl_(std::move(o.impl_)),
      registered_bases_(std::move(o.registered_bases_)),
      fusion_buffers_(std::move(o.fusion_buffers_)),
      fusion_workspace_dev_(o.fusion_workspace_dev_),
      fusion_flag_dev_(o.fusion_flag_dev_),
      fusion_max_tokens_(o.fusion_max_tokens_),
      fusion_hidden_(o.fusion_hidden_),
      fusion_lamport_comm_bytes_(o.fusion_lamport_comm_bytes_) {
    o.signal_self_ = nullptr;
    o.rank_data_ = nullptr;
    o.fusion_workspace_dev_ = nullptr;
    o.fusion_flag_dev_ = nullptr;
    o.fusion_max_tokens_ = 0;
    o.fusion_hidden_ = 0;
    o.fusion_lamport_comm_bytes_ = 0;
}

CustomAllReduce& CustomAllReduce::operator=(CustomAllReduce&& o) noexcept {
    if (this != &o) {
        this->~CustomAllReduce();
        new (this) CustomAllReduce(std::move(o));
    }
    return *this;
}

void CustomAllReduce::register_buffer(NcclComm&, void*, std::size_t) {}

void CustomAllReduce::register_graph_buffers(NcclComm&) {}

bool CustomAllReduce::can_handle(const void*, std::size_t, cudaStream_t) const noexcept {
    return false;
}

bool CustomAllReduce::can_fuse_residual_rmsnorm(
    int, int, cudaStream_t) const noexcept {
    return false;
}

void CustomAllReduce::all_reduce_bf16(
    const void*, void*, std::size_t, cudaStream_t) {
    throw std::runtime_error(
        "custom_all_reduce: unavailable for this CUDA architecture");
}

void CustomAllReduce::all_reduce_residual_rmsnorm_bf16(
    const void*,
    void*,
    const void*,
    void*,
    int,
    int,
    float,
    cudaStream_t) {
    throw std::runtime_error(
        "custom_all_reduce: fused residual RMSNorm is unavailable");
}

}  // namespace pie_cuda_driver
