#include "custom_all_reduce.hpp"

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "distributed.hpp"

#include <flashinfer/comm/trtllm_allreduce.cuh>
#include <flashinfer/comm/trtllm_allreduce_fusion.cuh>
#include <flashinfer/comm/vllm_custom_all_reduce.cuh>

namespace pie_cuda_driver {

namespace {

// Helper to enable peer access from `device` to every other GPU in the
// group. Idempotent — subsequent calls just return cudaErrorPeerAccessAlreadyEnabled
// which we swallow.
void enable_peer_access(int self_device, int world_size) {
    for (int peer = 0; peer < world_size; ++peer) {
        if (peer == self_device) continue;
        const auto err = cudaDeviceEnablePeerAccess(peer, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            // Fail open — the constructor will throw at the IPC step if
            // peer access truly isn't available.
            std::cerr << "[custom_all_reduce] cudaDeviceEnablePeerAccess "
                      << self_device << "→" << peer
                      << " failed: " << cudaGetErrorString(err) << "\n";
        }
        // Reset the sticky error.
        (void)cudaGetLastError();
    }
}

// Look up the base allocation address for `ptr`. The vllm kernel needs
// the *base* pointer for the IPC handle exchange — sub-allocation
// pointers won't round-trip across processes correctly.
void* get_base_ptr(const void* ptr) {
    void* base = nullptr;
    const auto rc = cuPointerGetAttribute(&base,
        CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
        reinterpret_cast<CUdeviceptr>(ptr));
    if (rc != CUDA_SUCCESS) {
        throw std::runtime_error(
            "custom_all_reduce: cuPointerGetAttribute(RANGE_START_ADDR) failed");
    }
    return base;
}

void all_gather_host_bytes(NcclComm& comm,
                           const void* send,
                           void* recv,
                           std::size_t bytes)
{
    if (bytes == 0) return;
    void* d_send = nullptr;
    void* d_recv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_send, bytes));
    CUDA_CHECK(cudaMalloc(&d_recv, bytes * comm.world_size()));
    CUDA_CHECK(cudaMemcpy(d_send, send, bytes, cudaMemcpyHostToDevice));
    comm.all_gather_bytes(d_send, d_recv, bytes, /*stream=*/nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(recv, d_recv, bytes * comm.world_size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
}

std::size_t align_up(std::size_t n, std::size_t a) {
    return ((n + a - 1) / a) * a;
}

}  // namespace

// Default ctor defined here (not = default in header) so the compiler
// doesn't have to see the full vllm::CustomAllreduce type to handle the
// implicit member initialisation of std::unique_ptr<vllm::CustomAllreduce>
// at every use site of the wrapper.
CustomAllReduce::CustomAllReduce() = default;

CustomAllReduce::CustomAllReduce(NcclComm& comm,
                                 bool same_process,
                                 std::size_t max_bytes,
                                 std::size_t rank_data_bytes,
                                 int fusion_max_tokens,
                                 int fusion_hidden) {
    rank_       = comm.rank();
    world_size_ = comm.world_size();
    same_process_ = same_process;
    max_bytes_ = max_bytes;
    if (world_size_ < 2 || world_size_ > 8 || (world_size_ % 2) != 0) {
        throw std::runtime_error(
            "custom_all_reduce: vllm kernel supports world_size ∈ {2,4,6,8}; got " +
            std::to_string(world_size_));
    }

    // Best-effort peer access; cudaIpcOpenMemHandle below will surface a
    // hard error if NVLink P2P isn't actually available between this
    // GPU and any peer.
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    enable_peer_access(dev, world_size_);

    // Allocate the local Signal struct plus the temporary region needed by
    // flashinfer's 2-stage algorithm. TP=2 uses the 1-stage kernel, but
    // matching vLLM's layout keeps this wrapper valid for fully-connected
    // larger groups too.
    const std::size_t signal_bytes = sizeof(vllm::Signal) + max_bytes_;
    CUDA_CHECK(cudaMalloc(&signal_self_, signal_bytes));
    CUDA_CHECK(cudaMemset(signal_self_, 0, signal_bytes));

    signal_peers_.resize(world_size_);
    if (same_process_) {
        std::uint64_t self_ptr = reinterpret_cast<std::uint64_t>(signal_self_);
        std::vector<std::uint64_t> all_ptrs(world_size_);
        all_gather_host_bytes(comm, &self_ptr, all_ptrs.data(), sizeof(self_ptr));
        for (int r = 0; r < world_size_; ++r) {
            signal_peers_[r] =
                reinterpret_cast<vllm::Signal*>(all_ptrs[static_cast<std::size_t>(r)]);
        }
    } else {
        // Exchange IPC handles for the Signal across ranks. ncclAllGather
        // does the byte transfer; cudaIpcOpenMemHandle on each peer's
        // handle gives us a pointer mapped into our address space.
        cudaIpcMemHandle_t self_signal_handle{};
        CUDA_CHECK(cudaIpcGetMemHandle(&self_signal_handle, signal_self_));
        std::vector<cudaIpcMemHandle_t> all_signal_handles(world_size_);
        all_gather_host_bytes(comm, &self_signal_handle, all_signal_handles.data(),
                              sizeof(cudaIpcMemHandle_t));

        for (int r = 0; r < world_size_; ++r) {
            if (r == rank_) {
                signal_peers_[r] = static_cast<vllm::Signal*>(signal_self_);
                continue;
            }
            void* peer_signal = nullptr;
            const auto rc = cudaIpcOpenMemHandle(
                &peer_signal, all_signal_handles[static_cast<std::size_t>(r)],
                cudaIpcMemLazyEnablePeerAccess);
            if (rc != cudaSuccess) {
                throw std::runtime_error(
                    std::string("custom_all_reduce: cudaIpcOpenMemHandle(rank=") +
                    std::to_string(r) + ") failed: " + cudaGetErrorString(rc));
            }
            signal_peers_[r] = static_cast<vllm::Signal*>(peer_signal);
        }
    }

    // RankData scratch — one 64-byte slot per registered buffer or graph
    // all-reduce site. vLLM uses 8 MiB, enough for ~131k graph addresses.
    rank_data_bytes_ = std::max<std::size_t>(rank_data_bytes,
                                             sizeof(vllm::RankData));
    CUDA_CHECK(cudaMalloc(&rank_data_, rank_data_bytes_));

    impl_ = std::make_unique<vllm::CustomAllreduce>(
        signal_peers_.data(), rank_data_, rank_data_bytes_,
        rank_, world_size_, full_nvlink_);

    if (world_size_ == 2 && fusion_max_tokens > 0 && fusion_hidden > 0) {
        fusion_max_tokens_ = fusion_max_tokens;
        fusion_hidden_ = fusion_hidden;

        constexpr std::size_t kAlign = 1ull << 21;
        constexpr std::size_t kBarrierFlagCount = 256;
        const std::size_t elem_bytes = sizeof(__nv_bfloat16);
        const std::size_t buffer_bytes = align_up(
            static_cast<std::size_t>(world_size_) *
                static_cast<std::size_t>(fusion_max_tokens_) *
                static_cast<std::size_t>(fusion_hidden_) * elem_bytes,
            kAlign);
        const std::size_t flag_bytes = align_up(
            static_cast<std::size_t>(world_size_) * kBarrierFlagCount *
                sizeof(std::int32_t),
            kAlign);
        fusion_lamport_comm_bytes_ = std::min<std::size_t>(
            static_cast<std::size_t>(world_size_) *
                static_cast<std::size_t>(fusion_max_tokens_) *
                static_cast<std::size_t>(fusion_hidden_) * elem_bytes,
            2145386496ull);
        const std::size_t lamport_bytes =
            align_up(fusion_lamport_comm_bytes_ * 3, kAlign);

        fusion_buffers_.resize(3, nullptr);
        CUDA_CHECK(cudaMalloc(&fusion_buffers_[0], buffer_bytes));
        CUDA_CHECK(cudaMalloc(&fusion_buffers_[1], flag_bytes));
        CUDA_CHECK(cudaMalloc(&fusion_buffers_[2], lamport_bytes));

        // Lamport slots use negative zero as the empty sentinel.
        CUDA_CHECK((flashinfer::trtllm_allreduce::lamportInitialize<
            __nv_bfloat16>(
            fusion_buffers_[2], lamport_bytes / elem_bytes, nullptr)));

        CUDA_CHECK(cudaMalloc(&fusion_flag_dev_, 5 * sizeof(std::uint32_t)));
        const std::uint32_t flags[5] = {
            0u, 0u, 0u,
            static_cast<std::uint32_t>(fusion_lamport_comm_bytes_),
            0u,
        };
        CUDA_CHECK(cudaMemcpy(fusion_flag_dev_, flags, sizeof(flags),
                              cudaMemcpyHostToDevice));

        std::vector<void*> workspace;
        workspace.reserve(static_cast<std::size_t>(3 * world_size_ + 1));
        for (void* local_buf : fusion_buffers_) {
            if (same_process_) {
                std::uint64_t self_ptr =
                    reinterpret_cast<std::uint64_t>(local_buf);
                std::vector<std::uint64_t> all_ptrs(world_size_);
                all_gather_host_bytes(
                    comm, &self_ptr, all_ptrs.data(), sizeof(self_ptr));
                for (int r = 0; r < world_size_; ++r) {
                    workspace.push_back(reinterpret_cast<void*>(
                        all_ptrs[static_cast<std::size_t>(r)]));
                }
            } else {
                cudaIpcMemHandle_t self_handle{};
                CUDA_CHECK(cudaIpcGetMemHandle(&self_handle, local_buf));
                std::vector<cudaIpcMemHandle_t> all_handles(world_size_);
                all_gather_host_bytes(comm, &self_handle, all_handles.data(),
                                      sizeof(cudaIpcMemHandle_t));
                for (int r = 0; r < world_size_; ++r) {
                    if (r == rank_) {
                        workspace.push_back(local_buf);
                    } else {
                        void* peer = nullptr;
                        const auto rc = cudaIpcOpenMemHandle(
                            &peer, all_handles[static_cast<std::size_t>(r)],
                            cudaIpcMemLazyEnablePeerAccess);
                        if (rc != cudaSuccess) {
                            throw std::runtime_error(
                                std::string("custom_all_reduce: fusion "
                                            "cudaIpcOpenMemHandle(rank=") +
                                std::to_string(r) + ") failed: " +
                                cudaGetErrorString(rc));
                        }
                        workspace.push_back(peer);
                    }
                }
            }
        }
        workspace.push_back(fusion_flag_dev_);
        CUDA_CHECK(cudaMalloc(&fusion_workspace_dev_,
                              workspace.size() * sizeof(void*)));
        CUDA_CHECK(cudaMemcpy(fusion_workspace_dev_, workspace.data(),
                              workspace.size() * sizeof(void*),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::cerr << "[custom_all_reduce] initialised (world=" << world_size_
              << ", rank=" << rank_
              << ", mode=" << (same_process_ ? "same-process" : "ipc")
              << ", NVLink=" << (full_nvlink_ ? "yes" : "no")
              << ")\n";
}

CustomAllReduce::~CustomAllReduce() {
    // Close peer signal handles before freeing local state. The vllm
    // CustomAllreduce destructor takes care of buffers it opened via
    // `register_buffer`'s IPC path. Iterate by signal_peers_ size, not
    // world_size_, so the moved-from-empty-but-world_size_=2 case stays
    // correct (move ctor empties the vector but doesn't reset world_size_).
    impl_.reset();
    for (std::size_t r = 0; !same_process_ && r < signal_peers_.size(); ++r) {
        if (static_cast<int>(r) == rank_) continue;
        if (signal_peers_[r] != nullptr) {
            cudaIpcCloseMemHandle(signal_peers_[r]);
        }
    }
    if (signal_self_) cudaFree(signal_self_);
    if (rank_data_)   cudaFree(rank_data_);
    for (void* buf : fusion_buffers_) {
        if (buf) cudaFree(buf);
    }
    if (fusion_workspace_dev_) cudaFree(fusion_workspace_dev_);
    if (fusion_flag_dev_) cudaFree(fusion_flag_dev_);
}

CustomAllReduce::CustomAllReduce(CustomAllReduce&& o) noexcept
    : rank_(o.rank_), world_size_(o.world_size_), full_nvlink_(o.full_nvlink_),
      same_process_(o.same_process_), max_bytes_(o.max_bytes_),
      signal_self_(o.signal_self_), signal_peers_(std::move(o.signal_peers_)),
      rank_data_(o.rank_data_), rank_data_bytes_(o.rank_data_bytes_),
      impl_(std::move(o.impl_)),
      registered_bases_(std::move(o.registered_bases_)),
      fusion_buffers_(std::move(o.fusion_buffers_)),
      fusion_workspace_dev_(o.fusion_workspace_dev_),
      fusion_flag_dev_(o.fusion_flag_dev_),
      fusion_max_tokens_(o.fusion_max_tokens_),
      fusion_hidden_(o.fusion_hidden_),
      fusion_lamport_comm_bytes_(o.fusion_lamport_comm_bytes_)
{
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

bool CustomAllReduce::can_handle(const void* input, std::size_t bytes,
                                 cudaStream_t stream) const noexcept {
    if (!impl_ || input == nullptr) return false;
    if (bytes == 0 || bytes > max_bytes_ || (bytes % 16) != 0) return false;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) return false;
    if (status == cudaStreamCaptureStatusActive) return true;
    void* base = nullptr;
    try {
        base = get_base_ptr(input);
    } catch (...) {
        return false;
    }
    if (registered_bases_.find(base) == registered_bases_.end()) return false;
    // The vllm one-shot kernel handles arbitrary sizes correctly; the
    // question is just where NCCL takes over on raw bandwidth. On
    // NVLink/NVSwitch the crossover is around a few MB for 2 ranks,
    // less for higher TP. Be generous on small TP (2 ranks); tighter
    // thresholds for larger TP where the kernel becomes the bottleneck.
    if (world_size_ <= 2)         return bytes < max_bytes_;
    if (world_size_ <= 4)         return bytes < 1 * 1024 * 1024;
    return bytes < 256 * 1024;
}

bool CustomAllReduce::can_fuse_residual_rmsnorm(
    int tokens, int hidden, cudaStream_t /*stream*/) const noexcept {
    if (std::getenv("PIE_DISABLE_AR_FUSION") != nullptr) return false;
    if (fusion_workspace_dev_ == nullptr) return false;
    if (tokens <= 0 || tokens > fusion_max_tokens_) return false;
    if (hidden != fusion_hidden_) return false;
    if (world_size_ != 2) return false;
    return (hidden % 8) == 0;
}

void CustomAllReduce::register_buffer(NcclComm& comm, void* buf,
                                      std::size_t /*buf_bytes*/)
{
    if (!impl_) return;
    void* self_base = get_base_ptr(buf);
    if (registered_bases_.find(self_base) != registered_bases_.end()) return;

    std::vector<void*> peer_bases(world_size_);
    if (same_process_) {
        std::uint64_t self_ptr = reinterpret_cast<std::uint64_t>(self_base);
        std::vector<std::uint64_t> all_ptrs(world_size_);
        all_gather_host_bytes(comm, &self_ptr, all_ptrs.data(), sizeof(self_ptr));
        for (int r = 0; r < world_size_; ++r) {
            peer_bases[static_cast<std::size_t>(r)] =
                reinterpret_cast<void*>(all_ptrs[static_cast<std::size_t>(r)]);
        }
    } else {
        cudaIpcMemHandle_t self_handle{};
        CUDA_CHECK(cudaIpcGetMemHandle(&self_handle, self_base));
        std::vector<cudaIpcMemHandle_t> all_handles(world_size_);
        all_gather_host_bytes(comm, &self_handle, all_handles.data(),
                              sizeof(cudaIpcMemHandle_t));

        // Open peer handles, build the per-rank pointer array, and register.
        for (int r = 0; r < world_size_; ++r) {
            if (r == rank_) {
                peer_bases[r] = self_base;
            } else {
                void* peer = nullptr;
                const auto rc = cudaIpcOpenMemHandle(
                    &peer, all_handles[static_cast<std::size_t>(r)],
                    cudaIpcMemLazyEnablePeerAccess);
                if (rc != cudaSuccess) {
                    throw std::runtime_error(
                        std::string("custom_all_reduce: register_buffer "
                                    "cudaIpcOpenMemHandle(rank=") +
                        std::to_string(r) + ") failed: " +
                        cudaGetErrorString(rc));
                }
                peer_bases[r] = peer;
            }
        }
    }
    impl_->register_buffer(peer_bases.data());
    registered_bases_[self_base] = self_base;
}

void CustomAllReduce::register_graph_buffers(NcclComm& comm)
{
    if (!impl_) return;
    const std::size_t num_buffers = impl_->graph_unreg_buffers_.size();
    if (num_buffers == 0) return;

    if (same_process_) {
        std::vector<std::uint64_t> local(num_buffers);
        for (std::size_t i = 0; i < num_buffers; ++i) {
            local[i] = reinterpret_cast<std::uint64_t>(
                impl_->graph_unreg_buffers_[i]);
        }
        std::vector<std::uint64_t> gathered(num_buffers * world_size_);
        all_gather_host_bytes(comm, local.data(), gathered.data(),
                              num_buffers * sizeof(std::uint64_t));

        impl_->check_rank_data_capacity(num_buffers);
        std::vector<vllm::RankData> rank_data(num_buffers);
        for (std::size_t i = 0; i < num_buffers; ++i) {
            for (int r = 0; r < world_size_; ++r) {
                const std::size_t idx =
                    static_cast<std::size_t>(r) * num_buffers + i;
                rank_data[i].ptrs[r] =
                    reinterpret_cast<void*>(gathered[idx]);
            }
        }
        CUDA_CHECK(cudaMemcpy(impl_->d_rank_data_base_, rank_data.data(),
                              sizeof(vllm::RankData) * num_buffers,
                              cudaMemcpyHostToDevice));
        impl_->d_rank_data_base_ += num_buffers;
        impl_->graph_unreg_buffers_.clear();
        return;
    }

    auto [self_handles, self_offsets] = impl_->get_graph_buffer_ipc_meta();
    const std::size_t handle_bytes = self_handles.size();
    std::vector<char> all_handles(handle_bytes * world_size_);
    all_gather_host_bytes(comm, self_handles.data(), all_handles.data(),
                          handle_bytes);

    std::vector<std::int64_t> all_offsets(num_buffers * world_size_);
    all_gather_host_bytes(comm, self_offsets.data(), all_offsets.data(),
                          num_buffers * sizeof(std::int64_t));

    std::vector<std::string> handles(world_size_);
    std::vector<std::vector<std::int64_t>> offsets(world_size_);
    for (int r = 0; r < world_size_; ++r) {
        handles[r].assign(all_handles.data() +
                              static_cast<std::size_t>(r) * handle_bytes,
                          handle_bytes);
        offsets[r].assign(
            all_offsets.begin() + static_cast<std::ptrdiff_t>(
                static_cast<std::size_t>(r) * num_buffers),
            all_offsets.begin() + static_cast<std::ptrdiff_t>(
                static_cast<std::size_t>(r + 1) * num_buffers));
    }
    impl_->register_graph_buffers(handles, offsets);
}

void CustomAllReduce::all_reduce_bf16(const void* input, void* output,
                                      std::size_t count, cudaStream_t stream)
{
    if (!impl_) {
        throw std::runtime_error("custom_all_reduce: not initialised");
    }
    // The vllm kernel registers buffers by base address; passing a
    // sub-pointer works as long as the *base* was registered. We pass
    // the input pointer directly (kernel does its own offset math on
    // the registered RankData).
    impl_->allreduce<__nv_bfloat16>(
        stream,
        const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(input)),
        static_cast<__nv_bfloat16*>(output),
        static_cast<int>(count),
        /*block_limit=*/36, /*threads=*/512);
}

void CustomAllReduce::all_reduce_residual_rmsnorm_bf16(
    const void* input,
    void* residual_inout,
    const void* rms_gamma,
    void* norm_out,
    int tokens,
    int hidden,
    float eps,
    cudaStream_t stream)
{
    if (!can_fuse_residual_rmsnorm(tokens, hidden, stream)) {
        throw std::runtime_error(
            "custom_all_reduce: fused residual RMSNorm is unavailable");
    }
    using namespace flashinfer::trtllm_allreduce_fusion;
    AllReduceFusionParams<__nv_bfloat16> params{};
    params.nranks = world_size_;
    params.rank = rank_;
    params.size = tokens * hidden;
    params.hidden_dim = hidden;
    params.workspace = reinterpret_cast<void**>(fusion_workspace_dev_);
    params.allreduce_in = const_cast<void*>(input);
    params.allreduce_out = nullptr;
    params.residual_in = residual_inout;
    params.residual_out = residual_inout;
    params.norm_out = norm_out;
    params.quant_out = nullptr;
    params.scale_out = nullptr;
    params.rms_gamma = const_cast<void*>(rms_gamma);
    params.rms_eps = eps;
    params.scale_factor = nullptr;
    params.use_oneshot = true;
    params.layout = flashinfer::QuantizationSFLayout::SWIZZLED_128x4;
    params.stream = stream;
    params.pattern = AllReduceFusionPattern::kARResidualRMSNorm;
    params.trigger_completion_at_end = false;
    CUDA_CHECK((allreduce_fusion_op<__nv_bfloat16>(
        params, /*launch_with_pdl=*/false, /*use_fp32_acc=*/true)));
}

}  // namespace pie_cuda_driver
