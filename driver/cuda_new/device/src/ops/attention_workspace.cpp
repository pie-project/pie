#include "ops/attention_workspace.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace pie_cuda_device::ops {

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("AttentionWorkspace: ") + what +
                                 ": " + cudaGetErrorString(e));
    }
}
}  // namespace

AttentionWorkspace AttentionWorkspace::allocate(
    std::size_t float_workspace_bytes,
    std::size_t int_workspace_bytes)
{
    AttentionWorkspace ws;
    check(cudaMalloc(&ws.float_buf_, float_workspace_bytes), "cudaMalloc(float)");
    check(cudaMalloc(&ws.int_buf_, int_workspace_bytes), "cudaMalloc(int)");
    check(cudaMallocHost(&ws.page_locked_int_, int_workspace_bytes),
          "cudaMallocHost(int)");
    ws.float_bytes_ = float_workspace_bytes;
    ws.int_bytes_ = int_workspace_bytes;
    return ws;
}

AttentionWorkspace::AttentionWorkspace(AttentionWorkspace&& o) noexcept
    : float_buf_(o.float_buf_),
      int_buf_(o.int_buf_),
      page_locked_int_(o.page_locked_int_),
      float_bytes_(o.float_bytes_),
      int_bytes_(o.int_bytes_)
{
    o.float_buf_ = nullptr;
    o.int_buf_ = nullptr;
    o.page_locked_int_ = nullptr;
    o.float_bytes_ = 0;
    o.int_bytes_ = 0;
}

AttentionWorkspace& AttentionWorkspace::operator=(AttentionWorkspace&& o) noexcept {
    if (this != &o) {
        if (float_buf_) cudaFree(float_buf_);
        if (int_buf_) cudaFree(int_buf_);
        if (page_locked_int_) cudaFreeHost(page_locked_int_);
        float_buf_ = o.float_buf_;
        int_buf_ = o.int_buf_;
        page_locked_int_ = o.page_locked_int_;
        float_bytes_ = o.float_bytes_;
        int_bytes_ = o.int_bytes_;
        o.float_buf_ = nullptr;
        o.int_buf_ = nullptr;
        o.page_locked_int_ = nullptr;
        o.float_bytes_ = 0;
        o.int_bytes_ = 0;
    }
    return *this;
}

AttentionWorkspace::~AttentionWorkspace() {
    if (float_buf_) cudaFree(float_buf_);
    if (int_buf_) cudaFree(int_buf_);
    if (page_locked_int_) cudaFreeHost(page_locked_int_);
}

bool decode_supports_gqa(int gqa) {
    return gqa == 1 || gqa == 2 || gqa == 3 || gqa == 4 || gqa == 8;
}

}  // namespace pie_cuda_device::ops
