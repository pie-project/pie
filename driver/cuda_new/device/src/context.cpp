#include "context.hpp"

namespace pie_cuda_device {

namespace {
thread_local std::string g_last_error;
}  // namespace

void set_last_error(std::string msg) { g_last_error = std::move(msg); }

const char* last_error() {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

}  // namespace pie_cuda_device
