// The recording implementations behind stub/cuda_runtime.h.
//
// Pins are real host memory named `pin#K` in creation order; events are
// 1-byte allocations named `ev#K`. Symbolic names rather than addresses for
// the usual reason: a golden full of malloc's return values is a golden
// about malloc. Failure injection is one-shot and explicit, so a script's
// failure lands exactly where the script aimed it.

#include "cuda_runtime.h"

#include <cstdlib>
#include <map>

namespace oracle_cuda {
namespace {

std::vector<std::string> g_log;
std::map<const void*, std::string> g_pins;
std::map<const void*, std::string> g_events;
std::map<const void*, std::string> g_streams;
int g_next_pin = 0;
int g_next_event = 0;
bool g_fail_malloc = false;
bool g_fail_event = false;

std::string stream_name(cudaStream_t s) {
    if (s == nullptr) return "s0";
    auto it = g_streams.find(s);
    return it == g_streams.end() ? "unknown" : it->second;
}

std::string event_name(const void* e) {
    if (e == nullptr) return "null";
    auto it = g_events.find(e);
    return it == g_events.end() ? "unknown" : it->second;
}

}  // namespace

void reset_case() {
    g_log.clear();
    g_pins.clear();
    g_events.clear();
    g_streams.clear();
    g_next_pin = 0;
    g_next_event = 0;
    g_fail_malloc = false;
    g_fail_event = false;
}

const std::vector<std::string>& log() { return g_log; }
void note(const std::string& line) { g_log.push_back(line); }
void name_stream(cudaStream_t s, const std::string& name) { g_streams[s] = name; }

std::string pin_name(const void* ptr) {
    if (ptr == nullptr) return "null";
    auto it = g_pins.find(ptr);
    return it == g_pins.end() ? "unknown" : it->second;
}

void fail_next_malloc_host() { g_fail_malloc = true; }
void fail_next_event_create() { g_fail_event = true; }

}  // namespace oracle_cuda

cudaError_t cudaMallocHost(void** ptr, std::size_t size) {
    if (oracle_cuda::g_fail_malloc) {
        oracle_cuda::g_fail_malloc = false;
        oracle_cuda::note("pin-fail\x1f" + std::to_string(size));
        return 1;
    }
    *ptr = std::malloc(size == 0 ? 1 : size);
    const std::string name = "pin#" + std::to_string(oracle_cuda::g_next_pin++);
    oracle_cuda::g_pins[*ptr] = name;
    oracle_cuda::note("pin\x1f" + std::to_string(size) + "\x1f" + name);
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* ptr) {
    oracle_cuda::note("unpin\x1f" + oracle_cuda::pin_name(ptr));
    oracle_cuda::g_pins.erase(ptr);
    std::free(ptr);
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int) {
    if (oracle_cuda::g_fail_event) {
        oracle_cuda::g_fail_event = false;
        oracle_cuda::note("evc-fail");
        return 1;
    }
    *event = std::malloc(1);
    const std::string name = "ev#" + std::to_string(oracle_cuda::g_next_event++);
    oracle_cuda::g_events[*event] = name;
    oracle_cuda::note("evc\x1f" + name);
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    oracle_cuda::note("evd\x1f" + oracle_cuda::event_name(event));
    oracle_cuda::g_events.erase(event);
    std::free(event);
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    oracle_cuda::note("evs\x1f" + oracle_cuda::event_name(event));
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    oracle_cuda::note("evr\x1f" + oracle_cuda::event_name(event) + "\x1f" +
                      oracle_cuda::stream_name(stream));
    return cudaSuccess;
}
