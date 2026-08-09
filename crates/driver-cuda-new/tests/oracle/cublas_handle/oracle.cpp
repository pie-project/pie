// The CublasHandle oracle — gate-cublas.
//
// Compiles the REAL `gemm/gemm.cpp` — the whole 2.5k-line TU, over a full
// cublas/cublasLt declaration stub, with `--gc-sections` discarding the
// dispatchers — and drives the twenty lines this gate is about:
// `CublasHandle`'s constructor (create, then set-stream ONLY when a stream
// was given, then tensor-op math mode), the stream accessors, the
// destructor's destroy-if-nonnull, and the two constructor failure paths.
//
// The second failure is the one worth a case: `cublasSetMathMode` failing
// throws out of a half-built object, so the destructor never runs and the
// created handle LEAKS — the transcript shows a create with no destroy.
// That is the C++'s real behaviour, recorded rather than papered over; the
// Rust port must reproduce the refusal (its RAII cannot leak, and the
// parity maps the difference explicitly).
//
// The stub tree doubles as the foundation for a future act_x_w dispatch
// oracle — the quantized-routing logic in this same TU is the gemm operand
// family's real port surface.

#include <cstdio>
#include <stdexcept>
#include <string>

#include "gemm/gemm.hpp"

using pie_cuda_driver::kernels::gemm::CublasHandle;

namespace {

constexpr char SEP = '\x1f';
std::string g_case;
int g_next_handle = 0;
bool g_fail_create = false;
bool g_fail_math = false;
cudaStream_t g_last_stream = nullptr;

void note(const std::string& body) {
    std::printf("%s%c%s\n", g_case.c_str(), SEP, body.c_str());
}

std::string stream_name(cudaStream_t s) {
    if (s == nullptr) return "s0";
    if (s == reinterpret_cast<cudaStream_t>(0xA0)) return "sA";
    if (s == reinterpret_cast<cudaStream_t>(0xB0)) return "sB";
    return "s?";
}

std::string handle_name(cublasHandle_t h) {
    if (h == nullptr) return "null";
    return "h#" + std::to_string(reinterpret_cast<std::uintptr_t>(h) - 1);
}

}  // namespace

// ── the cuBLAS recorders ────────────────────────────────────────────────────

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    if (g_fail_create) {
        g_fail_create = false;
        note("create FAIL");
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    *handle = reinterpret_cast<cublasHandle_t>(
        static_cast<std::uintptr_t>(g_next_handle + 1));
    note("create " + handle_name(*handle));
    ++g_next_handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    note("destroy " + handle_name(handle));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
    note("set-stream " + handle_name(handle) + " " + stream_name(stream));
    g_last_stream = stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream) {
    *stream = g_last_stream;
    note("get-stream " + handle_name(handle) + " -> " +
         stream_name(g_last_stream));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    if (g_fail_math) {
        g_fail_math = false;
        note("math-mode FAIL");
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    note("math-mode " + handle_name(handle) + " mode=" +
         std::to_string(static_cast<int>(mode)));
    return CUBLAS_STATUS_SUCCESS;
}

int main() {
    cudaStream_t sA = reinterpret_cast<cudaStream_t>(0xA0);
    cudaStream_t sB = reinterpret_cast<cudaStream_t>(0xB0);

    // a. Default construction: no set-stream call at all.
    g_case = "a-default";
    {
        g_last_stream = nullptr;
        CublasHandle h;
        note("handle=" + handle_name(h.handle()));
        note("stream=" + stream_name(h.stream()));
    }

    // b. Stream-bound construction, rebind, and the getter.
    g_case = "b-stream";
    {
        CublasHandle h(sA);
        h.set_stream(sB);
        note("stream=" + stream_name(h.stream()));
    }

    // c. Create fails: the throw carries the status and the call name.
    g_case = "c-create-fail";
    {
        g_fail_create = true;
        try {
            CublasHandle h;
            note("no-throw");
        } catch (const std::exception& e) {
            note(std::string("threw ") + e.what());
        }
    }

    // d. Math mode fails: the throw leaves a half-built object, so the
    //    created handle leaks — a create row with no destroy row.
    g_case = "d-math-fail";
    {
        g_fail_math = true;
        try {
            CublasHandle h;
            note("no-throw");
        } catch (const std::exception& e) {
            note(std::string("threw ") + e.what());
        }
    }

    return 0;
}
