// PIE_GEMM_DETERMINISTIC, checked the only way that catches the failure this
// project keeps meeting: a knob that is documented, believed, and no longer
// read. Asserting the parser alone would not catch it -- the parser would keep
// returning true while the reader had been replaced by a constant -- so the
// live `dense_gemm_deterministic()` is exercised in a child process per case,
// with the variable actually set or actually cleared. It caches its answer in
// a function-local static, which is why each case needs its own process.
//
// No device and no GPU: the switch is host C++, so this runs anywhere the
// driver compiles.

#include <stdlib.h>  // setenv/unsetenv are POSIX, not in <cstdlib>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "ops/gemm_determinism.hpp"

using pie_cuda_driver::ops::dense_gemm_deterministic;
using pie_cuda_driver::ops::dense_gemm_deterministic_value;
using pie_cuda_driver::ops::kDenseGemmDeterministicEnv;

namespace {

struct Case {
    const char* value;  // nullptr = the variable is not set at all
    bool deterministic;
};

constexpr Case kCases[] = {
    {nullptr, false},
    {"", false},
    {"0", false},
    {"1", true},
    {"true", true},
    {"yes", true},
};

const char* spelling(const char* value) { return value ? value : "<unset>"; }

// argv[2] is what the parent set the environment to; assert the reader agrees.
int run_child(const char* expected) {
    const bool want = std::strcmp(expected, "on") == 0;
    const bool got = dense_gemm_deterministic();
    if (got != want) {
        std::fprintf(stderr,
            "gemm_determinism_test: %s=%s -> dense_gemm_deterministic()=%d, "
            "expected %d. The switch is no longer read from the environment, "
            "so a deterministic run is silently a tuned one.\n",
            kDenseGemmDeterministicEnv,
            spelling(getenv(kDenseGemmDeterministicEnv)), got, want);
        return 1;
    }
    return 0;
}

int run_parent(const char* self) {
    int failures = 0;
    for (const Case& c : kCases) {
        if (dense_gemm_deterministic_value(c.value) != c.deterministic) {
            std::fprintf(stderr,
                "gemm_determinism_test: dense_gemm_deterministic_value(%s) "
                "should be %d\n", spelling(c.value), c.deterministic);
            ++failures;
        }
        if (c.value == nullptr) {
            unsetenv(kDenseGemmDeterministicEnv);
        } else {
            setenv(kDenseGemmDeterministicEnv, c.value, 1);
        }
        const std::string cmd = std::string("'") + self + "' child " +
                                (c.deterministic ? "on" : "off");
        if (std::system(cmd.c_str()) != 0) {
            std::fprintf(stderr,
                "gemm_determinism_test: live read failed for %s=%s\n",
                kDenseGemmDeterministicEnv, spelling(c.value));
            ++failures;
        }
    }
    if (failures != 0) return 1;
    std::printf("gemm_determinism_test: OK (%zu cases)\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "child") == 0) {
        return run_child(argv[2]);
    }
    return run_parent(argv[0]);
}
