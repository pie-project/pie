// Hermetic test harness for fatal_backstop.hpp.
//
// Self-contained: includes only the header under test plus libc, so it
// compiles and runs anywhere with a C++20 compiler — no pie/ggml/Metal
// build required. The driver thread's fatal-signal contract is exercised
// directly here; the in-app path is the same install()/handler code wired
// into driver/portable/src/entry.cpp.
//
// Modes (driven by run-fatal-backstop-test.sh):
//   signum <SIG>         print the OS signal number for <SIG>, exit 0.
//   marked <SIG>         mark this as a driver thread, install, raise <SIG>.
//                        Expect: diagnostic on stderr + death by <SIG>.
//   unmarked <SIG>       install a recovering "prev" handler, then install
//                        the backstop WITHOUT marking the thread, raise.
//                        Expect: NO diagnostic + chain to prev recovers.
//   marked_recover <SIG> recovering prev + marked + install, raise.
//                        Expect: diagnostic AND chain to prev recovers.

#include "../src/fatal_backstop.hpp"

#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstring>

static sigjmp_buf g_jmp;
static volatile sig_atomic_t g_prev_ran = 0;

// Stands in for wasmtime's guard-page handler: records that it ran and
// "recovers" by jumping back to the sigsetjmp point (savesigs=1 restores
// the signal mask), exactly the recover-and-continue shape our chain must
// preserve for non-driver-thread faults on Linux.
static void recovering_prev(int /*sig*/, siginfo_t* /*info*/, void* /*ctx*/) {
    g_prev_ran = 1;
    siglongjmp(g_jmp, 1);
}

static void install_recovering_prev(int sig) {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = recovering_prev;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    ::sigaction(sig, &sa, nullptr);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <mode> <SIG>\n", argv[0]);
        return 64;
    }
    const char* mode = argv[1];
    const int sig = pie_driver_backstop::signal_from_name(argv[2]);
    if (sig <= 0) {
        std::fprintf(stderr, "bad signal: %s\n", argv[2]);
        return 64;
    }

    if (std::strcmp(mode, "signum") == 0) {
        std::printf("%d\n", sig);
        return 0;
    }

    if (std::strcmp(mode, "marked") == 0) {
        pie_driver_backstop::mark_driver_thread();
        pie_driver_backstop::install("portable", "/test/model.gguf");
        ::raise(sig);
        std::fprintf(stderr, "ERROR: survived raise() in marked mode\n");
        return 1;  // unreachable: the chain to SIG_DFL must terminate us
    }

    if (std::strcmp(mode, "unmarked") == 0) {
        install_recovering_prev(sig);
        // Deliberately do NOT mark this thread.
        pie_driver_backstop::install("portable", "/test/model.gguf");
        if (sigsetjmp(g_jmp, 1) == 0) {
            ::raise(sig);
            std::fprintf(stderr, "ERROR: raise() returned without recovery\n");
            return 1;
        }
        std::printf("RECOVERED prev_ran=%d\n", static_cast<int>(g_prev_ran));
        return 0;
    }

    if (std::strcmp(mode, "marked_recover") == 0) {
        install_recovering_prev(sig);
        pie_driver_backstop::mark_driver_thread();
        pie_driver_backstop::install("portable", "/test/model.gguf");
        if (sigsetjmp(g_jmp, 1) == 0) {
            ::raise(sig);
            std::fprintf(stderr, "ERROR: raise() returned without recovery\n");
            return 1;
        }
        std::printf("RECOVERED-MARKED prev_ran=%d\n",
                    static_cast<int>(g_prev_ran));
        return 0;
    }

    std::fprintf(stderr, "unknown mode: %s\n", mode);
    return 64;
}
