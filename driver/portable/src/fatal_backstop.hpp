// Process-level fatal-signal backstop for the embedded native driver.
//
// Why this exists
// ---------------
// The native driver runs as a *thread* inside the host process (the Rust
// server, which also hosts wasmtime). The C++ run wrappers only catch
// `std::exception`, the host's startup-exit detection only sees a clean
// nonzero return, and the driver is launched with
// `install_signal_handlers=0`. So an uncatchable native fault on the
// driver thread (a wild read past a truncated mmap, a bad kernel, a
// failed assert) kills the *whole* process with no surfaced cause — the
// app sees only a bare `crashed on signal=N`. #688 added a bounds guard
// for the one known trigger (a truncated GGUF); this is the general
// backstop for every other native fault.
//
// The wasmtime coexistence problem
// --------------------------------
// wasmtime owns SIGSEGV/SIGBUS (and SIGILL/SIGFPE) to implement wasm
// linear-memory guard pages and trap instructions. A naive fatal handler
// would fire on every *expected* guard-page fault and clobber wasm's
// recovery. The escape hatch is an invariant of this architecture:
// wasmtime executes wasm only on the host's worker threads, NEVER on the
// driver thread. So a fatal signal delivered while executing *on the
// driver thread* is always a real native driver fault, never a wasm trap.
//
// We therefore gate the *diagnostic* (never the chain) on whether the
// faulting thread could plausibly be carrying an expected wasm trap:
//
//   * macOS: wasmtime uses Mach exception ports for guard pages, so an
//     expected wasm trap is intercepted and recovered out-of-band and
//     NEVER delivered as a BSD signal to a sigaction handler. Hence *any*
//     signal that reaches this handler is already a genuine fatal fault —
//     on the main driver thread, a ggml compute worker, or a Metal
//     completion thread. We emit for all of them (the saved previous
//     disposition is just SIG_DFL, so we restore it and re-raise to die).
//     This is what surfaces an inference-time fault on a ggml worker
//     thread that entry.cpp never marked.
//   * Linux: wasmtime installs POSIX SIGSEGV/SIGBUS handlers, so an
//     expected wasm guard-page trap DOES reach a sigaction handler on a
//     host worker thread. There we must stay silent on any thread we did
//     not explicitly mark, and chain straight through to wasmtime's saved
//     handler — invoking `prev.sa_sigaction(sig, info, ctx)` with the
//     *same* `ucontext` the kernel will `sigreturn` into, so wasmtime's
//     in-handler recovery takes effect when we return. On Linux only the
//     thread that called `mark_driver_thread()` emits; ggml compute
//     workers are spawned inside ggml's default pthread pool (no pie-owned
//     init hook), so surfacing their faults there needs a ggml
//     worker-entry hook — deferred with the cuda/Linux follow-up.
//
// So the emit gate is: `t_on_driver_thread || !wasm traps can reach here`.
// `g_wasm_traps_reach_sigaction` defaults to the platform value above and
// is overridable so the hermetic test can exercise both regimes on one
// host.
//
// Everything that runs inside the signal handler is async-signal-safe:
// only `write(2)` of pre-sized buffers, `sigaction`, and `raise`. The
// flavor/model strings are snprintf'd into fixed buffers at install time
// (normal context), never inside the handler.

#pragma once

#include <atomic>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include <pthread.h>
#include <unistd.h>

namespace pie_driver_backstop {

inline constexpr int kFatalSignals[] = {SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT};
inline constexpr int kNumFatal =
    static_cast<int>(sizeof(kFatalSignals) / sizeof(kFatalSignals[0]));

// Previous dispositions (wasmtime's handler on Linux, SIG_DFL otherwise),
// one per entry in kFatalSignals. Populated once by install().
inline struct sigaction g_prev[kNumFatal];

// install() runs at most once per process; mark_driver_thread() runs once
// per driver thread.
inline std::atomic<bool> g_installed{false};

// Diagnostic context, fixed at install time so the handler only reads it.
inline char        g_flavor[32] = {0};
inline std::size_t g_flavor_len = 0;
inline char        g_model[1024] = {0};
inline std::size_t g_model_len = 0;

// Per-thread marker. Default-false on every thread; set true only on
// driver threads (see mark_driver_thread). Reading a thread_local in the
// handler is safe here: the driver links as a static lib into the host
// binary, so this resolves to local-exec TLS (a direct register+offset
// load, no __tls_get_addr call).
inline thread_local bool t_on_driver_thread = false;

// Whether an expected wasm guard-page trap can be delivered to this POSIX
// sigaction handler. False on Apple (wasmtime uses Mach exception ports),
// so any signal reaching us is a genuine fault on any thread — including
// unmarked ggml compute workers. True on Linux, where we must stay silent
// on unmarked threads to avoid emitting on recoverable wasm traps. A plain
// bool read in the handler is async-signal-safe; the test overrides it to
// exercise both regimes on one host.
#if defined(__APPLE__)
inline constexpr bool kWasmTrapsReachSigactionDefault = false;
#else
inline constexpr bool kWasmTrapsReachSigactionDefault = true;
#endif
inline bool g_wasm_traps_reach_sigaction = kWasmTrapsReachSigactionDefault;

// Write a string literal without strlen (async-signal-safe, length known
// at compile time). `s` must be a char array literal.
#define PIE_BACKSTOP_WLIT(s)                                        \
    do {                                                            \
        ssize_t pie_backstop_n_ =                                   \
            ::write(STDERR_FILENO, (s), sizeof(s) - 1);            \
        (void)pie_backstop_n_;                                      \
    } while (0)

inline void write_buf(const char* buf, std::size_t len) {
    if (len == 0) return;
    ssize_t n = ::write(STDERR_FILENO, buf, len);
    (void)n;
}

inline void write_signal_name(int sig) {
    switch (sig) {
        case SIGSEGV: PIE_BACKSTOP_WLIT("SIGSEGV"); break;
        case SIGBUS:  PIE_BACKSTOP_WLIT("SIGBUS");  break;
        case SIGILL:  PIE_BACKSTOP_WLIT("SIGILL");  break;
        case SIGFPE:  PIE_BACKSTOP_WLIT("SIGFPE");  break;
        case SIGABRT: PIE_BACKSTOP_WLIT("SIGABRT"); break;
        default:      PIE_BACKSTOP_WLIT("signal");  break;
    }
}

inline int fatal_index(int sig) {
    for (int i = 0; i < kNumFatal; ++i) {
        if (kFatalSignals[i] == sig) return i;
    }
    return -1;
}

inline void fatal_handler(int sig, siginfo_t* info, void* ctx) {
    // Emit on a marked driver thread, or — where wasm traps cannot reach a
    // sigaction handler (macOS Mach ports) — on any thread, since the
    // signal is then necessarily a genuine fault (e.g. a ggml worker).
    if (t_on_driver_thread || !g_wasm_traps_reach_sigaction) {
        // Format: "[pie-driver-<flavor>] fatal: caught <SIG> (model=<path>)"
        PIE_BACKSTOP_WLIT("[pie-driver-");
        write_buf(g_flavor, g_flavor_len);
        PIE_BACKSTOP_WLIT("] fatal: caught ");
        write_signal_name(sig);
        if (g_model_len > 0) {
            PIE_BACKSTOP_WLIT(" (model=");
            write_buf(g_model, g_model_len);
            PIE_BACKSTOP_WLIT(")");
        }
        PIE_BACKSTOP_WLIT("\n");
    }

    // Chain to the previous disposition. On a non-driver thread this is the
    // *only* thing we do, so wasm guard-page recovery is preserved exactly.
    const int idx = fatal_index(sig);
    if (idx >= 0) {
        const struct sigaction& prev = g_prev[idx];
        if (prev.sa_flags & SA_SIGINFO) {
            if (prev.sa_sigaction != nullptr) {
                // wasmtime's handler: it may rewrite `ctx` to recover. We
                // hand it the same ctx the kernel will sigreturn into, then
                // return so that recovery (or its own chain-to-default)
                // takes effect.
                prev.sa_sigaction(sig, info, ctx);
                return;
            }
        } else {
            if (prev.sa_handler == SIG_IGN) {
                return;  // previously ignored — honor that
            }
            if (prev.sa_handler != SIG_DFL && prev.sa_handler != nullptr) {
                prev.sa_handler(sig);
                return;
            }
        }
    }

    // Previous was SIG_DFL (or we somehow lost it): restore the default
    // disposition and re-raise so the process terminates with the correct
    // signal status (e.g. WTERMSIG == sig for the host to report).
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}

// Mark the calling thread as a driver thread. Call once at the top of each
// driver thread's run path, before any work that could fault. Idempotent
// and async-signal-safe.
inline void mark_driver_thread() { t_on_driver_thread = true; }

// Install the fatal-signal backstop process-wide. Safe to call from every
// driver thread; only the first call installs the handlers and records the
// diagnostic context. `flavor` is the driver flavor name (e.g. "portable");
// `model_path` may be null.
//
// Ordering note: drivers boot before the runtime creates wasmtime (see
// server::serve — drivers hand back caps, then bootstrap runs), so this
// installs *before* wasmtime registers its trap handlers. That is fine in
// either direction:
//   * wasmtime, installing after us, saves our handler as its chain target
//     and calls it for any fault it does not recognize as a wasm trap. A
//     driver-thread fault therefore still reaches us (we emit, then chain
//     to the default disposition we saved here and die with the right
//     status), and an expected wasm guard-page fault is handled and
//     recovered entirely inside wasmtime — it never reaches us.
//   * If the order were ever reversed, we are the first responder and chain
//     to wasmtime's saved handler, which recovers wasm faults and dies on
//     native ones.
// Either way the thread-local gate guarantees we only emit a diagnostic for
// faults on a driver thread, and wasm guard-page recovery is untouched.
inline void install(const char* flavor, const char* model_path) {
    bool expected = false;
    if (!g_installed.compare_exchange_strong(expected, true)) {
        return;  // already installed by an earlier driver thread
    }

    std::snprintf(g_flavor, sizeof(g_flavor), "%s", flavor ? flavor : "?");
    g_flavor_len = std::strlen(g_flavor);
    if (model_path != nullptr && model_path[0] != '\0') {
        std::snprintf(g_model, sizeof(g_model), "%s", model_path);
        g_model_len = std::strlen(g_model);
    }

    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = fatal_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    for (int i = 0; i < kNumFatal; ++i) {
        std::memset(&g_prev[i], 0, sizeof(g_prev[i]));
        ::sigaction(kFatalSignals[i], &sa, &g_prev[i]);
    }
}

// Map a signal name ("SIGSEGV", "SIGBUS", ...) to its number, or -1.
// Provided for tests that inject a named fault.
inline int signal_from_name(const char* name) {
    if (name == nullptr) return -1;
    if (std::strcmp(name, "SIGSEGV") == 0) return SIGSEGV;
    if (std::strcmp(name, "SIGBUS") == 0) return SIGBUS;
    if (std::strcmp(name, "SIGILL") == 0) return SIGILL;
    if (std::strcmp(name, "SIGFPE") == 0) return SIGFPE;
    if (std::strcmp(name, "SIGABRT") == 0) return SIGABRT;
    return -1;
}

}  // namespace pie_driver_backstop
