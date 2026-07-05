/* pie_ipc.h — C ABI for the in-process IPC mechanism (the vtable handoff
 * between the Rust runtime and a same-process C++ driver).
 *
 * The wire-schema descriptor types (PieFrameDesc / PieResponseFrameDesc
 * and friends) and the cabi accessors live in `pie_driver_abi.h`; this header
 * carries only the mechanism — the direct-FFI vtable. It is intentionally
 * separate so the schema header stays mechanism-free.
 *
 * In-process vtable — direct-FFI handoff between Rust (runtime) and a
 * same-process C++ driver. Exchanges PieFrameDesc / PieResponseFrameDesc
 * pointers directly; no rkyv encode/decode on this path. The shmem path
 * (used by Python drivers) goes through pie_parse_<t> / pie_build_<t>
 * with rkyv bytes (see pie_driver_abi.h).
 *
 * Lifetime contract:
 *   - `recv` writes *out_request pointing to a PieFrameDesc that remains
 *     valid until the matching `send_response(req_id)`.
 *   - Every slice pointer inside that descriptor (and its nested
 *     sub-descriptors) shares the same lifetime.
 *   - `send_response` must copy any data it needs synchronously; the
 *     response descriptor and its slices are invalid after the call
 *     returns.
 */

#ifndef PIE_IPC_H
#define PIE_IPC_H

#include <stddef.h>
#include <stdint.h>

/* PieFrameDesc / PieResponseFrameDesc come from the schema header. */
#include <pie_driver_abi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t PieRecvResult;

/* #11 prefetch-seam: the C++ NVRTC-warm trampoline the Rust InProcChannel calls
 * (fire-and-forget) once per attached program before `submit`, off-TTFT, to warm
 * the JIT PTX cache. `backend_ctx` is the driver's IR backend (the compile-cache
 * owner). Bindings marshal as parallel `(kind, key)` arrays — the SAME shape the
 * submit path conveys (the carrier ships only `(kind, key)`; the trampoline
 * reconstructs `ready = SubmitBound` identically) — so the prefetch-warm program
 * hash MATCHES the real submit-fire hash ⇒ compile-cache HIT ⇒ the TTFT win.
 * kind: 0 = Logits, 1 = Tensor(host_key), 2 = MtpLogits. */
typedef void (*PiePrefetchFn)(
    void*           backend_ctx,
    const uint8_t*  bytecode,
    size_t          bytecode_len,
    const uint8_t*  binds_kind,
    const uint32_t* binds_key,
    size_t          binds_len);

typedef struct PieInProcVTable {
    PieRecvResult (*recv)(
        void* ctx,
        const PieFrameDesc** out_request,
        uint32_t*            out_req_id);

    void (*send_response)(
        void*                       ctx,
        uint32_t                    req_id,
        const PieResponseFrameDesc* response);

    void* ctx;

    /* #11 prefetch-seam (trailing, additive — recv/send_response/ctx keep their
     * offsets, so this is a pure append). The Rust side installs its prefetch
     * entry point here; the driver calls it ONCE at backend-ready —
     * `register_prefetch(ctx, &trampoline, backend_ctx)` — handing over the C++
     * trampoline plus the IR-backend context. Optional: a non-JIT driver (metal)
     * simply never calls it ⇒ prefetch stays a no-op (fire-and-forget tolerates
     * an unregistered trampoline). May be NULL on the Rust side for transports
     * that don't support prefetch. */
    void (*register_prefetch)(
        void*         ctx,
        PiePrefetchFn prefetch,
        void*         backend_ctx);
} PieInProcVTable;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PIE_IPC_H */
