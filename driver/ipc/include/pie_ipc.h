/* pie_ipc.h — C ABI for the in-process IPC mechanism (the vtable handoff
 * between the Rust runtime and a same-process C++ driver).
 *
 * The wire-schema descriptor types (PieFrameDesc / PieResponseFrameDesc
 * and friends) and the cabi accessors live in `pie_schema.h`; this header
 * carries only the mechanism — the direct-FFI vtable. It is intentionally
 * separate so the schema header stays mechanism-free.
 *
 * In-process vtable — direct-FFI handoff between Rust (runtime) and a
 * same-process C++ driver. Exchanges PieFrameDesc / PieResponseFrameDesc
 * pointers directly; no rkyv encode/decode on this path. The shmem path
 * (used by Python drivers) goes through pie_parse_<t> / pie_build_<t>
 * with rkyv bytes (see pie_schema.h).
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
#include <pie_schema.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t PieRecvResult;

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
} PieInProcVTable;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PIE_IPC_H */
