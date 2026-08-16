//===-- geometry.cuh - the CSR arithmetic the driver does on device ---===//
//
// Three `__global__`s and nothing else. `geometry.cu` used to include this
// file and keep the launchers that fired the first two -- a split, not a copy,
// so that exactly ONE definition of each kernel existed. `2ef431d02` deleted
// that file, and this text was left with no launcher and no reader at all:
// `every_carried_file_is_reachable` is what said so. The launchers are Rust
// now, `driver_internal::{derive_kv_len, resolve_slot_to_block,
// compose_envelope_csr}`, over the root `src/layout.rs` declares for this
// file, and the one definition is the only one there ever needs to be.
//
// No kernel here has a row, and the reason is not geometry: the first two
// launch `<<<ceil(n/256), 256>>>` with their own bound check, which is exactly
// `LaunchRule::Elementwise`. It is that none is reachable from a model
// text. All three are called by the DRIVER while it composes a wave, not by a
// statement, so there is no fire whose operands a `Source` could name and
// inventing one would be a contract nothing checks. `new-horizon.md` §10.10
// put the extraction first precisely so that the row would be a later and
// separable decision; the decision has been taken, and it is that these are
// launched from `driver_internal` and belong to no `Family`.
//
// # What a device-composed decode wave needs, and why it is THIS file
//
// A `DecodeEnvelope` pass states its whole geometry on channels its own
// previous fire wrote: the token, the position, the pages, the page CSR, the
// KV length and the write descriptor. None of those values has ever been on
// the host -- the producing fire has not run when the consuming fire is
// converted -- so a driver that reads them back synchronizes between the two
// slots of one frame and gives up exactly the run-ahead the frame exists for.
// The deleted C++ driver said it in one sentence, and it is the reason all of
// this is on the device rather than beside it: device composition is *"the
// only path that can resolve a chained descriptor: the host readback fallback
// cannot see a value the producing fire has not committed yet, so a decode
// step that reads the prefill's sampled token never became ready."*
//
// The three kernels are the three shapes that arithmetic takes. Two of them
// are a pair and read as one: `derive_kv_len` goes from a CSR to a length, and
// `compose_envelope_csr` goes the other way -- from the length a program
// traced to the last page's occupancy the attention kernel indexes with. A
// wave composed on the device needs both directions because it is handed one
// of the two and must produce the other, and which one depends on which ports
// the program bound. `resolve_slot_to_block` is the third: a traced page is a
// WORKING-SET slot and the pool wants a physical block, and the dictionary
// between them is per-wave host data while the slot itself is not.
//
// # What `compose_envelope_csr` has and has not been through
//
// It COMPILES -- `every_instantiation_compiles` hands it to NVRTC with the
// rest -- and it has never run, because it has no caller. Neither do the two
// above it, and the reason is the same one: the driver-side half of device
// composition is not written. What blocks that half is not the geometry at
// all but the CHANNEL PLANE underneath it. `driver-cuda` gives each bound
// INSTANCE its own device ring (`program::channel::Rings`, one per
// `program::session::Session`) and moves a value between two instances
// through the pinned HOST mirror `serve::register_channel` allocates -- so
// the first decode of a request, whose token its PREFILL instance published,
// has no device cell to read at all, and the two rings do not even agree on
// which ring slot a given ticket names. The C++ driver had one ring per
// CHANNEL (`DeviceChannelRegistry::cell_base`), which is what made a cell's
// address a host constant and its contents a device value. Until that is
// ported, this kernel can compose a wave whose members chain off THEMSELVES
// and not one whose members chain off each other -- and a composition that
// serves only the second and later decodes of a request is worse than none,
// because the first one would silently read a zeroed cell.
//
// All three are handed to NVRTC through the carried header set rather than
// through an include path, so nothing here may reach for the C++ standard
// library -- `u32` is `pie_device.cuh`'s, which is what `<cstdint>`
// used to be.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

// One thread per request. Derives `kv_len[r]` from the CSR page descriptors,
// bit-identical to the host formula in request.rs (append_request_with_options):
//   page_count = kv_page_indptr[r+1] - kv_page_indptr[r]
//   kv_len[r]  = page_count == 0 ? 0
//                                : (page_count - 1) * page_size + last_page_len
// All arithmetic is u32 (matches the host's Vec<u32> column) so the device and
// host results are byte-for-byte equal — the M5 C1-FINAL handshake invariant.
__global__ void derive_kv_len(
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    u32 page_size,
    u32 num_requests,
    u32* __restrict__ kv_len) {
  const u32 r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= num_requests) {
    return;
  }
  const u32 page_count = kv_page_indptr[r + 1] - kv_page_indptr[r];
  kv_len[r] =
      page_count == 0u ? 0u : (page_count - 1u) * page_size + kv_last_page_lens[r];
}

// One thread per flattened page slot. Resolves a working-set slot id to its
// physical page-pool BlockId via the runtime-uploaded dictionary:
//   page_indices[i] = slot_to_block[pages[i]]
// An out-of-range slot id (>= num_slots) is a loud sentinel (0xFFFFFFFF), never
// a silent wrap — a corrupt/padding slot must fail visibly, not gather a wrong
// page. Slot id 0 is valid and resolved like any other.
__global__ void resolve_slot_to_block(
    const u32* __restrict__ pages,
    const u32* __restrict__ slot_to_block,
    u32 num_slots,
    u32 count,
    u32* __restrict__ page_indices) {
  const u32 i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const u32 slot = pages[i];
  page_indices[i] = slot < num_slots ? slot_to_block[slot] : 0xFFFFFFFFu;
}

// One member of a device-composed decode wave, as the compose below needs it.
// Every field is HOST knowledge: where the member's staged arrays sit, how
// wide its `pages` channel was declared, and where its own working set starts
// in the frame's concatenated translation. The VALUES the member traced are
// not here -- they are the device arrays the kernel reads -- and that split is
// the whole design: the host says WHERE, the device says WHAT, and the two
// meet at kernel time on the stream the producing fire's kernels already ran
// on.
//
// It lives here rather than being spelled again in Rust for
// `GatherTokenOp`'s reason (`gather_tokens.cuh:58`): a second copy would be
// the same five words until someone reordered a field, and then it would be a
// silent mis-read of every member of the wave.
struct EnvelopeMember {
    // How many page slots the member's `pages` channel DECLARES. The traced
    // count may be smaller and never larger; the declaration is what the host
    // reserved room for and what it planned the attention against.
    u32 page_bound;
    // Where this member's `page_bound` staged slots begin -- the host prefix
    // sum of `page_bound`, which is why it is not derived here.
    u32 page_src;
    // The member's segment base in the frame's concatenated `kv_translation`.
    // Added to every logical slot the member traced, so that one dictionary
    // serves a wave of several working sets: a bare slot id means page 0 of
    // SOME conversation, and without the base a co-batched member attends its
    // neighbour's history.
    u32 xlat_base;
};

// One block, one thread per member: the whole of a decode wave's page CSR,
// composed from what the members' programs traced.
//
// # Why one block, and why a serial scan inside it
//
// The output CSR is one shared object -- member `i`'s page span begins where
// every earlier member's ended -- so the members cannot be spread across
// blocks without a grid-wide barrier this launch has no reason to pay for. A
// decode wave is at most a few hundred lanes wide, which is one block. That is
// the same shape `graph_pad_rows` has and the same reason it is not a row.
//
// # The compaction, and the bug it is against
//
// A member's pages are staged at its DECLARED width and read out at its TRACED
// width, and the two differ whenever a program reserved room it has not filled
// yet. Writing the CSR without moving the pages would leave a hole between
// member `i`'s last page and member `i+1`'s first, and a CSR is contiguous by
// construction -- the attention kernel reads member `i+1`'s span starting at
// `kv_page_indptr[i+1]`, so a hole makes it attend the tail of its neighbour's
// declaration. So the spans are compacted as they are copied, which is what
// the deleted `compose_decode_envelopes` did with the same shared prefix scan.
//
// # Fail-stop rather than a silent wrong answer
//
// A traced CSR that is empty, decreasing, or wider than its declaration, and a
// traced `kv_len` that does not land inside the last of the pages it claims,
// are all the same event: the program stated a geometry the wave cannot honour,
// and there is no host that could have caught it because no host saw the value.
// Such a member is given ONE page -- its own page 0, which it always holds --
// its row is marked invalid so the KV write skips it, and `kills` is bumped so
// the driver can report growth loudly. Clamping quietly would have the member
// attend or overwrite pages belonging to another conversation.
__global__ void compose_envelope_csr(
    const EnvelopeMember* __restrict__ members,
    const u32* __restrict__ traced_page_indptr,
    const u32* __restrict__ traced_pages,
    const u32* __restrict__ traced_kv_len,
    const u32* __restrict__ traced_w_slot,
    const u32* __restrict__ token_ids,
    u32 member_count,
    u32 page_size,
    u32* __restrict__ kv_page_indptr,
    u32* __restrict__ kv_page_indices,
    u32* __restrict__ kv_last_page_lens,
    u32* __restrict__ w_slot_out,
    u8* __restrict__ row_valid,
    u32* __restrict__ kills) {
    extern __shared__ u32 page_offsets[];
    const u32 i = threadIdx.x;
    const bool live = i < member_count;

    EnvelopeMember member{};
    u32 traced_begin = 0;
    u32 pages = 0;
    u32 last_page_len = 0;
    bool valid = false;
    if (live) {
        member = members[i];
        traced_begin = traced_page_indptr[2u * i];
        const u32 traced_end = traced_page_indptr[2u * i + 1u];
        pages = traced_end >= traced_begin ? traced_end - traced_begin : 0u;
        valid = pages != 0u && pages <= member.page_bound;
        if (valid) {
            // The last page's occupancy, and the check is the same statement
            // read backwards: `kv_len` is the readable extent after this
            // pass's writes land, so it must fall inside the last of the
            // `pages` pages -- above what the earlier ones already hold and no
            // further than the last one can reach.
            const u32 floor = (pages - 1u) * page_size;
            const u32 ceiling = pages * page_size;
            const u32 len = traced_kv_len[i];
            if (len > floor && len <= ceiling) {
                last_page_len = len - floor;
            } else {
                valid = false;
            }
        }
        // The in-band skip. A device-resolved geometry may state that a row
        // produces nothing, and it spells that `-1` in the token -- which is
        // why a host-wire fire refuses the sentinel outright rather than
        // embedding it as a real token id. The row still occupies its lane and
        // still attends, so only its validity byte changes.
        if (token_ids[i] == 0xFFFFFFFFu) {
            row_valid[i] = 0u;
        }
        if (!valid) {
            pages = 1u;
            last_page_len = 1u;
            traced_begin = 0u;
            row_valid[i] = 0u;
            if (kills != nullptr) atomicAdd(kills, 1u);
        }
        page_offsets[i] = pages;
    }
    __syncthreads();

    if (i == 0u) {
        u32 cursor = 0u;
        kv_page_indptr[0] = 0u;
        for (u32 m = 0u; m < member_count; ++m) {
            const u32 count = page_offsets[m];
            page_offsets[m] = cursor;
            cursor += count;
            kv_page_indptr[m + 1u] = cursor;
        }
    }
    __syncthreads();

    if (!live) return;
    const u32 out = page_offsets[i];
    if (valid) {
        for (u32 p = 0u; p < pages; ++p) {
            kv_page_indices[out + p] =
                member.xlat_base + traced_pages[member.page_src + traced_begin + p];
        }
        w_slot_out[i] = member.xlat_base + traced_w_slot[i];
    } else {
        // Its own page 0: a lane that failed the checks above still has to name
        // a page the pool holds, and the first page of its own working set is
        // the one page every live member owns by construction.
        kv_page_indices[out] = member.xlat_base;
        w_slot_out[i] = member.xlat_base;
    }
    kv_last_page_lens[i] = last_page_len;
}

}  // namespace pie::layout
