// icb::rebind — the one shader the indirect-command-buffer plane adds.
//
// EVERY OTHER SHADER IN THIS CRATE COMPUTES A MODEL. This one computes the
// COMMANDS. It reads the fire descriptor (the same bytes
// `driver::fire::descriptor::pack` writes, header and class table), evaluates
// the derived `DescriptorAbi` — one law per moving component — and rewrites
// the indirect command buffer in place: grids, buffer offsets, staged
// scalars, and the pipeline of a slot whose entry picked another arm. One
// thread per slot, and no host walk anywhere.
//
// What it may write is `MTLIndirectComputeCommand`'s whole vocabulary and
// nothing beyond it: there is no `set_bytes` on an indirect compute command,
// which is why a scalar argument lives in a staged cell and this shader
// writes the CELL rather than the binding.
//
// The three law forms are `engine_metal::abi::Law`, and this is the only
// other place they are evaluated:
//
//     kind 0  const   v
//     kind 1  affine  v = base + Σ slope[k]·coord[k]
//     kind 2  ceil    v = mul · ⌈(α·rows + β) / div⌉
//
// The coordinates are not in the descriptor. They are read OUT of it, by the
// one linear functional per direction that `engine_metal::abi::Recipe` solves
// at load — the descriptor carries a class table and the laws are written in
// a basis of reachable directions, and the recipe is the inverse of the
// second in terms of the first.

#include <metal_stdlib>
#include <metal_command_buffer>

using namespace metal;

// Mirrored EXACTLY by `kernels_metal::icb` on the host side; the bytes are
// the interface and this is one half of the one layout.
#define ICB_MAX_AXES      4
#define ICB_MAX_PIPELINES 256
#define ICB_MAX_SLABS     128

// The descriptor's own layout, in 32-bit words — `driver::fire::descriptor`'s
// table read as `uint[]`, which is legal because every field of it is a
// little-endian `u32` on a 4-byte boundary and the one 64-bit field (a lane's
// word) is past everything this shader reads.
#define ICB_DESC_MAGIC    0u
#define ICB_DESC_VERSION  1u
#define ICB_DESC_ROWS     2u
#define ICB_DESC_LANES    3u
#define ICB_DESC_BUCKET   4u
#define ICB_DESC_CLASSES  5u
#define ICB_DESC_TABLE    8u   // first class record; four words each
#define ICB_CLASS_WORDS   4u
#define ICB_CLASS_ROWS    1u
#define ICB_CLASS_LANES   3u

struct IcbHandle { command_buffer icb; };
struct IcbPipes  { array<compute_pipeline_state, ICB_MAX_PIPELINES> p; };
struct IcbSlabs  { device char* p[ICB_MAX_SLABS]; };

struct IcbPlan {
    uint slots;
    uint axes;
    uint classes;
    uint magic;
    uint version;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct IcbLaw {
    long base;
    long slope[ICB_MAX_AXES];
    long mul;
    long alpha;
    long beta;
    long div;
    uint kind;      // 0 const, 1 affine, 2 ceil
    uint at_kind;   // 0 grid axis, 1 threadgroup axis, 2 argument
    uint at_index;
    uint arg_kind;  // 0 buffer offset, 1 four-byte scalar, 2 eight-byte scalar
    uint slab;
    uint cell;      // byte offset into the scalar arena
    uint pad0;
    uint pad1;
};

struct IcbBind {
    ulong offset;   // bytes into the reservation, or into the scalar arena
    uint index;     // the argument index
    uint kind;      // 0 reservation, 1 scalar cell, 2 absent
    uint slab;
    uint pad0;
};

struct IcbArm {
    uint pipe;
    uint law_at;
    uint law_count;
    uint bind_at;
    uint bind_count;
    uint lanes[3];
    uint group[3];
    uint pad0;
};

struct IcbSlot {
    uint arm_at;
    uint arm_count;
    uint pick;       // 0 one arm always, 1 a threshold on the window's rows
    uint threshold;
    uint rows_law;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct IcbPipe {
    uint width;   // threadExecutionWidth
    uint total;   // maxTotalThreadsPerThreadgroup
};

// The three law forms, evaluated. `rows` is the window's own row count, which
// is itself a law and is evaluated first.
static inline long icb_eval(device const IcbLaw& law,
                            thread const long* coord,
                            uint axes,
                            long rows) {
    if (law.kind == 0u) {
        return law.base;
    }
    if (law.kind == 1u) {
        long v = law.base;
        for (uint k = 0; k < axes; ++k) {
            v += law.slope[k] * coord[k];
        }
        return v;
    }
    long n = law.alpha * rows + law.beta;
    long q = n / law.div;
    if (n > 0 && (n % law.div) != 0) {
        q += 1;
    }
    return law.mul * q;
}

// The threadgroup a slot that stated none gets — `engine_metal::device::ctx::
// threadgroup`, in the one other place it has to be computed, because the
// grid it is derived from is what this shader just rewrote.
static inline uint3 icb_occupancy(IcbPipe pipe, uint3 lanes) {
    uint w = max(pipe.width, 1u);
    uint t = max(pipe.total, 1u);
    uint x = max(min(w, max(lanes.x, 1u)), 1u);
    uint y = max(min(t / x, max(lanes.y, 1u)), 1u);
    uint z = max(min(t / (x * y), max(lanes.z, 1u)), 1u);
    return uint3(x, y, z);
}

kernel void icb_rebind(device IcbHandle&        handle  [[buffer(0)]],
                       device IcbPipes&         pipes   [[buffer(1)]],
                       device IcbSlabs&         slabs   [[buffer(2)]],
                       constant IcbPlan&        plan    [[buffer(3)]],
                       device const uint*       desc    [[buffer(4)]],
                       device const long*       konst   [[buffer(5)]],
                       device const long*       coeff   [[buffer(6)]],
                       device const IcbSlot*    slotrow [[buffer(7)]],
                       device const IcbArm*     armrow  [[buffer(8)]],
                       device const IcbLaw*     lawrow  [[buffer(9)]],
                       device const IcbBind*    bindrow [[buffer(10)]],
                       device const IcbPipe*    piperow [[buffer(11)]],
                       device uint*             live    [[buffer(12)]],
                       device uint*             status  [[buffer(13)]],
                       device uint*             cells   [[buffer(14)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= plan.slots) {
        return;
    }
    // THE DESCRIPTOR IS CHECKED, NOT TRUSTED. A malformed one does not fault
    // on this side — it computes, over whatever rows the wrong numbers name —
    // so the two words `driver::fire::descriptor::unpack` checks are checked
    // here too, and a mismatch leaves every command exactly as it was.
    if (desc[ICB_DESC_MAGIC] != plan.magic) {
        if (gid == 0) { status[0] = 1u; }
        return;
    }
    if (desc[ICB_DESC_VERSION] != plan.version) {
        if (gid == 0) { status[0] = 2u; }
        return;
    }
    if (desc[ICB_DESC_CLASSES] != plan.classes) {
        if (gid == 0) { status[0] = 3u; }
        return;
    }

    // The coordinates, read out of the class table by the solved recipe.
    long coord[ICB_MAX_AXES];
    uint classes = plan.classes;
    for (uint k = 0; k < plan.axes; ++k) {
        long v = konst[k];
        for (uint c = 0; c < classes; ++c) {
            uint at = ICB_DESC_TABLE + ICB_CLASS_WORDS * c;
            long rows = (long)desc[at + ICB_CLASS_ROWS];
            long lanes = (long)desc[at + ICB_CLASS_LANES];
            v += coeff[k * 2u * classes + 2u * c] * rows;
            v += coeff[k * 2u * classes + 2u * c + 1u] * lanes;
        }
        coord[k] = v;
    }

    IcbSlot slot = slotrow[gid];
    long rows = icb_eval(lawrow[slot.rows_law], coord, plan.axes, 0);
    compute_command cmd(handle.icb, gid);

    // WALK'S RULE 1, ON THE DEVICE. A region with no rows is not dispatched
    // at all — the eager walk skips its nodes — so the slots standing in it
    // are reset. A reset command inside an executed range is skipped with no
    // error and no cost; a slot left live to exit immediately still costs a
    // dispatch boundary, which is why this is a reset and not a zero grid.
    if (rows <= 0) {
        if (live[gid] != 0u) {
            cmd.reset();
            live[gid] = 0u;
        }
        return;
    }

    uint which = slot.arm_at;
    if (slot.pick == 1u && rows >= (long)slot.threshold) {
        which = slot.arm_at + 1u;
    }
    IcbArm arm = armrow[which];
    // The liveness word carries WHICH ARM is encoded, not merely that
    // something is: a slot that was reset lost every binding it had, and a
    // slot that switched arms is a different entry with a different argument
    // list. Both are a full re-encode and one word says so.
    uint token = which + 1u;
    bool fresh = (live[gid] != token);

    cmd.set_compute_pipeline_state(pipes.p[arm.pipe]);
    if (fresh) {
        for (uint i = 0; i < arm.bind_count; ++i) {
            IcbBind bind = bindrow[arm.bind_at + i];
            if (bind.kind == 0u) {
                cmd.set_kernel_buffer(slabs.p[bind.slab] + bind.offset, bind.index);
            } else if (bind.kind == 1u) {
                cmd.set_kernel_buffer(cells + (bind.offset >> 2), bind.index);
            } else {
                cmd.set_kernel_buffer(cells, bind.index);
            }
        }
    }

    uint3 lanes = uint3(arm.lanes[0], arm.lanes[1], arm.lanes[2]);
    uint3 group = uint3(arm.group[0], arm.group[1], arm.group[2]);
    for (uint i = 0; i < arm.law_count; ++i) {
        device const IcbLaw& law = lawrow[arm.law_at + i];
        long v = icb_eval(law, coord, plan.axes, rows);
        if (law.at_kind == 0u) {
            lanes[law.at_index] = (uint)v;
        } else if (law.at_kind == 1u) {
            group[law.at_index] = (uint)v;
        } else if (law.arg_kind == 0u) {
            cmd.set_kernel_buffer(slabs.p[law.slab] + (ulong)v, law.at_index);
        } else if (law.arg_kind == 1u) {
            cells[law.cell >> 2] = (uint)v;
        } else {
            ulong wide = (ulong)v;
            cells[law.cell >> 2] = (uint)(wide & 0xffffffffUL);
            cells[(law.cell >> 2) + 1u] = (uint)(wide >> 32);
        }
    }

    uint3 grid = uint3(max(lanes.x, 1u), max(lanes.y, 1u), max(lanes.z, 1u));
    if (group.x == 0u && group.y == 0u && group.z == 0u) {
        group = icb_occupancy(piperow[arm.pipe], grid);
    }
    cmd.concurrent_dispatch_threads(grid, group);
    // EVERY SLOT CARRIES A BARRIER. An indirect command buffer's dispatches
    // are concurrent by kind and the walk assumes the serial pass a compute
    // encoder gives; measured, sixteen chained slots race without this.
    cmd.set_barrier();
    live[gid] = token;
}
