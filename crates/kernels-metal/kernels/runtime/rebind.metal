

#include <metal_stdlib>
#include <metal_command_buffer>

using namespace metal;

#define ICB_MAX_AXES      4
#define ICB_MAX_PIPELINES 256
#define ICB_MAX_SLABS     128

#define ICB_DESC_MAGIC    0u
#define ICB_DESC_VERSION  1u
#define ICB_DESC_ROWS     2u
#define ICB_DESC_LANES    3u
#define ICB_DESC_BUCKET   4u
#define ICB_DESC_CLASSES  5u
#define ICB_DESC_TABLE    8u
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
    uint kind;
    uint at_kind;
    uint at_index;
    uint arg_kind;
    uint slab;
    uint cell;
    uint pad0;
    uint pad1;
};

struct IcbBind {
    ulong offset;
    uint index;
    uint kind;
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
    uint pick;
    uint threshold;
    uint rows_law;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct IcbPipe {
    uint width;
    uint total;
};

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

    cmd.set_barrier();
    live[gid] = token;
}
