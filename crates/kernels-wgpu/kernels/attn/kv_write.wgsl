// The KV appends, contiguous and paged: one binding contract, one scatter.
//
// ## Where these numbers come from
//
// Not from Metal, and not from the file next door. This backend sends the
// scalars to a uniform block, so the BUFFER run closes up by however many
// scalars precede an operand — `kv_append_paged`'s row lists `head_dim`,
// `page_size` and `n_kv_heads` between its buffers, and the write page and
// offset land at 10 and 11 rather than at the 13 and 14 a Metal index would
// suggest. `kernels-vulkan`'s copy of this shader carried 9 and 10 until a
// SPIR-V audit compared its `OpDecorate Binding` set against the row: off by
// one, which made it read `ring_11` as the write page and the write page as the
// offset. Do not transcribe these. Read the row through
// `kernels_wgpu::bindings`: the storage numbers are derived from the operand
// order, not copied from a sibling backend. The deleted `dump_layout` example
// only printed that answer.
//
// The row also names `ring_4`, `ring_6`..`ring_9`, `ring_11` and `ring_15`,
// which belong to a shared ring ABI this kernel does not read. They are real
// entries of the bind group layout and this module simply does not declare
// them: WGSL requires a shader's bindings to be a SUBSET of the layout, not
// equal to it.
//
// ## Why the grid is half a head wide
//
// This is a pure scatter into a cache two other dispatches also write, and a
// bf16 pair shares one `u32`. `pie_store_bf16` is a read-modify-write, WGSL has
// no sub-word atomic, and the two invocations that would share a word here are
// not even in the same workgroup once `head_dim` exceeds the workgroup width —
// so there is nothing that could order them. Each invocation therefore owns a
// PAIR of channels and writes a whole word. The host may still launch a full
// head in x; the upper half exits at the guard, which is the harmless
// direction.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> k_new: array<u32>;
@group(0) @binding(1) var<storage, read_write> v_new: array<u32>;
@group(0) @binding(2) var<storage, read_write> k_dst: array<u32>;
@group(0) @binding(3) var<storage, read_write> v_dst: array<u32>;

// The bf16 half-index unpack, per source buffer. `pie_load_bf16(&k_new, i)` is
// the shared answer and cannot be CALLED: its `ptr<storage, array<u32>, read>`
// parameter is WGSL's `unrestricted_pointer_parameters`, which naga does not
// implement, so a module that calls it parses and then fails
// `create_shader_module`. The CONVERSION keeps one definition in
// `common/bf16.inc.wgsl`.
fn k_new_at(i: u32) -> f32 {
    let word = k_new[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn v_new_at(i: u32) -> f32 {
    let word = v_new[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

//#if defined(PIE_PAGED)

@group(0) @binding(10) var<storage, read_write> w_page: array<u32>;
@group(0) @binding(11) var<storage, read_write> w_off: array<u32>;

struct Params { head_dim: i32, page_size: i32, n_kv_heads: i32 }
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.y;
    let i = gid.z;
    let hd = u32(params.head_dim);

    // The pair `(d, d + 1)`, owned outright. See the note at the top.
    let d = gid.x * 2u;
    if (d >= hd) { return; }

    // `(slot, head, channel)` over the pool the driver allocated, addressed
    // with the STATEMENT's head shape -- both halves of it, which is why the
    // row hands this kernel `n_kv_heads` as well as `head_dim`. A grid built
    // from the fire's `[256, 16]` where the statement said `[512, 4]` wrote the
    // top half of no KV head and put heads 4..15 in the next token's rows.
    let row_stride = u32(params.n_kv_heads) * hd;
    let slot = w_page[i] * u32(params.page_size) + w_off[i];
    let dst = slot * row_stride + h * hd + d;
    let src = i * row_stride + h * hd + d;

    // `row_stride` and `hd` are both even at every point of the head-dim axis,
    // so `dst` is even and the pair is exactly one word.
    let at = dst >> 1u;
    if (at < arrayLength(&k_dst)) {
        k_dst[at] = pie_pack_bf16(k_new_at(src), k_new_at(src + 1u));
    }
    if (at < arrayLength(&v_dst)) {
        v_dst[at] = pie_pack_bf16(v_new_at(src), v_new_at(src + 1u));
    }
}

//#else

@group(0) @binding(4) var<storage, read_write> pos: array<i32>;

// `head_dim` at 0, then two 64-bit strides at 8 and 16 -- NOT at 4 and 12. A
// `vec2<u32>` aligns to eight, so the lone `i32` in front of it is followed by
// four bytes of padding, and a shell that packed this block by concatenation
// would write both strides four bytes low and this shader would read two halves
// of two different numbers. `uniform_layout()` derives the offsets; nothing at
// runtime would report the mismatch, because a uniform buffer is just bytes.
struct Params { head_dim: i32, k_head_stride: vec2<u32>, k_seq_stride: vec2<u32> }
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.y;
    let hd = u32(params.head_dim);

    let d = gid.x * 2u;
    if (d >= hd) { return; }

    // Only the LOW word of each stride is read, and the high word is not
    // silently dropped -- it cannot matter. Every term of this index is
    // unsigned, so no product exceeds the sum it belongs to; the sum is an
    // element index into a bound storage range, and a storage buffer's range is
    // itself a 32-bit quantity. A `u32` multiply exact modulo 2^32 is therefore
    // exact. The ABI keeps 64 bits because it is shared with `kernels-metal`,
    // where a >4 GiB buffer makes the concern real.
    let dst = h * params.k_head_stride.x + u32(pos[0]) * params.k_seq_stride.x + d;
    let src = h * hd + d;

    let at = dst >> 1u;
    if (at < arrayLength(&k_dst)) {
        k_dst[at] = pie_pack_bf16(k_new_at(src), k_new_at(src + 1u));
    }
    if (at < arrayLength(&v_dst)) {
        v_dst[at] = pie_pack_bf16(v_new_at(src), v_new_at(src + 1u));
    }
}

//#endif

// pie:instantiate kv_append_bfloat16
// pie:instantiate kv_append_paged_bfloat16 PIE_PAGED=1
