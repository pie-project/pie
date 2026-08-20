// Compact the rows a fire will actually sample.
//
// Ported from Metal's row gather with the same arithmetic. The two numbers it
// reads are the row pitch and the number of rows to gather, and the ORDER
// between them is the ABI: `width` then `count`, exactly as `RowGatherParams`
// spelled it and as `layout/row_gather.metal` still takes it, buffer for
// buffer. A swap binds the request count as the pitch and reads whole rows out
// of the wrong place -- both are plausible `uint`s, so nothing would report it.
//
// THE TWO WORDS ARE A PUSH BLOCK NOW, NOT A PACKED STRUCT. This bound
// `RowGatherParams` as a std430 storage buffer at binding 3 and sent plain
// scalars to a push range, so there was no trailing scalar slot to append a
// count to: `count` was `Ty::InPacked`, the driver wrote it into the struct's
// second field, and the shader read `p.count`. Getting that reading wrong is
// what the WGSL sibling records this plane doing once, by inheriting Metal's --
// where a packed slot IS the buffer and a trailing scalar lands in the same
// argument -- and folding the count into a push block that was otherwise empty.
// That pushed a word nothing read and left the struct's second field holding
// whatever the params buffer happened to contain, which bounds the gather by
// garbage.
//
// The block is not empty any more. `width` is a `Const<uint>` mark on the
// routine -- word 0 of the statement's own run, which is where the struct's
// first field read it from -- and the body passes the derived request count
// straight after it, so `driver-vulkan`'s `Encoder::fire` builds an eight-byte
// push range out of the two in the order the body stated them. One descriptor
// fewer, one staging slot fewer, and the same two numbers.
//
// The first kernel of this tree written in Slang, and so the one that had to
// prove the SPIR-V is indistinguishable from what `glslc` produced: same
// entrypoint name, same bindings in the same order, same workgroup size.
// `tests/gpu.rs` runs it on the card and compares against the same expectation
// it always did.

#include "common/bf16.slang"

PIE_BUFFER_RO(0, input_);
PIE_BUFFER_RW(1, out_);
[[vk::binding(2)]] StructuredBuffer<uint> rows;

struct Push { uint width; uint count; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint c = gid.x;
    uint i = gid.y;
    if (c >= pc.width || i >= pc.count) return;
    out_[i * pc.width + c] = input_[rows[i] * pc.width + c];
}

// pie:instantiate row_gather_bfloat16
