// Compact the rows a fire will actually sample.
//
// A prefill's stream is one row per TOKEN and its readout is one distribution
// per REQUEST, so the sampled rows are picked out before the lm head runs.
//
// ## The row this file exists to get right
//
// **The `@group(1)` block here holds BOTH words, and the order is the
// contract.** There used to be no `@group(1)` at all: the row stated five
// operands and the fifth, `count`, was `Ty::InPacked` -- a value the driver had
// to supply that got no slot of its own, because it was the second FIELD of the
// `RowGatherParams` struct binding 3 already carried, written while the driver
// filled that buffer.
//
// Folding `count` into a uniform block was the thing that could not be done
// THEN, because the block was empty: it would have pushed a word no shader read
// while leaving `params.count` holding whatever the params buffer happened to
// contain, so the guard below would bound the gather by garbage.
// `kernels-vulkan`'s table reader made exactly that mistake, inheriting Metal's
// reading, where a packed slot IS the buffer and a trailing scalar lands in the
// same argument.
//
// What changed is that `width` is a `Const<u32>` mark now rather than a struct
// field, so both words come from the BODY: it passes `width` and then the
// derived request count, and `driver-wgpu::lowering::routine::bind` packs
// body-passed scalars into the `@group(1)` block in the order the body passed
// them. `[width, count]` therefore lands with the layout `RowGatherParams` had,
// field for field -- the same two numbers, reached through the uniform path
// instead of through a staged storage struct, and neither of them a word the
// shader does not read.
//
// ## A gather moves bits, so nothing here widens a bf16
//
// `common/bf16.inc.wgsl` is not included. A copy reproduces the bit pattern,
// and rounding it through f32 and back would be a lossless operation at best
// and a rounding of a NaN at worst. Both tensors are `array<u32>` holding two
// bf16 apiece, so one invocation moves one WORD -- columns `2c` and `2c+1` of
// its row -- which also makes the write race-free without a sub-word atomic
// WGSL does not have.
//
// That word is whole only if `width` is EVEN, which is the same thing every
// bf16 body in this tree needs: an odd row pitch means row `i+1` starts in the
// middle of row `i`'s last word, and no invocation can own it. Every hidden
// size the tree loads is a multiple of 64.

@group(0) @binding(0) var<storage, read_write> input_: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;
@group(0) @binding(2) var<storage, read_write> rows: array<u32>;
// Width then count, exactly as `RowGatherParams` spelled it and as Metal's
// `row_gather.metal` still takes it, buffer for buffer. The statement states
// `[width]` -- which the routine reads as its one `Const<u32>` mark, slot 0 --
// and the body passes the derived request count straight after it, so the block
// is `[width, count]` and the order is the body's argument order.
struct Params { width: u32, count: u32 }
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let i = gid.y;
    let width = params.width;
    // `count` is the block's second word, and this is the only thing that
    // reads it.
    if (2u * c + 1u >= width || i >= params.count) { return; }

    let pitch = width >> 1u;
    let at = i * pitch + c;
    // The row count is a number the body derived and the driver packed, so it
    // can be right where the grid is wrong. Guard the destination anyway: a
    // WGSL store past the end is clamped rather than dropped, which corrupts
    // the last word instead of doing nothing.
    if (at >= arrayLength(&out_)) { return; }

    out_[at] = input_[rows[i] * pitch + c];
}

// pie:instantiate row_gather_bfloat16
