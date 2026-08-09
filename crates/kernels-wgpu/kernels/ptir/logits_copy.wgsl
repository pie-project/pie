// PTIR logits row staging.
//
// One dispatch stages every row the fire needs, `global_invocation_id.y`
// choosing the per-row params record. It used to copy a single row and be
// submitted as its own command buffer, once per row, so a sixteen-request fire
// paid sixteen round trips per token to move sixteen vocabulary rows -- about
// 3ms of a 23.5ms step, scaling linearly with the batch.
//
// The row is UNSTATED in the table, deliberately: this backend's channel-plane
// interpreter never dispatches it, and a row is filled when a text names its
// symbol. So the bindings are the sibling backends' -- source 0, destination 1,
// params 2 -- and there is no `@group(1)`, because every scalar the copy needs
// is a field of the params record.
//
// `common/bf16.inc.wgsl` is not included and nothing here widens a bf16: a copy
// reproduces a bit pattern, and both tensors are `array<u32>` holding two bf16
// apiece, so one invocation moves one WORD -- columns `2x` and `2x+1`. That
// also makes the store race-free, which an element-per-invocation body would
// not be: WGSL cannot write half a word and has no sub-word atomic. It needs
// `vocab` to be EVEN, the same thing every bf16 row pitch in this tree needs --
// an odd one starts the next row inside the previous row's last word.

struct PtirLogitsCopyParams {
    source_row: u32,
    destination_row: u32,
    vocab: u32,
    reserved: u32,
}

@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<PtirLogitsCopyParams>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The record count is the params buffer's own length. Guarding on it rather
    // than on a stated count is what makes a grid rounded up on the row axis
    // harmless: without it a clamped read would re-copy the last record, which
    // is a write nobody asked for to a row somebody else owns.
    let row = gid.y;
    if (row >= arrayLength(&params)) { return; }

    let p = params[row];
    let pitch = p.vocab >> 1u;
    let x = gid.x;
    if (2u * x + 1u >= p.vocab) { return; }

    let at = p.destination_row * pitch + x;
    // A WGSL store past the end is clamped, not dropped, so an unguarded tail
    // corrupts the last word of the destination rather than doing nothing.
    if (at >= arrayLength(&destination)) { return; }

    destination[at] = source[p.source_row * pitch + x];
}

// pie:instantiate copy_logits_bf16
