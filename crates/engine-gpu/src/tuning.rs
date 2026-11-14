//! The decisions every backend's registry holds alike, read through the
//! provided methods of [`Tactics`](crate::backend::Tactics).

use profile::Decision;

/// The row tile a routed matmul sorts its pairs into, by pairs per expert;
/// 1 leaves the pairs unsorted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(
    id = "kernel.moe.tile_rows",
    shape(per_expert: derived),
    constant(set(1, 16, 32, 64)),
    default = r#"[{ when = "per_expert < 2", value = 1 }, { when = "per_expert >= 32", value = 32 }, { value = 16 }]"#,
    tune = search
)]
pub struct MoeTileRows(pub u32);

/// The fewest rows per request that take the tiled attention kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.tile_rows_per_request", constant(range(1..=4096)), default = 32, tune = search)]
pub struct SdpaTileRowsPerRequest(pub u32);

/// The fewest keys a split slice holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "kernel.sdpa.decode.split_min_keys", constant(range(1..=65_536)), default = 256, tune = search)]
pub struct DecodeSplitMinKeys(pub u32);

/// The share of the device past the weights a fire's working set (the
/// activation span: the carve and the kernel scratch rooms) may take, in
/// percent; the rest is owed to the state the pool keeps (kv pages,
/// records, state slots). A policy, not a derivation: what a fire needs
/// and what the sequences between fires need are both served better by
/// more, and the load does not know how many sequences it will hold. The
/// cuda shell's decision of the same id and default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Decision)]
#[decision(id = "engine.working_set_percent", constant(range(10..=90)), default = 50, tune = engine)]
pub struct WorkingSetPercent(pub u32);
