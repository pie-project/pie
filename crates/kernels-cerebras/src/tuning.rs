//! The chip's calibration as profile decisions: the cost model's per-unit cycles and rates, and
//! [`Tuning`], the values a process prices under ([`get`]; the defaults until [`set`]).

use std::sync::{LazyLock, OnceLock};

use profile::{Decision, Domain, Facts, Profile, Registry, Var};

macro_rules! costs {
    ($($ty:ident, $field:ident = $id:literal, $default:literal, $doc:literal;)*) => {
        $(
            #[doc = $doc]
            #[derive(Clone, Copy, Debug, PartialEq, Decision)]
            #[decision(id = $id, constant(range(0..=1_000_000)), default = $default, tune = price)]
            pub struct $ty(pub f64);
        )*

        pub fn register(registry: &mut Registry) {
            $(registry.add::<$ty>();)*
        }

        /// The calibration one process prices under, read once from its profile.
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct Tuning {
            $(#[doc = $doc] pub $field: f64,)*
        }

        impl Tuning {
            /// The registry defaults; a test holds the two equal.
            pub const DEFAULT: Tuning = Tuning { $($field: $default,)* };

            #[must_use]
            pub fn of(profile: &Profile) -> Tuning {
                Tuning { $($field: profile.get::<$ty>().0,)* }
            }
        }
    };
}

costs! {
    DecodeMatvec, decode_matvec = "kernel.cost.decode.matvec_cycles", 0.4579326923076923,
        "Cycles a weight element a decode's weight PE multiplies (its stamped matvec phases).";
    DecodeKv, decode_kv = "kernel.cost.decode.kv_ms", 6.436e-6,
        "ms a KV element a decode's row PE reads (fitted).";
    DecodeChain, decode_chain = "kernel.cost.decode.chain_ms", 1.093e-6,
        "ms a cycle of a decode's reduction chains (fitted).";
    DecodeSpanRow, decode_span_row = "kernel.cost.decode.span_row_ms", 1.267e-5,
        "ms a row a decode layer's collectives cross down the columns (fitted).";
    DecodeSpanCol, decode_span_col = "kernel.cost.decode.span_col_ms", 0.0,
        "ms a column they cross along the rows (fitted).";
    DecodeHead, decode_head = "kernel.cost.decode.head_cycles", 0.537,
        "Cycles a head element a head PE multiplies (its stamped matvec phase).";
    DecodeLoopReduce, decode_loop_reduce = "kernel.cost.decode.loop_reduce_cycles", 1.0,
        "Cycles a cycle of the token loop's max-reduce.";
    LayerGdn, layer_gdn = "kernel.cost.decode.layer_gdn_ms", 8.018e-3,
        "ms a gated delta net layer's own work (least squares over the chip's steps).";
    LayerGlobal, layer_global = "kernel.cost.decode.layer_global_ms", 5.707e-3,
        "ms a full attention layer's own work.";
    LayerSliding, layer_sliding = "kernel.cost.decode.layer_sliding_ms", 10.653e-3,
        "ms a sliding-window attention layer's own work.";
    LayerSharedKv, layer_shared_kv = "kernel.cost.decode.layer_shared_kv_ms", 19.700e-3,
        "ms a layer's own work over another layer's KV.";
    LoopMerge, loop_merge = "kernel.cost.chain.loop_merge", 75.0,
        "Cycles a head control PE merges a received (value, id) pair and sends it on.";
    LoopWire, loop_wire = "kernel.cost.chain.loop_wire", 0.5,
        "Cycles a router hop of the token loop's pairs.";
    ChainAdd, chain_add = "kernel.cost.chain.add", 7.0,
        "Cycles a reduction chain member adds and forwards.";
    ChainMerge, chain_merge = "kernel.cost.chain.merge", 7.0,
        "Cycles a group end merges its group's sum with the level-2 sum.";
    ChainWire, chain_wire = "kernel.cost.chain.wire", 3.0,
        "Cycles a router hop of a chain's sums.";
    GemmTurn, gemm_turn = "kernel.cost.gemm.turn", 150.0,
        "Cycles a GEMM line's turn.";
    GemmPiece, gemm_piece = "kernel.cost.gemm.piece", 280.0,
        "Cycles a GEMM's piece past its turn and words.";
    GemmMacCall, gemm_mac_call = "kernel.cost.gemm.mac_call", 61.75,
        "Cycles a `mac32` call's fixed part.";
    GemmMacElem, gemm_mac_elem = "kernel.cost.gemm.mac_elem", 0.543,
        "Cycles a `mac32` multiply-accumulate.";
    GemmHop, gemm_hop = "kernel.cost.gemm.hop", 12.0,
        "Cycles a PE's hop of a GEMM all-reduce's chain.";
    GemmCall, gemm_call = "kernel.cost.gemm.call", 8.0,
        "Cycles a GEMM DSD op's fixed part.";
    GemmExp, gemm_exp = "kernel.cost.gemm.exp", 30.0,
        "Cycles a scalar exponent.";
    GemmKeyBlock, gemm_key_block = "kernel.cost.gemm.key_block", 11770.0,
        "Cycles a key block past its modelled terms.";
    GemmOp, gemm_op = "kernel.cost.gemm.op", 200.0,
        "Cycles a GEMM op's fixed part.";
    GemmMeasured, gemm_measured = "kernel.cost.gemm.measured", 1.0,
        "What a GEMM step measures over its price.";
    TiledWord, tiled_word = "kernel.cost.tiled.word", 1.0,
        "Cycles a wavelet received or sent.";
    TiledStreamWord, tiled_stream_word = "kernel.cost.tiled.stream_word", 1.17,
        "Cycles a wavelet of the weight stream.";
    TiledKey, tiled_key = "kernel.cost.tiled.key", 1350.0,
        "Cycles a key's turn through the hub's loop, its fixed part.";
    TiledKeyHop, tiled_key_hop = "kernel.cost.tiled.key_hop", 8.3,
        "Cycles a key's turn a row and column of the array.";
    TiledCall, tiled_call = "kernel.cost.tiled.call", 40.0,
        "Cycles a tiled DSD op's or call's fixed part.";
    TiledHop, tiled_hop = "kernel.cost.tiled.hop", 2.0,
        "Cycles a wavelet's hop through a router.";
    TiledScalar, tiled_scalar = "kernel.cost.tiled.scalar", 12.0,
        "Cycles a scalar statement.";
    TiledOp, tiled_op = "kernel.cost.tiled.op", 5100.0,
        "Cycles a tiled op's fixed part.";
    GridRow, grid_row = "kernel.cost.grid.row", 1150.0,
        "Cycles a grid row's fixed part.";
    GridFinal, grid_final = "kernel.cost.grid.final", 400.0,
        "Cycles a norm's scale and shift a group at the end of its pass.";
    StreamP1, stream_p1 = "kernel.cost.stream.p1", 0.55,
        "Cycles a weight-PE matvec element a row.";
    StreamDsdOp, stream_dsd_op = "kernel.cost.stream.dsd_op", 0.0,
        "Cycles a region DSD op's issue.";
    StreamDsdElem, stream_dsd_elem = "kernel.cost.stream.dsd_elem", 0.69,
        "Cycles a region DSD op's element.";
    StreamExpElem, stream_exp_elem = "kernel.cost.stream.exp_elem", 16.2,
        "Cycles a vector exp's element.";
    StreamRecipElem, stream_recip_elem = "kernel.cost.stream.recip_elem", 16.2,
        "Cycles a vector reciprocal's element.";
    StreamSumElem, stream_sum_elem = "kernel.cost.stream.sum_elem", 0.69,
        "Cycles a chunk sum's element.";
    StreamSumLevel, stream_sum_level = "kernel.cost.stream.sum_level", 20.0,
        "Cycles a chunk sum's level.";
    StreamScalarMath, stream_scalar_math = "kernel.cost.stream.scalar_math", 60.0,
        "Cycles a scalar math call.";
    StreamAllReduce, stream_all_reduce = "kernel.cost.stream.all_reduce", 210.0,
        "Cycles a segment all-reduce over a unit's CX roots.";
    StreamRotCall, stream_rot_call = "kernel.cost.stream.rot_call", 170.0,
        "Cycles a rotation table call.";
    StreamRotChange, stream_rot_change = "kernel.cost.stream.rot_change", 429.0,
        "Cycles a row whose position moved.";
    StreamRotLevel, stream_rot_level = "kernel.cost.stream.rot_level", 152.0,
        "Cycles a rotation level recomputed.";
    StreamRotDirectOp, stream_rot_direct_op = "kernel.cost.stream.rot_direct_op", 30.0,
        "Cycles a direct rotation's DSD op issue.";
    StreamStoreWord, stream_store_word = "kernel.cost.stream.store_word", 1.0,
        "Cycles a held word cast out or summed back.";
    StreamChainHop, stream_chain_hop = "kernel.cost.stream.chain_hop", 2.0,
        "Cycles a hop of the row chain.";
    StreamStore, stream_store = "kernel.cost.stream.store", 508.0,
        "Cycles a stored row.";
    StreamQuery, stream_query = "kernel.cost.stream.query", 1100.0,
        "Cycles a query.";
    StreamQueryEntry, stream_query_entry = "kernel.cost.stream.query_entry", 151.0,
        "Cycles an entry a query's PE keeps.";
    StreamFabricWord, stream_fabric_word = "kernel.cost.stream.fabric_word", 1.0,
        "Cycles a word a root sends to or takes from the fabric.";
    StreamForwardWord, stream_forward_word = "kernel.cost.stream.forward_word", 1.0,
        "Cycles a forwarded row's word at a CX root.";
    StreamMeasured, stream_measured = "kernel.cost.stream.measured", 1.41,
        "What a stream step measures over its price.";
    PlaceKvHop, place_kv_hop = "kernel.cost.place.kv_hop", 2.08,
        "Cycles a hop of a K / V round trip.";
}

/// The facts a profile assumes: the system it calibrates.
pub const FACTS: &[Var] = &[Var::fixed("system", Domain::Enum(&["cs3"]))];

static REGISTRY: LazyLock<Registry> = LazyLock::new(|| {
    let mut r = Registry::new("cerebras", FACTS, &[]);
    register(&mut r);
    r
});

/// The decisions a Cerebras profile states.
#[must_use]
pub fn registry() -> &'static Registry {
    &REGISTRY
}

/// A CS-3's facts.
#[must_use]
pub fn cs3() -> Facts {
    Facts::of(registry()).word("system", "cs3")
}

/// The calibration profile file `text` states (`origin` names it), the rest at the defaults.
pub fn load(text: &str, origin: &str) -> Result<Tuning, String> {
    registry()
        .load_user(text, origin)
        .map(|p| Tuning::of(&p))
        .map_err(|e| e.to_string())
}

static TUNING: OnceLock<Tuning> = OnceLock::new();

/// The calibration this process prices under: what [`set`] set, else the defaults.
#[must_use]
pub fn get() -> &'static Tuning {
    TUNING.get().unwrap_or(&Tuning::DEFAULT)
}

/// Price under `t` from now on: once a process (the engine at load, `wafer-compile --profile`);
/// another calibration after it is refused.
pub fn set(t: Tuning) -> Result<(), String> {
    let at = TUNING.get_or_init(|| t);
    if *at == t {
        Ok(())
    } else {
        Err("the chip's calibration is set already, to other values".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_tuning_is_the_registry_defaults() {
        assert_eq!(registry().check(), Vec::<String>::new());
        assert_eq!(Tuning::of(&registry().defaults()), Tuning::DEFAULT);
        assert_eq!(Tuning::DEFAULT.decode_matvec, 1524.0 / 3328.0);
    }
}
