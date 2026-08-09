//! Turning a lowered fire into dispatches. The executor's other half.
//!
//! [`executor`] resolves a launch's operands to addresses; [`geometry`] turns
//! a rectangle into a thread grid. This is the walk that uses both, and its
//! whole shape is the claim the crate rests on:
//!
//! ```text
//! for launch in &lowered.launches {
//!     let sig  = sig_in(KERNELS, symbol)?;   // the row states the contract
//!     let args = bind(lowered, launch, ..)?; // the driver resolves names
//!     let grid = eval(sig.launch, dims)?;    // the row names the rule
//! }
//! ```
//!
//! **There is no per-family branch and no per-kernel arm.** A symbol is a
//! name: on this backend an entry point is compiled from one, so a text that
//! states a symbol the table knows needs no code written to receive it. That
//! is the difference the north star measures — `driver-cuda-new`'s executor
//! grows an arm per kernel "beside the bridge", and this one does not grow at
//! all.
//!
//! # Portable, and that is deliberate
//!
//! Nothing here touches a Metal object. A dispatch is a symbol, a file, a
//! grid, a threadgroup and a list of resolved operands — all of which are
//! decided before any device is involved, and all of which are therefore
//! provable in a build with no GPU. [`encode`] is the half that needs one.
//!
//! [`executor`]: super::executor
//! [`geometry`]: super::geometry
//! [`encode`]: super::encode

use core::ops::Range;

use kernels::KernelSig;
use model_compiler::lower::{Arg, Launch, Lowered};

use super::executor::{BindRefusal, BoundArg, FireTable, Frame, Resolver, bind};
use super::geometry::{Dims, Ungeometric, eval};

/// The fire-invariant half of [`Dims`]: what every launch of one fire shares.
///
/// The rectangle states the rows and the operands state the widths, so these
/// are the only quantities left — and they are the fire's geometry, handed in
/// by the caller that already knows it. The driver does not derive them:
/// deriving a head count from a buffer size is exactly the "model definition
/// inside the driver" that `batch/geometry.rs`'s `DecodeGeometry` is retiring
/// for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Geometry {
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Channels a partial rope rotates.
    pub rotary_dims: u32,
    /// Experts the router scores.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
}

/// One encodable dispatch: everything a command encoder needs, and nothing
/// that needs a command encoder to compute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch<'a> {
    /// The entry point to run. Borrowed from [`Lowered::kernels`] — the
    /// lowering's own spelling, unmodified, because the shader exports it
    /// under that name.
    pub symbol: &'a str,
    /// The shader that defines `symbol`, from its row's [`KernelSig::file`].
    pub file: &'static str,
    /// Total threads per axis.
    pub grid: [u32; 3],
    /// Threads per threadgroup per axis.
    pub threadgroup: [u32; 3],
    /// Operands **in the order the kernel reads them**, when its row states
    /// them; in the trace's stated order when it does not.
    ///
    /// The two are not the same order, and assuming they were is what made
    /// every launch of every text misbind. The trace states inputs, outputs
    /// then weights — the compiler's convention — while `affine_qmv_fast`
    /// declares `w, scales, biases, x, y`. A row that states its operands
    /// says which slot takes what, and [`reorder`] applies it.
    ///
    /// A row that states none is bound positionally, which is what every row
    /// got before and is wrong for most of them. That is why
    /// `tests/text_conformance.rs` counts them.
    pub args: Vec<BoundArg>,
    /// Where each scalar binds, and how wide it is there.
    ///
    /// Three facts, because three are needed and the row states all of them.
    ///
    /// * **Which buffer.** Two spellings exist in the tree: `moe/route.metal`
    ///   takes `constant RouterParams&`, one buffer holding every field, and
    ///   `quant/qmv.metal` takes its two extents as separate buffers.
    /// * **Which scalar**, as a byte offset into this dispatch's staged run.
    /// * **How wide.** `attn/sdpa_vector.metal` declares its strides
    ///   `const constant size_t&` — **eight bytes** — while the trace's params
    ///   are `u32`. A driver that handed a four-byte slot to an eight-byte
    ///   read would give the kernel four bytes of the next scalar as the high
    ///   half of this one. The row's `Ty` says which, so the stage widens.
    pub param_slots: Vec<ParamSlot>,
    /// The scalar arguments the statement states, in its stated order.
    ///
    /// A kernel takes numbers no operand shape gives — a QKV split's two
    /// widths, a strided kernel's row pitch. The **text** states them; this
    /// forwards them without knowing what they mean, which is the difference
    /// between a driver that passes a constant and one that re-derives it
    /// from a config it had to understand.
    pub params: Vec<u32>,
    /// The layers this rectangle covers.
    pub layers: Range<u16>,
    /// Which traced op produced it — where a refusal points.
    pub op: u32,
}

/// One scalar's placement: which buffer, where in the staged run, how wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamSlot {
    /// The argument-table index this binds at.
    pub slot: usize,
    /// Byte offset into this dispatch's staged scalars.
    pub at: u32,
    /// Bytes the kernel reads there — four or eight for a scalar.
    pub bytes: u32,
    /// This slot is a POINTER to a struct holding every remaining scalar,
    /// rather than one scalar.
    ///
    /// Both spellings are in the tree and the row's `Ty` tells them apart: a
    /// `Buf` param is `constant RouterParams&` — one buffer, every field —
    /// while an `I32` param is `const constant int&`, one buffer per number.
    /// A packed slot's run is as long as the statement's scalars; a scalar
    /// slot's is its own width.
    pub packed: bool,
    /// Which of the statement's scalars this is, or `None` for a slot the row
    /// names past what the statement states.
    pub value: Option<u8>,
}

/// Why a launch could not become a dispatch.
///
/// Every variant is drift: a fire that cannot be dispatched was lowered
/// against a table or a binding other than the one loaded, and no retry
/// changes that.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Undispatchable {
    /// An operand did not resolve. See [`BindRefusal`].
    Unbound {
        /// The symbol whose launch refused.
        symbol: String,
        /// The traced op it came from.
        op: u32,
        /// Which rule could not be applied.
        why: BindRefusal,
    },
    /// The lowering states a symbol no `kernel!` row declares, so nothing
    /// knows its contract. `attn::split_qkv_bf16` is today's instance.
    NoRow {
        /// The symbol with no row.
        symbol: String,
        /// The traced op that named it.
        op: u32,
    },
    /// The row exists but does not say which shader defines the symbol.
    /// Metal compiles at run time from `(path, entry name)`, so a row without
    /// a file cannot be reached. Fill it in the row, demand-driven.
    NoFile {
        /// The symbol whose row states no file.
        symbol: String,
        /// The traced op that named it.
        op: u32,
    },
    /// The row states no launch rule, so no grid can be produced from it.
    /// A guessed grid runs a kernel over the wrong extent, which no hardware
    /// reports — see [`Ungeometric`].
    Ungeometric {
        /// The symbol whose row states no rule.
        symbol: String,
        /// The traced op that named it.
        op: u32,
        /// Which refusal the rule made.
        why: Ungeometric,
    },
    /// The rectangle sits under a conditional region, so whether it runs is a
    /// question this walk cannot answer.
    ///
    /// `GuardMode::Union` keeps every arm and tags it, for a backend that can
    /// turn the tree back into conditional graph nodes. **Metal has no such
    /// API**: `Stepper` re-encodes every step, so the merged rectangle list IS
    /// the encode loop and `GuardMode::Resolve` is the mode that fits — the
    /// guards are answered before a rectangle exists.
    ///
    /// Reaching here means a fire was lowered in `Union` mode and handed to
    /// this walk, which would encode **every arm of every guard
    /// unconditionally**. That is not a slower answer, it is a different one.
    Conditional {
        /// The symbol whose rectangle is conditional.
        symbol: String,
        /// The traced op that named it.
        op: u32,
        /// Which region of [`Lowered::conds`] it sits under.
        cond: u32,
    },
}

/// Elements per row of the operand that sizes this launch.
///
/// The rectangle's operands are stated **inputs, outputs, then weights**
/// ([`Launch::args`]), and a weight carries no row width because its extent is
/// the tensor's. So the last operand with a width is the launch's last
/// *output*, and an output's row width is what every rule in the vocabulary
/// means by "width": a projection's output width, a norm's row width, an MLP's
/// intermediate.
///
/// Zero when the launch states no widthed operand at all, which leaves the
/// rule to refuse rather than this to guess.
fn sizing_width(lowered: &Lowered, launch: &Launch) -> u32 {
    widths(lowered, launch).next_back().unwrap_or(0)
}

/// Elements per row of the launch's FIRST widthed operand — its first input.
///
/// What sizes a statement that reads one packed buffer and writes several: no
/// one output spells the grid, because each is a fraction of the work.
fn input_width(lowered: &Lowered, launch: &Launch) -> u32 {
    widths(lowered, launch).next().unwrap_or(0)
}

/// The row widths this launch's operands state, in the trace's order.
fn widths<'a>(lowered: &'a Lowered, launch: &Launch) -> impl DoubleEndedIterator<Item = u32> + 'a {
    lowered.args[launch.args.start as usize..launch.args.end as usize]
        .iter()
        .filter_map(|arg| match arg {
            Arg::Arena { width, .. } | Arg::Named { width, .. } => Some(*width),
            Arg::Weight(_) => None,
        })
}

/// The dims one launch evaluates its rule at.
#[must_use]
pub fn dims_of(lowered: &Lowered, launch: &Launch, geometry: Geometry) -> Dims {
    Dims {
        rows: launch.rows.end - launch.rows.start,
        width: sizing_width(lowered, launch),
        in_width: input_width(lowered, launch),
        q_heads: geometry.q_heads,
        kv_heads: geometry.kv_heads,
        head_dim: geometry.head_dim,
        rotary_dims: geometry.rotary_dims,
        n_experts: geometry.n_experts,
        experts_per_token: geometry.experts_per_token,
    }
}

/// Turn one launch into a dispatch, against `table`.
///
/// `table` is `kernels_metal::KERNELS` in every caller; it is a parameter so
/// that this module depends on the kernel *vocabulary* rather than on one
/// table, which is what lets a test state its own rows.
///
/// # Errors
///
/// [`Undispatchable`] naming the symbol and the traced op, in every case.
pub fn plan_one<'a, S: Resolver>(
    lowered: &'a Lowered,
    launch: &Launch,
    table: &'static [KernelSig],
    frame: Frame,
    geometry: Geometry,
    resolver: &mut S,
) -> Result<Dispatch<'a>, Undispatchable> {
    let symbol = &lowered.kernels[launch.kernel as usize];
    // A conditional rectangle's guard was NOT answered by the lowering, and
    // this walk has no way to answer it: encoding it would run every arm.
    if launch.cond != Launch::NO_COND {
        return Err(Undispatchable::Conditional {
            symbol: symbol.clone(),
            op: launch.op,
            cond: launch.cond,
        });
    }
    let sig = kernels::sig_in(table, symbol).ok_or_else(|| Undispatchable::NoRow {
        symbol: symbol.clone(),
        op: launch.op,
    })?;
    let file = sig.file.ok_or_else(|| Undispatchable::NoFile {
        symbol: symbol.clone(),
        op: launch.op,
    })?;
    let grid = eval(sig.launch, dims_of(lowered, launch, geometry)).map_err(|why| {
        Undispatchable::Ungeometric {
            symbol: symbol.clone(),
            op: launch.op,
            why,
        }
    })?;
    let bound = bind(lowered, launch, frame, resolver).map_err(|why| Undispatchable::Unbound {
        symbol: symbol.clone(),
        op: launch.op,
        why,
    })?;
    let args = reorder(sig, &bound.args, lowered, launch, resolver);
    let (param_slots, params) = param_layout(sig, args.len(), lowered, launch, resolver);
    Ok(Dispatch {
        symbol: bound.kernel,
        params,
        file,
        grid: grid.grid,
        threadgroup: grid.tg,
        args,
        param_slots,
        layers: bound.layers,
        op: launch.op,
    })
}

/// Every launch of a lowered fire, in order, as dispatches.
///
/// The whole executor. It does not branch on a family, a fire class or a
/// kernel — it walks what the compiler stated.
///
/// # Errors
///
/// The first [`Undispatchable`]. Nothing partial is returned: a fire that
/// cannot be dispatched whole would otherwise run its prefix and leave the
/// arena half-written, which is indistinguishable from a model that answers
/// nonsense.
pub fn plan<'a, S: Resolver>(
    lowered: &'a Lowered,
    table: &'static [KernelSig],
    frame: Frame,
    geometry: Geometry,
    resolver: &mut S,
) -> Result<Vec<Dispatch<'a>>, Undispatchable> {
    lowered
        .launches
        .iter()
        .map(|launch| plan_one(lowered, launch, table, frame, geometry, resolver))
        .collect()
}

/// The launch's operands in the order the KERNEL reads them.
///
/// A row states its buffers as [`Operand`]s, each carrying a [`Source`] that
/// says where the value comes from — the statement's `i`-th input, its `i`-th
/// result, the `i`-th weight it names, its `i`-th scalar. This walks the row
/// and picks.
///
/// # What "the statement's i-th input" means here
///
/// The trace concatenates inputs, then outputs, then weights, and the binder
/// keeps that order. A weight is an [`Arg::Weight`]; everything before the
/// last widthed operand is an input and the widthed ones after are the
/// results. That is the same reading `sizing_width` makes, and the two must
/// agree or a rule sizes on an operand the row calls an input.
///
/// A row that states no operands is returned unchanged — bound positionally,
/// which is what every row got before any of them stated anything.
///
/// A [`Source`] the row leaves `Unbound`, or one naming an operand the
/// statement does not have, contributes a **zero slot**: an operand that
/// addresses nothing, which is the same honest answer `resolve_arg` gives a
/// `scale.` marker. It is not silently skipped, because a skipped slot shifts
/// every operand after it.
fn reorder<S: Resolver>(
    sig: &'static KernelSig,
    bound: &[BoundArg],
    lowered: &Lowered,
    launch: &Launch,
    resolver: &mut S,
) -> Vec<BoundArg> {
    if sig.operands.is_empty() {
        return bound.to_vec();
    }
    let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
    let widthed: Vec<usize> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| !matches!(a, Arg::Weight(_)))
        .map(|(i, _)| i)
        .collect();
    let weights: Vec<usize> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, Arg::Weight(_)))
        .map(|(i, _)| i)
        .collect();
    // How many of the widthed operands are RESULTS: the row says, because the
    // row is what knows how many values its kernel produces. A QKV split
    // writes three and a norm writes one, and the trace states them all after
    // its inputs, so the split is at `len - results`.
    let results = sig
        .operands
        .iter()
        .filter_map(|o| match o.source {
            kernels::Source::Out(i) => Some(usize::from(i) + 1),
            _ => None,
        })
        .max()
        // ZERO when the row names no `Out` at all, and that is not a
        // degenerate case: `kv_append` writes the POOL and produces no traced
        // value, so it has inputs and no outputs.
        //
        // This defaulted to ONE, which took `v_new` -- the last input -- for
        // an output, so `In(1)` had nothing left to resolve to and the append
        // bound a region addressing nothing where the values were. Measured
        // against a real checkpoint: the K pages filled and the V pages were
        // entirely zero, every layer, and the attention that read them
        // answered zero without failing.
        //
        // A row with no operands at all returns above, so nothing needs the
        // old default.
        .unwrap_or(0)
        .min(widthed.len());
    let (ins, outs) = widthed.split_at(widthed.len() - results);

    let nothing = BoundArg {
        slice: crate::model::executor::Slice {
            address: 0,
            bytes: 0,
        },
        width: 0,
    };
    // The layer this statement runs in, for the state lookups. A rolled trace
    // states a span and an unrolled one states a layer; the span's first is
    // the answer either way, because a rolled statement runs once per layer
    // and reaches this once per layer with it.
    let layer = launch.layers.start;
    sig.operands
        .iter()
        .map(|operand| match operand.source {
            kernels::Source::In(i) => pick(bound, ins.get(i as usize)),
            kernels::Source::Out(i) => pick(bound, outs.get(i as usize)),
            kernels::Source::Weight(i) => pick(bound, weights.get(i as usize)),
            // The KV cache is STATE, not an operand: no traced value stands
            // for it, so the pointer comes from the driver's own pool through
            // the resolver rather than from the statement's args.
            kernels::Source::KvKeys => resolver
                .kv(layer, false)
                .map_or(nothing, |slice| BoundArg { slice, width: 0 }),
            kernels::Source::KvValues => resolver
                .kv(layer, true)
                .map_or(nothing, |slice| BoundArg { slice, width: 0 }),
            // The fire's own tables. The row names which; this forwards the
            // name and never reads what it means.
            kernels::Source::TokenIds => fire(resolver, FireTable::TokenIds),
            kernels::Source::Positions => fire(resolver, FireTable::Positions),
            kernels::Source::RequestOfToken => fire(resolver, FireTable::RequestOfToken),
            kernels::Source::KvPageIndices => fire(resolver, FireTable::KvPageIndices),
            kernels::Source::KvPageIndptr => fire(resolver, FireTable::KvPageIndptr),
            kernels::Source::KvWritePage => fire(resolver, FireTable::KvWritePage),
            kernels::Source::KvWriteOffset => fire(resolver, FireTable::KvWriteOffset),
            kernels::Source::RopeFrequencies => fire(resolver, FireTable::RopeFrequencies),
            kernels::Source::SamplingIndices => fire(resolver, FireTable::SamplingIndices),
            kernels::Source::AttentionMask => fire(resolver, FireTable::AttentionMask),
            kernels::Source::AttentionMaskEnabled => {
                fire(resolver, FireTable::AttentionMaskEnabled)
            }
            // A scalar does not come out of the operand list at all — it rides
            // `Dispatch::params`, bound at the slot the row placed it — so its
            // slot here addresses nothing and the encoder's binding is what
            // the kernel reads.
            _ => nothing,
        })
        .collect()
}

/// Where each of a statement's scalars binds, and how wide.
///
/// A row that states no operands has them appended as one packed struct after
/// the operands — the only convention available when nothing said otherwise,
/// and right for the `RouterParams` shape.
///
/// A row that states them places each itself, and its `Ty` gives the width.
/// The staged run is laid out in the row's order with each scalar naturally
/// aligned, so an eight-byte stride starts on an eight-byte boundary rather
/// than wherever the previous four-byte extent happened to end.
fn param_layout<S: Resolver>(
    sig: &'static KernelSig,
    operands: usize,
    lowered: &Lowered,
    launch: &Launch,
    resolver: &mut S,
) -> (Vec<ParamSlot>, Vec<u32>) {
    let mut params: Vec<u32> =
        lowered.params[launch.params.start as usize..launch.params.end as usize].to_vec();
    if sig.operands.is_empty() {
        return (
            vec![ParamSlot {
                slot: operands,
                at: 0,
                bytes: 4,
                packed: true,
                value: Some(0),
            }],
            params,
        );
    }
    let mut at = 0u32;
    let mut out = Vec::new();
    for (slot, operand) in sig.operands.iter().enumerate() {
        // A pool number is not one of the statement's scalars: the driver
        // resolves it and APPENDS it, so the slot points at a value the
        // statement never carried. That is the whole reason it is a `Source`
        // — a stride is the pool's shape, and a text that guessed one would be
        // right for a deployment and silently wrong for the next.
        // A field of the PRECEDING packed struct: append the value and bind
        // nothing. The packed slot's run covers every scalar after it, so the
        // field lands where the struct expects it.
        if operand.ty == kernels::Ty::InPacked {
            params.push(match operand.source {
                kernels::Source::RequestCount => lowered.n_requests,
                _ => 0,
            });
            continue;
        }
        let pooled = match operand.source {
            kernels::Source::KvHeadStride => Some(FireTable::KvHeadStride),
            kernels::Source::KvSeqStride => Some(FireTable::KvSeqStride),
            kernels::Source::KvPageSize => Some(FireTable::KvPageSize),
            _ => None,
        };
        let (which, bytes, packed) = if let Some(want) = pooled {
            let value = resolver.pool(want).unwrap_or(0);
            params.push(value);
            let i = u8::try_from(params.len() - 1).unwrap_or(u8::MAX);
            (
                Some(i),
                if operand.ty == kernels::Ty::Usize {
                    8
                } else {
                    4
                },
                false,
            )
        } else {
            match operand.source {
                kernels::Source::Param(i) | kernels::Source::ParamF32(i) => match operand.ty {
                    kernels::Ty::Usize => (Some(i), 8, false),
                    // A pointer where a scalar could be is the packed struct.
                    kernels::Ty::Buf | kernels::Ty::BufMut => (Some(i), 4, true),
                    _ => (Some(i), 4, false),
                },
                _ => continue,
            }
        };
        at = at.next_multiple_of(bytes);
        out.push(ParamSlot {
            slot,
            at,
            bytes,
            packed,
            value: which,
        });
        at += bytes;
    }
    (out, params)
}

/// One of the fire's tables, or a region addressing nothing.
fn fire<S: Resolver>(resolver: &mut S, table: FireTable) -> BoundArg {
    resolver.fire(table).map_or(
        BoundArg {
            slice: crate::model::executor::Slice {
                address: 0,
                bytes: 0,
            },
            width: 0,
        },
        |slice| BoundArg { slice, width: 0 },
    )
}

/// The bound operand at `at`, or one that addresses nothing.
///
/// Nothing rather than a skip: a skipped slot shifts every operand after it,
/// which turns one missing pointer into a whole misbound launch.
fn pick(bound: &[BoundArg], at: Option<&usize>) -> BoundArg {
    at.and_then(|&i| bound.get(i).copied()).unwrap_or(BoundArg {
        slice: crate::model::executor::Slice {
            address: 0,
            bytes: 0,
        },
        width: 0,
    })
}

/// The distinct `(file, entry point)` pairs a dispatch list needs compiled.
///
/// In first-use order, deduplicated: a fire naming one symbol 28 times
/// compiles it once. This is what the device half hands to
/// `Compiler::compile_batch`, and it is here rather than there because it is a
/// property of the list, not of the GPU.
#[must_use]
pub fn pipelines_needed<'a>(dispatches: &[Dispatch<'a>]) -> Vec<(&'static str, &'a str)> {
    let mut out: Vec<(&'static str, &'a str)> = Vec::new();
    for d in dispatches {
        let pair = (d.file, d.symbol);
        if !out.contains(&pair) {
            out.push(pair);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::executor::Slice;
    use kernels::{LaunchRule, kernel};
    use model_compiler::trace::ValueId;

    /// Answers every name, so a test is about the walk rather than the store.
    #[derive(Default)]
    struct Anything;

    impl Resolver for Anything {
        fn weight(&mut self, _: &str) -> Option<Slice> {
            Some(Slice {
                address: 0x1000,
                bytes: 1 << 30,
            })
        }
        fn named(&mut self, _: ValueId) -> Option<Slice> {
            Some(Slice {
                address: 0x2000,
                bytes: 1 << 30,
            })
        }
    }

    static TABLE: &[KernelSig] = &[
        kernel!(sized "sized", file = Some("f.metal"), launch = LaunchRule::Rms),
        kernel!(no_file "no_file", launch = LaunchRule::Rms),
        kernel!(no_rule "no_rule", file = Some("f.metal")),
    ];

    /// One launch of `symbol` over `rows`, with the args given.
    fn one(symbol: &str, rows: u32, args: Vec<Arg>) -> Lowered {
        Lowered {
            // One request: these fixtures state one row.
            n_requests: 1,
            // A dispatch fixture is one launch, not a whole fire, so it has no
            // exit to state.
            readout: None,

            launches: vec![Launch {
                kernel: 0,
                rows: 0..rows,
                layers: 0..1,
                op: 7,
                cond: Launch::NO_COND,
                params: 0..0,
                args: 0..args.len() as u32,
                peel: None,
            }],
            kernels: vec![symbol.to_string()],
            rectangles: 1,
            arena_bytes: 4096,
            value_offset: Vec::new(),
            value_owner: Vec::new(),
            epilogue_gather: usize::MAX,
            epilogue_norm: usize::MAX,
            args,
            params: Vec::new(),
            structural: Vec::new(),
            residue: Vec::new(),
            conds: Vec::new(),
        }
    }

    fn frame() -> Frame {
        Frame {
            arena: Slice {
                address: 0x8000,
                bytes: 4096,
            },
        }
    }

    #[test]
    fn the_sizing_width_is_the_last_output_not_the_first_input() {
        // Args are stated inputs, outputs, then weights, and a weight has no
        // row width. So the last widthed operand is the output, and that is
        // what every rule means by "width".
        let low = one(
            "sized",
            3,
            vec![
                Arg::Arena {
                    at: 0,
                    width: 11,
                    bytes: 2,
                },
                Arg::Arena {
                    at: 64,
                    width: 22,
                    bytes: 2,
                },
                Arg::Weight("w".into()),
            ],
        );
        assert_eq!(sizing_width(&low, &low.launches[0]), 22);
    }

    #[test]
    fn a_launch_evaluates_its_rows_and_the_fires_geometry_together() {
        let low = one(
            "sized",
            5,
            vec![Arg::Arena {
                at: 0,
                width: 64,
                bytes: 2,
            }],
        );
        let geometry = Geometry {
            q_heads: 16,
            kv_heads: 4,
            head_dim: 128,
            ..Geometry::default()
        };
        let dims = dims_of(&low, &low.launches[0], geometry);
        assert_eq!(dims.rows, 5, "the rectangle states the rows");
        assert_eq!(dims.width, 64, "the operand states the width");
        assert_eq!(dims.q_heads, 16, "the fire states the rest");
    }

    #[test]
    fn a_symbol_with_no_row_names_itself_and_the_op_it_came_from() {
        let low = one(
            "attn::split_qkv_bf16",
            1,
            vec![Arg::Arena {
                at: 0,
                width: 8,
                bytes: 16,
            }],
        );
        assert_eq!(
            plan(&low, TABLE, frame(), Geometry::default(), &mut Anything),
            Err(Undispatchable::NoRow {
                symbol: "attn::split_qkv_bf16".into(),
                op: 7
            })
        );
    }

    #[test]
    fn a_row_that_states_no_file_cannot_be_reached_on_this_backend() {
        // Metal compiles at run time from `(path, entry name)`. A row without
        // a file is not a kernel this driver can find.
        let low = one(
            "no_file",
            1,
            vec![Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            }],
        );
        assert!(matches!(
            plan(&low, TABLE, frame(), Geometry::default(), &mut Anything),
            Err(Undispatchable::NoFile { .. })
        ));
    }

    #[test]
    fn a_row_that_states_no_rule_refuses_rather_than_launching_something_plausible() {
        let low = one(
            "no_rule",
            1,
            vec![Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            }],
        );
        assert_eq!(
            plan(&low, TABLE, frame(), Geometry::default(), &mut Anything),
            Err(Undispatchable::Ungeometric {
                symbol: "no_rule".into(),
                op: 7,
                why: Ungeometric::Unstated
            })
        );
    }

    #[test]
    fn a_fire_that_cannot_be_dispatched_whole_returns_nothing_partial() {
        // A prefix of dispatches would leave the arena half-written, which is
        // indistinguishable from a model that answers nonsense.
        let mut low = one(
            "sized",
            1,
            vec![Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            }],
        );
        low.kernels.push("no_rule".into());
        let second = Launch {
            kernel: 1,
            ..low.launches[0].clone()
        };
        low.launches.push(second);
        assert!(plan(&low, TABLE, frame(), Geometry::default(), &mut Anything).is_err());
    }

    #[test]
    fn a_conditional_rectangle_refuses_because_metal_cannot_answer_a_guard() {
        // `GuardMode::Union` keeps every arm for a backend that can build
        // conditional graph nodes. Metal has no such API and re-encodes every
        // step, so a union-lowered fire reaching this walk would encode every
        // arm of every guard unconditionally — a different answer, not a
        // slower one.
        let mut low = one(
            "sized",
            1,
            vec![Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            }],
        );
        low.launches[0].cond = 3;
        assert_eq!(
            plan(&low, TABLE, frame(), Geometry::default(), &mut Anything),
            Err(Undispatchable::Conditional {
                symbol: "sized".into(),
                op: 7,
                cond: 3
            })
        );
    }

    #[test]
    fn one_symbol_named_many_times_is_compiled_once() {
        let d = Dispatch {
            symbol: "sized",
            file: "f.metal",
            grid: [1, 1, 1],
            threadgroup: [1, 1, 1],
            args: Vec::new(),
            param_slots: vec![ParamSlot {
                slot: 0,
                at: 0,
                bytes: 4,
                packed: true,
                value: Some(0),
            }],
            params: Vec::new(),
            layers: 0..1,
            op: 0,
        };
        let list = vec![
            d.clone(),
            d.clone(),
            Dispatch {
                symbol: "other",
                ..d
            },
        ];
        assert_eq!(
            pipelines_needed(&list),
            vec![("f.metal", "sized"), ("f.metal", "other")]
        );
    }
}
