//! Binding a lowered launch's operands.
//!
//! `model_compiler::lower` turns a traced fire into rectangles whose operands
//! are [`Arg`]s — an arena offset, a backend-named value, or a weight name.
//! This resolves them. Three rules, stated once, for every family there will
//! ever be: **the driver binds, it does not choose**
//! (`model-compiler/DSL-DESIGN.md`).
//!
//! What binding is NOT: dispatch. A bound launch still has to reach the
//! pipeline the compiler built for its symbol, and that is the executor's other
//! half. Splitting them keeps this one pure host logic — provable against a
//! real lowered trace with no GPU in the build.
//!
//! # An operand carries its own extent
//!
//! This is the one place the Metal binder differs from `driver-cuda-new`'s,
//! and it is deliberate. That one resolves to a bare `*mut c_void` and keeps
//! the arena's size in a neighbouring field, hand-compared at the call site. A
//! pointer whose length lives somewhere else is precisely the boundary this
//! crate exists to remove — see the crate docs and the kernel panics that
//! motivated them.
//!
//! Here every operand is a [`Slice`]: an address **and** what may be addressed
//! from it. A bound arg therefore cannot be passed anywhere without its extent
//! travelling with it, and the arena bound is a property of the value rather
//! than a convention two call sites have to share.
//!
//! It is address-and-extent rather than a `metal::Handle` so the module stays
//! portable: the binder is host logic and is provable against a real lowered
//! trace with no device in the build. A caller on the device path builds a
//! [`Slice`] from a `Handle`, whose own `slice` has already made the same
//! check.

use model_compiler::lower::{Arg, Launch, Lowered};
use model_compiler::trace::ValueId;

/// The frame's activation arena: one block of [`Lowered::arena_bytes`],
/// allocated per fire or reused across them. The binder only addresses it.
///
/// An arena reused across fires can be **smaller** than the new fire needs, and
/// a launch that addressed past it would corrupt whatever the allocator placed
/// next — so the extent travels with the address rather than beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// The arena.
    pub arena: Slice,
}

/// Resolves the names a trace states against the driver's stores.
///
/// The one thing that stays per-family is a **map** rather than a switch —
/// `lower.rs`'s own words — and this is that map's seam.
pub trait Resolver {
    /// The region a weight the trace names (`layer.3.q_proj`) lives in, or
    /// `None`.
    ///
    /// `None` is **drift, not absence**: a trace that names a weight the store
    /// lacks was traced against a different binding, and running it would bind
    /// some other tensor to the operand.
    fn weight(&mut self, name: &str) -> Option<Slice>;
    /// The region a backend-named value lives in — the values a seam exposes,
    /// the ones `Buffers::NAMED` marks.
    fn named(&mut self, value: ValueId) -> Option<Slice>;

    /// A layer's KV pages: its keys when `values` is clear, its values when
    /// set.
    ///
    /// The third question, and it is a different KIND from the other two. A
    /// weight and a named value are both things the TRACE mentions. The KV
    /// cache is not: a statement names it as STATE
    /// (`StateRef { store: KvCache, layer }`) because the cache outlives the
    /// fire and no traced value stands for it. So a kernel that reads or
    /// writes it has a pointer that cannot come from the statement's args, and
    /// every backend has answered that with a hand-written arm.
    ///
    /// [`Source::KvKeys`] and [`Source::KvValues`] let a row ask instead, and
    /// this is where the asking lands.
    ///
    /// Defaulted to `None` so a resolver with no pool — the binder's tests,
    /// the name-map checks — needs no answer for a question it never faces. A
    /// statement that asks and gets `None` binds a region addressing nothing,
    /// which is the same honest answer a missing scale gets.
    ///
    /// [`Source::KvKeys`]: kernels::Source::KvKeys
    /// [`Source::KvValues`]: kernels::Source::KvValues
    fn kv(&mut self, _layer: u16, _values: bool) -> Option<Slice> {
        None
    }

    /// One of the FIRE's own tables.
    ///
    /// The token ids a gather reads, the positions a rope reads, the page CSR
    /// an attention walks. A text cannot state these — they are this fire's
    /// data, not this model's structure — so the ROW names which table a slot
    /// wants and this answers.
    ///
    /// That division is the point rather than a convenience. A kernel wanting
    /// the positions and a driver that KNEW to bind them is the hand-written
    /// arm this crate exists to remove; a row that names the table is the same
    /// fact, stated once, where a reader looks.
    ///
    /// Defaulted to `None` for the same reason [`Resolver::kv`] is: a
    /// resolver with no fire has no answer to a question it never faces.
    fn fire(&mut self, _table: FireTable) -> Option<Slice> {
        None
    }

    /// One of the pool's geometry numbers, as a value.
    ///
    /// A stride is `max_ctx * head_dim` for the pool the DRIVER allocated, so
    /// no text can state it. Answered here beside the pages themselves, and
    /// returned as a number rather than an address because it rides the
    /// scalar channel: the row names it `Param`-like and the encoder stages it
    /// with the statement's own.
    ///
    /// `None` for a resolver with no pool, and a slot it does not fill is a
    /// slot the kernel reads as zero — which for a stride means every head
    /// reads the first.
    fn pool(&mut self, _which: FireTable) -> Option<u32> {
        None
    }
}

/// Which of the fire's tables a slot wants.
///
/// Mirrors the `kernels::Source` variants that name one, so the driver never
/// matches on the table's meaning — it forwards a name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FireTable {
    /// The tokens this fire runs.
    TokenIds,
    /// Each token's absolute position.
    Positions,
    /// Which request owns each token.
    RequestOfToken,
    /// The KV page translation.
    KvPageIndices,
    /// Its per-request CSR.
    KvPageIndptr,
    /// The custom attention mask.
    AttentionMask,
    /// The per-lane byte saying whether the mask applies.
    AttentionMaskEnabled,
    /// Elements between one KV head's pages and the next.
    KvHeadStride,
    /// Elements between one token and the next within a head.
    KvSeqStride,
    /// Token rows per page.
    KvPageSize,
    /// Per token: the physical page its KV row is written into.
    KvWritePage,
    /// Per token: the row within that page.
    KvWriteOffset,
    /// The rotary inverse frequencies, `[rotary_dims/2]` f32.
    RopeFrequencies,
}

/// Where an operand is: a device address and the bytes it may address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slice {
    /// GPU address of the first byte.
    pub address: u64,
    /// Bytes addressable from it.
    pub bytes: u64,
}

/// One resolved operand: where it is, and how wide one row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundArg {
    /// The region the operand addresses.
    pub slice: Slice,
    /// Elements per row, for the args that carry one ([`Arg::Arena`],
    /// [`Arg::Named`]); zero for a weight, whose extent is the tensor's.
    pub width: u32,
}

/// A launch with every operand resolved — what a dispatch consumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundLaunch<'a> {
    /// The kernel's symbol, resolved through [`Lowered::kernels`]. A **name**:
    /// on this backend an entry point is compiled from one, so a symbol the
    /// lowering states needs no arm written to receive it.
    pub kernel: &'a str,
    /// The rectangle, in the op's own row space.
    pub rows: core::ops::Range<u32>,
    /// The layer range.
    pub layers: core::ops::Range<u16>,
    /// Operands in the trace's stated order: inputs, outputs, weights.
    pub args: Vec<BoundArg>,
}

/// Why a launch refused to bind.
///
/// Every variant is a **drift diagnosis, not a runtime condition**. A fire that
/// cannot bind was lowered against a different binding than the one loaded, and
/// no retry changes that.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindRefusal {
    /// An arena operand addresses past the frame's arena.
    ArenaOutOfBounds {
        /// The offending offset.
        at: usize,
        /// The row width it wanted from there.
        width: u32,
        /// What the frame actually holds.
        arena_bytes: u64,
    },
    /// The trace names a weight the resolver does not hold.
    UnknownWeight(String),
    /// The trace names a seam value the resolver does not bind.
    UnknownNamed(ValueId),
}

/// The marker a constant rides the weight-name slot under.
///
/// `dsl::cuda::scalar_mul` puts a scalar here and says *"a binder never looks
/// for it"* — the value reaches the kernel as a dispatch constant, and the
/// operand slot exists only so the launch's arity holds. Binding a zero-length
/// region is the honest reading: there is an operand, and it addresses nothing.
const SCALE_PREFIX: &str = "scale.";

/// Resolve one operand.
///
/// # Errors
///
/// [`BindRefusal`] naming which of the three rules could not be applied.
pub fn resolve_arg<S: Resolver>(
    arg: &Arg,
    frame: Frame,
    resolver: &mut S,
) -> Result<BoundArg, BindRefusal> {
    Ok(match arg {
        Arg::Arena { at, width, .. } => {
            let at64 = *at as u64;
            let bytes = frame.arena.bytes;
            // The row is `width` elements from `at`; the arena must hold the
            // offset itself, and an operand that starts inside it but runs off
            // the end is the same defect one byte later.
            if at64 >= bytes {
                return Err(BindRefusal::ArenaOutOfBounds {
                    at: *at,
                    width: *width,
                    arena_bytes: bytes,
                });
            }
            BoundArg {
                slice: Slice {
                    address: frame.arena.address + at64,
                    bytes: bytes - at64,
                },
                width: *width,
            }
        }
        Arg::Named { value, width } => BoundArg {
            slice: resolver
                .named(*value)
                .ok_or(BindRefusal::UnknownNamed(*value))?,
            width: *width,
        },
        Arg::Weight(name) => {
            if name.starts_with(SCALE_PREFIX) {
                return Ok(BoundArg {
                    slice: Slice {
                        address: 0,
                        bytes: 0,
                    },
                    width: 0,
                });
            }
            BoundArg {
                slice: resolver
                    .weight(name)
                    .ok_or_else(|| BindRefusal::UnknownWeight(name.clone()))?,
                // A weight's extent is the tensor's, so there is no row width
                // to carry and zero is not a missing value.
                width: 0,
            }
        }
    })
}

/// Resolve every operand of one launch.
///
/// # Errors
///
/// The first [`BindRefusal`] any operand produces. Nothing partial is returned:
/// a launch binds whole or not at all, because a dispatch with some operands
/// resolved is a dispatch that would read whatever was in the others.
pub fn bind<'a, S: Resolver>(
    lowered: &'a Lowered,
    launch: &Launch,
    frame: Frame,
    resolver: &mut S,
) -> Result<BoundLaunch<'a>, BindRefusal> {
    let span = launch.args.start as usize..launch.args.end as usize;
    let mut args = Vec::with_capacity(span.len());
    for arg in &lowered.args[span] {
        args.push(resolve_arg(arg, frame, resolver)?);
    }
    Ok(BoundLaunch {
        kernel: &lowered.kernels[launch.kernel as usize],
        rows: launch.rows.clone(),
        layers: launch.layers.clone(),
        args,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// An arena at a recognisable base, so an offset is visible in the result.
    fn arena(bytes: u64) -> Frame {
        Frame {
            arena: Slice {
                address: 0x1_0000,
                bytes,
            },
        }
    }

    #[derive(Default)]
    struct Store {
        weights: BTreeMap<String, Slice>,
        named: BTreeMap<ValueId, Slice>,
    }

    impl Resolver for Store {
        fn weight(&mut self, name: &str) -> Option<Slice> {
            self.weights.get(name).copied()
        }
        fn named(&mut self, value: ValueId) -> Option<Slice> {
            self.named.get(&value).copied()
        }
    }

    fn slice(address: u64, bytes: u64) -> Slice {
        Slice { address, bytes }
    }

    #[test]
    fn an_arena_operand_addresses_its_offset_and_reports_what_is_left() {
        let frame = arena(4096);
        let mut store = Store::default();
        let bound = resolve_arg(&Arg::Arena { at: 256, width: 64, bytes: 2 }, frame, &mut store)
            .expect("inside the arena");
        assert_eq!(bound.slice.address, 0x1_0000 + 256);
        assert_eq!(bound.slice.bytes, 4096 - 256, "what it may still address");
        assert_eq!(bound.width, 64);
    }

    #[test]
    fn an_arena_operand_past_the_frame_is_refused_with_both_numbers() {
        // An arena reused across fires can be smaller than the new fire needs,
        // and a launch that addressed past it would corrupt whatever the
        // allocator placed next.
        let frame = arena(1024);
        let mut store = Store::default();
        assert_eq!(
            resolve_arg(&Arg::Arena { at: 4096, width: 8, bytes: 2 }, frame, &mut store),
            Err(BindRefusal::ArenaOutOfBounds {
                at: 4096,
                width: 8,
                arena_bytes: 1024
            })
        );
    }

    #[test]
    fn a_weight_the_store_lacks_is_drift_and_names_itself() {
        let frame = arena(64);
        let mut store = Store::default();
        assert_eq!(
            resolve_arg(&Arg::Weight("layer.3.q_proj".into()), frame, &mut store),
            Err(BindRefusal::UnknownWeight("layer.3.q_proj".into())),
            "a trace naming a weight the store lacks was traced against another binding"
        );
    }

    #[test]
    fn a_weight_carries_no_row_width_because_its_extent_is_the_tensors() {
        let frame = arena(64);
        let mut store = Store::default();
        store
            .weights
            .insert("layer.3.q_proj".into(), slice(0xABC0, 8192));
        let bound = resolve_arg(&Arg::Weight("layer.3.q_proj".into()), frame, &mut store)
            .expect("the store holds it");
        assert_eq!(bound.slice, slice(0xABC0, 8192));
        assert_eq!(bound.width, 0, "zero is not a missing value here");
    }

    #[test]
    fn a_scale_constant_binds_an_operand_that_addresses_nothing() {
        // `dsl::cuda::scalar_mul`: "a binder never looks for it". The slot
        // exists so the launch's arity holds; the value reaches the kernel as a
        // dispatch constant. The store is never asked.
        let frame = arena(64);
        let mut store = Store::default();
        let bound = resolve_arg(&Arg::Weight("scale.rope_theta".into()), frame, &mut store)
            .expect("a scale never refuses");
        assert_eq!(bound.slice, slice(0, 0));
        assert!(store.weights.is_empty(), "and nothing was looked up");
    }

    #[test]
    fn a_named_value_the_seam_does_not_bind_is_refused_by_id() {
        let frame = arena(64);
        let mut store = Store::default();
        assert_eq!(
            resolve_arg(&Arg::Named { value: 7, width: 4 }, frame, &mut store),
            Err(BindRefusal::UnknownNamed(7))
        );
        store.named.insert(7, slice(0xF00, 16));
        assert_eq!(
            resolve_arg(&Arg::Named { value: 7, width: 4 }, frame, &mut store)
                .expect("bound now")
                .slice,
            slice(0xF00, 16)
        );
    }
}
