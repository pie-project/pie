//! The checkpoint, resident: one device allocation, one row per
//! `Trace::params` — and, in the same allocation and the same table, the
//! adapter banks (design §8).
//!
//! # Why the banks live HERE and not in `store/`
//!
//! They are runtime-MUTABLE, which is `store/`'s whole subject, and the pull
//! toward putting them beside the kv pages is real. Three things decide it the
//! other way.
//!
//! **The table has to be one table.** `Def::Weight(i)` resolves through
//! `WeightTable` and nowhere else (`Run::tensor` is the crate's heart, and its
//! whole point is that provenance handling exists exactly once). A bank is a
//! `Def::Weight` — that is the seat design §8's open item asked for, and the
//! reason it is the right seat is that MoE already proved a routed bank needs
//! no `Def` of its own. Splitting the STORAGE while sharing the TABLE would
//! put one invariant in two modules; splitting both would mean a second
//! resolution path in `Run` for a value that is a weight.
//!
//! **What `store/` owns is per-SEQUENCE and this is not.** A kv page belongs
//! to a slot, a recurrent slab belongs to a slot, and `Pools::clear(slot)`
//! is a sentence about a sequence beginning. A bank row belongs to no
//! sequence: many lanes route to it in one fire, it outlives every one of
//! them, and nothing about it is per slot.
//!
//! **Mutability here is not the fire path's.** `store/` is written by
//! LAUNCHES, inside the graph, every fire. A bank is written by
//! `register_adapter`, between fires, on the host — the same instant and the
//! same mechanism the checkpoint itself was landed by. It is a second load of
//! a few rows, not a pool.
//!
//! What the banks do inherit from the pools is the pointer-stability rule
//! (`inputs.rs` argues it): the store is reserved at the model text's declared
//! capacity and never grows, so an address recorded into a graph stays the
//! address a later fire reads. Registering the thirty-second adapter moves
//! nothing and recaptures nothing.
//!
//! # The contract arrives; the family does not
//!
//! This module takes a [`ModelContract`] and never asks which model it
//! describes. That is decision #18 read from the residency side: the runtime
//! links `model`, traces the `Trace` and states the load contract; the shell
//! compiles the contract against a checkpoint and lands the bytes. A shell
//! that reached for `models::qwen_3` would be a shell that has to grow an arm
//! per family, which is exactly the shape the string-plan era had.
//!
//! # Names are the plan's, both sides
//!
//! The load contract publishes one `Visibility::Public` entry per plan param,
//! **under the param's own name** — `embed`, `layer.7.q_proj` — and the
//! checkpoint's spellings live inside the expressions. So the bijection this
//! module needs is `trace.params[i].name == published name`, which is the
//! property `model/tests/the_zt_contract_states_the_cut.rs` asserts for every
//! SKU in the catalog. Both directions are checked here anyway, because the
//! consequence of a hole is not a missing tensor: it is
//! [`WeightTable`]'s `None`, reached at the first fire, in a panic.
//!
//! # Why the transforms run on the host
//!
//! The plan is compiled at `BackendKind::Cuda` — that is what fixes the
//! alignment and the tile budget — but the arena handed to the executor is a
//! `Vec<u8>`, so `ArenaBacking::runs_named_kernels` is false and every cast
//! runs host-side. For the SKUs this shell serves today that is a handful of
//! bf16→f32 widenings on norm scales; the device path is a load-time
//! optimisation. Landing the bytes correctly first, quickly second, is the
//! right order.
//!
//! The device path is `arena::CudaArena`, beside this module, behind
//! `feature = "_cuda"`. It moved here from the loader, which had carried it
//! behind an optional `cuda` feature that no engine's `src/` ever turned on —
//! see that module's header for why the seam is an `ArenaBacking` the
//! consumer supplies rather than a second executor. Binding it is one line
//! here (hand `Execution::arena` a `CudaArena` over the store's allocation
//! instead of a `Vec<u8>`) and is not taken yet; `tests/gpu_transform_parity.rs`
//! is what holds the device answer bit-identical to the host one until it is.

/// The device load arena and its four transforms — the other answer to "why
/// the transforms run on the host" above, gated on a chosen runtime because
/// it is the only part of this crate's load path that calls one.
#[cfg(feature = "_cuda")]
pub mod arena;

use std::collections::BTreeMap;
use std::path::Path;

// The two readers. `zt` comes in by module because its door is spelled
// `parse` — the module path is what says what is parsed, and a bare `parse`
// beside `parse_metadata` would say nothing about which of the two doors
// this line opened.
use checkpoint::file::read::parse_metadata;
use checkpoint::file::zt;
use checkpoint::contract::ModelContract;
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
use checkpoint::plan::{LoadPlan, StorageTarget, compile, compile_streaming};
use checkpoint::types::{BackendKind, ScaleForm, TensorId};
use kernels_cuda::Tensor;
use kernels_cuda::linear::moe::GroupSeat;
use model_ir::{Dtype, ParamSource, Trace};

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::experts::Attachments;
use crate::run::{WeightRow, WeightTable};

/// What a matrix operand wants under cuBLAS, and what `cudaMalloc` itself
/// guarantees — so a view into the store is as aligned as its own allocation
/// would have been. The same number the loader's `StorageTarget` states, for
/// the same reason.
pub(crate) const ALIGN: u64 = 256;

/// One plane of a registered adapter, as the caller hands it over.
///
/// **FULL CAPACITY, NOT THE ADAPTER'S OWN RANK.** The bytes are exactly one
/// slot of `bank`, which is that bank's declared `[rank, in]` or `[out, rank]`
/// rectangle in the bank's own dtype. An adapter trained at a lower rank is
/// zero-padded by the CALLER, because only the caller knows the layout it is
/// padding: `A`'s unused ranks are trailing rows and `B`'s are a stride inside
/// every row, so "write the prefix and leave the rest" is right for one plane
/// and wrong for the other. A short plane is `Fault::Adapter` naming both
/// numbers, never a partial write.
#[derive(Debug, Clone, Copy)]
pub struct AdapterPlane<'a> {
    /// The bank param this plane fills, as `Trace::params` names it.
    pub bank: &'a str,
    /// One slot's worth of bytes.
    pub bytes: &'a [u8],
}

/// Every weight this model needs, on the device — the checkpoint's and the
/// banks'.
#[derive(Debug)]
pub struct Weights {
    store: Buffer,
    table: WeightTable,
    /// Bank name -> (param index, adapters, bytes per adapter slot). Built at
    /// load off `ParamSource::Registered`, which is the only place a bank is
    /// declared; `register_adapter` is a lookup in here and a write.
    banks: BTreeMap<String, Bank>,
    /// **The routed-expert tier** (alto design §7, wave D2), or `None` for the
    /// degenerate load whose device budget covers the whole table.
    ///
    /// `Some` says some routed bank of this plan is a device slab smaller than
    /// the bank, over a pinned host copy of all of it, behind a device-resident
    /// indirection table. Everything that reads it is written so that `None`
    /// is not an arm of a branch but the absence of one: the weight rows are
    /// `Dense`, the select kernels get `ExpertTable::RESIDENT`, and the fire
    /// path is the fire path this shell had before D2.
    experts: Option<crate::experts::Tier>,
    /// **Did this table come off the warm-boot artifact instead of the
    /// checkpoint?** (alto design §7's T2 tier.)
    ///
    /// The per-load half of what
    /// [`weight_cache::observed`](crate::weight_cache::observed) counts for
    /// the whole process. `true` means the host-side transform pipeline never
    /// ran: the bytes are the same bytes a cold load would have produced,
    /// which is what the artifact's key and digest are for.
    from_cache: bool,
    /// **THE ROTATING DENSE PUMP** (alto streaming §3 item 4, D2b), or `None`
    /// for a load with nothing to rotate — which is every load whose budget
    /// held the table, and every load whose spilled planes are all over the
    /// slot cap.
    ///
    /// `Some` says a set of dense planes the device budget gave up are read
    /// out of DEVICE SLOTS whose contents rotate during the fire rather than
    /// out of the pinned tier over UVA. The addresses in the weight table are
    /// the slots' and never move ([`crate::rotate`]'s whole premise); what
    /// moves is the bytes, at the walk's region boundaries.
    rotor: Option<crate::rotate::Rotor>,
}

/// One declared adapter bank: where its slots are and how big they are.
#[derive(Debug, Clone, Copy)]
struct Bank {
    /// Where the bank's first slot starts in the store.
    offset: u64,
    /// How many adapters the first axis seats — the ceiling
    /// `Budget::max_adapters` is checked against, stated as a shape.
    adapters: u32,
    /// One adapter's bytes in this bank.
    slot: u64,
    /// **ONE SLOT'S RECTANGLE**, which is the param's shape past the leading
    /// adapters axis — `[rank, in]` for an `A` and `[out, rank]` for a `B`.
    ///
    /// Recorded because the SHARED-BLOB resolver needs it and nothing else
    /// does: a file declares which way it was written (alto adapter §6.3's
    /// out-major statute) and the only thing that can contradict the file is
    /// the bank's own shape. `register_adapter` still checks bytes and only
    /// bytes — a plane is one slot — so this adds no arm to the verb.
    rows: u64,
    /// The trailing half of that rectangle.
    cols: u64,
    /// One element of this bank's declared dtype, in bytes.
    elem: u64,
}

/// **ONE BANK, AS THE SHARED-ADAPTER RESOLVER READS IT** (alto adapter §6.3).
///
/// Everything [`crate::blob`] needs to slice a `[layers, ...]` file into one
/// full-capacity plane per bank: the name it registers under, the capacity,
/// the slot's bytes, and the rectangle those bytes are. A flattened
/// [`Bank`] — the resolver lives in another module and a private field is
/// not a contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankSeat {
    /// The param's own name, which is what a registration names.
    pub name: String,
    /// How many adapters it seats.
    pub adapters: u32,
    /// One adapter's bytes.
    pub slot: u64,
    /// The leading axis of one slot.
    pub rows: u64,
    /// The trailing axes of one slot, multiplied out.
    pub cols: u64,
    /// One element in bytes.
    pub elem: u64,
}

/// The banks a plan declares, read off `ParamSource::Registered`.
///
/// **ONE STATEMENT OF WHAT A BANK IS, AND IT IS THE PARAM'S OWN.** No suffix
/// convention, no name matching, no pairing table: a bank is a param the model
/// text marked registered, its capacity is that param's leading axis, and one
/// adapter's slot is everything after it. `A` and `B` are two independent
/// banks here — the op pairs them, this module does not, and a registration
/// names each by the name the plan gave it.
fn banks(trace: &Trace, places: &[Place]) -> BTreeMap<String, Bank> {
    trace.params
        .iter()
        .zip(places)
        .filter(|(param, _)| param.source == ParamSource::Registered)
        .map(|(param, place)| {
            let adapters = u32::try_from(param.shape.first().copied().unwrap_or(0))
                .unwrap_or(u32::MAX);
            let slot = if adapters == 0 {
                0
            } else {
                place.bytes / u64::from(adapters)
            };
            // The slot's own rectangle: the param's shape with the adapters
            // axis cut off. `[adapters, rank, in]` is a `[rank, in]` slot and
            // `[adapters, out, rank]` is an `[out, rank]` one — the same
            // `rectangle` split the store's layout uses, one axis in.
            let (rows, cols) = rectangle(param.shape.get(1..).unwrap_or(&[]));
            (
                param.name.clone(),
                Bank {
                    offset: place.offset,
                    adapters,
                    slot,
                    rows,
                    cols,
                    elem: model_compiler::arena::elem_bytes(param.dtype).unwrap_or(0),
                },
            )
        })
        .collect()
}

/// **How many device bytes this plan's weight table demands resident**, before
/// a single one of them is allocated (alto design §7).
///
/// The same arithmetic [`Weights::resident`] does — every param's rows times
/// width times element, each rounded up to the handle alignment — read off the
/// PLAN alone. A pure function of the trace, so a load that a
/// [`Residency`](engine::load::Residency) budget cannot admit is
/// refused before the device is touched rather than after the store is full.
///
/// # Errors
///
/// [`Fault::Param`] for a param whose dtype the arena has no element size for
/// — the same refusal the residency itself would answer with.
pub fn device_demand(trace: &Trace) -> Result<u64> {
    let places = places(trace, &crate::experts::Plan::default())?;
    Ok(places.last().map_or(0, |place| place.offset + place.reserved))
}

/// **Every param's plane bytes, unaligned and unreduced** — the one place the
/// arithmetic `rows x width x element` is written.
///
/// Read by [`places`] to lay the store out and by
/// [`experts::Plan::of`](crate::experts::Plan::of) to divide a routed bank
/// into expert slots. Two readers, one statement: a residency plan that
/// computed a bank's size differently from the store that reserves it would
/// be a plan about a different table.
///
/// **THE TWO PACKED PLANES ARE DECLARED IN TWO DIFFERENT UNITS, AND
/// `elem_bytes` CAN ANSWER FOR NEITHER.** `model_dsl::Weight::planes` gives an
/// mxfp4 bank the rectangle it OCCUPIES — each 32-code block folded into a
/// trailing axis of sixteen, which is the block's bytes — and gives an MLX
/// affine bank its LOGICAL rectangle, four bits an element, because that is
/// the shape its points index it by. So neither is `elements x element size`,
/// and taking `elem_bytes`'s honest `None` for the four-bit code as a refusal
/// is what refused every quantized SKU at its first param — which is the
/// `no element size` this shell answered a gpt-oss load with until wave W-5.
///
/// The refusal STAYS for a storage element that genuinely has no byte size
/// (`Dtype::E2m1`, a nibble with no declared packing): a plane whose length
/// cannot be computed is not a plane a store can reserve, and it is said by
/// name rather than guessed at.
///
/// # Errors
///
/// [`Fault::Param`] for a param declared in a storage element that has no
/// byte size.
pub(crate) fn plane_bytes(trace: &Trace) -> Result<Vec<u64>> {
    trace
        .params
        .iter()
        .map(|param| {
            let (rows, width) = rectangle(&param.shape);
            Ok(match param.dtype {
                // The shape already folds a 32-code block into sixteen bytes,
                // so the rectangle IS the byte count.
                Dtype::Mxfp4 => rows.saturating_mul(width),
                // The shape is logical and the element is a nibble: two codes
                // to a byte, and an odd row rounds up rather than overlapping
                // the next one.
                // **AND THE TILED ROW IS THE SAME NIBBLES**, in the order a
                // fragment reads them. A repack is size-preserving up to the
                // band padding, and the padding is already in the shape
                // (`model_dsl::Weight::planes` bands the rows), so this is
                // one arm and not two.
                Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => {
                    rows.saturating_mul(width).div_ceil(2)
                }
                // The same bank at two bits, sixteen codes to the `u32` word
                // MLX packs them into: a row of `width` codes is `width / 4`
                // bytes, and the `div_ceil` is the four-bit arm's argument one
                // notch narrower. The group is not in this number — it sizes
                // the COMPANION planes, which are their own params with their
                // own rectangles — which is why all three groups share the
                // arm. `engine_metal::weights::plane_bytes` has said the same
                // since 2-bit opened; this shell's copy was missing, and the
                // two `mlxu2` catalog rows refused at `Ranking::of` for it,
                // uncapped included, before a device was touched.
                Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => {
                    rows.saturating_mul(width).div_ceil(4)
                }
                // The same bank at eight bits: one whole byte a code, so the
                // logical rectangle IS the byte count and nothing rounds.
                Dtype::U8g64 => rows.saturating_mul(width),
                // **A STORED QUANTIZATION TERM, WHOSE SHAPE IS ALREADY ITS
                // CONTAINER.** `model_dsl::Weight::planes` folded the logical
                // `[n, k]` into `[n, Dtype::row_bytes(k)]` — one braided plane,
                // scales inside the payload, which is what serving AS STORED
                // means — so this is the mxfp4 arm's sentence with a term in
                // place of a constant: the rectangle IS the byte count.
                Dtype::U2g16k
                | Dtype::I3g16k
                | Dtype::U4g32k
                | Dtype::U5g32k
                | Dtype::I6g16k => rows.saturating_mul(width),
                other => {
                    let element =
                        model_compiler::arena::elem_bytes(other).ok_or_else(|| Fault::Param {
                            name: param.name.clone(),
                            why: "is declared in a packed storage element that has no \
                                  element size",
                        })?;
                    rows.saturating_mul(width).saturating_mul(element)
                }
            })
        })
        .collect()
}

/// **EVERYTHING A LOAD CAN BE PLANNED AGAINST BEFORE A BYTE OF IT IS READ.**
///
/// Two facts, one plan compile — because both come off the same `LoadPlan` and
/// compiling it twice to learn them separately would be paying twice for one
/// answer.
#[derive(Debug, Clone)]
pub struct Prospect {
    /// The split-plane pairings: which other params move when a packed bank
    /// moves. What [`experts::Plan::of`] budgets groups with.
    ///
    /// [`experts::Plan::of`]: crate::experts::Plan::of
    pub planes: Attachments,
    /// **The warm-artifact key a FULLY RESIDENT load of this deployment
    /// forms** — the name of the T2 source, and the reason a CAPPED load can
    /// find one at all.
    ///
    /// A load's own key is a function of the layout it chose, so a capped load
    /// and an uncapped load of one deployment key differently and neither can
    /// find the other's file. The fix is not a second key format: it is that
    /// the RESIDENT layout is a pure function of the trace
    /// (`places(trace, &Plan::default())`), so any load can compute the key
    /// the resident one would have used. That is this number, and it is what
    /// makes "boot once uncapped, then serve capped out of the artifact" a
    /// sentence an operator can act on.
    pub resident_key: u64,
    /// **THE SERVING ARTIFACT'S NAME** — and since §M it is here, beside the
    /// resident one, because it is the same KIND of number: a function of the
    /// deployment and not of this load's budgets.
    ///
    /// Format 2's key mixed both layouts and every rung, so it could only be
    /// formed once a `Plan` existed and a different budget named a different
    /// file. §M's mixes the RANKING instead (`Ranking::images`), which no
    /// budget touches — so one file serves any budget pair on this setup and
    /// the name can be formed here, out of the one metadata parse and the one
    /// plan compile this call already pays for.
    ///
    /// `0` for a load plan that will not serialize, which is the load that
    /// forms no key at all and therefore neither reads nor writes the file.
    pub tier_key: u64,
    /// **The priority ranking this trace declares**, before any budget cuts
    /// it — what [`experts::Plan::cut`] turns into a residency and what the
    /// serving artifact's payload is laid out from.
    ///
    /// [`experts::Plan::cut`]: crate::experts::Plan::cut
    pub ranking: crate::experts::Ranking,
}

/// **A ROUTED BANK'S OTHER DEVICE PLANES AND THE T2 SOURCE'S NAME, OFF THE
/// LOAD PLAN** — what [`experts::Plan::of`] and
/// [`Residency::admit_tiers`](engine::load::Residency::admit_tiers) need
/// before a byte is landed, and the reason they can be asked for without one.
///
/// A quantized bank is codes plus factors, and both are indexed by the number
/// the routing vector carries — so a residency decision that moved the codes
/// alone would leave the factors reading somebody else's expert
/// (`crate::experts`'s header states it whole). The pairing is the LOAD
/// PLAN's, read exactly as [`pairings`] reads it and never off a name; this
/// door exists so that the decision can be made BEFORE [`Weights::resident`]
/// reserves the store, which is what puts admission in front of the landing
/// rather than behind it.
///
/// Costs one metadata parse and one plan compile, and reads no tensor bytes.
///
/// # Errors
///
/// [`Fault::Load`] for a checkpoint the contract does not fit,
/// [`Fault::Param`] for an attachment this plan cannot resolve.
///
/// [`experts::Plan::of`]: crate::experts::Plan::of
pub fn prospect(trace: &Trace, contract: &ModelContract, path: &Path) -> Result<Prospect> {
    let metadata = if path.is_dir() {
        parse_metadata(path)?
    } else {
        zt::parse(path)?
    };
    let landing = compile(
        &metadata,
        contract,
        StorageTarget::for_backend(BackendKind::Cuda, 0, 1),
    )?;
    let index: BTreeMap<&str, usize> = trace
        .params
        .iter()
        .enumerate()
        .map(|(at, param)| (param.name.as_str(), at))
        .collect();
    let planes = attachments(&landing, &index)?;
    let ranking = crate::experts::Ranking::of(trace, &planes)?;
    Ok(Prospect {
        resident_key: resident_key(trace, &landing, path)?,
        tier_key: tier_identity(trace, &landing, path, &ranking)?.unwrap_or(0),
        ranking,
        planes,
    })
}

/// **THE SPLIT-PLANE PAIRINGS, AS A RESIDENCY DECISION WANTS THEM** — which
/// other params move when a packed bank moves, keyed by the bank's own row.
///
/// [`pairings`] read through one map instead of two: it answers with the
/// loader's `Pairing` and every caller then flattens it the same way. Stated
/// once because two callers now need it — [`prospect`], which asks before a
/// byte is landed, and [`Weights::resident`], which needs the same
/// [`Ranking`](crate::experts::Ranking) to name the file it is about to read.
///
/// # Errors
///
/// [`pairings`]', verbatim.
fn attachments(landing: &LoadPlan, index: &BTreeMap<&str, usize>) -> Result<Attachments> {
    let mut planes = Attachments::new();
    for (name, pairing) in pairings(landing, index)? {
        let Some(&at) = index.get(name) else {
            continue;
        };
        let mut companions = vec![pairing.scales];
        companions.extend(pairing.biases);
        planes.insert(at, companions);
    }
    Ok(planes)
}

/// **The key a FULLY RESIDENT load of this deployment forms** — see
/// [`Prospect::resident_key`].
///
/// A pure function of the trace, the recipe and the checkpoint: the layout it
/// hashes is `places(trace, &Plan::default())`, which is what an uncapped load
/// lays out, whatever THIS load's budgets are. So a capped load computes the
/// uncapped load's key exactly and names the file it wrote.
///
/// # Errors
///
/// [`Fault::Param`] for a param whose plane cannot be sized.
fn resident_key(trace: &Trace, landing: &LoadPlan, path: &Path) -> Result<u64> {
    let resident = places(trace, &crate::experts::Plan::default())?;
    let layout: Vec<(u64, u64, u64)> = resident
        .iter()
        .map(|place| (place.offset, place.bytes, place.reserved))
        .collect();
    let total = resident.last().map_or(0, |p| p.offset + p.reserved);
    // A plan that will not serialize is not a fault — it is a key this load
    // cannot form, and a load with no key neither reads nor writes the cache.
    // `0` is that answer, and no artifact is ever written under it because the
    // writer takes the same `None` path.
    let Ok(plan_json) = serde_json::to_vec(landing) else {
        return Ok(0);
    };
    Ok(crate::weight_cache::Identity {
        checkpoint: path,
        trace_name: &trace.name,
        plan_json: &plan_json,
        layout: &layout,
        total,
    }
    .key())
}

/// **THE KEY THIS DEPLOYMENT'S SERVING ARTIFACT CARRIES** (§M.3) — and
/// `None` for a load plan that will not serialize.
///
/// It used to be [`resident_key`]'s opposite in the one way that mattered: it
/// was **budget-DEPENDENT**, mixing this load's device layout, its host layout
/// and every param's rung, so a changed budget was a different key, a
/// different file, and a hundred gigabytes rewritten to say the same thing
/// about the same weights.
///
/// **None of that is here.** The artifact holds one image per plane in a
/// budget-free order and a boot cuts it, so what identifies the file is the
/// deployment and the sequence:
///
/// ```text
///   layout, total   what a FULLY RESIDENT load of this deployment lays out
///   images          (param, bytes, reserved) per image, in payload order
/// ```
///
/// The first line is `resident_key`'s own two fields, verbatim and for its
/// reason: they are the deployment's identity, a pure function of the trace
/// and the recipe. The second is what this file physically holds — the
/// ranking and every span in it — which is also a pure function of the trace
/// and the recipe (`Ranking::of` reads no budget). Two boots at two DIFFERENT
/// budgets therefore form the SAME key and read the SAME file, which is the
/// whole of §M.3.
///
/// # Errors
///
/// [`Fault::Param`] for a param whose plane cannot be sized.
fn tier_identity(
    trace: &Trace,
    landing: &LoadPlan,
    path: &Path,
    ranking: &crate::experts::Ranking,
) -> Result<Option<u64>> {
    let resident = places(trace, &crate::experts::Plan::default())?;
    let layout: Vec<(u64, u64, u64)> = resident
        .iter()
        .map(|place| (place.offset, place.bytes, place.reserved))
        .collect();
    // The offsets are left out on purpose: the images tile the payload
    // consecutively, so a span list in order states every offset it could
    // carry, and a key that mixed both would be mixing one fact twice.
    let images: Vec<(u64, u64, u64)> = ranking
        .images()
        .into_iter()
        .map(|(param, _at, bytes, reserved)| (param, bytes, reserved))
        .collect();
    // As `resident_key`: a plan that will not serialize is not a fault, it is
    // a key this load cannot form — and a load with no key neither reads nor
    // writes the artifact.
    let Ok(plan_json) = serde_json::to_vec(landing) else {
        return Ok(None);
    };
    Ok(Some(
        crate::weight_cache::tier::Identity {
            checkpoint: path,
            trace_name: &trace.name,
            plan_json: &plan_json,
            total: resident.last().map_or(0, |place| place.offset + place.reserved),
            layout: &layout,
            images: &images,
        }
        .key(),
    ))
}

/// **The serving artifact this deployment would name**, computed from the
/// outside — [`Prospect::resident_key`]'s twin, and `None` for a load plan
/// that does not serialize.
///
/// One metadata parse and one plan compile, reading no tensor bytes, exactly
/// as [`prospect`] costs — and it IS [`Prospect::tier_key`], reachable on its
/// own for a caller that holds neither. It no longer takes a
/// [`Plan`](crate::experts::Plan): the key stopped being a function of the
/// budgets when the artifact did (§M.3), so an operator naming a file and a
/// gate asserting what a boot wrote both ask the same question of the
/// deployment alone.
///
/// # Errors
///
/// [`Fault::Load`] for a checkpoint the contract does not fit,
/// [`Fault::Param`] for a param whose plane cannot be sized.
pub fn tier_key(trace: &Trace, contract: &ModelContract, path: &Path) -> Result<Option<u64>> {
    let metadata = if path.is_dir() {
        parse_metadata(path)?
    } else {
        zt::parse(path)?
    };
    let landing = compile(
        &metadata,
        contract,
        StorageTarget::for_backend(BackendKind::Cuda, 0, 1),
    )?;
    let index: BTreeMap<&str, usize> = trace
        .params
        .iter()
        .enumerate()
        .map(|(at, param)| (param.name.as_str(), at))
        .collect();
    let planes = attachments(&landing, &index)?;
    tier_identity(trace, &landing, path, &crate::experts::Ranking::of(trace, &planes)?)
}


/// **OPEN THIS DEPLOYMENT'S SERVING ARTIFACT AS A T2 SOURCE, OR ANSWER WHY
/// THERE IS NONE** (§K.6-T4, §M.3).
///
/// [`spill_source`]'s twin for the SECOND file, and the road that frees a
/// spilled deployment from the hundred-gigabyte resident snapshot that seeded
/// it: the serving artifact carries one image per plane of this trace, so it
/// answers for whatever this budget spills — and for whatever the NEXT budget
/// spills too, which is the difference §M made.
///
/// **THE KEY IS THE ADMISSION.** `key` is [`tier_key`]'s, which is
/// budget-FREE (§M.3): it mixes the deployment and the sequence, so a file
/// found under it describes this trace's images whatever rung a budget puts
/// them on. `Artifact::open` then checks the header's arithmetic, that the
/// images tile their own payload, the index digest and the key the filename
/// states.
///
/// **NOTHING IS HASHED HERE.** Whether the bytes are the bytes is
/// [`open_tiers`]'s question, asked once, of the same file — see its own doc
/// for why the answer is a verification up front and not a first-touch one.
///
/// `None` for a load offered no cache directory, a key it could not form, a
/// deployment that has never been imported, and every refusal
/// `Artifact::open` states.
#[must_use]
pub fn tier_spill(
    cache_dir: Option<&Path>,
    key: u64,
) -> Option<crate::weight_cache::tier::Artifact> {
    use crate::weight_cache::tier;

    if key == 0 {
        return None;
    }
    let artifact = tier::Artifact::open(&tier::path(cache_dir?, key)).ok()?;
    // **AND THE CONDITION THAT USED TO BE HERE IS GONE.** Format 2 also asked
    // whether the file's MAPPED SECTION was non-empty, because a file written
    // by a boot that spilled nothing carried no third image and was no source.
    // §M's file has no sections: it holds every plane of the trace, so any
    // artifact under this key is a source for any spill this deployment can
    // plan.
    (artifact.key() == key).then_some(artifact)
}

/// **OPEN THIS DEPLOYMENT'S SERVING ARTIFACT, OR SAY WHY THERE IS NONE**
/// (§K.4, §M.3).
///
/// [`spill_source`]'s twin for the OTHER file. It is asked before the pinned
/// tier is allocated, because whether that allocation pays for its
/// `write_bytes(0)` is a function of whether these bytes are on the disk
/// ([`Tier::open`](crate::experts::Tier::open)'s [`Fill`](crate::experts::Fill)),
/// and that question has to be asked before the allocation is made rather
/// than after.
///
/// # What an `Err` means, and it is three different things
///
/// ```text
///   absent    no file under this key. Since §M-3 that is NEWS: the key is
///             the filename, so a changed plan or recipe is a changed key,
///             and the census beside the refusal names the files that are
///             there instead of the one that is not
///   stale     another build's format — said out loud, with the remedy
///   rotten    a file under this key that does not describe itself, or whose
///             mapped images do not hash: counted, named, LEFT ALONE
/// ```
///
/// **AND NONE OF THEM DELETES ANYTHING** (§M.4). The file is the model on this
/// machine, not a cache of a boot: the boot that finds it wrong is not the
/// boot that can rewrite it, and the source it was imported from may not still
/// be here. Every refusal names `pie model import --prepare-only`;
/// [`tier::refuse`](crate::weight_cache::tier::refuse) is where that sentence
/// lives, and all three of these are built through it so that they have one
/// shape.
///
/// **AND NONE OF THEM RUNS A COLD LOAD EITHER** (§M-3). This used to answer
/// `Option` and every `None` meant "stream, transform, write". A serving load
/// has no such road left, so the answer is the SENTENCE — [`Miss`] — and the
/// caller decides where it goes: a `Fault::Residency` under `Intent::Serve`, a
/// printed line under `Intent::Prepare`, which is the run that then writes the
/// file the sentence is about.
///
/// # The refusal this door adds
///
/// **A FILE WHOSE IMAGES ARE NOT THIS TRACE'S.** The key mixes the sequence
/// (§M.3), so an index that disagrees with this boot's ranking cannot happen
/// without the key having been reused for another deployment. It is checked
/// anyway, image for image, because the alternative to a loud refusal here is
/// a pump of the wrong length onto a store of this one's — and it costs one
/// walk of an index already in hand.
///
/// **AND THE MAPPED-RUNG IMAGES ARE HASHED BEFORE A KERNEL IS POINTED AT
/// THEM.** The body says why that one is asked here, up front, and whole.
fn open_tiers(
    dir: &Path,
    key: u64,
    ranking: &crate::experts::Ranking,
    plan: &crate::experts::Plan,
    source: &Path,
) -> std::result::Result<crate::weight_cache::tier::Artifact, Miss> {
    use crate::weight_cache::Refused;
    use crate::weight_cache::tier;

    let path = tier::path(dir, key);
    let artifact = match tier::Artifact::open(&path) {
        Ok(artifact) => artifact,
        // **THE MISS THAT USED TO BE SILENT** (§M-3). While the cold path
        // existed this was the ordinary first boot and nobody needed told;
        // now it is the whole reason a deployment will not start, and the
        // sentence has to answer the question an operator actually has, which
        // is *why this key and not the one whose file is sitting right there*.
        // `absent` is that census.
        Err(Refused::Unreadable { .. } | Refused::NotAnArtifact) => {
            return Err(Miss::Absent(absent(dir, key, &path, source)));
        }
        // An operator who upgraded a build over a populated directory learns
        // why — and, since §M, what to run about it. Through `tier::refuse`
        // like every other refusal here, so the three read as one message.
        Err(Refused::StaleFormat { states, reads }) => {
            return Err(Miss::Refused(tier::refuse(
                &path,
                Some(source),
                &format!(
                    "states format {states} and this build reads {reads}, so its images \
                     cannot be cut by this one"
                ),
            )));
        }
        // A file under this key that does not describe itself: a header whose
        // arithmetic does not close, images that do not tile their own
        // payload, a name and a header that name different deployments. No
        // byte was hashed to find any of them, so this is where they are
        // counted.
        Err(other) => {
            tier::count_corrupt();
            return Err(Miss::Refused(tier::refuse(&path, Some(source), &format!("{other}"))));
        }
    };
    // **A FILE WHOSE ADAPTER BANKS WERE LIVE IS NOT A BOOT IMAGE.** The flag
    // states that the registered planes were zeros when the snapshot was
    // taken, which is what lets this file omit them and a restore leave the
    // store's own zeros in place. A file without it was written by something
    // that moved the call out of `Weights::resident`, and the right answer to
    // that is a refusal rather than a silent extra weight.
    if artifact.head().flags & tier::FLAG_ADAPTERS_ZEROED == 0 {
        return Err(Miss::Refused(tier::refuse(
            &path,
            Some(source),
            "does not state that its registered planes were zeros when it was written, \
             and a restore that left this store's zeros in place would be seating \
             whatever they held",
        )));
    }
    // ── **AND THE INDEX IS THIS BOOT'S RANKING, IMAGE FOR IMAGE.**
    let want = ranking.images();
    let holds = artifact.entries();
    if holds.len() != want.len() {
        tier::count_corrupt();
        return Err(Miss::Refused(tier::refuse(
            &path,
            Some(source),
            &format!(
                "carries {} plane images where this trace ranks {}; the key mixes the \
                 sequence (§M.3), so a file this far from the ranking that names it is a \
                 key that has been reused for another deployment",
                holds.len(),
                want.len(),
            ),
        )));
    }
    for (at, (param, offset, bytes, reserved)) in want.into_iter().enumerate() {
        let holds = holds[at];
        if u64::from(holds.id) != param
            || holds.offset != offset
            || holds.bytes != bytes
            || holds.reserved != reserved
        {
            tier::count_corrupt();
            return Err(Miss::Refused(tier::refuse(
                &path,
                Some(source),
                &format!(
                    "states image {at} as param {} at payload byte {} ({} of {} bytes) \
                     where this trace ranks param {param} at {offset} ({bytes} of \
                     {reserved}); the two were written from different traces",
                    holds.id, holds.offset, holds.bytes, holds.reserved,
                ),
            )));
        }
    }
    // ── AND THE MAPPED-RUNG IMAGES ARE HASHED BEFORE A KERNEL IS POINTED AT
    //    THEM (§K.5). The other rungs are verified AS THEIR BYTES CROSS,
    //    which is what keeps "always checksummed" from being a second pass
    //    over a hundred gigabytes. A T2 image never crosses: its bytes are
    //    read by the GPU, one page at a time, out of a mapping — there is no
    //    moment a reader could hash a page at, and no hook to hang a
    //    first-touch check on.
    //
    //    So they are hashed HERE, whole, once, and the cost is honest:
    //    parallel FNV chains over the images this budget spills, which on
    //    this SKU is a couple of seconds. It buys two things at that price.
    //    The first is §K.5 itself — *a silently-corrupt weight artifact
    //    produces garbage tokens with no error* — and the T2 images are the
    //    ones whose corruption no later check would catch. The second is
    //    free: the read that hashes them is a sequential NVMe pass that
    //    leaves them in the page cache, which is precisely the state the
    //    first fires would otherwise fault them into one page at a time.
    //
    //    **AND THE FILE SURVIVES A DISAGREEMENT** (§M.4). Format 2 deleted
    //    here, on the argument that a tier artifact was one this shell could
    //    re-form. It is the model now: an IMPORT rewrites it, and the
    //    decision to replace a hundred gigabytes belongs to the write that
    //    succeeds, not to the read that failed.
    let spilled: Vec<u32> = artifact
        .entries()
        .iter()
        .filter(|group| plan.mapped(usize::try_from(group.id).unwrap_or(usize::MAX)))
        .map(|group| group.id)
        .collect();
    if !spilled.is_empty()
        && let Err(why) = artifact.verify_entries(&spilled)
    {
        return Err(Miss::Refused(tier::refuse(&path, Some(source), &format!("{why}"))));
    }
    Ok(artifact)
}

/// **WHY THIS LOAD HAS NO SERVING ARTIFACT** — the sentence, and which kind of
/// news it is (§M-3).
///
/// Both variants carry a finished sentence, so nothing downstream formats
/// anything. What the variant decides is who SAYS it:
///
/// ```text
///   Absent    a prepare says nothing — it is about to write the file
///   Refused   a prepare SAYS IT, because "I am replacing a rotted 100 GiB
///             artifact" is the one line that run owes an operator
/// ```
///
/// A serving load refuses on either, with the sentence, and never prints from
/// here: it hands it to `Fault::Residency` and the runtime logs it once
/// against the load it refused.
enum Miss {
    /// No file under this key, and the census of what IS in the directory.
    Absent(String),
    /// A file this build will not read. Already counted at the door that
    /// found it.
    Refused(String),
}

impl Miss {
    /// The sentence, whichever kind of miss it is.
    fn why(&self) -> &str {
        match self {
            Miss::Absent(why) | Miss::Refused(why) => why,
        }
    }
}

/// **THE LOUD MISS** (§M-3): this key's file is not on the disk, said in a way
/// that distinguishes the two things that means.
///
/// The bug this closes is M-3's own finding. `tier_key` is the FILENAME, and
/// the key mixes the checkpoint, the load plan and the ranking — so an edit to
/// a model text, a new recipe, a rebuilt contract all produce a NEW key, and
/// before this wave the boot answered that by going cold beside a ~100 GiB
/// file it would never open again. Two files, twice the disk, and not one line
/// saying so.
///
/// So the refusal looks in the directory and answers the operator's real
/// question:
///
/// ```text
///   others found   the plan or the recipe changed. They are NAMED, with
///                  their sizes, because the follow-up question is always
///                  "then what is that hundred gigabytes" and the answer is
///                  "the previous recipe's, and nothing will read it again"
///   none found     this model was never prepared on this machine
/// ```
///
/// Four names at most: the list is a prompt to go and look, not an inventory,
/// and a cache directory with thirty deployments in it would bury the remedy
/// under its own census. They are newest-first ([`tier::others`]), so the four
/// shown are the ones most likely to be the recipe just replaced.
fn absent(dir: &Path, key: u64, path: &Path, source: &Path) -> String {
    use crate::weight_cache::tier;

    let others = tier::others(dir, key);
    let why = if others.is_empty() {
        format!(
            "is not there, and {dir:?} holds no serving artifact at all — this \
             deployment has never been prepared on this machine"
        )
    } else {
        let shown: Vec<String> = others
            .iter()
            .take(4)
            .map(|other| {
                let name = other
                    .file_name()
                    .map_or_else(|| other.display().to_string(), |name| {
                        name.to_string_lossy().into_owned()
                    });
                let bytes = other.metadata().map_or(0, |meta| meta.len());
                // GiB is the unit the disk cost is felt in and the unit every
                // other line here uses; a file under one is rendered in MiB
                // rather than as `0.0 GiB`, which reads as "empty" for a
                // number whose whole job is to be alarming.
                match bytes >= (1u64 << 30) {
                    true => format!("{name} ({:.1} GiB)", bytes as f64 / (1u64 << 30) as f64),
                    false => format!("{name} ({:.1} MiB)", bytes as f64 / (1u64 << 20) as f64),
                }
            })
            .collect();
        let rest = match others.len().saturating_sub(shown.len()) {
            0 => String::new(),
            more => format!(", and {more} more"),
        };
        format!(
            "is not there, and {} other serving artifact{} sit{} beside it in {dir:?}: \
             {}{rest}. The key is the filename and it is a function of the checkpoint, \
             the load plan and the ranking — so a changed model text, a changed recipe \
             or a changed contract is a changed key, and those files are this \
             deployment under a recipe this build no longer reads",
            others.len(),
            if others.len() == 1 { "" } else { "s" },
            if others.len() == 1 { "s" } else { "" },
            shown.join(", "),
        )
    };
    tier::refuse(path, Some(source), &why)
}

/// **THE MISS WITH NO KEY AT ALL** — the one refusal that is not about a file
/// (§M-3).
///
/// A streamed load reaches this when it was offered no weight cache directory,
/// or when the load plan would not serialize so no key could be formed
/// ([`tier_identity`]). Neither is a rotted artifact and neither has a
/// `pie model import` remedy on its own, so the sentence is its own rather
/// than [`tier::refuse`](crate::weight_cache::tier::refuse)'s shape — what an
/// operator has to change is the CONFIG, and it is named.
fn unkeyed(dir: Option<&Path>, source: &Path) -> Miss {
    Miss::Refused(match dir {
        None => format!(
            "engine-cuda: this load streams its expert banks and can only be served out \
             of a prepared serving artifact, and this deployment states no weight cache \
             directory to look in. Set `[model] weight_cache_dir` and run `{}`.",
            crate::weight_cache::tier::rebuild(Some(source)),
        ),
        Some(dir) => format!(
            "engine-cuda: this load streams its expert banks and can only be served out \
             of a prepared serving artifact under {dir:?}, and its load plan does not \
             serialize — so this build can form no key and can name no file. That is a \
             refusal about this BUILD and not about anything on the disk."
        ),
    })
}

/// **THE CUT: WHAT THIS BOOT'S BUDGETS MAKE OF A BUDGET-FREE ARTIFACT**
/// (§M.3).
///
/// The one place the file's ranking meets this load's plan. The artifact holds
/// one image per plane in an order no budget touches; `places` and `plan` say
/// where each of them goes THIS time, and this is that walk, done once, so
/// that the restore and its verification cannot disagree about which image is
/// on which rung.
///
/// ```text
///   pump    images the device store holds: [0 .. c1)
///   pinned  images the pinned tier holds:  [c1 .. c2)
///   mapped  images served where they lie:  [c2 ..  )
/// ```
///
/// The three cuts are RANK THRESHOLDS and not strict prefixes, because
/// `Plan::cut`'s walk goes on past a group too large for the tier it is being
/// offered (its own doc argues why). Nothing here needs them to be prefixes:
/// every image is addressed by its own index entry, so a rung is a SET of
/// entries and the ordering is what makes that set nearly contiguous rather
/// than what makes it addressable.
struct Cut {
    /// `(payload offset, bytes, store offset)` per image the store holds.
    pump: Vec<(u64, u64, u64)>,
    /// `(payload offset, span, first block, pinned offset)` per T1 image.
    pinned: Vec<(u64, u64, u64, u64)>,
    /// The params behind `pump`, for the digest that runs beside it.
    device_params: Vec<u32>,
    /// The params behind `pinned`, for the deferred seat's verify-in-place.
    pinned_params: Vec<u32>,
}

/// [`Cut`]'s walk. `Err` is a file that cannot be cut at all — which is a
/// claim about the FILE, because the plan and the ranking were checked against
/// each other in [`open_tiers`] before this ran.
fn cut_of(
    artifact: &crate::weight_cache::tier::Artifact,
    places: &[Place],
    plan: &crate::experts::Plan,
) -> std::result::Result<Cut, String> {
    let mut cut = Cut {
        pump: Vec::new(),
        pinned: Vec::new(),
        device_params: Vec::new(),
        pinned_params: Vec::new(),
    };
    // ── [0 .. c1): WHAT THE DEVICE STORE HOLDS.
    for entry in artifact.entries() {
        let param = usize::try_from(entry.id).unwrap_or(usize::MAX);
        let Some(place) = places.get(param) else {
            return Err(format!("carries an image for param {param}, which this trace has not"));
        };
        if place.reserved == 0 {
            continue;
        }
        // **A STREAMED BANK'S SLAB IS NOT PUMPED.** The store gives it
        // `resident` slots and `Tier::land` fills every one of them from the
        // pinned copy — one copy per slot, because a slot's stride and an
        // expert's stride are the same number only while the slots are the
        // first `resident` experts. Pumping the bank's prefix here would write
        // bytes that call overwrites, and would write them over the padding
        // `Buffer::zeroed` left, which is the one difference a cold boot's
        // store would still have from this one.
        if plan.resident(param).is_some() {
            continue;
        }
        if entry.bytes != place.bytes {
            return Err(format!(
                "states {} bytes for param {param} and this trace publishes {}",
                entry.bytes, place.bytes,
            ));
        }
        cut.pump
            .push((entry.offset, place.bytes.min(place.reserved), place.offset));
        cut.device_params.push(entry.id);
    }
    // ── [c1 .. c2): WHAT THE PINNED TIER HOLDS, at the tier's own offsets.
    for (param, host_at, _, reserved) in plan.host_layout() {
        let id = u32::try_from(param).unwrap_or(u32::MAX);
        let Some((entry, first_block)) = artifact.locate(id) else {
            return Err(format!("carries no image for param {param}, which this plan pins"));
        };
        if entry.reserved != reserved {
            return Err(format!(
                "gives param {param} a {}-byte image and the pinned tier seats it in \
                 {reserved}",
                entry.reserved,
            ));
        }
        cut.pinned.push((entry.offset, reserved, first_block, host_at));
        cut.pinned_params.push(id);
    }
    Ok(cut)
}

/// **SHOULD THIS WARM BOOT SERVE OUT OF THE FILE WHILE IT PAGE-LOCKS?**
/// (§L.1, phase L-1). The artifact to serve T1 from, or `None` for a boot that
/// makes its page-locked image up front the way every warm boot did.
///
/// # What it costs to say yes, and what it buys
///
/// A warm streamed boot of this SKU spends the great majority of its life in
/// two terms that produce no answer: `cudaHostAlloc` page-locking the whole
/// pinned image, and the read that fills it. Neither is needed to SERVE. A
/// packed select reads its plane bases out of a cell and dereferences them; it
/// cannot tell a page-locked address from a mapped one, and neither can a
/// dense bank's table entry. So the images this cut puts on T1 are verified
/// where they lie — once, before a kernel is pointed at them — and the tier
/// serves them out of the mapping. The first token arrives at meta + compile +
/// the device pump.
///
/// The window that follows is the honest cost and §L.5 states it: until the
/// background fill lands, every T1 plane a fire reads is an NVMe page fault
/// over HMM rather than a PCIe read out of page-locked memory. **This wave
/// buys the first token, not the throughput.**
///
/// # The three conditions, and why each is a condition
///
/// ```text
///   a warm artifact   there is nothing to serve out of without one
///   host_image > 0    a plan that seats nothing on T1 has nothing to defer
///   pageable_access   without HMM a mapped host pointer is not a device
///                     pointer at all — the same attribute T2 stands on
/// ```
///
/// **A DEVICE WITHOUT THE ATTRIBUTE FALLS BACK, IT IS NOT REFUSED** (§L,
/// hazard H4). `Tier::open`'s T2 arm refuses instead, and the difference is
/// which promise is being kept: a plan that SPILLED cannot be served at all
/// without the attribute, while this is a boot-time optimization over a road
/// that still works. A policy, not a refusal.
///
/// The artifact is opened a second time here, and deliberately — `resident`'s
/// own note on the T2 source argues the shape: `warm` is borrowed by the
/// restore and dropped at the end of the load, while THIS mapping has to
/// outlive it, because every T1 address the tier hands out points inside it
/// until the install. Two `mmap`s of one inode share their pages, so the
/// second costs a file descriptor and an address range.
fn defer_tiers(
    warm: Option<&crate::weight_cache::tier::Artifact>,
    plan: &crate::experts::Plan,
) -> Option<crate::weight_cache::tier::Artifact> {
    use crate::weight_cache::tier;

    let warm = warm?;
    if plan.host_image() == 0 || !crate::experts::pageable_access() {
        return None;
    }
    let artifact = tier::Artifact::open(warm.path()).ok()?;
    // The same comparison `open_tiers` makes and for the same reason: this is
    // a SECOND open of a path, and a file that changed under the boot between
    // the two is a file this tier must not seat itself over. Nothing is
    // counted here — the door that hashed the bytes owns that — and a
    // disagreement simply means the eager road. Every image this seat will
    // actually serve is checked one at a time by `Tier::open`'s deferred arm.
    (artifact.key() == warm.key()).then_some(artifact)
}

/// **Why a warm streamed boot did not get its images.**
///
/// It said "fell through to the cold load" until §M-3, and there is nothing to
/// fall through to: a `Serve` refuses on either of these and only a `Prepare`
/// goes on to write the file. What the distinction still decides is what the
/// operator is TOLD, which is the whole reason it is two variants and not one
/// string: bytes that do not hash to what the block table states are a file
/// that will be refused identically on every boot until somebody rebuilds it,
/// and a device or a disk that stopped answering is not the file's doing at
/// all — telling that operator to re-import would send them to fix the one
/// thing that is not broken. **Neither of them deletes it** (§M.4).
enum Rotten {
    /// The bytes are not the bytes. Counted at the door that hashed them —
    /// [`Artifact::verify_entries`](crate::weight_cache::tier::Artifact::verify_entries)
    /// and [`read_spans_into`](crate::weight_cache::tier::read_spans_into)
    /// each count what they find — and named here with the remedy.
    Bytes(String),
    /// The machine could not do it. Said out loud.
    Machine(String),
}

/// **The pinned tier's read targets, as something a scope thread may carry.**
///
/// A `*mut u8` is not `Send` and must not be, because nothing about a raw
/// pointer says who else is holding it. What makes THESE sound to move is the
/// sentence at their one construction site: the allocation is the tier's, the
/// tier belongs to a `Weights::resident` that has not returned, the spans are
/// disjoint because `Plan::host_layout` tiles the image, and the only other
/// thread in that scope writes the device store.
struct Destinations(Vec<crate::weight_cache::tier::Span>);

// SAFETY: as above — a list of disjoint spans of one allocation, moved once,
// into a thread that is their sole writer for as long as the scope is open.
unsafe impl Send for Destinations {}

/// **FILL THIS LOAD'S TIERS OUT OF THE ARTIFACT** (§K.4), or say why not.
///
/// Answers `None` when every image the cut names crossed AND verified, which
/// is the only answer that lets the caller skip the executor. Every `Some` is
/// the refusal's sentence — and, under [`Intent::Prepare`], leaves the store
/// zeroed and the pinned tier zeroed, the state a cold load starts from, so
/// the landing that follows starts from where it would have started.
///
/// # The shape
///
/// ```text
///   T1      one positioned read per image, straight into the pinned tier,
///           each hashing its own blocks from the bytes as they land
///   T0      one staged transfer per image, at the store's own offsets, with
///           the block digests running beside them over the same mapping
/// ```
///
/// **THE DEVICE ARM IS A LIST OF TRANSFERS NOW AND NOT ONE.** Format 2's
/// device section WAS the store's image, so a restore pumped it back as a
/// single copy at offset zero. §M's payload is the ranking's order and the
/// store's layout is `places`' — two orders, and the store's is the one that
/// moves with the budget — so the restore names each image's destination
/// itself. It costs nothing: `Lanes::pump` splits every transfer into 2 MiB
/// chunks and round-robins the lot across its lanes, so a hundred transfers
/// and one transfer of the same bytes are the same queue.
///
/// # And the deferred shape, which is the same shape (§L.3)
///
/// A tier opened over the mapping ([`Tier::deferring`](
/// crate::experts::Tier::deferring)) has nowhere to read its T1 images INTO,
/// so the host arm hashes them WHERE THEY LIE — `verify_entries` in the same
/// position, in the same scope, beside the same device pump. Everything else
/// about this function is unchanged, including both recoveries.
///
/// **Verify-first, and the alternative was rejected by name.** Trusting the
/// mapping and checking it lazily has no place to hang the check: nothing
/// hooks a first touch, the GPU faults each page in on its own, and the blast
/// radius of a wrong byte is tokens that have already been handed to a caller.
/// `open_tiers` wrote that argument for the mapped rung and it is the same
/// argument here.
///
/// Nothing is staged on the host for T1 — the read target IS the pinned tier —
/// and nothing is staged for T0 beyond the pump's own lanes. **Both
/// verifications happen as the bytes cross**, which is what keeps "always
/// verified" (§K.5) from being a second pass over 100 GiB.
///
/// # Why the host arm goes first
///
/// Because it is the one whose failure has an ordering consequence. The tier
/// was allocated WITHOUT its memset on the strength of this call covering
/// every byte of it, and reading it first means the window in which that
/// promise is outstanding is as short as the file allows.
///
/// # And what a refusal is now (§M-3)
///
/// It answers the SENTENCE rather than `false`, for [`open_tiers`]' reason:
/// there is no cold serving path left, so a serving caller turns this into a
/// `Fault::Residency` and only a prepare goes on to write the file. `None` is
/// a restore that happened.
///
/// **AND A SERVING REFUSAL PAYS NO RECOVERY.** The zeroing below exists so
/// that the load which CONTINUES starts where a cold boot starts; a load that
/// is about to return an error continues nowhere, its `Buffer` and its `Tier`
/// are dropped on the way out, and `Tier::undefer` in particular would
/// page-lock tens of gigabytes for a `Weights` that will never exist. So it is
/// spent under [`Intent::Prepare`] and skipped under [`Intent::Serve`].
///
/// # Errors
///
/// [`Fault::Device`] only, and only for the zeroing a refusal owes the store.
/// A refusal is not an error here: a prepare has a way to produce these bytes,
/// and taking it is what a `Some` lets it do.
fn restore_tiers(
    artifact: &crate::weight_cache::tier::Artifact,
    source: &Path,
    places: &[Place],
    store: &mut Buffer,
    tier: &mut crate::experts::Tier,
    intent: Intent,
) -> Result<Option<String>> {
    use crate::weight_cache::tier;

    let deferring = tier.deferring();
    match fill_tiers(artifact, places, store, tier) {
        Ok(()) => {
            tier::count_restored();
            // ── **AND THE WINDOW OPENS HERE, NOT EARLIER** (§L.4). The
            //    background fill is armed AFTER the scope above has closed,
            //    which is after `Lanes::standard()` has opened its own pinned
            //    buffers and freed them again. Two reasons, and the charter
            //    names the first: a `cudaHostAlloc` over the whole image holds
            //    the runtime's memory-manager lock for tens of seconds, and a
            //    lane pool trying to allocate behind it waits (hazard H1). The
            //    second is the disk — the pump and the host verify above are
            //    the road to the first token, and a third reader on the same
            //    NVMe is bandwidth taken off it.
            if deferring {
                tier::count_deferred();
                arm_refill(tier);
            }
            Ok(None)
        }
        Err(rotten) => {
            // **TWO SENTENCES, AND ONLY ONE OF THEM IS ABOUT THE FILE.** A
            // rotted image is `tier::refuse`'s shape like every other refusal
            // an artifact earns; a read the MACHINE would not do is not the
            // artifact's fault and must not tell an operator to rebuild it —
            // it names the file, says what failed, and stops.
            let why = match &rotten {
                Rotten::Bytes(why) => tier::refuse(artifact.path(), Some(source), why),
                Rotten::Machine(why) => format!(
                    "engine-cuda: the serving artifact {:?} could not be read back \
                     ({why}). The file is left exactly where it is: this is a device or \
                     a disk that stopped answering rather than an artifact that rotted, \
                     and rebuilding it would fix nothing.",
                    artifact.path(),
                ),
            };
            if intent == Intent::Serve {
                return Ok(Some(why));
            }
            // **WHATEVER CROSSED IS THROWN AWAY**, whichever arm failed. The
            // store goes back to `Buffer::zeroed`'s state for
            // `weight_cache::restore`'s reason — a half-filled store is not a
            // state the cold load has ever started from — and the pinned tier
            // goes back to it because the allocation skipped its memset on
            // this call's promise (`Fill::Restored`). Zeroing a tier that was
            // in fact filled correctly, because the OTHER arm rotted, costs
            // one memset on the rarest path there is; the alternative is a
            // second piece of state to reason about.
            //
            // **AND A DEFERRED SEAT OWES MORE THAN A MEMSET** (§L.3): it has
            // no allocation at all, and no verified mapping left to serve out
            // of. `Tier::undefer` makes the page-locked image the deferred arm
            // declined to make, zeroed, which is the state the prepare's cold
            // landing below starts from.
            store.zero_span(0, store.bytes())?;
            match deferring {
                true => tier.undefer()?,
                false => tier.zero_host(),
            }
            Ok(Some(why))
        }
    }
}

/// [`restore_tiers`] without the counting or the zeroing — so that every
/// refusal below is one `?` and the recovery is stated once.
fn fill_tiers(
    artifact: &crate::weight_cache::tier::Artifact,
    places: &[Place],
    store: &mut Buffer,
    tier: &crate::experts::Tier,
) -> std::result::Result<(), Rotten> {
    use crate::weight_cache::Refused;
    use crate::weight_cache::tier;

    let head = artifact.head();
    let cut = cut_of(artifact, places, tier.plan()).map_err(Rotten::Bytes)?;
    // ── T1'S DESTINATIONS. `Plan::host_layout` tiles the pinned image
    //    exactly — every bank's padded span, then every pinned plane's — so
    //    the spans below cover `0..Pinned::bytes` with no gap and no overlap,
    //    and every byte of the allocation is written by this call.
    // A DEFERRED SEAT HAS NO ALLOCATION TO MEASURE, and `Tier::open` already
    // measured the thing it has instead: it resolved every one of these images
    // against the file, or it refused.
    let seated = tier.deferred_image();
    let host = tier.host();
    let covers: u64 = cut.pinned.iter().map(|(_, span, _, _)| *span).sum();
    if seated.is_none() && covers != host.bytes() as u64 {
        return Err(Rotten::Bytes(format!(
            "answers for {covers} pinned bytes and this boot allocated {}",
            host.bytes(),
        )));
    }
    let into = Destinations(
        cut.pinned
            .iter()
            .map(|(at, span, first_block, host_at)| tier::Span {
                at: *at,
                len: *span,
                first_block: *first_block,
                // SAFETY: `host_at + span <= host.bytes()`, because the spans
                // tile the allocation and the sum was just checked against its
                // length.
                into: unsafe {
                    host.host().add(usize::try_from(*host_at).unwrap_or(usize::MAX))
                },
            })
            .collect(),
    );

    // ── T0'S. One transfer per image the store holds, at the store's own
    //    offsets, out of the mapping.
    let base = store.at(0).map_err(|why| Rotten::Machine(format!("{why}")))?;
    let payload = artifact.payload();
    let mut transfers = Vec::with_capacity(cut.pump.len());
    for (at, len, store_at) in &cut.pump {
        let from = usize::try_from(*at).unwrap_or(usize::MAX);
        let want = usize::try_from(*len).unwrap_or(usize::MAX);
        let Some(span) = payload.get(from..from.saturating_add(want)) else {
            return Err(Rotten::Bytes(format!(
                "maps {} payload bytes and an image wants {from}..{}",
                payload.len(),
                from + want,
            )));
        };
        // The one bounds check on this path: the pump takes device addresses,
        // so the span meets the store's length here rather than at
        // `Buffer::write`'s door, which this arm does not go through.
        store.at(store_at.saturating_add(*len)).map_err(|why| {
            Rotten::Machine(format!("an image does not fit the store: {why}"))
        })?;
        transfers.push(crate::staged_h2d::Transfer {
            dst: base + store_at,
            src: span.as_ptr(),
            len: *len,
        });
    }

    // ── AND THE TWO ARMS RUN AT THE SAME TIME.
    //
    // **BECAUSE NEITHER OF THEM IS WAITING ON A DISK.** Both are verified as
    // they arrive, one FNV chain per 64 MiB block, and an FNV-1a chain is a
    // serial multiply per byte — call it a gigabyte a second on one core,
    // which is well under what the NVMe behind either read will give. Run one
    // after the other and the machine spends half its time with idle cores and
    // a queue-depth of nothing; run them together and the whole restore costs
    // what its longer arm costs. The two touch nothing in common: one writes
    // page-locked host memory through positioned reads, the other writes
    // device memory through the staged pump's own lanes.
    let (read, moved, verified) = std::thread::scope(|scope| {
        // SAFETY: the spans in `into` are disjoint windows on the tier's own
        // allocation, they tile it, and this load is inside
        // `Weights::resident` — so no kernel has been enqueued against the
        // tier, no guest exists, and the landing sink that writes it has not
        // run. Nothing else in this scope names a byte of it: the device arm
        // below writes the STORE.
        let reading = scope.spawn(move || match seated {
            // ── **VERIFIED WHERE THEY LIE** (§L.3). There is nowhere to read
            //    them into: the tier is seated over these very images and the
            //    kernels will fault their pages in themselves. So the same
            //    chains hash the same bytes in the same position in this
            //    scope, and what the answer decides is the same thing —
            //    whether the load may go on. The read that hashes them also
            //    leaves them in the page cache, which is exactly the state the
            //    first fires would otherwise fault them into one page at a
            //    time.
            //
            //    AND IT IS THE TIER'S OWN MAPPING THAT IS HASHED, not this
            //    restore's view of the path — `Tier::deferred_image` argues
            //    the difference, and it is the difference between promising
            //    that the FILE was checked and promising that the BYTES SERVED
            //    were.
            Some(seated) => seated.verify_entries(&cut.pinned_params),
            None => {
                let into = into;
                let blocks = artifact.blocks();
                unsafe { tier::read_spans_into(artifact.path(), &head, blocks, &into.0) }
            }
        });
        let (moved, verified) = match transfers.is_empty() {
            true => (Ok(()), Ok(Ok(()))),
            false => {
                let mut lanes = match crate::staged_h2d::Lanes::standard() {
                    Ok(lanes) => lanes,
                    Err(why) => return (reading.join(), Err(why), Ok(Ok(()))),
                };
                // **THE DEVICE DIGESTS RUN BESIDE THEIR OWN LANES**, over the
                // same mapping the lanes are faulting in, exactly as the
                // resident restore's does.
                std::thread::scope(|inner| {
                    let hashing = inner.spawn(|| artifact.verify_entries(&cut.device_params));
                    let moved = lanes.pump(&transfers);
                    (moved, hashing.join())
                })
            }
        };
        (reading.join(), moved, verified)
    });

    // The host arm first, whichever failed: it is the one whose destination
    // the allocation skipped a memset for, so its answer is the one a reader
    // of this function is looking for.
    match read {
        Ok(Ok(())) => {}
        Ok(Err(Refused::IndexCorrupt { why })) => return Err(Rotten::Bytes(why)),
        Ok(Err(other)) => return Err(Rotten::Machine(other.to_string())),
        Err(_) => return Err(Rotten::Machine("a host reader panicked".to_string())),
    }
    moved.map_err(|why| Rotten::Machine(format!("staged upload failed: {why}")))?;
    match verified {
        Ok(Ok(())) => Ok(()),
        Ok(Err(why)) => Err(Rotten::Bytes(why.to_string())),
        Err(_) => Err(Rotten::Machine("a digest worker panicked".to_string())),
    }
}

/// **ARM THE BACKGROUND FILL BEHIND A DEFERRED SEAT** (§L.4).
///
/// Spawns the thread, hands the tier its end of the channel, and answers
/// nothing: a fill that cannot be armed is a seat that serves out of the
/// mapping for its whole life, which is slower and is not wrong. Every refusal
/// is a line an operator can read.
///
/// **WHAT CROSSES, AND IT IS THE WHOLE LIST**: a `PathBuf`, a `Head`, the
/// block table, the span list this seat's cut produced, and a device ordinal.
/// Not the tier, not the mapping, not the artifact — the thread re-opens the
/// file by path and reads it with positioned reads, so nothing on this side is
/// aliased and no lifetime has to be argued across a spawn.
fn arm_refill(tier: &mut crate::experts::Tier) {
    // The tier's OWN mapping, which is the one the arm above just verified —
    // so the digests this thread reads the file back against are the digests
    // of the bytes this seat is serving. See `Tier::deferred_image`.
    let Some(image) = tier.deferred_image() else {
        return;
    };
    let path = image.path().to_path_buf();
    let head = image.head();
    let blocks = image.blocks().to_vec();
    // The spans in the PAGE-LOCKED image's own coordinates: `host_at` is the
    // offset the install will re-form every address from, so the copy this
    // thread builds is laid out exactly as `Tier::reseat` expects it.
    let mut spans: Vec<(u64, u64, u64, u64)> = Vec::new();
    for (param, host_at, _, reserved) in tier.plan().host_layout() {
        let id = u32::try_from(param).unwrap_or(u32::MAX);
        let Some((entry, first_block)) = image.locate(id) else {
            eprintln!(
                "engine-cuda: the deferred tier's fill cannot find param {param} in \
                 {path:?}; it will serve out of the mapping for the life of this load"
            );
            return;
        };
        spans.push((entry.offset, reserved.min(entry.reserved), first_block, host_at));
    }
    let bytes = usize::try_from(tier.plan().host_image()).unwrap_or(usize::MAX);
    // `cudaSetDevice` is per-thread and does not travel with a spawn, so the
    // ordinal is read HERE — on the thread the shell bound — and carried.
    let ordinal = match crate::device::ctx::current() {
        Ok(ordinal) => ordinal,
        Err(why) => {
            eprintln!(
                "engine-cuda: the deferred tier cannot name its device ({why}); it will \
                 serve out of {path:?} for the life of this load"
            );
            return;
        }
    };
    let (send, filled) = std::sync::mpsc::channel();
    match std::thread::Builder::new()
        .name("pie-tier-refill".to_string())
        .spawn(move || refill(&path, &head, &blocks, &spans, bytes, ordinal, &send))
    {
        Ok(filling) => tier.arm_refill(filling, filled),
        Err(why) => eprintln!(
            "engine-cuda: the deferred tier's fill thread would not start ({why}); it \
             will serve out of the artifact for the life of this load"
        ),
    }
}

/// **THE FILL, ON ITS OWN THREAD**: bind, map, read, verify, page-lock, send.
///
/// The order is the order, and §L-3 moved one term of it: the mapping comes
/// before the open so that the bytes have somewhere to land, and the
/// PAGE-LOCK comes last, after they have landed and hashed. What that buys is
/// [`Pinning`](crate::device::Pinning)'s subject — a `cudaHostAlloc` of this
/// size holds the runtime's memory-manager lock against every other thread for
/// tens of seconds, and this thread is armed while the boot still has work to
/// do. `read_spans_into` does the reading and the hashing in one pass — one
/// FNV chain per [`TIER_BLOCK`](crate::weight_cache::tier::TIER_BLOCK) AS THE
/// BYTES LAND IN THIS ALLOCATION, which is a stronger claim than re-hashing
/// the mapping would be: it verifies the copy the tier is about to serve out
/// of and not a second view of the file. So the §L.3 corollary's "the
/// background copy does not trust the boot's verify" is not a second pass
/// bolted on here; it is what this call already is.
///
/// **A FAILURE SENDS NOTHING**, which is how [`Refill`](crate::experts) hears
/// it. Bytes that do not hash to the table are a claim about the FILE: counted
/// at the door that hashed them, named here with the remedy, and **the file is
/// left where it is** (§M.4) — the seat keeps serving the verified mapping it
/// has been serving all along, which is a performance failure and not a
/// correctness one.
fn refill(
    path: &Path,
    head: &crate::weight_cache::tier::Head,
    blocks: &[u64],
    spans: &[(u64, u64, u64, u64)],
    bytes: usize,
    ordinal: i32,
    out: &std::sync::mpsc::Sender<crate::device::Pinned>,
) {
    use crate::weight_cache::Refused;
    use crate::weight_cache::tier;

    if let Err(why) = crate::device::ctx::bind_thread(ordinal) {
        eprintln!("engine-cuda: the deferred tier's fill cannot bind device {ordinal} ({why})");
        return;
    }
    // **UNINITIALIZED, AND THE PROMISE IS KEPT ONE LINE BELOW.**
    // `read_spans_into` writes every span WHOLE — padding included, its own
    // doc states it — and these spans tile this mapping. Nothing reads a byte
    // of it before then: it is on this thread's stack and unreachable from the
    // tier until it crosses the channel, which is a narrower window than the
    // boot restore's.
    //
    // **AND IT IS NOT PAGE-LOCKED YET, WHICH IS THE WHOLE OF §L-3**
    // ([`Pinning`](crate::device::Pinning) measures it). A `cudaHostAlloc`
    // over sixty gigabytes holds the runtime's memory-manager lock for its
    // whole length and every other CUDA call on every other thread waits
    // behind it — the load's own remaining allocations, and the fires this
    // seat exists to serve during the window. So the image is mapped, filled
    // by the read below, and page-locked in ONE call at the end, by which time
    // the boot that armed this thread has finished. Measured at qwen4's
    // 60.2 GiB: a 3.4 s register where the allocation it replaces was ~36 s.
    let host = match crate::device::Pinning::uninit(bytes) {
        Ok(host) => host,
        Err(why) => {
            eprintln!(
                "engine-cuda: the deferred tier's fill could not map {bytes} bytes \
                 ({why}); the seat serves out of {path:?} for the life of this load"
            );
            return;
        }
    };
    // SAFETY: `host` is a mapping of exactly `bytes`, which is
    // `Plan::host_image`; the spans below are that plan's own `host_layout`,
    // which tiles it; and the mapping was made on this thread one line above,
    // has been handed to nobody, and no kernel, guest or other thread can name
    // a byte of it until it is sent.
    let spans: Vec<tier::Span> = spans
        .iter()
        .map(|(at, len, first_block, host_at)| tier::Span {
            at: *at,
            len: *len,
            first_block: *first_block,
            into: unsafe { host.host().add(usize::try_from(*host_at).unwrap_or(usize::MAX)) },
        })
        .collect();
    // SAFETY: as above.
    match unsafe { tier::read_spans_into(path, head, blocks, &spans) } {
        // **THE LOCK IS TAKEN OVER BYTES THAT ARE ALREADY VERIFIED**, which is
        // the other half of the reordering: a fill whose digests disagree
        // never page-locks anything at all, where the old order paid for the
        // whole allocation before it knew whether the file was worth one.
        Ok(()) => match host.lock() {
            Ok(host) => {
                let _ = out.send(host);
            }
            Err(why) => eprintln!(
                "engine-cuda: the deferred tier's fill could not page-lock {bytes} bytes \
                 ({why}); the seat serves out of {path:?} for the life of this load"
            ),
        },
        // **THE ONE REFUSAL WITH NO LOAD TO REFUSE.** This runs on the thread
        // §L armed, behind a seat that is already serving out of the verified
        // mapping, so the file is named, the remedy is named, and the fill
        // stops — see `tier::rebuild`'s `None` for why the command is spelled
        // with a slot in it here and with a path everywhere else.
        Err(Refused::IndexCorrupt { why }) => eprintln!("{}", tier::refuse(path, None, &why)),
        Err(other) => eprintln!(
            "engine-cuda: the deferred tier's fill could not read {path:?} back ({other}); \
             the seat serves out of the mapping and the file is left alone"
        ),
    }
}

/// **WRITE THIS DEPLOYMENT'S SERVING ARTIFACT** — one image per plane of the
/// ranking, whatever rung this boot happened to put it on (§M.3).
///
/// Called once, from inside [`Weights::resident`], after the executor has run
/// and before the tier is landed. Best effort in every direction: the load
/// already succeeded and the images this reads from are the answer, so every
/// refusal below is a counted line rather than a fault.
///
/// # The peak this does not add
///
/// No image is ever held. Each is streamed to the file in
/// [`TIER_BLOCK`](crate::weight_cache::tier::TIER_BLOCK) pieces, out of
/// whichever rung this boot put it on: `Buffer::read` for a plane in the
/// store, `Pinned::view` for one on T1 — a window on the page-locked bytes,
/// where `Pinned::read` would have allocated a second copy of a tier that may
/// be sixty gigabytes — and a window on the source mapping for one on T2.
///
/// # What the file says, and what THIS boot has to do with it
///
/// The index is [`Ranking::images`](crate::experts::Ranking::images) verbatim:
/// the sequence, hottest first, then the dense routed banks whole. **Not one
/// entry of it is a function of the budgets** — which is the point, and the
/// difference from every format before 3. What the budgets decide is only
/// where this writer READS each image from, and the three rungs hold the same
/// bytes (§M.3's measured fact), so two boots at two different budgets write
/// the same file.
///
/// **A BANK IS WRITTEN WHOLE.** Its image is every expert, read out of the
/// pinned tier — which holds all of them authoritatively — and never out of
/// the device slab, which holds `resident` of them and is a cache over it.
///
/// **AND THE REGISTERED PLANES ARE NOT WRITTEN AT ALL.**
/// [`FLAG_ADAPTERS_ZEROED`](crate::weight_cache::tier::FLAG_ADAPTERS_ZEROED)
/// is what states that, and the ordering that makes it true is structural
/// rather than checked: this runs INSIDE `Weights::resident`, which has not yet
/// returned a `Weights` — and `register_adapter` is a method on one. So no
/// registration can have happened, every adapter bank still holds what
/// `Buffer::zeroed` left, and a restore leaves this store's own zeros in place
/// rather than reading a hundred megabytes of them back. A caller that moves
/// this call out of the constructor owes the flag a new argument or must stop
/// setting it.
fn write_tiers(
    dir: Option<&Path>,
    key: u64,
    trace: &Trace,
    landed: &[bool],
    ranking: &crate::experts::Ranking,
    places: &[Place],
    tier: &crate::experts::Tier,
    store: &Buffer,
) {
    use crate::weight_cache::Group;
    use crate::weight_cache::tier;

    // **NOTHING IS WRITTEN FOR A LOAD THAT DID NOT LAND.** The refusal for an
    // unpublished plane is thirty lines below this call, in the table build,
    // and it is the load's own answer; what must not happen in between is a
    // FILE under this key describing an image with a hole in it — because the
    // warm boot that reads it skips the executor, and therefore skips the
    // check that would have caught the hole. Same condition as the table's,
    // said here first: a registered plane is one the checkpoint does not have.
    if let Some(at) = (0..trace.params.len())
        .find(|at| !landed[*at] && trace.params[*at].source == ParamSource::Checkpoint)
    {
        tier::decline(&format!(
            "`{}` is a plan param the load contract never published, so this load is \
             about to be refused and its images describe a model with a hole in it",
            trace.params[at].name,
        ));
        return;
    }

    let entries: Vec<Group> = ranking
        .images()
        .into_iter()
        .map(|(param, at, bytes, reserved)| Group {
            id: u32::try_from(param).unwrap_or(u32::MAX),
            plane: 0,
            offset: at,
            bytes,
            reserved,
        })
        .collect();
    tier::store(dir, key, &entries, tier::FLAG_ADAPTERS_ZEROED, |param, at, into| {
        let param = usize::try_from(param).unwrap_or(usize::MAX);
        // The rungs are asked in the order that makes each answer a whole
        // image: T1 holds a bank's every expert and a pinned plane whole, the
        // mapping holds a spilled plane whole, and the store holds everything
        // else — where `place.bytes` is what the checkpoint published into it.
        if let Some(host_at) = tier.host_offset(param) {
            let span = tier
                .host()
                .view(host_at + at, into.len() as u64)
                .ok_or_else(|| {
                    format!("the pinned tier holds no bytes for param {param} at {at}")
                })?;
            into.copy_from_slice(span);
            return Ok(());
        }
        if let Some(bytes) = tier.mapped_plane(param) {
            let from = usize::try_from(at).unwrap_or(usize::MAX);
            let span = bytes
                .get(from..from.saturating_add(into.len()))
                .ok_or_else(|| format!("param {param} is short of {at} on the mapping"))?;
            into.copy_from_slice(span);
            return Ok(());
        }
        let place = places
            .get(param)
            .ok_or_else(|| format!("param {param} is not a plane of this trace"))?;
        store
            .read(place.offset + at, into)
            .map_err(|why| format!("reading the device store for param {param} at {at}: {why}"))
    });
}

/// The transform arena's backing: RAM when the machine has room for it beside
/// everything this load has already spent, a file-backed map when it does not
/// — see [`Scratch::fitting`] and the construction site in
/// [`Weights::resident`] for the argument.
enum Scratch {
    Ram(Vec<u8>),
    Disk(SpillArena),
}

impl Scratch {
    /// `arena` bytes of scratch, spilled to disk when RAM will not hold it
    /// with a safety share left for the rest of the load.
    ///
    /// **THE PINNED TIER IS NOT AN ARGUMENT HERE, BECAUSE IT IS ALREADY
    /// SPENT.** The doc that stood here said this weighed the arena against
    /// "the pinned bytes the tiers are about to lock", and the call site has
    /// never passed those: `Tier::open` runs well above this line and its
    /// `cudaHostAlloc` has already moved `MemAvailable` and this cgroup's
    /// `memory.current` by the whole of T1. So [`available_memory`] READS the
    /// tier rather than being told about it, and adding it to `need` would
    /// have charged the same gigabytes twice and spilled arenas that fit.
    ///
    /// What the second argument really is: the bytes this load put on the
    /// MAPPED tier — the ones no allocation has accounted for, because they
    /// are a file the kernel will page in behind the executor. Charged as
    /// headroom, which is what the page cache holding them costs.
    fn fitting(arena: usize, mapped: u64) -> Result<Scratch> {
        let need = arena as u64 + mapped + (2 << 30);
        if need <= available_memory() {
            return Ok(Scratch::Ram(vec![0u8; arena]));
        }
        eprintln!(
            "engine-cuda: the load's {arena}-byte transform arena does not fit \
             what is left of this machine's memory beside its {mapped} mapped \
             bytes; spilling the arena to disk"
        );
        SpillArena::new(arena).map(Scratch::Disk)
    }

    fn as_mut(&mut self) -> &mut [u8] {
        match self {
            Scratch::Ram(vec) => vec.as_mut_slice(),
            Scratch::Disk(map) => map.as_mut(),
        }
    }
}

/// What the machine will actually give this process: the tighter of the
/// kernel's `MemAvailable` and this cgroup's remaining allowance. Zero is
/// never answered — a machine whose accounting cannot be read gets the RAM
/// arena and the failure mode that has always existed.
fn available_memory() -> u64 {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let rest = line.strip_prefix("MemAvailable:")?;
                let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
                Some(kb * 1024)
            })
        });
    let cgroup = || -> Option<u64> {
        let max: u64 = std::fs::read_to_string("/sys/fs/cgroup/memory.max")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        let current: u64 = std::fs::read_to_string("/sys/fs/cgroup/memory.current")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        Some(max.saturating_sub(current))
    }();
    match (meminfo, cgroup) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => u64::MAX,
    }
}

/// A writable file-backed map, sized once and unlinked on drop — the
/// transform arena's disk spelling. `MAP_SHARED` over a temporary file is
/// what makes the dirty pages the KERNEL's problem: it writes them back and
/// reclaims under memory pressure, where an anonymous map of the same size
/// is unreclaimable and dies by OOM instead.
struct SpillArena {
    at: *mut u8,
    len: usize,
}

// SAFETY: `at` is a private MAP_SHARED mapping this struct alone owns; the
// file behind it is unlinked at creation, so no other process can reach it.
unsafe impl Send for SpillArena {}

impl SpillArena {
    fn new(len: usize) -> Result<SpillArena> {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("pie-arena-{}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|why| Fault::Load(checkpoint::error::Error::Checkpoint(format!(
                "the arena spill file {} does not open: {why}",
                path.display()
            ))))?;
        // Unlinked immediately: the mapping keeps the storage alive, and a
        // crashed load leaves no 100 GiB file behind.
        let _ = std::fs::remove_file(&path);
        file.set_len(len as u64).map_err(|why| {
            Fault::Load(checkpoint::error::Error::Checkpoint(format!(
                "the arena spill file does not grow to {len} bytes: {why}"
            )))
        })?;
        // SAFETY: a fresh shared mapping over a file this fn just created and
        // sized; length and protections are stated, fd may close after mmap.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len.max(1),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                std::os::fd::AsRawFd::as_raw_fd(&file),
                0,
            )
        };
        if at == libc::MAP_FAILED {
            return Err(Fault::Load(checkpoint::error::Error::Checkpoint(
                "the arena spill file does not map".to_string(),
            )));
        }
        Ok(SpillArena {
            at: at.cast(),
            len,
        })
    }

    fn as_mut(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is `len` writable bytes this struct owns.
        unsafe { std::slice::from_raw_parts_mut(self.at, self.len) }
    }
}

impl Drop for SpillArena {
    fn drop(&mut self) {
        // SAFETY: unmapping the mapping this struct created.
        unsafe {
            libc::munmap(self.at.cast(), self.len.max(1));
        }
    }
}

/// One quantized weight's other planes, as rows of [`places`]: the scales
/// always, and — for an affine scheme, whose codes centre on a stored zero
/// point — the biases beside them.
#[derive(Debug, Clone, Copy)]
struct Pairing {
    scales: usize,
    biases: Option<usize>,
}

/// **THE SPLIT-PLANE PAIRINGS THIS LOAD PLAN STATES**, by the code plane's own
/// name.
///
/// Every entry is recorded by whoever declared the scale tensor, at the point
/// of declaring it (`QuantAttachment`) — a contract that shipped its own
/// scales says so with `TensorContract::scaling`, and an encode the loader
/// performs records the id itself. So the pair is READ here and never
/// reconstructed: a `.scales` suffix matched against a param name is how a
/// scale tensor gets read as the wrong bank's, silently, in a model that
/// computes.
///
/// **ONE BANK FORM ON THIS PLANE.** The cuda shell stamps exactly one
/// quantized routed point — mxfp4, e2m1 codes under raw e8m0 exponents
/// (`linear::moe::matmul_select_bias` and its bias-free twin) — so an affine
/// codec's `Bf16AffineFactors`, whose bank is `code * scale + zero`, is
/// refused by name rather than seated without the half that centres it.
///
/// # Errors
///
/// [`Fault::Param`] for an attachment naming a plane this trace does not
/// declare, or a scale form this shell has no point for.
fn pairings<'a>(
    landing: &'a LoadPlan,
    index: &BTreeMap<&str, usize>,
) -> Result<BTreeMap<&'a str, Pairing>> {
    // Keyed by the id's own number: `TensorId` is `Hash` and `Eq` and not
    // `Ord`, and a derive on the loader's type is not this shell's to add.
    let named: BTreeMap<u32, &str> = landing
        .tensors
        .iter()
        .map(|decl| (decl.id.0, decl.name.as_str()))
        .collect();
    let mut out = BTreeMap::new();
    for attachment in &landing.attachments {
        let Some(name) = named.get(&attachment.tensor.0) else {
            continue;
        };
        // Only a plane the trace declares becomes a weight row; a contract may
        // compute internal tensors, and an attachment on one of those is a
        // pairing about bytes no `Def::Weight` names.
        if !index.contains_key(name) {
            continue;
        }
        let row = |id: TensorId, what: &'static str| -> Result<usize> {
            named
                .get(&id.0)
                .and_then(|plane| index.get(plane))
                .copied()
                .ok_or_else(|| Fault::Param {
                    name: (*name).to_string(),
                    why: what,
                })
        };
        let biases = match attachment.scale_form {
            ScaleForm::RawE8M0 => None,
            // The affine bank: `code * scale + zero`, so the pairing is a
            // TRIPLE and seating it without the zero points would be the
            // right spread around the wrong centre — a model that computes
            // and is wrong. Required, not optional.
            ScaleForm::Bf16AffineFactors => Some(row(
                attachment.zero_point_tensor.ok_or_else(|| Fault::Param {
                    name: (*name).to_string(),
                    why: "is an affine bank whose attachment names no zero-point \
                          tensor; `code * scale` alone is the wrong centre",
                })?,
                "is an affine bank whose zero points this plan does not publish as a \
                 param of their own",
            )?),
            ScaleForm::F32Factors => {
                return Err(Fault::Param {
                    name: (*name).to_string(),
                    why: "carries a scale form no point this shell stamps reads: the \
                          cuda plane's split-plane banks are mxfp4 codes under raw \
                          e8m0 exponents and MLX affine codes under bf16 factor pairs",
                });
            }
        };
        out.insert(
            *name,
            Pairing {
                scales: row(
                    attachment.scale_tensor,
                    "is a quantized weight whose scales this plan does not publish as a \
                     param of their own",
                )?,
                biases,
            },
        );
    }
    Ok(out)
}

/// **WHICH OF THE TWO RUNS THIS IS** (§M-3: `.wiki/alto/zt-as-serving-artifact.md`).
///
/// [`Weights::resident`] has always been two functions wearing one name: the
/// COLD half — decode the checkpoint, run the landing transforms, materialize
/// a store, write a serving artifact out of it — and the WARM half, which
/// opens that artifact, cuts it against this boot's budgets and pumps. Both
/// were reachable from a serve, and which one ran was decided by whether a
/// file happened to be on the disk.
///
/// §M-3 makes it a decision instead, and gives the two halves different
/// owners:
///
/// ```text
///   Serve     the warm half, and NOTHING ELSE. A streamed load with no
///             artifact it can cut REFUSES, naming the command that makes
///             one. It never streams, never transforms, never writes.
///   Prepare   both halves, warm first: an artifact that opens and restores
///             is one this box can serve and the run is over; anything else
///             is said out loud and the cold half writes the file.
/// ```
///
/// # Why this is a parameter and not a `cache_dir` that is `None`
///
/// Because those are opposite meanings. `cache_dir: None` says *this
/// deployment has no artifact and wants none* — which under `Serve` is now a
/// refusal for a streamed plan, since there is no other road. The prepare run
/// is the one that MUST have a directory. Overloading the option would have
/// made the writer indistinguishable from the feature being off.
///
/// # And why the fully-resident path does not read it
///
/// A model that fits device memory outright plans no tier, forms no
/// [`tier_key`], and lands out of the checkpoint every time; its own
/// whole-table cache ([`crate::weight_cache::store_indexed`]) is an
/// accelerator that a miss simply skips. Nothing in this wave touches it, and
/// the gate below is `plan.streams()` for exactly that reason.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Intent {
    /// A load that will serve fires. Warm or refused.
    Serve,
    /// `pie model import`'s run, through [`Shell::prepare`](crate::Shell). The
    /// only writer of serving artifacts there is.
    Prepare,
}

impl Weights {
    /// Land `contract` against the checkpoint at `path`.
    ///
    /// `path` is a snapshot directory or a single container; a directory is
    /// discovered the way `pie model import` discovers one, a file is read
    /// directly.
    ///
    /// # Errors
    ///
    /// [`Fault::Load`] for a checkpoint the contract does not fit,
    /// [`Fault::Param`] for a plan and a contract that do not name the same
    /// tensors, [`Fault::Device`] for the residency itself.
    ///
    /// # The warm boot
    ///
    /// `cache_dir` is the weight artifact cache's directory, typed from
    /// `Boot` (article 9: shells read no environment). With a directory, the
    /// device table is keyed on this load's whole recipe
    /// ([`weight_cache::Identity`]) and, on a match, read STRAIGHT TO THE
    /// DEVICE: the plan compile still runs (it is milliseconds, and it is what
    /// validates the contract against the checkpoint), and everything after it
    /// — the executor's host-side dequant, cast and repack, and the
    /// per-tensor uploads — does not.
    ///
    /// A corrupt artifact is never trusted and never silently retried: it is
    /// counted, said out loud, and LEFT ON THE DISK (§M.4).
    ///
    /// # And a STREAMED load is warm or it is nothing (§M-3)
    ///
    /// `intent` is the whole of the difference and [`Intent`] argues it. In
    /// one line: a `Serve` whose `plan.streams()` and which cannot open,
    /// cut and verify a serving artifact returns [`Fault::Residency`] naming
    /// `pie model import --prepare-only`; it does not stream the checkpoint,
    /// does not run the landing transforms, and cannot write a file. Only
    /// `Prepare` reaches [`write_tiers`], which is the only caller of
    /// [`tier::store`](crate::weight_cache::tier::store), which is the only
    /// door that replaces one.
    ///
    /// **THE FULLY-RESIDENT PATH IS UNTOUCHED.** A plan that streams nothing
    /// asks none of this: no [`tier_key`], no artifact, no refusal. It lands
    /// out of the checkpoint and its whole-table cache stays what it was — an
    /// accelerator whose miss costs a load, not a deployment.
    ///
    /// # Errors
    ///
    /// [`Fault::Residency`] for a streamed `Serve` with no artifact it can
    /// cut, beside the errors above.
    pub fn resident(
        trace: &Trace,
        contract: &ModelContract,
        path: &Path,
        cache_dir: Option<&Path>,
        plan: crate::experts::Plan,
        stream: *mut core::ffi::c_void,
        intent: Intent,
    ) -> Result<Weights> {
        let (metadata, snapshot) = if path.is_dir() {
            (parse_metadata(path)?, path)
        } else {
            (zt::parse(path)?, path.parent().unwrap_or(Path::new(".")))
        };

        // tp=1: the plan's `Shard::Cut` segments still describe the whole
        // tensor, and a rank of one takes all of them. `tp_rank`/`tp_size`
        // are the loader's whole notion of a rank, so a shell that grows
        // tensor parallelism states it here and nowhere else.
        let target = StorageTarget::for_backend(BackendKind::Cuda, 0, 1);
        let landing = compile(&metadata, contract, target.clone())?;

        // The plan's own name -> row map, and the pairings it states. Read by
        // the landing sink to place an arriving tensor, by the table build to
        // resolve an attachment's planes onto rows, and — since §M — by the
        // RANKING, which is what names the file this boot is about to read. So
        // it is stated once above all three rather than inside whichever
        // branch used to be its only reader.
        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        let ranking = crate::experts::Ranking::of(trace, &attachments(&landing, &index)?)?;

        let places = places(trace, &plan)?;
        let total = places.last().map_or(0, |p| p.offset + p.reserved);
        let mut store = Buffer::zeroed(usize::try_from(total).unwrap_or(usize::MAX))?;
        // **THE TIER IS OPENED BEFORE THE BYTES ARRIVE**, because a streamed
        // bank's plane does not land in the store at all: it lands in the
        // PINNED tier, whole, and the slab takes a copy of its first
        // `resident` slots afterwards. `None` for the degenerate plan, and
        // then nothing below this line is different from what it was.
        // ── THE OTHER KEY, AND IT IS FORMED BEFORE THE TIER IS ALLOCATED
        //    (§K.4, phase T-3). It used to sit below `Tier::open`, because a
        //    phase that only WROTE the file could form its name at any point
        //    before the write. A phase that reads it cannot: whether the
        //    pinned tier — tens of gigabytes of page-locked memory — pays for
        //    its `write_bytes(0)` is a function of whether these bytes are on
        //    the disk, and that question has to be asked before the allocation
        //    is made rather than after.
        //
        //    Formed only when there is a directory and a plan that streams, so
        //    a fully-resident load pays for neither the second `places` walk
        //    nor the second serialization. `plan.streams()` is the condition
        //    `experts.is_some()` stood for.
        let tiers = match cache_dir.filter(|_| plan.streams()) {
            Some(_) => tier_identity(trace, &landing, path, &ranking)?,
            None => None,
        };
        // ── AND THE ARTIFACT ITSELF, OPENED BEFORE THE TIER IT WILL FILL.
        //    `Ok` only for a file this build can cut; every refusal is
        //    `open_tiers`', built there and counted there — and none of them
        //    deletes it (§M.4).
        //
        //    **AND THIS IS WHERE A SERVING LOAD ENDS IF THERE IS NOTHING TO
        //    CUT** (§M-3). Before the pinned tier is allocated, before a byte
        //    of the checkpoint is read, and with the sentence the miss built:
        //    a streamed serve has one road to these weights and this is the
        //    door onto it. A PREPARE goes on — that is the run whose whole job
        //    is to make the thing that is missing — and it says so first,
        //    unless the miss was merely that the file is not there yet, which
        //    is the ordinary shape of an import and not news.
        //
        //    **AND A SERVING ARTIFACT MAKES THE WHOLE QUESTION MOOT**
        //    (§M-4d). Since §M-4b the model's own `.zt` holds every plane of
        //    the trace: it IS the file this door goes looking for, so a
        //    deployment served from one has nothing to prepare and no second
        //    file to miss. Asked here, before the miss is built, because the
        //    miss is a REFUSAL — `open_tiers` is where a streamed serve ends
        //    when there is nothing to cut — and refusing a load whose weights
        //    are sitting in the checkpoint it was pointed at would be this
        //    wave breaking the thing it exists to make.
        let serving = crate::checkpoint_serving::Serving::open(path, trace);
        // Kept past the move below: a PREPARE writes a tier file and a
        // deployment served out of its own `.zt` has nothing for one to hold.
        let served_from_checkpoint = serving.is_some();
        let opened = match (cache_dir, tiers) {
            (Some(dir), Some(key)) => open_tiers(dir, key, &ranking, &plan, path),
            (dir, _) => Err(unkeyed(dir, path)),
        };
        let warm = match opened {
            Ok(artifact) => Some(artifact),
            // A plan that streams nothing never wanted one; the miss it was
            // handed is about a file it does not read.
            Err(_) if !plan.streams() => None,
            // Nor did a plan whose planes come out of the checkpoint itself.
            // `None` and not a refusal: there is no tier artifact, which is
            // the point, and `defer_tiers` reads that as "no deferred seat"
            // exactly as it does for every load that has no second file.
            Err(_) if serving.is_some() => None,
            Err(miss) => match intent {
                Intent::Serve => return Err(Fault::Residency(miss.why().to_string())),
                Intent::Prepare => {
                    if let Miss::Refused(why) = &miss {
                        eprintln!("{why}");
                    }
                    None
                }
            },
        };
        // ── THE T2 SOURCE, OPENED BEFORE THE TIER THAT POINTS INTO IT — AND
        //    THERE ARE TWO FILES IT CAN BE NOW (§K.6-T4).
        //
        //    **THIS DEPLOYMENT'S SERVING ARTIFACT FIRST.** It carries one
        //    image per plane of this trace, so it answers for whatever THIS
        //    budget spills — and, since §M, for whatever the next one spills
        //    too — and it arrives with every other image beside it, so a hit
        //    here is the whole load off one file and the resident snapshot
        //    that seeded the deployment is no longer needed on this disk at
        //    all. `open_tiers` has already hashed the spilled images by the
        //    time this asks (see its doc), so the plane windows below are
        //    windows on bytes that have been checked.
        //
        //    **THE WHOLE-TABLE RESIDENT ARTIFACT SECOND**, which is where
        //    every spilled deployment starts: the file a fully-resident load
        //    wrote, under the key that load would have formed —
        //    `resident_key`'s doc says why a capped load can name it. It is
        //    the bootstrap, and the run it feeds is the one that writes the
        //    tier artifact above.
        //
        //    **AND THAT SECOND ARM SURVIVES §M-3, AS PREPARE'S ONLY ROAD
        //    IN.** It is tempting to read "the boot is warm-only" as "the
        //    bootstrap is dead", and it is the reverse: a SERVE reaching this
        //    line already holds a `warm`, so it takes the first arm and the
        //    second is a formality. A PREPARE of a spilled deployment holds
        //    no `warm` by definition — the file it is about to write is the
        //    one that does not exist — and its landing still has to read the
        //    spilled planes from somewhere. That somewhere is the whole-table
        //    artifact an uncapped boot wrote, exactly as it was in §K.6-T4.
        //    Deleting this arm would make a spilled deployment unpreparable
        //    and therefore unservable, which is the opposite of the wave.
        //    `Residency::admit_tiers` still counts either file as a source
        //    for the same reason: it is asked by `Cuda::prepare` as well as
        //    by `Engine::load`, and a statute that demanded the tier artifact
        //    would refuse the run that creates it.
        //
        //    `None` for a plan that spills nothing, which is every plan whose
        //    host budget held its groups — and for a spilled plan with neither
        //    file, which `Residency::admit_tiers` refused before the store was
        //    reserved.
        //
        //    **THE FILE IS OPENED TWICE ON A WARM SPILLED BOOT**, once as
        //    `warm` and once here, and deliberately: `warm` is borrowed by the
        //    restore and dropped, while this mapping has to outlive the load
        //    because the weight rows point into it. Two `mmap`s of one inode
        //    share their pages, so the second costs a file descriptor and an
        //    address range.
        //    **AND THE FIRST ARM IS GATED ON `warm`** (§M.4). It used to stand
        //    on `tier_spill` alone, because a file whose bytes did not hash
        //    was DELETED by the door that found out — so by the time this ran
        //    there was nothing left to open. Nothing deletes now, and a
        //    parseable file with a rotted image would otherwise be mapped
        //    here and served, one page at a time, with no door left to catch
        //    it. `warm` is `open_tiers`' answer, and `open_tiers` is the door
        //    that hashed the spilled images. Since §M-3 a serving load with
        //    no `warm` never reaches this line at all, so the gate is what a
        //    PREPARE reads: it says "do not map a file this run has just
        //    refused; map the bootstrap and rewrite the refused one".
        // **AND THE SOURCE IS THE CHECKPOINT, WHICH IS THE ONLY ONE LEFT**
        // (§M-4d step 3). Since §M-4b the `.zt` `pie model import` writes IS
        // the serving file — every plane of the trace, under this SKU's names,
        // in the order a boot reads them — so a deployment served from one
        // needs no second file beside it and no `[model] weight_cache_dir` to
        // find it.
        //
        // The two roads that stood here are gone. A tier artifact a prepare
        // wrote and the whole-table snapshot an uncapped boot left behind were
        // both a SECOND COPY of the model on the disk, which is the cost §M-4
        // exists to stop paying, and the two doors were measured equal within
        // noise before either was removed (W1@c128: 10,441 tok/s through the
        // snapshot door against 10,269 through this one).
        //
        // `Serving::open` answering `None` is now a refusal and not a
        // fallback: a spilled load whose checkpoint is not a serving artifact
        // has nowhere to read the planes it did not keep, and
        // `Residency::admit` says so by name above this line.
        let source = match plan.spill_demand() > 0 {
            true => serving.map(crate::experts::Spill::Serving),
            false => None,
        };
        // ── AND WHETHER THE PAGE-LOCKED IMAGE IS MADE AT THIS INSTANT AT ALL
        //    (§L.1, phase L-1). `defer_tiers` states the policy and the price;
        //    a `None` from it is every road this file had before §L.
        let deferred = defer_tiers(warm.as_ref(), &plan);
        let mut experts = match plan.streams() {
            // **AND THE MEMSET IS SKIPPED EXACTLY WHEN A FILE WILL COVER IT.**
            // `restore_tiers` reads every T1 image — padding included, and
            // `Plan::host_layout` tiles the allocation — straight into it
            // before anything reads a byte out of it, and owes it
            // `Tier::zero_host` if it cannot. **A DEFERRED SEAT SKIPS THE
            // ALLOCATION TOO**, and owes it `Tier::undefer`.
            true => {
                let fill = match (deferred, warm.is_some()) {
                    (Some(artifact), _) => crate::experts::Fill::Deferred(artifact),
                    (None, true) => crate::experts::Fill::Restored,
                    (None, false) => crate::experts::Fill::Cold,
                };
                Some(crate::experts::Tier::open(plan.clone(), source, fill)?)
            }
            false => None,
        };

        // ── THE WARM BOOT (alto design §7). The device table is a function of
        //    the checkpoint, this recipe, and this layout — so the second boot
        //    of the same deployment can read what the first one computed. The
        //    key is composed from all three; a mismatch of any of them is a
        //    miss, and a miss just runs the load below.
        //
        //    NOTHING IS PAID FOR A FEATURE THAT IS OFF: with no directory
        //    there is no key to form, so the layout is not collected and the
        //    plan is not serialized. A load that was never offered a cache
        //    costs exactly what it cost before this block existed.
        // **TWO KEYS, TWO PATHS, AND A LOAD FORMS EXACTLY ONE OF THEM**
        // (§K.2, phase T-2). The statute this replaces said that a streamed
        // load formed NO key: the artifact is a snapshot of the DEVICE STORE,
        // and a streamed load's store is a cache over a pinned tier that file
        // says nothing about — so restoring one would fill the slabs and
        // leave T1 empty, which is a table whose non-resident entries point
        // at zeros.
        //
        // What retires it is not a wider artifact but a SECOND one. A
        // resident load keys the store exactly as it always did; a streamed
        // load keys the SEQUENCE with `tier_identity`, which mixes what the
        // ranking lays out on top of everything the resident identity mixes —
        // and since §M.3 no budget and no rung, so one file serves any cut.
        // The two live in one directory under one key space and cannot be
        // handed to each other's reader: different magics, different
        // extensions (`weight_cache::tier`'s header argues both).
        let key = cache_dir.filter(|_| experts.is_none()).and_then(|_| {
            let layout: Vec<(u64, u64, u64)> = places
                .iter()
                .map(|place| (place.offset, place.bytes, place.reserved))
                .collect();
            // The plan IS the recipe, so it is hashed WHOLE rather than
            // summarized: a plan that grows a field is covered the moment it
            // exists. A plan that will not serialize is not a fault — it is a
            // key this load cannot form, and a load with no key neither reads
            // nor writes the cache.
            let plan_json = serde_json::to_vec(&landing).ok()?;
            Some(
                crate::weight_cache::Identity {
                    checkpoint: path,
                    trace_name: &trace.name,
                    plan_json: &plan_json,
                    layout: &layout,
                    total,
                }
                .key(),
            )
        });
        // ── THE WARM STREAMED BOOT (§K.4, §M.3). Every image out of one
        //    file, cut by THIS boot's budgets: the store's planes pumped to
        //    their own offsets, the T1 images read straight into the pinned
        //    tier's own bytes, the T2 images left where they lie — and all of
        //    them verified. A `true` here skips exactly what a resident
        //    restore skips — the source walk, every `TileMap`, `Finalize`'s
        //    read-back, the sink's copies, the arena and the spill — and
        //    leaves the load running only what it would have run anyway:
        //    `Tier::land`, the rotor, the pools.
        //
        //    A refusal is where the two intents part (§M-3). Under `Serve`
        //    it is the end of the load: the sentence goes into a
        //    `Fault::Residency`, nothing is zeroed, and nothing is written —
        //    a serving path that rewrote a hundred gigabytes on the strength
        //    of one bad block is the thing this wave removed. Under `Prepare`
        //    both images go back to zeros, the file stays on the disk with a
        //    sentence beside it (§M.4), and the cold branch below rewrites it,
        //    which is the only door that ever replaces one.
        //
        //    **TWO KEYS, TWO PATHS, AND A LOAD FORMS EXACTLY ONE OF THEM.**
        //    `key` is `None` for a streamed load and `tiers` is `None` for a
        //    resident one, so these two restores are alternatives and not a
        //    sequence.
        let from_cache = match key {
            Some(key) => crate::weight_cache::restore(cache_dir, key, &mut store)?,
            None => match (warm, experts.as_mut()) {
                // **AND A REFUSAL HERE ENDS A SERVE TOO** (§M-3). The images
                // opened, the index was this trace's, and then the bytes did
                // not hash — or the machine would not read them. There is no
                // second road: `restore_tiers` states the sentence and skips
                // the recovery for a load that is about to return.
                (Some(artifact), Some(tier)) => {
                    match restore_tiers(&artifact, path, &places, &mut store, tier, intent)? {
                        None => true,
                        Some(why) => match intent {
                            Intent::Serve => return Err(Fault::Residency(why)),
                            Intent::Prepare => {
                                eprintln!("{why}");
                                false
                            }
                        },
                    }
                }
                _ => false,
            },
        };

        let landed = if from_cache {
            // EVERY PARAM IS LANDED, because the blob is the whole table —
            // the layout is part of the key, so a restore that matched wrote
            // exactly the bytes this `places` describes. The same sentence
            // holds for the streamed restore, one rung further out: the
            // serving artifact's key mixes the RANKING (§M.3), `open_tiers`
            // checked the file's index against it image for image, and the
            // cut then routed every one of those images to the rung this
            // `places` and this `Plan::host_layout` put it on.
            //
            // **AND THE WRITE IS SKIPPED WITH THE BRANCH.** A hit that
            // restored must not re-write what it just read; it does not
            // reach `write_tiers` at all, which is a stronger statement than
            // that function's own already-on-the-disk arm.
            vec![true; places.len()]
        } else {
            // Where every finalized tensor goes, whichever shape produces it:
            // the sink is the load's own placement rule and knows nothing
            // about how the executor held the bytes on the way here.
            let mut sink = Landing {
                store: &mut store,
                experts: experts.as_ref(),
                plan: &plan,
                places: &places,
                index: &index,
                landed: vec![false; places.len()],
            };
            // ── **A STREAMED LOAD HAS NO ARENA AT ALL** (§K.3 follow-on).
            //    The arena is where a transform's intermediates live while the
            //    schedule runs, and a resident load needs one: its image is
            //    the device store's, staged host-side and pumped over. A
            //    STREAMED load's image is not staged anywhere — each tensor
            //    goes to the pinned tier or the slab as it finalizes — so the
            //    executor's own streaming residency owns every buffer and
            //    frees it at its last use, and what the load holds at once is
            //    one tensor's chain rather than the whole model.
            //
            //    What that subtracts is the traffic §K.0 measured: the write
            //    of the whole image into the arena, `Finalize` reading the
            //    whole image back out of it, and — the one that actually hurt
            //    — the `MAP_SHARED` spill file `Scratch::fitting` falls to
            //    when the image will not fit beside the tiers, which turned
            //    both of those into disk. The 4-bit flash load planned a
            //    102 GiB arena; now it plans none.
            //
            //    The plan is compiled a SECOND TIME for it, and that is the
            //    whole price: `compile_streaming` is this same contract
            //    against this same CUDA target, minus the two passes that
            //    exist to serve an arena (it says there why the target cannot
            //    be weakened to `Unknown` instead). Milliseconds, against a
            //    hundred gigabytes of disk.
            //
            //    **AND THE SECOND PLAN IS NEVER A KEY.** `landing` — the
            //    coalesced one — is what `resident_key` and `tier_identity`
            //    hash, above and below this branch, exactly as they did
            //    before. It has to be: `prospect` and `tier_key` compile from
            //    OUTSIDE the load, holding no residency to state, so a key
            //    formed here off a streaming-shaped plan would name a file no
            //    caller could ever find. This one is compiled, run, dropped.
            let landed = if plan.streams() {
                let streaming = compile_streaming(&metadata, contract, target)?;
                Execution::new(&streaming, snapshot)
                    .streaming()
                    .sink(&mut sink)
                    .run()?;
                sink.landed
            } else {
                // Host memory, because the transforms run host-side; dropped
                // the moment the load is over, and only the finalized tensors
                // the sink took survive.
                //
                // **AND HOST MEMORY HAS A SECOND SPELLING** (qwen4 flash). The
                // arena is planned at the image's own size, and an image can
                // outgrow the RAM the machine will actually give this process.
                // So an arena that will not fit beside the tiers goes to a
                // FILE-BACKED map instead: dirty file pages are the kernel's
                // to write back and reclaim under pressure, which turns an OOM
                // kill into a slower load. Asking the machine how much room it
                // has is the same class of question as asking the card its
                // memory. The pinned tier does not appear in this call and
                // must not: `Tier::open` allocated it above, so the machine
                // has already counted it and `Scratch::fitting` asks the
                // machine. (A load that reaches this arm streams nothing, so
                // its `spill_demand` is zero and the mapped term with it. The
                // argument is stated where it is answered, not where it
                // happens to be non-zero.)
                let bytes = usize::try_from(landing.memory.arena_bytes()).unwrap_or(0);
                let mut scratch = Scratch::fitting(bytes, plan.spill_demand())?;
                let mut backing: &mut [u8] = scratch.as_mut();
                Execution::new(&landing, snapshot)
                    .arena(&mut backing)
                    .sink(&mut sink)
                    .run()?;
                let landed = sink.landed;
                drop(scratch);
                landed
            };

            // **THE ARTIFACT IS WRITTEN FROM THE STORE, NOT FROM THE
            // TRANSFORMS.** What is cached is what is resident, which is the
            // only thing the digest can be a claim about. Best-effort in every
            // direction: a declined write is a counted line, not a failed
            // load.
            if let Some(key) = key {
                // **WITH THE PLANE-GROUP INDEX**, which is what turns the
                // artifact from a boot accelerator into a serving-time T2
                // source (alto streaming §0, build order item 2). One entry
                // per param at plane zero: a split-plane bank is already two
                // `Trace::params` rows on this plane, so each is its own
                // group and the index's `plane` axis stays at zero — it is
                // for a shell that puts both planes under one id, which this
                // one does not.
                //
                // This runs on the RESIDENT path only (`key` is `None` for a
                // streamed load), so the index it writes describes the whole
                // table — which is exactly the file a later capped load maps.
                let groups: Vec<crate::weight_cache::Group> = places
                    .iter()
                    .enumerate()
                    .map(|(at, place)| crate::weight_cache::Group {
                        id: u32::try_from(at).unwrap_or(u32::MAX),
                        plane: 0,
                        offset: place.offset,
                        bytes: place.bytes,
                        reserved: place.reserved,
                    })
                    .collect();
                crate::weight_cache::store_indexed(cache_dir, key, &groups, &store);
            }
            // **AND THE STREAMED LOAD'S SERVING ARTIFACT** (§K.3, §M.3), from
            // the same instant and for the same reason: what is written is
            // what is materialized, which is the only thing a digest can be a
            // claim about.
            //
            // **HERE, AND NOT AFTER `Tier::land`.** The landing is what this
            // file is a function of; the ladder is what a load does with it
            // afterwards. Taking the snapshot before the slabs are seated
            // keeps the file a function of the RANKING alone — two boots at
            // two different budgets write the same bytes — and keeps a
            // promotion that has already happened out of an image a later boot
            // cuts as its starting position.
            //
            // **AND ONLY A PREPARE REACHES IT** (§M-3). This is the whole of
            // the rebuild door: `write_tiers` -> `tier::store` -> the
            // verify-then-replace, and `Intent::Prepare` is the only way in.
            // A serving load that got this far streams nothing (it would have
            // refused above), so the condition is belt-and-braces — but it is
            // the STATEMENT of the property, and a future caller that lands a
            // streamed serve back on the cold arm should meet it here rather
            // than discover it by finding a hundred gigabytes rewritten
            // underneath a serving deployment.
            //
            //    **AND A DEPLOYMENT SERVED OUT OF ITS OWN `.zt` HAS NOTHING
            //    TO WRITE** (§M-4d step 2). The artifact `pie model import`
            //    wrote already holds every plane of this trace, ranked, with
            //    a digest table apiece — a tier file beside it would be a
            //    second copy of the model on the disk that no boot will ever
            //    open, because `Spill::Serving` is asked first and answers.
            //    So a prepare of one still runs (it is the check that this
            //    BOX can serve the artifact, which no format test can make)
            //    and writes nothing.
            //
            //    **AND THIS IS STILL REACHABLE, WHICH DECIDES WHETHER §M-4d's
            //    LAST STEP IS A DELETION OR A PORT.** The tempting reading is
            //    that nothing writes a tier file any more — step 2 stopped it
            //    for artifact deployments and step 3a refuses a spilled load
            //    that is not one, so `Fill::{Restored, Deferred}` would have
            //    no inputs and could simply go. That reading is WRONG, and the
            //    case it misses is narrow and real: a plan that STREAMS but
            //    does not SPILL, on a checkpoint that is not a serving
            //    artifact. `experts` is `Some` because the plan streams;
            //    `admit_tiers` passes because `spilled` is zero, so 3a's
            //    refusal never fires; and a prepare of it writes a tier file
            //    that the next boot reads back as `warm`.
            //
            //    So the tier road still carries the PINNED image for such a
            //    deployment, and retiring `Fill::Restored` would take a
            //    working optimization off it rather than removing dead
            //    machinery.
            //
            //    **AND IT IS NOT A CORNER — IT IS THE COMMON CASE, MEASURED.**
            //    Swept over the catalog at `Platform::Cuda`, sixteen device
            //    budgets from the ranking's floor to its full against four
            //    host budgets: **all 42 SKUs have budget pairs that stream
            //    without spilling**, and it is typically more than half of
            //    them — `dsv4-flash-bf16` 60 of 64, `gemma4-31b-bf16` 37 of
            //    64, `dsv4-base-bf16` 56 of 64. A capped deployment that fits
            //    device-plus-host is the ordinary shape of a streamed load,
            //    not an edge of it.
            //
            //    So the answer is to PORT the two fills at the `.zt` and not
            //    to delete them. Deleting would make every capped deployment
            //    rebuild its page-locked image on every boot — which is the
            //    tens of seconds of memory bandwidth `Fill::Restored` exists
            //    to skip — and artifact deployments already take `Fill::Cold`
            //    today, so the wave would END by making the common case
            //    slower than the world it replaced. The port's own hazard is
            //    written at `experts::Fill::Restored`.
            if let (Intent::Prepare, false, Some(key), Some(tier)) =
                (intent, served_from_checkpoint, tiers, experts.as_ref())
            {
                write_tiers(cache_dir, key, trace, &landed, &ranking, &places, tier, &store);
            }
            landed
        };

        // The pairings this landing states, read once. Empty for every SKU
        // whose weights are all dense, which is every catalog row but
        // gpt-oss's — and an empty map costs one walk of a vector nobody has
        // pushed to.
        let pairings = pairings(&landing, &index)?;

        let mut table = Vec::with_capacity(places.len());
        for (at, place) in places.iter().enumerate() {
            // A REGISTERED PLANE IS ONE THE CHECKPOINT DOES NOT HAVE, and
            // demanding it is exactly what would refuse every adapter-capable
            // load (design §8's open item, from the residency side). It is
            // already reserved — `places` sizes every param, whatever its
            // provenance — and already zeroed, and a zeroed low-rank `A`
            // makes the whole bank the identity until something registers
            // into it. What lands here is `register_adapter`'s business.
            if !landed[at] && trace.params[at].source == ParamSource::Checkpoint {
                return Err(Fault::Param {
                    name: trace.params[at].name.clone(),
                    why: "is a plan param the load contract never published",
                });
            }
            // **A SPLIT-PLANE BANK IS TWO HANDLES UNDER ONE `Def::Weight`**
            // (alto streaming §3 item 6, wave W-5). The pairing is the load
            // plan's — stated by whoever declared the scale tensor rather than
            // guessed from a `.scales` suffix, which is how a scale tensor
            // gets read as the wrong bank's — and `Run::planes` is the one
            // resolution that answers with both.
            //
            // **BOTH PLANES ARE BOUND AS `U8`.** The mxfp4 select reads them
            // as `const u8*` and does its own block arithmetic
            // (`quant.cuh`); the handle's dtype is what a `{:?}` prints, and
            // naming an element the launch never indexes by would be a number
            // nothing means. `rows x width` is the BYTE rectangle for both,
            // because `plane_bytes` sizes an mxfp4 code plane at exactly its
            // declared rectangle and an e8m0 exponent plane at one byte an
            // entry.
            //
            // **A STREAMED DENSE BANK IS STILL ONE DENSE HANDLE**, and it has
            // to be: `Run::tensor` is the one resolution path for it and a
            // second kind of weight row would be a second one. What the handle
            // names is the SLAB — `resident` slots at the store's own address
            // — and the table that says which expert is in which of them rides
            // beside it in the row, as two device addresses the select kernel
            // reads (alto design §7, wave D2). `rows` is the slot count rather
            // than the expert count, which is honest: the numbers on a weight
            // handle are what a `{:?}` prints, and no entry reads a bank's
            // rows back as a promise (see `rectangle`).
            let row = match pairings.get(trace.params[at].name.as_str()) {
                // **AND A STREAMED GROUP CARRIES ITS SEAT** (alto streaming §3
                // item 3, wave B7). The two handles are where the plan seated
                // the planes; the seat is the fixed-address cell the select
                // reads the LIVE pair out of and the counter it notes the
                // routing in, so that the ladder can move the group without
                // touching a captured graph. Two zeros for a group the store
                // holds whole with no tier open, which is every group of a
                // fully-resident load.
                Some(pairing) => WeightRow::Planes {
                    // **THE ORDER THE PLANES ARE IN, OFF THE DECLARATION**
                    // (§J4b). `Dtype::U4g64tiled` is the model text saying
                    // this projection's three rectangles were written by
                    // `pie model import` through the tiled relabelling; the
                    // bytes are the same bytes and the rectangle is the same
                    // rectangle, so the declaration is the only thing that
                    // could say it. `dispatch::linear` is where it is spent.
                    repacked: place.dtype == Dtype::U4g64tiled,
                    codes: packed(experts.as_ref(), &store, &places, at)?,
                    scales: packed(experts.as_ref(), &store, &places, pairing.scales)?,
                    biases: match pairing.biases {
                        Some(biases) => {
                            Some(packed(experts.as_ref(), &store, &places, biases)?)
                        }
                        None => None,
                    },
                    seat: experts
                        .as_ref()
                        .and_then(|tier| tier.group_handles(at))
                        .map_or(GroupSeat::RESIDENT, |seat| GroupSeat {
                            cell: seat.cell,
                            hits: seat.hits,
                        }),
                },
                None => {
                    let handle = Tensor::new(
                        address(experts.as_ref(), &store, place.offset, at)?,
                        place.rows,
                        place.width,
                        place.dtype,
                    );
                    match experts.as_ref().and_then(|tier| tier.handles(at)) {
                        None => WeightRow::Dense(handle),
                        Some(handles) => WeightRow::Streamed {
                            slab: handle,
                            table: handles.table,
                            counts: handles.counts,
                        },
                    }
                }
            };
            table.push(Some(row));
        }
        // ── THE SLABS AND THE FIRST TABLE. Every resident slot filled from
        //    the pinned copy the sink just wrote, and every entry published —
        //    the resident ones into the slab, the rest at their pinned bytes
        //    over UVA. Synchronous, at load, on the shell's own stream: this
        //    is the same instant the checkpoint itself lands at, and no fire
        //    has been enqueued behind it.
        if let Some(tier) = experts.as_mut() {
            let slabs: Vec<u64> = tier
                .plan()
                .banks()
                .iter()
                .map(|bank| store.at(places[bank.param].offset))
                .collect::<Result<_>>()?;
            // **AND WHERE THE STORE PUT EVERY PLANE IT HOLDS** — what turns
            // a T0 packed group into a berth the ladder can displace a group
            // out of (wave B7). Only the loader can answer it: the offsets are
            // its, and `store.at` is the one place they become addresses.
            let store_at: Vec<(usize, u64)> = tier
                .plan()
                .seated()
                .iter()
                .flat_map(|group| group.planes.clone())
                .map(|plane| Ok((plane.param, store.at(places[plane.param].offset)?)))
                .collect::<Result<_>>()?;
            tier.land(&slabs, &store_at, stream)?;
        }
        Ok(Weights {
            store,
            table: WeightTable(table),
            banks: banks(trace, &places),
            experts,
            from_cache,
            rotor: None,
        })
    }

    /// **ARM THE ROTATING DENSE PUMP** — the second half of D2b, and the one
    /// call that turns a spilled dense plane from "read where it lies" into
    /// "copied ahead into a slot whose address never moves" (alto streaming §3
    /// item 4).
    ///
    /// Called once, at load, after [`Weights::resident`] and with the
    /// `CompiledModel` in hand — because the pump works at REGION granularity
    /// and a region is a thing the compiler decides, while the residency plan
    /// is decided before the model is compiled. That split is
    /// `prefetch::Schedule`'s own (`Schedule::of` reads nodes,
    /// `Schedule::against` projects onto regions), and this is its consumer.
    ///
    /// **THE CANDIDATES ARE T1 DENSE PLANES AND NOTHING ELSE.** A routed bank
    /// is the dynamic demand shape and has its own tier; a group held
    /// `Held::Mapped` is T2, whose pages are not page-locked, so an async H2D
    /// out of it is not async and the promotion it wants is streaming §3 item
    /// 3's, not this one. Both stay exactly where they were.
    ///
    /// **AND THE WEIGHT ROWS ARE REWRITTEN HERE, NOT AT `resident`.** A
    /// rotated plane's handle names its slot; everything else about the row —
    /// the rectangle, the dtype — is what `places` already said, because the
    /// only thing that changed is where the bytes are.
    ///
    /// Answers whether a pump was armed. A load that declines one is a correct
    /// load and not a refusal: the planes are served by the tier that was
    /// serving them.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for a slot, an event or a stream the runtime refused,
    /// and [`Fault::Residency`] for a tier that cannot answer a planned
    /// tenant's host address — which is a plan and a tier that disagree.
    pub fn rotate(
        &mut self,
        trace: &Trace,
        compiled: &model_compiler::CompiledModel,
    ) -> Result<bool> {
        let Some(tier) = self.experts.as_ref() else {
            return Ok(false);
        };
        // A group of one plane, not routed, page-locked: the shape a spilled
        // DENSE plane takes (`experts::Plan::of` makes it a `GroupPlan` with
        // `experts: 0`, which is the point of making a dense plane a group).
        let candidates: Vec<(usize, u64)> = tier
            .plan()
            .groups()
            .iter()
            .filter(|group| {
                !group.routed
                    && group.held == crate::experts::Held::Pinned
                    && group.planes.len() == 1
            })
            .map(|group| (group.param, group.bytes))
            .collect();
        if candidates.is_empty() {
            return Ok(false);
        }
        let schedule = model_compiler::prefetch::Schedule::of(trace);
        let rotation = match crate::rotate::Rotation::plan(
            &schedule,
            compiled,
            &candidates,
            crate::rotate::SLOT_CAP,
            crate::rotate::ARENA_CAP,
        ) {
            Ok(rotation) => rotation,
            // **A DECLINE IS NOT A FAULT.** Nothing rotates, every plane is
            // read where it lies, and the load is the load it was before this
            // method existed.
            Err(_why) => return Ok(false),
        };
        // Each tenant's page-locked source, in the rotation's own order. The
        // tier owns these bytes for the life of the load, which is what makes
        // the raw pointer sound to hold.
        let mut source: Vec<*const u8> = Vec::with_capacity(rotation.tenants().len());
        for tenant in rotation.tenants() {
            // **THE PLANE'S OWN ADDRESS AND NOT A BASE PLUS AN OFFSET**
            // (§M.3). It was `serving_host() + host_offset()` while a deferred
            // seat served T1 out of one contiguous section; the serving
            // artifact holds one image per plane in a ranking order a budget
            // cuts, so the tier answers per plane and there is no base to add
            // to. The address it hands back — the pinned allocation's, or the
            // artifact mapping's while the window is open — outlives this
            // `Weights`, because the tier holds the mapping for its whole life
            // for exactly this reason.
            let at = tier.serving_host_of(tenant.param).ok_or_else(|| {
                Fault::Residency(format!(
                    "`{}` was planned to rotate and the pinned tier seats no bytes for it",
                    trace.params[tenant.param].name
                ))
            })?;
            source.push(at);
        }
        let rotor = crate::rotate::Rotor::open(rotation, source)?;
        // ── AND THE ROWS NOW NAME THE SLOTS.
        for tenant in rotor.rotation().tenants() {
            let Some(seat) = rotor.seat(tenant.param) else {
                continue;
            };
            let param = &trace.params[tenant.param];
            let (rows, width) = rectangle(&param.shape);
            self.table.0[tenant.param] = Some(WeightRow::Dense(Tensor::new(
                seat,
                u32::try_from(rows).unwrap_or(u32::MAX),
                u32::try_from(width).unwrap_or(u32::MAX),
                param.dtype,
            )));
        }
        self.rotor = Some(rotor);
        Ok(true)
    }

    /// **The rotating dense pump this load armed**, or `None`. What the fire
    /// path hands its cursor at the region seam.
    #[must_use]
    pub fn rotor(&self) -> Option<&crate::rotate::Rotor> {
        self.rotor.as_ref()
    }

    /// **Does this load rotate dense planes during a fire?**
    ///
    /// Read by the fire path for one reason: a rotating load takes the EAGER
    /// walk, whatever mode the shell is in. See [`crate::rotate`]'s header for
    /// the capture-legality argument — the same shape as design §6's sentence
    /// about a buffered fire.
    #[must_use]
    pub fn rotating(&self) -> bool {
        self.rotor.is_some()
    }

    /// **The routed-expert tier this load opened**, or `None` for a load whose
    /// banks are resident (alto design §7).
    #[must_use]
    pub fn experts(&self) -> Option<&crate::experts::Tier> {
        self.experts.as_ref()
    }

    /// The same, mutably — what the promotion between two fires is driven
    /// through.
    pub fn experts_mut(&mut self) -> Option<&mut crate::experts::Tier> {
        self.experts.as_mut()
    }

    /// **Is the whole weight table on the device?** What
    /// [`LoadFacts::weights_resident`](engine::load::LoadFacts)
    /// reports, answered rather than assumed.
    #[must_use]
    pub fn all_resident(&self) -> bool {
        self.experts.is_none()
    }

    /// **Did this table come off the warm-boot artifact?** (design §7.)
    ///
    /// `true` says the host-side transform pipeline did not run for this
    /// load. What a caller reports as
    /// [`LoadFacts::weights_from_cache`](engine::load::LoadFacts).
    #[must_use]
    pub fn from_cache(&self) -> bool {
        self.from_cache
    }

    /// The digest of what is actually resident on the device.
    ///
    /// Not the size and not the source: the BYTES. What a gate compares
    /// between a cold load and a warm one, because "the same number of bytes
    /// from a file that claims the same key" is precisely the claim a cache
    /// is not allowed to make on its own word.
    ///
    /// # Errors
    ///
    /// A device failure reading the store back.
    pub fn digest(&self) -> Result<u64> {
        crate::weight_cache::digest_of(&self.store)
    }

    /// Write one adapter's planes into the banks (design §8).
    ///
    /// **A POOL WRITE AND A TABLE ROW, AND NOT A RECAPTURE** (decision 17).
    /// Nothing about the graph key is a function of a bank's contents — the
    /// key is the fire's composition — so this is a `cudaMemcpy` per plane
    /// onto an address that was reserved at load and will not move. The
    /// per-lane id is what a fire says afterwards, and it says it in a
    /// submission.
    ///
    /// **RE-REGISTERING ZEROES THE SLOT FIRST**, because the planes are
    /// full-capacity and a caller that skipped one would otherwise leave the
    /// previous adapter's plane in place beside the new one's — an adapter
    /// that is half of each. A bank this call does not name keeps whatever it
    /// held, which is what makes a per-site registration expressible; naming
    /// every site is the caller's business and `Fault::Adapter` names any
    /// bank it invented.
    ///
    /// # Errors
    ///
    /// [`Fault::Adapter`] for a bank this plan does not declare, an id past
    /// the bank's capacity, or a plane whose bytes are not one slot's;
    /// [`Fault::Device`] for the copy.
    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        // Checked whole before anything is written: a registration that
        // refuses halfway leaves a bank holding an adapter nobody described.
        for plane in planes {
            let bank = self.banks.get(plane.bank).ok_or_else(|| Fault::Adapter {
                bank: plane.bank.to_string(),
                why: "is not a bank this plan declares; a bank is a weight the model \
                      text marked `registered`, and this plan marked none by that name"
                    .to_string(),
            })?;
            if id >= bank.adapters {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} adapters and this registration is id {id}; capacity is \
                         a shape the model text declared, so the fix is the model text \
                         and not a retry",
                        bank.adapters
                    ),
                });
            }
            if plane.bytes.len() as u64 != bank.slot {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} bytes per adapter and this plane carries {}; a plane \
                         is one whole slot, zero-padded by the caller past its own rank",
                        bank.slot,
                        plane.bytes.len()
                    ),
                });
            }
        }
        for plane in planes {
            let bank = self.banks[plane.bank];
            let at = bank.offset + u64::from(id) * bank.slot;
            self.store
                .zero_span(at, usize::try_from(bank.slot).unwrap_or(0))?;
            self.store.write(at, plane.bytes)?;
        }
        Ok(())
    }

    /// The banks this load declared: name, capacity, and bytes per slot.
    /// What a caller sizes its planes against, and what a gate asserts on.
    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.banks
            .iter()
            .map(|(name, bank)| (name.as_str(), bank.adapters, bank.slot))
            .collect()
    }

    /// **THE BANKS, AS THE SHARED-ADAPTER RESOLVER READS THEM** — name,
    /// capacity, slot bytes, the slot's rectangle and its element size.
    ///
    /// [`Weights::banks`]'s longer twin, and a second method rather than a
    /// widened one because the two have different readers: a caller sizing a
    /// plane by hand wants three numbers, and [`crate::blob`] slicing a
    /// `[layers, ...]` file needs the shape to check the out-major statute
    /// against (alto adapter §6.3).
    #[must_use]
    pub fn seats(&self) -> Vec<BankSeat> {
        self.banks
            .iter()
            .map(|(name, bank)| BankSeat {
                name: name.clone(),
                adapters: bank.adapters,
                slot: bank.slot,
                rows: bank.rows,
                cols: bank.cols,
                elem: bank.elem,
            })
            .collect()
    }

    /// **HOW MANY ADAPTERS THIS LOAD CAN HOLD RESIDENT AT ONCE** — the
    /// smallest capacity any declared bank states, and zero for a plan that
    /// declares none.
    ///
    /// The SMALLEST, because an adapter occupies one slot of every bank it
    /// fills: a load whose `A` seats eight and whose `B` seats four holds
    /// four adapters, and a fifth would have nowhere to put its `B`. Model
    /// texts declare the two alike today and this is the honest reading of a
    /// text that does not.
    ///
    /// This is `slots` in the sense alto adapter §3.3 fixes it: CONCURRENT
    /// RESIDENCY, not a catalog. A hundred adapters may exist as files.
    #[must_use]
    pub fn adapter_seats(&self) -> u32 {
        self.banks
            .values()
            .map(|bank| bank.adapters)
            .min()
            .unwrap_or(0)
    }

    /// The table a fire resolves `Def::Weight(i)` through.
    #[must_use]
    pub fn table(&self) -> &WeightTable {
        &self.table
    }

    /// Every byte the store holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() as u64
    }
}

/// **Where a param's bytes actually are** — the device store, or the pinned
/// tier for a plane of a packed group the store does not hold.
///
/// One place asks the question, because a handle built off `store.at(offset)`
/// for a plane that reserved nothing there would name the NEXT param's bytes:
/// right-looking, resident, and wrong. `Tier::offloaded_at` answers the pinned
/// tier's address for a T1 group, the mapping's for a T2 one, and `None` for
/// everything the store does hold — which is every plane of every load that
/// does not stream a packed bank.
fn address(
    tier: Option<&crate::experts::Tier>,
    store: &Buffer,
    offset: u64,
    param: usize,
) -> Result<u64> {
    match tier.and_then(|tier| tier.offloaded_at(param)) {
        Some(elsewhere) => Ok(elsewhere),
        None => store.at(offset),
    }
}

/// One plane of a split-plane bank, as the mxfp4 select reads it: raw bytes.
///
/// The dtype is `U8` and the rectangle is the BYTE rectangle — see the seating
/// comment in [`Weights::resident`] for why both are the honest numbers and
/// not a widening.
fn packed(
    tier: Option<&crate::experts::Tier>,
    store: &Buffer,
    places: &[Place],
    param: usize,
) -> Result<Tensor> {
    let place = places[param];
    // **THE HANDLE'S RECTANGLE IS THE BYTE RECTANGLE**, made true rather
    // than assumed. The plan declares a plane in its own elements — a bf16
    // factor plane in factors, a four-bit code plane in codes — and the two
    // entries that read a plane's width back (the affine gather's group
    // recovery, the dense affine point's bit-width and grouping guards)
    // read it as bytes, because bytes are what a `U8`-bound handle can
    // honestly mean. Handing the element count through was the qwen4 first
    // light's one silent wrong number: the n-gram gather recovered group
    // eighty from a five-factor row and scaled forty-eight of every sixty
    // table columns by the wrong pair.
    let width = match place.dtype {
        // The shape already folds a 32-code block into sixteen bytes.
        Dtype::Mxfp4 => place.width,
        // Two codes to a byte; `plane_bytes` rounds the TOTAL up, and a
        // row's width is even for every whole-group bank this arm serves.
        Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => place.width.div_ceil(2),
        // One byte a code.
        Dtype::U8g64 => place.width,
        other => model_compiler::arena::elem_bytes(other)
            .and_then(|element| u32::try_from(element).ok())
            .map(|element| place.width.saturating_mul(element))
            .ok_or_else(|| Fault::Param {
                name: format!("param {param}"),
                why: "is a packed plane in a storage element that has no element size",
            })?,
    };
    Ok(Tensor::new(
        address(tier, store, place.offset, param)?,
        place.rows,
        width,
        Dtype::U8,
    ))
}

/// Where one param's plane sits in the store.
#[derive(Debug, Clone, Copy)]
struct Place {
    offset: u64,
    /// The plane's own bytes — the whole tensor the checkpoint publishes.
    bytes: u64,
    /// **What the DEVICE STORE gives it**, rounded up to the next handle
    /// alignment.
    ///
    /// `bytes.next_multiple_of(ALIGN)` for everything but a STREAMED routed
    /// bank, whose slab seats `resident` of its experts and whose remaining
    /// experts live in the pinned tier (alto design §7). So `bytes` is what
    /// the checkpoint publishes and this is what the device holds, and the
    /// two are the same number for every plane of a fully-resident load.
    reserved: u64,
    rows: u32,
    width: u32,
    dtype: Dtype,
}

/// The store's layout, decided before a byte is read.
///
/// STATED AHEAD, NOT ACCUMULATED, so that the length the checkpoint publishes
/// meets a length the plan predicted. A sink that allocated as it went would
/// take whatever arrived; this one refuses a plane that is not the size its
/// own declaration says it is, which is the only cheap check there is that a
/// contract and a plan describe the same model.
fn places(trace: &Trace, plan: &crate::experts::Plan) -> Result<Vec<Place>> {
    let bytes = plane_bytes(trace)?;
    let mut out = Vec::with_capacity(trace.params.len());
    let mut at = 0u64;
    for (index, param) in trace.params.iter().enumerate() {
        let (rows, width) = rectangle(&param.shape);
        let plane = bytes[index];
        // **A STREAMED BANK RESERVES ITS SLOTS AND NOT ITS EXPERTS** (alto
        // design §7, wave D2). `plane` stays the bank's WHOLE size — it is
        // what the checkpoint publishes and what the landing sink checks a
        // published tensor against — and `held` is what the device store
        // gives it: `resident` slots of the same stride, which is the slab
        // the indirection table points into. For every other param, and for
        // every param of an uncapped load, the two are the same number and
        // this line is the line that was here before.
        // **A PACKED GROUP THE STORE DOES NOT HOLD RESERVES NOTHING HERE.**
        // Its planes are not a slab over a source; they ARE the source — read
        // over UVA out of the pinned tier, or over HMM out of the mapped
        // artifact — so the store gives them no bytes at all and the plane
        // that follows them starts where they would have. `bytes` still says
        // what the checkpoint publishes, which is what the landing sink checks
        // an arriving tensor against.
        let held = if plan.streamed_whole(index) {
            0
        } else {
            match plan.resident(index) {
                Some(resident) if rows > 0 => plane / rows * u64::from(resident),
                _ => plane,
            }
        };
        out.push(Place {
            offset: at,
            bytes: plane,
            reserved: held.next_multiple_of(ALIGN),
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype: param.dtype,
        });
        at += held.next_multiple_of(ALIGN);
    }
    Ok(out)
}

/// A declared shape, read as `rows x width` — the IR's own rule, and the one
/// the arena carve reads too.
///
/// The numbers are honest rather than load-bearing: `linear.matmul` takes its
/// `m`, `n` and `k` from the ACTIVATION and the RESULT, never from the
/// weight, and the norm entries read a weight's pointer alone. So a handle
/// here says what the plan declared, which is what a `{:?}` in a panic should
/// print, and nothing reads it back as a promise.
fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
    }
}

/// The sink that puts each finalized tensor where the layout said it goes.
struct Landing<'a> {
    store: &'a mut Buffer,
    /// The pinned tier a streamed bank's plane lands in instead of the store.
    experts: Option<&'a crate::experts::Tier>,
    plan: &'a crate::experts::Plan,
    places: &'a [Place],
    index: &'a BTreeMap<&'a str, usize>,
    landed: Vec<bool>,
}

impl TensorSink for Landing<'_> {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> std::result::Result<(), LoadError> {
        let at = *self.index.get(name).ok_or_else(|| {
            LoadError::Contract(format!(
                "the load contract publishes `{name}`, which this plan does not \
                 name — the two were not written from each other"
            ))
        })?;
        let place = self.places[at];
        if bytes.len() as u64 != place.bytes {
            return Err(LoadError::Contract(format!(
                "`{name}` lands {} bytes and the plan declares {} — a plane read \
                 at the wrong width is a model that computes",
                bytes.len(),
                place.bytes
            )));
        }
        // **A STREAMED BANK GOES TO T1 AND NOT TO THE STORE** (alto design
        // §7). Pinned is the authoritative copy of every expert; the slab is
        // filled from it afterwards, and a table entry for a non-resident
        // expert names these very bytes. `Pinned::write` answers `false` for
        // a span past the allocation, which cannot happen — the tier was
        // opened from the same plan this offset came from — and is turned
        // into a sentence rather than swallowed.
        // **A MAPPED GROUP'S BYTES GO NOWHERE.** They are already on disk in
        // the artifact this load maps, in exactly the form the executor just
        // computed — that is what the key asserts, and it is streaming §0's
        // precondition read from the landing side. Writing them into the store
        // is impossible (it reserved nothing for them) and writing them into
        // the pinned tier is the tier the host budget just refused. So the
        // plane is COUNTED AS LANDED and dropped, which is honest: the load
        // did produce it, and the copy it would have made is the one T2 exists
        // to avoid.
        //
        // (The transform still RAN, which is a cost this wave pays and names:
        // landing the resident planes straight out of the artifact — the
        // executor never running at all — is the next step and is written up
        // in this module's header.)
        if self.plan.mapped(at) {
            self.landed[at] = true;
            return Ok(());
        }
        let streamed = self.plan.resident(at).is_some() || self.plan.pinned(at);
        match self.experts.filter(|_| streamed) {
            Some(tier) => {
                let host_at = tier.host_offset(at).ok_or_else(|| {
                    LoadError::Internal(format!(
                        "`{name}` is a streamed routed bank the tier did not seat"
                    ))
                })?;
                if !tier
                    .host()
                    .write(usize::try_from(host_at).unwrap_or(usize::MAX), bytes)
                {
                    return Err(LoadError::Internal(format!(
                        "`{name}` does not fit the pinned tier at offset {host_at}"
                    )));
                }
            }
            None => self
                .store
                .write(place.offset, bytes)
                .map_err(|fault| LoadError::Internal(fault.to_string()))?,
        }
        self.landed[at] = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use model_dsl::Platform;

    use super::*;

    #[test]
    fn the_store_is_laid_out_aligned_disjoint_and_in_plan_order() {
        let trace =
            models::trace_of("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships the SKU");
        let trace = trace(Platform::Cuda);
        let places = places(&trace, &crate::experts::Plan::default())
            .expect("every param of a bf16 SKU has an element size");

        assert_eq!(places.len(), trace.params.len());
        let mut end = 0u64;
        for (place, param) in places.iter().zip(&trace.params) {
            assert!(place.offset >= end, "`{}` overlaps its predecessor", param.name);
            assert_eq!(place.offset % ALIGN, 0, "`{}` is misaligned", param.name);
            assert!(place.bytes > 0, "`{}` reserves nothing", param.name);
            end = place.offset + place.reserved;
        }

        // The embedding is the SKU's largest plane and its first: 248320
        // rows of 1024 bf16, and the head is tied to it, so it is landed once.
        assert_eq!(places[0].offset, 0);
        assert_eq!(places[0].rows, 248_320);
        assert_eq!(places[0].width, 1024);
        assert_eq!(places[0].bytes, 248_320 * 1024 * 2);
    }

    /// **THE THREE REFUSALS A WARM-ONLY BOOT CAN PRINT, IN ONE PLACE** (§M-3).
    ///
    /// The messages are the deliverable of this wave as much as the code is:
    /// an operator meeting one of them has a deployment that will not start,
    /// and what they do next is whatever the sentence says. So the sentence is
    /// asserted rather than inspected, and asserted HERE, on the host, with no
    /// device and no checkpoint — because a claim that only a `-- --ignored`
    /// GPU gate can check is a claim that goes stale between waves.
    ///
    /// Two of the three are built by [`absent`], which is the one this wave
    /// ADDS: the census that turns M-3's silent miss into the loudest thing
    /// the boot says. The third is the shape every refused-file message shares
    /// ([`tier::refuse`](crate::weight_cache::tier::refuse)), and it is checked
    /// through the stale-format wording because that arm needs no bytes.
    #[test]
    fn the_refusals_name_what_is_wrong_and_the_command_that_fixes_it() {
        use crate::weight_cache::tier;

        let dir = std::env::temp_dir().join(format!(
            "pie-refusals-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |since| since.as_nanos()),
        ));
        std::fs::create_dir_all(&dir).expect("a temporary directory");
        let source = Path::new("/models/gpt-oss-20b.zt");
        let key = 0x1234_5678_9abc_def0u64;

        // ── (1) NEVER PREPARED. An empty directory is a deployment that has
        //    not been imported on this box, and nothing else.
        let said = absent(&dir, key, &tier::path(&dir, key), source);
        assert!(
            said.contains("has never been prepared on this machine"),
            "{said}"
        );
        assert!(
            said.contains("pie model import --prepare-only /models/gpt-oss-20b.zt"),
            "the remedy is spelled against the checkpoint the load names: {said}"
        );
        assert!(
            said.contains("no cold serving path left"),
            "and it says there is nowhere else to go: {said}"
        );

        // ── (2) THE PLAN OR THE RECIPE CHANGED. The same absence with other
        //    files beside it is a DIFFERENT operator situation and the wave's
        //    whole finding: the key is the filename, so a changed model text
        //    orphans a hundred gigabytes under a name nothing will open again.
        let stale = tier::path(&dir, 0xdead_beef_dead_beefu64);
        std::fs::write(&stale, b"not a real artifact, and this door reads no bytes")
            .expect("a sibling artifact");
        let said = absent(&dir, key, &tier::path(&dir, key), source);
        assert!(
            said.contains("1 other serving artifact"),
            "the census counts them: {said}"
        );
        assert!(
            said.contains("deadbeefdeadbeef.tiers"),
            "and NAMES them, which is the whole point: {said}"
        );
        assert!(
            said.contains("a changed key"),
            "and says why this deployment stopped naming that file: {said}"
        );
        assert!(
            said.contains("nothing here deletes it"),
            "and that it will not tidy anything away on its own (§M.4) — said \
             ONCE, in the shared clause, rather than twice: {said}"
        );
        assert_eq!(
            said.matches("deletes").count(),
            1,
            "the census must not repeat the policy the shared tail states: {said}"
        );
        assert!(said.contains(".tiers (0.0 MiB)"), "sub-GiB reads as MiB: {said}");
        assert!(
            said.contains("pie model import --prepare-only /models/gpt-oss-20b.zt"),
            "same remedy, same shape: {said}"
        );

        // ── (3) A FILE THAT WILL NOT BE READ. The third shape, checked
        //    through the arm that needs no payload.
        let said = tier::refuse(
            &tier::path(&dir, key),
            Some(source),
            "states format 2 and this build reads 3, so its images cannot be cut by this one",
        );
        assert!(said.contains("states format 2"), "{said}");
        assert!(
            said.contains("nothing here rewrites it and nothing here deletes it"),
            "the middle clause is the §M.4 policy, said the same way every time: {said}"
        );
        assert!(
            said.contains("pie model import --prepare-only /models/gpt-oss-20b.zt"),
            "{said}"
        );

        // ── AND THE ONE THAT IS NOT ABOUT A FILE. No cache directory is a
        //    config to change, not an artifact to rebuild, so it names the
        //    key an operator has to set.
        let Miss::Refused(said) = unkeyed(None, source) else {
            panic!("a missing directory is not an absent file");
        };
        assert!(said.contains("weight_cache_dir"), "{said}");
        assert!(
            said.contains("pie model import --prepare-only /models/gpt-oss-20b.zt"),
            "{said}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
