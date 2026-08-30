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
//! that reached for `model::qwen_3` would be a shell that has to grow an arm
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
use checkpoint::plan::{LoadPlan, StorageTarget, compile};
use checkpoint::types::{BackendKind, ScaleForm, TensorId};
use kernels_cuda::Tensor;
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
/// (`Dtype::Fp4`, a nibble with no declared packing): a plane whose length
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
                Dtype::MlxU4 => rows.saturating_mul(width).div_ceil(2),
                // The same bank at eight bits: one whole byte a code, so the
                // logical rectangle IS the byte count and nothing rounds.
                Dtype::MlxU8 => rows.saturating_mul(width),
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
    let mut planes = Attachments::new();
    for (name, pairing) in pairings(&landing, &index)? {
        let Some(&at) = index.get(name) else {
            continue;
        };
        planes.insert(at, vec![pairing.scales]);
    }
    Ok(Prospect {
        planes,
        resident_key: resident_key(trace, &landing, path)?,
    })
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

/// **Open the T2 source for this load, or answer why there is none.**
///
/// The artifact a FULLY RESIDENT load of this deployment wrote, under
/// [`Prospect::resident_key`]. `None` for every load that was offered no cache
/// directory, and for one whose deployment has not booted uncapped yet — which
/// is not an error here: it becomes
/// [`Residency::admit_tiers`](engine::load::Residency::admit_tiers)'s refusal,
/// with the sentence that says what to do about it.
///
/// **THE KEY IS CHECKED AGAINST THE FILE'S OWN.** `Artifact::open` validates
/// the format, the index and the lengths, and the key lives in the FILENAME —
/// so a file moved or renamed under a key it does not carry would open and
/// serve another deployment's weights. The header states its key; it is read
/// back and compared here.
#[must_use]
pub fn spill_source(cache_dir: Option<&Path>, key: u64) -> Option<crate::weight_cache::Artifact> {
    if key == 0 {
        return None;
    }
    let dir = cache_dir?;
    let artifact = crate::weight_cache::Artifact::open(&crate::weight_cache::artifact_path(dir, key))
        .ok()?;
    (artifact.key() == key).then_some(artifact)
}

/// One quantized weight's other plane, as a row of [`places`].
#[derive(Debug, Clone, Copy)]
struct Pairing {
    scales: usize,
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
        match attachment.scale_form {
            ScaleForm::RawE8M0 => {}
            ScaleForm::Bf16AffineFactors | ScaleForm::F32Factors => {
                return Err(Fault::Param {
                    name: (*name).to_string(),
                    why: "carries a scale form no point this shell stamps reads. The \
                          cuda plane has one split-plane bank: mxfp4 e2m1 codes under \
                          raw e8m0 exponents. An affine bank is `code * scale + zero` \
                          and seating it without its zero points is the right spread \
                          around the wrong centre — a model that computes and is wrong \
                          — so it is refused here instead",
                });
            }
        }
        out.insert(
            *name,
            Pairing {
                scales: row(
                    attachment.scale_tensor,
                    "is a quantized weight whose scales this plan does not publish as a \
                     param of their own",
                )?,
            },
        );
    }
    Ok(out)
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
    /// `Boot` (article 9: shells read no environment). `None` turns the
    /// feature off entirely — no reads, no writes. With a directory, the
    /// device table is keyed on this load's whole recipe
    /// ([`weight_cache::Identity`]) and, on a match, read STRAIGHT TO THE
    /// DEVICE: the plan compile still runs (it is milliseconds, and it is what
    /// validates the contract against the checkpoint), and everything after it
    /// — the executor's host-side dequant, cast and repack, and the
    /// per-tensor uploads — does not. A cold load writes the artifact on its
    /// way out, and declines the write rather than filling the disk.
    ///
    /// A corrupt artifact is never trusted and never silently retried: it is
    /// counted, said out loud, deleted, and followed by the full load.
    pub fn resident(
        trace: &Trace,
        contract: &ModelContract,
        path: &Path,
        cache_dir: Option<&Path>,
        plan: crate::experts::Plan,
        stream: *mut core::ffi::c_void,
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
        let landing = compile(&metadata, contract, target)?;

        let places = places(trace, &plan)?;
        let total = places.last().map_or(0, |p| p.offset + p.reserved);
        let mut store = Buffer::zeroed(usize::try_from(total).unwrap_or(usize::MAX))?;
        // **THE TIER IS OPENED BEFORE THE BYTES ARRIVE**, because a streamed
        // bank's plane does not land in the store at all: it lands in the
        // PINNED tier, whole, and the slab takes a copy of its first
        // `resident` slots afterwards. `None` for the degenerate plan, and
        // then nothing below this line is different from what it was.
        // ── THE T2 SOURCE, OPENED BEFORE THE TIER THAT POINTS INTO IT.
        //    The artifact a fully-resident load of this deployment wrote, found
        //    under the key that load would have formed — `resident_key`'s doc
        //    says why a capped load can name it. `None` for a plan that spills
        //    nothing, which is every plan whose host budget held its groups.
        let source = match plan.spill_demand() > 0 {
            true => spill_source(cache_dir, resident_key(trace, &landing, path)?),
            false => None,
        };
        let mut experts = match plan.streams() {
            true => Some(crate::experts::Tier::open(plan.clone(), source)?),
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
        // **A STREAMED LOAD FORMS NO KEY.** The artifact is a snapshot of the
        // DEVICE STORE, and a streamed load's store is a cache over a pinned
        // tier the artifact says nothing about — so restoring one would fill
        // the slabs and leave T1 empty, which is a table whose non-resident
        // entries point at zeros. The tiers get their own artifact the day
        // T2's background prefetch does (design §7); until then a load that
        // streams neither reads nor writes the cache, and says so here rather
        // than by a digest mismatch later.
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

        let from_cache = match key {
            Some(key) => crate::weight_cache::restore(cache_dir, key, &mut store)?,
            None => false,
        };

        // The plan's own name -> row map. Read by the landing sink to place an
        // arriving tensor and by the pairing below to resolve an attachment's
        // planes onto rows, so it is stated once above both rather than inside
        // the branch that used to be its only reader.
        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();

        let landed = if from_cache {
            // EVERY PARAM IS LANDED, because the blob is the whole table —
            // the layout is part of the key, so a restore that matched wrote
            // exactly the bytes this `places` describes.
            vec![true; places.len()]
        } else {
            // The executor's own arena: where a transform's intermediates live
            // while it runs. Host memory, because the transforms run host-side;
            // it is dropped the moment the load is over, and only the finalized
            // tensors the sink took survive.
            let mut scratch = vec![0u8; usize::try_from(landing.memory.arena_bytes()).unwrap_or(0)];
            let mut backing: &mut [u8] = &mut scratch;

            let mut sink = Landing {
                store: &mut store,
                experts: experts.as_ref(),
                plan: &plan,
                places: &places,
                index: &index,
                landed: vec![false; places.len()],
            };
            Execution::new(&landing, snapshot)
                .arena(&mut backing)
                .sink(&mut sink)
                .run()?;

            let landed = sink.landed;
            drop(scratch);
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
                Some(pairing) => WeightRow::Planes {
                    codes: packed(experts.as_ref(), &store, &places, at)?,
                    scales: packed(experts.as_ref(), &store, &places, pairing.scales)?,
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
            tier.land(&slabs, stream)?;
        }
        Ok(Weights {
            store,
            table: WeightTable(table),
            banks: banks(trace, &places),
            experts,
            from_cache,
        })
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
    Ok(Tensor::new(
        address(tier, store, place.offset, param)?,
        place.rows,
        place.width,
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
            model::trace_of("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships the SKU");
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
}
