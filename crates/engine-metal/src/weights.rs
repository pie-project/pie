//! The checkpoint, resident: one device allocation, one row per
//! `Trace::params` — and, in the same allocation and table, the adapter
//! banks (kept here, not in `store/`, since the table must resolve
//! `Def::Weight` in one place and a bank row belongs to no sequence).
//!
//! A streamed load still gets one allocation and one table: only the
//! reservation for the bands it streams shrinks. Weight rows are `u32`
//! handles into [`Handles`](crate::device::Handles), load-lived and sealed by the caller.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;

// The two readers; `zt` comes in by module since its door is spelled `parse`.
use checkpoint::file::read::parse_metadata;
use checkpoint::file::Metadata;
use checkpoint::file::serve;
use checkpoint::file::zt;
use checkpoint::contract::{ModelContract, TensorContract};
use checkpoint::error::Error as LoadError;
use checkpoint::executor::{Execution, sink::TensorSink};
// The warm arm's reader: `Artifact::open` maps and hashes nothing, `locate`
// answers where every plane lies, `plane` the published bytes there.
use checkpoint::file::serve::Artifact;
use checkpoint::plan::{LoadPlan, StorageTarget, compile, compile_streaming};
// `Stamp::of` fills the four policy fields from the profile's constants;
// `Mismatch::refuse` is the one place the refusal is written.
use checkpoint::serving::{self, Stamp};
use checkpoint::types::{BackendKind, ScaleForm, TensorId};
use kernels_metal::Tensor;
use model_ir::{Dtype, ParamSource, Trace};

use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};
use crate::experts::{Attachments, Plan, Source, Tier};
use crate::gather;
use crate::host_source::HostSource;
use crate::weight_store::Store;
use crate::mapping::{self, Mapping};
use crate::run::{WeightRow, WeightTable};

/// What a matrix operand wants, and what a Metal buffer's own base is
/// already aligned to.
pub(crate) const ALIGN: u64 = 256;

/// One plane of a registered adapter. Full capacity, not the adapter's
/// own rank: zero-padded by the caller for a lower-rank adapter.
#[derive(Debug, Clone, Copy)]
pub struct AdapterPlane<'a> {
    /// The bank param this plane fills, as `Trace::params` names it.
    pub bank: &'a str,
    /// One slot's worth of bytes.
    pub bytes: &'a [u8],
}

/// Every weight this model needs, on the device — the checkpoint's and the banks'.
#[derive(Debug)]
pub struct Weights {
    /// The writable reservation: the whole store for a cold load, or (for
    /// warm) only what this shell still writes, since checkpoint planes
    /// are `PROT_READ` views onto [`Weights::mapped`].
    store: Store,
    /// The artifact itself, mapped and bound — non-empty for a warm load.
    /// A `Vec` and not one buffer, since `maxBufferLength` bounds one buffer.
    mapped: Vec<Buffer>,
    table: WeightTable,
    /// The routed-expert tier, opened only for a streamed load. `None` is
    /// full residency. `RefCell` since the seat swap happens inside the walk.
    tier: Option<RefCell<Tier>>,
    /// The gathered row slab, opened only for a load that gathers. `None`
    /// is a resident table, the common case.
    rows: Option<RefCell<gather::Slab>>,
    /// Bank name -> (param index, adapters, bytes per adapter slot), built
    /// off `ParamSource::Registered`.
    banks: BTreeMap<String, Bank>,
    /// How much of this load the warm arm computed rather than read —
    /// `(planes, bytes)`, `(0, 0)` for a cold load.
    residue: (usize, u64),
    /// Banks decoded to dense bf16 at load for the ops that have no
    /// quantized arm (`crate::decoded`), each its own buffer.
    decoded: Vec<Buffer>,
}

/// One declared adapter bank: where its slots are and how big they are.
#[derive(Debug, Clone, Copy)]
struct Bank {
    /// Where the bank's first slot starts in the store.
    offset: u64,
    /// How many adapters the first axis seats — the ceiling
    /// `Budget::max_adapters` is checked against.
    adapters: u32,
    /// One adapter's bytes in this bank.
    slot: u64,
    rows: u64,
    cols: u64,
    elem: u64,
}

/// One bank, as the adapter resolver reads it. A flattened [`Bank`],
/// since a private field is not a contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankSeat {
    /// The param's own name, which is what a registration names.
    pub name: String,
    pub adapters: u32,
    pub slot: u64,
    /// The leading axis of one slot.
    pub rows: u64,
    /// The trailing axes of one slot, multiplied out.
    pub cols: u64,
    pub elem: u64,
}


/// The banks a plan declares, read off `ParamSource::Registered`. `seat`
/// answers where a bank's first slot sits, the one thing the two load
/// arms disagree on, as a parameter rather than a branch.
fn banks(
    trace: &Trace,
    places: &[Place],
    seat: impl Fn(usize) -> u64,
) -> BTreeMap<String, Bank> {
    trace.params
        .iter()
        .zip(places)
        .enumerate()
        .filter(|(_, (param, _))| param.source == ParamSource::Registered)
        .map(|(at, (param, place))| {
            let adapters =
                u32::try_from(param.shape.first().copied().unwrap_or(0)).unwrap_or(u32::MAX);
            // The declared plane: a registered bank is never routed, so
            // `full` and the declared size agree.
            let slot = if adapters == 0 {
                0
            } else {
                place.full / u64::from(adapters)
            };
            // The slot's rectangle: the param's shape with the adapters axis cut off.
            let (rows, cols) = rectangle(param.shape.get(1..).unwrap_or(&[]));
            (
                param.name.clone(),
                Bank {
                    offset: seat(at),
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

impl Weights {
    /// Land `contract` against the checkpoint at `path` (a snapshot
    /// directory or a single container). The caller seals `handles` once
    /// every load-time row is minted.
    ///
    /// # Errors
    ///
    /// [`Fault::Recipe`] for a serving artifact stamped for another
    /// deployment, [`Fault::Load`] for a checkpoint the contract does not
    /// fit, [`Fault::Param`] for a plan/contract mismatch,
    /// [`Fault::Ceiling`]/[`Fault::Device`]/[`Fault::Deviceless`] on
    /// reservation failure.
    pub fn resident(
        device: &Context,
        handles: &Handles,
        trace: &Trace,
        contract: &ModelContract,
        path: &Path,
        plan: &Plan,
    ) -> Result<Weights> {
        // Before the metadata parse: this refusal is about the trace, not
        // the file, so it must not read as a checkpoint-side sentence.
        readable_plane_orders(trace)?;
        let gather = plan.gathered();

        // Still before anything is reserved (a cross-recipe boot must cost
        // one manifest read and nothing else).
        serves_this_deployment(path, trace.platform.backend(), &trace.name)?;

        let (metadata, snapshot) = if path.is_dir() {
            (parse_metadata(path)?, path)
        } else {
            (zt::parse(path)?, path.parent().unwrap_or(Path::new(".")))
        };

        // tp=1: `Shard::Cut` segments still describe the whole tensor, and
        // a rank of one takes all of them.
        let target = StorageTarget::for_backend(BackendKind::Metal, 0, 1);
        let landing = compile(&metadata, contract, target.clone())?;

        let places = places(trace, plan, plan.gathered())?;
        let index: BTreeMap<&str, usize> = trace
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        // Read before either arm, since both seat the same banks: which
        // three rows make one quantized weight is a fact about the
        // compiled load plan, not about where the bytes came from.
        let pairings = pairings(&landing, &index)?;

        // The warm arm, tried before anything is reserved: a serving
        // artifact whose stamp agreed holds every plane already landed, so
        // the whole load is map-and-mint, no reservation or copy. Any miss
        // falls back to the cold path and says why, as a sentence rather
        // than a fault — a raw snapshot or an ordinary `.zt` is expected to miss.
        match warm(
            device,
            handles,
            trace,
            contract,
            &metadata,
            snapshot,
            target.clone(),
            path,
            plan,
            &places,
            &pairings,
        ) {
            Ok(weights) => return Ok(weights),
            // `None`: never claimed to be a serving artifact (ordinary
            // load, nothing to log). `Some`: claimed it and could not be
            // served whole — the surprising case, logged below.
            Err(None) => {}
            Err(Some(why)) => {
                let held = std::fs::metadata(path).map(|it| it.len()).unwrap_or(0);
                let staging = std::env::temp_dir();
                eprintln!(
                    "engine-metal: {} loads cold — {why}.\n\
                     engine-metal: that road reads all {:.2} GiB of it through the executor \
                     and stages {:.2} GiB of routed bands under {}, where {} free; the warm \
                     arm would have read no weight byte at all.",
                    path.display(),
                    held as f64 / (1u64 << 30) as f64,
                    plan.source_bytes() as f64 / (1u64 << 30) as f64,
                    staging.display(),
                    match crate::host_source::free_bytes(&staging) {
                        Some(free) => format!("{:.2} GiB is", free as f64 / (1u64 << 30) as f64),
                        None => "an unknown amount is".to_string(),
                    },
                );
            }
        }

        let spans: Vec<(u64, u64)> = places.iter().map(|p| (p.offset, p.reserved)).collect();
        let mut store = Store::zeroed(device, &spans, device.max_buffer())?;

        // The host band table, sized off the plan and empty for a
        // full-residency load. A mapping, not a `Vec`, so the kernel can
        // page these bytes back to an unlinked file rather than swap.
        let mut host = HostSource::open(plan.source_bytes())?;
        // The gathered class's own staging; a load that gathers nothing
        // opens zero bytes and is untouched.
        let mut table = HostSource::open(gather.source_bytes())?;
        let mut sink = Landing {
            store: &mut store,
            host: &mut host,
            table: &mut table,
            plan,
            gather,
            places: &places,
            index: &index,
            landed: vec![false; places.len()],
        };
        // A streamed (or gathered) load hands no arena: each tensor goes
        // straight to the device store or host table, rather than staging
        // the whole image. Compiled a second time for this; the second
        // plan is dropped here and never read by `pairings` below.
        let landed = if plan.streams() || gather.gathers() {
            let streaming = compile_streaming(&metadata, contract, target)?;
            Execution::new(&streaming, snapshot)
                .streaming()
                .sink(&mut sink)
                .run()?;
            sink.landed
        } else {
            // The executor's arena: host memory, since transforms run
            // host-side; dropped once the load is over.
            let mut scratch = vec![0u8; usize::try_from(landing.memory.arena_bytes()).unwrap_or(0)];
            let mut backing: &mut [u8] = &mut scratch;
            Execution::new(&landing, snapshot)
                .arena(&mut backing)
                .sink(&mut sink)
                .run()?;
            let landed = sink.landed;
            drop(scratch);
            landed
        };

        let mut weight_table = Vec::with_capacity(places.len());
        for (at, place) in places.iter().enumerate() {
            // A registered plane is one the checkpoint does not have — it is
            // already reserved and zeroed, and `register_adapter` fills it later.
            if !landed[at] && trace.params[at].source == ParamSource::Checkpoint {
                return Err(Fault::Param {
                    name: trace.params[at].name.clone(),
                    why: "is a plan param the load contract never published",
                });
            }
            let dense = |place: &Place| -> Result<Tensor> {
                Ok(Tensor::new(
                    store.bind(handles, place.offset, place.bytes)?,
                    place.rows,
                    place.width,
                    place.dtype,
                ))
            };
            weight_table.push(Some(match pairings.get(trace.params[at].name.as_str()) {
                Some(pairing) => WeightRow::Planes(kernels_metal::Bank {
                    codes: dense(place)?,
                    scales: dense(&places[pairing.scales])?,
                    biases: pairing.biases.map(|at| dense(&places[at])).transpose()?,
                    group: pairing.group,
                    bits: pairing.bits,
                }),
                None => WeightRow::Dense(dense(place)?),
            }));
        }
        // Opened after the landing, before the first fire: host table full,
        // store seats reserved and zeroed, no command buffer committed yet.
        let offsets: Vec<u64> = places.iter().map(|place| place.offset).collect();
        let tier = plan
            .streams()
            .then(|| Tier::open(plan, &store, Source::landed(plan, host), &offsets).map(RefCell::new))
            .transpose()?;
        // Same instant as the tier, but seeds nothing (a hashed row space
        // has no likely prefix).
        let rows = gather
            .gathers()
            .then(|| {
                gather::Slab::open(gather, &store, Source::from_host(table, gather.host_bands()), &offsets)
                    .map(RefCell::new)
            })
            .transpose()?;
        Ok(Weights {
            store,
            mapped: Vec::new(),
            table: WeightTable(weight_table),
            tier,
            rows,
            banks: banks(trace, &places, |at| places[at].offset),
            // A cold load computes nothing separately: everything lands together.
            residue: (0, 0),
            decoded: Vec::new(),
        })
    }

    /// Did this load map its checkpoint instead of reading it? `true` is a
    /// warm load, every plane served off its own mapped pages.
    #[must_use]
    pub fn warm(&self) -> bool {
        !self.mapped.is_empty()
    }

    /// How many reservations the mapped artifact took, `0` for a cold load.
    #[must_use]
    pub fn windows(&self) -> usize {
        self.mapped.len()
    }

    /// What this load computed rather than read — `(planes, bytes)`, `(0, 0)` if nothing.
    #[must_use]
    pub fn residue(&self) -> (usize, u64) {
        self.residue
    }

    /// The gathered row slab this load opened, or `None` if tables are
    /// resident.
    #[must_use]
    pub fn rows(&self) -> Option<&RefCell<gather::Slab>> {
        self.rows.as_ref()
    }

    /// The routed-expert tier this load opened, or `None` for full residency.
    #[must_use]
    pub fn tier(&self) -> Option<&RefCell<Tier>> {
        self.tier.as_ref()
    }

    /// Write one adapter's planes into the banks. Re-registering zeroes
    /// the slot first; a bank this call does not name keeps whatever it held.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a bank this plan does not declare, an id
    /// past capacity, or a plane whose bytes are not one slot's;
    /// [`Fault::Ceiling`]/[`Fault::Deviceless`] on reservation failure.
    pub fn register_adapter(&mut self, id: u32, planes: &[AdapterPlane<'_>]) -> Result<()> {
        // Checked whole before anything is written: a registration that
        // refuses halfway leaves a bank holding an adapter nobody described.
        for plane in planes {
            let bank = self.banks.get(plane.bank).ok_or_else(|| Fault::Adapter {
                bank: plane.bank.to_string(),
                why: "not a bank this plan declares; a bank is a weight the model text \
                      marked `registered`, and this plan marked none by that name"
                    .to_string(),
            })?;
            if id >= bank.adapters {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} adapters while this registration is id {id}; capacity \
                         is a shape the model text declared, so the fix is the model \
                         text and not a retry",
                        bank.adapters
                    ),
                });
            }
            if plane.bytes.len() as u64 != bank.slot {
                return Err(Fault::Adapter {
                    bank: plane.bank.to_string(),
                    why: format!(
                        "seats {} bytes per adapter while this plane carries {}; a plane \
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
            self.store.zero_span(at, bank.slot)?;
            self.store.write(at, plane.bytes)?;
        }
        Ok(())
    }

    /// The banks this load declared: name, capacity, and bytes per slot.
    #[must_use]
    pub fn banks(&self) -> Vec<(&str, u32, u64)> {
        self.banks
            .iter()
            .map(|(name, bank)| (name.as_str(), bank.adapters, bank.slot))
            .collect()
    }

    /// The banks, as the adapter resolver reads them. [`Weights::banks`]'s
    /// longer twin, for [`crate::adapter::planes_of`].
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

    /// How many adapters this load can hold resident at once — the
    /// smallest capacity any declared bank states, zero if none.
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

    /// Decode, once, every split-plane bank an op with no quantized arm
    /// reads as one dense plane (the MLA absorbs' `kv_b`), and rebind its
    /// row as that plane. Runs before `Handles::seal`, so the minted handles
    /// live as long as the load.
    ///
    /// # Errors
    ///
    /// [`Fault::Param`] for a bank this decoder does not unpack,
    /// [`Fault::Device`]/[`Fault::Ceiling`] for the buffer.
    pub fn decode_absorbed(
        &mut self,
        device: &Context,
        handles: &Handles,
        trace: &Trace,
    ) -> Result<()> {
        for at in crate::decoded::absorbed_weights(trace) {
            let Some(Some(WeightRow::Planes(bank))) = self.table.0.get(at).copied() else {
                continue;
            };
            let param = &trace.params[at];
            let (n, k) = match param.shape.as_slice() {
                [n, k] => (*n as usize, *k as usize),
                _ => {
                    return Err(Fault::Param {
                        name: param.name.clone(),
                        why: "is not a two-axis plane, the only shape decoded at load",
                    });
                }
            };
            let bits = bank.bits as usize;
            let group = bank.group as usize;
            let codes = handles.read(bank.codes.buf, (n * k * bits / 8) as u64)?;
            let scales = handles.read(bank.scales.buf, (n * k / group * 2) as u64)?;
            let biases = bank
                .biases
                .map(|b| handles.read(b.buf, (n * k / group * 2) as u64))
                .transpose()?;
            let plane = crate::decoded::decode_affine(
                &codes,
                &scales,
                biases.as_deref(),
                n,
                k,
                group,
                bits,
            )
            .map_err(|why| Fault::Device {
                call: "decode_absorbed",
                why: format!("{}: {why}", param.name),
            })?;
            let mut buffer = Buffer::zeroed(device, plane.len() as u64)?;
            buffer.write(0, &plane)?;
            let handle = handles.bind(&buffer, 0, plane.len() as u64)?;
            self.table.0[at] = Some(WeightRow::Dense(Tensor::new(
                handle,
                u32::try_from(n).unwrap_or(u32::MAX),
                u32::try_from(k).unwrap_or(u32::MAX),
                Dtype::Bf16,
            )));
            self.decoded.push(buffer);
        }
        Ok(())
    }

    /// Every byte this load's weight reservations hold: the store for a
    /// cold load, or the whole mapped file plus banks for a warm one.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.store.bytes() + self.mapped.iter().map(Buffer::bytes).sum::<u64>()
    }
}

/// A routed bank's other device planes, off the load plan — read
/// structurally (via [`pairings`]), never off a name.
///
/// # Errors
///
/// [`Fault::Load`] for a checkpoint the contract does not fit,
/// [`Fault::Param`] for an attachment this plan cannot resolve.
///
/// [`experts::Plan::of`]: crate::experts::Plan::of
pub fn attachments(trace: &Trace, contract: &ModelContract, path: &Path) -> Result<Attachments> {
    readable_plane_orders(trace)?;
    let metadata = if path.is_dir() {
        parse_metadata(path)?
    } else {
        zt::parse(path)?
    };
    let landing = compile(
        &metadata,
        contract,
        StorageTarget::for_backend(BackendKind::Metal, 0, 1),
    )?;
    let index: BTreeMap<&str, usize> = trace
        .params
        .iter()
        .enumerate()
        .map(|(at, param)| (param.name.as_str(), at))
        .collect();
    let mut out = Attachments::new();
    for (name, pairing) in pairings(&landing, &index)? {
        let Some(&at) = index.get(name) else {
            continue;
        };
        let mut planes = vec![pairing.scales];
        planes.extend(pairing.biases);
        out.insert(at, planes);
    }
    Ok(out)
}

/// One quantized weight's other planes, as rows of [`places`].
#[derive(Debug, Clone, Copy)]
struct Pairing {
    scales: usize,
    /// The zero points, for an affine scheme; `None` for a symmetric one.
    biases: Option<usize>,
    group: u32,
    bits: u32,
}

/// Which params are quantized weights, and what their other planes are.
/// Read structurally off `LoadPlan::attachments`, never off a `.scales`/
/// `.biases` name suffix.
fn pairings<'a>(
    landing: &'a LoadPlan,
    index: &BTreeMap<&str, usize>,
) -> Result<BTreeMap<&'a str, Pairing>> {
    // Keyed by the id's number: `TensorId` is `Hash`/`Eq` but not `Ord`.
    let named: BTreeMap<u32, &str> = landing
        .tensors
        .iter()
        .map(|decl| (decl.id.0, decl.name.as_str()))
        .collect();
    let mut out = BTreeMap::new();
    for attachment in &landing.attachments {
        let Some(of) = named.get(&attachment.tensor.0) else {
            continue;
        };
        // Only a plane the trace declares becomes a weight row.
        if !index.contains_key(of) {
            continue;
        }
        let name = of;
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
        let bits = match attachment.scale_form {
            // E2M1 is four bits by its own name; mxfp4 has no affine point.
            ScaleForm::RawE8M0 => 4,
            ScaleForm::Bf16AffineFactors => match landing.affine_point_of(name) {
                Some((_, bits)) => bits,
                None => {
                    return Err(Fault::Param {
                        name: (*name).to_string(),
                        why: "carries affine scale factors and no quantized encoding for \
                              them to be factors OF; the point a kernel is selected at is \
                              the tensor's own `QuantSpec`, and this one has none",
                    });
                }
            },
            // F32Factors expands to f32 multipliers no kernel this shell
            // stamps reads: scales are always read in the stored width.
            ScaleForm::F32Factors => {
                return Err(Fault::Param {
                    name: (*name).to_string(),
                    why: "wants its scales expanded to f32 factors, and every quantized \
                          point this shell stamps reads them in the width they are stored",
                });
            }
        };
        // An affine bank without its zero points is refused, not
        // downgraded: `code * scale` alone computes but is silently wrong.
        let biases = match (attachment.scale_form, attachment.zero_point_tensor) {
            (_, Some(id)) => Some(row(
                id,
                "is an affine bank whose zero points this plan does not publish as a \
                 param of their own",
            )?),
            (ScaleForm::Bf16AffineFactors, None) => {
                return Err(Fault::Param {
                    name: (*name).to_string(),
                    why: "is an affine bank whose scales are half of its dequantization, \
                          and this plan names no zero points for the other half; a \
                          contract states them with `TensorContract::offsetting`",
                });
            }
            (_, None) => None,
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
                group: attachment.group_size,
                bits,
            },
        );
    }
    Ok(out)
}

/// A plane order this shell has no reader for is refused by name.
/// `Dtype::U4g64tiled` is `U4g64`'s codes in fragment order, written for
/// CUDA; this shell's quant kernels index row-major. The net under the
/// declaration; [`serves_this_deployment`] is the net under the file.
///
/// # Errors
///
/// [`Fault::Param`] naming the first param declared in an order this shell
/// cannot read.
pub(crate) fn readable_plane_orders(trace: &Trace) -> Result<()> {
    match trace
        .params
        .iter()
        .find(|param| param.dtype == Dtype::U4g64tiled)
    {
        None => Ok(()),
        Some(param) => Err(Fault::Param {
            name: param.name.clone(),
            why: "is declared U4g64tiled — MLX affine codes in m16n8k16 fragment order, which \
                  this shell has no reader for: its qmm and qmv arms index an affine bank \
                  row-major and would answer nonsense off a relaid plane. The order is \
                  `kernels_cuda::linear::tiled`'s, and a model text reaches it only by asking \
                  for it: `model_dsl::place` resolves a placed dtype against the platform the \
                  declaration is read for, and this platform's answer is the canonical \
                  row-major sibling. So either this plane came out of an artifact converted \
                  FOR the cuda shell — convert it again on this box, or serve it there — or a \
                  text stated the order outright, in which case it is the text that has to ask",
        }),
    }
}

/// Is this artifact for this deployment? — the `pie.serving/1` stamp gate,
/// asked before a byte of the machine is spent. A raw snapshot or
/// ordinary `.zt` has no stamp and proceeds; a stamped artifact is
/// checked field by field (`Stamp::check`), and a request stating no
/// precision is refused before the file is read.
///
/// # Errors
///
/// [`Fault::Recipe`] for a stamp that disagrees, for a serving version this
/// build does not read, and for a load that states no precision.
pub(crate) fn serves_this_deployment(path: &Path, backend: &str, sku: &str) -> Result<()> {
    // A snapshot directory holds no file-level attributes and can carry no stamp.
    if path.is_dir() {
        return Ok(());
    }
    let artifact = match serve::stamp_of(path) {
        // No `pie.serving/1` key: an ordinary checkpoint, the only outcome that proceeds.
        Ok(None) => return Ok(()),
        Ok(Some(stamp)) => stamp,
        // Claims a serving profile but its stamp does not read back — refused
        // rather than served as the ordinary checkpoint it also is.
        Err(why) => return Err(Fault::Recipe(format!("checkpoint: {why}"))),
    };
    let deployment = Stamp::of(backend, sku);
    artifact.check(&deployment).map_err(|mismatch| {
        Fault::Recipe(mismatch.refuse(&path.display().to_string()))
    })
}

/// Where one param's plane sits in the store.
#[derive(Debug, Clone, Copy)]
struct Place {
    offset: u64,
    /// What the store reserves for it — the plane's own bytes when held
    /// whole, `slots x stride` for a streamed band.
    bytes: u64,
    /// Those bytes, rounded up to the next view alignment.
    reserved: u64,
    /// What the plane declares. Equal to [`bytes`](Place::bytes) unless
    /// streamed, in which case it is the whole bank.
    full: u64,
    /// Whether a residency plan holds this param CPU-side.
    streamed: bool,
    /// Which class: `false` routed-expert tier, `true` gathered row slab.
    gathered: bool,
    rows: u32,
    width: u32,
    dtype: Dtype,
}

/// What every param's plane declares, in bytes — stated once so
/// [`places`] and [`experts::Plan`] cannot disagree. Two packed dtypes
/// use units `elem_bytes` answers for neither.
///
/// # Errors
///
/// [`Fault::Param`] for a param declared in a storage element that has no
/// byte size.
///
/// [`experts::Plan`]: crate::experts::Plan
pub(crate) fn plane_bytes(trace: &Trace) -> Result<Vec<u64>> {
    trace
        .params
        .iter()
        .map(|param| {
            let (rows, width) = rectangle(&param.shape);
            Ok(match param.dtype {
                Dtype::Mxfp4 => rows.saturating_mul(width),
                // Sizing is a fact about the rectangle, not who can read the
                // order (`readable_plane_orders` refuses U4g64tiled elsewhere).
                Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => {
                    rows.saturating_mul(width).div_ceil(2)
                }
                // Two bits a code, four codes a byte; none of the three
                // groups (32/64/128) admits a width not a multiple of four.
                Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => {
                    rows.saturating_mul(width).div_ceil(4)
                }
                // A whole byte a code; no row can round up into the next.
                Dtype::U8g64 => rows.saturating_mul(width),
                other => {
                    let element =
                        model_compiler::arena::elem_bytes(other).ok_or_else(|| Fault::Param {
                            name: param.name.clone(),
                            why: "is declared in a packed storage element that has no element \
                                  size",
                        })?;
                    rows.saturating_mul(width).saturating_mul(element)
                }
            })
        })
        .collect()
}

/// Where one param's bytes are, once the warm arm has resolved them.
#[derive(Debug, Clone, Copy)]
enum Seat {
    /// A view onto the mapped artifact, at its serving manifest's offset.
    Artifact(u64),
    /// A slot of the small writable reservation: an adapter bank, or a
    /// residue plane the landing below computes.
    Store(u64),
}

/// How much of a plane the offset probe compares — head and tail catch a
/// shift as cheaply as the whole payload would.
const PROBE: usize = 32;

/// The warm arm: serve the artifact off its own mapped pages, bound once
/// at the artifact's own layout, never `places`'s device-store layout.
/// Adapter banks and computed residue planes get a small writable
/// reservation via [`banks`]. Every refusal is `Option<String>`, not a
/// [`Fault`]: the cold path serves everything this declines. A view is
/// minted at `blob + plane`, so both offsets are checked against [`ALIGN`].
#[allow(clippy::too_many_arguments)]
fn warm(
    device: &Context,
    handles: &Handles,
    trace: &Trace,
    contract: &ModelContract,
    metadata: &Metadata,
    snapshot: &Path,
    target: StorageTarget,
    path: &Path,
    plan: &Plan,
    places: &[Place],
    pairings: &BTreeMap<&str, Pairing>,
) -> std::result::Result<Weights, Option<String>> {
    let gather = plan.gathered();
    // Nothing to map: a directory is not one container.
    if path.is_dir() {
        return Err(None);
    }
    // The fits-in-memory rule is about what the device reads: a streamed
    // band is never bound off the mapping, so this arm's wired set is
    // still the dense floor plus `slots` seats, same as cold.
    let artifact = Artifact::open(path).map_err(|_| None)?;
    let spans = artifact.spans();
    // Recovered off the offsets themselves; the profile stores none.
    let align = serving::alignment(&spans);
    if align == 0 || !align.is_multiple_of(ALIGN) {
        return Err(Some(format!(
            "its serving offsets are aligned to {align} and a matrix operand on this \
             device wants {ALIGN}"
        )));
    }
    // A plane's `(file offset, length)` by the name a `Trace::param` calls it.
    let locate = |name: &str| -> Option<(u64, u64)> {
        let located = artifact.locate(name).ok()?;
        Some((located.at + located.plane.offset, located.plane.len))
    };

    // Planes kept under the checkpoint's own name. A miss falls to the
    // contract: `Expr::Src(name)` alone means the bytes ARE the plane;
    // anything else must be computed and falls back whole.
    let verbatim: BTreeMap<&str, &str> = contract
        .tensors
        .iter()
        .filter_map(|tensor| match &tensor.expr {
            checkpoint::contract::Expr::Src(from) => {
                Some((tensor.name.as_str(), from.as_str()))
            }
            _ => None,
        })
        .collect();

    // The resolution pass: every param seated, or none.
    let mut seats = Vec::with_capacity(places.len());
    // Planes that get a view onto the mapping, as `mapping::cut` wants it:
    // `(name, offset, length)`, covering exactly these and nothing else.
    let mut covers: Vec<(&str, u64, u64)> = Vec::with_capacity(places.len());
    let mut objects: Vec<&str> = Vec::with_capacity(places.len());
    let mut landed: Vec<TensorContract> = Vec::new();
    let mut store_bytes = 0u64;
    let mut residue_bytes = 0u64;
    // Where every plane this arm reads out of the file lies, param-indexed.
    let mut from_file: Vec<Option<u64>> = vec![None; places.len()];
    // `param index -> offset of expert 0`, streamed bands alone, for `Source::artifact`.
    let mut bands: BTreeMap<usize, u64> = BTreeMap::new();
    // The same, for the gathered class, kept apart (two readers, two lifetimes).
    let mut rows_of: BTreeMap<usize, u64> = BTreeMap::new();
    let mut stored = 0u64;
    for (index, param) in trace.params.iter().enumerate() {
        let place = &places[index];
        // A registered plane is one the artifact does not have; reserved
        // at capacity and zero until something registers into it.
        if param.source == ParamSource::Registered {
            seats.push(Seat::Store(store_bytes));
            objects.push(param.name.as_str());
            store_bytes += place.reserved;
            continue;
        }
        // The plane's own name first, and the name it is stored under second
        // — see `verbatim`.
        let object = locate(&param.name)
            .map(|found| (param.name.as_str(), found))
            .or_else(|| {
                verbatim
                    .get(param.name.as_str())
                    .and_then(|from| locate(from).map(|found| (*from, found)))
            });
        let Some(object) = object else {
            // The residue: computed and landed below, capped by the
            // ceiling. A streamed band is never a residue.
            if place.streamed {
                return Err(Some(format!(
                    "its {} `{}` is a plane this load holds CPU-side and the artifact \
                     carries in no form — its seat is a slab and cannot be landed whole \
                     into that seat",
                    if place.gathered { "gathered table" } else { "routed band" },
                    param.name,
                )));
            }
            let Some(residue_of) = residue_of(contract, &param.name) else {
                return Err(Some(format!(
                    "it carries no plane `{}` and this contract declares none — the \
                     artifact and the trace were not written from each other",
                    param.name,
                )));
            };
            landed.push(residue_of);
            seats.push(Seat::Store(store_bytes));
            objects.push(param.name.as_str());
            store_bytes += place.reserved;
            residue_bytes += place.full;
            continue;
        };
        let (object, (offset, length)) = object;
        // `full` is the declared width on both roads; `bytes` (the store's
        // reservation) differs from it only for a streamed band, so the
        // second half of this check is for resident planes alone.
        if length != place.full || (!place.streamed && place.bytes != place.full) {
            return Err(Some(format!(
                "its plane `{}` (stored as `{object}`) is {length} bytes and this plan \
                 declares {}",
                param.name, place.full,
            )));
        }
        if !offset.is_multiple_of(ALIGN) {
            return Err(Some(format!(
                "its plane `{}` (stored as `{object}`) lies at {offset}, and a matrix \
                 operand on this device wants {ALIGN}-byte alignment",
                param.name,
            )));
        }
        from_file[index] = Some(offset);
        stored += place.full;
        objects.push(object);
        if place.streamed {
            // A streamed band gets no view (read-only mapping); its
            // `Tensor` is `slots` seats in the writable reservation.
            if place.gathered {
                rows_of.insert(index, offset);
            } else {
                bands.insert(index, offset);
            }
            seats.push(Seat::Store(store_bytes));
            store_bytes += place.reserved;
            continue;
        }
        // A plane a window has to cover: windows are cut over bound planes
        // only, never the whole file, since a window wires whole and a
        // streamed band (read by the CPU, never bound) must stay unwired.
        covers.push((param.name.as_str(), offset, place.full));
        seats.push(Seat::Artifact(offset));
    }

    // The residue has a ceiling (a sixteenth of what the file stores) so
    // "warm" keeps meaning warm rather than a slower cold load in disguise.
    if residue_bytes.saturating_mul(16) > stored {
        return Err(Some(format!(
            "{} of its {} plane(s) are ones this load computes rather than reads — {} \
             bytes against {stored} stored, which is not a warm load with a residue but \
             a cold load with a mapping",
            landed.len(),
            places.len(),
            residue_bytes,
        )));
    }
    // Read before `landed` is moved into the residue contract below. Zero
    // is the interesting answer: the artifact carried everything.
    let residue_planes = landed.len();

    // The mapping, and the probe that says the offsets are the file's.
    let map = Mapping::of(path).map_err(|why| Some(why.to_string()))?;
    for (index, param) in trace.params.iter().enumerate() {
        let Some(offset) = from_file[index] else {
            continue;
        };
        let length = places[index].full;
        let from = usize::try_from(offset).unwrap_or(usize::MAX);
        let upto = usize::try_from(offset.saturating_add(length)).unwrap_or(usize::MAX);
        let mine = map.get(from..upto).ok_or_else(|| {
            Some(format!(
                "its plane `{}` lies at {offset}..{upto} and the file holds {} bytes",
                param.name,
                map.len(),
            ))
        })?;
        let published = artifact
            .plane(objects[index])
            .map_err(|why| {
                Some(format!(
                    "its plane `{}` has no zero-copy view ({why})",
                    param.name
                ))
            })?;
        let ends = |bytes: &[u8]| {
            let head = bytes.get(..PROBE.min(bytes.len())).unwrap_or_default().to_vec();
            let tail = bytes
                .get(bytes.len().saturating_sub(PROBE)..)
                .unwrap_or_default()
                .to_vec();
            (head, tail)
        };
        if published.len() != mine.len() || ends(published) != ends(mine) {
            return Err(Some(format!(
                "its manifest puts `{}` at {offset} and the bytes there are not the ones \
                 the container publishes for it — two readings of one offset that do not \
                 agree, so nothing here binds either",
                param.name,
            )));
        }
    }

    // ── the two reservations, and the residue landed into the small one.
    let spans: Vec<(u64, u64)> = places
        .iter()
        .enumerate()
        .filter_map(|(index, place)| match seats[index] {
            Seat::Store(offset) => Some((offset, place.reserved)),
            Seat::Artifact(_) => None,
        })
        .collect();
    debug_assert_eq!(
        spans.last().map_or(0, |&(offset, reserved)| offset + reserved),
        store_bytes,
        "the store's spans pack to the bytes the arm counted"
    );
    let mut store = Store::zeroed(device, &spans, device.max_buffer())
        .map_err(|why| Some(why.to_string()))?;
    if !landed.is_empty() {
        let into: BTreeMap<&str, (u64, u64)> = trace
            .params
            .iter()
            .enumerate()
            .filter_map(|(index, param)| match seats[index] {
                Seat::Store(offset) if param.source != ParamSource::Registered => {
                    Some((param.name.as_str(), (offset, places[index].full)))
                }
                _ => None,
            })
            .collect();
        // A contract of the residue alone, compiled and run on its own.
        let only = ModelContract {
            alignment: contract.alignment,
            tensors: landed,
            // Emptied: `residue_of` admits only a leaf naming no other
            // entry, so a residue plane is never a group member.
            groups: Vec::new(),
        };
        let plan = compile(metadata, &only, target).map_err(|why| {
            Some(format!(
                "the {} plane(s) it does not store do not compile ({why})",
                only.tensors.len(),
            ))
        })?;
        let mut scratch = vec![0u8; usize::try_from(plan.memory.arena_bytes()).unwrap_or(0)];
        let mut backing: &mut [u8] = &mut scratch;
        let mut sink = Residue {
            store: &mut store,
            into: &into,
            landed: 0,
        };
        Execution::new(&plan, snapshot)
            .arena(&mut backing)
            .sink(&mut sink)
            .run()
            .map_err(|why| Some(format!("the plane(s) it does not store do not land ({why})")))?;
        if sink.landed != only.tensors.len() {
            return Err(Some(format!(
                "the residue landing published {} of {} plane(s)",
                sink.landed,
                only.tensors.len(),
            )));
        }
    }
    // The windows (`mapping::cut`): an artifact past `maxBufferLength` is
    // bound in several reservations, cut around `covers` (bound planes
    // only) so no plane straddles a boundary and no window wires a band.
    let cuts = mapping::cut(&map, mapping::ceiling(device.max_buffer()), &covers)
        .map_err(|why| Some(why.to_string()))?;
    let mut files = Vec::with_capacity(cuts.len());
    for cut in &cuts {
        files.push(
            Buffer::window(device, std::sync::Arc::clone(&map), *cut)
                .map_err(|why| Some(why.to_string()))?,
        );
    }
    let row = |index: usize| -> std::result::Result<Tensor, Option<String>> {
        let place = &places[index];
        let handle = match seats[index] {
            Seat::Artifact(offset) => {
                // The last cut starting at or before offset, whole length included.
                let at = cuts
                    .iter()
                    .rposition(|cut| cut.holds(offset, place.bytes))
                    .ok_or_else(|| {
                        Some(format!(
                            "`{}` lies at {offset} for {} bytes, which no one of this \
                             artifact's {} mapped window(s) holds whole",
                            trace.params[index].name,
                            place.bytes,
                            cuts.len(),
                        ))
                    })?;
                let view = cuts[at].view(offset).ok_or_else(|| {
                    Some(format!("`{}` does not seat in its window", trace.params[index].name))
                })?;
                handles.bind(&files[at], view, place.bytes)
            }
            Seat::Store(offset) => store.bind(handles, offset, place.bytes),
        }
        .map_err(|why| Some(format!("`{}` does not bind ({why})", trace.params[index].name)))?;
        Ok(Tensor::new(handle, place.rows, place.width, place.dtype))
    };
    let mut table = Vec::with_capacity(places.len());
    for (index, param) in trace.params.iter().enumerate() {
        table.push(Some(match pairings.get(param.name.as_str()) {
            Some(pairing) => WeightRow::Planes(kernels_metal::Bank {
                codes: row(index)?,
                scales: row(pairing.scales)?,
                biases: pairing.biases.map(row).transpose()?,
                group: pairing.group,
                bits: pairing.bits,
            }),
            None => WeightRow::Dense(row(index)?),
        }));
    }
    let seated: Vec<u64> = seats
        .iter()
        .map(|seat| match seat {
            Seat::Store(offset) | Seat::Artifact(offset) => *offset,
        })
        .collect();
    // The tier, over the artifact itself: source is the same mapping the
    // resident planes bind through (`Arc::clone(&map)`), no staging file,
    // offsets straight out of the manifest (`bands`). `seated` is this
    // arm's own destination packing, not `places`'s.
    let tier = plan
        .streams()
        .then(|| {
            Tier::open(
                plan,
                &store,
                Source::artifact(std::sync::Arc::clone(&map), bands),
                &seated,
            )
            .map(RefCell::new)
        })
        .transpose()
        .map_err(|why| Some(format!("its routed tier does not open ({why})")))?;
    // The row slab, over that same mapping — read by `memcpy` alone, so
    // its pages stay pageable rather than wired.
    let rows = gather
        .gathers()
        .then(|| {
            gather::Slab::open(
                gather,
                &store,
                Source::artifact(std::sync::Arc::clone(&map), rows_of),
                &seated,
            )
            .map(RefCell::new)
        })
        .transpose()
        .map_err(|why| Some(format!("its gathered row slab does not open ({why})")))?;
    Ok(Weights {
        store,
        mapped: files,
        table: WeightTable(table),
        tier,
        rows,
        banks: banks(trace, places, |at| seated[at]),
        residue: (residue_planes, residue_bytes),
        decoded: Vec::new(),
    })
}

/// Can the warm arm land this plane on its own? The entry must be
/// self-contained: it names no other entry ([`Expr::Out`]) and pairs with
/// no scales/zero-points. `None` if the contract does not declare it.
///
/// [`Expr::Out`]: checkpoint::contract::Expr::Out
fn residue_of(contract: &ModelContract, name: &str) -> Option<TensorContract> {
    let entry = contract.tensors.iter().find(|it| it.name == name)?;
    if entry.scales.is_some() || entry.zero_points.is_some() {
        return None;
    }
    let mut names_another = false;
    entry.expr.visit(&mut |node| {
        if matches!(node, checkpoint::contract::Expr::Out(_)) {
            names_another = true;
        }
    });
    (!names_another).then(|| entry.clone())
}

/// The sink the residue landing publishes into: the small writable
/// reservation. [`Landing`]'s sibling, not a mode of it.
struct Residue<'a> {
    store: &'a mut Store,
    /// Published name -> (offset in the store, the bytes the plan declares).
    into: &'a BTreeMap<&'a str, (u64, u64)>,
    /// How many planes actually arrived, checked against how many were asked for.
    landed: usize,
}

impl TensorSink for Residue<'_> {
    fn publish(&mut self, name: &str, bytes: &[u8]) -> std::result::Result<(), LoadError> {
        let Some((offset, full)) = self.into.get(name).copied() else {
            // Not a refusal: a contract entry may compute an internal
            // tensor the trace does not name.
            return Ok(());
        };
        if bytes.len() as u64 != full {
            return Err(LoadError::Contract(format!(
                "`{name}` lands {} bytes and the plan declares {full} — a plane read at \
                 the wrong width is a model that computes",
                bytes.len(),
            )));
        }
        self.store
            .write(offset, bytes)
            .map_err(|fault| LoadError::Internal(fault.to_string()))?;
        self.landed += 1;
        Ok(())
    }
}

/// The store's layout, decided before a byte is read. `plan` and `gather`
/// shrink a bank below its declaration (reserving `slots` seats instead
/// of the whole leading axis) while `full` keeps the declared size.
fn places(trace: &Trace, plan: &Plan, gather: &gather::Plan) -> Result<Vec<Place>> {
    let bytes = plane_bytes(trace)?;
    let mut out = Vec::with_capacity(trace.params.len());
    let mut at = 0u64;
    for (index, param) in trace.params.iter().enumerate() {
        let (rows, width) = rectangle(&param.shape);
        let full = bytes[index];
        let seated = plan.resident(index);
        let gathered = gather.resident(index);
        let (reserve, rows) = match seated.or(gathered) {
            Some(slots) if rows > 0 => (full / rows * u64::from(slots), u64::from(slots)),
            _ => (full, rows),
        };
        out.push(Place {
            offset: at,
            bytes: reserve,
            reserved: reserve.next_multiple_of(ALIGN),
            full,
            streamed: seated.is_some() || gathered.is_some(),
            gathered: gathered.is_some(),
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype: param.dtype,
        });
        at += reserve.next_multiple_of(ALIGN);
    }
    Ok(out)
}

/// A declared shape, read as `rows x width`. Honest rather than
/// load-bearing: `linear.matmul` takes its dimensions from elsewhere.
fn rectangle(shape: &[u64]) -> (u64, u64) {
    match shape.split_first() {
        Some((rows, rest)) => (*rows, rest.iter().product()),
        None => (1, 1),
    }
}

/// The sink that puts each finalized tensor where the layout said it goes.
/// A branch on the plan, not a second sink.
struct Landing<'a> {
    store: &'a mut Store,
    /// Every expert of every streamed band, at the offsets
    /// [`Plan::host_at`](crate::experts::Plan::host_at) states.
    host: &'a mut [u8],
    /// Every row of every gathered table, at [`gather::Plan::host_at`]'s
    /// offsets — a second mapping, not a second sink.
    table: &'a mut [u8],
    plan: &'a Plan,
    gather: &'a gather::Plan,
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
        // The DECLARED size, not the reserved one: a streamed band reserves
        // `slots` seats and the checkpoint still publishes every expert.
        if bytes.len() as u64 != place.full {
            return Err(LoadError::Contract(format!(
                "`{name}` lands {} bytes and the plan declares {} — a plane read \
                 at the wrong width is a model that computes",
                bytes.len(),
                place.full
            )));
        }
        if place.streamed {
            let (what, at_host, into) = if place.gathered {
                (
                    "gathered table",
                    self.gather.host_at(at).unwrap_or(0),
                    &mut *self.table,
                )
            } else {
                ("streamed band", self.plan.host_at(at).unwrap_or(0), &mut *self.host)
            };
            let from = usize::try_from(at_host).unwrap_or(usize::MAX);
            let into = into
                .get_mut(from..from + bytes.len())
                .ok_or_else(|| {
                    LoadError::Internal(format!(
                        "`{name}` is a {what} whose {} bytes leave its host source at \
                         offset {from}",
                        bytes.len()
                    ))
                })?;
            into.copy_from_slice(bytes);
        } else {
            self.store
                .write(place.offset, bytes)
                .map_err(|fault| LoadError::Internal(fault.to_string()))?;
        }
        self.landed[at] = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use checkpoint::contract::Expr;
    use checkpoint::types::{DType, Encoding};
    use model_ir::Platform;

    use super::*;

    /// The residue rule, pinned: the line between a leaf this arm can run
    /// on its own and the top of a chain whose other links are not here.
    #[test]
    fn the_residue_is_a_leaf_and_a_chain_is_not() {
        let entry = |name: &str, expr: Expr| TensorContract {
            name: name.to_string(),
            expr,
            shape: None,
            encoding: Encoding::Raw(DType::F32),
            scales: None,
            zero_points: None,
            visibility: Default::default(),
        };
        let contract = ModelContract {
            alignment: 256,
            tensors: vec![
                // A cast of one stored tensor.
                entry(
                    "leaf",
                    Expr::Cast {
                        src: Box::new(Expr::Src("stored".into())),
                        to: Encoding::Raw(DType::F32),
                    },
                ),
                // The top of a chain.
                entry(
                    "chained",
                    Expr::Cast {
                        src: Box::new(Expr::Out("leaf".into())),
                        to: Encoding::Raw(DType::F32),
                    },
                ),
            ],
            groups: Vec::new(),
        };

        assert!(
            residue_of(&contract, "leaf").is_some(),
            "a cast of one stored tensor is a plane this arm can land by itself"
        );
        assert!(
            residue_of(&contract, "chained").is_none(),
            "an entry that names another entry is a chain, and the cold path is what \
             runs chains"
        );
        assert!(
            residue_of(&contract, "nowhere").is_none(),
            "a plane the contract does not declare is not a residue at all"
        );
    }

    #[test]
    fn the_store_is_laid_out_aligned_disjoint_and_in_plan_order() {
        let trace =
            models::sku("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships the SKU").trace;
        let trace = trace(Platform::Metal);
        let places = places(&trace, &Plan::default(), &gather::Plan::default())
            .expect("every param of a bf16 SKU has an element size");

        assert_eq!(places.len(), trace.params.len());
        let mut end = 0u64;
        for (place, param) in places.iter().zip(&trace.params) {
            assert!(
                place.offset >= end,
                "`{}` overlaps its predecessor",
                param.name
            );
            assert_eq!(place.offset % ALIGN, 0, "`{}` is misaligned", param.name);
            assert!(place.bytes > 0, "`{}` reserves nothing", param.name);
            end = place.offset + place.reserved;
        }

        assert_eq!(places[0].offset, 0);
        assert_eq!(places[0].rows, 248_320);
        assert_eq!(places[0].width, 1024);
        assert_eq!(places[0].bytes, 248_320 * 1024 * 2);
    }
}
