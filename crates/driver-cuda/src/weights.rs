//! The checkpoint, resident: one device allocation, one row per
//! `Plan::params`.
//!
//! # The contract arrives; the family does not
//!
//! This module takes a [`ModelContract`] and never asks which model it
//! describes. That is decision #18 read from the residency side: the engine
//! links `model`, traces the `Plan` and states the load contract; the shell
//! compiles the contract against a checkpoint and lands the bytes. A shell
//! that reached for `model::qwen_3` would be a shell that has to grow an arm
//! per family, which is exactly the shape the string-plan era had.
//!
//! # Names are the plan's, both sides
//!
//! The load contract publishes one `Visibility::Public` entry per plan param,
//! **under the param's own name** — `embed`, `layer.7.q_proj` — and the
//! checkpoint's spellings live inside the expressions. So the bijection this
//! module needs is `plan.params[i].name == published name`, which is the
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
//! bf16→f32 widenings on norm scales; the device path
//! (`model_loader::executor::cuda`) is a load-time optimisation, and it does
//! not build against the current `kernels-cuda` (its imports name a `quant`
//! module and `In`/`Out` marks that the menlo rewrite retired). Landing the
//! bytes correctly first, quickly second, is the right order.

use std::collections::BTreeMap;
use std::path::Path;

use kernels_cuda::Tensor;
use model_ir::{Dtype, Plan};
use model_loader::checkpoint;
use model_loader::contract::ModelContract;
use model_loader::error::Error as LoadError;
use model_loader::executor::{Execution, sink::TensorSink};
use model_loader::plan::{StorageTarget, compile};
use model_loader::types::BackendKind;

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::run::{WeightRow, WeightTable};

/// What a matrix operand wants under cuBLAS, and what `cudaMalloc` itself
/// guarantees — so a view into the store is as aligned as its own allocation
/// would have been. The same number the loader's `StorageTarget` states, for
/// the same reason.
const ALIGN: u64 = 256;

/// Every weight this model needs, on the device.
#[derive(Debug)]
pub struct Weights {
    store: Buffer,
    table: WeightTable,
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
    pub fn resident(plan: &Plan, contract: &ModelContract, path: &Path) -> Result<Weights> {
        let (metadata, snapshot) = if path.is_dir() {
            (checkpoint::read::parse_checkpoint_metadata(path)?, path)
        } else {
            (
                checkpoint::zt::parse_checkpoint(path)?,
                path.parent().unwrap_or(Path::new(".")),
            )
        };

        // tp=1: the plan's `Shard::Cut` segments still describe the whole
        // tensor, and a rank of one takes all of them. `tp_rank`/`tp_size`
        // are the loader's whole notion of a rank, so a shell that grows
        // tensor parallelism states it here and nowhere else.
        let target = StorageTarget::for_backend(BackendKind::Cuda, 0, 1);
        let landing = compile(&metadata, contract, target)?;

        let places = places(plan)?;
        let total = places.last().map_or(0, |p| p.offset + p.reserved);
        let mut store = Buffer::zeroed(usize::try_from(total).unwrap_or(usize::MAX))?;

        // The executor's own arena: where a transform's intermediates live
        // while it runs. Host memory, because the transforms run host-side;
        // it is dropped the moment the load is over, and only the finalized
        // tensors the sink took survive.
        let mut scratch = vec![0u8; usize::try_from(landing.memory.arena_bytes()).unwrap_or(0)];
        let mut backing: &mut [u8] = &mut scratch;

        let index: BTreeMap<&str, usize> = plan
            .params
            .iter()
            .enumerate()
            .map(|(at, param)| (param.name.as_str(), at))
            .collect();
        let mut sink = Landing {
            store: &mut store,
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

        let mut table = Vec::with_capacity(places.len());
        for (at, place) in places.iter().enumerate() {
            if !landed[at] {
                return Err(Fault::Param {
                    name: plan.params[at].name.clone(),
                    why: "is a plan param the load contract never published",
                });
            }
            // Every row of this catalog is one dense handle. A split-plane
            // bank (`WeightRow::Planes`) arrives the day a SKU with mxfp4
            // banks does, through the load plan's `attachments` — the
            // pairing is stated there rather than guessed from a `.scales`
            // suffix, which is how a scale tensor gets read as the wrong
            // one's.
            table.push(Some(WeightRow::Dense(Tensor::new(
                store.at(place.offset)?,
                place.rows,
                place.width,
                place.dtype,
            ))));
        }
        Ok(Weights {
            store,
            table: WeightTable(table),
        })
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

/// Where one param's plane sits in the store.
#[derive(Debug, Clone, Copy)]
struct Place {
    offset: u64,
    /// The plane's own bytes.
    bytes: u64,
    /// Those bytes, rounded up to the next handle alignment.
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
fn places(plan: &Plan) -> Result<Vec<Place>> {
    let mut out = Vec::with_capacity(plan.params.len());
    let mut at = 0u64;
    for param in &plan.params {
        let element =
            model_compiler::arena::elem_bytes(param.dtype).ok_or_else(|| Fault::Param {
                name: param.name.clone(),
                why: "is declared in a packed storage element that has no element size",
            })?;
        let (rows, width) = rectangle(&param.shape);
        let bytes = rows.saturating_mul(width).saturating_mul(element);
        out.push(Place {
            offset: at,
            bytes,
            reserved: bytes.next_multiple_of(ALIGN),
            rows: u32::try_from(rows).unwrap_or(u32::MAX),
            width: u32::try_from(width).unwrap_or(u32::MAX),
            dtype: param.dtype,
        });
        at += bytes.next_multiple_of(ALIGN);
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
        self.store
            .write(place.offset, bytes)
            .map_err(|fault| LoadError::Internal(fault.to_string()))?;
        self.landed[at] = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use model_dsl::Plane;

    use super::*;

    #[test]
    fn the_store_is_laid_out_aligned_disjoint_and_in_plan_order() {
        let trace =
            model::trace_of("qwen35-d0.8b-bf16-kv-bf16").expect("the catalog ships the SKU");
        let plan = trace(Plane::Cuda);
        let places = places(&plan).expect("every param of a bf16 SKU has an element size");

        assert_eq!(places.len(), plan.params.len());
        let mut end = 0u64;
        for (place, param) in places.iter().zip(&plan.params) {
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
