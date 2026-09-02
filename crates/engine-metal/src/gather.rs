//! The gathered row slab: a third residency class for a plane whose demand
//! per fire is static and sparse (a token's PLE n-gram hash rows) rather
//! than a router's dynamic choice (`crate::experts`).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use model_compiler::CompiledModel;
use model_exec::fire::MaskSpan;
use model_ir::ops::{Attention, Layout};
use model_ir::{Def, Operation, Trace, ValueId};

use crate::weight_store::Store;
use crate::device::alloc::Buffer;
use crate::error::{Fault, Result};
use crate::experts::{Attachments, Source};
use crate::device::Handles;
use kernels_metal::Tensor;

/// The plane a gathered load holds CPU-side, and everything needed to build
/// a slab over it. One and not many, since only one exists (see
/// [`Plan::of`]'s refusal for a second).
#[derive(Debug, Clone, Default)]
pub struct Table {
    /// The code plane's own name, for a refusal that names it.
    pub name: String,
    /// `Trace::params` indices of the three planes, code plane first. A
    /// symmetric bank has two.
    pub params: Vec<usize>,
    /// The table's DECLARED row count — `padded_vocab`, the number the trace
    /// hands `embed_concat` as its vocabulary guard.
    pub rows: u64,
    /// How many rows the slab seats.
    pub seats: u32,
    /// One row of each plane, in `params` order.
    pub strides: Vec<u64>,
    /// `param -> byte offset of row 0` in the CPU-side source, packed in
    /// `params` order from zero — the cold arm's layout, and the one the
    /// landing sink and the slab share.
    host_of: BTreeMap<usize, u64>,
    host_bytes: u64,
    device_bytes: u64,
}

impl Table {
    /// What the store would have reserved for this table held resident
    /// (every plane whole, at the store's own alignment). The gate's
    /// arithmetic: a gathered load's weight reservation is a resident
    /// one's, minus this, plus the slab.
    #[must_use]
    pub fn stored(&self) -> u64 {
        self.strides
            .iter()
            .map(|stride| (self.rows * stride).next_multiple_of(crate::weights::ALIGN))
            .sum()
    }

    /// What the slab reserves instead.
    #[must_use]
    pub fn slab(&self) -> u64 {
        self.device_bytes
    }
}

/// The residency plan for the gathered class: empty for every load that
/// holds its tables whole, which is every load but a capped Flash-Next.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    table: Option<Table>,
}

impl Plan {
    /// Plans the gathered class for `trace` under `budget`. `None`
    /// (uncapped), or any budget that already holds the whole weight
    /// table, plans the empty (resident) load. The plane is found
    /// structurally: the table param of a `Layout::EmbedConcat` node, so a
    /// trace with no `EmbedConcat` plans the empty gather.
    ///
    /// # Errors
    ///
    /// [`Fault::Param`] for an `EmbedConcat` whose table does not resolve
    /// to a weight, a table plane with no declared rows, or a second
    /// gathered table. [`Fault::Residency`] when even the row slab does
    /// not fit the budget.
    pub fn of(
        trace: &Trace,
        planes: &Attachments,
        budget: Option<u64>,
        max_tokens: u32,
    ) -> Result<Plan> {
        let Some(budget) = budget else {
            return Ok(Plan::default());
        };
        let bytes = crate::weights::plane_bytes(trace)?;
        let full: u64 = bytes
            .iter()
            .map(|plane| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        if budget >= full {
            return Ok(Plan::default());
        }

        // hashed heads, off the hasher's own node: primes.len() is both the
        // head count and the id-vector width, so seats = row ceiling x heads.
        let mut heads: Option<usize> = None;
        for node in &trace.nodes {
            let Operation::Attention(op) = &node.op else {
                continue;
            };
            let primes = match op {
                Attention::PleNgramIds { primes, .. }
                | Attention::PleNgramIdsChunked { primes, .. } => primes,
                _ => continue,
            };
            heads = Some(heads.map_or(primes.len(), |held: usize| held.max(primes.len())));
        }

        let mut found: Option<Table> = None;
        for node in &trace.nodes {
            let Operation::Layout(Layout::EmbedConcat { table, .. }) = &node.op else {
                continue;
            };
            let codes = weight_of(trace, *table)?;
            if let Some(held) = &found {
                if held.params.first() == Some(&codes) {
                    continue;
                }
                return Err(Fault::Param {
                    name: trace.params[codes].name.clone(),
                    why: "is a second gathered table in one plan; this class holds one, \
                          because one is what exists — a second wants the group vocabulary \
                          the routed tier has and this one deliberately does not",
                });
            }
            let Some(heads) = heads else {
                return Err(Fault::Param {
                    name: trace.params[codes].name.clone(),
                    why: "is read by a concatenating gather in a plan that carries no PLE \
                          hasher; the gathered class is the hasher's demand shape and there \
                          is no static demand to serve without one",
                });
            };
            let mut params = vec![codes];
            params.extend(planes.get(&codes).into_iter().flatten().copied());
            let rows = trace.params[codes].shape.first().copied().unwrap_or(0);
            if rows == 0 {
                return Err(Fault::Param {
                    name: trace.params[codes].name.clone(),
                    why: "declares no rows, and a row slab over a table with no rows has no \
                          stride to seat",
                });
            }
            let mut strides = Vec::with_capacity(params.len());
            for &at in &params {
                let plane_rows = trace.params[at].shape.first().copied().unwrap_or(0);
                if plane_rows != rows {
                    return Err(Fault::Param {
                        name: trace.params[at].name.clone(),
                        why: "is a companion plane of a gathered table whose leading axis is \
                              not the table's; a seat is one row of every plane at the same \
                              index, and two row counts make that untrue",
                    });
                }
                strides.push(bytes[at] / rows);
            }
            // seat count: fire's row ceiling times heads per row. Dedup
            // only shrinks real demand, so this is a bound, not an estimate.
            let seats = u64::from(max_tokens).saturating_mul(heads as u64).min(rows);
            let seats = u32::try_from(seats).unwrap_or(u32::MAX).max(1);
            let mut host_of = BTreeMap::new();
            let mut host_bytes = 0u64;
            let mut device_bytes = 0u64;
            for (at, &param) in params.iter().enumerate() {
                host_of.insert(param, host_bytes);
                host_bytes += bytes[param];
                device_bytes += (u64::from(seats) * strides[at])
                    .next_multiple_of(crate::weights::ALIGN);
            }
            found = Some(Table {
                name: trace.params[codes].name.clone(),
                params,
                rows,
                seats,
                strides,
                host_of,
                host_bytes,
                device_bytes,
            });
        }

        let Some(table) = found else {
            return Ok(Plan::default());
        };
        // dense floor restated for this class alone: what's left after the
        // table leaves the store, plus the slab.
        let gathered: BTreeSet<usize> = table.params.iter().copied().collect();
        let rest: u64 = bytes
            .iter()
            .enumerate()
            .filter(|(at, _)| !gathered.contains(at))
            .map(|(_, plane)| plane.next_multiple_of(crate::weights::ALIGN))
            .sum();
        if budget < table.device_bytes {
            return Err(Fault::Residency(format!(
                "`device_weight_budget` is {budget} bytes and `{}` alone is {} of them, so \
                 it is held CPU-side and read through a row slab of {} seats — which is \
                 {} bytes, and the budget does not hold even that. The slab is sized by the \
                 fire's row ceiling and not by the table's {} rows, so this number does not \
                 shrink: raise the budget past it, or lower `max_tokens`.",
                table.name,
                bytes[table.params[0]],
                table.seats,
                table.device_bytes,
                table.rows,
            )));
        }
        let _ = rest;
        Ok(Plan { table: Some(table) })
    }

    /// Does this load gather any plane?
    #[must_use]
    pub fn gathers(&self) -> bool {
        self.table.is_some()
    }

    /// The gathered table, or `None` for a resident load.
    #[must_use]
    pub fn table(&self) -> Option<&Table> {
        self.table.as_ref()
    }

    /// How many rows the store reserves for `param`, or `None` if this
    /// class does not hold it — the same question `experts::Plan::resident`
    /// answers for a band.
    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        let table = self.table.as_ref()?;
        table.params.contains(&param).then_some(table.seats)
    }

    /// Where `param`'s whole plane lies in the CPU-side source.
    #[must_use]
    pub fn host_at(&self, param: usize) -> Option<u64> {
        self.table.as_ref()?.host_of.get(&param).copied()
    }

    /// Every param this class holds — `experts::Plan::of` takes it as an
    /// EXCLUSION, so the table's bytes leave both the full table and the dense
    /// floor before the expert slab is sized.
    #[must_use]
    pub fn params(&self) -> BTreeSet<usize> {
        self.table
            .as_ref()
            .map(|t| t.params.iter().copied().collect())
            .unwrap_or_default()
    }

    /// What the slab costs on the device.
    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.table.as_ref().map_or(0, |t| t.device_bytes)
    }

    /// What the CPU-side source has to hold.
    #[must_use]
    pub fn source_bytes(&self) -> u64 {
        self.table.as_ref().map_or(0, |t| t.host_bytes)
    }

    /// `param -> byte offset of row 0` in the CPU-side source — what
    /// `experts::Source::from_host` is built over on the cold arm, and the one
    /// arithmetic the landing sink and the slab share.
    #[must_use]
    pub fn host_bands(&self) -> BTreeMap<usize, u64> {
        self.table
            .as_ref()
            .map(|t| t.host_of.clone())
            .unwrap_or_default()
    }

    /// A gathered read's vocabulary guard for `param`'s table: `seats`,
    /// not `padded_vocab`, since the read is against the slab and
    /// [`Slab::segment`] remaps an unanswerable id to `seats` itself.
    #[must_use]
    pub fn vocab(&self, param: usize) -> Option<u32> {
        let table = self.table.as_ref()?;
        (table.params.first() == Some(&param)).then_some(table.seats)
    }
}

/// The param a `Def::Weight` value names.
fn weight_of(trace: &Trace, id: ValueId) -> Result<usize> {
    match trace.values.get(id.0 as usize).map(|decl| &decl.def) {
        Some(Def::Weight(w)) => Ok(*w as usize),
        _ => Err(Fault::Param {
            name: format!("value {}", id.0),
            why: "is read as a concatenating gather's table and is not a weight; a gathered \
                  plane is a `Def::Weight` row and nothing else resolves there",
        }),
    }
}

/// Where the walk is cut for the gathered class, one entry per region of
/// the compiled template: the id vector the hasher in that region writes,
/// or `None`. Two regions each carrying one hasher is normal (prefill and
/// decode arms), writing disjoint rows of one merged vector.
///
/// # Errors
///
/// [`Fault::Residency`] when one region holds two hashers writing
/// different id vectors.
pub fn cuts(trace: &Trace, compiled: &CompiledModel) -> Result<Vec<Option<ValueId>>> {
    let mut out = Vec::with_capacity(compiled.template().len());
    for (at, region) in compiled.template().iter().enumerate() {
        let mut here: Option<ValueId> = None;
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            let Operation::Attention(op) = &node.op else {
                continue;
            };
            let ids = match op {
                Attention::PleNgramIds { ngram_ids, .. }
                | Attention::PleNgramIdsChunked { ngram_ids, .. } => *ngram_ids,
                _ => continue,
            };
            if let Some(first) = here {
                if first != ids {
                    return Err(Fault::Residency(format!(
                        "region {at} holds two n-gram hashers landing different id vectors \
                         (values {} and {}), and a gathered load cuts its command buffer \
                         after EACH one — a single cut behind both would seat the first \
                         arm's rows and then read the second arm's raw table ids as seats. \
                         Raise `device_weight_budget` to hold the table whole, or bake an \
                         artifact whose regions carry one hasher each.",
                        first.0, ids.0
                    )));
                }
                continue;
            }
            here = Some(ids);
        }
        out.push(here);
    }
    Ok(out)
}

/// One plane of the gathered table, seated: where its rows are on both sides,
/// and how wide one row is.
#[derive(Debug)]
struct Band {
    /// Byte offset of seat 0 in the device weight store.
    at: u64,
    /// Byte offset of row 0 in the CPU-side source.
    from: u64,
    stride: u64,
}

/// What one fire seated, for a gate to read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Residency {
    /// The code plane, by name.
    pub name: String,
    /// The table's declared rows.
    pub rows: u64,
    /// How many of them the slab seats.
    pub seats: u32,
    /// How many DISTINCT rows the last fire demanded.
    pub demanded: u32,
}

/// The row slab: the CPU-side table, a wired slab of seats, and the seat
/// map between them. Holds a retain of the weight store (cloning a
/// `Buffer` retains, not copies).
#[derive(Debug)]
pub struct Slab {
    store: Store,
    source: Source,
    bands: Vec<Band>,
    /// The table's declared rows — an id at or past it is unaddressable and
    /// goes to the guard rather than to a seat.
    rows: u64,
    seats: u32,
    name: String,
    /// Table row -> seat, for this fire. A map, not a vector: the id space
    /// is the vocabulary (hundreds of millions of rows) but occupancy is
    /// only thousands.
    seat_of: HashMap<i32, u32>,
    /// Which row is in which seat, ascending — the observable half.
    in_seat: Vec<i32>,
    /// The next free seat. Reset at every fire boundary; see the module
    /// header for why there is no clock.
    next: u32,
    fires: u64,
    copies: u64,
}

impl Slab {
    /// Opens the slab over a landed or mapped source. `offsets` is
    /// `param -> byte offset of seat 0 in the store`. Nothing is seeded:
    /// unlike `Tier::open`'s identity-prefix copy, a hashed row space has
    /// no such prefix, so the slab opens zeroed.
    ///
    /// # Errors
    ///
    /// [`Fault::Param`] for a plan param the store laid down no offset for.
    pub fn open(plan: &Plan, store: &Store, source: Source, offsets: &[u64]) -> Result<Slab> {
        let table = plan.table.as_ref().ok_or_else(|| Fault::Param {
            name: "the gathered table".to_string(),
            why: "is opened over a plan that gathers nothing",
        })?;
        let mut bands = Vec::with_capacity(table.params.len());
        for (at, &param) in table.params.iter().enumerate() {
            let seat0 = offsets.get(param).copied().ok_or_else(|| Fault::Param {
                name: format!("param {param}"),
                why: "is a gathered plane the store laid down no offset for",
            })?;
            let from = source.at(param).ok_or_else(|| Fault::Param {
                name: format!("param {param}"),
                why: "is a gathered plane the CPU-side source states no row-0 offset for",
            })?;
            bands.push(Band {
                at: seat0,
                from,
                stride: table.strides[at],
            });
        }
        let mut source = source;
        source.settle();
        Ok(Slab {
            store: store.clone(),
            source,
            bands,
            rows: table.rows,
            seats: table.seats,
            name: table.name.clone(),
            seat_of: HashMap::new(),
            in_seat: vec![-1; table.seats as usize],
            next: 0,
            fires: 0,
            copies: 0,
        })
    }

    /// The fire boundary, and this class's whole eviction discipline.
    /// Called once per fire, before the walk: everything the previous fire
    /// seated is released at once. Within a fire nothing is ever released.
    pub fn fire(&mut self) {
        self.seat_of.clear();
        for seat in &mut self.in_seat {
            *seat = -1;
        }
        self.next = 0;
        self.fires += 1;
    }

    /// The cut's work: reads the ids the hasher just wrote, seats every
    /// row they name, and rewrites each id to its seat. The caller has
    /// committed and waited, which makes the read/write a legal memcpy.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for an id vector this fire minted no row for;
    /// [`Fault::Residency`] for a fire demanding more distinct rows than
    /// the slab seats (`Plan::of`'s sizing makes this unreachable).
    pub fn segment(
        &mut self,
        arena: &mut Buffer,
        handles: &Handles,
        ids: ValueId,
        rect: Tensor,
        span: MaskSpan,
    ) -> Result<()> {
        if span.rows == 0 {
            return Ok(());
        }
        let width = u64::from(rect.width);
        let base = {
            let row = handles.get(rect.buf).ok_or_else(|| Fault::Unbound {
                what: format!(
                    "handle {}, the n-gram id vector of value {}, which this fire minted no \
                     row for",
                    rect.buf, ids.0
                ),
            })?;
            row.offset()
        };
        let first = base + u64::from(span.row_offset) * width * 4;
        let count = usize::try_from(u64::from(span.rows) * width).unwrap_or(usize::MAX);
        let mut raw = vec![0u8; count * 4];
        arena.read(first, &mut raw)?;
        // seat then rewrite in one pass, so an already-seen row costs no
        // second copy.
        for entry in raw.chunks_exact_mut(4) {
            let id = i32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
            if id < 0 {
                // already unaddressable; left as the hasher wrote it, the
                // gather's own guard zeroes it.
                continue;
            }
            if u64::from(id.unsigned_abs()) >= self.rows {
                // id the table can't answer: goes to the guard's zero arm
                // at the slab's own height (Plan::vocab).
                entry.copy_from_slice(&(self.seats as i32).to_le_bytes());
                continue;
            }
            let seat = self.seat(id)?;
            entry.copy_from_slice(&(seat as i32).to_le_bytes());
        }
        arena.write(first, &raw)?;
        Ok(())
    }

    /// Where row `row` sits, seating it if this fire has not already.
    fn seat(&mut self, row: i32) -> Result<u32> {
        if let Some(&seat) = self.seat_of.get(&row) {
            return Ok(seat);
        }
        if self.next >= self.seats {
            return Err(Fault::Residency(format!(
                "this fire demands more than {} distinct rows of `{}`, which is every seat \
                 the gathered row slab has. The slab is sized `max_tokens x heads` and a \
                 fire may not present more token rows than `max_tokens`, so this is a \
                 composition wider than the budget it was planned against: raise \
                 `device_weight_budget` to hold the table whole, or lower `max_tokens`.",
                self.seats, self.name,
            )));
        }
        let seat = self.next;
        self.next += 1;
        self.copy(seat, row)?;
        self.seat_of.insert(row, seat);
        self.in_seat[seat as usize] = row;
        Ok(seat)
    }

    /// One row of every plane, from the CPU-side source into the slab. An
    /// overwrite in place inside a fixed reservation, which is what bounds
    /// wired residency: nothing here grows, and the GPU only ever touches
    /// the slab's pages.
    fn copy(&mut self, seat: u32, row: i32) -> Result<()> {
        for band in 0..self.bands.len() {
            let (into, from, stride) = {
                let band = &self.bands[band];
                (
                    band.at + u64::from(seat) * band.stride,
                    band.from + u64::from(row.unsigned_abs()) * band.stride,
                    band.stride,
                )
            };
            let from = usize::try_from(from).unwrap_or(usize::MAX);
            let len = usize::try_from(stride).unwrap_or(usize::MAX);
            let source = self.source.get(from, len).ok_or_else(|| Fault::Ceiling {
                what: "bytes of the gathered row source",
                need: (from + len) as u64,
                have: self.source.len(),
            })?;
            self.store.write(into, source)?;
            self.copies += 1;
        }
        Ok(())
    }

    /// What the last fire seated — the gate's observable.
    #[must_use]
    pub fn residency(&self) -> Residency {
        Residency {
            name: self.name.clone(),
            rows: self.rows,
            seats: self.seats,
            demanded: self.next,
        }
    }

    /// `(row copies, fires)` — a register, read by nothing and branched on by
    /// nothing.
    #[must_use]
    pub fn motion(&self) -> (u64, u64) {
        (self.copies, self.fires)
    }

    /// What the CPU-side source is made of, and what is behind it — the same
    /// pair `experts::Tier` publishes, and for the same gates.
    #[must_use]
    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }

    /// `(the backing file's size, its link count)`.
    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        self.source.backing()
    }
}
