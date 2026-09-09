use std::collections::{BTreeMap, BTreeSet, HashMap};

use model_compiler::CompiledModel;
use model_exec::fire::MaskSpan;
use model_ir::ops::{Attention, Layout};
use model_ir::{Def, Operation, Trace, ValueId};

use crate::device::Handles;
use crate::device::alloc::Buffer;
use crate::error::{Fault, Result};
use crate::experts::{Attachments, Source};
use crate::weight_store::Store;
use kernels_metal::Tensor;

#[derive(Debug, Clone, Default)]
pub struct Table {
    pub name: String,
    pub params: Vec<usize>,
    pub rows: u64,
    pub seats: u32,
    pub strides: Vec<u64>,
    host_of: BTreeMap<usize, u64>,
    host_bytes: u64,
    device_bytes: u64,
}

impl Table {
    #[must_use]
    pub fn stored(&self) -> u64 {
        self.strides
            .iter()
            .map(|stride| (self.rows * stride).next_multiple_of(crate::weights::ALIGN))
            .sum()
    }

    #[must_use]
    pub fn slab(&self) -> u64 {
        self.device_bytes
    }
}

#[derive(Debug, Clone, Default)]
pub struct Plan {
    table: Option<Table>,
}

impl Plan {
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
            let seats = u64::from(max_tokens).saturating_mul(heads as u64).min(rows);
            let seats = u32::try_from(seats).unwrap_or(u32::MAX).max(1);
            let mut host_of = BTreeMap::new();
            let mut host_bytes = 0u64;
            let mut device_bytes = 0u64;
            for (at, &param) in params.iter().enumerate() {
                host_of.insert(param, host_bytes);
                host_bytes += bytes[param];
                device_bytes +=
                    (u64::from(seats) * strides[at]).next_multiple_of(crate::weights::ALIGN);
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
                table.name, bytes[table.params[0]], table.seats, table.device_bytes, table.rows,
            )));
        }
        let _ = rest;
        Ok(Plan { table: Some(table) })
    }

    #[must_use]
    pub fn gathers(&self) -> bool {
        self.table.is_some()
    }

    #[must_use]
    pub fn table(&self) -> Option<&Table> {
        self.table.as_ref()
    }

    #[must_use]
    pub fn resident(&self, param: usize) -> Option<u32> {
        let table = self.table.as_ref()?;
        table.params.contains(&param).then_some(table.seats)
    }

    #[must_use]
    pub fn host_at(&self, param: usize) -> Option<u64> {
        self.table.as_ref()?.host_of.get(&param).copied()
    }

    #[must_use]
    pub fn params(&self) -> BTreeSet<usize> {
        self.table
            .as_ref()
            .map(|t| t.params.iter().copied().collect())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn device_demand(&self) -> u64 {
        self.table.as_ref().map_or(0, |t| t.device_bytes)
    }

    #[must_use]
    pub fn source_bytes(&self) -> u64 {
        self.table.as_ref().map_or(0, |t| t.host_bytes)
    }

    #[must_use]
    pub fn host_bands(&self) -> BTreeMap<usize, u64> {
        self.table
            .as_ref()
            .map(|t| t.host_of.clone())
            .unwrap_or_default()
    }

    #[must_use]
    pub fn vocab(&self, param: usize) -> Option<u32> {
        let table = self.table.as_ref()?;
        (table.params.first() == Some(&param)).then_some(table.seats)
    }
}

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

#[derive(Debug)]
struct Band {
    at: u64,
    from: u64,
    stride: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Residency {
    pub name: String,
    pub rows: u64,
    pub seats: u32,
    pub demanded: u32,
}

#[derive(Debug)]
pub struct Slab {
    store: Store,
    source: Source,
    bands: Vec<Band>,
    rows: u64,
    seats: u32,
    name: String,
    seat_of: HashMap<i32, u32>,
    in_seat: Vec<i32>,
    next: u32,
    fires: u64,
    copies: u64,
}

impl Slab {
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

    pub fn fire(&mut self) {
        self.seat_of.clear();
        self.in_seat.fill(-1);
        self.next = 0;
        self.fires += 1;
    }

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
        for entry in raw.as_chunks_mut::<4>().0 {
            let id = i32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
            if id < 0 {
                continue;
            }
            if u64::from(id.unsigned_abs()) >= self.rows {
                entry.copy_from_slice(&(self.seats as i32).to_le_bytes());
                continue;
            }
            let seat = self.seat(id)?;
            entry.copy_from_slice(&(seat as i32).to_le_bytes());
        }
        arena.write(first, &raw)?;
        Ok(())
    }

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

    #[must_use]
    pub fn residency(&self) -> Residency {
        Residency {
            name: self.name.clone(),
            rows: self.rows,
            seats: self.seats,
            demanded: self.next,
        }
    }

    #[must_use]
    pub fn motion(&self) -> (u64, u64) {
        (self.copies, self.fires)
    }

    #[must_use]
    pub fn source_kind(&self) -> &'static str {
        self.source.kind()
    }

    #[must_use]
    pub fn backing(&self) -> Option<(u64, u64)> {
        self.source.backing()
    }
}
