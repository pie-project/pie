use model_ir::{Trace, ValueId};

use crate::error::{Fault, Result};

pub use model_exec::store::kv::{
    Geometry, Paging, Reader, Seat, SpaceFacts, indptr, reads, row_of, space_of,
};

pub fn geometry(paging: &Paging, seats: &[Seat]) -> Result<Geometry> {
    Ok(model_exec::store::kv::geometry(paging, seats)?)
}

pub fn geometry_with(paging: &Paging, seats: &[Seat], tables: &[&[u32]]) -> Result<Geometry> {
    Ok(model_exec::store::kv::geometry_with(paging, seats, tables)?)
}

pub fn width_of(trace: &Trace, value: ValueId) -> Result<u64> {
    Ok(model_exec::store::kv::width_of(trace, value)?)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Facts {
    pub rows: Vec<Option<SpaceFacts>>,

    pub plans: Vec<Option<SpaceFacts>>,
}

impl Facts {
    pub fn row(&self, at: usize, name: &str) -> Result<SpaceFacts> {
        self.rows
            .get(at)
            .copied()
            .flatten()
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "cache `{name}`, which no attention op reads — so nothing states its heads"
                ),
            })
    }
}

pub fn probe(trace: &Trace) -> Result<Facts> {
    let mut out = Facts {
        rows: vec![None; trace.caches.len()],
        plans: vec![None; trace.values.len()],
    };

    for node in &trace.nodes {
        let Some(read) = reads(&node.op) else {
            continue;
        };
        let Some(row) = row_of(trace, read.cache) else {
            continue;
        };
        let seat = out.rows.get_mut(row).ok_or_else(|| Fault::Unbound {
            what: format!("cache row {row}, which this plan does not declare"),
        })?;
        match seat {
            None => {
                *seat = Some(SpaceFacts {
                    head_dim: read.head_dim,
                    kv_heads: read.kv_heads.unwrap_or(0),
                    q_heads: 0,
                    window: None,
                });
            }
            Some(known) => {
                if known.head_dim != read.head_dim {
                    return Err(Fault::Unbound {
                        what: format!(
                            "cache row {row}, whose readers disagree about its shape: \
                             head_dim {} against head_dim {}",
                            known.head_dim, read.head_dim
                        ),
                    });
                }
                if let Some(heads) = read.kv_heads {
                    if known.kv_heads != 0 && known.kv_heads != heads {
                        return Err(Fault::Unbound {
                            what: format!(
                                "cache row {row}, whose readers state {} and {heads} kv heads",
                                known.kv_heads
                            ),
                        });
                    }
                    known.kv_heads = heads;
                }
            }
        }
    }

    for node in &trace.nodes {
        let Some(read) = reads(&node.op) else {
            continue;
        };
        let width = width_of(trace, read.q)?;
        if read.head_dim == 0 || width % u64::from(read.head_dim) != 0 {
            return Err(Fault::Unbound {
                what: format!(
                    "plan value {}, whose query rectangle is {width} wide and whose head \
                     width is {} — not a whole number of heads",
                    read.plan.0, read.head_dim
                ),
            });
        }
        let kv_heads = row_of(trace, read.cache)
            .and_then(|row| out.rows.get(row).copied().flatten())
            .map_or(0, |row| row.kv_heads);
        let stated = SpaceFacts {
            head_dim: read.head_dim,
            kv_heads,
            q_heads: u32::try_from(width / u64::from(read.head_dim)).unwrap_or(u32::MAX),
            window: read.window,
        };
        let seat = out
            .plans
            .get_mut(read.plan.0 as usize)
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "plan value {}, which this plan does not declare",
                    read.plan.0
                ),
            })?;
        match seat {
            None => *seat = Some(stated),
            Some(known) if *known == stated => {}
            Some(known) => {
                return Err(Fault::Unbound {
                    what: format!(
                        "plan value {}, whose launches read it at two readings — {known:?} \
                         against {stated:?}. A schedule is carved for ONE of them (the \
                         window sizes its kv chunking, the head width its tile), so the \
                         model text mints a second plan for the second reading rather \
                         than sharing this one",
                        read.plan.0
                    ),
                });
            }
        }
    }
    Ok(out)
}
