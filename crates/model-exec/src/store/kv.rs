use model_ir::{Attention, CacheRow, Def, Dim, Operation, Trace, Ty, ValueId};

use crate::store::{Fault, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceFacts {
    pub head_dim: u32,
    pub kv_heads: u32,
    pub q_heads: u32,
    pub window: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reader {
    pub q: ValueId,
    pub plan: ValueId,
    pub cache: ValueId,
    pub head_dim: u32,
    pub kv_heads: Option<u32>,
    pub window: Option<u32>,
}

#[must_use]
pub fn reads(op: &Operation) -> Option<Reader> {
    let Operation::Attention(op) = op else {
        return None;
    };
    match op {
        Attention::Decode {
            q,
            plan,
            cache,
            window,
            head_dim,
            ..
        }
        | Attention::DecodeLse {
            q,
            plan,
            cache,
            window,
            head_dim,
            ..
        }
        | Attention::DecodeRel {
            q,
            plan,
            cache,
            window,
            head_dim,
            ..
        } => Some(Reader {
            q: *q,
            plan: *plan,
            cache: *cache,
            head_dim: *head_dim,
            kv_heads: None,
            window: *window,
        }),
        Attention::Masked {
            q,
            plan,
            cache,
            window,
            head_dim,
            kv_heads,
            ..
        } => Some(Reader {
            q: *q,
            plan: *plan,
            cache: *cache,
            head_dim: *head_dim,
            kv_heads: Some(*kv_heads),
            window: *window,
        }),
        Attention::Prefill {
            q,
            plan,
            cache,
            window,
            head_dim,
            kv_heads,
            ..
        }
        | Attention::PrefillLse {
            q,
            plan,
            cache,
            window,
            head_dim,
            kv_heads,
            ..
        }
        | Attention::PrefillRel {
            q,
            plan,
            cache,
            window,
            head_dim,
            kv_heads,
            ..
        } => Some(Reader {
            q: *q,
            plan: *plan,
            cache: *cache,
            head_dim: *head_dim,
            kv_heads: Some(*kv_heads),
            window: *window,
        }),
        _ => None,
    }
}

#[must_use]
pub fn row_of(trace: &Trace, cache: ValueId) -> Option<usize> {
    let Def::Cache(row) = trace.values.get(cache.0 as usize)?.def else {
        return None;
    };
    match trace.caches.get(row as usize)? {
        CacheRow::Kv { .. } => Some(row as usize),
        CacheRow::State { .. } => None,
    }
}

#[must_use]
pub fn space_of(trace: &Trace, cache: ValueId) -> Option<u32> {
    let Def::Cache(row) = trace.values.get(cache.0 as usize)?.def else {
        return None;
    };
    match trace.caches.get(row as usize)? {
        CacheRow::Kv { space, .. } => Some(*space),
        CacheRow::State { .. } => None,
    }
}

pub fn width_of(trace: &Trace, value: ValueId) -> Result<u64> {
    let decl = trace
        .values
        .get(value.0 as usize)
        .ok_or_else(|| Fault::Unbound {
            what: format!("value {}, which its own plan does not declare", value.0),
        })?;
    let Ty::Tensor { shape, .. } = &decl.ty else {
        return Err(Fault::Unbound {
            what: format!(
                "value {}, which declares a host struct, as a rectangle",
                value.0
            ),
        });
    };
    let mut width = 1u64;
    for dim in shape.iter().skip(1) {
        match dim {
            Dim::Const(n) => width = width.saturating_mul(*n),
            other => {
                return Err(Fault::Unbound {
                    what: format!(
                        "value {}, whose width carries the symbolic dim {other:?}",
                        value.0
                    ),
                });
            }
        }
    }
    Ok(width)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Paging {
    pub page_size: u32,
    pub pages_per_slot: u32,
    pub slots: u32,
    pub pages: u64,
}

impl Paging {
    pub fn of(page_size: u32, context: u32, slots: u32, pages: u64) -> Result<Paging> {
        if page_size == 0 {
            return Err(Fault::Ceiling {
                what: "tokens per page",
                need: 1,
                have: 0,
            });
        }
        Ok(Paging {
            page_size,
            pages_per_slot: context.div_ceil(page_size).max(1),
            slots,
            pages: pages.max(1),
        })
    }

    #[must_use]
    pub fn pages(&self) -> u64 {
        self.pages
    }

    #[must_use]
    pub fn context(&self) -> u32 {
        self.pages_per_slot.saturating_mul(self.page_size)
    }

    #[must_use]
    pub fn base(&self, slot: u32) -> u64 {
        u64::from(slot) * u64::from(self.pages_per_slot)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seat {
    pub slot: u32,
    pub have: u32,
    pub rows: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Geometry {
    pub indptr: Vec<i32>,
    pub indices: Vec<i32>,
    pub last_page_len: Vec<i32>,
    pub kv_len: Vec<i32>,
    pub write_page: Vec<i32>,
    pub write_offset: Vec<i32>,
}

impl Geometry {
    pub fn pad_to(&mut self, lanes: usize) {
        pad_indptr(&mut self.indptr, lanes);
        self.last_page_len
            .resize(self.last_page_len.len().max(lanes), 0);
        self.kv_len.resize(self.kv_len.len().max(lanes), 0);
    }
}

pub fn pad_indptr(indptr: &mut Vec<i32>, lanes: usize) {
    let Some(&last) = indptr.last() else {
        return;
    };
    indptr.resize(indptr.len().max(lanes + 1), last);
}

pub fn geometry(paging: &Paging, seats: &[Seat]) -> Result<Geometry> {
    geometry_with(paging, seats, &[])
}

pub fn geometry_with(paging: &Paging, seats: &[Seat], tables: &[&[u32]]) -> Result<Geometry> {
    let rows: u64 = seats.iter().map(|s| u64::from(s.rows)).sum();
    let mut out = Geometry {
        indptr: Vec::with_capacity(seats.len() + 1),
        indices: Vec::new(),
        last_page_len: Vec::with_capacity(seats.len()),
        kv_len: Vec::with_capacity(seats.len()),
        write_page: Vec::with_capacity(rows as usize),
        write_offset: Vec::with_capacity(rows as usize),
    };
    out.indptr.push(0);

    for (lane, seat) in seats.iter().enumerate() {
        let table = tables.get(lane).copied().unwrap_or(&[]);
        let after = u64::from(seat.have) + u64::from(seat.rows);
        let pages = after.div_ceil(u64::from(paging.page_size)).max(1);

        if table.is_empty() {
            if seat.slot >= paging.slots {
                return Err(Fault::Ceiling {
                    what: "kv slots",
                    need: u64::from(seat.slot) + 1,
                    have: u64::from(paging.slots),
                });
            }
            if after > u64::from(paging.context()) {
                return Err(Fault::Ceiling {
                    what: "kv tokens in one slot",
                    need: after,
                    have: u64::from(paging.context()),
                });
            }
            let base = paging.base(seat.slot);
            for page in 0..pages {
                out.indices.push(narrow(base + page, "page indices")?);
            }
            for token in 0..u64::from(seat.rows) {
                let at = u64::from(seat.have) + token;
                out.write_page.push(narrow(
                    base + at / u64::from(paging.page_size),
                    "page indices",
                )?);
                out.write_offset
                    .push(narrow(at % u64::from(paging.page_size), "write offsets")?);
            }
        } else {
            if (table.len() as u64) < pages {
                return Err(Fault::Ceiling {
                    what: "kv pages this lane stated",
                    need: pages,
                    have: table.len() as u64,
                });
            }
            for &page in &table[..pages as usize] {
                out.indices.push(narrow(u64::from(page), "page indices")?);
            }
            for token in 0..u64::from(seat.rows) {
                let at = u64::from(seat.have) + token;
                let page = table[(at / u64::from(paging.page_size)) as usize];
                out.write_page
                    .push(narrow(u64::from(page), "page indices")?);
                out.write_offset
                    .push(narrow(at % u64::from(paging.page_size), "write offsets")?);
            }
        }

        out.indptr
            .push(narrow(out.indices.len() as u64, "kv indptr")?);
        out.last_page_len.push(narrow(
            after - (pages - 1) * u64::from(paging.page_size),
            "last page length",
        )?);
        out.kv_len.push(narrow(after, "kv length")?);
    }
    Ok(out)
}

pub fn indptr(seats: &[Seat]) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(seats.len() + 1);
    let mut at = 0u64;
    out.push(0);
    for seat in seats {
        at += u64::from(seat.rows);
        out.push(narrow(at, "qo indptr")?);
    }
    Ok(out)
}

fn narrow(n: u64, what: &'static str) -> Result<i32> {
    i32::try_from(n).map_err(|_| Fault::Ceiling {
        what,
        need: n,
        have: u64::try_from(i32::MAX).unwrap_or(u64::MAX),
    })
}

#[cfg(test)]
mod tests {

    use super::*;

    fn paging() -> Paging {
        Paging::of(16, 64, 4, 16).expect("a page size of 16 spells geometry")
    }

    #[test]
    fn kv_every_case() {
        a_prefill_writes_its_own_prompt_and_then_attends_it();
        a_decode_step_appends_one_row_past_what_the_slot_holds();
        a_sequence_past_its_slots_pages_is_refused_rather_than_wrapped();
    }

    fn a_prefill_writes_its_own_prompt_and_then_attends_it() {
        let g = geometry(
            &paging(),
            &[Seat {
                slot: 1,
                have: 0,
                rows: 20,
            }],
        )
        .expect("one lane of 20 rows pages");

        assert_eq!(g.indptr, vec![0, 2]);
        assert_eq!(g.indices, vec![4, 5], "slot 1's block starts at page 4");
        assert_eq!(g.kv_len, vec![20], "the length is AFTER the append");
        assert_eq!(g.last_page_len, vec![4]);
        assert_eq!(g.write_page.len(), 20);
        assert_eq!(
            &g.write_page[..17],
            &[4; 16].iter().chain(&[5]).copied().collect::<Vec<_>>()[..]
        );
        assert_eq!(g.write_offset[0], 0);
        assert_eq!(g.write_offset[15], 15);
        assert_eq!(g.write_offset[16], 0, "row 16 opens the second page");
    }

    fn a_decode_step_appends_one_row_past_what_the_slot_holds() {
        let g = geometry(
            &paging(),
            &[Seat {
                slot: 0,
                have: 20,
                rows: 1,
            }],
        )
        .expect("one decode row pages");

        assert_eq!(g.kv_len, vec![21]);
        assert_eq!(g.indptr, vec![0, 2]);
        assert_eq!(g.last_page_len, vec![5]);
        assert_eq!(
            g.write_page,
            vec![1],
            "token 20 is page 1 of slot 0's block"
        );
        assert_eq!(g.write_offset, vec![4]);
    }

    fn a_sequence_past_its_slots_pages_is_refused_rather_than_wrapped() {
        let refusal = geometry(
            &paging(),
            &[Seat {
                slot: 0,
                have: 60,
                rows: 8,
            }],
        );
        assert!(
            matches!(
                refusal,
                Err(Fault::Ceiling {
                    need: 68,
                    have: 64,
                    ..
                })
            ),
            "a slot that overruns its block must name the numbers: {refusal:?}"
        );
    }
}
