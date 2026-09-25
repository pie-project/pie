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
        }
        | Attention::DecodeSelected {
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
        }
        | Attention::MaskedLse {
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
        }
        | Attention::PrefillSelected {
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
    pub window: Option<Windowed>,
}

/// The windowed kv spaces' paging: `tokens` is the widest window a
/// windowed row is read through, `fire_pages` the most one fire's rows add
/// past the windows of the lanes they land in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Windowed {
    pub tokens: u32,
    pub fire_pages: u32,
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
            window: None,
        })
    }

    /// The paging with windowed spaces read through `tokens` and fires of
    /// at most `max_tokens` rows. A window that spans the context holds
    /// every page, so it is no window at all.
    #[must_use]
    pub fn windowed(mut self, tokens: Option<u32>, max_tokens: u32) -> Paging {
        self.window = tokens
            .filter(|&tokens| tokens < self.context())
            .map(|tokens| Windowed {
                tokens,
                fire_pages: max_tokens.div_ceil(self.page_size) + 1,
            });
        self
    }

    /// The windowed pages one sequence holds between fires: its window,
    /// which may straddle one more page than it fills.
    #[must_use]
    pub fn window_held(&self) -> u32 {
        self.window.map_or(self.pages_per_slot, |w| {
            (w.tokens.div_ceil(self.page_size) + 1).min(self.pages_per_slot)
        })
    }

    /// The windowed pages a slot-seated lane cycles through: its window
    /// and the most one fire's rows add past it.
    #[must_use]
    pub fn window_ring(&self) -> u32 {
        self.window.map_or(self.pages_per_slot, |w| {
            (self.window_held() + w.fire_pages).min(self.pages_per_slot)
        })
    }

    /// The windowed pool beside `pages` kv pages: page 0, the null page
    /// every page behind a window reads, one sequence's window and fire
    /// rows, then a page for every two kv pages past that sequence's. A
    /// short sequence holds as many windowed pages as kv pages, so a pool
    /// split by long sequences alone would seat far fewer short ones than
    /// the kv pages do; the half keeps them within a few percent while a
    /// long one still costs its window.
    #[must_use]
    pub fn window_pages_at(&self, pages: u64) -> u64 {
        match self.window {
            None => pages,
            Some(_) => {
                let past = pages.saturating_sub(u64::from(self.pages_per_slot));
                1 + pages.min(u64::from(self.window_ring()) + past / 2)
            }
        }
    }

    #[must_use]
    pub fn window_pages(&self) -> u64 {
        self.window_pages_at(self.pages)
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

/// A lane's table in a windowed space, as long as its table in the full
/// one so the lengths, schedules and mask columns read the same: a page
/// wholly behind the window of the lane's first row reads the null page 0,
/// the rest the windowed ids the lane was handed (`ids`, aligned with
/// `table`). A lane handed none keeps its own pages one past the null
/// page, refused past the pool rather than folded onto a live one, and a
/// lane seated by slot cycles through a ring of the slot's pages.
pub fn window_table(paging: &Paging, seat: &Seat, table: &[u32], ids: &[u32]) -> Result<Vec<u32>> {
    let Some(window) = paging.window else {
        return Ok(table.to_vec());
    };
    let page_size = u64::from(paging.page_size);
    let pages = (u64::from(seat.have) + u64::from(seat.rows))
        .div_ceil(page_size)
        .max(1);
    let first = u64::from(seat.have.saturating_sub(window.tokens)) / page_size;
    let ring = u64::from(paging.window_ring());
    let capacity = paging.window_pages();
    (0..pages)
        .map(|page| {
            if page < first {
                return Ok(0);
            }
            let id = if !ids.is_empty() {
                u64::from(*ids.get(page as usize).ok_or(Fault::Ceiling {
                    what: "windowed pages this lane stated",
                    need: page + 1,
                    have: ids.len() as u64,
                })?)
            } else if table.is_empty() {
                1 + u64::from(seat.slot) * ring + page % ring
            } else {
                u64::from(table.get(page as usize).copied().unwrap_or(u32::MAX)) + 1
            };
            if id == 0 {
                return Err(Fault::Unbound {
                    what: format!(
                        "windowed page {page} of a lane holding {} tokens, which its window \
                         still reads and the runtime has released",
                        seat.have
                    ),
                });
            }
            if id >= capacity {
                return Err(Fault::Ceiling {
                    what: "windowed kv pages",
                    need: id + 1,
                    have: capacity,
                });
            }
            narrow(id, "windowed page indices").map(|id| id as u32)
        })
        .collect()
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
        a_windowed_table_reads_the_null_page_behind_the_window();
    }

    fn a_windowed_table_reads_the_null_page_behind_the_window() {
        let paging = Paging::of(16, 256, 1, 16)
            .expect("a page size of 16 spells geometry")
            .windowed(Some(32), 16);
        let seat = Seat {
            slot: 0,
            have: 70,
            rows: 1,
        };
        let table = [4, 3, 2, 1, 0];
        assert_eq!(
            window_table(&paging, &seat, &table, &[0, 0, 3, 4, 2]).expect("the ids stand"),
            vec![0, 0, 3, 4, 2],
            "tokens 38 on sit from page 2, so pages 0 and 1 read the null page"
        );
        assert_eq!(
            window_table(&paging, &seat, &table, &[]).expect("the lane's own pages"),
            vec![0, 0, 3, 2, 1],
        );
        assert!(
            window_table(&paging, &seat, &table, &[5, 1, 0, 4, 2]).is_err(),
            "a page the window still reads is never the null page"
        );
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
