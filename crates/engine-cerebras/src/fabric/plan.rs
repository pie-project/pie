//! The engine's planner: the split of every op by what its entry states,
//! within the terms of the fabric.

use kernels_cerebras::error::{Error, refuse};
use kernels_cerebras::linear::gemm::FABRIC_SUM_WORDS;
use kernels_cerebras::phase::{BlockPlan, Reduce};
use kernels_cerebras::plan::{
    BlockSplit, Blocks, Fabric, Footprint, Held, LSE_GROUPS, LaneSplit, Merge, PageSplit,
    PagedLanes, Pages, Plan, Product, Routed, RoutedSplit, Slots, Spread,
};

/// The most words one array holds: a word address of 16 bits (the probe's
/// 8448-word array fails a 14-bit-plus-shift relocation).
pub const ARRAY_WORDS: u64 = 8192;

/// The planner over a set of terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Planner {
    pub fabric: Fabric,
}

impl Plan for Planner {
    fn fabric(&self) -> Fabric {
        self.fabric
    }

    fn holds(&self, held: Held) -> bool {
        held.words <= self.fabric.pe_words && held.array <= ARRAY_WORDS
    }

    fn tile(&self, op: &'static str, footprint: Footprint) -> Result<(u32, u32), Error> {
        tile(&self.fabric, op, footprint)
    }

    fn grid(&self, product: Product) -> Option<(u32, u32, u32)> {
        grid(&self.fabric, product)
    }

    fn spread(&self, spread: &Spread<'_>) -> Option<u32> {
        self::spread(&self.fabric, spread)
    }

    fn blocks(&self, blocks: &Blocks<'_>) -> Option<BlockSplit> {
        self::blocks(&self.fabric, blocks)
    }

    fn slots(&self, op: &'static str, slots: &Slots<'_>) -> Result<LaneSplit, Error> {
        self::slots(&self.fabric, op, slots)
    }

    fn routed(&self, routed: &Routed<'_>) -> Option<RoutedSplit> {
        self::routed(&self.fabric, routed)
    }

    fn pages(&self, op: &'static str, pages: &Pages<'_>) -> Result<PageSplit, Error> {
        self::pages(&self.fabric, op, pages)
    }

    fn merge(&self, merge: Merge) -> bool {
        self::merge(&self.fabric, merge)
    }
}

/// Whether the partials of `merge` combine on the fabric: when the terms
/// merge there at all and there is more than one, an lse merge gathering
/// at most [`LSE_GROUPS`] copies unless it spans the resident row.
pub fn merge(fab: &Fabric, merge: Merge) -> bool {
    let (group, capped) = match merge {
        Merge::Sum { group } | Merge::Pick { group } => (group, false),
        Merge::Lse { group, row } => (group, !row && group > LSE_GROUPS),
    };
    fab.fabric_sum && group > 1 && !capped
}

/// The `(row groups, column groups)` a row-wise op of `footprint` spreads
/// over so that one PE's share fits the terms' budget, each share's rows
/// within an array: the fewest PEs win, rows first; the resident placement
/// runs it whole on every PE of its row.
pub fn tile(fabric: &Fabric, op: &'static str, footprint: Footprint) -> Result<(u32, u32), Error> {
    let Footprint {
        rows,
        width,
        per_row,
        per_col,
        whole,
        merge,
        unit,
    } = footprint;
    // A fabric merge's words are kept whatever the split.
    let whole = whole + if fabric.fabric_sum { merge } else { 0 };
    let budget = fabric.pe_words;
    let (rows, width, unit) = (rows.max(1), width.max(1), unit.max(1));
    if fabric.resident_pes.is_some() {
        let words = per_row * u64::from(rows) + per_col * u64::from(width) + whole;
        return if words <= budget && per_row * u64::from(rows) <= ARRAY_WORDS {
            Ok((1, 1))
        } else {
            Err(refuse(
                op,
                format!("the phase's {words} words exceed a resident PE's {budget}"),
            ))
        };
    }
    let units = width / unit;
    let fits = |rg: u32, cg: u32| {
        let words = per_row * u64::from(rows / rg) / u64::from(cg)
            + per_col * u64::from(width / cg)
            + whole;
        words <= budget && per_row * u64::from(rows / rg) / u64::from(cg) <= ARRAY_WORDS
    };
    let mut best: Option<(u32, u32)> = None;
    for cg in (1..=units).filter(|d| units.is_multiple_of(*d)) {
        for rg in (1..=rows).filter(|d| rows.is_multiple_of(*d)) {
            if fits(rg, cg) {
                if best.is_none_or(|(br, bc)| rg * cg < br * bc) {
                    best = Some((rg, cg));
                }
                break;
            }
        }
    }
    best.ok_or_else(|| {
        refuse(
            op,
            format!(
                "one row of {width} ({per_row} words with its planes) does not fit a PE's {budget} even one column of {unit} at a time"
            ),
        )
    })
}

/// A grid `(rg, cg, kg)` with its `(words moved, PEs)` key.
type GridPick = ((u32, u32, u32), (u64, u32));

/// The `(row groups, column groups, depth groups)` grid of PEs a `m × k`
/// by `n × k` product spreads over. PE `(r, c, d)` holds rows `r` and depth
/// slice `d` of the activations, weight rows `c` at depth `d`, and a
/// partial sum of output block `(r, c)`, which the host adds over `d`: the
/// words moved are the activations once per column group, the weight once
/// per row group, and the output once per depth group. The grid moving the
/// fewest words whose shares fit [`Fabric::pe_words`] (arrays within
/// `ARRAY_WORDS`) on at most [`Fabric::max_pes`] PEs wins, the fewer PEs the
/// better among equals.
pub fn grid(fab: &Fabric, product: Product) -> Option<(u32, u32, u32)> {
    let Product { m, k, n, packed } = product;
    // A weight no row of PEs holds (the vocabulary projection, by default)
    // is multiplied on the host.
    let per = if packed { 2 } else { 1 };
    if u64::from(n) * u64::from(k) / per > u64::from(fab.max_pes) * fab.shard_words {
        return None;
    }
    let budget = fab.pe_words;
    let most = fab.max_pes;
    let fabric = fab.fabric_sum;
    let divisors = |x: u32| (1..=x).filter(move |d| x.is_multiple_of(*d));
    let (m64, k64, n64) = (u64::from(m), u64::from(k), u64::from(n));
    let mut best: Option<GridPick> = None;
    for rg in divisors(m) {
        for cg in divisors(n) {
            for kg in divisors(k) {
                if packed && !(k / kg).is_multiple_of(2) {
                    continue;
                }
                let a = (m64 / u64::from(rg)) * (k64 / u64::from(kg));
                let w = (n64 / u64::from(cg)) * (k64 / u64::from(kg)) / per;
                let y = (m64 / u64::from(rg)) * (n64 / u64::from(cg));
                let extra = if kg > 1 && fabric {
                    y + FABRIC_SUM_WORDS
                } else {
                    0
                };
                if a.max(w).max(y) > ARRAY_WORDS
                    || a + w + y + extra > budget
                    || rg * cg * kg > most
                {
                    continue;
                }
                let moved = m64 * k64 * u64::from(cg)
                    + n64 * k64 * u64::from(rg)
                    + m64 * n64 * u64::from(kg);
                let key = (moved, rg * cg * kg);
                if best.is_none_or(|(_, k)| key < k) {
                    best = Some(((rg, cg, kg), key));
                }
                break;
            }
        }
    }
    best.map(|(g, _)| g)
}

/// The most PEs one collective over a row of PEs spans.
const ROW_PES: u32 = 256;

/// The groups a [`Spread`] splits into: the fewest that divide its units
/// and whose share fits a PE; groups combining over the fabric span at
/// most one rectangle row and take the collectives' words, and under the
/// resident placement are the whole row or one; any other split is capped
/// at [`Fabric::max_pes`].
pub fn spread(fab: &Fabric, spread: &Spread<'_>) -> Option<u32> {
    let Spread {
        units,
        collective,
        held,
    } = *spread;
    let fits = |n: u32| {
        let h = held(n);
        let collectives = if collective && n > 1 {
            FABRIC_SUM_WORDS
        } else {
            0
        };
        h.array <= ARRAY_WORDS && h.words + collectives <= fab.pe_words
    };
    let divides = |n: u32| units.is_multiple_of(n);
    if !collective {
        return (1..=units)
            .filter(|n| divides(*n) && *n <= fab.max_pes)
            .find(|n| fits(*n));
    }
    if let Some(n) = fab.resident_pes {
        return [n, 1].into_iter().find(|n| divides(*n) && fits(*n));
    }
    (1..=ROW_PES.min(units))
        .filter(|n| divides(*n))
        .find(|n| fits(*n))
}

/// How blocks of a state split over PEs: all the rows when a split holds
/// them, else a window of them halved until one does; the split, the
/// fewest PEs, each `b` group taking the first `a` grouping that fits.
pub fn blocks(fab: &Fabric, blocks: &Blocks<'_>) -> Option<BlockSplit> {
    let Blocks {
        rows,
        a,
        b,
        code,
        held,
        merge: merged,
    } = *blocks;
    let held = |window: u32, ag: u32, bg: u32| {
        let lse = Merge::Lse {
            group: ag,
            row: false,
        };
        let merged = if merge(fab, lse) {
            merged(window, ag, bg)
        } else {
            0
        };
        held(window, ag, bg) + merged
    };
    let budget = fab.pe_words.saturating_sub(code);
    let divisors = |x: u32| (1..=x).filter(move |d| x.is_multiple_of(*d));
    let mut window = rows;
    loop {
        let mut best: Option<(u32, u32, u32)> = None;
        for bg in divisors(b) {
            for ag in divisors(a) {
                if held(window, ag, bg) <= budget {
                    let pes = ag * bg;
                    if best.is_none_or(|(p, ..)| pes < p) {
                        best = Some((pes, ag, bg));
                    }
                    break;
                }
            }
        }
        match best {
            Some((_, a_groups, b_groups)) => {
                return Some(BlockSplit {
                    window,
                    a_groups,
                    b_groups,
                });
            }
            None if window > 1 => window = window.div_ceil(2),
            None => return None,
        }
    }
}

/// How lanes over a recurrent bank spread over PEs: whole slots on one PE
/// when the bank and the words beside it fit [`Fabric::pe_words`], else each
/// slot split along `b` and then (when `a_splits`) `a` until one block plus
/// the words beside it fits, as many lanes a PE as the rest allows.
pub fn slots(fab: &Fabric, op: &'static str, slots: &Slots<'_>) -> Result<LaneSplit, Error> {
    let Slots {
        elements,
        lanes,
        a,
        b,
        w,
        a_splits,
        beside,
    } = *slots;
    if lanes == 0 || (elements + beside(1, 1) <= fab.pe_words && elements <= ARRAY_WORDS) {
        return Ok(LaneSplit {
            pes: 1,
            lanes_per_pe: lanes.max(1),
            block: None,
        });
    }
    let divisors = |n: u32| (1..=n).filter(move |d| n.is_multiple_of(*d));
    let block_words = |ag: u32, bg: u32| u64::from(a / ag) * u64::from(b / bg) * u64::from(w);
    let mut found = None;
    'search: for bg in divisors(b) {
        for ag in divisors(if a_splits { a } else { 1 }) {
            if block_words(ag, bg) + beside(ag, bg) <= fab.pe_words {
                found = Some((ag, bg));
                break 'search;
            }
        }
    }
    let Some((a_groups, b_groups)) = found else {
        let budget = fab.pe_words;
        return Err(refuse(
            op,
            format!(
                "one row of a slot's state ({w} words) and the phase's other words ({}) exceed a PE's {budget}",
                beside(a, b)
            ),
        ));
    };
    let block = BlockPlan {
        a,
        b,
        w,
        a_groups,
        b_groups,
    };
    let words = block.words().max(1);
    // The bank is one array: its slot rows within ARRAY_WORDS too.
    let room = fab
        .pe_words
        .saturating_sub(beside(a_groups, b_groups))
        .min(ARRAY_WORDS);
    let lanes_per_pe = (room / words).clamp(1, u64::from(lanes)) as u32;
    let lane_groups = lanes.div_ceil(lanes_per_pe);
    Ok(LaneSplit {
        pes: lane_groups * a_groups * b_groups,
        lanes_per_pe,
        block: Some(block),
    })
}

/// How lanes over a routed bank spread over PEs: the fewest PEs whose
/// lanes' blocks plus the planes fit (a split of `b` summed on the fabric
/// holds the partial again and the collectives' words), lanes a PE dividing
/// the lanes; the partials of a `b` split add on the fabric or the host as
/// [`Fabric::fabric_sum`] says.
pub fn routed(fab: &Fabric, routed: &Routed<'_>) -> Option<RoutedSplit> {
    let Routed {
        lanes,
        a,
        b,
        block,
        planes,
        partial,
    } = *routed;
    let budget = fab.pe_words;
    let divisors = |x: u32| (1..=x).filter(move |d| x.is_multiple_of(*d));
    let mut best: Option<(u32, u32, u32, u32)> = None;
    for bg in divisors(b) {
        for ag in divisors(a) {
            let words = block(ag, bg);
            let mut beside = planes(ag, bg);
            if bg > 1 && fab.fabric_sum {
                beside += partial(ag, bg) + FABRIC_SUM_WORDS;
            }
            if words + beside > budget || words > ARRAY_WORDS {
                continue;
            }
            let most = ((budget - beside) / words.max(1))
                .min(ARRAY_WORDS / words.max(1))
                .clamp(1, u64::from(lanes)) as u32;
            let per_pe = (1..=most)
                .rev()
                .find(|d| lanes.is_multiple_of(*d))
                .unwrap_or(1);
            let pes = (lanes / per_pe) * ag * bg;
            if best.is_none_or(|(p, ..)| pes < p) {
                best = Some((pes, per_pe, ag, bg));
            }
            break;
        }
    }
    best.map(|(pes, lanes_per_pe, a_groups, b_groups)| RoutedSplit {
        pes,
        lanes_per_pe,
        a_groups,
        b_groups,
        sum: (b_groups > 1).then_some(if fab.fabric_sum {
            Reduce::FabricSum { group: b_groups }
        } else {
            Reduce::Sum
        }),
    })
}

/// How a paged pool's lanes spread over PEs, and the rows of one phase:
/// all the rows (at most [`Fabric::attn_rows`]) when a split holds them,
/// else (`windows`) a window of them halved until one does. The resident
/// placement strides the pool over its row; else a PE holds as many lanes'
/// pages as fit beside the rest, a lane's pages past that split over page
/// groups, a page past that its heads and rows; the fewest PEs win.
pub fn pages(fab: &Fabric, op: &'static str, pages: &Pages<'_>) -> Result<PageSplit, Error> {
    let p = pages;
    // A fabric merge's words are kept whatever the split.
    let beside = |window: u32, hg: u32| {
        let merged = if fab.fabric_sum {
            (p.merge)(window, hg)
        } else {
            0
        };
        (p.beside)(window, hg) + merged
    };
    let mut window = if p.windows {
        p.rows.max(1).min(fab.attn_rows.unwrap_or(u32::MAX)).max(1)
    } else {
        p.rows
    };
    loop {
        let halve = p.windows && window > 1;
        if let Some(pes) = fab.resident_pes {
            let beside = beside(window, 1);
            match strided(fab, pes, p, beside) {
                Some(lanes) => {
                    return Ok(PageSplit {
                        window,
                        lanes: Some(lanes),
                    });
                }
                None if halve => {
                    window = window.div_ceil(2);
                    continue;
                }
                None => {
                    return Err(refuse(
                        op,
                        format!(
                            "the resident row of {pes} PEs does not hold the pool's {} pages beside the phase's other {beside} words",
                            p.pool_pages
                        ),
                    ));
                }
            }
        }
        match paged(fab, op, p, &|hg| beside(window, hg)) {
            Ok(lanes) => return Ok(PageSplit { window, lanes }),
            Err(_) if halve => window = window.div_ceil(2),
            Err(e) => return Err(e),
        }
    }
}

/// The paged lanes of [`pages`] for one window (`beside(head_groups)` the
/// words beside pages), or `None` when one PE holds the pool.
fn paged(
    fab: &Fabric,
    op: &'static str,
    p: &Pages<'_>,
    beside: &dyn Fn(u32) -> u64,
) -> Result<Option<PagedLanes>, Error> {
    let lanes = p.lanes;
    if lanes == 0 || (p.elements + beside(1) <= fab.pe_words && p.elements <= ARRAY_WORDS) {
        return Ok(None);
    }
    let max_pages = p.max_pages.max(1);
    let heads = p.heads.max(1);
    let page_rows = p.page_rows.max(1);
    let mut best: Option<PagedLanes> = None;
    for hg in (1..=heads).filter(|g| heads.is_multiple_of(*g)) {
        for ag in (1..=page_rows).filter(|g| page_rows.is_multiple_of(*g)) {
            // A bank plane is one array: its pages within ARRAY_WORDS too.
            let room = fab.pe_words.saturating_sub(beside(hg)).min(2 * ARRAY_WORDS);
            let pair = (p.page_words / u64::from(hg) / u64::from(ag)).max(1);
            let pages_per_pe = u32::try_from(room / pair).unwrap_or(u32::MAX);
            if pages_per_pe == 0 {
                continue;
            }
            let (lanes_per_pe, page_groups) = if pages_per_pe >= max_pages {
                ((pages_per_pe / max_pages).clamp(1, lanes), 1)
            } else {
                (1, max_pages.div_ceil(pages_per_pe))
            };
            let pes = lanes.div_ceil(lanes_per_pe) * page_groups * hg * ag;
            if best.as_ref().is_none_or(|b| pes < b.pes) {
                best = Some(PagedLanes {
                    pes,
                    lanes_per_pe,
                    page_groups,
                    head_groups: hg,
                    row_groups: ag,
                    strided_pages: None,
                });
            }
            break;
        }
    }
    match best {
        Some(b) => Ok(Some(b)),
        None => Err(refuse(
            op,
            format!(
                "one page's {} words and the phase's other {} words exceed a PE's {} even one kv head and one page row at a time",
                p.page_words,
                beside(heads),
                fab.pe_words
            ),
        )),
    }
}

/// The resident placement of a pool over a row of `pes` PEs: every PE runs
/// every lane over the pages it holds (page `g` on PE `g % pes`), `None`
/// when the share and `beside` words do not fit a PE.
fn strided(fab: &Fabric, pes: u32, p: &Pages<'_>, beside: u64) -> Option<PagedLanes> {
    let share = p.pool_pages.div_ceil(pes).max(1);
    let words = u64::from(share) * p.page_words;
    (words / 2 <= ARRAY_WORDS && words + beside <= fab.pe_words).then_some(PagedLanes {
        pes,
        lanes_per_pe: p.lanes,
        page_groups: 1,
        head_groups: 1,
        row_groups: 1,
        strided_pages: Some(share),
    })
}
