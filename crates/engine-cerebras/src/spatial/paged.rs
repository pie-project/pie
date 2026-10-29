//! The paged, batched wafer program's host side (`payload/qwen2d_paged`'s
//! contract, the batched serve2d to come): B lanes a launch, each at its own
//! position through its own page table, KV pages owned on the wafer.
//!
//! The boundary is kept thin while the full program's API settles: the
//! per-pass table (what every layer grid reads) is built here and tested;
//! the symbols and launches it is written with are named in [`Symbols`],
//! one place to change when the program's names are final.
//!
//! The pool: NB · P pages of PS positions; pool page g lives whole on grid
//! row g % P as KV block g / P. A pass's table, per row, is
//! `[pos[B] | slot[B] | blk[B][E] | page[B][E] | vis[B][E]]` (u32; unused
//! entries 0xffffffff), written with one row-broadcast copy (Fork 8's
//! `qwen2d_paged/run.py` `tables`): lane b's RoPE position and KV write
//! slot, then its entries on row r, its pages that live on row r (at most
//! E): their block, logical page index and the bitmap of the page's slots
//! the lane attends (PS <= 32).

use crate::sdk::{DataType, Order, Rect, Runtime};

type R<T> = std::result::Result<T, String>;

/// An unused table entry.
pub const NONE: u32 = 0xffff_ffff;

/// The program's paging shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// Lanes a pass.
    pub lanes: usize,
    /// Grid rows (P): the pool's rows.
    pub rows: usize,
    /// Positions a page.
    pub page_size: usize,
    /// KV blocks a row (NB): pages = NB · P.
    pub blocks: usize,
    /// A lane's entries on one row (E).
    pub entries: usize,
}

impl Shape {
    pub fn pages(&self) -> usize {
        self.blocks * self.rows
    }

    /// Words of one row's table.
    pub fn row_words(&self) -> usize {
        2 * self.lanes + 3 * self.lanes * self.entries
    }

    /// The most pages one lane may hold (E on every row).
    pub fn max_page_refs(&self) -> usize {
        self.entries * self.rows
    }
}

/// One lane of a pass: the RoPE position it rotates by, the KV slot it
/// appends at, the pool pages its table names (logical page i is
/// `pages[i]`) and the slots it attends (`None`: every slot up to its own).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lane {
    pub pos: u32,
    pub slot: u32,
    pub pages: Vec<u32>,
    pub visible: Option<Vec<u32>>,
}

/// The per-pass table, `rows` rows of [`Shape::row_words`] words, for
/// `lanes` (at most [`Shape::lanes`]; the rest repeat lane 0 and are
/// ignored by the caller) and a launch of `steps` passes (a lane writes
/// slots `slot..slot + steps`). Refuses a page outside the pool, a lane
/// short of pages, and a lane with more than E pages on one row.
pub fn table(shape: &Shape, lanes: &[Lane], steps: u32) -> R<Vec<u32>> {
    let (b_n, e_n, p_n) = (shape.lanes, shape.entries, shape.rows);
    let ps = shape.page_size as u32;
    if ps > 32 {
        return Err(format!("pages of {ps} positions do not fit a 32-bit visibility bitmap"));
    }
    let first = lanes.first().ok_or("a pass with no lanes")?;
    if lanes.len() > b_n {
        return Err(format!("{} lanes for a program of {b_n}", lanes.len()));
    }
    let row_words = shape.row_words();
    let mut tab = vec![NONE; p_n * row_words];
    for b in 0..b_n {
        let lane = lanes.get(b).unwrap_or(first);
        let causal: Vec<u32>;
        let visible = match &lane.visible {
            Some(v) => v.as_slice(),
            None => {
                causal = (0..=lane.slot).collect();
                &causal
            }
        };
        let last_written = lane.slot + steps.max(1) - 1;
        let last = (last_written / ps).max(visible.iter().max().map_or(0, |v| v / ps)) as usize;
        if lane.pages.len() <= last {
            return Err(format!(
                "lane {b} reaches slot {} and names {} page(s) of {ps}",
                (last as u32 + 1) * ps - 1,
                lane.pages.len()
            ));
        }
        for r in 0..p_n {
            tab[r * row_words + b] = lane.pos;
            tab[r * row_words + b_n + b] = lane.slot;
        }
        let mut on_row = vec![0usize; p_n];
        for (logical, &g) in lane.pages[..=last].iter().enumerate() {
            let g = g as usize;
            if g >= shape.pages() {
                return Err(format!("lane {b} names page {g} of a pool of {}", shape.pages()));
            }
            let (r, block) = (g % p_n, g / p_n);
            let e = on_row[r];
            if e >= e_n {
                return Err(format!(
                    "lane {b} holds more than {e_n} page(s) on pool row {r}; the program reads \
                     at most {e_n} a row"
                ));
            }
            on_row[r] += 1;
            let lo = logical as u32 * ps;
            let bits = visible
                .iter()
                .filter(|&&v| v >= lo && v < lo + ps)
                .fold(0u32, |acc, &v| acc | (1 << (v - lo)));
            let base = r * row_words + 2 * b_n;
            tab[base + b * e_n + e] = block as u32;
            tab[base + b_n * e_n + b * e_n + e] = logical as u32;
            tab[base + 2 * b_n * e_n + b * e_n + e] = bits;
        }
    }
    Ok(tab)
}

/// What the program calls its pieces.
#[derive(Debug, Clone)]
pub struct Symbols {
    /// The per-pass table, on every PE of a grid (rows broadcast).
    pub table: &'static str,
    /// The ring's per-micro-batch tables at the head's feeders (serve2d_pipe).
    pub ring_table: &'static str,
    /// The VM's per-lane control (serve2d_pipe `VM` 1): (gmode_l, forced
    /// prefix) a virtual lane.
    pub vm_lanes: &'static str,
    /// Forced tokens `[B][CAP]` and every step's token `[B][CAP]`, at the
    /// head's final root.
    pub in_tokens: &'static str,
    pub out_tokens: &'static str,
    /// `compute(steps)`; `copy_page(src, dst)` then `copy_move(src, dst)`.
    pub compute: &'static str,
    pub copy_page: &'static str,
    pub copy_move: &'static str,
    /// The head's last logits rows, before 1/rms: serve2d's one sequence
    /// (`logits`), and the strip head's or a batched head's lanes
    /// (`logit_lanes`); each lane's 1/rms (`yscale`).
    pub logits: &'static str,
    pub logit_lanes: &'static str,
    pub yscale: &'static str,
    /// Every layer grid's K and V pools.
    pub kc: &'static str,
    pub vc: &'static str,
}

/// The wafer programs' runtime symbols (their placement symbols stay with
/// the placement that writes them).
pub const SYMBOLS: Symbols = Symbols {
    table: "tab",
    ring_table: "mtab",
    vm_lanes: "vmlane",
    in_tokens: "in_tokens",
    out_tokens: "out_tokens",
    compute: "compute",
    copy_page: "copy_page",
    copy_move: "copy_move",
    logits: "y",
    logit_lanes: "hlogits",
    yscale: "yscale",
    kc: "kc",
    vc: "vc",
};

impl Default for Symbols {
    fn default() -> Symbols {
        SYMBOLS
    }
}

/// Whether `rt`'s program is a paged one (it exports the table and the
/// page copies).
pub fn is_paged(rt: &Runtime, sym: &Symbols) -> bool {
    [sym.table, sym.copy_page, sym.copy_move]
        .iter()
        .all(|name| rt.rpc().get(name).is_ok())
}

/// The table onto every grid (`grids`: each layer grid's rectangle, P rows
/// high), one row-broadcast copy a grid.
pub fn write_table(rt: &mut Runtime, sym: &Symbols, shape: &Shape, grids: &[Rect], tab: &[u32]) -> R<()> {
    for &grid in grids {
        if grid.h as usize != shape.rows {
            return Err(format!("a grid of {} rows for a pool of {}", grid.h, shape.rows));
        }
        rt.memcpy_h2d_rowbcast(sym.table, tab, grid, shape.row_words(), DataType::Bits32)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// `copy_kv`'s whole pages, on the wafer: two launches a pair (the first
/// sets the routes; the boundary is the barrier the path needs).
pub fn copy_pages(rt: &mut Runtime, sym: &Symbols, pairs: &[(u32, u32)]) -> R<()> {
    for &(src, dst) in pairs {
        rt.launch(sym.copy_page, &[src, dst]).map_err(|e| e.to_string())?;
        rt.launch(sym.copy_move, &[src, dst]).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// A pool the host copies pages of (`KVCP` 0 builds export no copy
/// kernels): every layer grid's `kc`/`vc`, `NB` blocks of `cells` elements a
/// PE after one leading pad, block g / P of pool page g on grid row g % P.
pub struct HostPool {
    /// Each layer grid's top-left PE.
    pub grids: Vec<(usize, usize)>,
    /// Columns a grid row (PX); each holds its kv head's dims of a page.
    pub cols: usize,
    pub shape: Shape,
    /// Elements of one page on one PE (Dc · PS).
    pub cells: usize,
    /// 16-bit for bf16 KV (one element a u32), else 32-bit.
    pub data_type: DataType,
}

impl HostPool {
    /// Elements of `kc`/`vc` a PE.
    fn len(&self) -> usize {
        self.shape.blocks * self.cells + 1
    }
}

/// Block `bs` of every column of `src` into block `bd` of `dst` (rows of
/// `len` elements a PE, one leading pad).
pub fn patch_block(src: &[u32], dst: &mut [u32], len: usize, cells: usize, bs: usize, bd: usize) {
    for (a, d) in src.chunks_exact(len).zip(dst.chunks_exact_mut(len)) {
        d[1 + bd * cells..1 + (bd + 1) * cells].copy_from_slice(&a[1 + bs * cells..1 + (bs + 1) * cells]);
    }
}

/// `copy_kv`'s whole pages by the host, pair by pair in order: for each
/// grid and each of `kc`, `vc`, read the source row's array (and the
/// destination's, on another row), patch the block, write the destination
/// row back (memcpy takes no offset into a symbol). Only between launches:
/// a running launch appends to the same rows.
pub fn copy_pages_host(rt: &mut Runtime, pool: &HostPool, pairs: &[(u32, u32)]) -> R<()> {
    let (rows, len) = (pool.shape.rows, pool.len());
    let e = |e: crate::sdk::Error| e.to_string();
    for &(src, dst) in pairs {
        let (src, dst) = (src as usize, dst as usize);
        if src >= pool.shape.pages() || dst >= pool.shape.pages() {
            return Err(format!("page copy {src} -> {dst} outside a pool of {}", pool.shape.pages()));
        }
        if src == dst {
            continue;
        }
        let (rs, bs, rd, bd) = (src % rows, src / rows, dst % rows, dst / rows);
        for &(x0, y0) in &pool.grids {
            let row = |r: usize| Rect {
                x: x0 as i32,
                y: (y0 + r) as i32,
                w: pool.cols as i32,
                h: 1,
            };
            for sym in [SYMBOLS.kc, SYMBOLS.vc] {
                let mut a = vec![0u32; pool.cols * len];
                rt.memcpy_d2h(sym, &mut a, row(rs), len, Order::RowMajor, pool.data_type)
                    .map_err(e)?;
                let mut d = if rd == rs {
                    a.clone()
                } else {
                    let mut d = vec![0u32; pool.cols * len];
                    rt.memcpy_d2h(sym, &mut d, row(rd), len, Order::RowMajor, pool.data_type)
                        .map_err(e)?;
                    d
                };
                patch_block(&a, &mut d, len, pool.cells, bs, bd);
                rt.memcpy_h2d(sym, &d, row(rd), len, Order::RowMajor, pool.data_type)
                    .map_err(e)?;
            }
        }
    }
    Ok(())
}

/// Words of one page's host copy: every grid's `kc` then `vc`, each
/// column's `cells` of it.
pub fn page_words(pool: &HostPool) -> usize {
    pool.grids.len() * 2 * pool.cols * pool.cells
}

/// Block `b` of every column of a row (`len` elements a PE, one leading
/// pad), the columns' cells back to back.
pub fn take_block(row: &[u32], len: usize, cells: usize, b: usize) -> Vec<u32> {
    row.chunks_exact(len).flat_map(|c| c[1 + b * cells..1 + (b + 1) * cells].iter().copied()).collect()
}

/// [`take_block`]'s words back into block `b` of every column of `row`.
pub fn put_block(row: &mut [u32], len: usize, cells: usize, b: usize, words: &[u32]) {
    for (c, w) in row.chunks_exact_mut(len).zip(words.chunks_exact(cells)) {
        c[1 + b * cells..1 + (b + 1) * cells].copy_from_slice(w);
    }
}

/// Page `page`'s cells off the device ([`page_words`] of them, as the
/// pool's memcpy gives them): what the host keeps while its lane is
/// suspended. Only between launches.
pub fn read_page_host(rt: &mut Runtime, pool: &HostPool, page: u32) -> R<Vec<u32>> {
    let (rows, len) = (pool.shape.rows, pool.len());
    let page = page as usize;
    if page >= pool.shape.pages() {
        return Err(format!("page {page} outside a pool of {}", pool.shape.pages()));
    }
    let (r, b) = (page % rows, page / rows);
    let mut out = Vec::with_capacity(page_words(pool));
    for &(x0, y0) in &pool.grids {
        let rect = Rect { x: x0 as i32, y: (y0 + r) as i32, w: pool.cols as i32, h: 1 };
        for sym in [SYMBOLS.kc, SYMBOLS.vc] {
            let mut row = vec![0u32; pool.cols * len];
            rt.memcpy_d2h(sym, &mut row, rect, len, Order::RowMajor, pool.data_type)
                .map_err(|e| e.to_string())?;
            out.extend(take_block(&row, len, pool.cells, b));
        }
    }
    Ok(out)
}

/// [`read_page_host`]'s words written into page `page` (any page: a
/// resumed lane's pages come back where the pool has room).
pub fn write_page_host(rt: &mut Runtime, pool: &HostPool, page: u32, words: &[u32]) -> R<()> {
    let (rows, len) = (pool.shape.rows, pool.len());
    let page = page as usize;
    if page >= pool.shape.pages() {
        return Err(format!("page {page} outside a pool of {}", pool.shape.pages()));
    }
    if words.len() != page_words(pool) {
        return Err(format!("a page of {} words for pages of {}", words.len(), page_words(pool)));
    }
    let (r, b) = (page % rows, page / rows);
    let part = pool.cols * pool.cells;
    let mut parts = words.chunks_exact(part);
    for &(x0, y0) in &pool.grids {
        let rect = Rect { x: x0 as i32, y: (y0 + r) as i32, w: pool.cols as i32, h: 1 };
        for sym in [SYMBOLS.kc, SYMBOLS.vc] {
            let mut row = vec![0u32; pool.cols * len];
            rt.memcpy_d2h(sym, &mut row, rect, len, Order::RowMajor, pool.data_type)
                .map_err(|e| e.to_string())?;
            put_block(&mut row, len, pool.cells, b, parts.next().ok_or("short page")?);
            rt.memcpy_h2d(sym, &row, rect, len, Order::RowMajor, pool.data_type)
                .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: Shape = Shape {
        lanes: 4,
        rows: 4,
        page_size: 4,
        blocks: 4,
        entries: 2,
    };

    #[test]
    fn a_page_s_block_round_trips_word_for_word() {
        // Three columns of 2 blocks of 4 cells (one pad each): block 1 out of
        // one row, into block 0 of another, and back out the same.
        let (len, cells) = (1 + 2 * 4, 4);
        let row: Vec<u32> = (0..3 * len as u32).map(|i| 0x1000 + i).collect();
        let words = take_block(&row, len, cells, 1);
        assert_eq!(words.len(), 12);
        assert_eq!(&words[..4], &row[5..9]);
        let mut other = vec![0u32; 3 * len];
        put_block(&mut other, len, cells, 0, &words);
        assert_eq!(take_block(&other, len, cells, 0), words);
        // Nothing else moved: pads and block 1 stay zero.
        assert!(other.chunks_exact(len).all(|c| c[0] == 0 && c[5..].iter().all(|&w| w == 0)));
    }

    fn row(tab: &[u32], r: usize) -> &[u32] {
        let w = SHAPE.row_words();
        &tab[r * w..(r + 1) * w]
    }

    fn lane(pos: u32, pages: Vec<u32>) -> Lane {
        Lane {
            pos,
            slot: pos,
            pages,
            visible: None,
        }
    }

    #[test]
    fn pages_land_on_their_rows_in_logical_order() {
        // Lane 0 at slot 9 (pages 0..=2 used): pool pages 5, 2, 9.
        let tab = table(&SHAPE, &[lane(9, vec![5, 2, 9, 7])], 1).unwrap();
        let r1 = row(&tab, 1);
        assert_eq!(&r1[..4], &[9, 9, 9, 9]); // positions
        assert_eq!(&r1[4..8], &[9, 9, 9, 9]); // write slots
        // Page 5: row 1, block 1, logical 0; page 9: row 1, block 2, logical 2.
        assert_eq!(&r1[8..10], &[1, 2]);
        assert_eq!(&r1[16..18], &[0, 2]);
        // Logical 0 sees slots 0..4; logical 2 sees slots 8, 9 (bits 0, 1).
        assert_eq!(&r1[24..26], &[0b1111, 0b11]);
        // Page 2: row 2, block 0, logical 1, all four slots.
        let r2 = row(&tab, 2);
        assert_eq!(&r2[8..10], &[0, NONE]);
        assert_eq!(&r2[16..18], &[1, NONE]);
        assert_eq!(&r2[24..26], &[0b1111, NONE]);
        // Page 7 is past the slot: not in the table.
        assert!(row(&tab, 3)[8..].iter().all(|&w| w == NONE));
    }

    #[test]
    fn a_beam_rotates_apart_from_its_slot_and_sees_its_own_rows() {
        // Prompt slots 0..3, then this beam's earlier row at slot 5 and its
        // own at slot 7 (siblings at 4 and 6 hidden); RoPE position 5.
        let beam = Lane {
            pos: 5,
            slot: 7,
            pages: vec![0, 1],
            visible: Some(vec![0, 1, 2, 3, 5, 7]),
        };
        let tab = table(&SHAPE, &[beam], 1).unwrap();
        assert_eq!(row(&tab, 0)[0], 5);
        assert_eq!(row(&tab, 0)[4], 7);
        assert_eq!(row(&tab, 0)[24], 0b1111); // page 0
        assert_eq!(row(&tab, 1)[24], 0b1010); // page 1: slots 5 and 7
    }

    #[test]
    fn a_launch_of_several_steps_names_the_pages_it_writes() {
        // Slot 3 for 2 steps writes slot 4: page 1 must be in the table.
        let tab = table(&SHAPE, &[lane(3, vec![0, 1])], 2).unwrap();
        assert_eq!(row(&tab, 1)[8], 0);
        assert!(table(&SHAPE, &[lane(3, vec![0])], 2).is_err());
    }

    #[test]
    fn unused_lanes_repeat_the_first() {
        let tab = table(&SHAPE, &[lane(2, vec![3])], 1).unwrap();
        let r3 = row(&tab, 3);
        assert_eq!(&r3[..4], &[2, 2, 2, 2]);
        for b in 0..4 {
            assert_eq!(r3[8 + b * 2], 0);
            assert_eq!(r3[16 + b * 2], 0);
        }
    }

    #[test]
    fn too_many_pages_on_a_row_are_refused() {
        // Pages 0, 4, 8 all live on row 0; E = 2.
        let err = table(&SHAPE, &[lane(11, vec![0, 4, 8])], 1).unwrap_err();
        assert!(err.contains("more than 2"), "{err}");
    }

    #[test]
    fn a_lane_short_of_pages_is_refused() {
        assert!(table(&SHAPE, &[lane(4, vec![1])], 1).is_err());
    }

    #[test]
    fn patch_moves_one_block_of_every_column() {
        // Two columns, NB 3 blocks of 2 cells after a pad: block 2 of the
        // source into block 0 of the destination.
        let len = 7;
        let src: Vec<u32> = (0..2 * len as u32).collect();
        let mut dst = vec![100u32; 2 * len];
        patch_block(&src, &mut dst, len, 2, 2, 0);
        assert_eq!(dst, [100, 5, 6, 100, 100, 100, 100, 100, 12, 13, 100, 100, 100, 100]);
    }
}
