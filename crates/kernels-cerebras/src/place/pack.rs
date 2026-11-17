//! Packing blocks with no nets between them onto a bounded fabric: co-resident readings (each
//! a whole program, a rigid block). No price reads where such a block sits, so every placement
//! within the bounds costs the same; the problem's own placement is kept where it fits (its
//! program unchanged), else the blocks are packed in 2D, bottom-left, over their orders, the
//! smallest bounding box first.

use super::{Problem, Solution};
use crate::floor::{Coord, Rect};

type R<T> = Result<T, String>;

/// Orders tried at most (every order up to 7 blocks; past that, by height and by width).
const ORDERS: usize = 5040;

/// Whether `s` keeps every block within `w` x `h` and apart.
#[must_use]
pub fn fits(p: &Problem, s: &Solution, w: u32, h: u32) -> bool {
    let rs: Vec<Rect> = (0..p.blocks.len()).map(|i| s.rect(p, i)).collect();
    rs.iter().all(|r| r.x + r.w <= w && r.y + r.h <= h)
        && rs.iter().enumerate().all(|(i, a)| {
            rs[i + 1..].iter().all(|b| {
                !(a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h)
            })
        })
}

/// How far blocks that pack into no `w` x `h` fabric miss it: the nearest packing's overrun, in
/// columns with the height held (`dx`) or in rows with the width held (`dy`); the smaller is
/// the one stated, the other 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Miss {
    pub dx: u32,
    pub dy: u32,
}

impl std::fmt::Display for Miss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.dy > 0 {
            write!(f, "short {} rows", self.dy)
        } else {
            write!(f, "short {} columns", self.dx)
        }
    }
}

/// [`pack`], or how near it came ([`Miss`]): a cheap fit test a search asks of many candidates.
pub fn pack_within(p: &Problem, w: u32, h: u32) -> Result<Solution, Miss> {
    if let Ok(s) = pack(p, w, h) {
        return Ok(s);
    }
    // the nearest miss: the least overrun of the bounds' one side with the other held
    let orders = orders(p);
    let over = |bw: u32, bh: u32, horiz: bool| -> Option<u32> {
        orders
            .iter()
            .filter_map(|o| bottom_left(p, o, bw, bh))
            .map(|s| {
                if horiz {
                    s.w.saturating_sub(w)
                } else {
                    s.h.saturating_sub(h)
                }
            })
            .min()
    };
    let dy = over(w, u32::MAX / 2, false);
    let dx = over(u32::MAX / 2, h, true);
    Err(match (dx, dy) {
        (Some(dx), Some(dy)) if dy <= dx => Miss { dx: 0, dy },
        (Some(dx), _) => Miss { dx, dy: 0 },
        (None, Some(dy)) => Miss { dx: 0, dy },
        // (a block wider and taller than the bounds alike)
        (None, None) => Miss {
            dx: p
                .blocks
                .iter()
                .map(|b| b.rect.w.saturating_sub(w))
                .max()
                .unwrap_or(0),
            dy: p
                .blocks
                .iter()
                .map(|b| b.rect.h.saturating_sub(h))
                .max()
                .unwrap_or(0),
        },
    })
}

/// The orders the packing tries: every order up to 7 blocks; past that, by height and by width.
fn orders(p: &Problem) -> Vec<Vec<usize>> {
    let n = p.blocks.len();
    let mut orders: Vec<Vec<usize>> = Vec::new();
    if n <= 7 {
        let mut a: Vec<usize> = (0..n).collect();
        permutations(&mut a, 0, &mut orders);
    } else {
        let mut by_h: Vec<usize> = (0..n).collect();
        by_h.sort_by_key(|&i| std::cmp::Reverse(p.blocks[i].rect.h));
        let mut by_w: Vec<usize> = (0..n).collect();
        by_w.sort_by_key(|&i| std::cmp::Reverse(p.blocks[i].rect.w));
        orders.extend([by_h, by_w]);
    }
    orders.truncate(ORDERS);
    orders
}

/// `p`'s blocks within `w` x `h`: its own placement where it fits, else the packing with the
/// smallest bounding box (then the least height). The solution's fabric is the bounding box.
pub fn pack(p: &Problem, w: u32, h: u32) -> R<Solution> {
    let pinned = Solution::pinned(p);
    if fits(p, &pinned, w, h) {
        return Ok(pinned);
    }
    let orders = orders(p);
    let mut best: Option<(u64, u32, Solution)> = None;
    for o in &orders {
        let Some(s) = bottom_left(p, o, w, h) else {
            continue;
        };
        let area = u64::from(s.w) * u64::from(s.h);
        if best
            .as_ref()
            .is_none_or(|(a, bh, _)| (area, s.h) < (*a, *bh))
        {
            best = Some((area, s.h, s));
        }
    }
    best.map(|(_, _, s)| s).ok_or_else(|| {
        format!(
            "place: blocks {:?} pack into no {w} x {h} fabric",
            p.blocks
                .iter()
                .map(|b| (b.rect.w, b.rect.h))
                .collect::<Vec<_>>()
        )
    })
}

fn permutations(a: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if out.len() >= ORDERS {
        return;
    }
    if k == a.len() {
        out.push(a.clone());
        return;
    }
    for i in k..a.len() {
        a.swap(k, i);
        permutations(a, k + 1, out);
        a.swap(k, i);
    }
}

/// The blocks in `order`, each at the lowest row, then the westmost column, where it fits
/// beside the ones before it (candidate corners: the origin and each placed block's east and
/// south edges).
fn bottom_left(p: &Problem, order: &[usize], w: u32, h: u32) -> Option<Solution> {
    let mut at: Vec<Option<Rect>> = vec![None; p.blocks.len()];
    for &i in order {
        let (bw, bh) = (p.blocks[i].rect.w, p.blocks[i].rect.h);
        let placed: Vec<Rect> = at.iter().flatten().copied().collect();
        let mut xs = vec![0u32];
        let mut ys = vec![0u32];
        for r in &placed {
            xs.push(r.x + r.w);
            ys.push(r.y + r.h);
        }
        let mut cands: Vec<(u32, u32)> = ys
            .iter()
            .flat_map(|&y| xs.iter().map(move |&x| (y, x)))
            .collect();
        cands.sort_unstable();
        let hit = cands.into_iter().find(|&(y, x)| {
            x + bw <= w
                && y + bh <= h
                && placed
                    .iter()
                    .all(|r| !(x < r.x + r.w && r.x < x + bw && y < r.y + r.h && r.y < y + bh))
        })?;
        at[i] = Some(Rect::new(hit.1, hit.0, bw, bh));
    }
    let rs: Vec<Rect> = at
        .into_iter()
        .map(|r| r.expect("every block placed"))
        .collect();
    Some(Solution {
        w: rs.iter().map(|r| r.x + r.w).max().unwrap_or(0),
        h: rs.iter().map(|r| r.y + r.h).max().unwrap_or(0),
        at: rs.iter().map(|r| Coord::new(r.x, r.y)).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::{BlockKind, Problem};
    use crate::floor::Rect;
    use crate::wiring::Wiring;

    fn problem(sizes: &[(u32, u32)]) -> Problem {
        let mut x = 0;
        let rects = sizes
            .iter()
            .enumerate()
            .map(|(i, &(w, h))| {
                let r = Rect::new(x, 0, w, h);
                x += w;
                (BlockKind::Reading(i), r)
            })
            .collect();
        Problem::of(
            x,
            sizes.iter().map(|s| s.1).max().unwrap_or(0),
            rects,
            &Wiring::default(),
            &|_| super::super::NetKind::Other,
        )
        .unwrap()
    }

    /// Side by side where they fit (the problem's own placement); packed in 2D where they do
    /// not, within the bounds, apart, in the smallest box.
    #[test]
    fn readings_pack_side_by_side_else_in_2d() {
        let p = problem(&[(300, 400), (200, 300)]);
        let s = super::pack(&p, 762, 1172).unwrap();
        assert_eq!(s, super::Solution::pinned(&p));
        let p = problem(&[(500, 300), (200, 500), (70, 70)]);
        let s = super::pack(&p, 754, 1170).unwrap();
        assert!(super::fits(&p, &s, 754, 1170));
        assert!(s.w <= 754, "{s:?}");
        let p = problem(&[(700, 600), (700, 600)]);
        let s = super::pack(&p, 754, 1170);
        assert!(s.is_err());
        // the nearest miss: stacked, 30 rows over (side by side would be 646 columns over)
        assert_eq!(
            super::pack_within(&p, 754, 1170).unwrap_err(),
            super::Miss { dx: 0, dy: 30 }
        );
        assert_eq!(
            super::pack_within(&p, 754, 1170).unwrap_err().to_string(),
            "short 30 rows"
        );
        // klein's readings side by side at 770 columns pack within the system's
        let p = problem(&[(256, 877), (384, 1014), (130, 148)]);
        let s = super::pack_within(&p, 755, 1170).unwrap();
        assert_eq!((s.w, s.h), (640, 1025));
        let p = problem(&[(256, 1100), (384, 1100), (130, 148)]);
        assert_eq!(
            super::pack_within(&p, 755, 1170).unwrap_err(),
            super::Miss { dx: 15, dy: 0 }
        );
    }
}
