//! A keep: how a fold sorts a set's buffer.
//!
//! Each buffer column of a set's extent (column `c` is slot `folded + c`,
//! the buffer a fire found and then its own rows) is committed, left
//! buffered or dropped. The committed columns, in order, become the set's
//! committed rows at `folded ..`; the buffered ones follow them, in order;
//! the dropped ones are released. Committed rows below `folded` never move.
//! A fold of `n` slots is the keep that commits the first `n` columns and
//! buffers the rest.
//!
//! On a model with recurrent state the buffered columns are a suffix of the
//! extent: the recurrence has read every row it passed, dropped ones
//! included, so the record folds through the column before the suffix and
//! the next fire replays the suffix alone.

use ::engine::transfer::KvMove;

/// What a fold does with one buffer column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Commit,
    Buffer,
    Drop,
}

impl Outcome {
    /// The column's outcome as a guest states it: 0 commit, 1 buffer,
    /// 2 drop.
    pub fn of(byte: u8) -> Result<Outcome, KeepError> {
        match byte {
            0 => Ok(Outcome::Commit),
            1 => Ok(Outcome::Buffer),
            2 => Ok(Outcome::Drop),
            other => Err(KeepError::Outcome(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KeepError {
    #[error("a keep column states {0}; a column is 0 (commit), 1 (buffer) or 2 (drop)")]
    Outcome(u8),
    #[error(
        "the keep states {stated} column(s) and the set's extent past its committed \
         rows is {extent}: a keep states every buffer column"
    )]
    Extent { stated: u64, extent: u64 },
    #[error(
        "column {at} is buffered and column {after} after it is committed or dropped: on a \
         model with recurrent state the buffered columns are a suffix, since the recurrence \
         read every row it passed"
    )]
    Suffix { at: u64, after: u64 },
}

/// A keep over a set whose committed rows end at `folded`, as slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeepPlan {
    pub folded: u64,
    pub commits: u64,
    pub buffered: u64,
    pub dropped: u64,
    /// The columns the recurrence reads into the record: every column
    /// before the buffered suffix. A count, so the fold the engine runs is
    /// today's fold length.
    pub recurrent_fold: u64,
    /// Each kept column's slot, committed ones first, then buffered ones:
    /// the slot that lands at `folded + i` is `kept[i]`.
    pub kept: Vec<u64>,
}

impl KeepPlan {
    /// The set's committed length after the keep.
    #[must_use]
    pub fn folded_after(&self) -> u64 {
        self.folded + self.commits
    }

    /// One past the set's last row after the keep.
    #[must_use]
    pub fn end_after(&self) -> u64 {
        self.folded + self.commits + self.buffered
    }

    /// Whether no row moves: the committed columns are a prefix, the
    /// buffered ones the rest, and nothing is dropped. A plain fold.
    #[must_use]
    pub fn in_place(&self) -> bool {
        self.dropped == 0
            && self
                .kept
                .iter()
                .enumerate()
                .all(|(at, &slot)| slot == self.folded + at as u64)
    }

    /// The page the first moved row lands in: the boundary page, which also
    /// holds committed rows below `folded` where `folded` is mid-page.
    #[must_use]
    pub fn first_page(&self, page: u32) -> u64 {
        self.folded / u64::from(page.max(1))
    }

    /// The pages the rows from `first_page` on take after the keep.
    #[must_use]
    pub fn pages_after(&self, page: u32) -> u64 {
        let page = u64::from(page.max(1));
        self.end_after().div_ceil(page) - self.folded / page
    }

    /// Each slot from the boundary page's first on, after the keep, and the
    /// slot it comes from: the boundary page's committed head first (in
    /// place, as slots), then the kept columns.
    pub fn sources(&self, page: u32) -> impl Iterator<Item = (u64, u64)> + '_ {
        let start = self.first_page(page) * u64::from(page.max(1));
        let head = (start..self.folded).map(|slot| (slot, slot));
        let kept = self
            .kept
            .iter()
            .enumerate()
            .map(|(at, &slot)| (self.folded + at as u64, slot));
        head.chain(kept)
    }

    /// The token moves that land the rows from the boundary page on in
    /// `fresh` pages: `pages[i]` is the set's page `i` before the keep,
    /// `fresh[j]` its page `first_page + j` after. Every row moves, so no
    /// move reads a page another writes.
    #[must_use]
    pub fn moves(&self, page: u32, pages: &[u32], fresh: &[u32]) -> Vec<KvMove> {
        let size = u64::from(page.max(1));
        let first = self.first_page(page);
        self.sources(page)
            .map(|(dst, src)| KvMove {
                dst_page_id: fresh[(dst / size - first) as usize],
                dst_token_offset: (dst % size) as u32,
                src_page_id: pages[(src / size) as usize],
                src_token_offset: (src % size) as u32,
            })
            .collect()
    }
}

/// The keep `outcomes` states over a set whose committed rows end at
/// `folded` and whose rows end at `end`. `recurrent`: the model holds
/// recurrent state, and the buffered columns must be a suffix.
pub fn plan(
    folded: u64,
    end: u64,
    outcomes: &[Outcome],
    recurrent: bool,
) -> Result<KeepPlan, KeepError> {
    let extent = end.saturating_sub(folded);
    if outcomes.len() as u64 != extent {
        return Err(KeepError::Extent {
            stated: outcomes.len() as u64,
            extent,
        });
    }
    let mut commits = Vec::new();
    let mut buffered = Vec::new();
    let mut dropped = 0u64;
    let mut first_buffer: Option<u64> = None;
    for (at, outcome) in outcomes.iter().enumerate() {
        let at = at as u64;
        let slot = folded + at;
        match outcome {
            Outcome::Commit | Outcome::Drop => {
                if recurrent && let Some(buffered_at) = first_buffer {
                    return Err(KeepError::Suffix {
                        at: buffered_at,
                        after: at,
                    });
                }
                match outcome {
                    Outcome::Commit => commits.push(slot),
                    _ => dropped += 1,
                }
            }
            Outcome::Buffer => {
                first_buffer.get_or_insert(at);
                buffered.push(slot);
            }
        }
    }
    let recurrent_fold = first_buffer.unwrap_or(extent);
    let (commit_count, buffer_count) = (commits.len() as u64, buffered.len() as u64);
    commits.extend(buffered);
    Ok(KeepPlan {
        folded,
        commits: commit_count,
        buffered: buffer_count,
        dropped,
        recurrent_fold,
        kept: commits,
    })
}

/// The keep a fold of `count` slots is: the first `count` columns
/// committed, the rest buffered.
#[must_use]
pub fn fold_of(count: u64, extent: u64) -> Vec<Outcome> {
    (0..extent)
        .map(|at| {
            if at < count {
                Outcome::Commit
            } else {
                Outcome::Buffer
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use Outcome::{Buffer as B, Commit as C, Drop as D};

    #[test]
    fn a_fold_count_is_the_keep_that_commits_a_prefix_and_moves_nothing() {
        let plan = plan(10, 16, &fold_of(4, 6), true).unwrap();
        assert!(plan.in_place());
        assert_eq!((plan.commits, plan.buffered, plan.dropped), (4, 2, 0));
        assert_eq!(plan.folded_after(), 14);
        assert_eq!(plan.end_after(), 16);
        assert_eq!(plan.recurrent_fold, 4);
    }

    #[test]
    fn commits_land_first_then_buffered_rows_and_drops_are_released() {
        // Slots 8.. of a set folded at 8: commit 8, drop 9, buffer 10,
        // commit 11, drop 12. Dense: the buffer needn't be a suffix.
        let plan = plan(8, 13, &[C, D, B, C, D], false).unwrap();
        assert_eq!(plan.kept, vec![8, 11, 10]);
        assert_eq!((plan.commits, plan.buffered, plan.dropped), (2, 1, 2));
        assert_eq!(plan.folded_after(), 10);
        assert_eq!(plan.end_after(), 11);
        assert!(!plan.in_place());
    }

    #[test]
    fn a_recurrent_keep_buffers_a_suffix_and_folds_the_record_through_the_column_before_it() {
        let plan = plan(0, 6, &[C, D, C, D, B, B], true).unwrap();
        assert_eq!(plan.recurrent_fold, 4);
        assert_eq!(plan.kept, vec![0, 2, 4, 5]);
        let refused = super::plan(0, 4, &[C, B, D, B], true).unwrap_err();
        assert_eq!(refused, KeepError::Suffix { at: 1, after: 2 });
        // The same keep on a dense model is a keep.
        assert!(super::plan(0, 4, &[C, B, D, B], false).is_ok());
    }

    #[test]
    fn a_keep_states_every_buffer_column() {
        assert_eq!(
            plan(4, 10, &[C, C], false).unwrap_err(),
            KeepError::Extent {
                stated: 2,
                extent: 6
            }
        );
        assert_eq!(Outcome::of(3).unwrap_err(), KeepError::Outcome(3));
    }

    #[test]
    fn the_boundary_page_head_moves_with_the_kept_rows_into_fresh_pages() {
        // Pages of 4; folded at 6 (page 1 holds committed 4, 5); rows end
        // at 12. Keep: drop 6, 7, commit 8, buffer 9, drop 10, commit 11.
        let plan = plan(6, 12, &[D, D, C, B, D, C], false).unwrap();
        assert_eq!(plan.first_page(4), 1);
        assert_eq!(plan.end_after(), 9);
        assert_eq!(plan.pages_after(4), 2);
        let sources: Vec<_> = plan.sources(4).collect();
        assert_eq!(sources, vec![(4, 4), (5, 5), (6, 8), (7, 11), (8, 9)]);
        let moves = plan.moves(4, &[100, 101, 102], &[200, 201]);
        let at = |m: &KvMove| {
            (
                m.src_page_id,
                m.src_token_offset,
                m.dst_page_id,
                m.dst_token_offset,
            )
        };
        assert_eq!(
            moves.iter().map(at).collect::<Vec<_>>(),
            vec![
                (101, 0, 200, 0),
                (101, 1, 200, 1),
                (102, 0, 200, 2),
                (102, 3, 200, 3),
                (102, 1, 201, 0),
            ]
        );
    }
}
