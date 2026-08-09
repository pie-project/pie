//! Who owns which page of the cache, across fires.
//!
//! [`Frame::of`] refuses two requests in ONE fire that name the same page,
//! and that check is worth having, but it is the smaller half of the problem.
//! A fire is a moment; a request is a conversation. The page a request wrote
//! its history into in fire 7 must still be its own in fire 800, and nothing
//! in a plan, a lowering or a frame says so -- every test in this crate up to
//! here invented page numbers by hand, and would have kept passing if two
//! conversations had been handed the same page in successive fires.
//!
//! That failure is silent in exactly the way [`Unstageable::SharedPage`]
//! describes and `Frame::of` cannot see: both conversations append to the
//! page, each reads the other's tokens back as its own history, and no
//! shader, no barrier and no validation layer notices. The only place it can
//! be prevented is a book that outlives the fire.
//!
//! # What this is not
//!
//! Not an eviction policy. This refuses when it is out of pages rather than
//! choosing a victim, because choosing one is a scheduling decision -- which
//! conversation is worth less -- and a driver is the wrong place to make it.
//! [`Book::spare`] is here so the layer that DOES make it can.
//!
//! Not a store of tokens either. A page's bytes are the pool's; this holds
//! only the numbers.
//!
//! [`Frame::of`]: crate::resources::Frame::of
//! [`Unstageable::SharedPage`]: crate::resources::Unstageable::SharedPage

use std::collections::BTreeMap;

use crate::resources::{Request, Shape};

/// Why a conversation could not be given what it asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unhoused {
    /// The cache has no page left.
    ///
    /// Carries what it needed and what was free rather than just saying no,
    /// because the caller's next move -- evict, queue, or refuse the user --
    /// depends on the gap and not on the fact.
    NoPages {
        /// Pages this growth would have needed.
        wanted: usize,
        /// Pages not held by anybody.
        spare: usize,
    },
}

impl std::fmt::Display for Unhoused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPages { wanted, spare } => {
                write!(
                    f,
                    "this growth needs {wanted} more pages and {spare} are free"
                )
            }
        }
    }
}

impl std::error::Error for Unhoused {}

/// One conversation's place in the cache.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Seat {
    /// Its pages, in the order it filled them. **Order is meaning, not
    /// bookkeeping**: `kv_page_indices` is read through `kv_page_indptr` at
    /// `position / page_size`, so a permuted list is a permuted history.
    pages: Vec<u32>,
    /// How many tokens it has appended. Its next position.
    tokens: usize,
}

/// Who owns which page, and how far each conversation has got.
///
/// Keyed by a caller-chosen id rather than by an index into a slot table, so
/// that admitting and dropping conversations does not renumber the ones that
/// stay -- the same reason [`Request::samples`] is per-request.
///
/// [`Request::samples`]: crate::resources::Request::samples
#[derive(Clone, Debug)]
pub struct Book {
    shape: Shape,
    /// Pages nobody holds. Handed out from the END, so a page just released
    /// is the next one given. That is a guess about locality and nothing
    /// depends on it; a page's bytes are meaningless to its new owner either
    /// way, because attention reads a request's own length and never past it.
    free: Vec<u32>,
    seats: BTreeMap<u64, Seat>,
}

impl Book {
    /// A book over every page a shape has.
    #[must_use]
    pub fn over(shape: Shape) -> Self {
        Self {
            shape,
            // Reversed so the first hand-out is page 0. A book that started
            // at the last page would work identically and read wrong in every
            // test that prints one.
            free: (0..shape.pages).rev().collect(),
            seats: BTreeMap::new(),
        }
    }

    /// Pages nobody holds.
    #[must_use]
    pub fn spare(&self) -> usize {
        self.free.len()
    }

    /// Conversations with a seat.
    #[must_use]
    pub fn seated(&self) -> usize {
        self.seats.len()
    }

    /// How many tokens `who` has appended, which is also its next position.
    #[must_use]
    pub fn tokens(&self, who: u64) -> Option<usize> {
        self.seats.get(&who).map(|s| s.tokens)
    }

    /// The pages `who` holds, in fill order.
    #[must_use]
    pub fn pages(&self, who: u64) -> Option<&[u32]> {
        self.seats.get(&who).map(|s| s.pages.as_slice())
    }

    /// Give `who` room for `tokens` more, and say what to fire.
    ///
    /// The positions and the pages are returned TOGETHER, as one [`Request`],
    /// because the defect they prevent is that they disagree:
    /// [`Unstageable::PastItsPages`] is a position whose page is past the end
    /// of its own list, and every caller in this crate that built the two by
    /// hand could produce it. Built here they cannot disagree, because the
    /// page count is computed from the last position.
    ///
    /// Seats a conversation that has none. Admission and growth are the same
    /// operation on purpose: a first fire is a growth from zero tokens, and a
    /// separate `admit` would be a second place to get the arithmetic wrong.
    ///
    /// # Errors
    ///
    /// [`Unhoused::NoPages`]. **Nothing is taken when it refuses** -- a
    /// partially grown conversation would hold pages it could not use and the
    /// caller would have no way to learn how many.
    ///
    /// [`Unstageable::PastItsPages`]: crate::resources::Unstageable::PastItsPages
    pub fn grow(&mut self, who: u64, tokens: usize) -> Result<Request, Unhoused> {
        let seat = self.seats.entry(who).or_default();
        let first = seat.tokens;
        let after = first + tokens;
        let page_size = self.shape.page_size as usize;
        // The page the LAST token lands on, so a growth that exactly fills a
        // page does not take the next one. `after` is a count and `after - 1`
        // is that token's index; a growth of zero tokens needs nothing.
        let need = if after == 0 {
            0
        } else {
            (after - 1) / page_size + 1
        };
        let more = need.saturating_sub(seat.pages.len());
        if more > self.free.len() {
            let spare = self.free.len();
            // Undo the seat this call created, so a refused first growth
            // leaves no empty conversation behind for `seated` to count.
            if first == 0 && seat.pages.is_empty() {
                self.seats.remove(&who);
            }
            return Err(Unhoused::NoPages {
                wanted: more,
                spare,
            });
        }
        for _ in 0..more {
            let page = self.free.pop().expect("checked against free above");
            seat.pages.push(page);
        }
        seat.tokens = after;
        Ok(Request::of(
            (first..after)
                .map(|p| u32::try_from(p).unwrap_or(u32::MAX))
                .collect(),
            seat.pages.clone(),
        ))
    }

    /// Take `who`'s pages back.
    ///
    /// Returns how many were freed, and zero for a conversation that never
    /// had a seat -- releasing twice is not an error, because the caller that
    /// drops a conversation and the caller that reaps it are usually not the
    /// same one.
    pub fn release(&mut self, who: u64) -> usize {
        let Some(seat) = self.seats.remove(&who) else {
            return 0;
        };
        let n = seat.pages.len();
        // Released in reverse so that a conversation of pages [4, 5, 6] hands
        // 4 back last and gets 4 first if it is immediately re-seated.
        self.free.extend(seat.pages.into_iter().rev());
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(pages: u32, page_size: u32) -> Shape {
        Shape {
            layers: 1,
            kv_heads: 1,
            head_dim: 8,
            page_size,
            pages,
            bytes: 2,
        }
    }

    /// A page a conversation was given is still its own after another
    /// conversation has been seated and grown.
    ///
    /// The whole reason this module exists. Handled by hand -- which is what
    /// every earlier test did -- the second conversation starts at page 0
    /// again and the first one's history becomes the second one's.
    #[test]
    fn a_second_conversation_is_never_given_a_page_the_first_still_holds() {
        let mut book = Book::over(shape(8, 4));
        let a = book.grow(1, 6).expect("room for six");
        let b = book.grow(2, 9).expect("room for nine");
        let held: Vec<u32> = a.pages.iter().chain(&b.pages).copied().collect();
        let mut sorted = held.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), held.len(), "{held:?} names a page twice");
        assert_eq!(a.pages, vec![0, 1]);
        assert_eq!(b.pages, vec![2, 3, 4]);
        assert_eq!(book.spare(), 3);
    }

    /// Growing an existing conversation keeps the pages it already filled, in
    /// the order it filled them.
    ///
    /// Order is meaning: `kv_page_indices` is read at `position / page_size`,
    /// so a book that appended a new page at the FRONT, or that rebuilt the
    /// list from a set, would move every token that had already been written.
    #[test]
    fn a_growth_appends_pages_and_never_reorders_the_ones_already_filled() {
        let mut book = Book::over(shape(8, 4));
        let first = book.grow(1, 4).expect("room");
        assert_eq!(first.pages, vec![0]);
        assert_eq!(first.positions, vec![0, 1, 2, 3]);
        let second = book.grow(1, 3).expect("room");
        assert_eq!(second.pages, vec![0, 1], "the filled page is still first");
        assert_eq!(second.positions, vec![4, 5, 6], "positions continue");
        assert_eq!(book.tokens(1), Some(7));
    }

    /// A growth that exactly fills a page does not take the next one.
    ///
    /// The off-by-one this arithmetic invites. A book computing `after /
    /// page_size + 1` takes a page for a conversation that has no token to
    /// put in it, which is not wrong so much as a slow leak: one wasted page
    /// per conversation, at every page boundary.
    #[test]
    fn a_conversation_that_exactly_fills_a_page_does_not_hold_an_empty_one() {
        let mut book = Book::over(shape(8, 4));
        assert_eq!(book.grow(1, 4).expect("room").pages.len(), 1);
        assert_eq!(book.grow(2, 8).expect("room").pages.len(), 2);
        assert_eq!(book.spare(), 5);
        // And one token past the boundary does take it.
        assert_eq!(book.grow(1, 1).expect("room").pages.len(), 2);
        assert_eq!(book.spare(), 4);
    }

    /// A refused growth takes nothing.
    ///
    /// A book that handed out what it had and then refused would leave a
    /// conversation holding pages for tokens it was never told it could
    /// write, and the caller -- which only sees the error -- would have no
    /// way to learn how many.
    #[test]
    fn a_refusal_leaves_the_book_exactly_as_it_was() {
        let mut book = Book::over(shape(4, 4));
        book.grow(1, 5).expect("two pages");
        let before = book.clone();
        let err = book.grow(2, 12).expect_err("only two pages are free");
        assert_eq!(
            err,
            Unhoused::NoPages {
                wanted: 3,
                spare: 2
            }
        );
        assert_eq!(book.spare(), before.spare());
        assert_eq!(book.seated(), 1, "the refused conversation kept no seat");
        assert_eq!(book.pages(1), before.pages(1));
        assert!(book.pages(2).is_none());
    }

    /// Released pages come back and can be given to somebody else.
    #[test]
    fn releasing_a_conversation_returns_every_page_it_held() {
        let mut book = Book::over(shape(4, 4));
        book.grow(1, 5).expect("two pages");
        assert_eq!(book.spare(), 2);
        assert_eq!(book.release(1), 2);
        assert_eq!(book.spare(), 4);
        assert_eq!(book.seated(), 0);
        assert!(book.tokens(1).is_none());
        // Twice is not an error: the caller that drops a conversation and the
        // one that reaps it are usually not the same.
        assert_eq!(book.release(1), 0);
        let after = book.grow(2, 16).expect("the whole cache");
        assert_eq!(after.pages.len(), 4);
    }

    /// What a book hands out stages.
    ///
    /// The point of returning a whole [`Request`]: the positions and the
    /// pages cannot disagree, so `Frame::of` -- which refuses exactly that
    /// disagreement -- has nothing to refuse.
    #[test]
    fn every_request_a_book_builds_stages_without_refusal() {
        use crate::resources::Frame;

        let s = shape(16, 4);
        let mut book = Book::over(s);
        // Lengths chosen to straddle page boundaries in both directions.
        let requests: Vec<Request> = [(1u64, 1usize), (2, 4), (3, 5), (7, 9)]
            .into_iter()
            .map(|(who, n)| book.grow(who, n).expect("room"))
            .collect();
        Frame::of(s, &requests).expect("a book's requests stage");
        // And again after every one has grown, which is the fire a
        // hand-written test never reaches.
        let next: Vec<Request> = [1u64, 2, 3, 7]
            .into_iter()
            .map(|who| book.grow(who, 1).expect("room"))
            .collect();
        Frame::of(s, &next).expect("the second fire stages too");
    }
}
