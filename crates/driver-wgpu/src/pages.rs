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

use std::collections::{BTreeMap, BTreeSet};

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
    /// A shrink would drop a page somebody is sitting on.
    ///
    /// Names the conversation and the page rather than only the counts: the
    /// caller's next move is to evict that conversation or to pick a larger
    /// target, and neither is choosable from "it did not fit".
    Stranded {
        /// The conversation holding it.
        who: u64,
        /// The page it holds that the shrink would drop.
        page: u32,
        /// The target the shrink asked for.
        pages: u32,
    },
    /// A fork's source has no seat, or its destination already has one.
    ///
    /// Kept apart from [`Unhoused::NoPages`] because waiting does not fix it:
    /// no amount of eviction gives a conversation a history it never had.
    Unforkable(Unforkable),
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
            Self::Stranded { who, page, pages } => write!(
                f,
                "conversation {who} holds page {page}, which a cache of {pages} pages \
                 does not have"
            ),
            Self::Unforkable(why) => write!(f, "{why}"),
        }
    }
}

/// Why a fork could not be made.
///
/// An enum and not a message, so that [`Unhoused`] stays `Copy` and a caller
/// can match on the reason. Each is a different mistake by a different
/// caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unforkable {
    /// Source and destination are the same conversation.
    Itself,
    /// The destination already holds pages, and taking its seat would drop
    /// them without telling anybody.
    Taken,
    /// The source has no seat, so there is no history to copy.
    Absent,
}

impl std::fmt::Display for Unforkable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Itself => write!(f, "a conversation cannot be forked onto itself"),
            Self::Taken => write!(f, "the destination already holds pages"),
            Self::Absent => write!(f, "the source has no history to fork"),
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
    /// Which recurrent slot holds its gated-DeltaNet carry.
    ///
    /// # Why the book owns this and not the pool
    ///
    /// It is the same fact the pages are: WHERE this conversation's history
    /// lives. A linear-attention layer keeps its history in a state slab
    /// instead of a page table, and the two have to be seated together or a
    /// conversation reads its own keys beside somebody else's carry.
    ///
    /// Assigned as the lowest index no other seat holds, so a dropped
    /// conversation's slot is reused rather than the counter running away from
    /// a pool that has a fixed number of them. It is stable for the life of
    /// the seat, which is what makes the carry a carry.
    slot: u32,
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

    /// The recurrent seat `who` holds, or `None` if `who` has no seat here.
    ///
    /// The sibling of [`Self::pages`], for the other pool. A conversation's
    /// carry lives at one slot for its whole life ([`Self::grow`] seats it
    /// once and [`Self::fork`] copies it), so this is a property of the seat
    /// rather than of any one fire — which is what makes it readable from
    /// outside, and what a probe comparing two conversations' state needs in
    /// order to know which bytes are whose.
    #[must_use]
    pub fn slot(&self, who: u64) -> Option<u32> {
        self.seats.get(&who).map(|s| s.slot)
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
        // Before the seat is borrowed, because the answer depends on every
        // OTHER seat and `entry` holds the map.
        let slot = self.free_slot(who);
        let seat = self.seats.entry(who).or_default();
        seat.slot = slot;
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
        let slot = seat.slot;
        let mut request = Request::of(
            (first..after)
                .map(|p| u32::try_from(p).unwrap_or(u32::MAX))
                .collect(),
            seat.pages.clone(),
        );
        request.slot = slot;
        Ok(request)
    }

    /// The recurrent slot `who` already holds, or the lowest nobody does.
    ///
    /// Linear scan over the seats, which are few and are already a `BTreeMap`
    /// this crate walks per fire. A counter would be cheaper and would run
    /// away from a pool with a fixed slot count: a server that seats and drops
    /// conversations all day would hand out slot ten thousand while nine
    /// thousand of them stood empty.
    fn free_slot(&self, who: u64) -> u32 {
        if let Some(seat) = self.seats.get(&who) {
            return seat.slot;
        }
        let taken: BTreeSet<u32> = self.seats.values().map(|s| s.slot).collect();
        (0u32..).find(|s| !taken.contains(s)).unwrap_or(u32::MAX)
    }

    /// Seat `to` on fresh pages holding the same history as `from`.
    ///
    /// Returns the moves a caller must make, as `(source, destination)` page
    /// numbers in fill order: the BOOK owns who holds which page and the POOL
    /// owns what is in it, and neither can do the other's half. A caller that
    /// takes this list and does not perform it has seated a conversation on
    /// pages holding somebody else's history -- which is why the pairs come
    /// back as a value that has to be used rather than as a side effect.
    ///
    /// The destination's token count is the source's, so `to` continues where
    /// `from` is rather than starting empty on a full cache.
    ///
    /// # Errors
    ///
    /// [`Unhoused::Unforkable`] if `from` has no seat or `to` already has one
    /// -- overwriting a seated conversation would drop its pages without
    /// telling anybody. [`Unhoused::NoPages`] if the cache cannot hold a
    /// second copy, and **nothing is taken when it refuses**, as in
    /// [`Book::grow`].
    pub fn fork(&mut self, from: u64, to: u64) -> Result<Vec<(u32, u32)>, Unhoused> {
        if from == to {
            return Err(Unhoused::Unforkable(Unforkable::Itself));
        }
        if self.seats.contains_key(&to) {
            return Err(Unhoused::Unforkable(Unforkable::Taken));
        }
        let Some(seat) = self.seats.get(&from) else {
            return Err(Unhoused::Unforkable(Unforkable::Absent));
        };
        let wanted = seat.pages.len();
        if self.free.len() < wanted {
            return Err(Unhoused::NoPages {
                wanted,
                spare: self.free.len(),
            });
        }
        let tokens = seat.tokens;
        let sources = seat.pages.clone();
        let mut moves = Vec::with_capacity(wanted);
        let mut pages = Vec::with_capacity(wanted);
        for source in sources {
            // Taken the way `grow` takes them, so a fork and a growth cannot
            // disagree about which page is next.
            let page = self.free.pop().expect("checked above");
            moves.push((source, page));
            pages.push(page);
        }
        // A forked lane needs its OWN carry: the beam's whole point is that
        // the lanes diverge, and two lanes sharing a slot would each fold
        // their tokens into one state. The pages are copied above; the slot is
        // allocated fresh, and the carry it starts from is whatever the pool
        // holds there -- which is the same gap `copy_kv` fills for the pages
        // and which no caller has asked for yet.
        let slot = self.free_slot(to);
        self.seats.insert(to, Seat { pages, tokens, slot });
        Ok(moves)
    }

    /// Follow the pool to `pages`.
    ///
    /// Grows by adding the new page numbers to the free list, shrinks by
    /// dropping the free ones past the new end.
    ///
    /// # Errors
    ///
    /// [`Unhoused::Stranded`] if a SEATED conversation holds a page the
    /// shrink would drop, naming the conversation and the page. Nothing is
    /// changed when it refuses -- a book that had already dropped half its
    /// free list would leave the pool smaller than the book believes, and
    /// every fire after it would bind a page past the buffer, which this card
    /// answers with zeros rather than an error.
    ///
    /// Checked here rather than in the pool because the pool does not know
    /// who holds what. `Shell::resize_pool` calls this FIRST for exactly that
    /// reason: a refusal must arrive before any byte moves.
    pub fn resize(&mut self, pages: u32) -> Result<(), Unhoused> {
        if let Some((&who, page)) = self.seats.iter().find_map(|(who, seat)| {
            seat.pages
                .iter()
                .find(|p| **p >= pages)
                .map(|page| (who, *page))
        }) {
            return Err(Unhoused::Stranded { who, page, pages });
        }
        if pages > self.shape.pages {
            // Prepended, because the free list is taken from its END and the
            // low pages should still go out first: a growth that handed out
            // the new pages first would leave a pool whose used pages are its
            // last ones, and the next shrink would strand them.
            let mut fresh: Vec<u32> = (self.shape.pages..pages).rev().collect();
            fresh.append(&mut self.free);
            self.free = fresh;
        } else {
            self.free.retain(|p| *p < pages);
        }
        self.shape.pages = pages;
        Ok(())
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

    #[test]
    fn a_fork_takes_fresh_pages_and_keeps_the_source_where_it_was() {
        let mut book = Book::over(shape(8, 4));
        book.grow(1, 5).expect("a seat");
        let held = book.pages(1).expect("pages").to_vec();
        let spare = book.spare();

        let moves = book.fork(1, 2).expect("a fork");
        assert_eq!(moves.len(), held.len(), "one move per page");
        // The SOURCE keeps what it had. A fork that handed the source's pages
        // away would leave two conversations reading one cache.
        assert_eq!(book.pages(1).expect("pages"), held.as_slice());
        let taken: Vec<u32> = moves.iter().map(|(_, to)| *to).collect();
        assert_eq!(book.pages(2).expect("pages"), taken.as_slice());
        for page in &taken {
            assert!(!held.contains(page), "page {page} was handed out twice");
        }
        assert_eq!(book.tokens(2), book.tokens(1), "the history's length");
        assert_eq!(book.spare(), spare - held.len(), "pages left");
    }

    #[test]
    fn a_fork_that_cannot_fit_takes_nothing() {
        // Six pages: a five-token conversation at four per page needs two,
        // and a fork of it needs two more, leaving two. A second fork wants
        // two and there are two -- so the refusal has to be arranged with a
        // conversation that needs more than is left.
        let mut book = Book::over(shape(3, 4));
        book.grow(1, 9).expect("three pages");
        let spare = book.spare();
        assert_eq!(spare, 0, "the premise: nothing is free");

        let refused = book.fork(1, 2).expect_err("no room for a copy");
        assert!(
            matches!(
                refused,
                Unhoused::NoPages {
                    wanted: 3,
                    spare: 0
                }
            ),
            "{refused}"
        );
        // Nothing taken, and no half-seat left behind.
        assert!(book.pages(2).is_none(), "the refused fork seated anyway");
        assert_eq!(book.spare(), spare);
    }

    #[test]
    fn a_fork_will_not_overwrite_a_seat_or_invent_a_history() {
        let mut book = Book::over(shape(8, 4));
        book.grow(1, 5).expect("a seat");
        book.grow(2, 5).expect("a seat");
        let theirs = book.pages(2).expect("pages").to_vec();

        assert_eq!(
            book.fork(1, 2).expect_err("2 is seated"),
            Unhoused::Unforkable(Unforkable::Taken)
        );
        // ...and 2 still holds exactly what it did, rather than having been
        // silently dropped on the floor.
        assert_eq!(book.pages(2).expect("pages"), theirs.as_slice());

        assert_eq!(
            book.fork(9, 10).expect_err("9 has no history"),
            Unhoused::Unforkable(Unforkable::Absent)
        );
        assert_eq!(
            book.fork(1, 1).expect_err("onto itself"),
            Unhoused::Unforkable(Unforkable::Itself)
        );
    }

    #[test]
    fn a_growth_hands_out_the_pages_that_were_there_first() {
        let mut book = Book::over(shape(2, 4));
        book.grow(1, 5).expect("both pages");
        assert_eq!(book.spare(), 0, "the premise");

        book.resize(6).expect("room to grow into");
        assert_eq!(book.spare(), 4, "the new pages are free");
        // The conversation that was seated keeps exactly what it held.
        assert_eq!(book.pages(1).expect("pages"), &[0, 1]);
        // ...and the next hand-out is the LOWEST new page, not the highest.
        // A growth that handed out the top of the range first would leave the
        // used pages at the end, and the next shrink would strand them.
        book.grow(2, 1).expect("a page");
        assert_eq!(book.pages(2).expect("pages"), &[2]);
    }

    #[test]
    fn a_shrink_drops_only_free_pages_and_refuses_to_strand_a_seat() {
        let mut book = Book::over(shape(8, 4));
        book.grow(1, 9).expect("three pages");
        assert_eq!(book.pages(1).expect("pages"), &[0, 1, 2]);

        // Down to exactly what is held: allowed, and nothing is left free.
        book.resize(3).expect("a shrink to the high-water mark");
        assert_eq!(book.spare(), 0);
        assert_eq!(book.pages(1).expect("pages"), &[0, 1, 2]);

        // One page further is a refusal that NAMES what is in the way.
        let refused = book.resize(2).expect_err("page 2 is held");
        assert_eq!(
            refused,
            Unhoused::Stranded {
                who: 1,
                page: 2,
                pages: 2
            }
        );
        // ...and the refusal changed nothing, so the book still agrees with a
        // pool nobody resized.
        assert_eq!(book.spare(), 0);
        // A SECOND conversation, because 1 has three tokens of slack inside
        // the pages it already holds and would be seated without asking for
        // one.
        assert!(book.grow(2, 1).is_err(), "the cache is still full");
        book.resize(4).expect("room again");
        book.grow(2, 1).expect("a page to sit on");
    }

    /// **Two conversations do not share a recurrent slot, and one keeps its
    /// own across fires.**
    ///
    /// Both halves, because they fail in opposite directions and a table that
    /// got either one wrong would still look like it worked. A slot that moved
    /// between fires would lose a conversation's carry every step, and one
    /// shared between conversations would fold two histories into one state —
    /// and neither shows up in the output as anything but a fluent wrong
    /// answer, because a gated DeltaNet reads whatever is in the slab.
    #[test]
    fn a_seat_keeps_one_recurrent_slot_and_no_other_seat_has_it() {
        let mut book = Book::over(shape(16, 4));
        let a = book.grow(7, 3).expect("room for three");
        let b = book.grow(9, 3).expect("room for three");
        assert_ne!(a.slot, b.slot, "two conversations share a carry");

        let again = book.grow(7, 1).expect("room for one more");
        assert_eq!(a.slot, again.slot, "a conversation's carry moved under it");

        // The lowest FREE one, not the next one ever handed out: a pool has a
        // fixed number of slots, and a counter that only goes up would run off
        // the end of a slab that was mostly empty.
        book.release(7);
        let c = book.grow(11, 1).expect("room for one");
        assert_eq!(c.slot, a.slot, "the dropped seat's slot was not reused");
    }
}
