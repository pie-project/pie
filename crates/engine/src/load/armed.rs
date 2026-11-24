//! Which classes a load's lattice names and arms before ready, and the room
//! arming lends a scored fire's capture rows: one rule every engine that
//! records bodies follows: every body is captured before ready, and nothing
//! is built eagerly after it.
//!
//! A class is armed when a plain request lands in it, when a request plain
//! but for capturing scores does (a scored class: an eviction guest's
//! lanes), or when a request plain but for reading the drafts does and the
//! class writes a drafter's output (a drafter's verify). A scored fire's
//! body reads where its lanes' score rows lie through a table the fire
//! stages, so it is the same body wherever the rows lie; at arm time no
//! reader is bound, and arming lends the rows from room the load holds for
//! its synthetic fires, sized by the widest scored fire
//! ([`arming_capture_room`]). A scored class is armed at the lattice's
//! octaves alone ([`scored_buckets`]), beside every plain bucket: its fires
//! pad up to the next and replay there.

use poem_ir::{ClassSet, Request};

/// Whether only its query length and its capturing scores set `request`
/// apart.
#[must_use]
pub fn scores_only(request: &Request) -> bool {
    request.captures_scores() && request.departures() == usize::from(request.query_len() != 1) + 1
}

/// Whether only its query length and its reading the pass's drafts set
/// `request` apart.
#[must_use]
pub fn drafts_only(request: &Request) -> bool {
    request.drafts() && request.departures() == usize::from(request.query_len() != 1) + 1
}

/// The classes a lattice names and arms, of `landing` (each class's
/// requests): a plain request's, a scored one's ([`scores_only`]), and a
/// drafter's verify's ([`drafts_only`], where `drafter` says the class
/// writes an output only drafting classes write).
#[must_use]
pub fn lattice_classes(landing: &[Vec<Request>], drafter: impl Fn(usize) -> bool) -> ClassSet {
    ClassSet::of((0..landing.len()).filter(|&class| {
        let requests = &landing[class];
        requests.iter().any(Request::plain)
            || requests.iter().any(scores_only)
            || (requests.iter().any(drafts_only) && drafter(class))
    }))
}

/// The scored classes of `landing`: those a request plain but for capturing
/// scores lands in.
#[must_use]
pub fn scored_classes(landing: &[Vec<Request>]) -> ClassSet {
    ClassSet::of((0..landing.len()).filter(|&class| landing[class].iter().any(scores_only)))
}

/// The buckets a scored class is armed at, of a lattice's `buckets`
/// (ascending): its octaves, the power-of-two points, and its top where that
/// is none, so every fire the lattice holds has one. A structural subset of
/// the plain lattice, never a tuned count: a scored fire pads up to the next
/// of them ([`scored_bucket`]) and replays its body there.
#[must_use]
pub fn scored_buckets(buckets: &[u32]) -> Vec<u32> {
    let mut octaves: Vec<u32> = buckets
        .iter()
        .copied()
        .filter(|bucket| bucket.is_power_of_two())
        .collect();
    if let Some(&top) = buckets.last()
        && octaves.last() != Some(&top)
    {
        octaves.push(top);
    }
    octaves
}

/// The bucket a fire of `rows` rows holding a scored lane pads to: the
/// first of [`scored_buckets`] that holds it; none past the lattice's top.
#[must_use]
pub fn scored_bucket(buckets: &[u32], rows: u32) -> Option<u32> {
    scored_buckets(buckets)
        .into_iter()
        .find(|&bucket| bucket >= rows)
}

/// The bytes of a capturing lane's score rows: a key row of the load's
/// `context` and the page row of its pages of `page_size` right after it,
/// f32.
#[must_use]
pub fn capture_row_bytes(context: u32, page_size: u32) -> u64 {
    let pages = context.div_ceil(page_size.max(1));
    (u64::from(context) + u64::from(pages)) * size_of::<f32>() as u64
}

/// The room arming lends a scored fire's capture rows: a lane's rows for
/// each of the widest fire's `lanes`, where the lattice arms a scored class;
/// none where it arms none. The load holds it beside its synthetic fires'
/// state, counted before ready.
#[must_use]
pub fn arming_capture_room(scored: bool, lanes: u32, context: u32, page_size: u32) -> u64 {
    if scored {
        u64::from(lanes).saturating_mul(capture_row_bytes(context, page_size))
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scored_class_is_armed_beside_the_plain_ones() {
        let plain = Request::new(1, false);
        let scored = Request::new(1, false).capturing_scores(true);
        let both = Request::new(1, false).capturing_scores(true).adapted(true);
        let landing = vec![vec![plain], vec![scored], vec![both]];
        let armed = lattice_classes(&landing, |_| false);
        assert!(armed.contains(0) && armed.contains(1));
        assert!(
            !armed.contains(2),
            "a scored lane with an adapter is not plain but for scores"
        );
        assert_eq!(scored_classes(&landing), ClassSet::of([1usize]));
    }

    #[test]
    fn a_scored_fire_pads_to_the_lattices_next_octave() {
        let lattice = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96];
        assert_eq!(scored_buckets(&lattice), [1, 2, 4, 8, 16, 32, 64, 96]);
        assert_eq!(scored_bucket(&lattice, 5), Some(8));
        assert_eq!(scored_bucket(&lattice, 64), Some(64));
        assert_eq!(scored_bucket(&lattice, 65), Some(96));
        assert_eq!(scored_bucket(&lattice, 97), None);
        assert_eq!(scored_buckets(&[1, 2, 4, 256]), [1, 2, 4, 256]);
    }

    #[test]
    fn the_room_is_the_widest_scored_fires_rows() {
        assert_eq!(capture_row_bytes(256, 16), (256 + 16) * 4);
        assert_eq!(capture_row_bytes(250, 16), (250 + 16) * 4);
        assert_eq!(arming_capture_room(true, 64, 256, 16), 64 * 272 * 4);
        assert_eq!(arming_capture_room(false, 64, 256, 16), 0);
    }
}
