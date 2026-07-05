//! Binary Run-Length Encoded (BRLE) boolean sequences.
//!
//! `Brle` is part of the wire schema (it appears inside
//! [`crate::schema::ForwardRequest`] via `Vec<Brle>`) AND is the type
//! the runtime/drivers manipulate directly. Putting the type and its
//! operations here keeps the schema single-source: no duplicate `Brle`
//! types, no conversion at the wire boundary.
//!
//! ## Encoding
//! - `[false, false, true, true, true, false]` → `[2, 3, 1]`
//! - `[true, true, false]` → `[0, 2, 1]` (zero-length false prefix)
//!
//! The starts-with-false invariant: a buffer that represents a sequence
//! beginning with `true` always has a leading `0` (zero-length false
//! run) so that even buffer indices map to false runs and odd indices
//! to true runs.

use std::collections::BTreeSet;
use std::iter::FusedIterator;

use pie_driver_abi_derive::schema;

/// A Binary Run-Length Encoding (BRLE) structure.
///
/// `total_size` is `u64` rather than `usize` because the wire schema
/// requires fixed width. The accessor methods take/return `usize` for
/// convenience and cast at the boundary.
#[derive(Default, PartialEq, Eq, Hash)]
#[schema]
pub struct Brle {
    /// The buffer of run lengths. Even indices = false-run lengths,
    /// odd indices = true-run lengths.
    pub buffer: Vec<u32>,
    /// Total boolean count this BRLE represents.
    pub total_size: u64,
}

// Public API
impl Brle {
    /// Creates a new `Brle` instance representing `size` `false` values.
    pub fn new(size: usize) -> Self {
        if size == 0 {
            Self {
                buffer: vec![],
                total_size: 0,
            }
        } else {
            Self {
                buffer: vec![size as u32],
                total_size: size as u64,
            }
        }
    }

    /// Creates a new `Brle` instance representing `size` `true` values.
    /// The starts-with-False convention requires a zero-length false-run
    /// prefix, so the buffer is `[0, size]`.
    pub fn all_true(size: usize) -> Self {
        if size == 0 {
            Self {
                buffer: vec![],
                total_size: 0,
            }
        } else {
            Self {
                buffer: vec![0u32, size as u32],
                total_size: size as u64,
            }
        }
    }

    /// Creates a `Brle` from an owned run-length buffer.
    pub fn from_vec(buffer: Vec<u32>) -> Self {
        let total_size: u64 = buffer.iter().map(|&x| x as u64).sum();
        Self { buffer, total_size }
    }

    /// Creates a `Brle` from a packed bitmask (`&[u32]`).
    ///
    /// Allocates a new buffer each call. For hot paths, prefer
    /// [`Brle::fill_from_bitmask`] which reuses an existing buffer.
    pub fn from_bitmask(bitmask: &[u32], total_size: usize) -> Self {
        let mut brle = Self {
            buffer: Vec::with_capacity(32),
            total_size: 0,
        };
        brle.fill_from_bitmask(bitmask, total_size);
        brle
    }

    /// Fills this `Brle` from a packed bitmask (`&[u32]`), reusing the
    /// internal buffer to avoid allocation.
    ///
    /// Each bit in the bitmask represents a boolean value (bit set = `true`).
    /// Bit 0 of word 0 is index 0, bit 31 of word 0 is index 31, etc.
    pub fn fill_from_bitmask(&mut self, bitmask: &[u32], total_size: usize) {
        self.buffer.clear();
        self.total_size = total_size as u64;

        if total_size == 0 {
            return;
        }

        let num_words = total_size.div_ceil(32);
        let words = &bitmask[..num_words];

        let mut prev_pos: u32 = 0;
        let mut prev_msb: u64 = 0;

        // Fuse two adjacent u32s into a u64 (little-endian layout).
        #[inline(always)]
        fn fuse(lo: u32, hi: u32) -> u64 {
            lo as u64 | ((hi as u64) << 32)
        }

        let full_u32s = total_size / 32;
        let batch_u32s = full_u32s & !15;

        for (batch_nr, chunk) in words[..batch_u32s].chunks_exact(16).enumerate() {
            let w0 = fuse(chunk[0], chunk[1]);
            let w1 = fuse(chunk[2], chunk[3]);
            let w2 = fuse(chunk[4], chunk[5]);
            let w3 = fuse(chunk[6], chunk[7]);
            let w4 = fuse(chunk[8], chunk[9]);
            let w5 = fuse(chunk[10], chunk[11]);
            let w6 = fuse(chunk[12], chunk[13]);
            let w7 = fuse(chunk[14], chunk[15]);

            let or_all = w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7;

            if or_all == 0 && prev_msb == 0 {
                continue;
            }

            if or_all == u64::MAX {
                let and_all = w0 & w1 & w2 & w3 & w4 & w5 & w6 & w7;
                if and_all == u64::MAX && prev_msb == 1 {
                    prev_msb = 1;
                    continue;
                }
            }

            let batch = [w0, w1, w2, w3, w4, w5, w6, w7];
            let batch_base = (batch_nr as u32) * 512;
            for k in 0..8u32 {
                let w64 = batch[k as usize];
                let shifted = (w64 << 1) | prev_msb;
                let mut tr = w64 ^ shifted;
                prev_msb = w64 >> 63;

                if tr == 0 {
                    continue;
                }

                let base = batch_base + k * 64;
                while tr != 0 {
                    let bit = tr.trailing_zeros();
                    let global = base + bit;
                    self.buffer.push(global - prev_pos);
                    prev_pos = global;
                    tr &= tr.wrapping_sub(1);
                }
            }
        }

        let remaining_pairs = &words[batch_u32s..full_u32s];
        let rem_base_bits = (batch_u32s as u32) * 32;
        for (p, pair) in remaining_pairs.chunks_exact(2).enumerate() {
            let w64 = fuse(pair[0], pair[1]);
            let shifted = (w64 << 1) | prev_msb;
            let mut tr = w64 ^ shifted;
            prev_msb = w64 >> 63;

            if tr == 0 {
                continue;
            }

            let base = rem_base_bits + (p as u32) * 64;
            while tr != 0 {
                let bit = tr.trailing_zeros();
                let global = base + bit;
                self.buffer.push(global - prev_pos);
                prev_pos = global;
                tr &= tr.wrapping_sub(1);
            }
        }

        let u32_processed = batch_u32s + (remaining_pairs.len() & !1);
        let mut i = u32_processed;
        while i < num_words {
            let is_last = i == num_words - 1;
            let bits_in_word = if is_last && !total_size.is_multiple_of(32) {
                total_size % 32
            } else {
                32
            };

            let w = if bits_in_word < 32 {
                words[i] & ((1u32 << bits_in_word) - 1)
            } else {
                words[i]
            };

            let shifted = (w << 1) | prev_msb as u32;
            let mut transitions = w ^ shifted;
            if bits_in_word < 32 {
                transitions &= (1u32 << bits_in_word) - 1;
            }

            let base = (i as u32) * 32;
            while transitions != 0 {
                let bit = transitions.trailing_zeros();
                let global = base + bit;
                self.buffer.push(global - prev_pos);
                prev_pos = global;
                transitions &= transitions.wrapping_sub(1);
            }

            prev_msb = (w >> 31) as u64;
            i += 1;
        }

        let final_run = total_size as u32 - prev_pos;
        if final_run > 0 || self.buffer.is_empty() {
            self.buffer.push(final_run);
        }
    }

    /// Creates a `Brle` from a slice of booleans.
    pub fn from_slice(v: &[bool]) -> Self {
        if v.is_empty() {
            return Self::new(0);
        }

        let mut buffer = Vec::new();
        let mut current_val = false;
        let mut count = 0;

        if v[0] {
            buffer.push(0);
            current_val = true;
        }

        for &val in v {
            if val == current_val {
                count += 1;
            } else {
                buffer.push(count);
                current_val = val;
                count = 1;
            }
        }
        buffer.push(count);

        Self {
            buffer,
            total_size: v.len() as u64,
        }
    }

    /// Returns the total number of booleans in the sequence.
    #[inline]
    pub fn len(&self) -> usize {
        self.total_size as usize
    }

    /// Returns `true` if the sequence is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_size == 0
    }

    /// Decodes the `Brle` into a `Vec<bool>`.
    pub fn to_vec(&self) -> Vec<bool> {
        let mut vec = Vec::with_capacity(self.len());
        for (value, start, end) in self.iter_runs() {
            let run_len = end - start;
            for _ in 0..run_len {
                vec.push(value);
            }
        }
        vec
    }

    /// Checks the boolean values at a given set of indices.
    pub fn is_masked(&self, indices: &[usize]) -> Vec<bool> {
        if indices.is_empty() {
            return Vec::new();
        }

        let mut indexed_indices: Vec<(usize, usize)> =
            indices.iter().copied().enumerate().collect();
        indexed_indices.sort_unstable_by_key(|&(_, index)| index);

        let mut results = vec![false; indices.len()];
        let mut run_iter = self.iter_runs();
        let mut current_run = run_iter.next();

        for &(original_pos, query_index) in &indexed_indices {
            if query_index >= self.len() {
                panic!(
                    "Index {} is out of bounds for Brle of length {}",
                    query_index, self.total_size
                );
            }
            while let Some((value, run_start, run_end)) = current_run {
                if query_index >= run_start && query_index < run_end {
                    results[original_pos] = value;
                    break;
                }
                current_run = run_iter.next();
            }
        }
        results
    }

    /// Checks if all boolean values within a specified range `start..end`
    /// are equal to a given `expected_value`.
    pub fn is_range_all_value(&self, start: usize, end: usize, expected_value: bool) -> bool {
        if start >= end {
            return true;
        }
        if end > self.len() {
            return false;
        }

        let mut pos_covered = start;

        for (run_value, run_start, run_end) in self.iter_runs() {
            let intersect_start = run_start.max(pos_covered);
            let intersect_end = run_end.min(end);

            if intersect_start < intersect_end {
                if run_value != expected_value {
                    return false;
                }
                pos_covered = intersect_end;

                if pos_covered >= end {
                    return true;
                }
            }

            if run_end >= end {
                break;
            }
        }

        pos_covered >= end
    }

    /// Sets a range of booleans to a specified value.
    pub fn mask_range(&mut self, start: usize, end: usize, flag: bool) {
        if start >= end {
            return;
        }
        let ranges = vec![(start, end)];
        self.mask_internal(&ranges, flag);
    }

    /// Sets multiple, potentially non-contiguous, indices to a specified value.
    pub fn mask(&mut self, indices: &[usize], flag: bool) {
        if indices.is_empty() {
            return;
        }

        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_unstable();
        sorted_indices.dedup();

        let mut ranges = Vec::new();
        if sorted_indices.is_empty() {
            return;
        }

        let mut range_start = sorted_indices[0];
        let mut range_end = range_start + 1;

        for &index in sorted_indices.iter().skip(1) {
            if index == range_end {
                range_end = index + 1;
            } else {
                ranges.push((range_start, range_end));
                range_start = index;
                range_end = index + 1;
            }
        }
        ranges.push((range_start, range_end));

        self.mask_internal(&ranges, flag);
    }

    /// Appends a boolean value to the end of the sequence.
    pub fn append(&mut self, flag: bool) {
        if self.buffer.is_empty() {
            if flag {
                self.buffer.extend(&[0, 1]);
            } else {
                self.buffer.push(1);
            }
        } else {
            let last_run_is_true = !(self.buffer.len() - 1).is_multiple_of(2);
            if last_run_is_true == flag {
                *self.buffer.last_mut().unwrap() += 1;
            } else {
                self.buffer.push(1);
            }
        }
        self.total_size += 1;
    }

    /// Extends this `Brle` with another one.
    pub fn extend(&mut self, other: &Self) {
        if other.is_empty() {
            return;
        }
        if self.is_empty() {
            *self = other.clone();
            return;
        }

        let self_last_run_is_true = !(self.buffer.len() - 1).is_multiple_of(2);
        let other_first_run_is_true = other.buffer.first() == Some(&0) && other.buffer.len() > 1;

        if self_last_run_is_true == other_first_run_is_true {
            let other_first_run_len = if other_first_run_is_true {
                other.buffer[1]
            } else {
                other.buffer[0]
            };
            let other_slice_start = if other_first_run_is_true { 2 } else { 1 };

            *self.buffer.last_mut().unwrap() += other_first_run_len;
            self.buffer
                .extend_from_slice(&other.buffer[other_slice_start..]);
        } else if other_first_run_is_true {
            self.buffer.extend_from_slice(&other.buffer[1..]);
        } else {
            self.buffer.extend_from_slice(&other.buffer);
        }
        self.total_size += other.total_size;
    }

    /// Removes the boolean value at a specific index.
    pub fn remove(&mut self, index: usize) {
        if index < self.len() {
            self.remove_range(index, index + 1);
        }
    }

    /// Removes a range of boolean values. The range is exclusive (`start..end`).
    pub fn remove_range(&mut self, start: usize, end: usize) {
        let end = end.min(self.len());
        if start >= end {
            return;
        }

        let head = self.slice(0, start);
        let tail = self.slice(end, self.len());

        let mut new_brle = head;
        new_brle.extend(&tail);
        *self = new_brle;
    }

    /// OR-set bits in `out` for pages whose entire
    /// `[p*page_size, (p+1)*page_size)` range is False under this BRLE
    /// (including the implicit-False tail past `total_size`). Used by
    /// the page-trim optimization in the wire-format builder.
    pub fn droppable_page_bits(
        &self,
        page_size: u32,
        num_pages: u32,
        total_seq_len: u32,
        out: &mut [u64],
    ) {
        if num_pages == 0 || page_size == 0 {
            return;
        }
        let mut covered: u32 = 0;
        for (value, start, end) in self.iter_runs() {
            covered = end as u32;
            if !value {
                set_page_bits_in_range(start as u32, end as u32, page_size, num_pages, out);
            }
        }
        if covered < total_seq_len {
            set_page_bits_in_range(covered, total_seq_len, page_size, num_pages, out);
        }
    }

    /// Append a trimmed copy of this BRLE to `out`, with `skip_ranges` removed.
    ///
    /// Returns the new total size (number of bits in the appended BRLE).
    pub fn write_skipping(&self, skip_ranges: &[(u32, u32)], out: &mut Vec<u32>) -> u32 {
        let mut last_value: Option<bool> = None;
        let mut new_total: u32 = 0;
        let mut skip_idx: usize = 0;

        for (value, start, end) in self.iter_runs() {
            let s = start as u32;
            let e = end as u32;

            let mut skipped: u32 = 0;
            while skip_idx < skip_ranges.len() {
                let (rs, re) = skip_ranges[skip_idx];
                if rs >= e {
                    break;
                }
                let overlap_s = rs.max(s);
                let overlap_e = re.min(e);
                if overlap_s < overlap_e {
                    skipped += overlap_e - overlap_s;
                }
                if re <= e {
                    skip_idx += 1;
                } else {
                    break;
                }
            }

            let raw_len = e - s;
            debug_assert!(skipped <= raw_len);
            let eff_len = raw_len - skipped;
            if eff_len == 0 {
                continue;
            }
            new_total += eff_len;

            match last_value {
                None => {
                    if value {
                        out.push(0);
                    }
                    out.push(eff_len);
                    last_value = Some(value);
                }
                Some(lv) if lv == value => {
                    *out.last_mut().unwrap() += eff_len;
                }
                Some(_) => {
                    out.push(eff_len);
                    last_value = Some(value);
                }
            }
        }

        new_total
    }
}

/// OR-set bits in `out` for every page `p` in `[0, num_pages)` such that
/// the entire range `[p*page_size, (p+1)*page_size)` lies inside `[s, e)`.
#[inline]
fn set_page_bits_in_range(s: u32, e: u32, page_size: u32, num_pages: u32, out: &mut [u64]) {
    if s >= e {
        return;
    }
    let p_lo = s.div_ceil(page_size);
    let p_hi = (e / page_size).min(num_pages);
    if p_lo < p_hi {
        set_bits(out, p_lo, p_hi);
    }
}

/// OR-set bits `[lo, hi)` in `out` (treated as a packed u64 bitmask).
///
/// Shared with runtime callers (e.g. `inference::request::TrimPlan`):
/// the bit-range stamping pattern recurs whenever we need to OR a
/// contiguous range of page indices into a packed bitmap.
#[inline]
pub fn set_bits(out: &mut [u64], lo: u32, hi: u32) {
    if lo >= hi {
        return;
    }
    let word_lo = (lo / 64) as usize;
    let bit_lo = lo % 64;
    let word_hi = (hi / 64) as usize;
    let bit_hi = hi % 64;
    if word_lo == word_hi {
        let mask = ((1u64 << bit_hi).wrapping_sub(1)) & !((1u64 << bit_lo).wrapping_sub(1));
        out[word_lo] |= mask;
        return;
    }
    out[word_lo] |= !((1u64 << bit_lo).wrapping_sub(1));
    for w in &mut out[word_lo + 1..word_hi] {
        *w = u64::MAX;
    }
    if bit_hi > 0 {
        out[word_hi] |= (1u64 << bit_hi).wrapping_sub(1);
    }
}

// Internal implementation and iterators
impl Brle {
    /// Returns an iterator over the runs, yielding `(value, start_index, end_index)`.
    pub fn iter_runs(&self) -> RunIterator<'_> {
        RunIterator {
            buffer: &self.buffer,
            index: 0,
            current_pos: 0,
        }
    }

    /// Creates a new `Brle` representing a slice of the current one.
    fn slice(&self, start: usize, end: usize) -> Self {
        let end = end.min(self.len());
        if start >= end {
            return Self::new(0);
        }

        let new_size = end - start;
        let mut new_buffer = Vec::new();

        for (val, r_start, r_end) in self.iter_runs() {
            let slice_r_start = r_start.max(start);
            let slice_r_end = r_end.min(end);

            if slice_r_start < slice_r_end {
                let len = (slice_r_end - slice_r_start) as u32;

                if new_buffer.is_empty() {
                    if val {
                        new_buffer.push(0);
                    }
                    new_buffer.push(len);
                } else {
                    let last_run_is_true = (new_buffer.len() - 1) % 2 != 0;
                    if last_run_is_true == val {
                        *new_buffer.last_mut().unwrap() += len;
                    } else {
                        new_buffer.push(len);
                    }
                }
            }
        }

        Self {
            buffer: new_buffer,
            total_size: new_size as u64,
        }
    }

    /// The core masking logic. Processes a set of pre-sorted, disjoint ranges.
    fn mask_internal(&mut self, ranges: &[(usize, usize)], flag: bool) {
        if ranges.is_empty() || self.total_size == 0 {
            return;
        }

        let total = self.len();
        let mut events = BTreeSet::new();
        events.insert(0);
        events.insert(total);

        for &(start, end) in ranges {
            let clamped_start = start.min(total);
            let clamped_end = end.min(total);
            if clamped_start < clamped_end {
                events.insert(clamped_start);
                events.insert(clamped_end);
            }
        }

        for run in self.iter_runs() {
            events.insert(run.1);
            events.insert(run.2);
        }

        let mut new_buffer = Vec::new();
        let mut run_iter = self.iter_runs();
        let mut range_iter = ranges.iter().peekable();
        let mut current_run = run_iter.next();

        let event_points: Vec<_> = events.into_iter().collect();
        for window in event_points.windows(2) {
            let start = window[0];
            let end = window[1];
            if start >= end {
                continue;
            }

            let mid_point = start + (end - start) / 2;

            let is_masked = loop {
                match range_iter.peek() {
                    Some(&&(r_start, r_end)) => {
                        if mid_point >= r_end {
                            range_iter.next();
                            continue;
                        }
                        break mid_point >= r_start && mid_point < r_end;
                    }
                    None => break false,
                }
            };

            let value = if is_masked {
                flag
            } else {
                while current_run.is_some() && mid_point >= current_run.unwrap().2 {
                    current_run = run_iter.next();
                }
                current_run
                    .expect("Should always find a run for a valid midpoint")
                    .0
            };

            let len = (end - start) as u32;

            let should_merge = if new_buffer.last().is_some() {
                let last_val_is_true = (new_buffer.len() - 1) % 2 != 0;
                last_val_is_true == value
            } else {
                false
            };

            if should_merge {
                *new_buffer.last_mut().unwrap() += len;
            } else {
                if new_buffer.is_empty() && value {
                    new_buffer.push(0);
                }
                new_buffer.push(len);
            }
        }
        self.buffer = new_buffer;
    }
}

/// An iterator over the runs of a `Brle` instance.
#[derive(Debug)]
pub struct RunIterator<'a> {
    buffer: &'a [u32],
    index: usize,
    current_pos: usize,
}

impl<'a> Iterator for RunIterator<'a> {
    type Item = (bool, usize, usize); // (value, start_index, end_index)

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.buffer.len() {
            let run_len = self.buffer[self.index] as usize;
            let value = !self.index.is_multiple_of(2);

            let start = self.current_pos;
            let end = self.current_pos + run_len;

            self.current_pos = end;
            self.index += 1;

            if run_len > 0 {
                return Some((value, start, end));
            }
        }
        None
    }
}

impl FusedIterator for RunIterator<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create Vec<bool> without type inference ambiguity (pyo3/serde_json).
    fn bv(v: &[bool]) -> Vec<bool> {
        v.to_vec()
    }

    // -- Encoding correctness -------------------------------------------------

    #[test]
    fn roundtrip_complex_pattern() {
        let pattern = vec![
            false, false, true, true, true, false, true, false, false, false,
        ];
        let b = Brle::from_slice(&pattern);
        assert_eq!(b.to_vec(), pattern);
        assert_eq!(b.len(), 10);
        assert_eq!(b.buffer, vec![2, 3, 1, 1, 3]);
    }

    #[test]
    fn from_slice_leading_true_run() {
        let b = Brle::from_slice(&[true, true, false]);
        assert_eq!(b.buffer, vec![0, 2, 1]);
        assert_eq!(b.to_vec(), vec![true, true, false]);
    }

    #[test]
    fn iter_runs_skips_zero_length_prefix() {
        let b = Brle::from_slice(&[true, true, true]);
        let runs: Vec<_> = b.iter_runs().collect();
        assert_eq!(runs, vec![(true, 0, 3)]);
    }

    // -- Masking --------------------------------------------------------------

    #[test]
    fn mask_range_carves_hole_in_trues() {
        let mut b = Brle::from_slice(&[true; 6]);
        b.mask_range(1, 5, false);
        assert_eq!(b.to_vec(), vec![true, false, false, false, false, true]);
    }

    #[test]
    fn mask_overwrite_creates_sandwich() {
        let mut b = Brle::new(10);
        b.mask_range(0, 10, true);
        b.mask_range(3, 7, false);
        assert_eq!(
            b.to_vec(),
            vec![
                true, true, true, false, false, false, false, true, true, true
            ]
        );
    }

    #[test]
    fn mask_scatter_coalesces_adjacent_indices() {
        let mut b = Brle::new(8);
        b.mask(&[7, 1, 5, 3], true);
        assert_eq!(
            b.to_vec(),
            vec![false, true, false, true, false, true, false, true]
        );
    }

    #[test]
    fn mask_scatter_deduplicates() {
        let mut b = Brle::new(4);
        b.mask(&[1, 1, 1], true);
        assert_eq!(b.to_vec(), vec![false, true, false, false]);
    }

    // -- Queries --------------------------------------------------------------

    #[test]
    fn is_masked_preserves_input_order() {
        let b = Brle::from_slice(&[false, true, true, false]);
        assert_eq!(b.is_masked(&[3, 1, 0, 2]), bv(&[false, true, false, true]));
    }

    #[test]
    fn is_range_all_value_boundary() {
        let b = Brle::from_slice(&[false, false, true, true, false]);
        assert!(b.is_range_all_value(2, 4, true));
        assert!(!b.is_range_all_value(1, 4, true));
        assert!(!b.is_range_all_value(0, 10, false));
        assert!(b.is_range_all_value(5, 3, true));
    }

    // -- Structural mutations -------------------------------------------------

    #[test]
    fn extend_merges_matching_boundary_runs() {
        let mut a = Brle::from_slice(&[false, true]);
        let b = Brle::from_slice(&[true, false]);
        a.extend(&b);
        assert_eq!(a.to_vec(), vec![false, true, true, false]);
        assert_eq!(a.buffer, vec![1, 2, 1]);
    }

    #[test]
    fn remove_range_splices_correctly() {
        let mut b = Brle::from_slice(&[false, true, true, true, false]);
        b.remove_range(1, 4);
        assert_eq!(b.to_vec(), vec![false, false]);
    }

    #[test]
    fn append_creates_proper_prefix_for_leading_true() {
        let mut b = Brle::new(0);
        b.append(true);
        assert_eq!(b.buffer, vec![0, 1]);
        b.append(true);
        assert_eq!(b.buffer, vec![0, 2]);
        b.append(false);
        assert_eq!(b.to_vec(), vec![true, true, false]);
    }

    // -- Stress ---------------------------------------------------------------

    #[test]
    fn large_mask_and_verify() {
        let mut b = Brle::new(100);
        b.mask_range(10, 20, true);
        b.mask_range(50, 80, true);

        for i in 0..100 {
            let expected = (10..20).contains(&i) || (50..80).contains(&i);
            assert_eq!(b.is_masked(&[i]), bv(&[expected]), "mismatch at index {i}");
        }
    }

    #[test]
    fn large_alternating_roundtrip() {
        let pattern: Vec<bool> = (0..1000).map(|i| i % 2 == 0).collect();
        let b = Brle::from_slice(&pattern);
        assert_eq!(b.to_vec(), pattern);
        assert_eq!(b.buffer.len(), 1001);
    }

    // -- from_bitmask correctness ---------------------------------------------

    fn naive_from_bitmask(bm: &[u32], total: usize) -> Brle {
        let bools: Vec<bool> = (0..total)
            .map(|i| (bm[i / 32] >> (i % 32)) & 1 != 0)
            .collect();
        Brle::from_slice(&bools)
    }

    fn assert_bitmask_matches_naive(bm: &[u32], total: usize) {
        let fast = Brle::from_bitmask(bm, total);
        let naive = naive_from_bitmask(bm, total);
        assert_eq!(
            fast.buffer, naive.buffer,
            "buffer mismatch for total_size={total}"
        );
        assert_eq!(fast.total_size, naive.total_size);
    }

    #[test]
    fn from_bitmask_roundtrip_simple() {
        let bm = [0b1011u32];
        let b = Brle::from_bitmask(&bm, 4);
        assert_eq!(b.to_vec(), vec![true, true, false, true]);
    }

    #[test]
    fn from_bitmask_roundtrip_all_false() {
        let bm = [0u32; 4];
        let b = Brle::from_bitmask(&bm, 128);
        assert_eq!(b.len(), 128);
        assert_eq!(b.buffer, vec![128]);
    }

    #[test]
    fn from_bitmask_roundtrip_all_true() {
        let bm = [u32::MAX; 4];
        let b = Brle::from_bitmask(&bm, 128);
        assert_eq!(b.len(), 128);
        assert_eq!(b.buffer, vec![0, 128]);
    }

    #[test]
    fn from_bitmask_partial_last_word() {
        let bm = [0x3FFu32];
        let b = Brle::from_bitmask(&bm, 10);
        assert_eq!(b.len(), 10);
        assert!(b.to_vec().iter().all(|&v| v));
    }

    #[test]
    fn from_bitmask_vs_naive_knuth_hash() {
        let mut bm = [0u32; 128];
        for (i, word) in bm.iter_mut().enumerate() {
            *word = (i as u32).wrapping_mul(2654435761);
        }
        assert_bitmask_matches_naive(&bm, 4000);
    }

    #[test]
    fn from_bitmask_vs_naive_edge_sizes() {
        let bm = [0xA5A5A5A5u32; 256];
        for &total in &[
            1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 1023, 1024, 1025, 4096,
            8191, 8192,
        ] {
            assert_bitmask_matches_naive(&bm, total);
        }
    }

    #[test]
    fn from_bitmask_vs_naive_all_zeros_128k() {
        let bm = vec![0u32; 4000];
        assert_bitmask_matches_naive(&bm, 128_000);
    }

    #[test]
    fn from_bitmask_vs_naive_all_ones_128k() {
        let bm = vec![u32::MAX; 4000];
        assert_bitmask_matches_naive(&bm, 128_000);
    }

    #[test]
    fn from_bitmask_vs_naive_sparse_128k() {
        let mut bm = vec![0u32; 4000];
        for i in (0..128_000usize).step_by(1280) {
            bm[i / 32] |= 1u32 << (i % 32);
        }
        assert_bitmask_matches_naive(&bm, 128_000);
    }

    #[test]
    fn from_bitmask_vs_naive_dense_random_128k() {
        let mut bm = vec![0u32; 4000];
        let mut rng = 0x12345678u64;
        for w in bm.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = (rng >> 32) as u32;
        }
        assert_bitmask_matches_naive(&bm, 128_000);
    }

    #[test]
    fn from_bitmask_vs_naive_alternating_words() {
        let bm: Vec<u32> = (0..4000)
            .map(|i| if i % 2 == 0 { 0 } else { u32::MAX })
            .collect();
        assert_bitmask_matches_naive(&bm, 128_000);
    }

    #[test]
    fn fill_from_bitmask_reuses_buffer() {
        let bm1 = [0xFFu32; 4];
        let bm2 = [0u32; 4];
        let mut brle = Brle::from_bitmask(&bm1, 128);
        let cap_after_first = brle.buffer.capacity();
        brle.fill_from_bitmask(&bm2, 128);
        assert!(brle.buffer.capacity() >= cap_after_first);
        assert_eq!(brle.buffer, vec![128]);
    }

    // -- droppable_page_bits --------------------------------------------------

    fn page_bits(b: &Brle, ps: u32, num_pages: u32, total_seq_len: u32) -> Vec<u64> {
        let words = (num_pages as usize).div_ceil(64);
        let mut out = vec![0u64; words.max(1)];
        b.droppable_page_bits(ps, num_pages, total_seq_len, &mut out);
        out
    }

    fn bit_set(words: &[u64], i: u32) -> bool {
        let w = (i / 64) as usize;
        let b = i % 64;
        (words[w] >> b) & 1 != 0
    }

    fn collect_set_bits(words: &[u64], num: u32) -> Vec<u32> {
        (0..num).filter(|&i| bit_set(words, i)).collect()
    }

    #[test]
    fn droppable_pages_causal_mask_yields_none() {
        let b = Brle::all_true(48);
        let bits = page_bits(&b, 16, 3, 48);
        assert_eq!(collect_set_bits(&bits, 3), Vec::<u32>::new());
    }

    #[test]
    fn droppable_pages_attention_sink_pattern() {
        let b = Brle::from_vec(vec![0, 4, 252, 64]);
        assert_eq!(b.len(), 320);
        let bits = page_bits(&b, 16, 20, 320);
        let expected: Vec<u32> = (1..=15).collect();
        assert_eq!(collect_set_bits(&bits, 20), expected);
    }

    #[test]
    fn droppable_pages_window_pattern() {
        let b = Brle::from_vec(vec![240, 80]);
        assert_eq!(b.len(), 320);
        let bits = page_bits(&b, 16, 20, 320);
        let expected: Vec<u32> = (0..=14).collect();
        assert_eq!(collect_set_bits(&bits, 20), expected);
    }

    #[test]
    fn droppable_pages_partial_page_false_run_not_eligible() {
        let b = Brle::from_vec(vec![5, 22, 5]);
        assert_eq!(b.len(), 32);
        let bits = page_bits(&b, 16, 2, 32);
        assert_eq!(collect_set_bits(&bits, 2), Vec::<u32>::new());
    }

    #[test]
    fn droppable_pages_aligned_false_run_eligible() {
        let b = Brle::from_vec(vec![0, 16, 16, 16]);
        assert_eq!(b.len(), 48);
        let bits = page_bits(&b, 16, 3, 48);
        assert_eq!(collect_set_bits(&bits, 3), vec![1]);
    }

    #[test]
    fn droppable_pages_implicit_false_tail() {
        let b = Brle::all_true(32);
        let bits = page_bits(&b, 16, 4, 64);
        assert_eq!(collect_set_bits(&bits, 4), vec![2, 3]);
    }

    #[test]
    fn droppable_pages_or_accumulates_into_existing_bits() {
        let b = Brle::from_vec(vec![0, 16, 16, 16]);
        let mut out = vec![0u64; 1];
        out[0] |= 1u64 << 5;
        b.droppable_page_bits(16, 3, 48, &mut out);
        assert!(bit_set(&out, 1));
        assert!(bit_set(&out, 5));
    }

    #[test]
    fn droppable_pages_cross_word_boundary() {
        let b = Brle::from_vec(vec![0, 16, 16 * 98, 16]);
        let total = 16 * 100;
        assert_eq!(b.len(), total);
        let bits = page_bits(&b, 16, 100, total as u32);
        let expected: Vec<u32> = (1..=98).collect();
        assert_eq!(collect_set_bits(&bits, 100), expected);
    }

    // -- write_skipping --------------------------------------------------------

    fn rebuild(buffer: Vec<u32>) -> Brle {
        Brle::from_vec(buffer)
    }

    #[test]
    fn write_skipping_no_skips_is_identity() {
        let b = Brle::from_vec(vec![0, 4, 252, 64]);
        let mut out = Vec::new();
        let new_total = b.write_skipping(&[], &mut out);
        assert_eq!(new_total, b.len() as u32);
        assert_eq!(out, b.buffer);
    }

    #[test]
    fn write_skipping_drops_middle_false_run_collapses_trues() {
        let b = Brle::from_vec(vec![4, 16, 16, 4]);
        let mut out = Vec::new();
        let new_total = b.write_skipping(&[(20, 36)], &mut out);
        assert_eq!(new_total, 24);
        let r = rebuild(out);
        assert_eq!(r.buffer, vec![4, 20]);
        assert_eq!(r.to_vec(), {
            let mut v = vec![false; 4];
            v.extend(vec![true; 20]);
            v
        });
    }

    #[test]
    fn write_skipping_drops_partial_false_run() {
        let b = Brle::from_vec(vec![32, 8]);
        let mut out = Vec::new();
        let new_total = b.write_skipping(&[(4, 20)], &mut out);
        assert_eq!(new_total, 24);
        assert_eq!(out, vec![16, 8]);
    }

    #[test]
    fn write_skipping_multiple_skips_merge_trues() {
        let b = Brle::from_vec(vec![0, 4, 252, 64]);
        let mut out = Vec::new();
        let new_total = b.write_skipping(&[(16, 32), (32, 48)], &mut out);
        assert_eq!(new_total, 4 + 220 + 64);
        assert_eq!(out, vec![0, 4, 220, 64]);
    }

    #[test]
    fn write_skipping_drops_leading_false_run_keeps_zero_prefix() {
        let b = Brle::from_vec(vec![16, 16]);
        let mut out = Vec::new();
        let new_total = b.write_skipping(&[(0, 16)], &mut out);
        assert_eq!(new_total, 16);
        assert_eq!(out, vec![0, 16]);
    }

    #[test]
    fn write_skipping_into_nonempty_buffer_appends() {
        let b = Brle::from_vec(vec![0, 4, 252, 64]);
        let mut out = vec![99u32, 100, 101];
        let _ = b.write_skipping(&[(16, 32)], &mut out);
        assert_eq!(&out[..3], &[99, 100, 101]);
        assert_eq!(&out[3..], &[0, 4, 236, 64]);
    }

    #[test]
    fn write_skipping_skip_at_boundary_between_runs() {
        let b = Brle::from_vec(vec![16, 16]);
        let mut out = Vec::new();
        let new_total = b.write_skipping(&[(12, 24)], &mut out);
        assert_eq!(new_total, 20);
        assert_eq!(out, vec![12, 8]);
    }

    #[test]
    fn write_skipping_empty_brle_yields_empty_output() {
        let b = Brle::new(0);
        let mut out = vec![1u32, 2, 3];
        let new_total = b.write_skipping(&[], &mut out);
        assert_eq!(new_total, 0);
        assert_eq!(out, vec![1, 2, 3]);
    }
}
