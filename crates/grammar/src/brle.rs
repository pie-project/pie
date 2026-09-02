//! Binary Run-Length Encoded (BRLE) boolean sequences. `RunMask` is part of
//! the submission schema (the engine contract's `fire::Mask` is the same run
//! encoding) and is also the type the runtime/engines manipulate directly,
//! so there's no duplicate type or wire-boundary conversion.
//!
//! ## Encoding
//! - `[false, false, true, true, true, false]` -> `[2, 3, 1]`
//! - `[true, true, false]` -> `[0, 2, 1]` (zero-length false prefix)
//!
//! Starts-with-false invariant: a sequence beginning with `true` always has
//! a leading `0` run, so even buffer indices are false runs, odd are true.

use std::collections::BTreeSet;
use std::iter::FusedIterator;

/// A Binary Run-Length Encoding (BRLE) structure.
///
/// `total_size` is `u64` rather than `usize` because the wire schema
/// requires fixed width. The accessor methods take/return `usize` for
/// convenience and cast at the boundary.
#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RunMask {
    /// The buffer of run lengths. Even indices = false-run lengths,
    /// odd indices = true-run lengths.
    pub buffer: Vec<u32>,
    /// Total boolean count this BRLE represents.
    pub total_size: u64,
}

// Public API
impl RunMask {
    /// Creates a new `RunMask` instance representing `size` `false` values.
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

    /// Creates a new `RunMask` instance representing `size` `true` values.
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

    /// Creates a `RunMask` from an owned run-length buffer.
    pub fn from_vec(buffer: Vec<u32>) -> Self {
        let total_size: u64 = buffer.iter().map(|&x| x as u64).sum();
        Self { buffer, total_size }
    }

    /// Creates a `RunMask` from a packed bitmask (`&[u32]`).
    ///
    /// Allocates a new buffer each call. For hot paths, prefer
    /// [`RunMask::fill_from_bitmask`] which reuses an existing buffer.
    pub fn from_bitmask(bitmask: &[u32], total_size: usize) -> Self {
        let mut brle = Self {
            buffer: Vec::with_capacity(32),
            total_size: 0,
        };
        brle.fill_from_bitmask(bitmask, total_size);
        brle
    }

    /// Fills this `RunMask` from a packed bitmask (`&[u32]`), reusing the
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

    /// Creates a `RunMask` from a slice of booleans.
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

    /// Decodes the `RunMask` into a `Vec<bool>`.
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
                    "Index {} is out of bounds for RunMask of length {}",
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

    /// Extends this `RunMask` with another one.
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
impl RunMask {
    /// Returns an iterator over the runs, yielding `(value, start_index, end_index)`.
    pub fn iter_runs(&self) -> RunIterator<'_> {
        RunIterator {
            buffer: &self.buffer,
            index: 0,
            current_pos: 0,
        }
    }

    /// Creates a new `RunMask` representing a slice of the current one.
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

/// An iterator over the runs of a `RunMask` instance.
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

    // -- Encoding correctness -------------------------------------------------

    #[test]
    fn roundtrip_complex_pattern() {
        let pattern = vec![
            false, false, true, true, true, false, true, false, false, false,
        ];
        let b = RunMask::from_slice(&pattern);
        assert_eq!(b.to_vec(), pattern);
        assert_eq!(b.len(), 10);
        assert_eq!(b.buffer, vec![2, 3, 1, 1, 3]);
    }

    #[test]
    fn from_slice_leading_true_run() {
        let b = RunMask::from_slice(&[true, true, false]);
        assert_eq!(b.buffer, vec![0, 2, 1]);
        assert_eq!(b.to_vec(), vec![true, true, false]);
    }

    #[test]
    fn iter_runs_skips_zero_length_prefix() {
        let b = RunMask::from_slice(&[true, true, true]);
        let runs: Vec<_> = b.iter_runs().collect();
        assert_eq!(runs, vec![(true, 0, 3)]);
    }

    // -- Masking --------------------------------------------------------------

    // -- Queries --------------------------------------------------------------

    // -- Structural mutations -------------------------------------------------

    // -- Stress ---------------------------------------------------------------

    // -- from_bitmask correctness ---------------------------------------------

    // -- droppable_page_bits --------------------------------------------------

    // -- write_skipping --------------------------------------------------------

}
