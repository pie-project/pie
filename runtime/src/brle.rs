use std::collections::BTreeSet;
use std::iter::FusedIterator;

/// A Binary Run-Length Encoding (BRLE) structure.
///
/// This structure efficiently stores a large sequence of booleans by encoding
/// the lengths of consecutive runs of `false` and `true` values.
/// The sequence always begins with a run of `false`s, which may be zero-length.
///
/// # Encoding
/// - `[false, false, true, true, true, false]` is encoded as `[2, 3, 1]`.
/// - `[true, true, false]` is encoded as `[0, 2, 1]`.
///
/// This makes it highly memory-efficient for data with long, contiguous runs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Brle {
    /// The buffer of run lengths. Even indices are for `false` runs, odd for `true`.
    pub buffer: Vec<u32>,
    /// The total number of boolean values represented.
    pub total_size: usize,
}

// Public API
impl Brle {
    /// Creates a new `Brle` instance representing `size` `false` values.
    ///
    /// # Arguments
    /// * `size` - The total number of elements in the boolean sequence.
    pub fn new(size: usize) -> Self {
        if size == 0 {
            Self {
                buffer: vec![],
                total_size: 0,
            }
        } else {
            Self {
                buffer: vec![size as u32],
                total_size: size,
            }
        }
    }

    /// Creates a `Brle` from an owned run-length buffer.
    pub fn from_vec(buffer: Vec<u32>) -> Self {
        let total_size = buffer.iter().map(|&x| x as usize).sum();
        Self { buffer, total_size }
    }

    /// Creates a `Brle` from a packed bitmask (`&[u32]`).
    ///
    /// Allocates a new buffer each call. For hot paths, prefer
    /// [`fill_from_bitmask`] which reuses an existing buffer.
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
    ///
    /// Uses XOR-based transition detection on `u64` chunks with 8-wide batch
    /// skipping: groups of 8 u64s (512 bits) are OR/AND-reduced and checked
    /// in a single branch. Uniform batches cost ~4 ops for 512 bits.
    ///
    /// # Arguments
    /// * `bitmask` - The packed bitmask words.
    /// * `total_size` - The total number of boolean values (may be less than
    ///   `bitmask.len() * 32` if the last word is partial).
    pub fn fill_from_bitmask(&mut self, bitmask: &[u32], total_size: usize) {
        self.buffer.clear();
        self.total_size = total_size;

        if total_size == 0 {
            return;
        }

        let num_words = (total_size + 31) / 32;
        let words = &bitmask[..num_words];

        let mut prev_pos: u32 = 0;
        let mut prev_msb: u64 = 0;

        // Fuse two adjacent u32s into a u64 (little-endian layout).
        // On x86_64, LLVM optimises this to a single 64-bit load.
        #[inline(always)]
        fn fuse(lo: u32, hi: u32) -> u64 {
            lo as u64 | ((hi as u64) << 32)
        }

        // Process u32 words in groups of 16 (= 8 u64s = 512 bits).
        // `chunks_exact` guarantees each chunk has exactly 16 elements,
        // so all indexing within the chunk is bounds-check-free.
        let full_u32s = total_size / 32;
        let batch_u32s = full_u32s & !15; // round down to multiple of 16

        for (batch_nr, chunk) in words[..batch_u32s].chunks_exact(16).enumerate() {
            let w0 = fuse(chunk[0],  chunk[1]);
            let w1 = fuse(chunk[2],  chunk[3]);
            let w2 = fuse(chunk[4],  chunk[5]);
            let w3 = fuse(chunk[6],  chunk[7]);
            let w4 = fuse(chunk[8],  chunk[9]);
            let w5 = fuse(chunk[10], chunk[11]);
            let w6 = fuse(chunk[12], chunk[13]);
            let w7 = fuse(chunk[14], chunk[15]);

            // Fast uniform check via OR-reduction.
            let or_all = w0 | w1 | w2 | w3 | w4 | w5 | w6 | w7;

            if or_all == 0 && prev_msb == 0 {
                continue;
            }

            // Lazy AND: only compute if or_all suggests possible all-ones.
            if or_all == u64::MAX {
                let and_all = w0 & w1 & w2 & w3 & w4 & w5 & w6 & w7;
                if and_all == u64::MAX && prev_msb == 1 {
                    prev_msb = 1;
                    continue;
                }
            }

            // Slow path: process each u64 individually.
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
                    let bit = tr.trailing_zeros() as u32;
                    let global = base + bit;
                    self.buffer.push(global - prev_pos);
                    prev_pos = global;
                    tr &= tr.wrapping_sub(1);
                }
            }
        }

        // Process remaining u32 pairs that didn't fill a batch of 16.
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
                let bit = tr.trailing_zeros() as u32;
                let global = base + bit;
                self.buffer.push(global - prev_pos);
                prev_pos = global;
                tr &= tr.wrapping_sub(1);
            }
        }


        // Handle the remaining 0-1 u32 words (possibly partial last word).
        let u32_processed = batch_u32s + (remaining_pairs.len() & !1);
        let mut i = u32_processed;
        while i < num_words {
            let is_last = i == num_words - 1;
            let bits_in_word = if is_last && total_size % 32 != 0 {
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

        // Final run: remaining bits after the last transition.
        let final_run = total_size as u32 - prev_pos;
        if final_run > 0 || self.buffer.is_empty() {
            self.buffer.push(final_run);
        }
    }


    /// Creates a `Brle` from a slice of booleans.
    ///
    /// This method efficiently scans the slice and constructs the run-length encoded buffer.
    ///
    /// # Arguments
    /// * `v` - A slice of booleans to encode.
    pub fn from_slice(v: &[bool]) -> Self {
        if v.is_empty() {
            return Self::new(0);
        }

        let mut buffer = Vec::new();
        let mut current_val = false;
        let mut count = 0;

        // Handle the initial run of `false`s, which could be zero-length.
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
        buffer.push(count); // Push the last run

        Self {
            buffer,
            total_size: v.len(),
        }
    }

    /// Returns the total number of booleans in the sequence.
    pub fn len(&self) -> usize {
        self.total_size
    }

    /// Returns `true` if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.total_size == 0
    }

    /// Decodes the `Brle` into a `Vec<bool>`.
    ///
    /// Note: This can consume a large amount of memory if the total size is large.
    pub fn to_vec(&self) -> Vec<bool> {
        let mut vec = Vec::with_capacity(self.total_size);
        // The iterator yields a 3-element tuple. Destructure it correctly.
        for (value, start, end) in self.iter_runs() {
            let run_len = end - start;
            for _ in 0..run_len {
                vec.push(value);
            }
        }
        vec
    }

    /// Checks the boolean values at a given set of indices.
    ///
    /// This is highly optimized to check multiple indices in a single pass over the data.
    ///
    /// # Arguments
    /// * `indices` - A slice of indices to check. The indices do not need to be sorted.
    ///
    /// # Returns
    /// A `Vec<bool>` containing the value at each corresponding index.
    pub fn is_masked(&self, indices: &[usize]) -> Vec<bool> {
        if indices.is_empty() {
            return Vec::new();
        }

        // To preserve the original order of results, we pair indices with their original position.
        let mut indexed_indices: Vec<(usize, usize)> =
            indices.iter().copied().enumerate().collect();
        indexed_indices.sort_unstable_by_key(|&(_, index)| index);

        let mut results = vec![false; indices.len()];
        let mut run_iter = self.iter_runs();
        let mut current_run = run_iter.next();

        for &(original_pos, query_index) in &indexed_indices {
            if query_index >= self.total_size {
                panic!(
                    "Index {} is out of bounds for Brle of length {}",
                    query_index, self.total_size
                );
            }
            // Advance through runs until we find the one containing the query_index
            while let Some((value, run_start, run_end)) = current_run {
                if query_index >= run_start && query_index < run_end {
                    results[original_pos] = value;
                    break; // Found the value for this index, move to the next index
                }
                // This index is past the current run, so get the next run
                current_run = run_iter.next();
            }
        }
        results
    }

    /// Checks if all boolean values within a specified range `start..end`
    /// are equal to a given `expected_value`.
    ///
    /// This operation is efficient and avoids creating a new `Brle` instance.
    ///
    /// # Arguments
    /// * `start` - The starting index of the range (inclusive).
    /// * `end` - The ending index of the range (exclusive).
    /// * `expected_value` - The boolean value to check for.
    ///
    /// # Returns
    /// `true` if every value in the range is `expected_value`, `false` otherwise.
    /// Returns `true` for an empty range (`start >= end`).
    pub fn is_range_all_value(&self, start: usize, end: usize, expected_value: bool) -> bool {
        if start >= end {
            return true;
        }
        if end > self.total_size {
            return false;
        }

        // `pos_covered` tracks the end of the last verified segment within our target range.
        let mut pos_covered = start;

        for (run_value, run_start, run_end) in self.iter_runs() {
            // Find the intersection of the current run `[run_start, run_end)`
            // and the part of the range we still need to check, `[pos_covered, end)`.
            let intersect_start = run_start.max(pos_covered);
            let intersect_end = run_end.min(end);

            // If there's a valid intersection...
            if intersect_start < intersect_end {
                // ...check if the value matches what we expect.
                if run_value != expected_value {
                    // Mismatch found. The range is not uniform.
                    return false;
                }
                // This segment is correct. Update our coverage marker.
                pos_covered = intersect_end;

                // If we have covered the entire target range, we are done.
                if pos_covered >= end {
                    return true;
                }
            }

            // Optimization: If the current run already extends beyond our target range,
            // we don't need to check any subsequent runs.
            if run_end >= end {
                break;
            }
        }

        // After checking all relevant runs, verify if the entire range was covered.
        pos_covered >= end
    }

    /// Sets a range of booleans to a specified value.
    /// The range is exclusive of `end` (`start..end`).
    ///
    /// This is an alias for the more general `mask` function.
    ///
    /// # Arguments
    /// * `start` - The starting index of the range (inclusive).
    /// * `end` - The ending index of the range (exclusive).
    /// * `flag` - The boolean value to set.
    pub fn mask_range(&mut self, start: usize, end: usize, flag: bool) {
        if start >= end {
            return;
        }
        let ranges = vec![(start, end)];
        self.mask_internal(&ranges, flag);
    }

    /// Sets multiple, potentially non-contiguous, indices to a specified value.
    ///
    /// This method is the core of the efficient modification API. It processes all
    /// updates in a single pass by converting indices into ranges and then merging
    /// them with the existing runs.
    ///
    /// # Arguments
    /// * `indices` - A slice of indices to set.
    /// * `flag` - The boolean value to set.
    pub fn mask(&mut self, indices: &[usize], flag: bool) {
        if indices.is_empty() {
            return;
        }

        // Step 1: Sort and group indices into contiguous ranges.
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
                // Extend the current range
                range_end = index + 1;
            } else {
                // Finish the old range and start a new one
                ranges.push((range_start, range_end));
                range_start = index;
                range_end = index + 1;
            }
        }
        ranges.push((range_start, range_end)); // Push the last range

        // Step 2: Use the internal mask implementation
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
            let last_run_is_true = (self.buffer.len() - 1) % 2 != 0;
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

        let self_last_run_is_true = (self.buffer.len() - 1) % 2 != 0;
        let other_first_run_is_true = other.buffer.get(0) == Some(&0) && other.buffer.len() > 1;

        if self_last_run_is_true == other_first_run_is_true {
            // Merge the last run of self with the first run of other.
            let other_first_run_len = if other_first_run_is_true {
                other.buffer[1]
            } else {
                other.buffer[0]
            };
            let other_slice_start = if other_first_run_is_true { 2 } else { 1 };

            *self.buffer.last_mut().unwrap() += other_first_run_len;
            self.buffer
                .extend_from_slice(&other.buffer[other_slice_start..]);
        } else {
            // No merge needed. If other starts with a zero-length false run, skip it.
            if other_first_run_is_true {
                self.buffer.extend_from_slice(&other.buffer[1..]);
            } else {
                self.buffer.extend_from_slice(&other.buffer);
            }
        }
        self.total_size += other.total_size;
    }

    /// Removes the boolean value at a specific index.
    pub fn remove(&mut self, index: usize) {
        if index < self.total_size {
            self.remove_range(index, index + 1);
        }
    }

    /// Removes a range of boolean values. The range is exclusive (`start..end`).
    pub fn remove_range(&mut self, start: usize, end: usize) {
        let end = end.min(self.total_size);
        if start >= end {
            return;
        }

        let head = self.slice(0, start);
        let tail = self.slice(end, self.total_size);

        let mut new_brle = head;
        new_brle.extend(&tail);
        *self = new_brle;
    }
}

// Internal implementation and iterators
impl Brle {
    /// Returns an iterator over the runs, yielding `(value, start_index, end_index)`.
    pub fn iter_runs(&self) -> RunIterator {
        RunIterator {
            buffer: &self.buffer,
            index: 0,
            current_pos: 0,
        }
    }

    /// Creates a new `Brle` representing a slice of the current one.
    fn slice(&self, start: usize, end: usize) -> Self {
        let end = end.min(self.total_size);
        if start >= end {
            return Self::new(0);
        }

        let new_size = end - start;
        let mut new_buffer = Vec::new();

        // Iterate through the runs of the original brle
        for (val, r_start, r_end) in self.iter_runs() {
            // Calculate the intersection of the current run [r_start, r_end)
            // and the desired slice [start, end).
            let slice_r_start = r_start.max(start);
            let slice_r_end = r_end.min(end);

            if slice_r_start < slice_r_end {
                // There is an overlap.
                let len = (slice_r_end - slice_r_start) as u32;

                if new_buffer.is_empty() {
                    // This is the first run of the new slice.
                    if val {
                        // if it starts with true
                        new_buffer.push(0);
                    }
                    new_buffer.push(len);
                } else {
                    // Not the first run. Check if we can merge with the previous run.
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
            total_size: new_size,
        }
    }

    /// The core masking logic. Processes a set of pre-sorted, disjoint ranges.
    fn mask_internal(&mut self, ranges: &[(usize, usize)], flag: bool) {
        if ranges.is_empty() || self.total_size == 0 {
            return;
        }

        let mut events = BTreeSet::new();
        events.insert(0);
        events.insert(self.total_size);

        for &(start, end) in ranges {
            let clamped_start = start.min(self.total_size);
            let clamped_end = end.min(self.total_size);
            if clamped_start < clamped_end {
                events.insert(clamped_start);
                events.insert(clamped_end);
            }
        }

        for run in self.iter_runs() {
            events.insert(run.1); // run start
            events.insert(run.2); // run end
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
        // Skip zero-length runs so consumers never see (value, N, N)
        while self.index < self.buffer.len() {
            let run_len = self.buffer[self.index] as usize;
            let value = self.index % 2 != 0;

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
    fn bv(v: &[bool]) -> Vec<bool> { v.to_vec() }

    // -- Encoding correctness -------------------------------------------------

    #[test]
    fn roundtrip_complex_pattern() {
        let pattern = vec![
            false, false, true, true, true, false, true, false, false, false,
        ];
        let b = Brle::from_slice(&pattern);
        assert_eq!(b.to_vec(), pattern);
        assert_eq!(b.len(), 10);
        // Verify internal encoding: [2, 3, 1, 1, 3]
        assert_eq!(b.buffer, vec![2, 3, 1, 1, 3]);
    }

    #[test]
    fn from_slice_leading_true_run() {
        // Leading true requires a zero-length false run prefix
        let b = Brle::from_slice(&[true, true, false]);
        assert_eq!(b.buffer, vec![0, 2, 1]);
        assert_eq!(b.to_vec(), vec![true, true, false]);
    }

    #[test]
    fn iter_runs_skips_zero_length_prefix() {
        let b = Brle::from_slice(&[true, true, true]);
        let runs: Vec<_> = b.iter_runs().collect();
        // Zero-length false run at buffer[0] should be skipped
        assert_eq!(runs, vec![(true, 0, 3)]);
    }

    // -- Masking (the complex algorithm) --------------------------------------

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
            vec![true, true, true, false, false, false, false, true, true, true]
        );
    }

    #[test]
    fn mask_scatter_coalesces_adjacent_indices() {
        // Scattered indices [1,3,5,7] should produce alternating pattern
        let mut b = Brle::new(8);
        b.mask(&[7, 1, 5, 3], true); // intentionally unsorted
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
        assert!(b.is_range_all_value(2, 4, true));    // exact true run
        assert!(!b.is_range_all_value(1, 4, true));   // bleeds into false
        assert!(!b.is_range_all_value(0, 10, false));  // beyond total_size
        assert!(b.is_range_all_value(5, 3, true));     // reversed → empty → true
    }

    // -- Structural mutations -------------------------------------------------

    #[test]
    fn extend_merges_matching_boundary_runs() {
        let mut a = Brle::from_slice(&[false, true]); // ends with true
        let b = Brle::from_slice(&[true, false]);      // starts with true
        a.extend(&b);
        assert_eq!(a.to_vec(), vec![false, true, true, false]);
        // Middle true runs should be merged in the buffer
        assert_eq!(a.buffer, vec![1, 2, 1]);
    }

    #[test]
    fn remove_range_splices_correctly() {
        let mut b = Brle::from_slice(&[false, true, true, true, false]);
        b.remove_range(1, 4); // remove the true block
        assert_eq!(b.to_vec(), vec![false, false]);
    }

    #[test]
    fn append_creates_proper_prefix_for_leading_true() {
        let mut b = Brle::new(0);
        b.append(true);
        assert_eq!(b.buffer, vec![0, 1]); // zero false prefix + one true
        b.append(true);
        assert_eq!(b.buffer, vec![0, 2]); // extends the true run
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
            assert_eq!(
                b.is_masked(&[i]),
                bv(&[expected]),
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn large_alternating_roundtrip() {
        let pattern: Vec<bool> = (0..1000).map(|i| i % 2 == 0).collect();
        let b = Brle::from_slice(&pattern);
        assert_eq!(b.to_vec(), pattern);
        // 1000 alternating values, starting with true → 1001 buffer entries
        // (zero-length false prefix + 1000 singleton runs)
        assert_eq!(b.buffer.len(), 1001);
    }

    // -- from_bitmask correctness ------------------------------------------------

    /// Naive reference: expand bitmask to bools, feed to from_slice.
    fn naive_from_bitmask(bm: &[u32], total: usize) -> Brle {
        let bools: Vec<bool> = (0..total)
            .map(|i| (bm[i / 32] >> (i % 32)) & 1 != 0)
            .collect();
        Brle::from_slice(&bools)
    }

    fn assert_bitmask_matches_naive(bm: &[u32], total: usize) {
        let fast = Brle::from_bitmask(bm, total);
        let naive = naive_from_bitmask(bm, total);
        assert_eq!(fast.buffer, naive.buffer,
            "buffer mismatch for total_size={total}");
        assert_eq!(fast.total_size, naive.total_size);
    }

    #[test]
    fn from_bitmask_roundtrip_simple() {
        // 0b1011 = bits 0,1,3 set → [true, true, false, true]
        let bm = [0b1011u32];
        let b = Brle::from_bitmask(&bm, 4);
        assert_eq!(b.to_vec(), vec![true, true, false, true]);
    }

    #[test]
    fn from_bitmask_roundtrip_all_false() {
        let bm = [0u32; 4];
        let b = Brle::from_bitmask(&bm, 128);
        assert_eq!(b.len(), 128);
        assert_eq!(b.buffer, vec![128]); // single false run
    }

    #[test]
    fn from_bitmask_roundtrip_all_true() {
        let bm = [u32::MAX; 4];
        let b = Brle::from_bitmask(&bm, 128);
        assert_eq!(b.len(), 128);
        assert_eq!(b.buffer, vec![0, 128]); // zero false prefix + 128 true
    }

    #[test]
    fn from_bitmask_partial_last_word() {
        // 10 bits: word = 0b11_1111_1111 (all set)
        let bm = [0x3FFu32];
        let b = Brle::from_bitmask(&bm, 10);
        assert_eq!(b.len(), 10);
        assert!(b.to_vec().iter().all(|&v| v));
    }

    #[test]
    fn from_bitmask_vs_naive_knuth_hash() {
        // Deterministic pseudo-random pattern (128 words = 4096 bits)
        let mut bm = [0u32; 128];
        for i in 0..128 {
            bm[i] = (i as u32).wrapping_mul(2654435761);
        }
        assert_bitmask_matches_naive(&bm, 4000); // partial last word
    }

    #[test]
    fn from_bitmask_vs_naive_edge_sizes() {
        // Test sizes that hit every batch/u64/u32 boundary edge:
        //   BATCH=8 u64 = 512 bits = 16 u32 words
        let bm = [0xA5A5A5A5u32; 256]; // 256 words = 8192 bits
        for &total in &[
            1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129,         // u32/u64 edges
            511, 512, 513, 1023, 1024, 1025,                      // batch edges
            4096, 8191, 8192,                                     // full buffer
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
        // Set ~100 scattered single bits
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
        let bm: Vec<u32> = (0..4000).map(|i| if i % 2 == 0 { 0 } else { u32::MAX }).collect();
        assert_bitmask_matches_naive(&bm, 128_000);
    }

    #[test]
    fn fill_from_bitmask_reuses_buffer() {
        let bm1 = [0xFFu32; 4]; // 128 bits
        let bm2 = [0u32; 4];
        let mut brle = Brle::from_bitmask(&bm1, 128);
        let cap_after_first = brle.buffer.capacity();
        brle.fill_from_bitmask(&bm2, 128);
        // Buffer should be reused, not reallocated
        assert!(brle.buffer.capacity() >= cap_after_first);
        assert_eq!(brle.buffer, vec![128]); // all false
    }

    // -- benchmark ---------------------------------------------------------------

    #[test]
    #[ignore] // cargo test --lib -- brle::tests::bench_from_bitmask --ignored --nocapture
    fn bench_from_bitmask() {
        use std::time::Instant;

        const VOCAB: usize = 128_000;
        const WORDS: usize = (VOCAB + 31) / 32;
        const ITERS: usize = 10_000;

        // Pattern 1: all zeros (one run)
        let bm_zeros = vec![0u32; WORDS];

        // Pattern 2: all ones (one run)
        let bm_ones = vec![u32::MAX; WORDS];

        // Pattern 3: sparse — ~100 tokens allowed (scattered single bits)
        let mut bm_sparse = vec![0u32; WORDS];
        for i in (0..VOCAB).step_by(VOCAB / 100) {
            bm_sparse[i / 32] |= 1u32 << (i % 32);
        }

        // Pattern 4: dense — ~50% random fill
        let mut bm_dense = vec![0u32; WORDS];
        let mut rng = 0x12345678u64;
        for w in bm_dense.iter_mut() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = (rng >> 32) as u32;
        }

        // Pattern 5: word-aligned blocks (alternating 0x0000/0xFFFF words)
        let bm_blocks: Vec<u32> = (0..WORDS)
            .map(|i| if i % 2 == 0 { 0 } else { u32::MAX })
            .collect();

        let patterns: &[(&str, &[u32])] = &[
            ("all_zeros", &bm_zeros),
            ("all_ones", &bm_ones),
            ("sparse_100", &bm_sparse),
            ("dense_50pct", &bm_dense),
            ("word_blocks", &bm_blocks),
        ];

        // --- Benchmark from_bitmask (allocating) ---
        eprintln!("\n=== Brle::from_bitmask [alloc] (vocab={VOCAB}, iters={ITERS}) ===");
        eprintln!("{:<15} {:>10} {:>10} {:>8}", "pattern", "total", "per_call", "runs");

        for &(name, bm) in patterns {
            for _ in 0..100 {
                std::hint::black_box(Brle::from_bitmask(bm, VOCAB));
            }

            let t = Instant::now();
            let mut last = Brle::new(0);
            for _ in 0..ITERS {
                last = std::hint::black_box(Brle::from_bitmask(
                    std::hint::black_box(bm), VOCAB,
                ));
            }
            let elapsed = t.elapsed();
            let per_call = elapsed / ITERS as u32;
            eprintln!("{:<15} {:>10?} {:>10?} {:>8}", name, elapsed, per_call, last.buffer.len());
        }

        // --- Benchmark fill_from_bitmask (warm buffer, no alloc) ---
        eprintln!("\n=== Brle::fill_from_bitmask [warm] (vocab={VOCAB}, iters={ITERS}) ===");
        eprintln!("{:<15} {:>10} {:>10} {:>8}", "pattern", "total", "per_call", "runs");

        for &(name, bm) in patterns {
            let mut brle = Brle::from_bitmask(bm, VOCAB); // warm up buffer
            for _ in 0..100 {
                brle.fill_from_bitmask(bm, VOCAB);
            }

            let t = Instant::now();
            for _ in 0..ITERS {
                brle.fill_from_bitmask(std::hint::black_box(bm), VOCAB);
                std::hint::black_box(&brle);
            }
            let elapsed = t.elapsed();
            let per_call = elapsed / ITERS as u32;
            eprintln!("{:<15} {:>10?} {:>10?} {:>8}", name, elapsed, per_call, brle.buffer.len());
        }
    }
}
