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
#[derive(Debug, Clone, PartialEq, Eq)]
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
        if self.index >= self.buffer.len() {
            return None;
        }

        let run_len = self.buffer[self.index] as usize;
        let value = self.index % 2 != 0;

        let start = self.current_pos;
        let end = self.current_pos + run_len;

        self.current_pos = end;
        self.index += 1;

        Some((value, start, end))
    }
}

impl FusedIterator for RunIterator<'_> {}
