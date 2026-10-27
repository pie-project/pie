use std::collections::BTreeSet;
use std::iter::FusedIterator;

#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
pub struct RunMask {
    pub buffer: Vec<u32>,
    pub total_size: u64,
}

impl RunMask {
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

    #[inline]
    pub fn len(&self) -> usize {
        self.total_size as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_size == 0
    }

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

    pub fn remove(&mut self, index: usize) {
        if index < self.len() {
            self.remove_range(index, index + 1);
        }
    }

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
}

impl RunMask {
    pub fn iter_runs(&self) -> RunIterator<'_> {
        RunIterator {
            buffer: &self.buffer,
            index: 0,
            current_pos: 0,
        }
    }

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

#[derive(Debug)]
pub struct RunIterator<'a> {
    buffer: &'a [u32],
    index: usize,
    current_pos: usize,
}

impl<'a> Iterator for RunIterator<'a> {
    type Item = (bool, usize, usize);

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

    #[test]
    fn brle_every_case() {
        roundtrip_complex_pattern();
        from_slice_leading_true_run();
        iter_runs_skips_zero_length_prefix();
    }

    fn roundtrip_complex_pattern() {
        let pattern = vec![
            false, false, true, true, true, false, true, false, false, false,
        ];
        let b = RunMask::from_slice(&pattern);
        assert_eq!(b.to_vec(), pattern);
        assert_eq!(b.len(), 10);
        assert_eq!(b.buffer, vec![2, 3, 1, 1, 3]);
    }

    fn from_slice_leading_true_run() {
        let b = RunMask::from_slice(&[true, true, false]);
        assert_eq!(b.buffer, vec![0, 2, 1]);
        assert_eq!(b.to_vec(), vec![true, true, false]);
    }

    fn iter_runs_skips_zero_length_prefix() {
        let b = RunMask::from_slice(&[true, true, true]);
        let runs: Vec<_> = b.iter_runs().collect();
        assert_eq!(runs, vec![(true, 0, 3)]);
    }
}
