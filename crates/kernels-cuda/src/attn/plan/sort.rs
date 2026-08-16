use super::heap::{Comp, make_heap, sort_heap};

/// libstdc++'s `_S_threshold`: runs of 16 or fewer are left to insertion sort.
const S_THRESHOLD: usize = 16;

/// `std::__lg(n)`: `floor(log2(n))`, and the depth limit is twice it.
fn lg(n: usize) -> u32 {
    usize::BITS - 1 - n.leading_zeros()
}

/// `std::sort(first, last, comp)`, as libstdc++ implements it.
pub fn sort<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    if v.is_empty() {
        return;
    }
    let len = v.len();
    introsort_loop(v, 0, len, 2 * lg(len), c);
    final_insertion_sort(v, c);
}

/// `std::__introsort_loop`.
fn introsort_loop<T: Copy, C: Comp<T>>(
    v: &mut [T],
    first: usize,
    mut last: usize,
    mut depth_limit: u32,
    c: &C,
) {
    while last - first > S_THRESHOLD {
        if depth_limit == 0 {
            partial_sort(v, first, last, c);
            return;
        }
        depth_limit -= 1;
        let cut = unguarded_partition_pivot(v, first, last, c);
        introsort_loop(v, cut, last, depth_limit, c);
        last = cut;
    }
}

/// `std::__partial_sort(first, middle, last)` with `middle == last`.
fn partial_sort<T: Copy, C: Comp<T>>(v: &mut [T], first: usize, last: usize, c: &C) {
    make_heap(&mut v[first..last], c);
    sort_heap(&mut v[first..last], c);
}

/// `std::__unguarded_partition_pivot`.
fn unguarded_partition_pivot<T: Copy, C: Comp<T>>(
    v: &mut [T],
    first: usize,
    last: usize,
    c: &C,
) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, c);
    unguarded_partition(v, first + 1, last, first, c)
}

/// `std::__move_median_to_first(result, a, b, c)`.
fn move_median_to_first<T: Copy, C: Comp<T>>(
    v: &mut [T],
    result: usize,
    a: usize,
    b: usize,
    cc: usize,
    c: &C,
) {
    if c.lt(&v[a], &v[b]) {
        if c.lt(&v[b], &v[cc]) {
            v.swap(result, b);
        } else if c.lt(&v[a], &v[cc]) {
            v.swap(result, cc);
        } else {
            v.swap(result, a);
        }
    } else if c.lt(&v[a], &v[cc]) {
        v.swap(result, a);
    } else if c.lt(&v[b], &v[cc]) {
        v.swap(result, cc);
    } else {
        v.swap(result, b);
    }
}

/// `std::__unguarded_partition(first, last, pivot)`.
fn unguarded_partition<T: Copy, C: Comp<T>>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    c: &C,
) -> usize {
    loop {
        while c.lt(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while c.lt(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

/// `std::__final_insertion_sort`.
fn final_insertion_sort<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    if v.len() > S_THRESHOLD {
        insertion_sort(v, 0, S_THRESHOLD, c);
        for i in S_THRESHOLD..v.len() {
            unguarded_linear_insert(v, i, c);
        }
    } else {
        let len = v.len();
        insertion_sort(v, 0, len, c);
    }
}

/// `std::__insertion_sort`.
fn insertion_sort<T: Copy, C: Comp<T>>(v: &mut [T], first: usize, last: usize, c: &C) {
    if first == last {
        return;
    }
    for i in first + 1..last {
        if c.lt(&v[i], &v[first]) {
            let val = v[i];
            v.copy_within(first..i, first + 1);
            v[first] = val;
        } else {
            unguarded_linear_insert(v, i, c);
        }
    }
}

/// `std::__unguarded_linear_insert`.
fn unguarded_linear_insert<T: Copy, C: Comp<T>>(v: &mut [T], mut last: usize, c: &C) {
    let val = v[last];
    let mut next = last - 1;
    while c.lt(&val, &v[next]) {
        v[last] = v[next];
        last = next;
        next = next.wrapping_sub(1);
    }
    v[last] = val;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Descending by the third element, which is `PrefillSM90Plan`'s
    #[test]
    fn equal_keys_come_out_in_libstdcxx_order() {
        let mut v: Vec<(i32, i32, i32)> = (0..20).map(|i| (i, i, 128)).collect();
        sort(&mut v, &|a: &(i32, i32, i32), b: &(i32, i32, i32)| a.2 > b.2);
        let order: Vec<i32> = v.iter().map(|t| t.0).collect();
        assert_eq!(order.len(), 20);
        assert_ne!(order, (0..20).collect::<Vec<_>>());
    }

    /// A sort is still a sort.
    #[test]
    fn the_range_ends_up_ordered() {
        let mut v: Vec<i32> = (0..200).map(|i| (i * 37) % 101).collect();
        sort(&mut v, &|a: &i32, b: &i32| a > b);
        assert!(v.windows(2).all(|w| w[0] >= w[1]));
    }

    /// Short ranges never reach the quicksort at all.
    #[test]
    fn a_short_range_is_insertion_sorted() {
        let mut v = [3i32, 1, 2];
        sort(&mut v, &|a: &i32, b: &i32| a < b);
        assert_eq!(v, [1, 2, 3]);
    }
}
