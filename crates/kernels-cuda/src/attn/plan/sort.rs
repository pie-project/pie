use super::heap::{Comp, make_heap, sort_heap};

const S_THRESHOLD: usize = 16;

fn lg(n: usize) -> u32 {
    usize::BITS - 1 - n.leading_zeros()
}

pub fn sort<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    if v.is_empty() {
        return;
    }
    let len = v.len();
    introsort_loop(v, 0, len, 2 * lg(len), c);
    final_insertion_sort(v, c);
}

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

fn partial_sort<T: Copy, C: Comp<T>>(v: &mut [T], first: usize, last: usize, c: &C) {
    make_heap(&mut v[first..last], c);
    sort_heap(&mut v[first..last], c);
}

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
