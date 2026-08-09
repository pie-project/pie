/// A comparator, in libstdc++'s sense: "is `a` ordered before `b`".
pub trait Comp<T> {
    /// `comp(a, b)`.
    fn lt(&self, a: &T, b: &T) -> bool;
}

impl<T, F: Fn(&T, &T) -> bool> Comp<T> for F {
    fn lt(&self, a: &T, b: &T) -> bool {
        self(a, b)
    }
}

/// libstdc++'s `__push_heap(first, holeIndex, topIndex, value, comp)`.
fn push_heap_hole<T: Copy, C: Comp<T>>(v: &mut [T], mut hole: isize, top: isize, value: T, c: &C) {
    let mut parent = (hole - 1) / 2;
    while hole > top && c.lt(&v[parent as usize], &value) {
        v[hole as usize] = v[parent as usize];
        hole = parent;
        parent = (hole - 1) / 2;
    }
    v[hole as usize] = value;
}

/// libstdc++'s `__adjust_heap(first, holeIndex, len, value, comp)`.
fn adjust_heap<T: Copy, C: Comp<T>>(v: &mut [T], mut hole: isize, len: isize, value: T, c: &C) {
    let top = hole;
    let mut second_child = hole;
    while second_child < (len - 1) / 2 {
        second_child = 2 * (second_child + 1);
        if c.lt(&v[second_child as usize], &v[(second_child - 1) as usize]) {
            second_child -= 1;
        }
        v[hole as usize] = v[second_child as usize];
        hole = second_child;
    }
    if (len & 1) == 0 && second_child == (len - 2) / 2 {
        second_child = 2 * (second_child + 1);
        v[hole as usize] = v[(second_child - 1) as usize];
        hole = second_child - 1;
    }
    push_heap_hole(v, hole, top, value, c);
}

/// `std::push_heap(first, last, comp)` — the last element joins the heap.
pub fn push_heap<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    let len = v.len() as isize;
    if len < 2 {
        return;
    }
    let value = v[(len - 1) as usize];
    push_heap_hole(v, len - 1, 0, value, c);
}

/// `std::pop_heap(first, last, comp)` — the top moves to the back.
pub fn pop_heap<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    let len = v.len() as isize;
    if len < 2 {
        return;
    }
    let value = v[(len - 1) as usize];
    v[(len - 1) as usize] = v[0];
    adjust_heap(v, 0, len - 1, value, c);
}

/// `std::make_heap(first, last, comp)`.
pub fn make_heap<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    let len = v.len() as isize;
    if len < 2 {
        return;
    }
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[parent as usize];
        adjust_heap(v, parent, len, value, c);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `std::sort_heap(first, last, comp)`.
pub fn sort_heap<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    let mut len = v.len();
    while len > 1 {
        pop_heap(&mut v[..len], c);
        len -= 1;
    }
}

/// FlashInfer's `MinHeap`: `(cta index, accumulated cost)`, cheapest on top.
#[derive(Clone, Debug)]
pub struct MinHeap {
    heap: Vec<(i32, f32)>,
}

/// `MinHeap::compare` — a max-heap on the reversed cost order, i.e. a min-heap.
fn min_heap_comp(a: &(i32, f32), b: &(i32, f32)) -> bool {
    a.1 > b.1
}

impl MinHeap {
    /// `MinHeap(capacity)`: entries `(0, 0.0) ..= (capacity - 1, 0.0)`.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self { heap: (0..capacity as i32).map(|i| (i, 0.0)).collect() }
    }

    /// `insert(element)` — `push_back` then `std::push_heap`.
    pub fn insert(&mut self, element: (i32, f32)) {
        self.heap.push(element);
        push_heap(&mut self.heap, &min_heap_comp);
    }

    /// `pop()` — `std::pop_heap`, then take the back.
    pub fn pop(&mut self) -> (i32, f32) {
        pop_heap(&mut self.heap, &min_heap_comp);
        self.heap.pop().expect("MinHeap::pop on an empty heap")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-equal costs are the case the schedulers spend most of their time in,
    #[test]
    fn ties_come_back_in_libstdcxx_order() {
        let round_robin = |cap: u32, steps: usize| {
            let mut h = MinHeap::new(cap);
            (0..steps)
                .map(|_| {
                    let (idx, cost) = h.pop();
                    h.insert((idx, cost + 1.0));
                    idx
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(round_robin(4, 12), vec![0, 2, 3, 1, 0, 3, 1, 2, 0, 1, 2, 3]);
        assert_eq!(round_robin(3, 9), vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(
            round_robin(8, 16),
            vec![0, 2, 6, 7, 5, 1, 4, 3, 0, 6, 4, 3, 2, 7, 1, 5]
        );
    }

    /// Unequal costs, so the sift path is exercised rather than the tie path.
    #[test]
    fn uneven_costs_follow_the_same_path() {
        let mut h = MinHeap::new(4);
        let costs = [3.0f32, 1.0, 1.0, 2.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let got: Vec<(i32, f32)> = costs
            .iter()
            .map(|&c| {
                let (idx, cost) = h.pop();
                h.insert((idx, cost + c));
                (idx, cost)
            })
            .collect();
        let want = [
            (0, 0.0),
            (2, 0.0),
            (3, 0.0),
            (1, 0.0),
            (2, 1.0),
            (3, 1.0),
            (1, 2.0),
            (2, 2.0),
            (0, 3.0),
            (1, 3.0),
            (2, 3.0),
            (0, 4.0),
        ];
        assert_eq!(got, want);
    }

    /// Once costs differ, the cheapest CTA is the one that comes back.
    #[test]
    fn the_cheapest_cta_is_the_one_that_comes_back() {
        let mut h = MinHeap::new(3);
        let (a, _) = h.pop();
        h.insert((a, 100.0));
        let (b, _) = h.pop();
        h.insert((b, 50.0));
        let (c, cost) = h.pop();
        assert!(cost < 50.0, "a loaded CTA came back before an idle one");
        assert_ne!(c, a);
        assert_ne!(c, b);
    }

    /// `make_heap` + `sort_heap` is the depth-limit path of [`super::sort`],
    #[test]
    fn make_then_sort_orders_the_range() {
        let mut v = [5i32, 3, 9, 1, 7, 7, 2];
        let lt = |a: &i32, b: &i32| a < b;
        make_heap(&mut v, &lt);
        sort_heap(&mut v, &lt);
        assert_eq!(v, [1, 2, 3, 5, 7, 7, 9]);
    }
}
