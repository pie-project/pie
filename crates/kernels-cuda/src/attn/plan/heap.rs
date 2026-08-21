
pub trait Comp<T> {

    fn lt(&self, a: &T, b: &T) -> bool;
}

impl<T, F: Fn(&T, &T) -> bool> Comp<T> for F {
    fn lt(&self, a: &T, b: &T) -> bool {
        self(a, b)
    }
}

fn push_heap_hole<T: Copy, C: Comp<T>>(v: &mut [T], mut hole: isize, top: isize, value: T, c: &C) {
    let mut parent = (hole - 1) / 2;
    while hole > top && c.lt(&v[parent as usize], &value) {
        v[hole as usize] = v[parent as usize];
        hole = parent;
        parent = (hole - 1) / 2;
    }
    v[hole as usize] = value;
}

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

pub fn push_heap<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    let len = v.len() as isize;
    if len < 2 {
        return;
    }
    let value = v[(len - 1) as usize];
    push_heap_hole(v, len - 1, 0, value, c);
}

pub fn pop_heap<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    let len = v.len() as isize;
    if len < 2 {
        return;
    }
    let value = v[(len - 1) as usize];
    v[(len - 1) as usize] = v[0];
    adjust_heap(v, 0, len - 1, value, c);
}

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

pub fn sort_heap<T: Copy, C: Comp<T>>(v: &mut [T], c: &C) {
    let mut len = v.len();
    while len > 1 {
        pop_heap(&mut v[..len], c);
        len -= 1;
    }
}

#[derive(Clone, Debug)]
pub struct MinHeap {
    heap: Vec<(i32, f32)>,
}

fn min_heap_comp(a: &(i32, f32), b: &(i32, f32)) -> bool {
    a.1 > b.1
}

impl MinHeap {

    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self { heap: (0..capacity as i32).map(|i| (i, 0.0)).collect() }
    }

    pub fn insert(&mut self, element: (i32, f32)) {
        self.heap.push(element);
        push_heap(&mut self.heap, &min_heap_comp);
    }

    pub fn pop(&mut self) -> (i32, f32) {
        pop_heap(&mut self.heap, &min_heap_comp);
        self.heap.pop().expect("MinHeap::pop on an empty heap")
    }
}
