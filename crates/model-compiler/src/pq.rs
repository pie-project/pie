//! A PQ-tree (Booth & Lueker, JCSS 1976): canonical representation of every
//! permutation under which a family of subsets is simultaneously consecutive.

use std::fmt::{self, Debug, Formatter};

/// One class, as a leaf of the tree.
///
/// A `u8`: `class_order` hands the engine a `Vec<u8>` per fire, so 256
/// classes is the ceiling; [`crate::layout`] declines to seriate a plan
/// past it rather than truncating one.
pub type Leaf = u8;

/// A node of the tree. Children are indices into [`PqTree::nodes`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    /// One class.
    Leaf(Leaf),
    /// Children in any order.
    P(Vec<usize>),
    /// Children in this order or its reverse, and no other.
    Q(Vec<usize>),
}

impl Node {
    fn children(&self) -> &[usize] {
        match self {
            Node::Leaf(_) => &[],
            Node::P(kids) | Node::Q(kids) => kids,
        }
    }
}

/// What one node of the pertinent subtree turned out to be, after its own
/// reduction.
///
/// `Partial(split)`: a partial node is a Q-node whose first `split`
/// children are empty and the rest are full. Every template that consumes
/// a partial child splices it in on that promise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    Empty,
    Full,
    Partial(usize),
}

/// Every ordering of a class set under which a family of subsets is
/// simultaneously an interval.
///
/// Built by [`universe`](PqTree::universe) (constrains nothing) and
/// narrowed by [`reduce`](PqTree::reduce). [`frontier`](PqTree::frontier)
/// is the canonical member; [`admits`](PqTree::admits) decides membership
/// for any other.
///
/// Canonical after every reduction: P-node children sort by least leaf, a
/// Q-node picks the orientation with the smaller leaf first, single-child
/// nodes collapse, and a two-child Q-node is stored as a P-node — so two
/// trees admitting the same orderings compare equal.
#[derive(Clone, PartialEq, Eq)]
pub struct PqTree {
    nodes: Vec<Node>,
    root: usize,
    frontier: Vec<Leaf>,
}

impl PqTree {
    /// The tree that constrains nothing: one P-node over `leaves` classes, so
    /// every one of the `leaves!` orderings is feasible.
    ///
    /// # Panics
    ///
    /// If `leaves` exceeds 256, which is what a [`Leaf`] can name.
    #[must_use]
    pub fn universe(leaves: usize) -> PqTree {
        assert!(
            leaves <= usize::from(u8::MAX) + 1,
            "a PQ-tree over {leaves} classes; a leaf is a u8",
        );
        let mut nodes: Vec<Node> = (0..leaves).map(|l| Node::Leaf(l as Leaf)).collect();
        let root = nodes.len();
        nodes.push(Node::P((0..leaves).collect()));
        let mut tree = PqTree {
            nodes,
            root,
            frontier: Vec::new(),
        };
        tree.canonicalise();
        tree
    }

    /// How many classes the tree orders.
    #[must_use]
    pub fn leaves(&self) -> usize {
        self.frontier.len()
    }

    /// The canonical feasible ordering: the leaves, left to right.
    #[must_use]
    pub fn frontier(&self) -> &[Leaf] {
        &self.frontier
    }

    /// Narrow the tree to the orderings under which `set` is consecutive.
    ///
    /// `set` is ascending and duplicate-free. Answers `false` iff no such
    /// ordering exists, leaving the tree untouched.
    ///
    /// A set of fewer than two classes, or one holding every class, is not
    /// a constraint and is accepted without touching anything.
    pub fn reduce(&mut self, set: &[Leaf]) -> bool {
        if set.len() < 2 || set.len() >= self.frontier.len() {
            return true;
        }

        let mut full = vec![0usize; self.nodes.len()];
        let mut total = vec![0usize; self.nodes.len()];
        self.count(self.root, set, &mut full, &mut total);
        if full[self.root] != set.len() {
            // A class this tree does not have. Nothing to reduce against.
            return false;
        }

        let snapshot = self.clone();
        let pertinent = self.pertinent_root(&full, set.len());
        if self.reduce_at(pertinent, true, &full, &total).is_some() {
            self.canonicalise();
            true
        } else {
            *self = snapshot;
            false
        }
    }

    /// Is `order` one of the orderings this tree admits?
    #[must_use]
    pub fn admits(&self, order: &[Leaf]) -> bool {
        if order.len() != self.frontier.len() {
            return false;
        }
        let mut seen = vec![false; 256];
        for &leaf in order {
            if std::mem::replace(&mut seen[usize::from(leaf)], true) {
                return false;
            }
        }
        self.admits_at(self.root, order)
    }

    /// How many maximal runs `set` breaks into under `order` — the `r` of
    /// [`Fallback::Split`](crate::Fallback), 1 exactly when `set` is an
    /// interval. Classes of `set` not in `order` aren't counted.
    #[must_use]
    pub fn runs(order: &[Leaf], set: &[Leaf]) -> u32 {
        let mut runs = 0;
        let mut inside = false;
        for leaf in order {
            let hit = set.contains(leaf);
            if hit && !inside {
                runs += 1;
            }
            inside = hit;
        }
        runs
    }

    /// Is `set` an interval of `order`? The property the pass exists to
    /// obtain: a windowed consumer whose classes are one run is one kernel
    /// over pointer plus extent.
    #[must_use]
    pub fn is_interval(order: &[Leaf], set: &[Leaf]) -> bool {
        PqTree::runs(order, set) <= 1
    }

    // -- the reduction -----------------------------------------------------

    /// Post-order: how many leaves each subtree has, and how many of them are
    /// in `set`.
    fn count(&self, n: usize, set: &[Leaf], full: &mut [usize], total: &mut [usize]) {
        match &self.nodes[n] {
            Node::Leaf(l) => {
                total[n] = 1;
                full[n] = usize::from(set.contains(l));
            }
            Node::P(kids) | Node::Q(kids) => {
                let (mut f, mut t) = (0, 0);
                for &c in kids {
                    self.count(c, set, full, total);
                    f += full[c];
                    t += total[c];
                }
                full[n] = f;
                total[n] = t;
            }
        }
    }

    /// The lowest node whose subtree holds every leaf of the set — the only
    /// node the reduction has to restructure.
    fn pertinent_root(&self, full: &[usize], want: usize) -> usize {
        let mut n = self.root;
        loop {
            let kids = self.nodes[n].children();
            let mut hits = 0;
            let mut only = n;
            for &c in kids {
                if full[c] > 0 {
                    hits += 1;
                    only = c;
                }
            }
            if hits == 1 && full[only] == want {
                n = only;
            } else {
                return n;
            }
        }
    }

    fn reduce_at(&mut self, n: usize, root: bool, full: &[usize], total: &[usize]) -> Option<Mark> {
        match self.nodes[n].clone() {
            Node::Leaf(_) => Some(if full[n] == 1 { Mark::Full } else { Mark::Empty }),
            Node::P(kids) => self.reduce_p(n, &kids, root, full, total),
            Node::Q(kids) => self.reduce_q(n, kids, root, full, total),
        }
    }

    /// Classify every child, descending only into ones neither wholly in
    /// nor wholly out (already consecutive, no rewriting needed).
    fn marks(&mut self, kids: &[usize], full: &[usize], total: &[usize]) -> Option<Vec<Mark>> {
        let mut marks = Vec::with_capacity(kids.len());
        for &c in kids {
            let mark = if full[c] == 0 {
                Mark::Empty
            } else if full[c] == total[c] {
                Mark::Full
            } else {
                self.reduce_at(c, false, full, total)?
            };
            marks.push(mark);
        }
        Some(marks)
    }

    /// Templates P1 through P6.
    fn reduce_p(
        &mut self,
        n: usize,
        kids: &[usize],
        root: bool,
        full: &[usize],
        total: &[usize],
    ) -> Option<Mark> {
        let marks = self.marks(kids, full, total)?;
        let mut empty: Vec<usize> = Vec::new();
        let mut filled: Vec<usize> = Vec::new();
        let mut partial: Vec<(usize, usize)> = Vec::new();
        for (&c, &mark) in kids.iter().zip(&marks) {
            match mark {
                Mark::Empty => empty.push(c),
                Mark::Full => filled.push(c),
                Mark::Partial(split) => partial.push((c, split)),
            }
        }

        // P1, both readings of it.
        if partial.is_empty() {
            if filled.is_empty() {
                return Some(Mark::Empty);
            }
            if empty.is_empty() {
                return Some(Mark::Full);
            }
        }

        if root {
            match partial.len() {
                // P2: the fulls become one child, free to sit among the empties.
                0 => {
                    let block = self.group(filled);
                    empty.push(block);
                    self.nodes[n] = Node::P(empty);
                }
                // P4: the partial child's fulls are at the tail; the node's own fulls join them.
                1 => {
                    let (q, _) = partial[0];
                    if !filled.is_empty() {
                        let block = self.group(filled);
                        match &mut self.nodes[q] {
                            Node::Q(kids) => kids.push(block),
                            Node::Leaf(_) | Node::P(_) => return None,
                        }
                    }
                    empty.push(q);
                    self.nodes[n] = Node::P(empty);
                }
                // P6: two partial children joined full end to full end, node's fulls between.
                2 => {
                    let mut merged = self.detach(partial[0].0)?;
                    if !filled.is_empty() {
                        let block = self.group(filled);
                        merged.push(block);
                    }
                    let mut far = self.detach(partial[1].0)?;
                    far.reverse();
                    merged.extend(far);
                    let joined = self.push(Node::Q(merged));
                    empty.push(joined);
                    self.nodes[n] = Node::P(empty);
                }
                // Three pertinent blocks, one line to lay them on: not C1P.
                _ => return None,
            }
            return Some(Mark::Full);
        }

        match partial.len() {
            // P3: fulls must reach an end, so the free permutation collapses
            // to a two-block Q-node.
            0 => {
                let head = self.group(empty);
                let tail = self.group(filled);
                self.nodes[n] = Node::Q(vec![head, tail]);
                Some(Mark::Partial(1))
            }
            // P5: the partial child absorbs the node's empties then fulls.
            1 => {
                let (q, split) = partial[0];
                let mut out = Vec::new();
                let mut at = 0;
                if !empty.is_empty() {
                    let head = self.group(empty);
                    out.push(head);
                    at += 1;
                }
                out.extend(self.detach(q)?);
                at += split;
                if !filled.is_empty() {
                    let tail = self.group(filled);
                    out.push(tail);
                }
                self.nodes[n] = Node::Q(out);
                Some(Mark::Partial(at))
            }
            _ => None,
        }
    }

    /// Templates Q1 through Q3.
    fn reduce_q(
        &mut self,
        n: usize,
        kids: Vec<usize>,
        root: bool,
        full: &[usize],
        total: &[usize],
    ) -> Option<Mark> {
        let marks = self.marks(&kids, full, total)?;

        // Q1.
        if marks.iter().all(|m| *m == Mark::Empty) {
            return Some(Mark::Empty);
        }
        if marks.iter().all(|m| *m == Mark::Full) {
            return Some(Mark::Full);
        }

        if root {
            // Q3: `E* P? F* P? E*`; fulls sit in one block with at most one
            // partial neighbour each side, or nothing works.
            let RootScan {
                head,
                first,
                block,
                second,
            } = scan_root(&marks)?;
            let mut out: Vec<usize> = kids[..head].to_vec();
            if let Some(at) = first {
                out.extend(self.detach(kids[at])?);
            }
            out.extend_from_slice(&kids[block.0..block.1]);
            if let Some(at) = second {
                let mut tail = self.detach(kids[at])?;
                tail.reverse();
                out.extend(tail);
                out.extend_from_slice(&kids[at + 1..]);
            } else {
                out.extend_from_slice(&kids[block.1..]);
            }
            self.nodes[n] = Node::Q(out);
            return Some(Mark::Full);
        }

        // Q2: `E* P? F*` up to reversal; reversing brings the fulls to the tail.
        let (mut kids, mut marks) = (kids, marks);
        let parsed = match scan_side(&marks) {
            Some(parsed) => parsed,
            None => {
                kids.reverse();
                marks.reverse();
                scan_side(&marks)?
            }
        };
        let (head, at) = parsed;
        let mut out: Vec<usize> = kids[..head].to_vec();
        let mut split = head;
        match at {
            Some(at) => {
                let Mark::Partial(inner) = marks[at] else {
                    return None;
                };
                out.extend(self.detach(kids[at])?);
                split += inner;
                out.extend_from_slice(&kids[at + 1..]);
            }
            None => out.extend_from_slice(&kids[head..]),
        }
        self.nodes[n] = Node::Q(out);
        Some(Mark::Partial(split))
    }

    /// One child standing for a block of them: itself when there is one, a
    /// fresh P-node when there are several. Moves as a unit outside,
    /// permutes freely inside.
    fn group(&mut self, mut block: Vec<usize>) -> usize {
        if block.len() == 1 {
            return block.pop().expect("just measured");
        }
        self.push(Node::P(block))
    }

    /// Take a partial node's children, leaving it for
    /// [`canonicalise`](PqTree::canonicalise) to drop. Always a Q-node
    /// ([`Mark::Partial`]'s promise), dissolved into the parent.
    fn detach(&mut self, n: usize) -> Option<Vec<usize>> {
        match &mut self.nodes[n] {
            Node::Q(kids) => Some(std::mem::take(kids)),
            Node::Leaf(_) | Node::P(_) => None,
        }
    }

    fn push(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    // -- the canonical form ------------------------------------------------

    /// Rebuild the arena in frontier order (the normal form), and recompute
    /// the frontier. Also the garbage collector: dissolved nodes aren't reachable.
    fn canonicalise(&mut self) {
        let mut fresh = Vec::with_capacity(self.nodes.len());
        let mut frontier = Vec::with_capacity(self.frontier.len());
        let root = self.rebuild(self.root, &mut fresh, &mut frontier);
        self.nodes = fresh;
        self.root = root;
        self.frontier = frontier;
    }

    fn rebuild(&self, n: usize, fresh: &mut Vec<Node>, frontier: &mut Vec<Leaf>) -> usize {
        match &self.nodes[n] {
            Node::Leaf(l) => {
                frontier.push(*l);
                fresh.push(Node::Leaf(*l));
                fresh.len() - 1
            }
            Node::P(kids) => {
                let mut order = kids.clone();
                order.sort_by_key(|&c| self.least(c));
                if order.len() == 1 {
                    return self.rebuild(order[0], fresh, frontier);
                }
                let ids: Vec<usize> = order
                    .iter()
                    .map(|&c| self.rebuild(c, fresh, frontier))
                    .collect();
                fresh.push(Node::P(ids));
                fresh.len() - 1
            }
            Node::Q(kids) => {
                if kids.len() == 1 {
                    return self.rebuild(kids[0], fresh, frontier);
                }
                let mut order = kids.clone();
                if self.least(order[order.len() - 1]) < self.least(order[0]) {
                    order.reverse();
                }
                let ids: Vec<usize> = order
                    .iter()
                    .map(|&c| self.rebuild(c, fresh, frontier))
                    .collect();
                // A two-child Q-node admits the same orders as a two-child
                // P-node; stored as P so equal trees compare equal.
                fresh.push(if ids.len() == 2 {
                    Node::P(ids)
                } else {
                    Node::Q(ids)
                });
                fresh.len() - 1
            }
        }
    }

    fn least(&self, n: usize) -> Leaf {
        match &self.nodes[n] {
            Node::Leaf(l) => *l,
            Node::P(kids) | Node::Q(kids) => {
                kids.iter().map(|&c| self.least(c)).min().unwrap_or(Leaf::MAX)
            }
        }
    }

    fn size(&self, n: usize) -> usize {
        match &self.nodes[n] {
            Node::Leaf(_) => 1,
            Node::P(kids) | Node::Q(kids) => kids.iter().map(|&c| self.size(c)).sum(),
        }
    }

    fn owns(&self, n: usize, leaf: Leaf) -> bool {
        match &self.nodes[n] {
            Node::Leaf(l) => *l == leaf,
            Node::P(kids) | Node::Q(kids) => kids.iter().any(|&c| self.owns(c, leaf)),
        }
    }

    fn admits_at(&self, n: usize, order: &[Leaf]) -> bool {
        match &self.nodes[n] {
            Node::Leaf(l) => order == [*l],
            Node::P(kids) => {
                let mut used = vec![false; kids.len()];
                let mut at = 0;
                while at < order.len() {
                    let head = order[at];
                    let Some(i) = (0..kids.len()).find(|&i| !used[i] && self.owns(kids[i], head))
                    else {
                        return false;
                    };
                    let size = self.size(kids[i]);
                    if at + size > order.len() || !self.admits_at(kids[i], &order[at..at + size]) {
                        return false;
                    }
                    used[i] = true;
                    at += size;
                }
                used.iter().all(|u| *u)
            }
            Node::Q(kids) => {
                if self.admits_seq(kids, order) {
                    return true;
                }
                let mut reversed = kids.clone();
                reversed.reverse();
                self.admits_seq(&reversed, order)
            }
        }
    }

    fn admits_seq(&self, kids: &[usize], order: &[Leaf]) -> bool {
        let mut at = 0;
        for &c in kids {
            let size = self.size(c);
            if at + size > order.len() || !self.admits_at(c, &order[at..at + size]) {
                return false;
            }
            at += size;
        }
        at == order.len()
    }
}

/// `E* P? F*` — the shape a Q-node below the pertinent root must read.
/// Answers how many empties lead, and which child is partial.
fn scan_side(marks: &[Mark]) -> Option<(usize, Option<usize>)> {
    let mut at = 0;
    while at < marks.len() && marks[at] == Mark::Empty {
        at += 1;
    }
    let head = at;
    let mut partial = None;
    if at < marks.len() && matches!(marks[at], Mark::Partial(_)) {
        partial = Some(at);
        at += 1;
    }
    while at < marks.len() && marks[at] == Mark::Full {
        at += 1;
    }
    (at == marks.len()).then_some((head, partial))
}

/// What [`scan_root`] read off a pertinent root's children.
struct RootScan {
    /// How many empty children lead.
    head: usize,
    /// The partial child on the empty side of the full block, if there is one.
    first: Option<usize>,
    /// The half-open span of the full children.
    block: (usize, usize),
    /// The partial child on the far side of it, if there is one.
    second: Option<usize>,
}

/// `E* P? F* P? E*` — the shape a Q-node AT the pertinent root must read,
/// where the fulls may sit in the middle.
fn scan_root(marks: &[Mark]) -> Option<RootScan> {
    let mut at = 0;
    while at < marks.len() && marks[at] == Mark::Empty {
        at += 1;
    }
    let head = at;
    let mut first = None;
    if at < marks.len() && matches!(marks[at], Mark::Partial(_)) {
        first = Some(at);
        at += 1;
    }
    let block = at;
    while at < marks.len() && marks[at] == Mark::Full {
        at += 1;
    }
    let span = (block, at);
    let mut second = None;
    if at < marks.len() && matches!(marks[at], Mark::Partial(_)) {
        second = Some(at);
        at += 1;
    }
    while at < marks.len() && marks[at] == Mark::Empty {
        at += 1;
    }
    (at == marks.len()).then_some(RootScan {
        head,
        first,
        block: span,
        second,
    })
}

/// The tree as its frontier and its shape: `[0 (1 3) 2]` is a P-node over
/// leaf 0, a Q-node over 1 and 3, and leaf 2. `()` are Q-nodes (order fixed,
/// reversal free), `[]` are P-nodes (order free).
impl Debug for PqTree {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.say(self.root, f)
    }
}

impl PqTree {
    fn say(&self, n: usize, f: &mut Formatter<'_>) -> fmt::Result {
        let (kids, open, close) = match &self.nodes[n] {
            Node::Leaf(l) => return write!(f, "{l}"),
            Node::P(kids) => (kids, '[', ']'),
            Node::Q(kids) => (kids, '(', ')'),
        };
        write!(f, "{open}")?;
        for (i, &c) in kids.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            self.say(c, f)?;
        }
        write!(f, "{close}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tree shape via [`Debug`]: `[]` P-node, `()` Q-node. Asserting shape
    // (not just frontier) tests the templates, not one witness ordering.
    fn shape(tree: &PqTree) -> String {
        format!("{tree:?}")
    }

    /// Every permutation of `0..n`, for the exhaustive feasible-set checks.
    fn permutations(n: usize) -> Vec<Vec<Leaf>> {
        let mut out = Vec::new();
        let mut order: Vec<Leaf> = (0..n as Leaf).collect();
        fn walk(order: &mut Vec<Leaf>, at: usize, out: &mut Vec<Vec<Leaf>>) {
            if at == order.len() {
                out.push(order.clone());
                return;
            }
            for i in at..order.len() {
                order.swap(at, i);
                walk(order, at + 1, out);
                order.swap(at, i);
            }
        }
        walk(&mut order, 0, &mut out);
        out
    }

    // The tree's feasible set, checked against its definition: admitted iff
    // every inserted constraint is an interval of it.
    fn set_is_exactly(tree: &PqTree, sets: &[&[Leaf]]) {
        for order in permutations(tree.leaves()) {
            let want = sets.iter().all(|s| PqTree::is_interval(&order, s));
            assert_eq!(
                tree.admits(&order),
                want,
                "{order:?} against {}",
                shape(tree),
            );
        }
    }

    #[test]
    fn a_q_node_whose_fulls_are_not_at_an_end_is_the_failure() {
        // {A,B},{B,C},{C,A}: pairwise overlapping with no common interval
        // order — the smallest non-C1P instance.
        let mut tree = PqTree::universe(3);
        assert!(tree.reduce(&[0, 1]));
        assert!(tree.reduce(&[0, 2]));
        assert_eq!(shape(&tree), "(1 0 2)");
        let before = tree.clone();
        assert!(!tree.reduce(&[1, 2]));
        assert_eq!(tree, before, "a failed reduction leaves the tree alone");
        set_is_exactly(&tree, &[&[0, 1], &[0, 2]]);
    }

    #[test]
    fn runs_counts_the_launches_a_split_would_take() {
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[1, 2]), 1);
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[0, 2]), 2);
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[0, 2, 3]), 2);
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[0, 3]), 2);
        assert_eq!(PqTree::runs(&[3, 1, 0, 2], &[0, 1]), 1);
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[]), 0);
        // A class the fire doesn't carry isn't a break: only present rows
        // are checked for contiguity.
        assert_eq!(PqTree::runs(&[0, 2], &[0, 1, 2]), 1);
        assert_eq!(PqTree::runs(&[0, 1, 2], &[0, 2]), 2);
    }

    #[test]
    fn admits_refuses_anything_that_is_not_a_permutation_of_the_leaves() {
        let tree = PqTree::universe(3);
        assert!(tree.admits(&[2, 0, 1]));
        assert!(!tree.admits(&[0, 1]));
        assert!(!tree.admits(&[0, 1, 1]));
        assert!(!tree.admits(&[0, 1, 2, 3]));
    }

}
