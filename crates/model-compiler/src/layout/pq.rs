//! A PQ-tree (Booth & Lueker, JCSS 1976): the canonical representation of
//! EVERY permutation of a ground set under which a family of subsets is
//! simultaneously consecutive.
//!
//! WHY A TREE AND NOT A SOLVER. A one-shot C1P decider answers "yes, and here
//! is an ordering"; the fire path needs "yes, and here is the whole set of
//! orderings", because the ordering it wants is the feasible one CLOSEST TO
//! LAST FIRE — re-binding a kernel node's pointers costs ~0.11 us and
//! re-recording the graph costs everything, so pointer churn is the thing
//! worth minimising and a single witness ordering cannot be asked about it
//! (`tart/evidence/layout_planning.md`, the tile-alignment section). So the
//! tree ships whole inside [`LayoutOrder`](crate::LayoutOrder), and
//! [`admits`](PqTree::admits) is the question the stability pick will ask it.
//!
//! # The shape
//!
//! Three node kinds, and the whole algebra is in them:
//!
//! - a **leaf** is one class;
//! - a **P-node**'s children may be permuted arbitrarily — `k!` orders;
//! - a **Q-node**'s children have a fixed order, up to reversal — 2 orders.
//!
//! The frontier — the leaves, left to right — is one feasible ordering; the
//! orderings the tree admits are exactly the frontiers reachable by permuting
//! P-nodes and reversing Q-nodes. A single P-node over every leaf is "nothing
//! is constrained yet"; a tree that has gone NULL is "no ordering works", and
//! here that is a `false` out of [`reduce`](PqTree::reduce) rather than a null
//! object, since the caller's answer to it is to withdraw the constraint and
//! keep the tree it had.
//!
//! # The reduction
//!
//! [`reduce`](PqTree::reduce) is Booth-Lueker's REDUCE: refine the tree so
//! that the leaves of `set` become consecutive in every frontier that
//! survives. It runs as a recursive descent over the PERTINENT SUBTREE — the
//! subtree rooted at the lowest node containing all of `set` — classifying
//! each node as EMPTY (no leaf of `set`), FULL (nothing but) or PARTIAL, and
//! rewriting it by the classical templates:
//!
//! | template | node | where | what it does |
//! |---|---|---|---|
//! | L1 | leaf | anywhere | full iff its class is in the set |
//! | P1 | P | anywhere | wholly empty or wholly full: nothing to do |
//! | P2 | P | pertinent root | group the full children under a new P-node |
//! | P3 | P | below the root | becomes a Q-node: `[P(empty), P(full)]` |
//! | P4 | P | pertinent root | one partial child; the full children join its full end |
//! | P5 | P | below the root | one partial child; it absorbs the empties and the fulls and becomes this node |
//! | P6 | P | pertinent root | two partial children, merged into one Q-node full-end to full-end |
//! | Q1 | Q | anywhere | wholly empty or wholly full: nothing to do |
//! | Q2 | Q | below the root | children read `E* P? F*` up to reversal; the partial is spliced in |
//! | Q3 | Q | pertinent root | children read `E* P? F* P? E*`; both partials are spliced in |
//!
//! A P-node below the pertinent root with two partial children, or a Q-node
//! whose children do not read as the pattern above, is the failure: those are
//! exactly the shapes Tucker's forbidden submatrices name.
//!
//! **THE REDUCTION IS ATOMIC.** Booth-Lueker mutates in place and leaves a
//! wreck behind on failure; the caller here needs the tree it had, because the
//! layout pass's answer to an infeasible constraint is to withdraw it and send
//! its consumer to the fallback menu rather than to refuse the plan. So
//! `reduce` snapshots, and a `false` return means the tree is untouched.
//!
//! **CORRECTNESS OVER ASYMPTOTICS.** The published algorithm is linear only
//! with the bubble pass, parent pointers, pertinent-child counts and a
//! template dispatch that never rescans a node; this is the same templates
//! written as a plain recursion over an index arena, which is `O(n)` per
//! reduction with a small quadratic in the canonicalisation. The ground set is
//! the plan's CLASS list — six, today, on the biggest catalog SKU — and the
//! whole thing runs once per load.

use std::fmt::{self, Debug, Formatter};

/// One class, as a leaf of the tree.
///
/// A `u8` BECAUSE THE FRONTIER IS THE FIRE PATH'S. `class_order` hands the
/// driver a `Vec<u8>` per fire (design §3), so 256 classes is the ceiling the
/// whole seam is spelled at; [`crate::layout`] declines to seriate a plan past
/// it rather than truncating one.
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
/// `Partial(split)` carries the INVARIANT THE WHOLE RECURSION RESTS ON: a
/// partial node is a Q-node whose first `split` children are empty and whose
/// remaining children are full. Every template that consumes a partial child
/// splices it in on that promise, and reversing a Q-node is free, so "the
/// fulls are at the tail" costs nothing to guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    Empty,
    Full,
    Partial(usize),
}

/// Every ordering of a class set under which a family of subsets is
/// simultaneously an interval.
///
/// Built by [`universe`](PqTree::universe) — which constrains nothing — and
/// narrowed one subset at a time by [`reduce`](PqTree::reduce). The
/// [`frontier`](PqTree::frontier) is the canonical member of the set, and
/// [`admits`](PqTree::admits) decides membership for any other.
///
/// CANONICAL AFTER EVERY REDUCTION, which is what makes `PartialEq` mean what
/// a reader expects and what makes the bake deterministic: children of a
/// P-node are stored in ascending order of their least leaf, a Q-node is
/// stored in whichever of its two orientations puts the smaller leaf first,
/// single-child nodes are collapsed, and a two-child Q-node — which admits
/// exactly what a two-child P-node admits — is stored as the P-node. Two trees
/// that admit the same orderings by the same structure are the same value.
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
    ///
    /// ONE MEMBER OF THE SET AND NOT THE SET, and v1's layout answer is this
    /// one — the stability pick that chooses a different member per fire is
    /// the documented later refinement (design §3).
    #[must_use]
    pub fn frontier(&self) -> &[Leaf] {
        &self.frontier
    }

    /// Narrow the tree to the orderings under which `set` is consecutive.
    ///
    /// `set` is ascending and duplicate-free. Answers `false` iff no such
    /// ordering exists — and then the tree is EXACTLY WHAT IT WAS, so the
    /// caller may withdraw the constraint and carry on (see the module docs).
    ///
    /// A set of fewer than two classes, or one holding every class, is not a
    /// constraint at all and is accepted without touching anything.
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
    ///
    /// THE QUESTION THE FIRE PATH WILL ASK. A stability pick proposes last
    /// fire's ordering and needs to know whether it is still feasible before
    /// it may keep the pointers it has; that is this, and it is the reason the
    /// tree ships rather than a witness (design §3).
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
    /// [`Fallback::Split`](crate::Fallback), and 1 exactly when `set` is an
    /// interval.
    ///
    /// Classes of `set` that `order` does not carry are not counted; a fire
    /// orders the classes it has.
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

    /// Is `set` an interval of `order`? The property the whole pass exists to
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
    /// node the reduction has to restructure, and the only one whose template
    /// may leave the fulls in the MIDDLE rather than at an end.
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

    /// Classify every child, descending only into the ones that are neither
    /// wholly in nor wholly out: an empty or a full subtree is already
    /// consecutive with itself and wants no rewriting.
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
                // P2. The fulls become one child, free to sit anywhere among
                // the empties; nothing above this node ever looks again.
                0 => {
                    let block = self.group(filled);
                    empty.push(block);
                    self.nodes[n] = Node::P(empty);
                }
                // P4. The one partial child already has its fulls at the tail,
                // so the node's own fulls join them there.
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
                // P6. Two partial children, joined full end to full end with
                // the node's own fulls between them — the only template that
                // builds a Q-node the reduction will never be able to reverse
                // apart again.
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
                // Three pertinent blocks and one line to lay them on: the
                // instance is not C1P.
                _ => return None,
            }
            return Some(Mark::Full);
        }

        match partial.len() {
            // P3. Below the pertinent root the fulls must reach an END of this
            // node's frontier, so the free permutation collapses to two blocks
            // in a fixed order — which is a Q-node.
            0 => {
                let head = self.group(empty);
                let tail = self.group(filled);
                self.nodes[n] = Node::Q(vec![head, tail]);
                Some(Mark::Partial(1))
            }
            // P5. The partial child absorbs the node: empties, then the
            // child's own empties and fulls, then the fulls.
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
            // Q3: `E* P? F* P? E*`. A Q-node's order is already fixed, so
            // unlike P6 there is nothing to choose — either the fulls sit in
            // one block with at most one partial neighbour on each side, or no
            // reversal of anything will make them consecutive.
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

        // Q2: `E* P? F*`, up to reversal. Reversing a Q-node's child list is
        // free; reversing it is how the fulls are brought to the tail, which
        // is the invariant the parent's template is owed.
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

    /// One child standing for a block of them: the child itself when there is
    /// one, a fresh P-node when there are several. The block permutes freely
    /// inside itself and moves as a unit outside, which is exactly what "the
    /// empty children" and "the full children" mean to every template above.
    fn group(&mut self, mut block: Vec<usize>) -> usize {
        if block.len() == 1 {
            return block.pop().expect("just measured");
        }
        self.push(Node::P(block))
    }

    /// Take a partial node's children, leaving it behind for
    /// [`canonicalise`](PqTree::canonicalise) to drop. A partial node is
    /// always a Q-node — that is [`Mark::Partial`]'s promise — and it is
    /// always dissolved into its parent rather than kept, because the parent's
    /// template has just fixed the order it was free in.
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

    /// Rebuild the arena in frontier order, applying the normal form the type
    /// docs promise, and recompute the frontier while doing it. Also the
    /// garbage collector: the nodes a reduction dissolved are simply not
    /// reachable from the root and do not come along.
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
                // A two-child Q-node admits both orders of its children and so
                // does a two-child P-node. One spelling, so that equal trees
                // compare equal.
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

/// `E* P? F*` — the shape a Q-node BELOW the pertinent root must read, so that
/// its fulls reach the end that faces its parent's. Answers how many empties
/// lead, and which child is the partial one.
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

/// The tree as its frontier and its shape: `[0 (1 3) 2]` is a P-node over leaf
/// 0, a Q-node over 1 and 3, and leaf 2. Round brackets are Q-nodes (order
/// fixed, reversal free), square brackets are P-nodes (order free).
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

    /// The tree's shape, as [`Debug`] spells it: `[]` a P-node, `()` a
    /// Q-node. Asserting the SHAPE and not only the frontier is what makes
    /// these tests about the templates rather than about one witness ordering
    /// — a brute force that happened to return the same permutation would not
    /// produce the same brackets.
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

    /// The tree's feasible set, checked against the definition it is supposed
    /// to represent: an ordering is admitted iff every constraint inserted so
    /// far is an interval of it. This is the property a PQ-tree exists to
    /// have, and the only test that can tell a real one from a lucky one.
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
    fn a_free_tree_is_one_p_node_and_admits_everything() {
        let tree = PqTree::universe(4);
        assert_eq!(shape(&tree), "[0 1 2 3]");
        assert_eq!(tree.frontier(), [0, 1, 2, 3]);
        set_is_exactly(&tree, &[]);
    }

    #[test]
    fn a_set_of_one_class_or_of_every_class_constrains_nothing() {
        let mut tree = PqTree::universe(4);
        assert!(tree.reduce(&[2]));
        assert!(tree.reduce(&[0, 1, 2, 3]));
        assert_eq!(shape(&tree), "[0 1 2 3]");
        set_is_exactly(&tree, &[]);
    }

    #[test]
    fn template_p2_groups_the_full_children_under_the_root() {
        // The pertinent root is the whole tree, no child is partial: the two
        // full leaves become one P-node, free to sit anywhere among the rest.
        let mut tree = PqTree::universe(4);
        assert!(tree.reduce(&[1, 3]));
        assert_eq!(shape(&tree), "[0 [1 3] 2]");
        assert_eq!(tree.frontier(), [0, 1, 3, 2]);
        set_is_exactly(&tree, &[&[1, 3]]);
    }

    #[test]
    fn the_two_facts_that_lexicographic_order_cannot_seat() {
        // The worked example: four classes over `qo_one` x `masked`, one
        // consumer windowed on each fact. Ascending class order fails —
        // {01, 11} is 1 and 3 with 2 between them — and the Gray-coded order
        // 00, 01, 11, 10 seats both. P2 then P4, and the P4 is what turns the
        // grouped pair into a Q-node.
        let mut tree = PqTree::universe(4);
        assert!(tree.reduce(&[1, 3]), "the qo_one classes");
        assert!(tree.reduce(&[2, 3]), "the masked classes");
        assert_eq!(shape(&tree), "[0 (1 3 2)]");
        assert_eq!(tree.frontier(), [0, 1, 3, 2]);

        assert!(!PqTree::is_interval(&[0, 1, 2, 3], &[1, 3]));
        assert!(PqTree::is_interval(tree.frontier(), &[1, 3]));
        assert!(PqTree::is_interval(tree.frontier(), &[2, 3]));
        assert!(!tree.admits(&[0, 1, 2, 3]));
        set_is_exactly(&tree, &[&[1, 3], &[2, 3]]);
    }

    #[test]
    fn template_p3_then_p5_below_the_pertinent_root() {
        // Three nested constraints. The third's pertinent root is the whole
        // tree, so the inner P-node is reduced BELOW it: its own partial child
        // fires P3 (a P-node with no partial child becomes a two-block
        // Q-node), and the P-node above it fires P5 (the partial child
        // absorbs its parent's empties and fulls).
        let mut tree = PqTree::universe(8);
        assert!(tree.reduce(&[0, 1, 2, 3]));
        assert!(tree.reduce(&[0, 1]));
        assert_eq!(shape(&tree), "[[[0 1] 2 3] 4 5 6 7]");
        assert!(tree.reduce(&[0, 2, 4]));
        assert_eq!(shape(&tree), "[(3 1 0 2 4) 5 6 7]");
        assert_eq!(tree.frontier(), [3, 1, 0, 2, 4, 5, 6, 7]);
        for set in [&[0, 1, 2, 3][..], &[0, 1], &[0, 2, 4]] {
            assert!(PqTree::is_interval(tree.frontier(), set), "{set:?}");
        }
    }

    #[test]
    fn template_p6_joins_two_partial_children_full_end_to_full_end() {
        let mut tree = PqTree::universe(6);
        assert!(tree.reduce(&[0, 1]));
        assert!(tree.reduce(&[2, 3]));
        assert_eq!(shape(&tree), "[[0 1] [2 3] 4 5]");
        // Two partial children at the pertinent root, and only one place for
        // the fulls to meet: 1 0 | 2 3.
        assert!(tree.reduce(&[0, 2]));
        assert_eq!(shape(&tree), "[(1 0 2 3) 4 5]");
        set_is_exactly(&tree, &[&[0, 1], &[2, 3], &[0, 2]]);
    }

    #[test]
    fn template_q3_splices_a_partial_child_in_at_each_end() {
        // Build a Q-node of three blocks, then constrain a set that straddles
        // it: the pertinent root is the Q-node, its middle block is full and
        // the two outer blocks are partial — `E* P? F* P? E*` with both
        // partials present, which is the template's whole point.
        let mut tree = PqTree::universe(8);
        assert!(tree.reduce(&[0, 1]));
        assert!(tree.reduce(&[2, 3]));
        assert!(tree.reduce(&[4, 5]));
        assert!(tree.reduce(&[0, 1, 2, 3]));
        assert!(tree.reduce(&[2, 3, 4, 5]));
        assert_eq!(shape(&tree), "[([0 1] [2 3] [4 5]) 6 7]");
        assert!(tree.reduce(&[1, 2, 3, 4]));
        assert_eq!(shape(&tree), "[(0 1 [2 3] 4 5) 6 7]");
        assert_eq!(tree.frontier(), [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn template_q2_reduces_a_q_node_below_the_pertinent_root() {
        let mut tree = PqTree::universe(8);
        for set in [&[0, 1][..], &[2, 3], &[4, 5], &[0, 1, 2, 3], &[2, 3, 4, 5]] {
            assert!(tree.reduce(set));
        }
        assert!(tree.reduce(&[1, 2, 3, 4]));
        assert_eq!(shape(&tree), "[(0 1 [2 3] 4 5) 6 7]");
        // Now a set reaching OUT of the Q-node: the pertinent root is the root
        // P-node, so the Q-node is reduced as a non-root — `E* P? F*` — and
        // has to bring its fulls to the end that faces leaf 6.
        assert!(tree.reduce(&[3, 4, 5, 6]));
        assert_eq!(shape(&tree), "[(0 1 2 3 4 5 6) 7]");
        for set in [
            &[0, 1][..],
            &[2, 3],
            &[4, 5],
            &[0, 1, 2, 3],
            &[2, 3, 4, 5],
            &[1, 2, 3, 4],
            &[3, 4, 5, 6],
        ] {
            assert!(PqTree::is_interval(tree.frontier(), set), "{set:?}");
        }
    }

    #[test]
    fn a_q_node_whose_fulls_are_not_at_an_end_is_the_failure() {
        // {A,B}, {B,C}, {C,A}: three sets pairwise overlapping with no common
        // interval order — the smallest non-C1P instance there is, and the
        // shape Tucker's characterisation names.
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
    fn a_reduction_that_fails_deep_still_leaves_the_tree_alone() {
        // The failure is inside a subtree several templates down, so the
        // atomicity claim is about an undo of real work and not of nothing.
        let mut tree = PqTree::universe(6);
        for set in [&[0, 1][..], &[2, 3], &[0, 1, 2, 3], &[1, 2]] {
            assert!(tree.reduce(set));
        }
        let before = tree.clone();
        assert!(!tree.reduce(&[0, 2]));
        assert_eq!(tree, before);
    }

    #[test]
    fn runs_counts_the_launches_a_split_would_take() {
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[1, 2]), 1);
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[0, 2]), 2);
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[0, 2, 3]), 2);
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[0, 3]), 2);
        assert_eq!(PqTree::runs(&[3, 1, 0, 2], &[0, 1]), 1);
        assert_eq!(PqTree::runs(&[0, 1, 2, 3], &[]), 0);
        // A class the fire does not carry is not a break either: what is
        // asked is whether the rows PRESENT are contiguous.
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

    #[test]
    fn one_class_and_no_classes_are_both_trees() {
        let tree = PqTree::universe(1);
        assert_eq!(shape(&tree), "0");
        assert_eq!(tree.frontier(), [0]);
        assert!(tree.admits(&[0]));

        let mut none = PqTree::universe(0);
        assert!(none.frontier().is_empty());
        assert!(none.reduce(&[]));
    }

    #[test]
    fn the_same_constraints_build_the_same_tree() {
        let sets: [&[Leaf]; 4] = [&[0, 1], &[2, 3], &[0, 1, 2, 3], &[1, 2]];
        let mut once = PqTree::universe(6);
        let mut twice = PqTree::universe(6);
        for set in sets {
            assert_eq!(once.reduce(set), twice.reduce(set));
        }
        assert_eq!(once, twice);
        assert_eq!(once.frontier(), twice.frontier());
    }
}
