use core::ffi::c_void;
use core::fmt;

use model_exec::law::Refuse;

use crate::device::graph::Graph;
use crate::device::nodes::{self, Node, Param, Walked};
use crate::error::Result;

pub use model_exec::law::{At, Component, Law};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Topology {
    pub hash: u64,
    pub nodes: usize,
    pub links: usize,
}

impl Topology {
    #[must_use]
    pub fn of(walked: &Walked) -> Topology {
        let mut hash = FNV_OFFSET;
        for node in &walked.nodes {
            fold(&mut hash, &node.depth.to_le_bytes());
            fold(&mut hash, &node.kind.to_le_bytes());
            fold(&mut hash, node.symbol.as_bytes());
            fold(&mut hash, b"|");
        }
        let mut edges: Vec<(usize, u32, &str, usize, u32, &str)> = walked
            .links
            .iter()
            .filter_map(|(from, to)| {
                let a = walked.nodes.get(*from)?;
                let b = walked.nodes.get(*to)?;
                Some((
                    a.depth,
                    a.kind,
                    a.symbol.as_str(),
                    b.depth,
                    b.kind,
                    b.symbol.as_str(),
                ))
            })
            .collect();
        edges.sort_unstable();
        for (ad, ak, asym, bd, bk, bsym) in &edges {
            fold(&mut hash, &ad.to_le_bytes());
            fold(&mut hash, &ak.to_le_bytes());
            fold(&mut hash, asym.as_bytes());
            fold(&mut hash, b">");
            fold(&mut hash, &bd.to_le_bytes());
            fold(&mut hash, &bk.to_le_bytes());
            fold(&mut hash, bsym.as_bytes());
            fold(&mut hash, b"|");
        }
        Topology {
            hash,
            nodes: walked.nodes.len(),
            links: edges.len(),
        }
    }
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "#{:016x} ({} nodes, {} edges)",
            self.hash, self.nodes, self.links
        )
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fold(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ambiguous {
    pub depth: usize,
    pub symbol: String,
    pub at: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub nodes: usize,
    pub kernels: usize,
    pub readable: usize,
    pub opaque: usize,
    pub classes: usize,
    pub ambiguous: usize,
    pub pairs: usize,
    pub components: usize,
}

impl fmt::Display for Census {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} nodes ({} kernel, {} readable, {} opaque), {} components, \
             {} ambiguous in {} classes ({} adjacent pairs)",
            self.nodes,
            self.kernels,
            self.readable,
            self.opaque,
            self.components,
            self.ambiguous,
            self.classes,
            self.pairs,
        )
    }
}

#[derive(Clone, Debug)]
pub struct NodeMap {
    topology: Topology,
    nodes: Vec<Node>,
    classes: Vec<Ambiguous>,
    class_of: Vec<Option<usize>>,
    census: Census,
}

impl NodeMap {
    pub fn of(graph: &Graph) -> Result<NodeMap> {
        Ok(NodeMap::from_walk(nodes::walk(graph)?))
    }

    #[must_use]
    pub fn from_walk(walked: Walked) -> NodeMap {
        let topology = Topology::of(&walked);
        let Walked {
            nodes, ambiguous, ..
        } = walked;

        let mut classes: Vec<Ambiguous> = Vec::new();
        let mut class_of: Vec<Option<usize>> = vec![None; nodes.len()];
        let mut at = 0usize;
        while at < nodes.len() {
            let mut end = at + 1;
            while end < nodes.len()
                && nodes[end].depth == nodes[at].depth
                && nodes[end].symbol == nodes[at].symbol
                && nodes[end].kind == nodes[at].kind
            {
                end += 1;
            }
            if end - at > 1 {
                let class = classes.len();
                for held in &mut class_of[at..end] {
                    *held = Some(class);
                }
                classes.push(Ambiguous {
                    depth: nodes[at].depth,
                    symbol: nodes[at].symbol.clone(),
                    at: (at..end).collect(),
                });
            }
            at = end;
        }

        let kernels = nodes.iter().filter(|node| node.kernel()).count();
        let readable = nodes
            .iter()
            .filter(|node| node.kernel() && node.opaque.is_none())
            .count();
        let components = nodes
            .iter()
            .filter(|node| node.kernel())
            .map(|node| 7 + node.params.iter().map(words).sum::<usize>())
            .sum();
        let census = Census {
            nodes: nodes.len(),
            kernels,
            readable,
            opaque: kernels - readable,
            classes: classes.len(),
            ambiguous: class_of.iter().filter(|held| held.is_some()).count(),
            pairs: ambiguous,
            components,
        };

        NodeMap {
            topology,
            nodes,
            classes,
            class_of,
            census,
        }
    }

    #[must_use]
    pub fn topology(&self) -> Topology {
        self.topology
    }

    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    #[must_use]
    pub fn node(&self, at: usize) -> Option<&Node> {
        self.nodes.get(at)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[must_use]
    pub fn classes(&self) -> &[Ambiguous] {
        &self.classes
    }

    #[must_use]
    pub fn ambiguous(&self, at: usize) -> bool {
        self.class_of.get(at).copied().flatten().is_some()
    }

    #[must_use]
    pub fn census(&self) -> Census {
        self.census
    }
}

fn words(param: &Param) -> usize {
    param.bytes.len().div_ceil(8).max(1)
}

#[derive(Clone, Debug)]
pub struct Patch {
    pub at: usize,
    pub node: *mut c_void,
    pub entry: *mut c_void,
    pub func: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
    pub params: Vec<Param>,
    pub moved: Vec<Component>,
}

impl Patch {
    #[must_use]
    pub fn block(&self) -> Vec<u8> {
        let len = self
            .params
            .iter()
            .map(|param| param.offset + param.size)
            .max()
            .unwrap_or(0);
        let mut block = vec![0u8; len];
        for param in &self.params {
            let take = param.size.min(param.bytes.len());
            block[param.offset..param.offset + take].copy_from_slice(&param.bytes[..take]);
        }
        block
    }
}

#[derive(Clone, Debug)]
pub enum Diff {
    NotSameTopology {
        held: Topology,
        brought: Topology,
    },
    Aligned {
        patches: Vec<Patch>,
        unmoved: usize,
        agreed: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refused {
    Ambiguous {
        depth: usize,
        symbol: String,
        at: usize,
        count: usize,
        differing: usize,
        component: Component,
        more: usize,
    },
    Opaque {
        at: usize,
        depth: usize,
        symbol: String,
        why: &'static str,
        component: Component,
        more: usize,
    },
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refused::Ambiguous {
                depth,
                symbol,
                at,
                count,
                differing,
                component,
                more,
            } => write!(
                f,
                "{count} nodes at depth {depth} run `{symbol}` and the canonical order \
                 tells them apart by an enumeration index the driver never promised \
                 twice; {differing} of them differ between the two captures (first at \
                 node {at}, component {}). Aligning them by guess can hand one \
                 node the other's buffer and the mistake COMPUTES, so no patch is \
                 derived{}",
                component.at,
                plural(*more),
            ),
            Refused::Opaque {
                at,
                depth,
                symbol,
                why,
                component,
                more,
            } => write!(
                f,
                "node {at} at depth {depth} runs `{symbol}`, its {} moved \
                 between the two captures, and its parameter block was never readable \
                 ({why}); `cudaGraphExecKernelNodeSetParams` restates a node in full, \
                 so a component of it cannot be written on its own{}",
                component.at,
                plural(*more),
            ),
        }
    }
}

impl Refused {
    #[must_use]
    pub fn reason(&self) -> Refuse {
        match self {
            Refused::Ambiguous { .. } => Refuse::Ambiguous,
            Refused::Opaque { .. } => Refuse::Opaque,
        }
    }
}

impl std::error::Error for Refused {}

fn plural(more: usize) -> String {
    match more {
        0 => String::new(),
        1 => " (and 1 more refusal in this diff)".to_string(),
        n => format!(" (and {n} more refusals in this diff)"),
    }
}

#[allow(clippy::too_many_lines)]
pub fn diff(held: &NodeMap, brought: &Walked) -> core::result::Result<Diff, Refused> {
    let topology = Topology::of(brought);
    let mismatch = || Diff::NotSameTopology {
        held: held.topology,
        brought: topology,
    };
    if topology != held.topology || brought.nodes.len() != held.nodes.len() {
        return Ok(mismatch());
    }

    for (was, now) in held.nodes.iter().zip(&brought.nodes) {
        if was.depth != now.depth || was.kind != now.kind || was.symbol != now.symbol {
            return Ok(mismatch());
        }
    }

    let moved: Vec<Vec<Component>> = held
        .nodes
        .iter()
        .zip(&brought.nodes)
        .enumerate()
        .map(|(at, (was, now))| {
            if was.kernel() {
                moved_of(at as u32, was, now)
            } else {
                Vec::new()
            }
        })
        .collect();

    let mut refusals: Vec<Refused> = Vec::new();
    for class in &held.classes {
        let differing = class.at.iter().filter(|at| !moved[**at].is_empty()).count();
        if differing == 0 {
            continue;
        }
        let first = class
            .at
            .iter()
            .copied()
            .find(|at| !moved[*at].is_empty())
            .unwrap_or(class.at[0]);
        refusals.push(Refused::Ambiguous {
            depth: class.depth,
            symbol: class.symbol.clone(),
            at: first,
            count: class.at.len(),
            differing,
            component: moved[first][0].clone(),
            more: 0,
        });
    }
    for (at, node) in held.nodes.iter().enumerate() {
        if held.ambiguous(at) || moved[at].is_empty() {
            continue;
        }
        if let Some(why) = node.opaque.or(brought.nodes[at].opaque) {
            refusals.push(Refused::Opaque {
                at,
                depth: node.depth,
                symbol: node.symbol.clone(),
                why,
                component: moved[at][0].clone(),
                more: 0,
            });
        }
    }
    if !refusals.is_empty() {
        let more = refusals.len() - 1;
        let mut first = refusals.swap_remove(0);
        match &mut first {
            Refused::Ambiguous { more: count, .. } | Refused::Opaque { more: count, .. } => {
                *count = more;
            }
        }
        return Err(first);
    }

    let mut patches = Vec::new();
    let mut unmoved = 0usize;
    let mut agreed = 0usize;
    for (at, node) in held.nodes.iter().enumerate() {
        if !node.kernel() {
            continue;
        }
        if moved[at].is_empty() {
            if held.ambiguous(at) {
                agreed += 1;
            } else {
                unmoved += 1;
            }
            continue;
        }
        let now = &brought.nodes[at];
        patches.push(Patch {
            at,
            node: node.node,
            entry: now.entry,
            func: now.func,
            grid: now.grid,
            block: now.block,
            smem: now.smem,
            params: now.params.clone(),
            moved: moved[at].clone(),
        });
    }

    Ok(Diff::Aligned {
        patches,
        unmoved,
        agreed,
    })
}

fn moved_of(node: u32, was: &Node, now: &Node) -> Vec<Component> {
    let mut moved = Vec::new();
    let mut push = |at: At, value: i128| moved.push(Component::new(node, at, Law::Const(value)));
    if was.func != now.func {
        push(At::Entry, i128::from(now.func));
    }
    for axis in 0..3 {
        if was.grid[axis] != now.grid[axis] {
            push(At::Grid(axis as u8), i128::from(now.grid[axis]));
        }
    }
    for axis in 0..3 {
        if was.block[axis] != now.block[axis] {
            push(At::Block(axis as u8), i128::from(now.block[axis]));
        }
    }
    if was.smem != now.smem {
        push(At::Shared, i128::from(now.smem));
    }
    if was.params.len() != now.params.len() {
        push(At::Shape, now.params.len() as i128);
        return moved;
    }
    for (at, (old, new)) in was.params.iter().zip(&now.params).enumerate() {
        if old.offset != new.offset || old.size != new.size || old.bytes.len() != new.bytes.len() {
            push(At::Shape, new.size as i128);
            continue;
        }
        for word in 0..words(old) {
            let from = word * 8;
            let to = (from + 8).min(old.bytes.len());
            if from >= old.bytes.len() {
                break;
            }
            if old.bytes[from..to] != new.bytes[from..to] {
                push(
                    At::Arg {
                        at: at as u16,
                        word: word as u16,
                    },
                    word_at(new, word),
                );
            }
        }
    }
    moved
}

fn word_at(param: &Param, word: usize) -> i128 {
    let from = word * 8;
    let to = (from + 8).min(param.bytes.len());
    let mut bytes = [0u8; 8];
    bytes[..to - from].copy_from_slice(&param.bytes[from..to]);
    i128::from(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(at: usize, depth: usize, symbol: &str, func: u64, args: &[u64]) -> Node {
        Node {
            at,
            depth,
            kind: 0,
            symbol: symbol.to_string(),
            func,
            node: core::ptr::without_provenance_mut(0x1000 + at),
            entry: core::ptr::without_provenance_mut(0x2000 + func as usize),
            grid: [1, 1, 1],
            block: [32, 1, 1],
            smem: 0,
            params: args
                .iter()
                .enumerate()
                .map(|(cell, value)| Param {
                    offset: cell * 8,
                    size: 8,
                    bytes: value.to_le_bytes().to_vec(),
                })
                .collect(),
            opaque: None,
        }
    }

    fn chain(nodes: Vec<Node>) -> Walked {
        let links = (1..nodes.len()).map(|at| (at - 1, at)).collect();
        Walked {
            ambiguous: 0,
            edges: nodes.len().saturating_sub(1),
            links,
            nodes,
        }
    }

    fn aligned(diff: Diff) -> (Vec<Patch>, usize, usize) {
        match diff {
            Diff::Aligned {
                patches,
                unmoved,
                agreed,
            } => (patches, unmoved, agreed),
            Diff::NotSameTopology { held, brought } => {
                panic!("expected an alignment, got {held} against {brought}")
            }
        }
    }

    #[test]
    fn map_every_case() {
        two_readings_of_one_graph_fingerprint_the_same();
        one_scalar_that_moved_is_the_only_component_the_patch_names();
        a_node_whose_block_was_never_read_refuses_to_be_rewritten();
    }

    fn two_readings_of_one_graph_fingerprint_the_same() {
        let a = chain(vec![
            node(0, 0, "load", 1, &[0xdead_0000]),
            node(1, 1, "gemm", 2, &[0xdead_0000, 8]),
        ]);
        let b = chain(vec![
            node(0, 0, "load", 1, &[0xbeef_0000]),
            node(1, 1, "gemm", 2, &[0xbeef_0000, 9]),
        ]);
        assert_eq!(
            Topology::of(&a),
            Topology::of(&b),
            "arguments are what a rebind moves; a fingerprint that saw them \
             would call every fire a new shape"
        );
    }

    fn one_scalar_that_moved_is_the_only_component_the_patch_names() {
        let held = NodeMap::from_walk(chain(vec![
            node(0, 0, "load", 1, &[0xdead_0000]),
            node(1, 1, "gemm", 2, &[0xdead_0000, 8]),
        ]));
        let brought = chain(vec![
            node(0, 0, "load", 1, &[0xdead_0000]),
            node(1, 1, "gemm", 2, &[0xdead_0000, 9]),
        ]);
        let (patches, unmoved, agreed) = aligned(diff(&held, &brought).expect("no refusal"));
        assert_eq!(patches.len(), 1, "one node moved");
        assert_eq!(patches[0].at, 1);
        assert_eq!(
            patches[0].moved,
            vec![Component::new(1, At::Arg { at: 1, word: 0 }, Law::Const(9))],
            "the component names the node, the word, and the value that rides"
        );
        assert_eq!(patches[0].params[1].word(), Some(9), "the NEW value rides");
        assert_eq!((unmoved, agreed), (1, 0));
    }

    fn a_node_whose_block_was_never_read_refuses_to_be_rewritten() {
        let mut blind = node(0, 0, "opaque", 1, &[1]);
        blind.opaque = Some("a kernelParams cell is null");
        let held = NodeMap::from_walk(chain(vec![blind.clone()]));
        let mut brought = blind;
        brought.grid = [4, 1, 1];

        let refused = diff(&held, &chain(vec![brought])).expect_err("unreadable is unwritable");
        assert_eq!(refused.reason(), Refuse::Opaque);
        let Refused::Opaque { component, .. } = &refused else {
            panic!("the block is what refused, not {refused}")
        };
        assert_eq!(component.at, At::Grid(0), "{refused}");
        assert_eq!(component.law, Law::Const(4), "the grid the new capture wants");
    }

}
