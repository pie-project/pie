//! Where every tensor a plan publishes ENDS UP, by name.
//!
//! A resident load produces one arena and a map from the contract's names into
//! it. Deriving that map is pure plan-walking — no device, no bytes, no I/O —
//! and it was nonetheless written twice, once per backend, because the
//! executor returns the arena and not an account of it.
//!
//! The two copies had already diverged, which is the argument for this module
//! rather than a preference about where code lives:
//!
//! * only one of them resolved [`StorageInstr::CreateView`] chains, so on the
//!   other every alias — `layer.3.attn_norm` onto
//!   `model.layers.3.input_layernorm.weight`, and every unfused fallback view
//!   into a fused projection — missed the arena and arrived through the sink
//!   as a SECOND resident copy of bytes already on the device. In a
//!   qwen3-0.6B plan that is 140 of 141 tensors.
//! * only the other counted what a plan leaves outside the arena before
//!   allocating, so the first could not size a buffer to hold everything and
//!   had to allocate again, per tensor, afterwards.
//!
//! Both are answers to questions the plan already contains. Asking it once,
//! here, is what makes them the same answer.
//!
//! # Who reads this today, measured
//!
//! `driver-metal`'s `weights/stage.rs`, and nothing else — the two copies this
//! module replaced were Metal's and CUDA's, and CUDA's has not come back yet.
//! That makes it a candidate for moving out beside its one reader, and it is
//! the wrong move: walking a `StorageInstr` stream to find out where a plan
//! puts things is a question about the PLAN, and answering it inside a driver
//! is precisely the arrangement that produced the two divergences above. Five
//! files in this crate point at it for that reason. It stays where the
//! question is.

use std::collections::{BTreeMap, HashMap};

use crate::plan::{LoadPlan, StorageInstr};
use crate::types::BufferId;

/// One tensor's place in the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// Byte offset from the start of the arena.
    pub offset: u64,
    /// Length in bytes.
    pub bytes: u64,
}

/// What an executed plan will hold, known before a byte is read.
#[derive(Clone, Debug, Default)]
pub struct Published {
    /// Every tensor that lands IN the arena, by the contract's name.
    ///
    /// Includes views, which have no `persistent_offset` of their own and
    /// whose offset is their base's plus the view's — see the module doc for
    /// what treating them as absent costs.
    pub in_arena: BTreeMap<String, Span>,
    /// Tensors the plan publishes with no arena offset, by name, with the
    /// bytes the plan declares for each.
    ///
    /// They arrive through the [`sink`](crate::executor::sink), because they
    /// have no offset to be written at. Named here — with their sizes, and in
    /// name order — so a caller can budget for them BEFORE the load rather
    /// than discover them during it. A backend whose arena is one allocation
    /// can then size it for everything and write once; one that allocates per
    /// tensor afterwards does not have to.
    pub outside: BTreeMap<String, u64>,
}

/// A view's base buffer and the offset into it, per hop.
type ViewOf = HashMap<BufferId, (BufferId, u64, u64)>;

/// How deep a chain of views may go before the walk gives up.
///
/// A view of a view is legal and the chains are short and acyclic by
/// construction, so this is a guard against a malformed plan rather than a
/// search limit: reaching it means the plan has a cycle, and the honest
/// answer is then "this buffer has no offset" rather than a hang.
const MAX_VIEW_HOPS: usize = 16;

/// Name every tensor the plan publishes and say where it lands.
///
/// Reads the plan only. The result is exactly as true before the load as
/// after it, which is what lets a caller allocate the arena, run the plan into
/// it, and name the contents without a second pass over anything.
#[must_use]
pub fn publish_spans(plan: &LoadPlan) -> Published {
    let names: HashMap<_, _> = plan
        .tensors
        .iter()
        .map(|t| (t.id, t.name.as_str()))
        .collect();

    // VIEWS ARE NOT COPIES. A `CreateView` gives a buffer no
    // `persistent_offset` of its own even when its input has one, so a walk
    // that reads only `persistent_offset` sees an alias as absent.
    let mut view_of: ViewOf = HashMap::new();
    for instr in &plan.instrs {
        if let StorageInstr::CreateView {
            input,
            output,
            view,
            ..
        } = instr
        {
            let bytes = view
                .stride
                .dims
                .iter()
                .try_fold(1u64, |n, d| u64::try_from(d.count).ok().map(|c| n * c))
                .unwrap_or(0)
                * u64::from(view.stride.element_bytes);
            view_of.insert(
                *output,
                (*input, view.offset + view.stride.base_offset, bytes),
            );
        }
    }

    let offset_of: HashMap<_, _> = plan
        .buffers
        .iter()
        .filter_map(|b| Some((b.id, b.persistent_offset?)))
        .collect();
    let arena_offset = |mut id: BufferId| -> Option<u64> {
        let mut extra = 0;
        for _ in 0..MAX_VIEW_HOPS {
            if let Some(base) = offset_of.get(&id) {
                return Some(base + extra);
            }
            let (parent, delta, _) = *view_of.get(&id)?;
            extra += delta;
            id = parent;
        }
        None
    };

    let mut in_arena = BTreeMap::new();
    let mut outside = BTreeMap::new();
    for b in &plan.buffers {
        let Some(tensor) = b.tensor else { continue };
        let Some(name) = names.get(&tensor) else {
            continue;
        };
        let Some(offset) = arena_offset(b.id) else {
            outside.insert((*name).to_string(), b.bytes);
            continue;
        };
        // A view's own `bytes` is zero — its length lives in the extent the
        // `CreateView` carries.
        let bytes = if b.bytes == 0 {
            view_of.get(&b.id).map_or(0, |(_, _, len)| *len)
        } else {
            b.bytes
        };
        in_arena.insert((*name).to_string(), Span { offset, bytes });
    }

    // A name reached through both an arena buffer and a bare one is IN the
    // arena: the bare buffer is the one that has no offset, and publishing the
    // tensor twice would make a caller upload bytes that are already there.
    outside.retain(|name, _| !in_arena.contains_key(name));

    Published { in_arena, outside }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extent::{Dim, Extent};
    use crate::plan::{BufferDecl, DestExtent, StorageTarget};
    use crate::types::{DType, Encoding, InstrId, TensorDecl, TensorId, Visibility};

    fn plan_with(
        buffers: Vec<BufferDecl>,
        instrs: Vec<StorageInstr>,
        names: &[(u32, &str)],
    ) -> LoadPlan {
        let mut plan = LoadPlan::empty(StorageTarget::default());
        plan.tensors = names
            .iter()
            .map(|(id, name)| TensorDecl {
                id: TensorId(*id),
                name: (*name).to_string(),
                shape: Vec::new(),
                encoding: Encoding::Raw(DType::BF16),
                alignment: 1,
                visibility: Visibility::Public,
            })
            .collect();
        plan.buffers = buffers;
        plan.instrs = instrs;
        plan
    }

    fn buffer(id: u32, tensor: Option<u32>, offset: Option<u64>, bytes: u64) -> BufferDecl {
        BufferDecl {
            id: BufferId(id),
            tensor: tensor.map(TensorId),
            ty: crate::contract::TensorType::raw(vec![bytes as i64], DType::U8),
            bytes,
            alignment: 1,
            temporary: false,
            persistent_offset: offset,
            scratch_offset: None,
        }
    }

    /// The divergence this module exists to end: an alias has no
    /// `persistent_offset`, and reading only that field reports it as absent.
    #[test]
    fn a_view_of_a_resident_buffer_is_in_the_arena() {
        let view = DestExtent {
            buffer: BufferId(1),
            offset: 64,
            stride: Extent {
                base_offset: 0,
                element_bytes: 2,
                dims: vec![Dim {
                    count: 8,
                    src_stride: 1,
                    dst_stride: 1,
                }],
            },
        };
        let plan = plan_with(
            vec![
                buffer(0, Some(0), Some(1024), 256),
                buffer(1, Some(1), None, 0),
            ],
            vec![StorageInstr::CreateView {
                id: InstrId(0),
                input: BufferId(0),
                output: BufferId(1),
                view,
            }],
            &[(0, "base"), (1, "alias")],
        );
        let published = publish_spans(&plan);
        assert_eq!(
            published.in_arena.get("alias"),
            Some(&Span {
                offset: 1024 + 64,
                bytes: 16
            }),
            "a view's offset is its base's plus its own, and its length comes \
             from the extent the CreateView carries"
        );
        assert!(published.outside.is_empty());
    }

    /// A chain terminating in nothing is reported as outside, with the bytes a
    /// caller has to budget, rather than looked for forever.
    #[test]
    fn a_buffer_with_no_offset_and_no_base_is_outside() {
        let plan = plan_with(vec![buffer(0, Some(0), None, 128)], vec![], &[(0, "loose")]);
        let published = publish_spans(&plan);
        assert!(published.in_arena.is_empty());
        assert_eq!(published.outside.get("loose"), Some(&128));
    }
}
