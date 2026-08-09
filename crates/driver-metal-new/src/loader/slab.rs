//! A bounded set of expert slots, filled from the checkpoint on demand.
//!
//! This is the mechanism that lets a model outgrow the machine, and it
//! exists because the obvious alternative does not work here. Mapping a bank
//! and asking for residency does not page it lazily: `requestResidency`
//! wires every page whether or not a kernel reads it, measured at 18.4 GB
//! wired for a streamed Qwen3-30B-A3B against 1.5 GB at rest. An Apple
//! Silicon GPU has no demand paging, and a kernel that touched a page the
//! residency set had let go would abort its command buffer rather than
//! fault it back. So the bytes the GPU can reach have to be a fixed, wired
//! region whose CONTENTS change, and changing them is an explicit copy the
//! host makes.
//!
//! What makes that copy cheap is the same fact the mapping work leaned on: a
//! converted checkpoint holds each expert bank as plain bytes in the file,
//! so paging an expert in is a copy from a mapping and not a transform.
//! There is no decode step here and no plan to run, which is why this takes
//! byte slices rather than a `LoadPlan`.
//!
//! # A slot is a set of parallel bands, not one band
//!
//! An expert is not one tensor. Qwen3-MoE routes to `gate_proj`, `up_proj`
//! and `down_proj` together; gpt-oss adds scales beside each. They have
//! DIFFERENT band sizes, so they cannot share a stride -- but they must
//! share a slot NUMBER, because the kernels read one `expert_ids` buffer and
//! every routed projection in the layer indexes with it. A per-tensor cache
//! would hand `gate_proj` slot 3 and `down_proj` slot 1 for the same expert,
//! and there is no single number to write into `expert_ids` that means both.
//!
//! So there is one slot index over (layer, expert) instances and one slab
//! per TENSOR KIND, all addressed by the same slot number. Occupying a slot
//! moves every tensor of that expert, which is also what makes the residency
//! claim true: a slot is resident or it is not, and no kernel can find half
//! of one.
//!
//! Two things are deliberately NOT here:
//!
//! * The Metal allocation. The slabs are [`Region`]s, so the eviction rule
//!   and the byte movement are both testable against ordinary host memory
//!   with no device. The eviction rule itself is
//!   [`model_loader::group_slot`], shared with CUDA rather than restated,
//!   because two backends deciding residency by two rules is two ways for
//!   the same checkpoint to thrash.
//! * Which expert a token wants. That is the router's answer, read back
//!   between segments, and it belongs to the caller.
//!
//! Single-threaded, like the forward pass that drives it.

use model_loader::group_slot::GroupSlotIndex;

use crate::region::Region;

/// One routed tensor kind across every mixture layer.
///
/// `layers[l]` is layer `l`'s expert-major bank -- a `[experts * rows,
/// cols]` tensor's bytes -- and expert `e`'s band is the `band_bytes` range
/// at `e * band_bytes`, which is exactly what the kernel would have read had
/// the whole bank been resident.
#[derive(Clone, Debug)]
pub struct SlabTensor<'a> {
    /// The runtime name past the layer prefix, e.g. `mlp.experts.gate_proj`.
    /// Diagnostics only; nothing here matches on it.
    pub suffix: String,
    /// One expert's bytes for this tensor kind.
    pub band_bytes: u64,
    /// Layer `l`'s bank. Each slice must hold `experts * band_bytes`.
    pub layers: Vec<&'a [u8]>,
}

/// Why an [`ExpertSlab`] was not built, or a slot not produced.
#[derive(Debug, PartialEq, Eq)]
pub enum SlabError {
    /// No routed tensors were declared.
    NoRoutedTensors,
    /// One slab per tensor kind; the counts disagree.
    SlabCount {
        /// Slabs handed in.
        got: usize,
        /// Tensor kinds declared.
        want: usize,
    },
    /// Zero experts or zero slots is not a small cache; it cannot run.
    EmptyCache,
    /// The first tensor kind declares no mixture layers.
    NoLayers,
    /// A tensor kind spans a different layer count than its siblings, which
    /// would silently make (layer, expert) mean a different expert for it.
    LayerSpan {
        /// The tensor kind at fault.
        suffix: String,
        /// Its layer count.
        got: usize,
        /// The layer count of the rest.
        want: usize,
    },
    /// A tensor kind with zero-byte bands pages nothing.
    ZeroBand {
        /// The tensor kind at fault.
        suffix: String,
    },
    /// A layer's bank is too short to hold every expert's band. The C++
    /// held raw pointers and could only check them against null; a slice
    /// carries its length, so the real precondition is checked.
    ShortBank {
        /// The tensor kind at fault.
        suffix: String,
        /// The layer whose bank is short.
        layer: usize,
        /// The bank's length.
        got: u64,
        /// `experts * band_bytes`.
        want: u64,
    },
    /// A slab region cannot hold `slots` bands of its tensor kind.
    ShortSlab {
        /// The tensor kind at fault.
        suffix: String,
        /// The region's length.
        got: u64,
        /// `slots * band_bytes`.
        want: u64,
    },
    /// A (layer, expert) outside the grid. The expert id is the router's
    /// answer read back from the device -- data, not a caller bug -- so a
    /// corrupted readback fails the fire instead of the process.
    OutsideGrid {
        /// The layer asked for.
        layer: u32,
        /// The expert asked for.
        expert: u32,
        /// The grid's layer count.
        layers: u32,
        /// The grid's experts per layer.
        experts: u32,
    },
    /// Every slot is pinned by the batch in flight: it wants more experts at
    /// once than the slab has slots. Refused rather than evicting a slot a
    /// kernel may still read.
    BatchOverSubscribed {
        /// The slab's slot count, all pinned.
        num_slots: u32,
    },
    /// A band copy left its region; carries the region layer's report.
    Copy(String),
}

impl std::fmt::Display for SlabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlabError::NoRoutedTensors => write!(f, "expert slab: no routed tensors"),
            SlabError::SlabCount { got, want } => {
                write!(
                    f,
                    "expert slab: one slab per tensor kind, got {got} for {want}"
                )
            }
            SlabError::EmptyCache => write!(f, "expert slab: an empty cache cannot run"),
            SlabError::NoLayers => write!(f, "expert slab: no mixture layers"),
            SlabError::LayerSpan { suffix, got, want } => write!(
                f,
                "expert slab: '{suffix}' spans {got} layers against {want} for the rest"
            ),
            SlabError::ZeroBand { suffix } => {
                write!(f, "expert slab: '{suffix}' has zero-byte bands")
            }
            SlabError::ShortBank {
                suffix,
                layer,
                got,
                want,
            } => write!(
                f,
                "expert slab: '{suffix}' layer {layer} holds {got} bytes, needs {want}"
            ),
            SlabError::ShortSlab { suffix, got, want } => write!(
                f,
                "expert slab: '{suffix}' slab holds {got} bytes, needs {want}"
            ),
            SlabError::OutsideGrid {
                layer,
                expert,
                layers,
                experts,
            } => write!(
                f,
                "expert slab: ({layer}, {expert}) outside {layers} x {experts}"
            ),
            SlabError::BatchOverSubscribed { num_slots } => write!(
                f,
                "expert slab: all {num_slots} slots are pinned by the batch in \
                 flight, so it wants more experts at once than the slab holds"
            ),
            SlabError::Copy(report) => write!(f, "expert slab: {report}"),
        }
    }
}

impl std::error::Error for SlabError {}

/// The bounded expert cache: bookkeeping from the shared index, bytes moved
/// into `R` regions.
#[derive(Debug)]
pub struct ExpertSlab<'a, R: Region> {
    tensors: Vec<SlabTensor<'a>>,
    slabs: Vec<R>,
    experts: u32,
    layers: u32,
    slot_bytes: u64,
    index: GroupSlotIndex,
    hits: u64,
    misses: u64,
}

impl<'a, R: Region> ExpertSlab<'a, R> {
    /// Build the cache over `slots` slots.
    ///
    /// `slabs[t]` must hold `slots * tensors[t].band_bytes` (checked --
    /// after `slots` is clamped to the instance count, since slots past the
    /// last instance are never written).
    ///
    /// # Errors
    ///
    /// Every way the shape can lie is a named [`SlabError`]; see each
    /// variant. A budget that cannot hold one slot has not configured a
    /// small cache, it has configured one that cannot run.
    pub fn new(
        tensors: Vec<SlabTensor<'a>>,
        experts_per_layer: u32,
        slabs: Vec<R>,
        slots: u32,
    ) -> Result<Self, SlabError> {
        if tensors.is_empty() {
            return Err(SlabError::NoRoutedTensors);
        }
        if slabs.len() != tensors.len() {
            return Err(SlabError::SlabCount {
                got: slabs.len(),
                want: tensors.len(),
            });
        }
        if experts_per_layer == 0 || slots == 0 {
            return Err(SlabError::EmptyCache);
        }
        let layers = tensors[0].layers.len();
        if layers == 0 {
            return Err(SlabError::NoLayers);
        }
        let instances = u64::from(u32::try_from(layers).map_err(|_| SlabError::NoLayers)?)
            * u64::from(experts_per_layer);
        let slots = u64::from(slots)
            .min(instances)
            .try_into()
            .unwrap_or(u32::MAX);
        let mut slot_bytes = 0u64;
        for (tensor, slab) in tensors.iter().zip(&slabs) {
            // Not a shape check for its own sake: a tensor present on fewer
            // layers than the rest would silently make (layer, expert) mean
            // a different expert for it than for its siblings.
            if tensor.layers.len() != layers {
                return Err(SlabError::LayerSpan {
                    suffix: tensor.suffix.clone(),
                    got: tensor.layers.len(),
                    want: layers,
                });
            }
            if tensor.band_bytes == 0 {
                return Err(SlabError::ZeroBand {
                    suffix: tensor.suffix.clone(),
                });
            }
            let bank_bytes = u64::from(experts_per_layer) * tensor.band_bytes;
            for (layer, bank) in tensor.layers.iter().enumerate() {
                if (bank.len() as u64) < bank_bytes {
                    return Err(SlabError::ShortBank {
                        suffix: tensor.suffix.clone(),
                        layer,
                        got: bank.len() as u64,
                        want: bank_bytes,
                    });
                }
            }
            let slab_bytes = u64::from(slots) * tensor.band_bytes;
            if slab.len() < slab_bytes {
                return Err(SlabError::ShortSlab {
                    suffix: tensor.suffix.clone(),
                    got: slab.len(),
                    want: slab_bytes,
                });
            }
            slot_bytes += tensor.band_bytes;
        }
        Ok(ExpertSlab {
            experts: experts_per_layer,
            layers: u32::try_from(layers).expect("checked above"),
            slot_bytes,
            index: GroupSlotIndex::new(u32::try_from(instances).unwrap_or(u32::MAX), slots),
            tensors,
            slabs,
            hits: 0,
            misses: 0,
        })
    }

    /// How many slots the budget holds, after clamping.
    #[must_use]
    pub fn num_slots(&self) -> u32 {
        self.index.num_slots()
    }

    /// Mixture layers.
    #[must_use]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Routed experts per layer.
    #[must_use]
    pub fn experts_per_layer(&self) -> u32 {
        self.experts
    }

    /// Every routed tensor of one expert, which is what one slot costs.
    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// The whole cache's byte cost.
    #[must_use]
    pub fn slab_bytes(&self) -> u64 {
        u64::from(self.num_slots()) * self.slot_bytes
    }

    /// True when the slab holds every expert of every layer, so nothing can
    /// ever be evicted. The caller that knows this can skip the routing
    /// readback entirely -- which is the whole cost of streaming once the
    /// slab is warm.
    #[must_use]
    pub fn fully_resident(&self) -> bool {
        self.num_slots() == self.index.arity()
    }

    /// The slot holding layer `layer`'s expert `expert`, copying every one
    /// of its tensors in if it is not there.
    ///
    /// The slot stays pinned until [`end_batch`](Self::end_batch). A batch
    /// that wants more experts at once than the slab has slots is refused
    /// rather than evicting a slot a kernel may still read.
    ///
    /// # Errors
    ///
    /// [`SlabError::OutsideGrid`] for a wild routing readback,
    /// [`SlabError::BatchOverSubscribed`] for a pinned-out slab,
    /// [`SlabError::Copy`] if a band leaves its region.
    ///
    /// # Safety
    ///
    /// The GPU must not be reading the slab regions: the copy is a plain
    /// host write into shared storage, and the only thing that establishes
    /// exclusivity is a step boundary. Between segments -- where the router
    /// readback that produced `expert` was taken -- is such a boundary.
    pub unsafe fn ensure_resident(&mut self, layer: u32, expert: u32) -> Result<u32, SlabError> {
        if layer >= self.layers || expert >= self.experts {
            return Err(SlabError::OutsideGrid {
                layer,
                expert,
                layers: self.layers,
                experts: self.experts,
            });
        }
        let key = layer * self.experts + expert;
        if let Some(slot) = self.index.find(key) {
            self.index.touch_and_pin(slot);
            self.hits += 1;
            return Ok(slot);
        }
        let acquired = self
            .index
            .acquire(key)
            .map_err(|err| SlabError::BatchOverSubscribed {
                num_slots: err.num_slots,
            })?;
        self.misses += 1;
        for (tensor, slab) in self.tensors.iter().zip(&self.slabs) {
            let band = usize::try_from(tensor.band_bytes).expect("band fits usize");
            let src_at = usize::try_from(u64::from(expert) * tensor.band_bytes)
                .expect("band offset fits usize");
            let src = &tensor.layers[layer as usize][src_at..src_at + band];
            // SAFETY: the caller's contract covers the GPU; the offsets were
            // established at construction (`ShortSlab` refused a region that
            // cannot hold `slots` bands).
            unsafe {
                slab.write(u64::from(acquired.slot) * tensor.band_bytes, src)
                    .map_err(|err| SlabError::Copy(err.to_string()))?;
            }
        }
        Ok(acquired.slot)
    }

    /// Every slot becomes evictable again. Slot numbers handed out since the
    /// last call must not be used past this point.
    pub fn end_batch(&mut self) {
        self.index.unpin_all();
    }

    /// Batch-boundary hit count.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// How many slots were filled from the checkpoint.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// The traffic streaming has cost so far.
    #[must_use]
    pub fn bytes_paged_in(&self) -> u64 {
        self.misses * self.slot_bytes
    }

    /// Tensor kind `t`'s slab region, for binding.
    #[must_use]
    pub fn slab(&self, t: usize) -> &R {
        &self.slabs[t]
    }

    /// Where tensor kind `t`'s band for `slot` starts within its slab.
    #[must_use]
    pub fn slot_offset(&self, t: usize, slot: u32) -> u64 {
        u64::from(slot) * self.tensors[t].band_bytes
    }

    /// The declared tensor kinds, in slab order.
    #[must_use]
    pub fn tensors(&self) -> &[SlabTensor<'a>] {
        &self.tensors
    }
}

#[cfg(test)]
mod tests {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    use super::*;

    /// A `Vec`-backed region, standing in for a heap slot.
    #[derive(Debug)]
    struct Host(Vec<u8>);

    impl Host {
        fn new(len: usize) -> Self {
            Host(vec![0; len])
        }
    }

    // SAFETY: the pointer is the Vec's own allocation and `len` its length;
    // nothing else aliases it while the test holds the slab.
    unsafe impl Region for Host {
        fn contents(&self) -> NonNull<c_void> {
            NonNull::new(self.0.as_ptr().cast_mut().cast()).expect("vec allocates")
        }
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
    }

    /// Two tensor kinds with different band sizes over 2 layers x 3 experts;
    /// every byte says which (kind, layer, expert) it belongs to.
    fn banks() -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let gate: Vec<Vec<u8>> = (0..2)
            .map(|l| (0..3).flat_map(|e| [10 * l + e; 4]).collect())
            .collect();
        let down: Vec<Vec<u8>> = (0..2)
            .map(|l| (0..3).flat_map(|e| [100 + 10 * l + e; 2]).collect())
            .collect();
        (gate, down)
    }

    fn slab_over<'a>(gate: &'a [Vec<u8>], down: &'a [Vec<u8>], slots: u32) -> ExpertSlab<'a, Host> {
        ExpertSlab::new(
            vec![
                SlabTensor {
                    suffix: "mlp.experts.gate_proj".into(),
                    band_bytes: 4,
                    layers: gate.iter().map(Vec::as_slice).collect(),
                },
                SlabTensor {
                    suffix: "mlp.experts.down_proj".into(),
                    band_bytes: 2,
                    layers: down.iter().map(Vec::as_slice).collect(),
                },
            ],
            3,
            vec![Host::new(4 * slots as usize), Host::new(2 * slots as usize)],
            slots,
        )
        .expect("a well-shaped slab builds")
    }

    #[test]
    fn a_slot_is_every_tensor_of_one_expert_or_nothing() {
        let (gate, down) = banks();
        let mut slab = slab_over(&gate, &down, 2);
        let slot = unsafe { slab.ensure_resident(1, 2) }.unwrap();
        // Both bands moved, addressed by the same slot number.
        assert_eq!(slab.slab(0).0[slot as usize * 4..][..4], [12, 12, 12, 12]);
        assert_eq!(slab.slab(1).0[slot as usize * 2..][..2], [112, 112]);
        assert_eq!(slab.slot_bytes(), 6);
        assert_eq!(slab.bytes_paged_in(), 6);
    }

    #[test]
    fn a_hit_costs_nothing_and_a_miss_evicts_the_lru() {
        let (gate, down) = banks();
        let mut slab = slab_over(&gate, &down, 2);
        let first = unsafe { slab.ensure_resident(0, 0) }.unwrap();
        let second = unsafe { slab.ensure_resident(0, 1) }.unwrap();
        slab.end_batch();
        assert_eq!(unsafe { slab.ensure_resident(0, 0) }.unwrap(), first);
        assert_eq!(slab.hits(), 1);
        // 0's hit re-pinned it, so the third expert takes 1's slot.
        let third = unsafe { slab.ensure_resident(0, 2) }.unwrap();
        assert_eq!(third, second);
        assert_eq!(slab.slab(0).0[third as usize * 4..][..4], [2, 2, 2, 2]);
        assert_eq!(slab.misses(), 3);
    }

    #[test]
    fn a_batch_wanting_more_experts_than_slots_refuses_rather_than_corrupting() {
        let (gate, down) = banks();
        let mut slab = slab_over(&gate, &down, 2);
        unsafe { slab.ensure_resident(0, 0) }.unwrap();
        unsafe { slab.ensure_resident(0, 1) }.unwrap();
        assert_eq!(
            unsafe { slab.ensure_resident(0, 2) },
            Err(SlabError::BatchOverSubscribed { num_slots: 2 })
        );
    }

    #[test]
    fn a_wild_routing_readback_is_data_not_a_crash() {
        let (gate, down) = banks();
        let mut slab = slab_over(&gate, &down, 2);
        assert_eq!(
            unsafe { slab.ensure_resident(0, 7) },
            Err(SlabError::OutsideGrid {
                layer: 0,
                expert: 7,
                layers: 2,
                experts: 3
            })
        );
    }

    #[test]
    fn slots_clamp_to_the_instance_count_and_full_residency_is_knowable() {
        let (gate, down) = banks();
        let slab = slab_over(&gate, &down, 64);
        assert_eq!(slab.num_slots(), 6, "2 layers x 3 experts");
        assert!(slab.fully_resident());
        assert_eq!(slab.slab_bytes(), 6 * 6);
    }

    #[test]
    fn a_short_bank_is_refused_at_construction_not_found_by_a_copy() {
        let (gate, down) = banks();
        let short = [vec![0u8; 4 * 3], vec![0u8; 4 * 3 - 1]];
        let err = ExpertSlab::new(
            vec![
                SlabTensor {
                    suffix: "gate".into(),
                    band_bytes: 4,
                    layers: short.iter().map(Vec::as_slice).collect(),
                },
                SlabTensor {
                    suffix: "down".into(),
                    band_bytes: 2,
                    layers: down.iter().map(Vec::as_slice).collect(),
                },
            ],
            3,
            vec![Host::new(4 * 2), Host::new(2 * 2)],
            2,
        )
        .expect_err("a bank one byte short cannot hold its last expert");
        assert_eq!(
            err,
            SlabError::ShortBank {
                suffix: "gate".into(),
                layer: 1,
                got: 11,
                want: 12
            }
        );
        drop(gate);
    }

    #[test]
    fn a_tensor_on_fewer_layers_than_its_siblings_is_refused() {
        let (gate, down) = banks();
        let err = ExpertSlab::new(
            vec![
                SlabTensor {
                    suffix: "gate".into(),
                    band_bytes: 4,
                    layers: gate.iter().map(Vec::as_slice).collect(),
                },
                SlabTensor {
                    suffix: "down".into(),
                    band_bytes: 2,
                    layers: down.iter().take(1).map(Vec::as_slice).collect(),
                },
            ],
            3,
            vec![Host::new(4 * 2), Host::new(2 * 2)],
            2,
        )
        .expect_err("(layer, expert) must mean the same expert for every kind");
        assert_eq!(
            err,
            SlabError::LayerSpan {
                suffix: "down".into(),
                got: 1,
                want: 2
            }
        );
    }
}
