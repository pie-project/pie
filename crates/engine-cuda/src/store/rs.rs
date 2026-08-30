//! **The buffered-activation pool** — the device half of `RsVerb::Buffer` and
//! `RsVerb::FoldBuffered` (alto design §6, survey §9).
//!
//! # What is buffered, and dev's economy
//!
//! Not the recurrent state, and not the recurrence's outputs: the **inputs of
//! the conv+scan**, which for a gated-delta layer are the two in-projection
//! planes the mixer computes before anything stateful happens — `qkv` (the
//! conv's rows, `conv_dim` wide) and `ba` (the `[b | a]` projection the gate
//! prep reads, `2·V_h` wide). A few kilobytes per token against a full state
//! copy per request, which is exactly why dev's programming model can afford
//! to keep a speculative window recoverable and why the ping-pong slot design
//! is superseded (survey §9's last paragraph).
//!
//! `z` is NOT buffered. It is the gate the post-recurrence `rmsnorm_gated`
//! applies, and a fold-buffered replay is state-only — it computes no output
//! anybody reads, so the one in-projection plane the recurrence does not feed
//! is the one plane the replay does not need.
//!
//! # The shape
//!
//! ```text
//! pool[layer][page_slot][plane][token]       bf16
//!            └─ `Budget`-many, one per state slot (dev: rs_buffer_num_slots)
//!                        │      └─ `page_tokens` rows = the kv page size
//!                        └─ plane 0 `qkv`, then plane 1 `ba`
//! ```
//!
//! One allocation for the whole pool, pointer-stable for the load's lifetime
//! (article 7), cut by arithmetic: a page slot holds `page_tokens` rows of
//! every plane, and a layer holds `slots` page slots. A lane's buffer is a RUN
//! of page slots stated by the verb ([`engine::PageRange`]), page-major
//! from buffer token zero — so buffer token `i` lives in page slot
//! `run.page_index + i / page_tokens` at in-page row `i % page_tokens`, which
//! is the addressing both the scatter and the gather do and the only thing the
//! two have to agree about.
//!
//! Plane-major INSIDE a page slot rather than token-major across the two,
//! because that is what makes one page's share of one plane a single
//! contiguous run: the copies are then `cudaMemcpyAsync`s per (lane, page,
//! plane) and nothing needs a strided kernel — dev's own choice
//! (`qwen3_5_forward.cpp:704-772`, three d2d memcpys per page).
//!
//! # Why the planes are keyed by VALUE and not by layer
//!
//! The scatter has to happen where a plane is live and the gather has to
//! happen before its reader runs, and the two readers are two different ops:
//! `attention.ssm_causal_conv1d_chunked` takes `qkv`,
//! `attention.ssm_gdn_prep` takes `ba`. So [`Planes`] is a map from the
//! operand's `ValueId` to the plane it is buffered as, and each dispatch arm
//! asks about its own operand — no arm has to know what the arm beside it will
//! read, and a family that grows a third plane grows one row in this map.
//!
//! It also settles the arm question for free. A GDN mixer traces TWICE, once
//! for the decode split and once for the chunked one, so a layer has two `ba`
//! values and two conv inputs. Only the chunked ones are in this map, because
//! only the chunked scans carry the `commit_len` and `write_state_mask` seats
//! that make a buffered fold dispatchable at all (`attn/ssm.cuh`) — a
//! single-token buffered write is expressible in the contract and is refused
//! by name rather than served on an arm that cannot truncate.

use std::collections::HashMap;

use model_ir::{Attention, Dim, Dtype, Operands, Operation, Trace, Ty, ValueId};

use crate::device::Buffer;
use crate::error::{Fault, Result};
use crate::store::kv::Paging;

/// The element the in-projection planes are staged at — the same `bf16` the
/// state slabs are, and for the same reason: `attn/ssm.cuh` instantiates every
/// state-taking entry at `__nv_bfloat16`, and the planes this pool holds are
/// the activations those entries read.
pub const PLANE_DTYPE: Dtype = Dtype::Bf16;

/// Bytes of one buffered element.
const ELEMENT: u64 = 2;

/// One buffered plane: which layer's block of the pool it lives in, where
/// inside a page slot its rows begin, and how wide one token's row is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plane {
    /// Which recurrent layer's block of the pool, in plan order.
    pub layer: u32,
    /// Elements from the front of a page slot to this plane's first row.
    pub at: u64,
    /// Elements one token writes.
    pub width: u64,
}

/// Every buffered plane a plan declares, keyed by the operand that reads it.
#[derive(Debug, Clone, Default)]
pub struct Planes(HashMap<u32, Plane>);

impl Planes {
    /// The plane this operand is buffered as, or `None` for a value no
    /// chunked recurrent op reads.
    #[must_use]
    pub fn of(&self, id: ValueId) -> Option<Plane> {
        self.0.get(&id.0).copied()
    }

    /// How many planes were resolved.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Does this plan buffer anything at all?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// **The pool, and the plan's reading of it.**
#[derive(Debug)]
pub struct Buffers {
    slab: Buffer,
    planes: Planes,
    /// Tokens one page slot holds — the kv page size (dev's rule).
    page_tokens: u32,
    /// Page slots per layer — the state-slot count, which is what the
    /// `Budget` already sized the recurrent banks by.
    slots: u32,
    /// Elements one page slot holds: `page_tokens × Σ plane widths`.
    per_page: u64,
    /// How many recurrent layers the plan declares.
    layers: u32,
}

impl Buffers {
    /// Reserve the pool one plan needs at one deployment's budget, or `None`
    /// for a plan with no chunked recurrent layer to buffer.
    ///
    /// # The structure is read off the plan, not configured
    ///
    /// A layer's chunked scan (`attention.ssm_gated_delta_chunked`) names the
    /// `qkv` it consumes and the `gates` it consumes; the node that WROTE that
    /// `qkv` is the chunked conv and the node that wrote those `gates` is the
    /// prep, so following the two operands backwards is what picks the chunked
    /// arm's planes out of a layer that also traced a decode arm. Nothing here
    /// is matched by name or by position.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a recurrent layer whose planes this shell cannot
    /// size — a symbolic row width, a scan whose operands no node defines, a
    /// plane written after the scan that reads it, or per-layer widths that
    /// disagree — and [`Fault::Device`](crate::Fault::Device) for the
    /// allocation.
    pub fn reserve(trace: &Trace, paging: Paging) -> Result<Option<Buffers>> {
        let Some((planes, per_token, layers)) = read(trace, paging.page_size)? else {
            return Ok(None);
        };
        let per_page = u64::from(paging.page_size) * per_token;
        let bytes = per_page
            .saturating_mul(u64::from(paging.slots))
            .saturating_mul(u64::from(layers))
            .saturating_mul(ELEMENT);
        Ok(Some(Buffers {
            slab: Buffer::zeroed(usize::try_from(bytes).unwrap_or(usize::MAX))?,
            planes,
            page_tokens: paging.page_size,
            slots: paging.slots,
            per_page,
            layers,
        }))
    }
}

/// **Read the plan's chunked recurrent layers**, with no allocation and no
/// device in reach — the half of [`Buffers::reserve`] that can be exercised
/// on a machine with no GPU, and the half that can refuse.
///
/// Answers `None` for a plan with no chunked gated-delta scan at all (every
/// dense text, and the KDA families until they are given the buffer
/// vocabulary), the planes keyed by the operand that reads them otherwise.
///
/// # Errors
///
/// [`Fault::Unbound`] — see [`Buffers::reserve`].
pub fn read(trace: &Trace, page_tokens: u32) -> Result<Option<(Planes, u64, u32)>> {
    {
        let paging = page_tokens;
        // Which node defines which value, so a scan's operands can be walked
        // back to the ops that wrote them.
        let mut defined: HashMap<u32, usize> = HashMap::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            let mut outs = Vec::new();
            node.op.outputs(&mut outs);
            for out in outs {
                defined.insert(out.0, at);
            }
        }

        let mut planes: HashMap<u32, Plane> = HashMap::new();
        let mut per_token: Option<u64> = None;
        let mut layers = 0u32;
        for (at, node) in trace.nodes.iter().enumerate() {
            let Operation::Attention(Attention::SsmGatedDeltaChunked { qkv, gates, .. }) = &node.op
            else {
                continue;
            };
            let conv = defined.get(&qkv.0).copied().and_then(|n| {
                match &trace.nodes[n].op {
                    Operation::Attention(Attention::SsmCausalConv1dChunked { x, .. }) => {
                        Some((n, *x))
                    }
                    _ => None,
                }
            });
            let prep = defined.get(&gates.0).copied().and_then(|n| {
                match &trace.nodes[n].op {
                    Operation::Attention(Attention::SsmGdnPrep { ba, .. }) => Some((n, *ba)),
                    _ => None,
                }
            });
            let (Some((conv_at, x)), Some((prep_at, ba))) = (conv, prep) else {
                return Err(Fault::Unbound {
                    what: format!(
                        "the chunked recurrence at node {at}, whose `qkv` and `gates` are not \
                         written by a chunked conv and a gate prep — this shell buffers the \
                         two in-projection planes those ops read and knows no third shape"
                    ),
                });
            };
            // **THE GATHER HAS TO LAND BEFORE ITS READER**, and the two
            // readers are two nodes. The scatter is order-free (both planes
            // are still live wherever either op runs), but a replay
            // OVERWRITES them — so the walk must reach each plane's own op
            // with the plane not yet consumed, which is true exactly when
            // each op stands before the scan that named it. Checked here
            // rather than assumed, because it is a property of the model
            // text and not of this shell.
            if conv_at >= at || prep_at >= at {
                return Err(Fault::Unbound {
                    what: format!(
                        "the chunked recurrence at node {at} reads planes written at nodes \
                         {conv_at} and {prep_at}, which do not both stand before it"
                    ),
                });
            }
            let qkv_width = width_of(trace, x).ok_or_else(|| unsized_plane("the chunked conv's rows", x))?;
            let ba_width = width_of(trace, ba).ok_or_else(|| unsized_plane("the gate prep's `[b | a]`", ba))?;
            let page = u64::from(paging);
            planes.insert(
                x.0,
                Plane {
                    layer: layers,
                    at: 0,
                    width: qkv_width,
                },
            );
            planes.insert(
                ba.0,
                Plane {
                    layer: layers,
                    at: page * qkv_width,
                    width: ba_width,
                },
            );
            // **ONE WIDTH FOR EVERY LAYER**, which is dev's one `stash_width`
            // and is true of every family that ships one: the in-projection
            // geometry is per model, not per layer. A plan that disagreed
            // would need a per-layer stride table, so it is refused rather
            // than mis-addressed.
            let here = qkv_width + ba_width;
            match per_token {
                None => per_token = Some(here),
                Some(first) if first == here => {}
                Some(first) => {
                    return Err(Fault::Unbound {
                        what: format!(
                            "recurrent layer {layers} buffers {here} elements a token where \
                             layer 0 buffers {first} — this pool cuts one page-slot stride \
                             for every layer"
                        ),
                    });
                }
            }
            layers += 1;
        }
        let Some(per_token) = per_token else {
            return Ok(None);
        };
        Ok(Some((Planes(planes), per_token, layers)))
    }
}

impl Buffers {
    /// The plan's reading: which operands are buffered planes.
    #[must_use]
    pub fn planes(&self) -> &Planes {
        &self.planes
    }

    /// Tokens one buffer page slot holds.
    #[must_use]
    pub fn page_tokens(&self) -> u32 {
        self.page_tokens
    }

    /// How many page slots a lane's run may name.
    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// How many recurrent layers this pool holds a block for.
    #[must_use]
    pub fn layers(&self) -> u32 {
        self.layers
    }

    /// Every byte this pool holds.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.slab.bytes() as u64
    }

    /// The device address of `plane`'s row `row` inside page slot
    /// `page_slot`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a page slot past the pool or a row past the
    /// page.
    pub fn row(&self, plane: Plane, page_slot: u32, row: u32) -> Result<u64> {
        if page_slot >= self.slots {
            return Err(Fault::Ceiling {
                what: "rs buffer page slots",
                need: u64::from(page_slot) + 1,
                have: u64::from(self.slots),
            });
        }
        if row >= self.page_tokens {
            return Err(Fault::Ceiling {
                what: "rs buffer page tokens",
                need: u64::from(row) + 1,
                have: u64::from(self.page_tokens),
            });
        }
        let block = (u64::from(plane.layer) * u64::from(self.slots) + u64::from(page_slot))
            * self.per_page;
        Ok(self.slab.ptr() + (block + plane.at + u64::from(row) * plane.width) * ELEMENT)
    }
}

fn unsized_plane(what: &str, id: ValueId) -> Fault {
    Fault::Unbound {
        what: format!(
            "{what} at value {}, whose row width is not a constant this shell can reserve a \
             buffered page for",
            id.0
        ),
    }
}

/// One value's declared row width, when it is a constant this shell can
/// reserve for.
fn width_of(trace: &Trace, id: ValueId) -> Option<u64> {
    let decl = trace.values.get(id.0 as usize)?;
    let Ty::Tensor { shape, .. } = &decl.ty else {
        return None;
    };
    match shape.last()? {
        Dim::Const(width) => Some(*width),
        _ => None,
    }
}

/// **The fold predicate and the accepted lengths, as one resident region**
/// (alto design §6's change (a), and §12 finding 4).
///
/// Four small vectors that are not staged inputs and must not be:
///
/// ```text
/// one        u32 == 1     the standing "committed" word
/// zero       u32 == 0     the standing "did not commit" word
/// commits    [lanes] u64  one commit-word ADDRESS per lane
/// indptr     [lanes+1] i32   the identity CSR
/// mask       [lanes] u8   what `mask_from_commit` writes and the scans read
/// commit_len [lanes] i32  where each request's accepted prefix ends
/// ```
///
/// **THE CSR IS THE IDENTITY, AND THAT IS A FINDING RATHER THAN A SHORTCUT.**
/// `channel::mask_from_commit` scatters a lane's commit word across the rows
/// the lane owns, through a row CSR — because a per-TOKEN predicate is what a
/// kv writer would want. The recurrent scans want a per-REQUEST one:
/// `attn/ssm.cuh`'s `row_persists(mask, r)` takes `r = blockIdx` over
/// requests in every chunked arm, so what the scan indexes is the lane and not
/// the token. Handing the kernel `indptr[l] = l` makes its scatter write
/// exactly one byte per lane, which is the shape the scan reads, with the
/// kernel unchanged.
///
/// **THE TWO STANDING WORDS ARE HOW A LANE WITH NO GUEST STAYS ALL-ONES.**
/// `mask_from_commit` reads a null pointer as "did not commit", which is the
/// right default for a channel and exactly the wrong one for a lane that
/// simply has no attachment — so an unattached lane points at `one` and folds,
/// and a lane whose verb is a buffered scatter points at `zero` and does not.
/// One kernel then carries both predicates, the pass's and the verb's, and
/// there is no second pass over the mask to fall out of step with the first.
#[derive(Debug)]
pub struct Predicate {
    region: Buffer,
    at_one: u64,
    at_zero: u64,
    at_commits: u64,
    at_indptr: u64,
    at_mask: u64,
    at_len: u64,
    lanes: u32,
}

impl Predicate {
    /// Carve the region one deployment's lane ceiling needs, and write the
    /// two standing words and the identity CSR into it once.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`](crate::Fault::Device) for the allocation or the two
    /// initial writes.
    pub fn reserve(max_lanes: u32) -> Result<Predicate> {
        let lanes = u64::from(max_lanes);
        let mut at = 0u64;
        let mut take = |bytes: u64| {
            let here = at;
            at += bytes.next_multiple_of(256);
            here
        };
        let at_one = take(4);
        let at_zero = take(4);
        let at_commits = take(lanes * 8);
        let at_indptr = take((lanes + 1) * 4);
        let at_mask = take(lanes);
        let at_len = take(lanes * 4);
        let mut region = Buffer::zeroed(usize::try_from(at).unwrap_or(usize::MAX))?;
        region.write(at_one, &1u32.to_le_bytes())?;
        let identity: Vec<u8> = (0..=max_lanes)
            .flat_map(|l| (l as i32).to_le_bytes())
            .collect();
        region.write(at_indptr, &identity)?;
        Ok(Predicate {
            region,
            at_one,
            at_zero,
            at_commits,
            at_indptr,
            at_mask,
            at_len,
            lanes: max_lanes,
        })
    }

    /// The standing "this pass committed" word — what an unattached lane's
    /// commit slot points at.
    #[must_use]
    pub fn always(&self) -> u64 {
        self.region.ptr() + self.at_one
    }

    /// The standing "this pass did not commit" word — what a lane whose verb
    /// is a buffered scatter points at.
    #[must_use]
    pub fn never(&self) -> u64 {
        self.region.ptr() + self.at_zero
    }

    /// Stage this fire's commit addresses and accepted lengths on `stream`.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] past the lane ceiling, [`Fault::Device`] for the
    /// copies.
    pub fn write(
        &mut self,
        stream: *mut core::ffi::c_void,
        commits: &[u64],
        lens: &[i32],
    ) -> Result<()> {
        if commits.len() as u64 > u64::from(self.lanes) {
            return Err(Fault::Ceiling {
                what: "rs fold predicates",
                need: commits.len() as u64,
                have: u64::from(self.lanes),
            });
        }
        let words: Vec<u8> = commits.iter().flat_map(|c| c.to_le_bytes()).collect();
        let at_commits = self.at_commits;
        let at_len = self.at_len;
        self.region.stage(stream, at_commits, &words)?;
        let lens: Vec<u8> = lens.iter().flat_map(|n| n.to_le_bytes()).collect();
        self.region.stage(stream, at_len, &lens)
    }

    /// Where the per-lane commit addresses live.
    #[must_use]
    pub fn commits(&self) -> u64 {
        self.region.ptr() + self.at_commits
    }

    /// Where the identity CSR lives.
    #[must_use]
    pub fn indptr(&self) -> u64 {
        self.region.ptr() + self.at_indptr
    }

    /// The fold predicate, as the handle a `RecurrentPool` seats.
    #[must_use]
    pub fn mask(&self, lanes: u32) -> kernels_cuda::Tensor {
        kernels_cuda::Tensor::new(self.region.ptr() + self.at_mask, lanes, 1, Dtype::U8)
    }

    /// The accepted lengths, as the handle a `RecurrentPool` seats.
    #[must_use]
    pub fn commit_len(&self, lanes: u32) -> kernels_cuda::Tensor {
        kernels_cuda::Tensor::new(self.region.ptr() + self.at_len, lanes, 1, Dtype::I32)
    }

    /// **Read the mask back**, for a gate that has to see the predicate the
    /// device wrote. Not on any fire path: it is a synchronous D2H.
    ///
    /// # Errors
    ///
    /// [`Fault::Device`] for the read.
    pub fn read_mask(&self, lanes: u32) -> Result<Vec<u8>> {
        let mut out = vec![0u8; lanes as usize];
        self.region.read(self.at_mask, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use model_dsl::Platform;

    /// **EVERY SKU IN THE CATALOG READS**, and the two that matter answer
    /// differently.
    ///
    /// [`read`](super::read) runs at LOAD, before any lane has asked for a
    /// recurrent verb — so a plan it cannot walk is a plan this shell refuses
    /// to load at all, and a family that grows a third in-projection plane
    /// would take every deployment of it down rather than refusing one fire.
    /// The whole catalog is swept for exactly that reason.
    ///
    /// Two answers, and both are correct answers:
    ///
    /// * `None` for a plan with no chunked gated-delta scan — every dense
    ///   text, and the KDA families until they are given the buffer
    ///   vocabulary. Such a load reserves no pool and a lane that names a
    ///   buffered verb against it is refused by name at the fire.
    /// * a reading for the GDN hybrids, whose per-token width must be the
    ///   conv's rows plus the gate prep's `[b | a]` — a number that is
    ///   nonzero and the SAME for every layer, which is what the pool's one
    ///   page-slot stride rests on.
    ///
    /// **AND AT LEAST ONE SKU MUST ANSWER THE SECOND WAY.** A sweep where
    /// every row read `None` would pass while proving nothing.
    #[test]
    fn every_sku_in_the_catalog_reads_its_buffered_planes_or_says_it_has_none() {
        let mut buffered = 0usize;
        let mut table: Vec<String> = Vec::new();
        for (sku, _, trace, _) in model::catalog() {
            let trace = trace(Platform::Cuda);
            let read = super::read(&trace, 16)
                .unwrap_or_else(|why| panic!("`{sku}` will not load: {why}"));
            match read {
                None => table.push(format!("  {sku}: no chunked recurrence")),
                Some((planes, per_token, layers)) => {
                    assert!(per_token > 0, "`{sku}` buffers zero elements a token");
                    assert_eq!(
                        planes.len(),
                        2 * layers as usize,
                        "`{sku}` reads {layers} recurrent layers and {} planes, and this \
                         shell buffers exactly two per layer",
                        planes.len(),
                    );
                    buffered += 1;
                    table.push(format!(
                        "  {sku}: {layers} layers x {per_token} elements a token"
                    ));
                }
            }
        }
        assert!(
            buffered > 0,
            "no SKU in the catalog buffers anything, so this sweep proves nothing:\n{}",
            table.join("\n"),
        );
        eprintln!("{}", table.join("\n"));
    }
}
