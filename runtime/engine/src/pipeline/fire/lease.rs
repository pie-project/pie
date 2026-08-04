//! Device-geometry page leasing (plan W3.3) + device-geometry pass detection.
//!
//! A device-geometry pass (its `pages`/`w_slot` descriptor ports are produced by
//! the program on-device — the generalized §6.2 beam) needs the runtime to grant
//! PHYSICAL pages the device writes new-token K/V into, since the host no longer
//! knows the per-fire geometry. Physical-space page ids from the start (§3.2):
//! the runtime seeds fire 0's pages and grants `B` fresh physical ids per submit;
//! the whole loop stays physical and the driver needs no slot→physical table.
//!
//! Lifecycle:
//!   * fire 0: `B` pages granted as the seed (one live page per lane).
//!   * each submit: `B` fresh pages granted, delivered to the program as a
//!     host-put on its fresh-page input channel (D1-coalesced).
//!   * after a fire commits: the harvested `w_cont` (per-lane "continued a shared
//!     tail in place" vs "forked a fresh page") says which fresh grants were
//!     UNUSED (a continuing heir did not consume its fresh page) — reclaim them.
//!   * pass drop / failure: reclaim everything.
//!
//! Pin float is bounded by `(run-ahead depth) × B` pages (each in-flight fire
//! holds at most its `B` fresh grants until it commits + reclaims).
//!
//! [`PageLease`] is the PURE bookkeeping half (grant / reclaim / free-list),
//! unit-tested here; the working-set page allocation + the host-put wiring live
//! in `pipeline/fire/mod.rs` (they need the arena / ws, which need a device).
//! [`detect_device_geometry`] + [`DevGeo`] are the paired bind-time detector
//! and per-pass lease bundle.
//!
//! Everything here — [`PageLease::new`]/[`seed`](PageLease::seed)/
//! [`grant`](PageLease::grant)/[`reclaim_after_fire`](PageLease::reclaim_after_fire)/
//! [`reclaim_all`](PageLease::reclaim_all), every [`DevGeo`] field, and
//! [`detect_device_geometry`] — is on the production device-geometry path in
//! `inferlet::host::forward` and `pipeline::fire`. The two items that are
//! not carry their own annotated `allow`.

/// A per-device-geometry-pass physical page lease. Tracks the pages granted to
/// each in-flight fire (FIFO) so unused fresh grants are reclaimed as fires
/// commit, and everything is reclaimed on drop.
#[derive(Debug, Default)]
pub struct PageLease {
    /// Beam / lane width `B` — the number of fresh pages granted per fire.
    pub b: usize,
    /// The reusable free-list of physical page ids reclaimed from prior fires
    /// (drawn from before allocating anew, bounding total page use).
    free: Vec<u32>,
    /// Per-in-flight-fire grants, submission order (FIFO). `pending[i]` are the
    /// `B` fresh page ids granted to the i-th oldest un-reclaimed fire.
    pending: std::collections::VecDeque<Vec<u32>>,
    /// The live seed pages (fire 0's one-page-per-lane grant), reclaimed only on
    /// pass drop.
    seed_pages: Vec<u32>,
}

impl PageLease {
    /// A fresh lease for `b` lanes.
    pub fn new(b: usize) -> Self {
        PageLease {
            b,
            free: Vec::new(),
            pending: std::collections::VecDeque::new(),
            seed_pages: Vec::new(),
        }
    }

    /// Record the fire-0 seed pages (one live page per lane); reclaimed on drop.
    pub fn seed(&mut self, pages: Vec<u32>) {
        self.seed_pages = pages;
    }

    /// Draw `B` fresh physical page ids for a submit: reuse the free-list first,
    /// then mint fresh ids via `alloc` (a monotonic page-id source or ws-backed
    /// allocator). Records the grant on the pending FIFO so it can be reclaimed
    /// after the fire commits.
    pub fn grant<F: FnMut() -> u32>(&mut self, mut alloc: F) -> Vec<u32> {
        let mut pages = Vec::with_capacity(self.b);
        for _ in 0..self.b {
            pages.push(self.free.pop().unwrap_or_else(&mut alloc));
        }
        self.pending.push_back(pages.clone());
        pages
    }

    /// Reclaim the UNUSED fresh grants of the oldest in-flight fire after it
    /// commits, using its harvested per-lane `w_cont` (`true` = the lane
    /// CONTINUED a shared tail in place, so its fresh page went unused →
    /// reclaimable; `false` = the lane FORKED onto its fresh page → keep it live).
    /// Returns the reclaimed ids (re-added to the free-list). No-op if no fire is
    /// pending.
    pub fn reclaim_after_fire(&mut self, w_cont: &[bool]) -> Vec<u32> {
        let Some(grant) = self.pending.pop_front() else {
            return Vec::new();
        };
        let mut reclaimed = Vec::new();
        for (lane, page) in grant.into_iter().enumerate() {
            // A continuing heir (w_cont true) didn't consume its fresh page.
            if w_cont.get(lane).copied().unwrap_or(false) {
                self.free.push(page);
                reclaimed.push(page);
            }
            // else: the lane forked onto `page` — it's now a live page (tracked
            // by the program's geometry, not the lease's pending set).
        }
        reclaimed
    }

    /// Reclaim EVERY page still held by the lease (pass drop / failure): all
    /// pending fires' grants + the seed pages. Returns the freed ids.
    pub fn reclaim_all(&mut self) -> Vec<u32> {
        let mut all = Vec::new();
        while let Some(grant) = self.pending.pop_front() {
            all.extend(grant);
        }
        all.extend(std::mem::take(&mut self.seed_pages));
        all.append(&mut self.free);
        all
    }

    /// Number of in-flight (un-reclaimed) fires — the pin-float depth × B
    /// bound. Asserted by this module's unit tests; no production reader.
    #[cfg(test)]
    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }
}

/// Physical-page leasing + channel bookkeeping for a device-geometry pass
/// (Track B / plan W3.3). The runtime seeds fire 0's `B` pages and grants `B`
/// fresh physical ids per submit (delivered as a host-put on the `fresh`
/// channel); after a fire commits it reclaims the UNUSED grants of continuing
/// heirs (harvested `w_cont`), and everything on drop.
pub struct DevGeo {
    /// The physical-page lease (grant / reclaim / free-list bookkeeping).
    pub lease: PageLease,
    /// Beam / lane width `B` — the fresh grants per fire.
    pub b: usize,
    /// Dense channel index of the host-writer `fresh`-page input channel — where
    /// the runtime injects each fire's grant.
    pub fresh_dense: usize,
    /// Dense channel index of the `w_cont` host-reader output ([B] bool) — read
    /// at finalize to reclaim continuing heirs' unused fresh pages.
    pub w_cont_dense: usize,
    /// The program binds an `AttnMask` descriptor channel (dense per-cell
    /// mask). Detected at bind and asserted by tests. The SOLO scheduling
    /// rule it names — the driver's composed multi-program batch does not
    /// merge dense device masks with other programs' geometry (v1 scope) —
    /// is carried by [`crate::pipeline::instance::BoundForwardPass`]'s
    /// `dense_mask` instead, which is the same binding read off the same
    /// ports and is available on every fire, not just the leased ones. This
    /// copy is redundant; the `allow` marks that rather than hiding it.
    #[allow(dead_code)]
    pub has_mask: bool,
    /// POOL-OWNED device geometry: the program reserved its own page pool and
    /// resolves every write target in-graph, so there is no `fresh`/`w_cont`
    /// leasing handshake. `lease`, `fresh_dense` and `w_cont_dense` are inert;
    /// the fire prepares explicit KV over the whole reserved span instead.
    pub pooled: bool,
}

impl DevGeo {
    /// A pool-owned device-geometry pass over `lanes` request rows.
    pub fn pooled(lanes: usize, has_mask: bool) -> Self {
        DevGeo {
            lease: PageLease::new(0),
            b: lanes,
            fresh_dense: usize::MAX,
            w_cont_dense: usize::MAX,
            has_mask,
            pooled: true,
        }
    }
}

/// Detect a POOL-OWNED device-geometry pass — the driver-side
/// `is_loop_carried_explicit_geometry_trace` contract: every descriptor port is
/// bound to a channel that the program itself re-publishes, so the driver can
/// resolve the complete geometry from the channel cells with no page-lease
/// handshake and no host replay.
///
/// This is the only executable home for a decode loop that carries a DENSE
/// device `AttnMask` (sliding-window / attention-sink / beam ancestry): the
/// DecodeEnvelope class composes batched lanes and has no per-lane mask state,
/// and the Host class cannot derive a device-sampled token. The mask is
/// required here so that mask-free decode loops keep the (batched, faster)
/// envelope path.
///
/// Returns the lane count (`EmbedTokens` extent).
pub fn detect_pooled_device_geometry(
    container: &pie_ir::container::TraceContainer,
) -> Option<usize> {
    use pie_ir::container::{ChanDType, PortSource};
    use pie_ir::registry::Port;
    use pie_ir::types::DType;

    let channel_of = |port: Port| {
        container
            .ports
            .iter()
            .find_map(|binding| match &binding.source {
                PortSource::Channel(channel) if binding.port == port => Some(*channel as usize),
                _ => None,
            })
    };
    let republished = |channel: usize| {
        container.stages.iter().any(|stage| {
            stage.ops.iter().any(|op| {
                matches!(op, pie_ir::op::Op::ChanPut { chan, .. } if *chan as usize == channel)
            })
        })
    };

    // A dense bool `AttnMask` is what rules out every other class.
    let mask = channel_of(Port::AttnMask)?;
    if !matches!(
        container.channels.get(mask)?.dtype,
        ChanDType::Concrete(DType::Bool)
    ) {
        return None;
    }
    if !republished(mask) {
        return None;
    }

    for port in [
        Port::EmbedTokens,
        Port::Positions,
        Port::Pages,
        Port::PageIndptr,
        Port::KvLen,
        Port::WSlot,
        Port::WOff,
    ] {
        let channel = channel_of(port)?;
        if !republished(channel) {
            return None;
        }
    }

    let tokens = container.channels.get(channel_of(Port::EmbedTokens)?)?;
    let dims = tokens.shape.dims();
    if dims.len() != 1 || dims[0] == 0 {
        return None;
    }
    if !matches!(
        tokens.dtype,
        ChanDType::Concrete(DType::I32) | ChanDType::Concrete(DType::U32)
    ) {
        return None;
    }
    Some(dims[0] as usize)
}

/// Detect a device-geometry pass: its geometry ports (`WSlot`/`WOff` write
/// descriptors — beam-specific; an ordinary decode carries the same ports but
/// keeps them host-derivable) bind DEVICE-produced channels, and `Pages` is
/// `[B, P]` (`P > 1`). Returns `(B, fresh_dense, w_cont_dense)`: the single
/// host-writer channel is `fresh`; the host-reader `[B]` bool channel is
/// `w_cont` (the reclaim signal). `None` for an ordinary decode.
pub fn detect_device_geometry(
    container: &pie_ir::container::TraceContainer,
) -> Option<(usize, usize, usize)> {
    use pie_ir::container::HostRole;
    use pie_ir::container::{ChanDType, PortSource};
    use pie_ir::registry::Port;
    use pie_ir::types::DType;

    let has_write_desc = container
        .ports
        .iter()
        .any(|p| matches!(p.port, Port::WSlot | Port::WOff));
    if !has_write_desc {
        return None;
    }
    // B from the [B, P] channel bound to the `Pages` port (P > 1 for a beam).
    let pages_ch = container
        .ports
        .iter()
        .find_map(|p| match (&p.port, &p.source) {
            (Port::Pages, PortSource::Channel(c)) => Some(*c as usize),
            _ => None,
        })?;
    let dims = container.channels.get(pages_ch)?.shape.dims();
    let b = if dims.len() == 2 && dims[1] > 1 {
        dims[0] as usize
    } else {
        return None;
    };

    // fresh = the single host-Writer channel; w_cont = the host-Reader bool.
    let fresh_dense = container
        .channels
        .iter()
        .position(|c| c.host_role == HostRole::Writer)?;
    let w_cont_dense = container.channels.iter().position(|c| {
        c.host_role == HostRole::Reader && matches!(c.dtype, ChanDType::Concrete(DType::Bool))
    })?;
    Some((b, fresh_dense, w_cont_dense))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A monotonic page-id allocator for tests (mints 1000, 1001, …).
    fn allocator() -> impl FnMut() -> u32 {
        let mut next = 1000u32;
        move || {
            let id = next;
            next += 1;
            id
        }
    }

    #[test]
    fn grant_mints_b_pages_and_tracks_in_flight() {
        let mut lease = PageLease::new(2);
        let mut alloc = allocator();
        let g0 = lease.grant(&mut alloc);
        assert_eq!(g0, vec![1000, 1001], "B=2 fresh pages");
        assert_eq!(lease.in_flight(), 1);
        let g1 = lease.grant(&mut alloc);
        assert_eq!(g1, vec![1002, 1003], "next fire mints the next B");
        assert_eq!(lease.in_flight(), 2, "run-ahead: two fires in flight");
    }

    #[test]
    fn reclaim_returns_only_continued_lanes() {
        let mut lease = PageLease::new(2);
        let mut alloc = allocator();
        let _g = lease.grant(&mut alloc); // [1000, 1001]
        // lane 0 continued (heir, fresh unused → reclaim), lane 1 forked (keep).
        let reclaimed = lease.reclaim_after_fire(&[true, false]);
        assert_eq!(
            reclaimed,
            vec![1000],
            "only the continuing lane's fresh page"
        );
        assert_eq!(lease.in_flight(), 0, "the oldest fire is retired");
    }

    #[test]
    fn reclaimed_pages_are_reused_before_minting() {
        let mut lease = PageLease::new(2);
        let mut alloc = allocator();
        let _g0 = lease.grant(&mut alloc); // [1000, 1001]
        lease.reclaim_after_fire(&[true, true]); // both continued → 1000,1001 freed
        // Next grant reuses the free-list (LIFO) before minting.
        let g1 = lease.grant(&mut alloc);
        assert_eq!(
            g1,
            vec![1001, 1000],
            "reused freed pages, none newly minted"
        );
    }

    #[test]
    fn reclaim_all_frees_pending_and_seed() {
        let mut lease = PageLease::new(2);
        let mut alloc = allocator();
        lease.seed(vec![500, 501]);
        let _g0 = lease.grant(&mut alloc); // [1000,1001]
        let _g1 = lease.grant(&mut alloc); // [1002,1003]
        let mut all = lease.reclaim_all();
        all.sort();
        assert_eq!(
            all,
            vec![500, 501, 1000, 1001, 1002, 1003],
            "everything freed on drop"
        );
        assert_eq!(lease.in_flight(), 0);
    }

    use pie_ir::container::{ChanDType, ChannelDecl, HostRole, PortBinding, PortSource};
    use pie_ir::container::{StageProgram, TraceContainer};
    use pie_ir::registry::{Port, Stage};
    use pie_ir::types::{DType, Shape};

    fn ch(shape: Shape, dtype: DType, role: HostRole) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded: false,
        }
    }

    /// A minimal device-geometry container: a `[B,P]` Pages channel, WSlot/WOff
    /// write descriptors, one host-Writer (`fresh`) + one host-Reader bool
    /// (`w_cont`). Channels: 0 pages[B,P], 1 w_slot[B], 2 w_off[B], 3 fresh[B]
    /// (Writer), 4 w_cont[B] bool (Reader).
    fn devgeo_container(b: u32, p: u32) -> TraceContainer {
        TraceContainer {
            names: vec![],
            channels: vec![
                ch(Shape::matrix(b, p), DType::U32, HostRole::None), // 0 pages
                ch(Shape::vector(b), DType::U32, HostRole::None),    // 1 w_slot
                ch(Shape::vector(b), DType::U32, HostRole::None),    // 2 w_off
                ch(Shape::vector(b), DType::U32, HostRole::Writer),  // 3 fresh
                ch(Shape::vector(b), DType::Bool, HostRole::Reader), // 4 w_cont
            ],
            ports: vec![
                PortBinding {
                    port: Port::Pages,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::WSlot,
                    source: PortSource::Channel(1),
                },
                PortBinding {
                    port: Port::WOff,
                    source: PortSource::Channel(2),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
            externs: vec![],
        }
    }

    #[test]
    fn detect_device_geometry_identifies_b_fresh_and_wcont() {
        let c = devgeo_container(2, 3);
        let (b, fresh, w_cont) = detect_device_geometry(&c).expect("device-geometry pass");
        assert_eq!(b, 2, "B from the [B,P] Pages channel");
        assert_eq!(fresh, 3, "fresh = the single host-Writer channel");
        assert_eq!(w_cont, 4, "w_cont = the host-Reader bool channel");
    }

    #[test]
    fn detect_device_geometry_rejects_plain_decode() {
        // A plain decode: KvLen only (no WSlot/WOff write descriptors), P == 1.
        let c = TraceContainer {
            names: vec![],
            channels: vec![ch(Shape::vector(1), DType::I32, HostRole::None)],
            ports: vec![PortBinding {
                port: Port::KvLen,
                source: PortSource::Channel(0),
            }],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
            externs: vec![],
        };
        assert!(
            detect_device_geometry(&c).is_none(),
            "no WSlot/WOff ⇒ not device-geometry"
        );
    }

    #[test]
    fn detect_device_geometry_rejects_single_page_width() {
        // WSlot/WOff present but Pages is [B,1] (P == 1) — not a multi-page beam.
        let mut c = devgeo_container(2, 1);
        // pages [B,1]
        c.channels[0] = ch(Shape::matrix(2, 1), DType::U32, HostRole::None);
        assert!(
            detect_device_geometry(&c).is_none(),
            "P == 1 ⇒ not device-geometry"
        );
    }
}

#[cfg(test)]
mod pooled_tests {
    use super::detect_pooled_device_geometry;
    use pie_ir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ir::op::Op;
    use pie_ir::registry::{Port, Stage};
    use pie_ir::types::{DType, Shape};

    fn chan(shape: Shape, dtype: DType) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    /// A masked device-carried decode loop: every descriptor port is bound to a
    /// channel the epilogue re-publishes (the driver's
    /// `is_loop_carried_explicit_geometry_trace` contract).
    fn masked_decode(lanes: u32, pool: u32) -> TraceContainer {
        let decls = [
            (Port::EmbedTokens, Shape::vector(lanes), DType::I32),
            (Port::Positions, Shape::vector(lanes), DType::U32),
            (Port::Pages, Shape::vector(lanes * 2), DType::U32),
            (Port::PageIndptr, Shape::vector(lanes + 1), DType::U32),
            (Port::KvLen, Shape::vector(lanes), DType::U32),
            (Port::WSlot, Shape::vector(lanes), DType::U32),
            (Port::WOff, Shape::vector(lanes), DType::U32),
            (Port::AttnMask, Shape::matrix(lanes, pool), DType::Bool),
        ];
        let mut container = TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![],
            ports: vec![],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
        };
        for (port, shape, dtype) in decls {
            let index = container.channels.len() as u32;
            container.channels.push(chan(shape, dtype));
            container.ports.push(PortBinding {
                port,
                source: PortSource::Channel(index),
            });
            container.stages[0].ops.push(Op::ChanPut {
                chan: index,
                value: 0,
            });
        }
        container
    }

    #[test]
    fn masked_loop_carried_decode_is_pooled_device_geometry() {
        assert_eq!(
            detect_pooled_device_geometry(&masked_decode(1, 128)),
            Some(1)
        );
        assert_eq!(
            detect_pooled_device_geometry(&masked_decode(2, 128)),
            Some(2)
        );
    }

    #[test]
    fn a_mask_free_decode_keeps_the_envelope_path() {
        let mut container = masked_decode(1, 128);
        let mask = container
            .ports
            .iter()
            .position(|binding| binding.port == Port::AttnMask)
            .expect("mask port");
        container.ports.remove(mask);
        assert_eq!(
            detect_pooled_device_geometry(&container),
            None,
            "without a dense device mask the decode envelope class still applies"
        );
    }

    #[test]
    fn a_host_driven_descriptor_is_not_pooled_device_geometry() {
        let mut container = masked_decode(1, 128);
        let pages = container
            .ports
            .iter()
            .find_map(|binding| match (&binding.port, &binding.source) {
                (Port::Pages, PortSource::Channel(channel)) => Some(*channel),
                _ => None,
            })
            .expect("pages channel");
        container.stages[0]
            .ops
            .retain(|op| !matches!(op, Op::ChanPut { chan, .. } if *chan == pages));
        assert_eq!(
            detect_pooled_device_geometry(&container),
            None,
            "a port the program does not re-publish is not device-resolvable"
        );
    }
}
