//! Device-geometry page leasing (grant/reclaim/free-list) and device-geometry
//! pass detection. Physical page ids only; the engine needs no slot->physical
//! table. Pin float is bounded by `(run-ahead depth) x B` pages.

/// Tracks pages granted to each in-flight fire (FIFO); unused fresh grants
/// are reclaimed as fires commit, and everything is reclaimed on drop.
#[derive(Debug, Default)]
pub struct PageLease {
    /// Beam / lane width: fresh pages granted per fire.
    pub b: usize,
    /// Free-list of reclaimed physical page ids, drawn before allocating anew.
    free: Vec<u32>,
    /// Per-in-flight-fire grants, FIFO. `pending[i]` = the `b` ids granted to
    /// the i-th oldest un-reclaimed fire.
    pending: std::collections::VecDeque<Vec<u32>>,
    /// Fire-0 seed pages (one per lane), reclaimed only on pass drop.
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

    /// Record the fire-0 seed pages; reclaimed on drop.
    pub fn seed(&mut self, pages: Vec<u32>) {
        self.seed_pages = pages;
    }

    /// Draw `b` fresh page ids: free-list first, then `alloc`. Records the
    /// grant on the pending FIFO for reclaim after the fire commits.
    pub fn grant<F: FnMut() -> u32>(&mut self, mut alloc: F) -> Vec<u32> {
        let mut pages = Vec::with_capacity(self.b);
        for _ in 0..self.b {
            pages.push(self.free.pop().unwrap_or_else(&mut alloc));
        }
        self.pending.push_back(pages.clone());
        pages
    }

    /// Reclaim the oldest in-flight fire's unused fresh grants, per lane
    /// `w_cont` (true = continued a shared tail, fresh page unused; false =
    /// forked onto it, keep live). Returns the reclaimed ids. No-op if empty.
    pub fn reclaim_after_fire(&mut self, w_cont: &[bool]) -> Vec<u32> {
        let Some(grant) = self.pending.pop_front() else {
            return Vec::new();
        };
        let mut reclaimed = Vec::new();
        for (lane, page) in grant.into_iter().enumerate() {
            if w_cont.get(lane).copied().unwrap_or(false) {
                self.free.push(page);
                reclaimed.push(page);
            }
        }
        reclaimed
    }

    /// Reclaim every page held (pass drop / failure). Returns the freed ids.
    pub fn reclaim_all(&mut self) -> Vec<u32> {
        let mut all = Vec::new();
        while let Some(grant) = self.pending.pop_front() {
            all.extend(grant);
        }
        all.extend(std::mem::take(&mut self.seed_pages));
        all.append(&mut self.free);
        all
    }

    // In-flight (un-reclaimed) fire count; test-only.
    #[cfg(test)]
    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }
}

/// Physical-page leasing + channel bookkeeping for a device-geometry pass.
pub struct DevGeo {
    /// The physical-page lease (grant / reclaim / free-list bookkeeping).
    pub lease: PageLease,
    /// Beam / lane width: fresh grants per fire.
    pub b: usize,
    /// Dense channel index of the host-writer `fresh`-page input channel.
    pub fresh_dense: usize,
    /// Dense channel index of the `w_cont` host-reader output ([b] bool).
    pub w_cont_dense: usize,
    /// Whether the program binds an `AttnMask` descriptor channel. Unread in
    /// production; kept for tests.
    #[allow(dead_code)]
    pub has_mask: bool,
    /// Pool-owned geometry: the program reserves its own page pool and
    /// resolves every write target in-graph, so `lease`/`fresh_dense`/
    /// `w_cont_dense` are inert.
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

/// Detect a pool-owned device-geometry pass: every descriptor port is bound
/// to a channel the program itself re-publishes, so the engine can resolve
/// geometry from the channel cells with no page-lease handshake. Requires a
/// dense device `AttnMask`; mask-free decode loops keep the envelope path.
///
/// Returns the lane count (`EmbedTokens` extent).
pub fn detect_pooled_device_geometry(
    container: &eta_ir::container::TraceContainer,
) -> Option<usize> {
    use eta_ir::container::{ChanDType, PortSource};
    use eta_ir::registry::Port;
    use eta_ir::types::Dtype;

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
                matches!(op, eta_ir::op::Op::ChanPut { chan, .. } if *chan as usize == channel)
            })
        })
    };

    let mask = channel_of(Port::AttnMask)?;
    if !matches!(
        container.channels.get(mask)?.dtype,
        ChanDType::Concrete(Dtype::Bool)
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
        ChanDType::Concrete(Dtype::I32) | ChanDType::Concrete(Dtype::U32)
    ) {
        return None;
    }
    Some(dims[0] as usize)
}

/// Detect a device-geometry pass: `WSlot`/`WOff` write descriptors bind
/// device-produced channels, and `Pages` is `[B, P]` (`P > 1`). Returns
/// `(B, fresh_dense, w_cont_dense)`; `None` for an ordinary decode.
pub fn detect_device_geometry(
    container: &eta_ir::container::TraceContainer,
) -> Option<(usize, usize, usize)> {
    use eta_ir::container::HostRole;
    use eta_ir::container::{ChanDType, PortSource};
    use eta_ir::registry::Port;
    use eta_ir::types::Dtype;

    let has_write_desc = container
        .ports
        .iter()
        .any(|p| matches!(p.port, Port::WSlot | Port::WOff));
    if !has_write_desc {
        return None;
    }
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

    let fresh_dense = container
        .channels
        .iter()
        .position(|c| c.host_role == HostRole::Writer)?;
    let w_cont_dense = container.channels.iter().position(|c| {
        c.host_role == HostRole::Reader && matches!(c.dtype, ChanDType::Concrete(Dtype::Bool))
    })?;
    Some((b, fresh_dense, w_cont_dense))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Monotonic page-id allocator for tests.
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
        // lane 0 continued (reclaim), lane 1 forked (keep).
        let reclaimed = lease.reclaim_after_fire(&[true, false]);
        assert_eq!(
            reclaimed,
            vec![1000],
            "only the continuing lane's fresh page"
        );
        assert_eq!(lease.in_flight(), 0, "the oldest fire is retired");
    }

    use eta_ir::container::{ChanDType, ChannelDecl, HostRole, PortBinding, PortSource};
    use eta_ir::container::{StageProgram, TraceContainer};
    use eta_ir::registry::{Port, Stage};
    use eta_ir::types::{Dtype, Shape};

    fn ch(shape: Shape, dtype: Dtype, role: HostRole) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded: false,
        }
    }

    // Channels: 0 pages[b,p], 1 w_slot[b], 2 w_off[b], 3 fresh[b] (Writer),
    // 4 w_cont[b] bool (Reader).
    fn devgeo_container(b: u32, p: u32) -> TraceContainer {
        TraceContainer {
            names: vec![],
            channels: vec![
                ch(Shape::matrix(b, p), Dtype::U32, HostRole::None), // 0 pages
                ch(Shape::vector(b), Dtype::U32, HostRole::None),    // 1 w_slot
                ch(Shape::vector(b), Dtype::U32, HostRole::None),    // 2 w_off
                ch(Shape::vector(b), Dtype::U32, HostRole::Writer),  // 3 fresh
                ch(Shape::vector(b), Dtype::Bool, HostRole::Reader), // 4 w_cont
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
        // Plain decode: KvLen only, no WSlot/WOff.
        let c = TraceContainer {
            names: vec![],
            channels: vec![ch(Shape::vector(1), Dtype::I32, HostRole::None)],
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
        // Pages is [b,1] (p == 1), not a multi-page beam.
        let mut c = devgeo_container(2, 1);
        c.channels[0] = ch(Shape::matrix(2, 1), Dtype::U32, HostRole::None);
        assert!(
            detect_device_geometry(&c).is_none(),
            "P == 1 ⇒ not device-geometry"
        );
    }
}

#[cfg(test)]
mod pooled_tests {
    use super::detect_pooled_device_geometry;
    use eta_ir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use eta_ir::op::Op;
    use eta_ir::registry::{Port, Stage};
    use eta_ir::types::{Dtype, Shape};

    fn chan(shape: Shape, dtype: Dtype) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    // Every descriptor port bound to a channel the epilogue re-publishes.
    fn masked_decode(lanes: u32, pool: u32) -> TraceContainer {
        let decls = [
            (Port::EmbedTokens, Shape::vector(lanes), Dtype::I32),
            (Port::Positions, Shape::vector(lanes), Dtype::U32),
            (Port::Pages, Shape::vector(lanes * 2), Dtype::U32),
            (Port::PageIndptr, Shape::vector(lanes + 1), Dtype::U32),
            (Port::KvLen, Shape::vector(lanes), Dtype::U32),
            (Port::WSlot, Shape::vector(lanes), Dtype::U32),
            (Port::WOff, Shape::vector(lanes), Dtype::U32),
            (Port::AttnMask, Shape::matrix(lanes, pool), Dtype::Bool),
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
