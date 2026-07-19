//! The neutral trace **builder** — the DSL crate's lowering core.
//!
//! [`Builder`] is the boundary-agnostic half of what used to be `ForwardPass`:
//! it takes descriptor-port bindings ([`bind_port`](Builder::bind_port)) and
//! stage closures ([`stage`](Builder::stage)), traces the closures once into
//! echo's canonical [`TraceContainer`], and runs the SDK span lints. It does
//! **not** bind (D6: the guest does not bind — `forward-pass.program` is the
//! authoritative gate); the author-facing `ForwardPass`/`Pipeline`/`WorkingSet`
//! lifetime objects live in `inferlet`, wrap the WIT resources, and drive this
//! builder.
//!
//! The subtle assembly machinery is preserved verbatim from the pre-A1
//! `ForwardPass::assemble`: gid re-key (interning order → declaration order),
//! `HostRole` derivation, terminal-output inference, and the reader auto-drain
//! drop. See [`crate`] A.5 invariants.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use pie_ptir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use pie_ptir::op::Op;
use pie_ptir::registry::{Port, Stage};

use crate::channel::Channel;
use crate::context::{self, ChannelRef, SinkCall};
use crate::error::{Span, TraceError, TraceErrors};

/// A descriptor-port source. Descriptor inputs are always channels; tensors
/// exist only inside traced stage closures.
pub struct PortInput(Channel);

impl From<&Channel> for PortInput {
    fn from(c: &Channel) -> PortInput {
        PortInput(c.clone())
    }
}
impl From<Channel> for PortInput {
    fn from(c: Channel) -> PortInput {
        PortInput(c)
    }
}

type StageClosure<'a> = Box<dyn Fn() + 'a>;

/// The neutral trace builder. Collects port bindings + stage closures, then
/// [`build`](Builder::build)s the canonical container.
pub struct Builder<'a> {
    ports: Vec<(Port, PortInput)>,
    stages: Vec<(Stage, StageClosure<'a>)>,
    vocab: u32,
    page_size: u32,
}

impl<'a> Builder<'a> {
    /// Create a neutral builder with engine-sourced trace constants.
    pub fn new(vocab: u32, page_size: u32) -> Builder<'a> {
        Builder {
            ports: Vec::new(),
            stages: Vec::new(),
            vocab,
            page_size,
        }
    }

    /// Bind a descriptor [`Port`] to a channel. Records the port's
    /// endpoint claim on the channel per its fixed consumption discipline
    /// ([`Port::consumes`]): the token-indexed family (`embed`, `positions`,
    /// `w_slot`/`w_off`) **takes**; geometry and masks **read** (§5.1). The
    /// claim drives host-role derivation and the span lints.
    #[track_caller]
    pub fn bind_port(&mut self, port: Port, source: impl Into<PortInput>) {
        let source = source.into();
        let span = Span::here();
        let mut st = source.0.state().borrow_mut();
        if port.consumes() {
            st.desc_takes.push(span);
        } else {
            st.desc_reads.push(span);
        }
        drop(st);
        self.ports.push((port, source));
    }

    /// Like [`bind_port`](Builder::bind_port), but WITHOUT recording the
    /// endpoint claim — for callers that already claimed eagerly at pass
    /// construction (`Channel::note_desc_claim`, F8).
    pub fn bind_port_recorded(&mut self, port: Port, source: impl Into<PortInput>) {
        self.ports.push((port, source.into()));
    }

    /// Attach a stage closure (traced once at [`build`](Builder::build)). A
    /// stage may be attached at most once; a second attach replaces the first.
    pub fn stage(&mut self, stage: Stage, body: impl Fn() + 'a) {
        if let Some(slot) = self.stages.iter_mut().find(|(s, _)| *s == stage) {
            slot.1 = Box::new(body);
        } else {
            self.stages.push((stage, Box::new(body)));
        }
    }

    /// Read-out rows for `intrinsics::logits()`: an explicit `Readout` channel,
    /// else the number of `EmbedIndptr` lanes.
    fn rows(&self) -> u32 {
        if let Some(channel) = self.channel_port(Port::Readout) {
            return (channel.shape().numel() as u32).max(1);
        }
        if let Some(channel) = self.channel_port(Port::EmbedIndptr) {
            return (channel.shape().numel() as u32).saturating_sub(1).max(1);
        }
        1
    }

    fn channel_port(&self, port: Port) -> Option<&Channel> {
        self.ports
            .iter()
            .find_map(|(bound, source)| (*bound == port).then_some(&source.0))
    }

    /// Trace + lint, returning the canonical [`Traced`] artifact: container
    /// bytes, dense-order channel identities, and names. Runs the SDK span
    /// lints only; authoritative validation is `forward-pass.program`'s result (D6).
    pub fn build(&self) -> Result<Traced, TraceErrors> {
        let rows = self.rows();
        let (result, channels) = crate::model::with_constants(self.vocab, self.page_size, || {
            context::with_session(|| self.record(rows))
        });
        let (stage_results, ports) = result;

        // The recorder interns channels in first-REFERENCE order (the order they
        // appear in the traced body, e.g. `embed(toks,…)` interns `toks` first).
        // But an inferlet DECLARES + indexes channels — seeds, host endpoints — in
        // DECLARATION order. Re-key the container to gid (declaration) order so the
        // two agree, remapping every channel reference (ChanTake / ChanRead /
        // ChanPut ops + descriptor `PortSource::Channel`). Without this, a
        // channel-0 [B,P] seed (e.g. beam `pages`) validates against whatever
        // channel happened to be referenced first (a [B] channel) → numel mismatch.
        let mut order: Vec<usize> = (0..channels.len()).collect();
        order.sort_by_key(|&i| channels[i].borrow().gid);
        let mut remap = vec![0u32; channels.len()];
        for (new_idx, &old_idx) in order.iter().enumerate() {
            remap[old_idx] = new_idx as u32;
        }
        let channels: Vec<ChannelRef> = order.iter().map(|&i| channels[i].clone()).collect();
        let stage_results: Vec<_> = stage_results
            .into_iter()
            .map(|mut r| {
                for op in &mut r.ops {
                    match op {
                        Op::ChanTake(c) | Op::ChanRead(c) => *c = remap[*c as usize],
                        Op::ChanPut { chan, .. } => *chan = remap[*chan as usize],
                        _ => {}
                    }
                }
                r
            })
            .collect();
        let ports: Vec<PortBinding> = ports
            .into_iter()
            .map(|mut p| {
                if let PortSource::Channel(ci) = &mut p.source {
                    *ci = remap[*ci as usize];
                }
                p
            })
            .collect();

        // Sink lint input (stage, sink).
        let sinks: Vec<(Stage, SinkCall)> = stage_results
            .iter()
            .flat_map(|r| r.sinks.iter().map(move |s| (r.stage, s.clone())))
            .collect();

        // Build echo's channel declarations with derived HostRole + seeded.
        let channel_decls: Vec<ChannelDecl> = channels
            .iter()
            .map(|c| {
                let st = c.borrow();
                let has_prog_put = !st.prog_puts.is_empty();
                let has_prog_consume = !st.prog_takes.is_empty() || !st.prog_reads.is_empty();
                let has_desc_use = !st.desc_takes.is_empty() || !st.desc_reads.is_empty();
                let has_host_put = !st.host_puts.is_empty();
                let host_consumes = !st.host_takes.is_empty() || !st.host_reads.is_empty();
                // A program-PRODUCED channel with NO program consumer (take/read),
                // NO descriptor binding, and NO host writer is a terminal OUTPUT the
                // host reads (e.g. beam `out`/`out_par`/`out_scr`): the guest's `take`
                // at runtime isn't visible at trace time, so infer Reader here.
                let is_terminal_output = has_prog_put
                    && !has_prog_consume
                    && !has_desc_use
                    && !has_host_put
                    && !st.seeded
                    && st.seed.is_none();
                // A seeded descriptor-only channel is replaceable through the
                // host `set` operation after bind. It therefore needs the same
                // device-visible Writer endpoint as an explicit host `put`,
                // even when its initial value came from the seed table.
                let seeded_descriptor_writer = st.seeded && has_desc_use && !has_prog_put;
                let host_role = if (has_host_put || seeded_descriptor_writer) && !has_prog_put {
                    HostRole::Writer
                } else if host_consumes && (!st.prog_takes.is_empty() || has_prog_put) {
                    // A host-consumed, pass-produced/loop-carried channel.
                    HostRole::Reader
                } else if is_terminal_output {
                    HostRole::Reader
                } else {
                    HostRole::None
                };
                let seeded = st.seeded || (has_host_put && has_prog_put);
                ChannelDecl {
                    shape: st.shape,
                    dtype: ChanDType::Concrete(st.dtype),
                    capacity: st.capacity,
                    host_role,
                    seeded,
                }
            })
            .collect();

        let stages: Vec<StageProgram> = stage_results
            .into_iter()
            .map(|r| StageProgram {
                stage: r.stage,
                ops: r.ops,
            })
            .collect();

        let mut ports = ports;
        ports.sort_by_key(|p| p.port as u8);

        let container = TraceContainer {
            externs: Vec::new(),
            names: Vec::new(),
            channels: channel_decls,
            ports,
            stages,
        };

        // SDK span lints (friendly, spans). Echo's authoritative bind lives on
        // the host at `forward-pass.program` (D6); native parity tests bind explicitly.
        let mut errs: Vec<TraceError> = Vec::new();
        crate::lint::lint(&channels, &sinks, &mut errs);
        if !errs.is_empty() {
            return Err(TraceErrors(errs));
        }

        let channel_order = channels.iter().map(|c| c.borrow().gid).collect();
        let channel_names = channels.iter().map(|c| c.borrow().name.clone()).collect();
        Ok(Traced {
            container,
            channel_order,
            channel_names,
        })
    }

    /// Assemble the raw container WITHOUT lint — for debugging emission.
    #[doc(hidden)]
    pub fn debug_container(&self) -> TraceContainer {
        let rows = self.rows();
        let (result, channels) = crate::model::with_constants(self.vocab, self.page_size, || {
            context::with_session(|| self.record(rows))
        });
        let (stage_results, ports) = result;
        let channel_decls: Vec<ChannelDecl> = channels
            .iter()
            .map(|c| {
                let st = c.borrow();
                ChannelDecl {
                    shape: st.shape,
                    dtype: ChanDType::Concrete(st.dtype),
                    capacity: st.capacity,
                    host_role: HostRole::None,
                    seeded: st.seeded,
                }
            })
            .collect();
        let stages = stage_results
            .into_iter()
            .map(|r| StageProgram {
                stage: r.stage,
                ops: r.ops,
            })
            .collect();
        let mut ports = ports;
        ports.sort_by_key(|p| p.port as u8);
        TraceContainer {
            externs: Vec::new(),
            names: Vec::new(),
            channels: channel_decls,
            ports,
            stages,
        }
    }

    /// Intern descriptor-port channels + trace each present stage (inside a session).
    fn record(&self, rows: u32) -> (Vec<context::StageResult>, Vec<PortBinding>) {
        let mut ports: Vec<PortBinding> = Vec::new();
        for (port, source) in &self.ports {
            let src = PortSource::Channel(context::intern_channel(source.0.state()));
            ports.push(PortBinding {
                port: *port,
                source: src,
            });
        }

        // Trace stages in canonical stage order (byte-stable container.stages).
        let mut results = Vec::new();
        for stage in [
            Stage::Prologue,
            Stage::OnAttnProj,
            Stage::OnAttn,
            Stage::Epilogue,
        ] {
            let Some((_, body)) = self.stages.iter().find(|(s, _)| *s == stage) else {
                continue;
            };
            let res = context::trace_stage(stage, rows, body);
            results.push(res);
        }
        (results, ports)
    }
}

/// A traced, linted forward pass: echo's canonical [`TraceContainer`] plus the
/// dense-order channel identities (gids) and names. Identity is the C3 hash
/// (FNV-1a over the canonical container bytes); binding is the host's job (D6).
#[derive(Debug)]
pub struct Traced {
    container: TraceContainer,
    channel_order: Vec<u64>,
    channel_names: Vec<String>,
}

impl Traced {
    /// The canonical trace container.
    pub fn container(&self) -> &TraceContainer {
        &self.container
    }
    /// Program-set identity hash (FNV-1a over the canonical container bytes, C3).
    pub fn identity_hash(&self) -> u64 {
        self.container.hash()
    }
    /// The canonical trace-container bytes.
    pub fn encode(&self) -> Vec<u8> {
        self.container.encode()
    }
    /// Channel identities (gids) by dense index — the builder↔bridge contract:
    /// the WIT channel-handle list must follow exactly this order (A.5).
    pub fn channel_order(&self) -> &[u64] {
        &self.channel_order
    }
    /// SDK channel names by dense index (debug).
    pub fn channel_names(&self) -> &[String] {
        &self.channel_names
    }
}
