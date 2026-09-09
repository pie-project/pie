use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::registry::{Port, Stage};

use crate::channel::Channel;
use crate::context::{self, ChannelRef, SinkCall};
use crate::error::{Span, TraceError, TraceErrors};

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

pub struct Builder<'a> {
    ports: Vec<(Port, PortInput)>,
    stages: Vec<(Stage, StageClosure<'a>)>,
    vocab: u32,
    page_size: u32,
    rows: Option<u32>,
}

impl<'a> Builder<'a> {
    pub fn new(vocab: u32, page_size: u32) -> Builder<'a> {
        Builder {
            ports: Vec::new(),
            stages: Vec::new(),
            vocab,
            page_size,
            rows: None,
        }
    }

    pub fn rows_hint(&mut self, rows: u32) {
        self.rows = Some(rows.max(1));
    }

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

    pub fn bind_port_recorded(&mut self, port: Port, source: impl Into<PortInput>) {
        self.ports.push((port, source.into()));
    }

    pub fn stage(&mut self, stage: Stage, body: impl Fn() + 'a) {
        if let Some(slot) = self.stages.iter_mut().find(|(s, _)| *s == stage) {
            slot.1 = Box::new(body);
        } else {
            self.stages.push((stage, Box::new(body)));
        }
    }

    fn rows(&self) -> u32 {
        if let Some(rows) = self.rows {
            return rows;
        }
        if let Some(channel) = self.channel_port(Port::Readout) {
            return saturating_rows(channel.shape().numel()).max(1);
        }
        if let Some(channel) = self.channel_port(Port::EmbedIndptr) {
            return saturating_rows(channel.shape().numel())
                .saturating_sub(1)
                .max(1);
        }
        1
    }

    fn channel_port(&self, port: Port) -> Option<&Channel> {
        self.ports
            .iter()
            .find_map(|(bound, source)| (*bound == port).then_some(&source.0))
    }

    pub fn build(&self) -> Result<Traced, TraceErrors> {
        let rows = self.rows();
        let (result, channels, names, authoring) =
            crate::model::with_constants(self.vocab, self.page_size, || {
                context::with_session(|| self.record(rows))
            });
        if !authoring.is_empty() {
            return Err(TraceErrors(authoring));
        }
        let (stage_results, ports) = result;

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
                    if let Some(chan) = op.channel_mut() {
                        *chan = remap[*chan as usize];
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

        let mut name_order: Vec<usize> = (0..names.len()).collect();
        name_order.sort_by(|&a, &b| names[a].cmp(&names[b]));
        let mut name_remap = vec![0u16; names.len()];
        for (new_idx, &old_idx) in name_order.iter().enumerate() {
            name_remap[old_idx] = new_idx as u16;
        }
        let names: Vec<String> = name_order.iter().map(|&i| names[i].clone()).collect();
        let stage_results: Vec<_> = stage_results
            .into_iter()
            .map(|mut r| {
                for op in &mut r.ops {
                    if let Some(name) = op.name_index_mut() {
                        *name = name_remap[*name as usize];
                    }
                }
                r
            })
            .collect();

        let sinks: Vec<(Stage, SinkCall)> = stage_results
            .iter()
            .flat_map(|r| r.sinks.iter().map(move |s| (r.stage, s.clone())))
            .collect();

        let channel_decls: Vec<ChannelDecl> = channels
            .iter()
            .map(|c| {
                let st = c.borrow();
                let has_prog_put = !st.prog_puts.is_empty();
                let has_prog_consume = !st.prog_takes.is_empty() || !st.prog_reads.is_empty();
                let has_desc_use = !st.desc_takes.is_empty() || !st.desc_reads.is_empty();
                let has_host_put = !st.host_puts.is_empty();
                let host_consumes = !st.host_takes.is_empty() || !st.host_reads.is_empty();
                let is_terminal_output = has_prog_put
                    && !has_prog_consume
                    && !has_desc_use
                    && !has_host_put
                    && !st.seeded
                    && st.seed.is_none();
                let seeded_latest_value_writer =
                    st.seeded && (has_desc_use || !st.prog_reads.is_empty()) && !has_prog_put;
                let host_role = if (has_host_put || seeded_latest_value_writer) && !has_prog_put {
                    HostRole::Writer
                } else if (host_consumes && (!st.prog_takes.is_empty() || has_prog_put))
                    || is_terminal_output
                {
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
            names,
            channels: channel_decls,
            ports,
            stages,
        };

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

    fn record(&self, rows: u32) -> (Vec<context::StageResult>, Vec<PortBinding>) {
        let mut ports: Vec<PortBinding> = Vec::new();
        for (port, source) in &self.ports {
            let src = PortSource::Channel(context::intern_channel(source.0.state()));
            ports.push(PortBinding {
                port: *port,
                source: src,
            });
        }

        let mut results = Vec::new();
        for &stage in Stage::ALL {
            let Some((_, body)) = self.stages.iter().find(|(s, _)| *s == stage) else {
                continue;
            };
            let res = context::trace_stage(stage, rows, body);
            results.push(res);
        }
        (results, ports)
    }
}

#[derive(Debug)]
pub struct Traced {
    container: TraceContainer,
    channel_order: Vec<u64>,
    channel_names: Vec<String>,
}

impl Traced {
    pub fn container(&self) -> &TraceContainer {
        &self.container
    }
    pub fn identity_hash(&self) -> u64 {
        self.container.hash()
    }
    pub fn encode(&self) -> Vec<u8> {
        self.container.encode()
    }
    pub fn channel_order(&self) -> &[u64] {
        &self.channel_order
    }
    pub fn channel_names(&self) -> &[String] {
        &self.channel_names
    }
}

fn saturating_rows(numel: u64) -> u32 {
    u32::try_from(numel).unwrap_or(u32::MAX)
}
