//! The device run a fire stages a host-resident routed bank's experts in.
//!
//! A routed bank whose planes lie on the host tier with no table of the
//! weight cache to read them through (its experts fit no expert shape) is
//! read by its select from a copy on the device: a fire reads its routes
//! back, copies each expert they name into this run, and renumbers the
//! routes to the experts' places in it. The run is bounded at load, laid
//! once in the activation's span beside the staged inputs, and entered in
//! the load's ledger: a fire allocates nothing and reads no expert across
//! the bus.
//!
//! A join's branches may run on side streams at once (`poem_compiler::
//! branch::split`, a region's `stream`), so two staged selects of one fire
//! can overlap: each stream whose regions run one stages into a lane of
//! its own.

use poem_ir::residency::{Residency, residency};
use poem_ir::{Def, Operands, Trace};

use crate::error::{Fault, Result};

/// What a run of the staging is aligned to.
const ALIGN: u64 = 256;

/// One select of a host-resident bank, as its staging is sized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Select {
    /// The bank's codes plane, by name.
    pub bank: String,
    /// The experts the bank holds.
    pub experts: u32,
    /// The routes a row picks.
    pub top_k: u32,
    /// The bytes an expert takes of each of its planes: codes, scales and
    /// the third (zeros, an NVFP4 bank's tensor scales), 0 where it has none.
    pub planes: [u64; 3],
    /// The stream its region runs on (0 the fire's main stream).
    pub stream: u32,
}

/// The staging a load lays for its host-resident banks: each plane's run
/// holds the most experts one fire's select can touch, `min(experts, rows
/// × top_k)` of its widest bank, and the routes' run one fire's renumbered
/// routes. The selects of one stream run one after another, so a lane of
/// these runs serves every layer on it; a lane a stream that runs one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Staging {
    /// The bytes of each run of a lane: codes, scales, third, routes.
    runs: [u64; 4],
    /// The streams that run a staged select, a lane each.
    streams: Vec<u32>,
    /// The rows a fire stages for at most.
    rows: u32,
    /// The banks it stages, by name.
    banks: Vec<String>,
}

/// The staging as a fire reads it: where each run lies and what it holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Staged {
    /// The first lane's runs, address and bytes: codes, scales, third,
    /// routes.
    runs: [(u64, u64); 4],
    /// From one lane to the next.
    stride: u64,
    /// Each lane's stream, in lane order.
    streams: std::sync::Arc<[u32]>,
    /// The rows a fire stages for at most.
    pub rows: u32,
}

impl Staged {
    /// The runs of `stream`'s lane; `None` for a stream the load laid none
    /// for.
    #[must_use]
    pub fn lane(&self, stream: u32) -> Option<[(u64, u64); 4]> {
        let lane = self.streams.iter().position(|&at| at == stream)? as u64;
        Some(
            self.runs
                .map(|(at, bytes)| (at + lane * self.stride, bytes)),
        )
    }
}

impl Staging {
    /// The staging `selects` need at fires of up to `rows` rows.
    #[must_use]
    pub fn bound(selects: &[Select], rows: u32) -> Staging {
        let mut runs = [0u64; 4];
        let mut banks: Vec<String> = Vec::new();
        let mut streams: Vec<u32> = Vec::new();
        for select in selects {
            if !streams.contains(&select.stream) {
                streams.push(select.stream);
            }
            let routes = u64::from(rows) * u64::from(select.top_k);
            let slots = u64::from(select.experts).min(routes);
            for (run, bytes) in runs.iter_mut().zip(select.planes) {
                *run = (*run).max(slots * bytes);
            }
            runs[3] = runs[3].max(routes * 4);
            if !banks.contains(&select.bank) {
                banks.push(select.bank.clone());
            }
        }
        if banks.is_empty() {
            return Staging::default();
        }
        streams.sort_unstable();
        Staging {
            runs,
            streams,
            rows,
            banks,
        }
    }

    /// The routed banks of `trace` that lie on the host (`host` of their
    /// codes' address) and that the weight cache tables none of (`tabled`,
    /// by param), each as `table` binds it, once for each stream
    /// (`stream_of` a node) whose nodes read it, whatever op reads it.
    pub fn selects(
        trace: &Trace,
        table: &crate::run::WeightTable,
        tabled: &[u32],
        host: impl Fn(u64) -> bool,
        stream_of: impl Fn(usize) -> u32,
    ) -> Result<Vec<Select>> {
        let classes = residency(trace);
        let mut streams: Vec<Vec<u32>> = vec![Vec::new(); trace.params.len()];
        let mut inputs = Vec::new();
        for (at, node) in trace.nodes.iter().enumerate() {
            inputs.clear();
            node.op.inputs(&mut inputs);
            for value in &inputs {
                if let Some(Def::Weight(w)) = trace.values.get(value.0 as usize).map(|v| &v.def)
                    && let Some(read) = streams.get_mut(*w as usize)
                    && !read.contains(&stream_of(at))
                {
                    read.push(stream_of(at));
                }
            }
        }
        let mut out = Vec::new();
        for (w, param) in trace.params.iter().enumerate() {
            let Residency::Expert(atom) = classes[w] else {
                continue;
            };
            let Some(Some(crate::run::WeightRow::Planes {
                codes,
                scales,
                biases,
                ..
            })) = table.0.get(w)
            else {
                continue;
            };
            if tabled.contains(&(w as u32)) || !host(codes.ptr) {
                continue;
            }
            if atom.top_k == 0 {
                return Err(Fault::Residency(format!(
                    "`{}` lies on the host and a fire stages the experts its routes name, \
                     which the load bounds by the routes a row picks; its select's routes \
                     state no constant width",
                    param.name
                )));
            }
            for &stream in &streams[w] {
                out.push(Select {
                    bank: param.name.clone(),
                    experts: codes.rows,
                    top_k: atom.top_k,
                    planes: [
                        u64::from(codes.width),
                        u64::from(scales.width),
                        biases.map_or(0, |third| u64::from(third.width)),
                    ],
                    stream,
                });
            }
        }
        Ok(out)
    }

    /// The bytes the staging takes of the activation's span.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.lane_bytes() * self.streams.len() as u64
    }

    fn lane_bytes(&self) -> u64 {
        self.runs
            .iter()
            .map(|run| run.next_multiple_of(ALIGN))
            .sum()
    }

    #[must_use]
    pub fn banks(&self) -> &[String] {
        &self.banks
    }

    /// The bytes of each run of a lane: codes, scales, third, routes.
    #[must_use]
    pub fn runs(&self) -> [u64; 4] {
        self.runs
    }

    /// The streams that run a staged select, a lane each.
    #[must_use]
    pub fn streams(&self) -> &[u32] {
        &self.streams
    }

    /// The rows a fire stages for at most.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// The fit's refusal of a load that lays this staging, its banks named:
    /// the staging is a run of the activation's span, a term of the fit as
    /// the workspaces are, so a card the fit cannot seat it on refuses the
    /// load there.
    #[must_use]
    pub fn short(&self, why: Fault) -> Fault {
        if self.banks.is_empty() {
            return why;
        }
        let why = match why {
            Fault::Residency(said) => said,
            other => other.to_string(),
        };
        Fault::Residency(format!(
            "{}. Of what this load lays, the routed bank(s) {} lie on the host with no \
             table of the weight cache to read them through, and a fire of {} rows stages the \
             experts it routes to on the device, a lane for each of {} stream(s): {} bytes of \
             staged experts. Serve fewer `max_forward_tokens`, or let the weight cache take \
             the bank(s).",
            why.trim_end_matches('.'),
            self.names(),
            self.rows,
            self.streams.len(),
            self.bytes(),
        ))
    }

    /// The staging laid from `base`; `None` where it stages nothing.
    #[must_use]
    pub fn at(&self, base: u64) -> Option<Staged> {
        if self.banks.is_empty() {
            return None;
        }
        let mut at = base;
        let runs = self.runs.map(|bytes| {
            let run = (at, bytes);
            at += bytes.next_multiple_of(ALIGN);
            run
        });
        Some(Staged {
            runs,
            stride: self.lane_bytes(),
            streams: self.streams.as_slice().into(),
            rows: self.rows,
        })
    }

    /// The load's fact: what is staged and for how many rows.
    #[must_use]
    pub fn fact(&self) -> Option<String> {
        (!self.banks.is_empty()).then(|| {
            format!(
                "the routed bank(s) {} lie on the host with no table of the \
                 weight cache: a fire reads its routes back and stages the experts they name \
                 into {} MiB laid at load (each plane the most experts {} rows route to, a \
                 lane for each of {} stream(s) that runs a select), so its fires walk \
                 eagerly, unrecorded",
                self.names(),
                self.bytes() >> 20,
                self.rows,
                self.streams.len(),
            )
        })
    }

    fn names(&self) -> String {
        self.banks
            .iter()
            .map(|bank| format!("`{bank}`"))
            .collect::<Vec<String>>()
            .join(", ")
    }
}
