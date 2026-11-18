use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use eta_compiler::codegen::cuda::region_analysis::RegionAnalysis;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_compiler::codegen::program::{Backend, EmittedKernel, emit_program};
use eta_compiler::plan::CompiledStage;
use eta_ir::container::{self, ContainerDecodeError, PortSource, TraceContainer};
use eta_ir::container_hash;
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, OutputDecl, Port};
use eta_ir::validate::{BoundTrace, ValidateError, bind, check_outputs};
use lru::LruCache;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pricing {
    pub channel_bytes: u64,
    pub num_channels: usize,
    pub rows: u32,
}

#[derive(Debug)]
pub struct RegisteredProgram {
    pub bytes: Vec<u8>,
    pub hash: u64,
    pub bound: BoundTrace,
    pub compiled_stages: Vec<CompiledStage>,
    emitted: Mutex<HashMap<Backend, Arc<EmittedProgram>>>,
    pub channel_accesses: Vec<(bool, bool)>,
    /// The channels declared at the row bound rather than a request's
    /// length (`TraceContainer::row_extent_channels`): an instance's channel
    /// carries up to `row_bound` rows there, padded with [`ROW_PAD`].
    pub row_extents: Vec<bool>,
    pub row_bound: u32,
    /// The outputs it reads, by name and the lane reading each.
    pub reads: Vec<(String, u32)>,
    launch: std::sync::OnceLock<LaunchPackage>,
    geometry_taint: std::sync::OnceLock<eta_compiler::eval::pareval::GeometryTaint>,
    shadow_plan: std::sync::OnceLock<Arc<crate::pipeline::fire::shadow::ShadowPlan>>,
    pub pricing: Pricing,
}

#[derive(Debug)]
pub struct EmittedProgram {
    pub emitter_version: u32,
    pub kernels: Vec<EmittedKernel>,
}

impl RegisteredProgram {
    pub fn launch(&self) -> &LaunchPackage {
        self.launch.get_or_init(|| {
            eta_compiler::codegen::launch::build(&self.bound, &self.compiled_stages)
        })
    }

    pub fn geometry_taint(&self) -> &eta_compiler::eval::pareval::GeometryTaint {
        self.geometry_taint
            .get_or_init(|| eta_compiler::eval::pareval::geometry_taint(&self.bound))
    }

    pub fn shadow_plan(&self) -> Arc<crate::pipeline::fire::shadow::ShadowPlan> {
        Arc::clone(self.shadow_plan.get_or_init(|| {
            Arc::new(crate::pipeline::fire::shadow::ShadowPlan::derive(
                &self.bound,
            ))
        }))
    }

    pub fn region_analysis(&self) -> Vec<RegionAnalysis> {
        eta_compiler::codegen::cuda::region_analysis::analyze_program(&self.compiled_stages)
    }

    pub fn emitted(&self, backend: &str) -> Option<Arc<EmittedProgram>> {
        let backend = Backend::parse(backend)?;
        let mut cache = self.emitted.lock().unwrap();
        Some(Arc::clone(cache.entry(backend).or_insert_with(|| {
            Arc::new(EmittedProgram {
                emitter_version: backend.emitter_version(),
                kernels: emit_program(backend, &self.compiled_stages, &self.bound),
            })
        })))
    }
}

#[derive(Debug)]
pub enum RegisterError {
    Decode(ContainerDecodeError),
    Bind(ValidateError),
    HashCollision(u64),
}

impl fmt::Display for RegisterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegisterError::Decode(e) => write!(f, "container decode failed: {e}"),
            RegisterError::Bind(e) => write!(f, "bind failed: {e}"),
            RegisterError::HashCollision(hash) => {
                write!(f, "program hash collision for 0x{hash:016x}")
            }
        }
    }
}
impl std::error::Error for RegisterError {}

pub const DEFAULT_CAPACITY: usize = 256;

/// What a row-extent channel carries past an instance's own rows: every
/// byte set, `-1` as an `i32` token (the SDK's `TOKEN_PAD`) and `u32::MAX`
/// as a position or slot. No fire reads it: the indptr spans the live rows.
pub const ROW_PAD: u32 = u32::MAX;

/// An instance's fill of a row-extent channel, `len` rows of `elem` bytes
/// in `bytes`, padded with [`ROW_PAD`] to the `rows` the program declares.
///
/// # Errors
///
/// A fill longer than the bound, by name: no fire binds that many rows.
pub fn pad_row_extent(mut bytes: Vec<u8>, elem: usize, rows: u32) -> Result<Vec<u8>, String> {
    let len = bytes.len() / elem.max(1);
    if len > rows as usize {
        return Err(format!(
            "a per-row channel of {len} rows exceeds the {rows} rows one fire binds \
             (`model.max-pass-rows()`)"
        ));
    }
    bytes.resize(rows as usize * elem, 0xff);
    Ok(bytes)
}

pub struct Registry {
    inner: LruCache<u64, Arc<RegisteredProgram>>,
    /// Registrations found by their hash, and those registered anew: a miss
    /// is a program identity of its own, so requests of one kind add none
    /// after the first.
    hits: u64,
    misses: u64,
}

impl Registry {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: LruCache::new(capacity),
            hits: 0,
            misses: 0,
        }
    }

    /// A program bound for one pass is checked again against the next's `lanes`.
    ///
    /// `rows` is the most rows one fire binds (the engine's max forward
    /// tokens, the SDK's `model.max-pass-rows()`): a host-fed per-row
    /// channel is declared at it, so programs differing only in a request's
    /// length register once.
    pub fn register(
        &mut self,
        bytes: Vec<u8>,
        lanes: Vec<Vec<OutputDecl>>,
        caps: EngineCaps,
        rows: u32,
    ) -> Result<Arc<RegisteredProgram>, RegisterError> {
        let salt = |mut hash: u64| {
            if std::env::var_os("PIE_ETA_NO_IDIOM").is_some() {
                hash ^= 0x9e37_79b9_7f4a_7c15;
            }
            if std::env::var_os("PIE_ETA_NO_FUSION").is_some() {
                hash ^= 0xbf58_476d_1ce4_e5b9;
            }
            hash
        };
        // A container already in its registered form is found by its bytes.
        let raw = salt(container_hash(&bytes));
        if let Some(hit) = self.inner.get(&raw)
            && hit.bytes == bytes
            && hit.row_bound == rows
        {
            check_outputs(&hit.bound.container, &lanes).map_err(RegisterError::Bind)?;
            self.hits += 1;
            return Ok(hit.clone());
        }
        let mut decoded = container::decode(&bytes).map_err(RegisterError::Decode)?;
        let row_extents = decoded.bind_row_extents(rows);
        let bytes = if row_extents.contains(&true) {
            decoded.encode()
        } else {
            bytes
        };
        let hash = salt(container_hash(&bytes));
        if let Some(hit) = self.inner.get(&hash) {
            if hit.bytes != bytes {
                return Err(RegisterError::HashCollision(hash));
            }
            check_outputs(&hit.bound.container, &lanes).map_err(RegisterError::Bind)?;
            self.hits += 1;
            return Ok(hit.clone());
        }
        tracing::debug!(
            hash = format_args!("0x{hash:016x}"),
            channels = ?decoded
                .channels
                .iter()
                .map(|channel| channel.shape.dims().to_vec())
                .collect::<Vec<_>>(),
            ops = decoded.stages.iter().map(|stage| stage.ops.len()).sum::<usize>(),
            "program registered anew"
        );
        let pricing = price(&decoded);
        let channel_accesses = Self::channel_accesses(&decoded);
        let reads = Self::reads(&decoded);
        let bound = bind(decoded, lanes, caps).map_err(RegisterError::Bind)?;
        let compiled_stages = eta_compiler::plan::compile_bound(&bound);
        let launch = std::sync::OnceLock::new();
        let entry = Arc::new(RegisteredProgram {
            bytes,
            hash,
            bound,
            compiled_stages,
            channel_accesses,
            row_extents,
            row_bound: rows,
            reads,
            launch,
            geometry_taint: std::sync::OnceLock::new(),
            shadow_plan: std::sync::OnceLock::new(),
            pricing,
            emitted: Mutex::new(HashMap::new()),
        });
        self.inner.put(hash, entry.clone());
        self.misses += 1;
        Ok(entry)
    }

    /// Registrations found by their hash and registered anew, so far.
    #[must_use]
    pub fn hits_and_misses(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    fn reads(container: &TraceContainer) -> Vec<(String, u32)> {
        let mut reads: Vec<(String, u32)> = container
            .stages
            .iter()
            .flat_map(|stage| stage.ops.iter())
            .filter_map(|op| match *op {
                Op::Output { index, lane, .. } => {
                    Some((container.names.get(index as usize)?.clone(), lane))
                }
                _ => None,
            })
            .collect();
        reads.sort_unstable();
        reads.dedup();
        reads
    }

    fn channel_accesses(container: &TraceContainer) -> Vec<(bool, bool)> {
        let mut accesses = vec![(false, false); container.channels.len()];
        for stage in &container.stages {
            for op in &stage.ops {
                match *op {
                    Op::ChanTake(channel) => accesses[channel as usize].0 = true,
                    Op::ChanPut { chan, .. } => accesses[chan as usize].1 = true,
                    _ => {}
                }
            }
        }
        for binding in &container.ports {
            let PortSource::Channel(channel) = &binding.source else {
                continue;
            };
            if matches!(
                binding.port,
                Port::EmbedTokens | Port::Positions | Port::Slots
            ) {
                accesses[*channel as usize].0 = true;
            }
        }
        accesses
    }

    pub fn lookup(&mut self, hash: u64) -> Option<Arc<RegisteredProgram>> {
        self.inner.get(&hash).cloned()
    }
}

fn price(c: &TraceContainer) -> Pricing {
    let channel_bytes = c
        .channels
        .iter()
        .map(|ch| {
            let elem = container::const_elem_size(ch.dtype.program_dtype()) as u64;
            let cells = (ch.capacity as u64) + 1;
            ch.shape.numel() * elem * cells
        })
        .sum();
    let port_len = |port| {
        c.ports
            .iter()
            .find(|p| p.port == port)
            .and_then(|p| match &p.source {
                container::PortSource::Const { shape, .. } => Some(shape.numel() as u32),
                container::PortSource::Channel(channel) => c
                    .channels
                    .get(*channel as usize)
                    .map(|decl| decl.shape.numel() as u32),
            })
    };
    let output_rows = || {
        c.stages
            .iter()
            .flat_map(|stage| stage.ops.iter())
            .filter_map(|op| match op {
                Op::Output { shape, .. } => shape.dims().first().copied(),
                _ => None,
            })
            .max()
    };
    let rows = port_len(eta_ir::registry::Port::Readout)
        .map(|readout| readout.max(1))
        .or_else(|| {
            port_len(eta_ir::registry::Port::EmbedIndptr)
                .map(|indptr| indptr.saturating_sub(1).max(1))
        })
        .or_else(output_rows)
        .unwrap_or(1);
    Pricing {
        channel_bytes,
        num_channels: c.channels.len(),
        rows,
    }
}

use std::sync::{LazyLock, MutexGuard};

static GLOBAL: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry::new(
        NonZeroUsize::new(DEFAULT_CAPACITY).expect("nonzero capacity"),
    ))
});

fn global() -> MutexGuard<'static, Registry> {
    GLOBAL.lock().expect("global lock poisoned")
}

/// Registers `bytes` for a model whose fires bind at most `rows` rows
/// ([`Registry::register`]).
pub fn register(
    bytes: Vec<u8>,
    lanes: Vec<Vec<OutputDecl>>,
    rows: u32,
) -> Result<Arc<RegisteredProgram>, RegisterError> {
    let eta = crate::model::model().eta_caps();
    let caps = EngineCaps {
        lora: eta.has_lora,
        attn_page_mask: eta.has_attn_page_mask,
    };
    global().register(bytes, lanes, caps, rows)
}

pub fn lookup(hash: u64) -> Option<Arc<RegisteredProgram>> {
    global().lookup(hash)
}

/// The served model's registrations found by their hash and registered
/// anew: the model status's `programs.hits` and `programs.misses`.
#[must_use]
pub fn hits_and_misses() -> (u64, u64) {
    global().hits_and_misses()
}

#[must_use]
pub fn with_host_codegen<'a>(
    plan: &'a ::engine::ProgramRegistration,
    engine_backend: Option<&str>,
) -> std::borrow::Cow<'a, ::engine::ProgramRegistration> {
    let Some(backend) = engine_backend.filter(|name| Backend::parse(name).is_some()) else {
        return std::borrow::Cow::Borrowed(plan);
    };
    let registered = lookup(plan.program_hash);

    let emitted = plan
        .emitted_kernels
        .is_empty()
        .then(|| registered.as_ref()?.emitted(backend))
        .flatten();

    let region_analysis = if plan.region_analysis.is_empty() && backend == "cuda" {
        registered
            .as_ref()
            .map(|program| program.region_analysis())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    if emitted.is_none() && region_analysis.is_empty() {
        return std::borrow::Cow::Borrowed(plan);
    }

    let mut next = plan.clone();
    if let Some(emitted) = emitted {
        next.emitter_version = emitted.emitter_version;
        next.emitted_kernels = emitted.kernels.clone();
    }
    if !region_analysis.is_empty() {
        next.region_analysis = region_analysis;
    }
    std::borrow::Cow::Owned(next)
}

#[cfg(test)]
mod pricing_tests {
    use super::price;
    use eta_ir::container::{StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::Stage;
    use eta_ir::types::{Dtype, Shape};

    #[test]
    fn a_float_lane_is_priced_by_its_output_rows() {
        let container = TraceContainer {
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(256, 64),
                    dtype: Dtype::F32,
                }],
            }],
            ..TraceContainer::default()
        };
        assert_eq!(price(&container).rows, 256);
        assert_eq!(price(&TraceContainer::default()).rows, 1);
    }
}

#[cfg(test)]
mod row_extent_tests {
    use super::{ROW_PAD, Registry, pad_row_extent};
    use eta_ir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Port, Stage};
    use eta_ir::types::{Dtype, Shape};
    use std::num::NonZeroUsize;

    const ROWS: u32 = 64;

    fn seeded(n: u32, dtype: Dtype) -> ChannelDecl {
        ChannelDecl {
            shape: Shape::vector(n),
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: HostRole::None,
            seeded: true,
        }
    }

    /// A prefill of `len` prompt tokens as an inferlet channels it
    /// (`Channel::from(&prompt)` and `Channel::from([0, len])`).
    fn prefill(len: u32) -> TraceContainer {
        TraceContainer {
            channels: vec![seeded(len, Dtype::I32), seeded(2, Dtype::U32)],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Channel(1),
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![],
            }],
            ..TraceContainer::default()
        }
    }

    fn logits() -> Vec<Vec<OutputDecl>> {
        vec![vec![OutputDecl {
            name: "logits".into(),
            axis: Axis::Row,
            width: 16,
            dtype: Dtype::F32,
            layers: 0,
            readout: true,
        }]]
    }

    /// Prompts of every length up to the bound are one program: the
    /// tokens' channel is declared at the row bound, the first registers it
    /// and the rest find it.
    #[test]
    fn prompts_of_every_length_register_one_program() {
        let mut registry = Registry::new(NonZeroUsize::new(8).unwrap());
        let programs: Vec<_> = [1u32, 7, 33, ROWS]
            .into_iter()
            .map(|len| {
                registry
                    .register(prefill(len).encode(), logits(), EngineCaps::default(), ROWS)
                    .expect("registers")
            })
            .collect();
        assert!(programs.iter().all(|program| program.hash == programs[0].hash));
        assert_eq!(registry.hits_and_misses(), (3, 1));
        let program = &programs[0];
        assert_eq!(program.row_extents, vec![true, false]);
        assert_eq!(
            program.bound.container.channels[0].shape,
            Shape::vector(ROWS),
            "the tokens are declared at the bound"
        );
        assert_eq!(program.bound.container.channels[1].shape, Shape::vector(2));
    }

    /// A channel an op reads keeps its length in the program: its shape is
    /// what the stage computes on.
    #[test]
    fn a_channel_a_stage_reads_keeps_its_length() {
        let read = |len: u32| {
            let mut container = prefill(len);
            container.stages[0].ops.push(eta_ir::op::Op::ChanRead(0));
            container
        };
        assert_eq!(read(3).row_extent_channels(), vec![false, false]);
        let mut registry = Registry::new(NonZeroUsize::new(8).unwrap());
        let mut hash = |len| {
            registry
                .register(read(len).encode(), logits(), EngineCaps::default(), ROWS)
                .expect("registers")
                .hash
        };
        assert_ne!(hash(3), hash(5));
    }

    /// A fill is padded with `ROW_PAD` to the bound; one longer than the
    /// bound is refused by name.
    #[test]
    fn a_fill_past_the_row_bound_is_refused_by_name() {
        let fill: Vec<u8> = [5i32, 6, 7].iter().flat_map(|t| t.to_le_bytes()).collect();
        let padded = pad_row_extent(fill, 4, 5).expect("fits");
        let rows: Vec<u32> = padded
            .chunks(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(rows, vec![5, 6, 7, ROW_PAD, ROW_PAD]);
        let long: Vec<u8> = vec![0; 4 * (ROWS as usize + 1)];
        let error = pad_row_extent(long, 4, ROWS).expect_err("past the bound");
        assert!(
            error.contains("65 rows exceeds the 64 rows one fire binds"),
            "{error}"
        );
    }
}
