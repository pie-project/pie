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
    /// A size the program declares does not fit the arithmetic it is
    /// priced in: `what` names it.
    Overflow(&'static str),
}

impl fmt::Display for RegisterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegisterError::Decode(e) => write!(f, "container decode failed: {e}"),
            RegisterError::Bind(e) => write!(f, "bind failed: {e}"),
            RegisterError::HashCollision(hash) => {
                write!(f, "program hash collision for 0x{hash:016x}")
            }
            RegisterError::Overflow(what) => {
                write!(f, "{what} overflows the arithmetic it is priced in")
            }
        }
    }
}
impl std::error::Error for RegisterError {}

pub const DEFAULT_CAPACITY: usize = 256;

pub struct Registry {
    inner: LruCache<u64, Arc<RegisteredProgram>>,
}

impl Registry {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: LruCache::new(capacity),
        }
    }

    /// A program bound for one pass is checked again against the next's `lanes`.
    pub fn register(
        &mut self,
        bytes: Vec<u8>,
        lanes: Vec<Vec<OutputDecl>>,
        caps: EngineCaps,
    ) -> Result<Arc<RegisteredProgram>, RegisterError> {
        let mut hash = container_hash(&bytes);
        if std::env::var_os("PIE_ETA_NO_IDIOM").is_some() {
            hash ^= 0x9e37_79b9_7f4a_7c15;
        }
        if std::env::var_os("PIE_ETA_NO_FUSION").is_some() {
            hash ^= 0xbf58_476d_1ce4_e5b9;
        }
        if let Some(hit) = self.inner.get(&hash) {
            if hit.bytes != bytes {
                return Err(RegisterError::HashCollision(hash));
            }
            check_outputs(&hit.bound.container, &lanes).map_err(RegisterError::Bind)?;
            return Ok(hit.clone());
        }
        let decoded = container::decode(&bytes).map_err(RegisterError::Decode)?;
        let pricing = price(&decoded)?;
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
            reads,
            launch,
            geometry_taint: std::sync::OnceLock::new(),
            shadow_plan: std::sync::OnceLock::new(),
            pricing,
            emitted: Mutex::new(HashMap::new()),
        });
        self.inner.put(hash, entry.clone());
        Ok(entry)
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

/// What `c` asks of the host and the device as it declares itself; a
/// declared size past the arithmetic it is priced in is refused by name.
fn price(c: &TraceContainer) -> Result<Pricing, RegisterError> {
    let mut channel_bytes = 0u64;
    for ch in &c.channels {
        let elem = container::const_elem_size(ch.dtype.program_dtype()) as u64;
        let cells = u64::from(ch.capacity)
            .checked_add(1)
            .ok_or(RegisterError::Overflow("a channel's cells"))?;
        let bytes = ch
            .shape
            .numel()
            .checked_mul(elem)
            .and_then(|bytes| bytes.checked_mul(cells))
            .ok_or(RegisterError::Overflow("a channel's bytes"))?;
        channel_bytes = channel_bytes
            .checked_add(bytes)
            .ok_or(RegisterError::Overflow("the program's channel bytes"))?;
    }
    let numel = |shape: eta_ir::types::Shape| {
        u32::try_from(shape.numel()).map_err(|_| RegisterError::Overflow("a port's length"))
    };
    let port_len = |port| -> Result<Option<u32>, RegisterError> {
        let Some(binding) = c.ports.iter().find(|p| p.port == port) else {
            return Ok(None);
        };
        match &binding.source {
            container::PortSource::Const { shape, .. } => numel(*shape).map(Some),
            container::PortSource::Channel(channel) => c
                .channels
                .get(*channel as usize)
                .map(|decl| numel(decl.shape))
                .transpose(),
        }
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
    let rows = match port_len(eta_ir::registry::Port::Readout)? {
        Some(readout) => readout.max(1),
        None => match port_len(eta_ir::registry::Port::EmbedIndptr)? {
            Some(indptr) => indptr.saturating_sub(1).max(1),
            None => output_rows().unwrap_or(1),
        },
    };
    Ok(Pricing {
        channel_bytes,
        num_channels: c.channels.len(),
        rows,
    })
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

pub fn register(
    bytes: Vec<u8>,
    lanes: Vec<Vec<OutputDecl>>,
) -> Result<Arc<RegisteredProgram>, RegisterError> {
    let eta = crate::model::model().eta_caps();
    let caps = EngineCaps {
        lora: eta.has_lora,
        attn_page_mask: eta.has_attn_page_mask,
    };
    global().register(bytes, lanes, caps)
}

pub fn lookup(hash: u64) -> Option<Arc<RegisteredProgram>> {
    global().lookup(hash)
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
        assert_eq!(price(&container).unwrap().rows, 256);
        assert_eq!(price(&TraceContainer::default()).unwrap().rows, 1);
    }

    #[test]
    fn a_channel_past_the_pricing_arithmetic_is_refused_by_name() {
        use eta_ir::container::{ChanDType, ChannelDecl, HostRole};
        let huge = Shape::new(&[u32::MAX, u32::MAX]).unwrap();
        let container = TraceContainer {
            channels: vec![ChannelDecl {
                shape: huge,
                dtype: ChanDType::Concrete(Dtype::F32),
                capacity: u32::MAX,
                host_role: HostRole::None,
                seeded: false,
            }],
            ..TraceContainer::default()
        };
        let refused = price(&container).unwrap_err();
        assert!(
            matches!(refused, super::RegisterError::Overflow("a channel's bytes")),
            "{refused}"
        );
    }

    #[test]
    fn a_port_longer_than_a_u32_is_refused_by_name() {
        use eta_ir::container::{ChanDType, ChannelDecl, HostRole, PortBinding, PortSource};
        use eta_ir::registry::Port;
        let container = TraceContainer {
            channels: vec![ChannelDecl {
                shape: Shape::new(&[1 << 20, 1 << 20]).unwrap(),
                dtype: ChanDType::Concrete(Dtype::Bool),
                capacity: 1,
                host_role: HostRole::None,
                seeded: false,
            }],
            ports: vec![PortBinding {
                port: Port::Readout,
                source: PortSource::Channel(0),
            }],
            ..TraceContainer::default()
        };
        let refused = price(&container).unwrap_err();
        assert!(
            matches!(refused, super::RegisterError::Overflow("a port's length")),
            "{refused}"
        );
    }
}
