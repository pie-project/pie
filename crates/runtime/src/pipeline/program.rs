use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use eta_compiler::codegen::cuda::region_analysis::RegionAnalysis;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_compiler::codegen::program::{Backend, EmittedKernel, emit_program};
use eta_compiler::plan::{CompiledStage, LibraryForms};

/// The library forms this process plans programs with: every form, less
/// what an engine it serves states its emitters lack
/// ([`Backend::library_forms`]), narrowed at boot before any program is
/// registered ([`serve_forms`]).
static PLAN_FORMS: Mutex<LibraryForms> = Mutex::new(LibraryForms::ALL);

/// Narrows the forms programs are planned with to what an engine states
/// its emitters serve (`engine::DeviceFacts::library_forms`, `engine` its
/// backend's name); logs once what it plans as generated regions instead.
pub fn serve_forms(engine: &str, states: LibraryForms) {
    let mut forms = PLAN_FORMS.lock().unwrap();
    let narrowed = forms.and(states);
    if narrowed != *forms {
        let mut generated = Vec::new();
        if forms.draw && !narrowed.draw {
            generated.push("the temperature-scaled gumbel draw over a top-p / top-k cut");
        }
        if forms.bf16_cut && !narrowed.bf16_cut {
            generated.push("a bf16 row's lane cut");
        }
        tracing::info!(
            engine,
            generated = %generated.join("; "),
            "the {engine} engine's emitters state no library form for these: programs are \
             planned with them as generated regions"
        );
    }
    *forms = narrowed;
}

fn plan_forms() -> LibraryForms {
    *PLAN_FORMS.lock().unwrap()
}
use eta_ir::container::{self, ContainerDecodeError, PortSource, TraceContainer};
use eta_ir::container_hash;
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, OutputDecl, Port};
use eta_ir::validate::{BoundTrace, Budget, ValidateError, bind_within, check_outputs};
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
    /// The channels declared at [`RegisteredProgram::row_bound`] rather than
    /// a request's length (`TraceContainer::row_extent_channels`): an
    /// instance's channel there carries its own rows, which the host
    /// resolves the ports from; its engine cell is the bound wide
    /// ([`fit_row_extent`]), and no fire reads it.
    pub row_extents: Vec<bool>,
    /// The most rows one fire binds, which a row-extent channel is declared
    /// at: the served model's `model.max-pass-rows()`.
    pub row_bound: u32,
    /// The outputs it reads, by name and the lane reading each.
    pub reads: Vec<(String, u32)>,
    launch: std::sync::OnceLock<LaunchPackage>,
    geometry_taint: std::sync::OnceLock<eta_compiler::eval::pareval::GeometryTaint>,
    shadow_plan: std::sync::OnceLock<Arc<crate::pipeline::fire::shadow::ShadowPlan>>,
    pub pricing: Pricing,
    /// Whether it states each beam's parent (`Port::Parents`): its lanes
    /// hold their own page lists of their set's reserve. Decided once, at
    /// registration, never walked per fire.
    pub binds_pages: bool,
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
    /// The program's channel cells ([`Pricing::channel_bytes`]) take more
    /// than an instance's state room holds, where they are laid.
    ChannelsPastRoom {
        bytes: u64,
        room: u64,
    },
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
            RegisterError::ChannelsPastRoom { bytes, room } => write!(
                f,
                "the program's channel cells take {bytes} bytes and an instance's state \
                 room on the device serving this model holds {room}; it can bind nowhere"
            ),
        }
    }
}
impl std::error::Error for RegisterError {}

pub const DEFAULT_CAPACITY: usize = 256;

/// What a row-extent channel's engine cell holds past an instance's own
/// rows: every byte set (`-1` as an `i32` token, `u32::MAX` as a position or
/// slot).
pub const ROW_PAD: u8 = 0xff;

/// An instance's row-extent channel as its engine cell holds it: `bytes`
/// of `elem`-byte rows, cut or padded with [`ROW_PAD`] to the `rows` the
/// program declares it at. The cell is the program's, never read by a fire:
/// its program's geometry is the host's, resolved from the instance's own
/// rows (a bind of it in another class is refused by name), so a prompt
/// longer than the bound loses nothing.
#[must_use]
pub fn fit_row_extent(mut bytes: Vec<u8>, elem: usize, rows: u32) -> Vec<u8> {
    bytes.resize(rows as usize * elem, ROW_PAD);
    bytes
}

/// A program's identity: `bytes` with each row-extent channel declared at
/// `rows` ([`TraceContainer::bind_row_extents`]), so programs that differ
/// only in a request's length are one; and which channels those are. `None`
/// keeps every extent (a float pass, which counts its rows from its
/// channels' lengths).
fn identity(bytes: Vec<u8>, rows: Option<u32>) -> Result<(Vec<u8>, Vec<bool>), RegisterError> {
    let Some(rows) = rows else {
        return Ok((bytes, Vec::new()));
    };
    let mut decoded = container::decode(&bytes).map_err(RegisterError::Decode)?;
    let row_extents = decoded.bind_row_extents(rows);
    if row_extents.contains(&true) {
        Ok((decoded.encode(), row_extents))
    } else {
        Ok((bytes, row_extents))
    }
}

/// Who registered a program: a guest process, or the runtime itself
/// (`None`: the standard samplers).
pub type Tenant = Option<uuid::Uuid>;

pub struct Registry {
    inner: LruCache<u64, Arc<RegisteredProgram>>,
    /// The tenant that brought each held program in: what bounds how much
    /// of the registry one tenant's registrations may churn
    /// ([`Registry::make_room`]).
    owners: HashMap<u64, Tenant>,
    /// Programs registered anew, and registrations that found one held:
    /// the model status's `programs.registered` / `programs.found`.
    registered: u64,
    found: u64,
}

impl Registry {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: LruCache::new(capacity),
            owners: HashMap::new(),
            registered: 0,
            found: 0,
        }
    }

    /// Room for one more program `tenant` brings in, in a full registry.
    /// Each tenant holding programs (and `tenant`) has an equal share of
    /// the registry's size: a tenant at or past its share gives up its own
    /// least recently used program; one under it takes the least recently
    /// used of a tenant past its share (one is, the registry being full).
    /// A tenant registering program after program so churns only its own
    /// share, never another tenant's programs within theirs.
    fn make_room(&mut self, tenant: Tenant) {
        let capacity = self.inner.cap().get();
        if self.inner.len() < capacity {
            return;
        }
        let mut held: HashMap<Tenant, usize> = HashMap::new();
        for owner in self.owners.values() {
            *held.entry(*owner).or_default() += 1;
        }
        let tenants = held.len() + usize::from(!held.contains_key(&tenant));
        let share = (capacity / tenants.max(1)).max(1);
        let own = held.get(&tenant).copied().unwrap_or(0);
        let victim = self
            .inner
            .iter()
            .rev()
            .map(|(hash, _)| *hash)
            .find(|hash| {
                let owner = self.owners.get(hash).copied().flatten();
                if own >= share {
                    owner == tenant
                } else {
                    held.get(&owner).copied().unwrap_or(0) > share
                }
            })
            .or_else(|| self.inner.peek_lru().map(|(hash, _)| *hash));
        if let Some(victim) = victim {
            self.inner.pop(&victim);
            self.owners.remove(&victim);
        }
    }

    /// A program bound for one pass is checked again against the next's
    /// `lanes`. It binds within `budget` (`validate::bind_within`), its
    /// channel cells too: past it, refused by name before anything of it is
    /// evaluated or laid. `tenant` brought it in ([`Registry::make_room`]).
    #[cfg(test)]
    pub fn register(
        &mut self,
        bytes: Vec<u8>,
        lanes: Vec<Vec<OutputDecl>>,
        caps: EngineCaps,
        budget: Budget,
        tenant: Tenant,
        rows: Option<u32>,
    ) -> Result<Arc<RegisteredProgram>, RegisterError> {
        let (bytes, row_extents) = identity(bytes, rows)?;
        let hash = container_hash(&bytes);
        if let Some(hit) = self.hit(hash, &bytes, &lanes)? {
            return Ok(hit);
        }
        let entry = Arc::new(RegisteredProgram::build(
            bytes,
            hash,
            lanes,
            caps,
            budget,
            (row_extents, rows.unwrap_or(0)),
        )?);
        Ok(self.insert(entry, tenant))
    }

    /// Programs registered anew, and registrations that found one held.
    #[must_use]
    pub fn counts(&self) -> (u64, u64) {
        (self.registered, self.found)
    }

    fn hit(
        &mut self,
        hash: u64,
        bytes: &[u8],
        lanes: &[Vec<OutputDecl>],
    ) -> Result<Option<Arc<RegisteredProgram>>, RegisterError> {
        let Some(hit) = self.inner.get(&hash) else {
            return Ok(None);
        };
        if hit.bytes != bytes {
            return Err(RegisterError::HashCollision(hash));
        }
        check_outputs(&hit.bound.container, lanes).map_err(RegisterError::Bind)?;
        let hit = hit.clone();
        self.found += 1;
        Ok(Some(hit))
    }

    /// A program another caller registered meanwhile wins, so every caller
    /// shares one entry; a new one takes its room as `tenant`'s
    /// ([`Registry::make_room`]).
    fn insert(&mut self, entry: Arc<RegisteredProgram>, tenant: Tenant) -> Arc<RegisteredProgram> {
        if let Some(raced) = self.inner.get(&entry.hash)
            && raced.bytes == entry.bytes
        {
            let raced = raced.clone();
            self.found += 1;
            return raced;
        }
        self.make_room(tenant);
        self.registered += 1;
        tracing::debug!(
            target: "pie::program",
            hash = format_args!("0x{:016x}", entry.hash),
            channels = ?entry
                .bound
                .container
                .channels
                .iter()
                .map(|decl| decl.shape)
                .collect::<Vec<_>>(),
            row_extents = ?entry.row_extents,
            "registered a program anew"
        );
        self.inner.put(entry.hash, entry.clone());
        self.owners.insert(entry.hash, tenant);
        entry
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

impl RegisteredProgram {
    /// Bind validates every channel index before anything else indexes by
    /// one; channel cells past `budget`'s room are refused before it binds.
    fn build(
        bytes: Vec<u8>,
        hash: u64,
        lanes: Vec<Vec<OutputDecl>>,
        caps: EngineCaps,
        budget: Budget,
        (row_extents, row_bound): (Vec<bool>, u32),
    ) -> Result<Self, RegisterError> {
        let decoded = container::decode(&bytes).map_err(RegisterError::Decode)?;
        let pricing = price(&decoded)?;
        // A program's channel cells lie in the `Channel` slots of the same
        // state room its stages do.
        if pricing.channel_bytes > budget.room_bytes {
            return Err(RegisterError::ChannelsPastRoom {
                bytes: pricing.channel_bytes,
                room: budget.room_bytes,
            });
        }
        let bound = bind_within(decoded, lanes, caps, budget).map_err(RegisterError::Bind)?;
        let channel_accesses = Registry::channel_accesses(&bound.container);
        let reads = Registry::reads(&bound.container);
        let compiled_stages = eta_compiler::plan::compile_bound_with(&bound, plan_forms());
        let binds_pages = bound
            .container
            .ports
            .iter()
            .any(|binding| binding.port == Port::Parents);
        Ok(Self {
            bytes,
            hash,
            bound,
            compiled_stages,
            channel_accesses,
            row_extents,
            row_bound,
            reads,
            launch: std::sync::OnceLock::new(),
            geometry_taint: std::sync::OnceLock::new(),
            shadow_plan: std::sync::OnceLock::new(),
            pricing,
            binds_pages,
            emitted: Mutex::new(HashMap::new()),
        })
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

/// Forgets the programs registered against the boot's model.
pub(crate) fn reset() {
    *PLAN_FORMS.lock().unwrap() = LibraryForms::ALL;
    *global() = Registry::new(NonZeroUsize::new(DEFAULT_CAPACITY).expect("nonzero capacity"));
}

/// Registers `bytes` for the served model. `rows` is the most rows one of
/// its fires binds (`model.max-pass-rows()`), which a row-extent channel is
/// declared at ([`identity`]); `None` for a float pass.
pub fn register(
    bytes: Vec<u8>,
    lanes: Vec<Vec<OutputDecl>>,
    tenant: Tenant,
    rows: Option<u32>,
) -> Result<Arc<RegisteredProgram>, RegisterError> {
    let eta = crate::model::model().eta_caps();
    let caps = EngineCaps {
        lora: eta.has_lora,
        attn_page_mask: eta.has_attn_page_mask,
    };
    register_with(bytes, lanes, caps, eta.budget, tenant, rows)
}

/// Programs the process registered anew since boot, and registrations that
/// found one held ([`Registry::counts`]).
#[must_use]
pub fn counts() -> (u64, u64) {
    global().counts()
}

/// Decode, bind and compile run outside the global lock, so a slow or refused
/// program never holds up, or poisons, every other process's registry access.
fn register_with(
    bytes: Vec<u8>,
    lanes: Vec<Vec<OutputDecl>>,
    caps: EngineCaps,
    budget: Budget,
    tenant: Tenant,
    rows: Option<u32>,
) -> Result<Arc<RegisteredProgram>, RegisterError> {
    let (bytes, row_extents) = identity(bytes, rows)?;
    let hash = container_hash(&bytes);
    if let Some(hit) = global().hit(hash, &bytes, &lanes)? {
        return Ok(hit);
    }
    let entry = Arc::new(RegisteredProgram::build(
        bytes,
        hash,
        lanes,
        caps,
        budget,
        (row_extents, rows.unwrap_or(0)),
    )?);
    Ok(global().insert(entry, tenant))
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

#[cfg(test)]
mod budget_tests {
    use super::{RegisterError, Registry};
    use eta_ir::container::{StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::{EngineCaps, Stage};
    use eta_ir::types::{Literal, Shape};
    use eta_ir::validate::{Budget, ValidateError};
    use std::num::NonZeroUsize;

    fn broadcast(shape: Shape) -> Vec<u8> {
        TraceContainer {
            stages: vec![StageProgram {
                stage: Stage::Prologue,
                ops: vec![
                    Op::Const(Literal::F32(0.5)),
                    Op::Broadcast { value: 0, shape },
                ],
            }],
            ..TraceContainer::default()
        }
        .encode()
    }

    /// A guest declaring a prologue past an instance's state room is refused
    /// at its registration by name, and nothing of it is held; one within
    /// it registers.
    #[test]
    fn an_oversized_prologue_is_refused_at_registration_by_name() {
        let mut registry = Registry::new(NonZeroUsize::new(4).unwrap());
        let budget = Budget {
            room_bytes: 64 << 20,
            submit: None,
        };
        let refused = registry
            .register(
                broadcast(Shape::matrix(1 << 16, 1 << 16)),
                vec![],
                EngineCaps::default(),
                budget,
                None,
                None,
            )
            .unwrap_err();
        assert!(
            matches!(
                refused,
                RegisterError::Bind(ValidateError::StagePastRoom {
                    stage: Stage::Prologue,
                    room,
                    ..
                }) if room == 64 << 20
            ),
            "{refused}"
        );
        assert!(refused.to_string().contains("state room"), "{refused}");
        assert_eq!(registry.inner.len(), 0, "a refused program is not held");

        let fits = registry
            .register(
                broadcast(Shape::vector(4096)),
                vec![],
                EngineCaps::default(),
                budget,
                None,
                None,
            )
            .expect("a 16 KiB prologue registers");
        assert_eq!(fits.bound.budget, Some(budget));
    }
}

#[cfg(test)]
mod tenant_tests {
    use super::{RegisterError, Registry, Tenant};
    use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::{EngineCaps, Stage};
    use eta_ir::types::{Dtype, Literal, Shape};
    use eta_ir::validate::Budget;
    use std::num::NonZeroUsize;

    const ROOM: Budget = Budget {
        room_bytes: 1 << 30,
        submit: None,
    };

    /// A distinct program for each `n`.
    fn program(n: u32) -> Vec<u8> {
        TraceContainer {
            stages: vec![StageProgram {
                stage: Stage::Prologue,
                ops: vec![
                    Op::Const(Literal::F32(1.0)),
                    Op::Broadcast {
                        value: 0,
                        shape: Shape::vector(n),
                    },
                ],
            }],
            ..TraceContainer::default()
        }
        .encode()
    }

    fn held(registry: &mut Registry, n: u32) -> bool {
        registry
            .inner
            .contains(&eta_ir::container_hash(&program(n)))
    }

    /// One tenant registering program after program churns only its own
    /// share of the registry (its size over the tenants holding programs):
    /// another tenant's programs within their share stay.
    #[test]
    fn one_tenant_churns_only_its_own_share_of_the_registry() {
        let mut registry = Registry::new(NonZeroUsize::new(4).unwrap());
        let (a, b): (Tenant, Tenant) = (
            Some(uuid::Uuid::from_u128(1)),
            Some(uuid::Uuid::from_u128(2)),
        );
        let register = |registry: &mut Registry, n: u32, tenant: Tenant| {
            registry
                .register(
                    program(n),
                    vec![],
                    EngineCaps::default(),
                    ROOM,
                    tenant,
                    None,
                )
                .expect("registers");
        };
        for n in 1..=4 {
            register(&mut registry, n, a);
        }
        // b's first program takes a's least recent: a is past its share (2).
        register(&mut registry, 100, b);
        assert!(!held(&mut registry, 1) && held(&mut registry, 100));
        register(&mut registry, 101, b);
        assert!(!held(&mut registry, 2));
        // a churns a thousand programs: only its own two slots turn over.
        for n in 5..1005 {
            register(&mut registry, n, a);
        }
        assert!(
            held(&mut registry, 100) && held(&mut registry, 101),
            "b's share stands"
        );
        assert!(held(&mut registry, 1003) && held(&mut registry, 1004));
        assert_eq!(registry.inner.len(), 4);
        // The runtime's own programs (no tenant) are a tenant like any other.
        register(&mut registry, 2000, None);
        assert!(held(&mut registry, 100) || held(&mut registry, 101));
        assert!(held(&mut registry, 2000));
    }

    /// A program whose channel cells pass an instance's state room is
    /// refused by name at its registration (`Pricing::channel_bytes`).
    #[test]
    fn channel_cells_past_the_room_are_refused_by_name() {
        let mut registry = Registry::new(NonZeroUsize::new(4).unwrap());
        let bytes = TraceContainer {
            channels: vec![ChannelDecl {
                shape: Shape::vector(1 << 20),
                dtype: ChanDType::Concrete(Dtype::F32),
                capacity: 7,
                host_role: HostRole::None,
                seeded: false,
            }],
            ..TraceContainer::default()
        }
        .encode();
        let room = Budget {
            room_bytes: 16 << 20,
            submit: None,
        };
        let refused = registry
            .register(bytes, vec![], EngineCaps::default(), room, None, None)
            .unwrap_err();
        assert!(
            matches!(
                refused,
                RegisterError::ChannelsPastRoom { bytes, room } if bytes == 32 << 20 && room == 16 << 20
            ),
            "{refused}"
        );
    }
}

#[cfg(test)]
mod register_tests {
    use super::{RegisterError, register_with};
    use eta_ir::container::{StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::{EngineCaps, Stage};
    use eta_ir::validate::{Budget, ValidateError};

    const ROOM: Budget = Budget {
        room_bytes: 64 << 20,
        submit: None,
    };

    #[test]
    fn an_out_of_range_channel_is_refused_and_the_registry_stays_usable() {
        let bad = TraceContainer {
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::ChanTake(5)],
            }],
            ..TraceContainer::default()
        };
        let err = register_with(
            bad.encode(),
            vec![],
            EngineCaps::default(),
            ROOM,
            None,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(err, RegisterError::Bind(ValidateError::Body { .. })),
            "{err}"
        );
        assert!(err.to_string().contains('5'), "{err}");
        let good = TraceContainer::default();
        let hash = register_with(
            good.encode(),
            vec![],
            EngineCaps::default(),
            ROOM,
            None,
            None,
        )
        .expect("a valid program registers after a refused one")
        .hash;
        assert!(super::lookup(hash).is_some());
    }
}

#[cfg(test)]
mod row_extent_tests {}
