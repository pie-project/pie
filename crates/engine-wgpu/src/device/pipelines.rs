use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use engine_gpu::error::{Fault, Result};

use super::ctx::{Context, Core};

pub struct Pipeline {
    pub name: &'static str,

    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,

    pub bindings: u32,

    pub used: Vec<bool>,

    pub read_only: Vec<bool>,

    pub uniform: Option<u32>,

    pub push_bytes: u32,
    pub local: [u32; 3],
}

struct Declared {
    bindings: u32,
    used: Vec<bool>,
    /// The storage buffers `main` reaches, which the pipeline's layout
    /// binds: those the kernel declares for itself, and the pool segments
    /// beside them (`pie_seg0..3`, as `prim/pool.slang` declares them).
    own: u32,
    pool: u32,
    read_only: Vec<bool>,
    uniform: Option<u32>,
    push_bytes: u32,
    local: [u32; 3],
}

/// Slang's WGSL with its `enable subgroups;` dropped: Naga refuses the
/// directive (gfx-rs/wgpu#5555) and takes the subgroup builtins under the
/// device's `SUBGROUP` feature instead. A browser's Tint needs it, so the
/// module keeps it everywhere else.
fn without_subgroup_enable(wgsl: &str) -> String {
    wgsl.lines()
        .filter(|line| line.trim() != "enable subgroups;")
        .flat_map(|line| [line, "\n"])
        .collect()
}

fn promote_storage(wgsl: &str) -> String {
    let mut out = String::with_capacity(wgsl.len() + 64);
    let mut rest = wgsl;
    while let Some(at) = rest.find("var<storage") {
        let (head, tail) = rest.split_at(at);
        out.push_str(head);
        let close = tail.find('>').map_or(tail.len(), |i| i + 1);
        let (decl, after) = tail.split_at(close);
        let inner: String = decl
            .trim_start_matches("var<storage")
            .trim_end_matches('>')
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        match inner.as_str() {
            "" | ",read" | ",read_write" => out.push_str("var<storage, read_write>"),
            _ => out.push_str(decl),
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

fn reflect(wgsl: &str) -> std::result::Result<Declared, String> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| e.emit_to_string(wgsl))?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| e.emit_to_string(wgsl))?;
    let main_at = module
        .entry_points
        .iter()
        .position(|e| e.name == "main")
        .ok_or_else(|| "no `main` entry point".to_string())?;
    let reached = info.get_entry_point(main_at);

    let mut storage: Vec<(u32, bool, bool)> = Vec::new();
    let (mut own, mut pool) = (0u32, 0u32);
    let mut uniform = None;
    let mut push_bytes = 0u32;
    for (handle, var) in module.global_variables.iter() {
        let Some(binding) = &var.binding else {
            continue;
        };
        if binding.group != 0 {
            return Err(format!(
                "binding {}.{} is not in group 0; this shell binds one group",
                binding.group, binding.binding
            ));
        }
        match var.space {
            naga::AddressSpace::Storage { access } => {
                if !reached[handle].is_empty() {
                    let segment = var
                        .name
                        .as_deref()
                        .is_some_and(|name| name.starts_with("pie_seg"));
                    *(if segment { &mut pool } else { &mut own }) += 1;
                }
                storage.push((
                    binding.binding,
                    !access.contains(naga::StorageAccess::STORE),
                    !reached[handle].is_empty(),
                ));
            }
            naga::AddressSpace::Uniform => {
                if uniform.is_some() {
                    return Err("two uniform blocks; this shell binds one".to_string());
                }
                uniform = Some(binding.binding);
                push_bytes = module.types[var.ty].inner.size(module.to_ctx());
            }
            _ => {}
        }
    }
    let bindings = storage.iter().map(|&(b, _, _)| b + 1).max().unwrap_or(0);
    let mut used = vec![false; bindings as usize];
    let mut read_only = vec![false; bindings as usize];
    for (b, ro, reached) in storage {
        used[b as usize] = reached;
        read_only[b as usize] = ro;
    }
    if let Some(u) = uniform
        && u != bindings
    {
        return Err(format!(
            "the uniform block sits at binding {u} but the storage bindings end at {bindings}"
        ));
    }
    let entry = module
        .entry_points
        .iter()
        .find(|e| e.name == "main")
        .ok_or_else(|| "no `main` entry point".to_string())?;
    Ok(Declared {
        bindings,
        used,
        own,
        pool,
        read_only,
        uniform,
        push_bytes,
        local: entry.workgroup_size,
    })
}

pub(crate) type View = (wgpu::Buffer, u64, u64);

type BindKey = (usize, Vec<View>, Vec<u8>);

#[derive(Default)]
struct BindCache {
    groups: HashMap<BindKey, wgpu::BindGroup>,

    chunks: Vec<super::ctx::Held>,
    used: u64,
    /// Where a load laid the arena ([`Pipelines::lay_uniforms`]): `chunks`
    /// are its two epochs, and a cached group's uniforms lie in its cache
    /// slot's ([`Arena`]). `None`: chunks are taken as the groups need them.
    laid: Option<Arena>,
}

/// The bind cache's uniform arena as a load lays it: two epochs of `slots`
/// cache slots each, slot `i` of epoch `e` the `stride` bytes at
/// `(e * slots + i) * stride` (in chunks of `per_chunk` slots). A group
/// takes the next slot of the epoch it is cached in; a full epoch evicts
/// every group and the cache fills the other one. A slot is written again
/// only two evictions after its group's: by then its group is gone from
/// the cache, and no frame still encoding reads it, as a frame binds no
/// more groups than an epoch holds (`slots` is at least a fire's
/// dispatches). A frame already submitted reads what its groups held: a
/// queue write lands after the work submitted before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Arena {
    pub(crate) slots: usize,
    pub(crate) stride: u64,
    pub(crate) per_chunk: u64,
    epoch: usize,
    next: usize,
}

impl Arena {
    pub(crate) fn new(slots: usize, stride: u64) -> Arena {
        Arena {
            slots: slots.max(1),
            stride,
            per_chunk: (UNIFORM_ARENA_CHUNK / stride.max(1)).max(1),
            epoch: 0,
            next: 0,
        }
    }

    /// The chunks both epochs take.
    pub(crate) fn chunks(&self) -> usize {
        (2 * self.slots as u64).div_ceil(self.per_chunk) as usize
    }

    /// Whether the epoch is full: the cache evicts every group, and the
    /// next takes the other epoch's first slot.
    pub(crate) fn full(&self) -> bool {
        self.next >= self.slots
    }

    /// Evicts the epoch: the other one fills from its first slot.
    pub(crate) fn flip(&mut self) {
        self.epoch ^= 1;
        self.next = 0;
    }

    /// The next slot of the epoch: its chunk and byte there.
    pub(crate) fn take(&mut self) -> (usize, u64) {
        let slot = (self.epoch * self.slots + self.next) as u64;
        self.next += 1;
        (
            (slot / self.per_chunk) as usize,
            (slot % self.per_chunk) * self.stride,
        )
    }
}

const BIND_CACHE_CAP: usize = 16 * 1024;
const UNIFORM_ARENA_CHUNK: u64 = 1 << 20;

/// One chunk of the bind cache's uniform arena.
fn arena_chunk(core: &Core) -> super::ctx::Held {
    core.held(
        &wgpu::BufferDescriptor {
            label: Some("pie uniform arena"),
            size: UNIFORM_ARENA_CHUNK,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
        engine_gpu::books::owners::UNIFORMS,
    )
}

#[derive(Default)]
pub struct Pipelines {
    built: RefCell<HashMap<&'static str, Arc<Pipeline>>>,
    /// The kernels built, in the order their pipelines were.
    order: RefCell<Vec<&'static str>>,
    binds: RefCell<BindCache>,
    bind_hits: Cell<u64>,
    bind_misses: Cell<u64>,
}

static BIND_TRAFFIC: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

#[must_use]
pub fn bind_traffic() -> (u64, u64) {
    (
        BIND_TRAFFIC[0].load(std::sync::atomic::Ordering::Relaxed),
        BIND_TRAFFIC[1].load(std::sync::atomic::Ordering::Relaxed),
    )
}

pub(crate) fn reset_bind_traffic() {
    BIND_TRAFFIC[0].store(0, std::sync::atomic::Ordering::Relaxed);
    BIND_TRAFFIC[1].store(0, std::sync::atomic::Ordering::Relaxed);
}

impl std::fmt::Debug for Pipelines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipelines")
            .field("pipelines", &self.built.borrow().len())
            .field("compiles", &self.order.borrow().len())
            .finish()
    }
}

impl Pipelines {
    #[must_use]
    pub fn new() -> Pipelines {
        Pipelines::default()
    }

    #[must_use]
    pub fn compiled(&self) -> u64 {
        self.order.borrow().len() as u64
    }

    /// The kernels whose pipelines were built after the first `from`, in
    /// the order built.
    #[must_use]
    pub fn built_since(&self, from: u64) -> Vec<&'static str> {
        let order = self.order.borrow();
        order[(from as usize).min(order.len())..].to_vec()
    }

    /// Holds kernel `key` of `file` to the storage buffers `device` binds a
    /// shader stage, both counts read from the module it is compiled from.
    pub fn hold(device: &Context, file: &'static str, key: &'static str) -> Result<()> {
        let (own, pool) = storage(file, key)?;
        engine_gpu::storage_buffers(key, own, pool, device.storage_buffers())
    }

    #[must_use]
    pub fn bind_traffic(&self) -> (u64, u64) {
        (self.bind_hits.get(), self.bind_misses.get())
    }

    pub(crate) fn bind_group(
        &self,
        core: &Arc<Core>,
        pipeline: &Pipeline,
        views: &[View],
        scalars: &[u8],
        uniform_bytes: u64,
    ) -> wgpu::BindGroup {
        let key: BindKey = (
            std::ptr::from_ref(pipeline) as usize,
            views.to_vec(),
            scalars.to_vec(),
        );
        if let Some(group) = self.binds.borrow().groups.get(&key) {
            self.bind_hits.set(self.bind_hits.get() + 1);
            BIND_TRAFFIC[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return group.clone();
        }
        self.bind_misses.set(self.bind_misses.get() + 1);
        BIND_TRAFFIC[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut binds = self.binds.borrow_mut();
        let binds = &mut *binds;
        match binds.laid.as_mut() {
            // A full epoch evicts every group its slots hold.
            Some(arena) if arena.full() => {
                binds.groups.clear();
                arena.flip();
            }
            Some(_) => {}
            None if binds.groups.len() >= BIND_CACHE_CAP => {
                binds.groups.clear();
                // The groups that read the earlier chunks are gone: keep
                // the one the next uniform is written to.
                let from = binds.chunks.len().saturating_sub(1);
                binds.chunks.drain(..from);
                binds.used = 0;
            }
            None => {}
        }
        // The group's slot of the laid arena, whether or not it binds
        // uniforms: slot and cache entry are one.
        let slot = binds.laid.as_mut().map(Arena::take);
        let uniform = pipeline.uniform.map(|binding| {
            let (written, at) = match slot {
                Some(slot) => slot,
                None => {
                    let align = u64::from(core.limits.min_uniform_buffer_offset_alignment).max(16);
                    let at = binds.used.next_multiple_of(align);
                    if binds.chunks.is_empty() || at + uniform_bytes > UNIFORM_ARENA_CHUNK {
                        binds.chunks.push(arena_chunk(core));
                        binds.used = 0;
                    }
                    let at = binds.used.next_multiple_of(align);
                    binds.used = at + uniform_bytes;
                    (binds.chunks.len() - 1, at)
                }
            };
            let chunk = (*binds.chunks[written]).clone();
            let mut padded = scalars.to_vec();
            padded.resize(uniform_bytes as usize, 0);
            core.queue.write_buffer(&chunk, at, &padded);
            (binding, chunk, at)
        });
        let mut entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::with_capacity(views.len() + 1);
        for (at, (buffer, offset, size)) in views.iter().enumerate() {
            if at >= pipeline.used.len() || !pipeline.used[at] {
                continue;
            }
            entries.push(wgpu::BindGroupEntry {
                binding: at as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: *offset,
                    size: std::num::NonZeroU64::new(*size),
                }),
            });
        }
        if let Some((binding, chunk, at)) = &uniform {
            entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: chunk,
                    offset: *at,
                    size: std::num::NonZeroU64::new(uniform_bytes),
                }),
            });
        }
        let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(pipeline.name),
            layout: &pipeline.layout,
            entries: &entries,
        });
        binds.groups.insert(key, group.clone());
        group
    }

    /// The uniform bytes a dispatch of any pipeline built takes, at the
    /// device's uniform offset alignment.
    #[must_use]
    pub fn uniform_stride(&self, device: &Context) -> u64 {
        let align = u64::from(device.core().limits.min_uniform_buffer_offset_alignment).max(16);
        self.built
            .borrow()
            .values()
            .map(|pipeline| u64::from(pipeline.push_bytes.next_multiple_of(16).max(16)))
            .max()
            .unwrap_or(16)
            .next_multiple_of(align)
    }

    /// Lays the bind cache's uniform arena ([`Arena`]): two epochs of a
    /// cache slot for every group the cache keeps, at least a fire's
    /// `dispatches`, each at the most any built pipeline's uniforms take.
    /// No dispatch takes a chunk past them. The groups cached before are
    /// dropped, their uniforms written again where they are next bound.
    pub fn lay_uniforms(&self, device: &Context, dispatches: u32) {
        let arena = Arena::new(
            BIND_CACHE_CAP.max(dispatches as usize),
            self.uniform_stride(device),
        );
        let core = device.core();
        let mut binds = self.binds.borrow_mut();
        binds.groups.clear();
        binds.chunks.truncate(arena.chunks());
        while binds.chunks.len() < arena.chunks() {
            let chunk = arena_chunk(core);
            binds.chunks.push(chunk);
        }
        binds.used = 0;
        binds.laid = Some(arena);
    }

    /// The pipeline of kernels-slang's module `key` (an instantiation at a
    /// tier, `kernels_slang::key`), from `file`; built on first use.
    pub fn get(
        &self,
        device: &Context,
        file: &'static str,
        key: &'static str,
    ) -> Result<Arc<Pipeline>> {
        if let Some(pipeline) = self.built.borrow().get(key) {
            return Ok(Arc::clone(pipeline));
        }
        let wgsl = module(file, key)?;
        let pipeline = Arc::new(build(device.core(), file, key, &wgsl)?);
        self.order.borrow_mut().push(key);
        self.built.borrow_mut().insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }
}

/// kernels-slang's module `key` of `file`, as this shell compiles it.
fn module(file: &'static str, key: &'static str) -> Result<String> {
    let wgsl = kernels_slang::wgsl(key).ok_or(Fault::Shader {
        file,
        entrypoint: key,
        why: "kernels-slang holds no WGSL module by that key (built without \
              `kernels-slang/wgsl`, or the instantiation is limited to SPIR-V)"
            .to_string(),
    })?;
    Ok(without_subgroup_enable(&promote_storage(wgsl)))
}

/// The storage buffers kernel `key` of `file` binds in its shader stage,
/// read from the module it is compiled from: those it declares for itself,
/// then the pool segments beside them. Only what `main` reaches counts, as
/// only that is in the pipeline's layout.
pub fn storage(file: &'static str, key: &'static str) -> Result<(u32, u32)> {
    let declared = reflect(&module(file, key)?).map_err(|why| Fault::Shader {
        file,
        entrypoint: key,
        why,
    })?;
    Ok((declared.own, declared.pool))
}

fn descriptor<'a>(
    name: &'static str,
    module: &'a wgpu::ShaderModule,
) -> wgpu::ComputePipelineDescriptor<'a> {
    wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: None,
        module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    }
}

fn compile(
    core: &Arc<Core>,
    name: &'static str,
    wgsl: &str,
) -> std::result::Result<wgpu::ComputePipeline, String> {
    crate::guest::validated(core, "create_compute_pipeline", || {
        let module = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
            });
        core.device
            .create_compute_pipeline(&descriptor(name, &module))
    })
}

fn build(
    core: &Arc<Core>,
    file: &'static str,
    name: &'static str,
    wgsl: &str,
) -> Result<Pipeline> {
    let shader = |why: String| Fault::Shader {
        file,
        entrypoint: name,
        why,
    };
    let declared = reflect(wgsl).map_err(shader)?;
    // Every kernel is held here, at the latest: one the adapter cannot
    // bind is refused by its counts, not as a pipeline that failed to
    // validate.
    engine_gpu::storage_buffers(
        name,
        declared.own,
        declared.pool,
        core.limits.max_storage_buffers_per_shader_stage,
    )?;
    let pipeline = compile(core, name, wgsl).map_err(shader)?;
    let layout = pipeline.get_bind_group_layout(0);
    Ok(Pipeline {
        name,
        pipeline,
        layout,
        bindings: declared.bindings,
        used: declared.used,
        read_only: declared.read_only,
        uniform: declared.uniform,
        push_bytes: declared.push_bytes,
        local: declared.local,
    })
}

#[cfg(test)]
mod tests {
    use super::Arena;

    /// Fills the cache past two epochs: no slot a live group holds (one of
    /// the epoch being filled, or of the one before, which a frame still
    /// encoding may bind) is taken again; the free-running wrap over the
    /// same bytes (the negative control) takes one.
    #[test]
    fn a_full_cache_overwrites_no_live_groups_uniforms() {
        let (slots, stride) = (16 * 1024, 256);
        let mut arena = Arena::new(slots, stride);
        assert_eq!(arena.chunks(), 8, "two epochs of 16K slots of 256 bytes");
        // Every live group's bytes: (chunk, byte) -> the epoch it is cached in.
        let mut live: std::collections::HashMap<(usize, u64), usize> = Default::default();
        let mut epoch = 0;
        for _ in 0..5 * slots + 7 {
            if arena.full() {
                arena.flip();
                epoch += 1;
                // Two evictions on, a group is in no cache and no frame.
                live.retain(|_, cached| *cached + 1 >= epoch);
            }
            let slot = arena.take();
            assert!(
                !live.contains_key(&slot),
                "slot {slot:?} of a live group taken again at epoch {epoch}"
            );
            live.insert(slot, epoch);
        }
        // The negative control: one ring over the same chunks, wrapping
        // as it runs out, overwrites a live group within the epoch after.
        let ring = (arena.chunks() as u64 * arena.per_chunk) as usize;
        let mut live: std::collections::HashMap<usize, usize> = Default::default();
        let mut overwrote = false;
        let mut epoch = 0;
        for at in 0..3 * slots {
            if at > 0 && at % slots == 0 {
                epoch += 1;
                live.retain(|_, cached| *cached + 1 >= epoch);
            }
            // A ring sized for one epoch's groups, as a free-running wrap
            // over the cache's slots lays it.
            let slot = at % (ring / 2);
            overwrote |= live.insert(slot, epoch).is_some();
        }
        assert!(overwrote, "the free-running wrap takes a live group's slot");
    }
}
