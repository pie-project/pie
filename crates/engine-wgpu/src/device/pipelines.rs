use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use kernels_wgpu::Fire;

use crate::error::{Fault, Result};

use super::ctx::{Context, Core};

pub struct Pipeline {
    pub name: &'static str,

    pub tier: kernels_wgpu::Capability,
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
    read_only: Vec<bool>,
    uniform: Option<u32>,
    push_bytes: u32,
    local: [u32; 3],
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
        read_only,
        uniform,
        push_bytes,
        local: entry.workgroup_size,
    })
}

type Cache = (Arc<Core>, wgpu::PipelineCache, Option<std::path::PathBuf>);

pub(crate) type View = (wgpu::Buffer, u64, u64);

type BindKey = (usize, Vec<View>, Vec<u8>);

#[derive(Default)]
struct BindCache {
    groups: HashMap<BindKey, wgpu::BindGroup>,

    chunks: Vec<wgpu::Buffer>,
    used: u64,
}

const BIND_CACHE_CAP: usize = 16 * 1024;
const UNIFORM_ARENA_CHUNK: u64 = 1 << 20;

#[derive(Default)]
pub struct Pipelines {
    built: RefCell<HashMap<&'static str, Arc<Pipeline>>>,
    cache: RefCell<Option<Cache>>,
    compiles: Cell<u64>,
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
            .field("compiles", &self.compiles.get())
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
        self.compiles.get()
    }

    pub fn warm(&self, device: &Context, fire: Fire) -> Result<()> {
        self.get(device, fire).map(|_| ())
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
        if binds.groups.len() >= BIND_CACHE_CAP {
            binds.groups.clear();
            binds.used = 0;
        }
        let uniform = pipeline.uniform.map(|binding| {
            let align = u64::from(core.limits.min_uniform_buffer_offset_alignment).max(16);
            let at = binds.used.next_multiple_of(align);
            if binds.chunks.is_empty() || at + uniform_bytes > UNIFORM_ARENA_CHUNK {
                binds
                    .chunks
                    .push(core.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("pie uniform arena"),
                        size: UNIFORM_ARENA_CHUNK,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }));
                binds.used = 0;
            }
            let at = binds.used.next_multiple_of(align);
            binds.used = at + uniform_bytes;
            let chunk = binds.chunks.last().expect("non-empty").clone();
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

    pub fn get(&self, device: &Context, fire: Fire) -> Result<Arc<Pipeline>> {
        let entrypoint = fire.entrypoint;
        if let Some(pipeline) = self.built.borrow().get(entrypoint) {
            return Ok(Arc::clone(pipeline));
        }

        let mut tiers = device.tiers();
        tiers.push(kernels_wgpu::Capability::Baseline);
        let cache = self.cache(device);
        let mut last = None;
        let mut built = None;
        for tier in tiers {
            let Ok(expanded) = kernels_wgpu::sources::at(entrypoint, tier) else {
                continue;
            };
            let name: &'static str = match tier {
                kernels_wgpu::Capability::Baseline => entrypoint,
                other => Box::leak(other.variant(entrypoint).into_boxed_str()),
            };
            let wgsl = promote_storage(&expanded.wgsl);
            match build(device.core(), cache.as_ref(), fire, name, tier, &wgsl) {
                Ok(pipeline) => {
                    built = Some(pipeline);
                    break;
                }
                Err(fault) if tier != kernels_wgpu::Capability::Baseline => last = Some(fault),
                Err(fault) => return Err(fault),
            }
        }
        let pipeline = Arc::new(built.ok_or_else(|| {
            last.unwrap_or(Fault::Shader {
                file: fire.file,
                entrypoint,
                why: "the kernel table holds no variant by that name".to_string(),
            })
        })?);
        self.compiles.set(self.compiles.get() + 1);
        self.built
            .borrow_mut()
            .insert(entrypoint, Arc::clone(&pipeline));
        Ok(pipeline)
    }

    fn cache(&self, device: &Context) -> Option<wgpu::PipelineCache> {
        if let Some((_, cache, _)) = self.cache.borrow().as_ref() {
            return Some(cache.clone());
        }
        let core = device.core();
        if !core.enabled.pipeline_cache {
            return None;
        }
        let path = device
            .pipeline_cache_path()
            .map(std::path::Path::to_path_buf);
        let initial = path.as_ref().and_then(|p| std::fs::read(p).ok());

        let cache = unsafe {
            core.device
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: Some("pie pipelines"),
                    data: initial.as_deref(),
                    fallback: true,
                })
        };
        *self.cache.borrow_mut() = Some((Arc::clone(core), cache.clone(), path));
        Some(cache)
    }

    pub fn persist(&self) -> Result<()> {
        let cache = self.cache.borrow();
        let Some((_, cache, Some(path))) = cache.as_ref() else {
            return Ok(());
        };
        let Some(bytes) = cache.get_data() else {
            return Ok(());
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| Fault::Device {
                call: "pipeline cache dir",
                why: e.to_string(),
            })?;
        }
        std::fs::write(path, bytes).map_err(|e| Fault::Device {
            call: "pipeline cache write",
            why: e.to_string(),
        })
    }
}

impl Drop for Pipelines {
    fn drop(&mut self) {
        let _ = self.persist();
    }
}

fn build(
    core: &Arc<Core>,
    cache: Option<&wgpu::PipelineCache>,
    fire: Fire,
    name: &'static str,
    tier: kernels_wgpu::Capability,
    wgsl: &str,
) -> Result<Pipeline> {
    let shader = |why: String| Fault::Shader {
        file: fire.file,
        entrypoint: fire.entrypoint,
        why,
    };
    let declared = reflect(wgsl).map_err(shader)?;
    let scope = core.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = core
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
        });
    let pipeline = core
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache,
        });
    if let Some(error) = pollster::block_on(scope.pop()) {
        return Err(shader(error.to_string()));
    }
    let layout = pipeline.get_bind_group_layout(0);
    Ok(Pipeline {
        name,
        tier,
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
