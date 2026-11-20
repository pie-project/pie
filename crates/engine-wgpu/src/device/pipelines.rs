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

/// A group by its layout, not its pipeline: kernels of one layout (the
/// variants of a kernel, by the rows they take) share what a dispatch binds
/// over the same views.
type BindKey = (wgpu::BindGroupLayout, Vec<View>);

/// A bind group layout by what it holds: each storage binding a kernel
/// reaches and whether it is read-only, and its uniform's binding.
type LayoutKey = (Vec<(u32, bool)>, Option<u32>);

#[derive(Default)]
struct BindCache {
    groups: HashMap<BindKey, wgpu::BindGroup>,
}

/// A bind group of `pipeline` over `views` (those it declares), its uniform
/// `uniform_bytes` of the frames' uniform buffer from its first byte, read
/// at a dispatch's dynamic offset.
pub(crate) fn group_of(
    core: &Arc<Core>,
    pipeline: &Pipeline,
    views: &[View],
    uniform_bytes: u64,
) -> Result<wgpu::BindGroup> {
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
    if let Some(binding) = pipeline.uniform {
        let buffer = core.uniform_buffer.get().ok_or_else(|| Fault::Device {
            call: "create_bind_group",
            why: "the frames' uniform buffer is made at load, and this bind is before it".into(),
        })?;
        entries.push(wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(uniform_bytes),
            }),
        });
    }
    let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(pipeline.name),
        layout: &pipeline.layout,
        entries: &entries,
    });
    core.take_error("create_bind_group")?;
    Ok(group)
}

#[derive(Default)]
pub struct Pipelines {
    built: RefCell<HashMap<&'static str, Arc<Pipeline>>>,
    /// The kernels built, in the order their pipelines were.
    order: RefCell<Vec<&'static str>>,
    binds: RefCell<BindCache>,
    /// One layout for every kernel that declares the same bindings.
    layouts: RefCell<HashMap<LayoutKey, wgpu::BindGroupLayout>>,
    bind_hits: Cell<u64>,
    bind_misses: Cell<u64>,
    /// Groups a fire made of its own frame's scratch views, which no cache
    /// keeps.
    scratch_groups: Cell<u64>,
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

    /// The bind groups made so far: each cache miss, and each group of a
    /// fire's scratch views ([`engine_gpu::serve::Serve::bound`]).
    #[must_use]
    pub fn bound(&self) -> u64 {
        self.bind_misses.get() + self.scratch_groups.get()
    }

    /// Counts a group a fire made of its own frame's scratch views.
    pub(crate) fn note_scratch_group(&self) {
        self.scratch_groups.set(self.scratch_groups.get() + 1);
    }

    #[must_use]
    pub fn bind_traffic(&self) -> (u64, u64) {
        (self.bind_hits.get(), self.bind_misses.get())
    }

    /// The bind group of `pipeline` over `views`, its uniform the frames'
    /// uniform buffer, which a dispatch reads at a dynamic offset (the
    /// frame's slot): one group for every set of views, whatever the scalars.
    pub(crate) fn bind_group(
        &self,
        core: &Arc<Core>,
        pipeline: &Pipeline,
        views: &[View],
        uniform_bytes: u64,
    ) -> Result<wgpu::BindGroup> {
        let key: BindKey = (pipeline.layout.clone(), views.to_vec());
        if let Some(group) = self.binds.borrow().groups.get(&key) {
            self.bind_hits.set(self.bind_hits.get() + 1);
            BIND_TRAFFIC[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(group.clone());
        }
        self.bind_misses.set(self.bind_misses.get() + 1);
        BIND_TRAFFIC[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let group = group_of(core, pipeline, views, uniform_bytes)?;
        self.binds.borrow_mut().groups.insert(key, group.clone());
        Ok(group)
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
        let pipeline = Arc::new(build(device.core(), &self.layouts, file, key, &wgsl)?);
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
    layout: &'a wgpu::PipelineLayout,
) -> wgpu::ComputePipelineDescriptor<'a> {
    wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(layout),
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
    layout: &wgpu::PipelineLayout,
) -> std::result::Result<wgpu::ComputePipeline, String> {
    crate::guest::validated(core, "create_compute_pipeline", || {
        let module = core
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
            });
        core.device
            .create_compute_pipeline(&descriptor(name, &module, layout))
    })
}

/// The bind group layout of a kernel as `declared`: each storage buffer
/// `main` reaches, read-only as it is declared, and its uniform block at a
/// dynamic offset (a dispatch's slot of the frames' uniform buffer). One for
/// every kernel declaring the same, so a group binds any of them.
fn bind_layout(
    core: &Arc<Core>,
    layouts: &RefCell<HashMap<LayoutKey, wgpu::BindGroupLayout>>,
    name: &'static str,
    declared: &Declared,
) -> wgpu::BindGroupLayout {
    let key: LayoutKey = (
        (0..declared.bindings)
            .filter(|&at| declared.used.get(at as usize).copied().unwrap_or(false))
            .map(|at| {
                (
                    at,
                    declared.read_only.get(at as usize).copied().unwrap_or(false),
                )
            })
            .collect(),
        declared.uniform,
    );
    layouts
        .borrow_mut()
        .entry(key)
        .or_insert_with(|| made_layout(core, name, declared))
        .clone()
}

fn made_layout(core: &Arc<Core>, name: &'static str, declared: &Declared) -> wgpu::BindGroupLayout {
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = (0..declared.bindings)
        .filter(|&at| declared.used.get(at as usize).copied().unwrap_or(false))
        .map(|at| wgpu::BindGroupLayoutEntry {
            binding: at,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: declared
                        .read_only
                        .get(at as usize)
                        .copied()
                        .unwrap_or(false),
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    if let Some(binding) = declared.uniform {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: None,
            },
            count: None,
        });
    }
    core.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(name),
            entries: &entries,
        })
}

fn build(
    core: &Arc<Core>,
    layouts: &RefCell<HashMap<LayoutKey, wgpu::BindGroupLayout>>,
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
    let layout = bind_layout(core, layouts, name, &declared);
    let pipelined = core
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let pipeline = compile(core, name, wgsl, &pipelined).map_err(shader)?;
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
