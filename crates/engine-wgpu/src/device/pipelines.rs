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

    /// The bytes one element of each storage binding's array steps, `0`
    /// where the binding is no runtime-sized array: the unit a rebased
    /// view's offset is counted in ([`Rebased`]).
    pub stride: Vec<u32>,

    /// This kernel's module with each storage binding read from an element
    /// offset ([`Rebased`]): built the first time a dispatch binds a window
    /// of a value from past the value's first aligned byte, which the
    /// load's bind walk does for each class region a fire lays past a
    /// value's first row. `None` once built where the module cannot be
    /// rebased.
    rebased: std::sync::OnceLock<Option<Arc<Rebased>>>,

    /// The WGSL the pipeline was built from, kept for its rebased twin.
    wgsl: String,
    file: &'static str,
}

/// A kernel's twin reading each storage binding `b` from element
/// `pie_rebase[b]` of the view bound: a window of a value (a class region's
/// rows of it, from the region's row offset) binds the value from its first
/// aligned byte, and the kernel steps over the elements before the window,
/// so the window binds the group the load built whatever row its region
/// starts at, never a copy of its rows (a WebGPU view starts only at the
/// device's storage offset alignment).
pub struct Rebased {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
    /// The binding of the offsets' uniform block, past the kernel's own.
    pub binding: u32,
    /// The bytes of that block: a u32 a storage binding, in vec4 rows.
    pub bytes: u32,
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
    stride: Vec<u32>,
    /// Each storage binding's variable name, by binding.
    names: Vec<Option<String>>,
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

    let mut storage: Vec<(u32, bool, bool, u32, Option<String>)> = Vec::new();
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
                let stride = match module.types[var.ty].inner {
                    naga::TypeInner::Array {
                        size: naga::ArraySize::Dynamic,
                        stride,
                        ..
                    } => stride,
                    _ => 0,
                };
                storage.push((
                    binding.binding,
                    !access.contains(naga::StorageAccess::STORE),
                    !reached[handle].is_empty(),
                    stride,
                    var.name.clone(),
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
    let bindings = storage.iter().map(|&(b, ..)| b + 1).max().unwrap_or(0);
    let mut used = vec![false; bindings as usize];
    let mut read_only = vec![false; bindings as usize];
    let mut stride = vec![0u32; bindings as usize];
    let mut names = vec![None; bindings as usize];
    for (b, ro, reached, step, name) in storage {
        used[b as usize] = reached;
        read_only[b as usize] = ro;
        stride[b as usize] = step;
        names[b as usize] = name;
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
        stride,
        names,
    })
}

/// The name of the rebased twin's offsets block ([`Rebased`]).
const REBASE: &str = "pie_rebase";

fn ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// `wgsl` with every index into storage variable `names[b]` stepped by
/// element `b` of the offsets block at `binding` (`name[i]` reading
/// `name[u32(i) + pie_rebase.w[b / 4][b % 4]]`), the block declared at the
/// module's end. Refused where a storage variable is named anywhere but
/// indexed (a pointer to it, its length), which no offset would reach.
fn rebase_wgsl(
    wgsl: &str,
    names: &[Option<String>],
    binding: u32,
) -> std::result::Result<String, String> {
    let by_name: HashMap<&str, usize> = names
        .iter()
        .enumerate()
        .filter_map(|(b, name)| Some((name.as_deref()?, b)))
        .collect();
    let rows = names.len().div_ceil(4).max(1);
    let bytes = wgsl.as_bytes();
    let mut out = String::with_capacity(wgsl.len() + wgsl.len() / 4);
    // The innermost open index each `[` closes: `Some(b)` where it indexes
    // binding `b`'s variable, its `]` closing the offset added.
    let mut open: Vec<Option<usize>> = Vec::new();
    // The source copied out up to here.
    let mut copied = 0usize;
    let mut at = 0usize;
    while at < bytes.len() {
        let c = bytes[at];
        if ident(c) && (at == 0 || !(ident(bytes[at - 1]) || bytes[at - 1] == b'.')) {
            let end = (at..bytes.len())
                .find(|&i| !ident(bytes[i]))
                .unwrap_or(bytes.len());
            let word = &wgsl[at..end];
            if let Some(&b) = by_name.get(word) {
                let next = (end..bytes.len()).find(|&i| !bytes[i].is_ascii_whitespace());
                match next.map(|i| bytes[i]) {
                    Some(b'[') => {
                        let i = next.expect("matched");
                        out.push_str(&wgsl[copied..end]);
                        out.push_str("[u32(");
                        open.push(Some(b));
                        at = i + 1;
                        copied = at;
                        continue;
                    }
                    // Its declaration: `var<storage, ..> name : array<T>;`.
                    Some(b':') if wgsl[..at].trim_end().ends_with('>') => {}
                    _ => {
                        return Err(format!(
                            "storage variable `{word}` is named other than indexed, which \
                             no element offset reaches"
                        ));
                    }
                }
            }
            at = end;
            continue;
        }
        match c {
            b'[' => open.push(None),
            b']' => {
                if let Some(Some(b)) = open.pop() {
                    out.push_str(&wgsl[copied..at]);
                    out.push_str(&format!(") + {REBASE}.w[{}][{}]", b / 4, b % 4));
                    copied = at;
                }
            }
            _ => {}
        }
        at += 1;
    }
    if !open.is_empty() {
        return Err("an index into a storage variable is never closed".to_string());
    }
    out.push_str(&wgsl[copied..]);
    out.push_str(&format!(
        "\nstruct PieRebase {{ w : array<vec4<u32>, {rows}>, }};\n\
         @binding({binding}) @group(0) var<uniform> {REBASE} : PieRebase;\n"
    ));
    Ok(out)
}

pub(crate) type View = (wgpu::Buffer, u64, u64);

/// A group by its layout, not its pipeline: kernels of one layout (the
/// variants of a kernel, by the rows they take) share what a dispatch binds
/// over the same views, at the same uniform size. One layout serves kernels
/// whose uniform blocks differ (a decode walk's scalars and a prefill tile's
/// two more), and a group binds its uniform at one size: a group made for
/// the smaller block fails validation on the larger.
type BindKey = (wgpu::BindGroupLayout, Vec<View>, u64);

/// A bind group layout by what it holds: each storage binding a kernel
/// reaches and whether it is read-only, its uniform's binding, and a
/// rebased twin's offsets block's ([`Rebased`]).
type LayoutKey = (Vec<(u32, bool)>, Option<u32>, Option<u32>);

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
    rebased: Option<&Rebased>,
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
    let layout = match rebased {
        Some(rebased) => {
            let buffer = core.uniform_buffer.get().ok_or_else(|| Fault::Device {
                call: "create_bind_group",
                why: "the frames' uniform buffer is made at load, and this bind is before it"
                    .into(),
            })?;
            entries.push(wgpu::BindGroupEntry {
                binding: rebased.binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(u64::from(rebased.bytes)),
                }),
            });
            &rebased.layout
        }
        None => &pipeline.layout,
    };
    let group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(pipeline.name),
        layout,
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
    /// Of `pipeline`'s twin `rebased` where a view is bound from before its
    /// first byte ([`Rebased`]).
    pub(crate) fn bind_group(
        &self,
        core: &Arc<Core>,
        pipeline: &Pipeline,
        views: &[View],
        uniform_bytes: u64,
        rebased: Option<&Rebased>,
    ) -> Result<wgpu::BindGroup> {
        let layout = rebased.map_or(&pipeline.layout, |rebased| &rebased.layout);
        let key: BindKey = (layout.clone(), views.to_vec(), uniform_bytes);
        if let Some(group) = self.binds.borrow().groups.get(&key) {
            self.bind_hits.set(self.bind_hits.get() + 1);
            BIND_TRAFFIC[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(group.clone());
        }
        self.bind_misses.set(self.bind_misses.get() + 1);
        BIND_TRAFFIC[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let group = group_of(core, pipeline, views, uniform_bytes, rebased)?;
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
        let pipeline = Arc::new(build(device.core(), &self.layouts, file, key, wgsl)?);
        self.order.borrow_mut().push(key);
        self.built.borrow_mut().insert(key, Arc::clone(&pipeline));
        Ok(pipeline)
    }

    /// `pipeline`'s rebased twin ([`Rebased`]), built on first use (counted
    /// as a pipeline built, as [`Self::get`]'s are); `None` where its
    /// module cannot be rebased.
    pub(crate) fn rebased(
        &self,
        device: &Context,
        pipeline: &Pipeline,
    ) -> Result<Option<Arc<Rebased>>> {
        if let Some(held) = pipeline.rebased.get() {
            return Ok(held.clone());
        }
        let built = build_rebased(device.core(), &self.layouts, pipeline)?;
        if built.is_some() {
            self.order.borrow_mut().push(pipeline.name);
        }
        Ok(pipeline.rebased.get_or_init(|| built).clone())
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
    rebase: Option<u32>,
) -> wgpu::BindGroupLayout {
    let key: LayoutKey = (
        (0..declared.bindings)
            .filter(|&at| declared.used.get(at as usize).copied().unwrap_or(false))
            .map(|at| {
                (
                    at,
                    declared
                        .read_only
                        .get(at as usize)
                        .copied()
                        .unwrap_or(false),
                )
            })
            .collect(),
        declared.uniform,
        rebase,
    );
    layouts
        .borrow_mut()
        .entry(key)
        .or_insert_with(|| made_layout(core, name, declared, rebase))
        .clone()
}

fn made_layout(
    core: &Arc<Core>,
    name: &'static str,
    declared: &Declared,
    rebase: Option<u32>,
) -> wgpu::BindGroupLayout {
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
    for binding in declared.uniform.into_iter().chain(rebase) {
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
    wgsl: String,
) -> Result<Pipeline> {
    let shader = |why: String| Fault::Shader {
        file,
        entrypoint: name,
        why,
    };
    let declared = reflect(&wgsl).map_err(shader)?;
    // Every kernel is held here, at the latest: one the adapter cannot
    // bind is refused by its counts, not as a pipeline that failed to
    // validate.
    engine_gpu::storage_buffers(
        name,
        declared.own,
        declared.pool,
        core.limits.max_storage_buffers_per_shader_stage,
    )?;
    let (pipeline, layout) =
        pipeline_of(core, layouts, name, &declared, None, &wgsl).map_err(shader)?;
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
        stride: declared.stride,
        rebased: std::sync::OnceLock::new(),
        wgsl,
        file,
    })
}

/// The pipeline of `wgsl` over `declared`'s bindings, and an offsets block
/// at `rebase` where it is a rebased twin's.
fn pipeline_of(
    core: &Arc<Core>,
    layouts: &RefCell<HashMap<LayoutKey, wgpu::BindGroupLayout>>,
    name: &'static str,
    declared: &Declared,
    rebase: Option<u32>,
    wgsl: &str,
) -> std::result::Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), String> {
    let layout = bind_layout(core, layouts, name, declared, rebase);
    let pipelined = core
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
    let pipeline = compile(core, name, wgsl, &pipelined)?;
    Ok((pipeline, layout))
}

/// `pipeline`'s rebased twin ([`Rebased`]): its module with every storage
/// index stepped by its binding's offset, the offsets block bound past the
/// kernel's own uniform. `None` where the module names a storage variable
/// other than by indexing it.
fn build_rebased(
    core: &Arc<Core>,
    layouts: &RefCell<HashMap<LayoutKey, wgpu::BindGroupLayout>>,
    pipeline: &Pipeline,
) -> Result<Option<Arc<Rebased>>> {
    let shader = |why: String| Fault::Shader {
        file: pipeline.file,
        entrypoint: pipeline.name,
        why,
    };
    let declared = reflect(&pipeline.wgsl).map_err(shader)?;
    let binding = declared.uniform.map_or(declared.bindings, |at| at + 1);
    let wgsl = match rebase_wgsl(&pipeline.wgsl, &declared.names, binding) {
        Ok(wgsl) => wgsl,
        Err(why) => {
            tracing::debug!(kernel = pipeline.name, %why, "no rebased twin");
            return Ok(None);
        }
    };
    let (twin, layout) = pipeline_of(core, layouts, pipeline.name, &declared, Some(binding), &wgsl)
        .map_err(shader)?;
    let bytes = 16 * u32::try_from(declared.bindings.div_ceil(4).max(1)).unwrap_or(u32::MAX);
    Ok(Some(Arc::new(Rebased {
        pipeline: twin,
        layout,
        binding,
        bytes,
    })))
}
