use std::sync::Arc;

use ash::vk;
use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_exec::{Extents, OpParams, ValueDesc, describe, layout};

use crate::device::Context;
use crate::device::alloc::{Buffer, Memory};
use crate::device::ctx::{Core, Frame};
use crate::error::{Fault, Result};

const DESC_WORDS: usize = 9;

const BINDINGS: u32 = 6;

const CFG_BINDING: u32 = 5;

const WORKGROUP: u32 = eta_compiler::codegen::wgsl::WORKGROUP;

const GROUPS_PER_CORE: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Cfg {
    value_count: u32,
    temporary: u32,
    op_count: u32,
    lane: u32,
}

const _: () = assert!(size_of::<Cfg>() == 16);

fn desc_words(d: &ValueDesc) -> [u32; DESC_WORDS] {
    let mut out = [0u32; DESC_WORDS];
    out[0] = d.len;
    out[1] = d.rows;
    out[2] = d.last;
    out[3] = d.rank;
    out[4] = d.dtype;
    for (at, &dim) in d.dims.iter().enumerate() {
        out[5 + at] = dim;
    }
    out
}

fn as_bytes<T: Copy>(items: &[T]) -> Vec<u8> {
    let mut out = Vec::with_capacity(size_of_val(items));
    for item in items {
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts((item as *const T).cast::<u8>(), size_of::<T>())
        });
    }
    out
}

struct Stage {
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,

    pipelines: Vec<vk::Pipeline>,

    groups: u32,
    set: vk::DescriptorSet,

    offsets: Vec<u64>,

    descriptors: Vec<ValueDesc>,
    descs: Buffer,
    params: Buffer,
    offs: Buffer,
    cfg: Buffer,

    heap: Buffer,

    staging: Buffer,

    landing: Buffer,

    dirty: Vec<(u64, u64)>,
}

pub struct Session {
    core: Arc<Core>,

    status: Buffer,
    stages: Vec<Stage>,
    pool: vk::DescriptorPool,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("stages", &self.stages.len())
            .field(
                "heap",
                &self.stages.iter().map(|s| s.heap.bytes()).sum::<u64>(),
            )
            .finish()
    }
}

impl Session {
    pub fn new(
        device: &Context,
        plans: &[LaunchStagePlan],
        code: &[crate::guest::Lowered],
        extents: &Extents,
    ) -> Result<Session> {
        if plans.is_empty() {
            return Err(Fault::Program {
                at: "guest::session",
                why: "a guest package with no stages has nothing to run".into(),
            });
        }
        if code.len() != plans.len() {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "{} stage plans but {} compiled forms; a session runs every stage or none",
                    plans.len(),
                    code.len()
                ),
            });
        }

        let mut status = Buffer::with(device, eta_exec::STATUS_BYTES as u64, Memory::Host)?;
        status.write(0, &[0u8; eta_exec::STATUS_BYTES])?;

        let core = device.core().clone();
        let mut stages = Vec::with_capacity(plans.len());
        for (plan, lowered) in plans.iter().zip(code) {
            stages.push(build_stage(device, &core, plan, lowered, extents)?);
        }

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((BINDINGS - 1) * stages.len() as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(stages.len() as u32),
        ];
        let pool = unsafe {
            core.device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(stages.len() as u32)
                        .pool_sizes(&sizes),
                    None,
                )
                .map_err(|e| core.fault("vkCreateDescriptorPool", e))?
        };

        let mut session = Session {
            core,
            status,
            stages,
            pool,
        };
        session.bind_sets()?;
        Ok(session)
    }

    fn bind_sets(&mut self) -> Result<()> {
        let d = &self.core.device;
        for at in 0..self.stages.len() {
            let set = unsafe {
                let layouts = [self.stages[at].set_layout];
                d.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.pool)
                        .set_layouts(&layouts),
                )
                .map_err(|e| self.core.fault("vkAllocateDescriptorSets", e))?[0]
            };
            let stage = &self.stages[at];
            let buffers = [
                (&self.status, vk::DescriptorType::STORAGE_BUFFER),
                (&stage.descs, vk::DescriptorType::STORAGE_BUFFER),
                (&stage.params, vk::DescriptorType::STORAGE_BUFFER),
                (&stage.offs, vk::DescriptorType::STORAGE_BUFFER),
                (&stage.heap, vk::DescriptorType::STORAGE_BUFFER),
                (&stage.cfg, vk::DescriptorType::UNIFORM_BUFFER),
            ];
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|(buffer, _)| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(buffer.slab().buffer)
                        .offset(0)
                        .range(buffer.bytes())]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buffers
                .iter()
                .zip(&infos)
                .enumerate()
                .map(|(binding, ((_, kind), info))| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding as u32)
                        .descriptor_type(*kind)
                        .buffer_info(info)
                })
                .collect();
            unsafe { d.update_descriptor_sets(&writes, &[]) };
            self.stages[at].set = set;
        }
        Ok(())
    }

    #[must_use]
    pub fn stages(&self) -> usize {
        self.stages.len()
    }

    #[must_use]
    pub fn span(&self, stage: usize, value: u32) -> Option<(u64, u64)> {
        let stage = self.stages.get(stage)?;
        let at = value as usize;
        Some((
            *stage.offsets.get(at)?,
            stage.descriptors.get(at)?.device_bytes(),
        ))
    }

    #[must_use]
    pub fn heap(&self, stage: usize) -> Option<&Buffer> {
        self.stages.get(stage).map(|stage| &stage.heap)
    }

    pub fn stage_in(&mut self, stage: usize, value: u32, bytes: &[u8]) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        if bytes.len() as u64 != len {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "value {value} of stage {stage} holds {len} bytes, and a root of \
                     {} was offered; a short root leaves the last fire's tail under it",
                    bytes.len()
                ),
            });
        }
        let stage = &mut self.stages[stage];
        stage.staging.write(at, bytes)?;
        stage.dirty.push((at, len));
        Ok(())
    }

    pub fn flush(&mut self, frame: &mut Frame, stage: usize) -> Result<()> {
        let Some(stage) = self.stages.get_mut(stage) else {
            return Ok(());
        };
        for (at, len) in stage.dirty.drain(..) {
            frame.copy(&stage.staging, at, &stage.heap, at, len)?;
        }
        Ok(())
    }

    pub fn copy_in(
        &mut self,
        frame: &mut Frame,
        stage: usize,
        value: u32,
        source: &Buffer,
        source_at: u64,
        len: u64,
    ) -> Result<()> {
        let (at, held) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        if len > held {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!(
                    "value {value} of stage {stage} holds {held} bytes and a copy of {len} \
                     was asked for"
                ),
            });
        }
        let stage = &self.stages[stage];
        frame.copy(source, source_at, &stage.heap, at, len)
    }

    pub fn read_back(&mut self, frame: &mut Frame, stage: usize, value: u32) -> Result<()> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let stage = &self.stages[stage];
        frame.copy(&stage.heap, at, &stage.landing, at, len)
    }

    pub fn taken(&self, stage: usize, value: u32) -> Result<Vec<u8>> {
        let (at, len) = self.span(stage, value).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("value {value} is not stage {stage}'s"),
        })?;
        let mut out = vec![0u8; len as usize];
        self.stages[stage].landing.read(at, &mut out)?;
        Ok(out)
    }

    pub fn dispatch(&self, frame: &Frame, at: usize) -> Result<()> {
        let stage = self.stages.get(at).ok_or_else(|| Fault::Program {
            at: "guest::session",
            why: format!("stage {at} is past this package's {}", self.stages.len()),
        })?;
        let d = &self.core.device;
        unsafe {
            let cmd = frame.cmd();

            let before = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[before],
                &[],
                &[],
            );
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                stage.layout,
                0,
                &[stage.set],
                &[],
            );

            let between = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            for (step, &pipeline) in stage.pipelines.iter().enumerate() {
                if step > 0 {
                    d.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[between],
                        &[],
                        &[],
                    );
                }
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                d.cmd_dispatch(cmd, stage.groups, 1, 1);
            }

            let after = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::TRANSFER_READ
                        | vk::AccessFlags::HOST_READ,
                );
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[after],
                &[],
                &[],
            );
        }
        Ok(())
    }

    pub fn status(&self) -> Result<Option<u32>> {
        let mut raw = [0u8; eta_exec::STATUS_BYTES];
        self.status.read(0, &mut raw)?;
        let code = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        Ok((code != 0).then_some(code))
    }

    pub fn clear_status(&mut self) -> Result<()> {
        self.status.write(0, &[0u8; eta_exec::STATUS_BYTES])
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            let d = &self.core.device;
            d.destroy_descriptor_pool(self.pool, None);
            for stage in &self.stages {
                for &pipeline in &stage.pipelines {
                    d.destroy_pipeline(pipeline, None);
                }
                d.destroy_pipeline_layout(stage.layout, None);
                d.destroy_descriptor_set_layout(stage.set_layout, None);
            }
        }
    }
}

fn groups_for(descriptors: &[ValueDesc], device: &Context) -> u32 {
    let widest = descriptors.iter().map(|d| d.len).max().unwrap_or(1).max(1);
    let wanted = widest.div_ceil(WORKGROUP);
    let cap = device.cores().max(1).saturating_mul(GROUPS_PER_CORE);
    wanted.clamp(1, cap)
}

fn alias_reshapes(plan: &LaunchStagePlan, values: &mut [u64]) {
    for (result, source) in eta_compiler::codegen::wgsl_analysis::analyze_stage(plan).aliases {
        let Some(&at) = values.get(source as usize) else {
            continue;
        };
        if let Some(slot) = values.get_mut(result as usize) {
            *slot = at;
        }
    }
}

fn build_stage(
    device: &Context,
    core: &Arc<Core>,
    plan: &LaunchStagePlan,
    lowered: &crate::guest::Lowered,
    extents: &Extents,
) -> Result<Stage> {
    let descriptors = plan
        .value_types
        .iter()
        .map(|value| {
            describe(value, extents).map_err(|why| Fault::Program {
                at: "guest::session",
                why: format!("a guest value's shape does not resolve against this fire: {why:?}"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut scratch = layout(&descriptors).map_err(|why| Fault::Program {
        at: "guest::session",
        why: format!("a guest pass's scratch does not fit: {why:?}"),
    })?;
    alias_reshapes(plan, &mut scratch.values);

    let desc_bytes: Vec<u8> = descriptors
        .iter()
        .flat_map(|d| as_bytes(&desc_words(d)))
        .collect();
    let offset_words: Vec<u32> = scratch
        .values
        .iter()
        .map(|&at| u32::try_from(at).unwrap_or(u32::MAX))
        .collect();

    let mut records = Vec::with_capacity(plan.ops.len());
    let mut result_base = 0u32;
    for op in &plan.ops {
        records.push(OpParams::of(
            op,
            result_base,
            eta_exec::OpRuntime::default(),
        ));
        result_base += u32::from(op.result_count);
    }
    let config = Cfg {
        value_count: u32::try_from(plan.value_types.len()).unwrap_or(u32::MAX),
        temporary: u32::try_from(scratch.temporary / 4).unwrap_or(u32::MAX),
        op_count: u32::try_from(plan.ops.len()).unwrap_or(u32::MAX),
        lane: 0,
    };

    let mut descs = Buffer::zeroed(device, desc_bytes.len().max(DESC_WORDS * 4) as u64)?;
    descs.write(0, &desc_bytes)?;
    let mut offs = Buffer::zeroed(device, (offset_words.len().max(1) * 4) as u64)?;
    offs.write(0, &as_bytes(&offset_words))?;
    let mut params = Buffer::zeroed(
        device,
        (records.len().max(1) * size_of::<OpParams>()) as u64,
    )?;
    if !records.is_empty() {
        params.write(0, &as_bytes(&records))?;
    }
    let mut cfg = Buffer::with(device, size_of::<Cfg>() as u64, Memory::Host)?;
    cfg.write(0, &as_bytes(&[config]))?;

    let bytes = scratch.total.max(256);
    let heap = Buffer::zeroed(device, bytes)?;
    let staging = Buffer::with(device, bytes, Memory::Host)?;
    let landing = Buffer::with(device, bytes, Memory::Host)?;

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..BINDINGS)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(if i == CFG_BINDING {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                })
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let d = &core.device;
    unsafe {
        let set_layout = d
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
            .map_err(|e| core.fault("vkCreateDescriptorSetLayout", e))?;
        let set_layouts = [set_layout];
        let layout = match d.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
            None,
        ) {
            Ok(l) => l,
            Err(e) => {
                d.destroy_descriptor_set_layout(set_layout, None);
                return Err(core.fault("vkCreatePipelineLayout", e));
            }
        };
        let module = match d.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&lowered.words),
            None,
        ) {
            Ok(m) => m,
            Err(e) => {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                return Err(core.fault("vkCreateShaderModule", e));
            }
        };

        let names: Vec<std::ffi::CString> = lowered
            .entries
            .iter()
            .map(|entry| {
                std::ffi::CString::new(entry.as_str())
                    .expect("an emitted entry name is a WGSL identifier and holds no nul")
            })
            .collect();
        let infos: Vec<vk::ComputePipelineCreateInfo> = names
            .iter()
            .map(|name| {
                vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(name),
                    )
                    .layout(layout)
            })
            .collect();
        if infos.is_empty() {
            d.destroy_shader_module(module, None);
            d.destroy_pipeline_layout(layout, None);
            d.destroy_descriptor_set_layout(set_layout, None);
            return Err(Fault::Program {
                at: "guest::session",
                why: "a stage emitted no dispatches, so nothing would run".into(),
            });
        }
        let built = d.create_compute_pipelines(vk::PipelineCache::null(), &infos, None);
        d.destroy_shader_module(module, None);
        match built {
            Ok(pipelines) => Ok(Stage {
                layout,
                set_layout,
                pipelines,
                groups: if lowered.strides_the_grid {
                    groups_for(&descriptors, device)
                } else {
                    1
                },
                set: vk::DescriptorSet::null(),
                offsets: scratch.values,
                descriptors,
                descs,
                params,
                offs,
                cfg,
                heap,
                staging,
                landing,
                dirty: Vec::new(),
            }),
            Err((_, e)) => {
                d.destroy_pipeline_layout(layout, None);
                d.destroy_descriptor_set_layout(set_layout, None);
                Err(core.fault("vkCreateComputePipelines", e))
            }
        }
    }
}
