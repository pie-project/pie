use crate::ffi_types::{
    PieLoaderBufferDeclSlice, PieLoaderBufferDeclView, PieLoaderBufferIdSlice, PieLoaderBytes,
    PieLoaderDType, PieLoaderDestExtentView, PieLoaderDimSpecSlice, PieLoaderDimSpecView,
    PieLoaderEncodingKind, PieLoaderI64Slice, PieLoaderMemoryPlanView, PieLoaderQuantScheme,
    PieLoaderSourceExtentView, PieLoaderStorageInstrKind, PieLoaderStorageInstrSlice,
    PieLoaderStorageInstrView, PieLoaderStorageProgramView, PieLoaderStridedExtentView,
    PieLoaderTensorDeclSlice, PieLoaderTensorDeclView, PieLoaderTileMapKind, PieLoaderU32Slice,
};
use crate::storage::{
    DestExtent, SourceExtent, StorageInstr, StorageProgram, StridedExtent, TileMapKind,
};
use crate::types::{BufferId, DType, Encoding, QuantScheme};

#[derive(Default)]
pub struct FfiArena {
    string_bytes: Vec<Box<[u8]>>,
    shapes: Vec<Box<[i64]>>,
    dim_specs: Vec<Box<[PieLoaderDimSpecView]>>,
    buffer_id_slices: Vec<Box<[u32]>>,
    tensor_views: Vec<PieLoaderTensorDeclView>,
    buffer_views: Vec<PieLoaderBufferDeclView>,
    instr_views: Vec<PieLoaderStorageInstrView>,
    schedule: Vec<u32>,
}

impl FfiArena {
    pub fn from_program(program: &StorageProgram) -> Self {
        let mut arena = Self::default();

        arena.tensor_views.reserve(program.tensors.len());
        for tensor in &program.tensors {
            let name = arena.push_string(&tensor.name);
            let shape = arena.push_i64_slice(&tensor.shape);
            let (encoding_kind, dtype, quant_scheme) = encoding_parts(&tensor.encoding);
            arena.tensor_views.push(PieLoaderTensorDeclView {
                id: tensor.id.0,
                name,
                dtype,
                encoding_kind,
                quant_scheme,
                shape,
                alignment: tensor.alignment,
            });
        }

        arena.buffer_views.reserve(program.buffers.len());
        for buffer in &program.buffers {
            arena.buffer_views.push(PieLoaderBufferDeclView {
                id: buffer.id.0,
                tensor_id: buffer.tensor.map(|id| id.0).unwrap_or(u32::MAX),
                has_tensor: buffer.tensor.is_some(),
                bytes: buffer.bytes,
                alignment: buffer.alignment,
                temporary: buffer.temporary,
            });
        }

        arena.instr_views.reserve(program.instrs.len());
        for instr in &program.instrs {
            let view = arena.instr_view(instr);
            arena.instr_views.push(view);
        }

        arena
            .schedule
            .extend(program.schedule.iter().map(|id| id.0));
        arena
    }

    pub fn view(&self, program: &StorageProgram) -> PieLoaderStorageProgramView {
        PieLoaderStorageProgramView {
            version: program.version,
            tensors: PieLoaderTensorDeclSlice {
                ptr: self.tensor_views.as_ptr(),
                len: self.tensor_views.len(),
            },
            buffers: PieLoaderBufferDeclSlice {
                ptr: self.buffer_views.as_ptr(),
                len: self.buffer_views.len(),
            },
            instrs: PieLoaderStorageInstrSlice {
                ptr: self.instr_views.as_ptr(),
                len: self.instr_views.len(),
            },
            schedule: PieLoaderU32Slice {
                ptr: self.schedule.as_ptr(),
                len: self.schedule.len(),
            },
            memory: PieLoaderMemoryPlanView {
                persistent_bytes: program.memory.persistent_bytes,
                temporary_peak_bytes: program.memory.temporary_peak_bytes,
                transform_scratch_peak_bytes: program.memory.transform_scratch_peak_bytes,
                checkpoint_read_bytes: program.memory.checkpoint_read_bytes,
                device_write_bytes: program.memory.device_write_bytes,
            },
        }
    }

    fn push_string(&mut self, value: &str) -> PieLoaderBytes {
        let bytes: Box<[u8]> = value.as_bytes().to_vec().into_boxed_slice();
        let ptr = bytes.as_ptr();
        let len = bytes.len();
        self.string_bytes.push(bytes);
        PieLoaderBytes { ptr, len }
    }

    fn push_i64_slice(&mut self, values: &[i64]) -> PieLoaderI64Slice {
        let shape: Box<[i64]> = values.to_vec().into_boxed_slice();
        let ptr = shape.as_ptr();
        let len = shape.len();
        self.shapes.push(shape);
        PieLoaderI64Slice { ptr, len }
    }

    fn push_dim_specs(&mut self, extent: &StridedExtent) -> PieLoaderDimSpecSlice {
        let dims: Box<[PieLoaderDimSpecView]> = extent
            .dims
            .iter()
            .map(|dim| PieLoaderDimSpecView {
                count: dim.count,
                src_stride: dim.src_stride,
                dst_stride: dim.dst_stride,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let ptr = dims.as_ptr();
        let len = dims.len();
        self.dim_specs.push(dims);
        PieLoaderDimSpecSlice { ptr, len }
    }

    fn push_buffer_ids(&mut self, values: &[BufferId]) -> PieLoaderBufferIdSlice {
        let ids: Box<[u32]> = values.iter().map(|id| id.0).collect::<Vec<_>>().into();
        let ptr = ids.as_ptr();
        let len = ids.len();
        self.buffer_id_slices.push(ids);
        PieLoaderBufferIdSlice { ptr, len }
    }

    fn instr_view(&mut self, instr: &StorageInstr) -> PieLoaderStorageInstrView {
        let empty_name = PieLoaderBytes::default();
        let empty_buffers = PieLoaderBufferIdSlice::default();
        let empty_source = PieLoaderSourceExtentView::default();
        let empty_dest = PieLoaderDestExtentView::default();
        match instr {
            StorageInstr::Allocate { id, buffer } => PieLoaderStorageInstrView {
                id: id.0,
                kind: PieLoaderStorageInstrKind::Allocate,
                buffer_id: buffer.0,
                source: empty_source,
                has_source: false,
                dest: empty_dest,
                has_dest: false,
                input_buffers: empty_buffers,
                output_buffers: empty_buffers,
                tile_kind: PieLoaderTileMapKind::None,
                transform_from: PieLoaderQuantScheme::None,
                transform_to: PieLoaderQuantScheme::None,
                name: empty_name,
            },
            StorageInstr::ExtentWrite { id, source, dest } => PieLoaderStorageInstrView {
                id: id.0,
                kind: PieLoaderStorageInstrKind::ExtentWrite,
                buffer_id: dest.buffer.0,
                source: self.source_view(source),
                has_source: true,
                dest: self.dest_view(dest),
                has_dest: true,
                input_buffers: empty_buffers,
                output_buffers: empty_buffers,
                tile_kind: PieLoaderTileMapKind::None,
                transform_from: PieLoaderQuantScheme::None,
                transform_to: PieLoaderQuantScheme::None,
                name: empty_name,
            },
            StorageInstr::TileMap {
                id,
                kind,
                source,
                dest,
                inputs,
                outputs,
                transform,
                ..
            } => PieLoaderStorageInstrView {
                id: id.0,
                kind: PieLoaderStorageInstrKind::TileMap,
                buffer_id: outputs.first().map(|id| id.0).unwrap_or(u32::MAX),
                source: source
                    .as_ref()
                    .map(|source| self.source_view(source))
                    .unwrap_or_default(),
                has_source: source.is_some(),
                dest: dest
                    .as_ref()
                    .map(|dest| self.dest_view(dest))
                    .unwrap_or_default(),
                has_dest: dest.is_some(),
                input_buffers: self.push_buffer_ids(inputs),
                output_buffers: self.push_buffer_ids(outputs),
                tile_kind: ffi_tile_kind(*kind),
                transform_from: transform
                    .from
                    .map(ffi_quant_scheme)
                    .unwrap_or(PieLoaderQuantScheme::None),
                transform_to: transform
                    .to
                    .map(ffi_quant_scheme)
                    .unwrap_or(PieLoaderQuantScheme::None),
                name: empty_name,
            },
            StorageInstr::CreateView {
                id,
                input,
                output,
                view,
                ..
            } => PieLoaderStorageInstrView {
                id: id.0,
                kind: PieLoaderStorageInstrKind::CreateView,
                buffer_id: output.0,
                source: empty_source,
                has_source: false,
                dest: self.dest_view(view),
                has_dest: true,
                input_buffers: self.push_buffer_ids(&[*input]),
                output_buffers: self.push_buffer_ids(&[*output]),
                tile_kind: PieLoaderTileMapKind::None,
                transform_from: PieLoaderQuantScheme::None,
                transform_to: PieLoaderQuantScheme::None,
                name: empty_name,
            },
            StorageInstr::Attach {
                id,
                tensor,
                metadata,
                ..
            } => PieLoaderStorageInstrView {
                id: id.0,
                kind: PieLoaderStorageInstrKind::Attach,
                buffer_id: tensor.0,
                source: empty_source,
                has_source: false,
                dest: empty_dest,
                has_dest: false,
                input_buffers: self.push_buffer_ids(metadata),
                output_buffers: self.push_buffer_ids(&[*tensor]),
                tile_kind: PieLoaderTileMapKind::None,
                transform_from: PieLoaderQuantScheme::None,
                transform_to: PieLoaderQuantScheme::None,
                name: empty_name,
            },
            StorageInstr::Release { id, buffer } => PieLoaderStorageInstrView {
                id: id.0,
                kind: PieLoaderStorageInstrKind::Release,
                buffer_id: buffer.0,
                source: empty_source,
                has_source: false,
                dest: empty_dest,
                has_dest: false,
                input_buffers: empty_buffers,
                output_buffers: empty_buffers,
                tile_kind: PieLoaderTileMapKind::None,
                transform_from: PieLoaderQuantScheme::None,
                transform_to: PieLoaderQuantScheme::None,
                name: empty_name,
            },
            StorageInstr::Finalize { id, tensor, name } => PieLoaderStorageInstrView {
                id: id.0,
                kind: PieLoaderStorageInstrKind::Finalize,
                buffer_id: tensor.0,
                source: empty_source,
                has_source: false,
                dest: empty_dest,
                has_dest: false,
                input_buffers: empty_buffers,
                output_buffers: self.push_buffer_ids(&[*tensor]),
                tile_kind: PieLoaderTileMapKind::None,
                transform_from: PieLoaderQuantScheme::None,
                transform_to: PieLoaderQuantScheme::None,
                name: self.push_string(name),
            },
        }
    }

    fn source_view(&mut self, source: &SourceExtent) -> PieLoaderSourceExtentView {
        PieLoaderSourceExtentView {
            file_id: source.file_id.0,
            tensor_id: source.tensor_id.0,
            file_offset: source.file_offset,
            span_bytes: source.span_bytes,
            stride: self.strided_view(&source.stride),
        }
    }

    fn dest_view(&mut self, dest: &DestExtent) -> PieLoaderDestExtentView {
        PieLoaderDestExtentView {
            buffer_id: dest.buffer.0,
            offset: dest.offset,
            stride: self.strided_view(&dest.stride),
        }
    }

    fn strided_view(&mut self, extent: &StridedExtent) -> PieLoaderStridedExtentView {
        PieLoaderStridedExtentView {
            base_offset: extent.base_offset,
            element_bytes: extent.element_bytes,
            dims: self.push_dim_specs(extent),
        }
    }
}

fn encoding_parts(
    encoding: &Encoding,
) -> (PieLoaderEncodingKind, PieLoaderDType, PieLoaderQuantScheme) {
    match encoding {
        Encoding::Raw(dtype) => (
            PieLoaderEncodingKind::Raw,
            ffi_dtype(*dtype),
            PieLoaderQuantScheme::None,
        ),
        Encoding::Quant(spec) => (
            PieLoaderEncodingKind::Quant,
            ffi_dtype(spec.logical_dtype),
            ffi_quant_scheme(spec.scheme),
        ),
    }
}

fn ffi_dtype(dtype: DType) -> PieLoaderDType {
    match dtype {
        DType::F32 => PieLoaderDType::F32,
        DType::F16 => PieLoaderDType::F16,
        DType::BF16 => PieLoaderDType::BF16,
        DType::F8E4M3 => PieLoaderDType::F8E4M3,
        DType::F8E5M2 => PieLoaderDType::F8E5M2,
        DType::I32 => PieLoaderDType::I32,
        DType::I16 => PieLoaderDType::I16,
        DType::I8 => PieLoaderDType::I8,
        DType::U32 => PieLoaderDType::U32,
        DType::U16 => PieLoaderDType::U16,
        DType::U8 => PieLoaderDType::U8,
        DType::Bool => PieLoaderDType::Bool,
    }
}

fn ffi_quant_scheme(scheme: QuantScheme) -> PieLoaderQuantScheme {
    match scheme {
        QuantScheme::None => PieLoaderQuantScheme::None,
        QuantScheme::Fp8E4M3 => PieLoaderQuantScheme::Fp8E4M3,
        QuantScheme::Fp8E5M2 => PieLoaderQuantScheme::Fp8E5M2,
        QuantScheme::Int8Symmetric => PieLoaderQuantScheme::Int8Symmetric,
        QuantScheme::Int8Asymmetric => PieLoaderQuantScheme::Int8Asymmetric,
        QuantScheme::AwqInt4 => PieLoaderQuantScheme::AwqInt4,
        QuantScheme::GptqInt4 => PieLoaderQuantScheme::GptqInt4,
        QuantScheme::Mxfp4E2M1E8M0 => PieLoaderQuantScheme::Mxfp4E2M1E8M0,
        QuantScheme::GgufQ4_0 => PieLoaderQuantScheme::GgufQ4_0,
        QuantScheme::GgufQ4K => PieLoaderQuantScheme::GgufQ4K,
        QuantScheme::GgufQ5_0 => PieLoaderQuantScheme::GgufQ5_0,
        QuantScheme::GgufQ5K => PieLoaderQuantScheme::GgufQ5K,
        QuantScheme::GgufQ8_0 => PieLoaderQuantScheme::GgufQ8_0,
    }
}

fn ffi_tile_kind(kind: TileMapKind) -> PieLoaderTileMapKind {
    match kind {
        TileMapKind::Cast => PieLoaderTileMapKind::Cast,
        TileMapKind::Decode => PieLoaderTileMapKind::Decode,
        TileMapKind::Encode => PieLoaderTileMapKind::Encode,
        TileMapKind::Transcode => PieLoaderTileMapKind::Transcode,
        TileMapKind::Reblock => PieLoaderTileMapKind::Reblock,
        TileMapKind::Reorder => PieLoaderTileMapKind::Reorder,
    }
}
