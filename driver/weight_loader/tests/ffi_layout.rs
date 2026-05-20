use std::ffi::CStr;
use std::ptr;

use pie_weight_loader::{
    PieLoaderBackendTargetView, PieLoaderBytes, PieLoaderCheckpointFileSlice,
    PieLoaderCheckpointFileView, PieLoaderCheckpointFormat, PieLoaderCheckpointTensorSlice,
    PieLoaderCheckpointTensorView, PieLoaderCompileInput, PieLoaderDType, PieLoaderEncodingKind,
    PieLoaderError, PieLoaderI64Slice, PieLoaderProgramHandle, PieLoaderQuantScheme,
    PieLoaderRuntimeAbiView, PieLoaderRuntimeByteSpanSlice, PieLoaderRuntimeByteSpanView,
    PieLoaderRuntimeSourceKind, PieLoaderRuntimeTensorContractSlice,
    PieLoaderRuntimeTensorContractView, PieLoaderSemanticRole, PieLoaderStatus,
    PieLoaderStorageInstrKind, PieLoaderTileMapKind, PieLoaderU32Slice, pie_loader_compile,
    pie_loader_error_free, pie_loader_program_free, pie_loader_program_view,
};

#[test]
fn empty_compile_returns_empty_program() {
    let input = PieLoaderCompileInput {
        target: PieLoaderBackendTargetView {
            tp_size: 1,
            ..PieLoaderBackendTargetView::default()
        },
        ..PieLoaderCompileInput::default()
    };
    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    assert!(!handle.is_null());
    assert!(error.message.is_null());

    let view = unsafe { pie_loader_program_view(handle) };
    assert_eq!(view.version, 1);
    assert_eq!(view.tensors.len, 0);
    assert_eq!(view.buffers.len, 0);
    assert_eq!(view.instrs.len, 0);
    assert_eq!(view.schedule.len, 0);

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

#[test]
fn null_input_reports_error() {
    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(ptr::null(), &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::NullArgument);
    assert!(handle.is_null());
    assert!(!error.message.is_null());
    let message = unsafe { CStr::from_ptr(error.message) }
        .to_string_lossy()
        .into_owned();
    assert!(message.contains("input"));
    unsafe { pie_loader_error_free(&mut error) };
    assert!(error.message.is_null());
}

#[test]
fn invalid_tp_rank_reports_structured_status() {
    let input = PieLoaderCompileInput {
        target: PieLoaderBackendTargetView {
            tp_rank: 2,
            tp_size: 2,
            ..PieLoaderBackendTargetView::default()
        },
        ..PieLoaderCompileInput::default()
    };
    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::InvalidInput);
    assert!(handle.is_null());
    unsafe { pie_loader_error_free(&mut error) };
}

#[test]
fn dense_contract_lowers_to_storage_program() {
    let file_path = bytes("model.safetensors");
    let tensor_name = bytes("model.embed_tokens.weight");
    let output_name = bytes("runtime.embed_tokens.weight");
    let abi_name = bytes("pie-cuda-test");
    let shape_values = [2_i64, 3_i64];
    let shape = PieLoaderI64Slice {
        ptr: shape_values.as_ptr(),
        len: shape_values.len(),
    };
    let files = [PieLoaderCheckpointFileView {
        id: 0,
        path: file_path,
        size_bytes: 12,
        format: PieLoaderCheckpointFormat::Safetensors,
    }];
    let tensors = [PieLoaderCheckpointTensorView {
        id: 7,
        name: tensor_name,
        file_id: 0,
        file_offset: 128,
        span_bytes: 12,
        dtype: PieLoaderDType::BF16,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape,
        ..PieLoaderCheckpointTensorView::default()
    }];
    let contracts = [PieLoaderRuntimeTensorContractView {
        output_name,
        source_kind: PieLoaderRuntimeSourceKind::DirectTensor,
        source_tensor_id: 7,
        source_tensor_ids: PieLoaderU32Slice::default(),
        byte_spans: Default::default(),
        metadata_tensor_ids: PieLoaderU32Slice::default(),
        source_contract_id: u32::MAX,
        semantic_role: PieLoaderSemanticRole::DirectTensor,
        layer: 0,
        has_layer: false,
        expert: 0,
        has_expert: false,
        axis: -1,
        start: 0,
        length: 0,
        dtype: PieLoaderDType::BF16,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape,
        alignment: 256,
        shard_axis: -1,
        ..PieLoaderRuntimeTensorContractView::default()
    }];
    let input = PieLoaderCompileInput {
        files: PieLoaderCheckpointFileSlice {
            ptr: files.as_ptr(),
            len: files.len(),
        },
        tensors: PieLoaderCheckpointTensorSlice {
            ptr: tensors.as_ptr(),
            len: tensors.len(),
        },
        runtime_abi: PieLoaderRuntimeAbiView {
            name: abi_name,
            version: 1,
            tensors: PieLoaderRuntimeTensorContractSlice {
                ptr: contracts.as_ptr(),
                len: contracts.len(),
            },
        },
        target: PieLoaderBackendTargetView {
            tp_size: 1,
            ..PieLoaderBackendTargetView::default()
        },
        ..PieLoaderCompileInput::default()
    };

    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    assert!(!handle.is_null());
    let view = unsafe { pie_loader_program_view(handle) };
    assert_eq!(view.tensors.len, 1);
    assert_eq!(view.buffers.len, 1);
    assert_eq!(view.instrs.len, 3);
    assert_eq!(view.schedule.len, 3);
    assert_eq!(view.memory.persistent_bytes, 12);
    assert_eq!(view.memory.checkpoint_read_bytes, 12);
    assert_eq!(view.memory.device_write_bytes, 12);
    let instrs = unsafe { std::slice::from_raw_parts(view.instrs.ptr, view.instrs.len) };
    assert_eq!(instrs[0].kind, PieLoaderStorageInstrKind::Allocate);
    assert_eq!(instrs[1].kind, PieLoaderStorageInstrKind::ExtentWrite);
    assert!(instrs[1].has_source);
    assert!(instrs[1].has_dest);
    assert_eq!(instrs[1].source.file_id, 0);
    assert_eq!(instrs[1].source.tensor_id, 7);
    assert_eq!(instrs[1].source.file_offset, 128);
    assert_eq!(instrs[1].source.span_bytes, 12);
    assert_eq!(instrs[1].dest.buffer_id, 0);
    assert_eq!(instrs[2].kind, PieLoaderStorageInstrKind::Finalize);

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

#[test]
fn cast_contract_lowers_to_source_tile_map() {
    let file_path = bytes("model.safetensors");
    let tensor_name = bytes("model.layers.0.mlp.up_proj.weight");
    let output_name = bytes("runtime.up.weight");
    let shape_values = [2_i64];
    let shape = PieLoaderI64Slice {
        ptr: shape_values.as_ptr(),
        len: shape_values.len(),
    };
    let files = [PieLoaderCheckpointFileView {
        id: 0,
        path: file_path,
        size_bytes: 8,
        format: PieLoaderCheckpointFormat::Safetensors,
    }];
    let tensors = [PieLoaderCheckpointTensorView {
        id: 0,
        name: tensor_name,
        file_id: 0,
        file_offset: 64,
        span_bytes: 8,
        dtype: PieLoaderDType::F32,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape,
        ..PieLoaderCheckpointTensorView::default()
    }];
    let contracts = [PieLoaderRuntimeTensorContractView {
        output_name,
        source_kind: PieLoaderRuntimeSourceKind::DirectTensor,
        source_tensor_id: 0,
        source_tensor_ids: PieLoaderU32Slice::default(),
        byte_spans: Default::default(),
        metadata_tensor_ids: PieLoaderU32Slice::default(),
        source_contract_id: u32::MAX,
        semantic_role: PieLoaderSemanticRole::DirectTensor,
        layer: 0,
        has_layer: false,
        expert: 0,
        has_expert: false,
        axis: -1,
        start: 0,
        length: 0,
        dtype: PieLoaderDType::BF16,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape,
        alignment: 1,
        shard_axis: -1,
        ..PieLoaderRuntimeTensorContractView::default()
    }];
    let input = compile_input(&files, &tensors, &contracts);

    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    let view = unsafe { pie_loader_program_view(handle) };
    let instrs = unsafe { std::slice::from_raw_parts(view.instrs.ptr, view.instrs.len) };
    assert_eq!(instrs.len(), 3);
    assert_eq!(instrs[1].kind, PieLoaderStorageInstrKind::TileMap);
    assert_eq!(instrs[1].tile_kind, PieLoaderTileMapKind::Cast);
    assert!(instrs[1].has_source);
    assert!(instrs[1].has_dest);
    assert_eq!(instrs[1].source.file_offset, 64);
    assert_eq!(instrs[1].source.span_bytes, 8);
    assert_eq!(instrs[1].dest.offset, 0);
    assert_eq!(view.memory.checkpoint_read_bytes, 8);
    assert_eq!(view.memory.device_write_bytes, 4);

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

#[test]
fn fp8_quant_contract_lowers_to_decode_with_metadata() {
    let file_path = bytes("model.safetensors");
    let weight_name = bytes("linear.weight");
    let scale_name = bytes("linear.weight_scale_inv");
    let output_name = bytes("runtime.linear.weight");
    let weight_shape_values = [2_i64, 4_i64];
    let scale_shape_values: [i64; 0] = [];
    let weight_shape = PieLoaderI64Slice {
        ptr: weight_shape_values.as_ptr(),
        len: weight_shape_values.len(),
    };
    let scale_shape = PieLoaderI64Slice {
        ptr: scale_shape_values.as_ptr(),
        len: scale_shape_values.len(),
    };
    let files = [PieLoaderCheckpointFileView {
        id: 0,
        path: file_path,
        size_bytes: 72,
        format: PieLoaderCheckpointFormat::Safetensors,
    }];
    let tensors = [
        PieLoaderCheckpointTensorView {
            id: 0,
            name: weight_name,
            file_id: 0,
            file_offset: 64,
            span_bytes: 8,
            dtype: PieLoaderDType::BF16,
            encoding_kind: PieLoaderEncodingKind::Quant,
            quant_scheme: PieLoaderQuantScheme::Fp8E4M3,
            shape: weight_shape,
            ..PieLoaderCheckpointTensorView::default()
        },
        PieLoaderCheckpointTensorView {
            id: 1,
            name: scale_name,
            file_id: 0,
            file_offset: 72,
            span_bytes: 2,
            dtype: PieLoaderDType::BF16,
            encoding_kind: PieLoaderEncodingKind::Raw,
            quant_scheme: PieLoaderQuantScheme::None,
            shape: scale_shape,
            ..PieLoaderCheckpointTensorView::default()
        },
    ];
    let metadata_ids = [1_u32];
    let contracts = [PieLoaderRuntimeTensorContractView {
        output_name,
        source_kind: PieLoaderRuntimeSourceKind::DirectTensor,
        source_tensor_id: 0,
        source_tensor_ids: PieLoaderU32Slice::default(),
        byte_spans: Default::default(),
        metadata_tensor_ids: PieLoaderU32Slice {
            ptr: metadata_ids.as_ptr(),
            len: metadata_ids.len(),
        },
        source_contract_id: u32::MAX,
        semantic_role: PieLoaderSemanticRole::DirectTensor,
        layer: 0,
        has_layer: false,
        expert: 0,
        has_expert: false,
        axis: -1,
        start: 0,
        length: 0,
        dtype: PieLoaderDType::BF16,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape: weight_shape,
        alignment: 1,
        shard_axis: -1,
        ..PieLoaderRuntimeTensorContractView::default()
    }];
    let input = compile_input(&files, &tensors, &contracts);

    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    let view = unsafe { pie_loader_program_view(handle) };
    let instrs = unsafe { std::slice::from_raw_parts(view.instrs.ptr, view.instrs.len) };
    assert_eq!(instrs.len(), 5);
    assert_eq!(instrs[1].kind, PieLoaderStorageInstrKind::Allocate);
    assert_eq!(instrs[2].kind, PieLoaderStorageInstrKind::ExtentWrite);
    assert_eq!(instrs[3].kind, PieLoaderStorageInstrKind::TileMap);
    assert_eq!(instrs[3].tile_kind, PieLoaderTileMapKind::Decode);
    assert_eq!(instrs[3].transform_from, PieLoaderQuantScheme::Fp8E4M3);
    assert_eq!(instrs[3].input_buffers.len, 1);
    assert!(instrs[3].has_source);
    assert!(instrs[3].has_dest);
    assert_eq!(instrs[3].source.file_offset, 64);
    assert_eq!(instrs[3].source.span_bytes, 8);
    assert_eq!(instrs[3].dest.offset, 0);
    assert_eq!(view.memory.checkpoint_read_bytes, 10);
    assert_eq!(view.memory.device_write_bytes, 18);

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

#[test]
fn semantic_role_contract_resolves_source_tensor() {
    let file_path = bytes("model.safetensors");
    let tensor_name = bytes("model.embed_tokens.weight");
    let output_name = bytes("runtime.embed_tokens.weight");
    let model_type = bytes("qwen3");
    let shape_values = [1_i64, 2_i64];
    let shape = PieLoaderI64Slice {
        ptr: shape_values.as_ptr(),
        len: shape_values.len(),
    };
    let files = [PieLoaderCheckpointFileView {
        id: 0,
        path: file_path,
        size_bytes: 4,
        format: PieLoaderCheckpointFormat::Safetensors,
    }];
    let tensors = [PieLoaderCheckpointTensorView {
        id: 42,
        name: tensor_name,
        file_id: 0,
        file_offset: 256,
        span_bytes: 4,
        dtype: PieLoaderDType::BF16,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape,
        ..PieLoaderCheckpointTensorView::default()
    }];
    let contracts = [PieLoaderRuntimeTensorContractView {
        output_name,
        source_kind: PieLoaderRuntimeSourceKind::Semantic,
        source_tensor_id: u32::MAX,
        source_tensor_ids: PieLoaderU32Slice::default(),
        byte_spans: Default::default(),
        metadata_tensor_ids: PieLoaderU32Slice::default(),
        source_contract_id: u32::MAX,
        semantic_role: PieLoaderSemanticRole::TokenEmbedding,
        layer: 0,
        has_layer: false,
        expert: 0,
        has_expert: false,
        axis: -1,
        start: 0,
        length: 0,
        dtype: PieLoaderDType::BF16,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape,
        alignment: 1,
        shard_axis: -1,
        ..PieLoaderRuntimeTensorContractView::default()
    }];
    let mut input = compile_input(&files, &tensors, &contracts);
    input.model.model_type = model_type;

    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    let view = unsafe { pie_loader_program_view(handle) };
    let instrs = unsafe { std::slice::from_raw_parts(view.instrs.ptr, view.instrs.len) };
    assert_eq!(instrs[1].source.tensor_id, 42);
    assert_eq!(instrs[1].source.file_offset, 256);

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

#[test]
fn byte_span_contract_lowers_to_explicit_extent_writes() {
    let file_path = bytes("model.safetensors");
    let a_name = bytes("a.weight");
    let b_name = bytes("b.weight");
    let output_name = bytes("runtime.assembled");
    let a_shape_values = [16_i64];
    let b_shape_values = [8_i64];
    let output_shape_values = [24_i64];
    let a_shape = PieLoaderI64Slice {
        ptr: a_shape_values.as_ptr(),
        len: a_shape_values.len(),
    };
    let b_shape = PieLoaderI64Slice {
        ptr: b_shape_values.as_ptr(),
        len: b_shape_values.len(),
    };
    let output_shape = PieLoaderI64Slice {
        ptr: output_shape_values.as_ptr(),
        len: output_shape_values.len(),
    };
    let files = [PieLoaderCheckpointFileView {
        id: 0,
        path: file_path,
        size_bytes: 160,
        format: PieLoaderCheckpointFormat::Safetensors,
    }];
    let tensors = [
        PieLoaderCheckpointTensorView {
            id: 0,
            name: a_name,
            file_id: 0,
            file_offset: 64,
            span_bytes: 16,
            dtype: PieLoaderDType::U8,
            encoding_kind: PieLoaderEncodingKind::Raw,
            quant_scheme: PieLoaderQuantScheme::None,
            shape: a_shape,
            ..PieLoaderCheckpointTensorView::default()
        },
        PieLoaderCheckpointTensorView {
            id: 1,
            name: b_name,
            file_id: 0,
            file_offset: 128,
            span_bytes: 8,
            dtype: PieLoaderDType::U8,
            encoding_kind: PieLoaderEncodingKind::Raw,
            quant_scheme: PieLoaderQuantScheme::None,
            shape: b_shape,
            ..PieLoaderCheckpointTensorView::default()
        },
    ];
    let spans = [
        PieLoaderRuntimeByteSpanView {
            source_tensor_id: 0,
            source_offset_bytes: 4,
            dest_offset_bytes: 0,
            span_bytes: 12,
        },
        PieLoaderRuntimeByteSpanView {
            source_tensor_id: 1,
            source_offset_bytes: 0,
            dest_offset_bytes: 12,
            span_bytes: 8,
        },
        PieLoaderRuntimeByteSpanView {
            source_tensor_id: 0,
            source_offset_bytes: 0,
            dest_offset_bytes: 20,
            span_bytes: 4,
        },
    ];
    let contracts = [PieLoaderRuntimeTensorContractView {
        output_name,
        source_kind: PieLoaderRuntimeSourceKind::ByteSpans,
        byte_spans: PieLoaderRuntimeByteSpanSlice {
            ptr: spans.as_ptr(),
            len: spans.len(),
        },
        dtype: PieLoaderDType::U8,
        shape: output_shape,
        alignment: 1,
        ..PieLoaderRuntimeTensorContractView::default()
    }];
    let input = compile_input(&files, &tensors, &contracts);

    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    let view = unsafe { pie_loader_program_view(handle) };
    let instrs = unsafe { std::slice::from_raw_parts(view.instrs.ptr, view.instrs.len) };
    assert_eq!(view.tensors.len, 1);
    assert_eq!(view.buffers.len, 1);
    assert_eq!(instrs.len(), 5);
    assert_eq!(instrs[1].kind, PieLoaderStorageInstrKind::ExtentWrite);
    assert_eq!(instrs[1].source.file_offset, 68);
    assert_eq!(instrs[1].dest.offset, 0);
    assert_eq!(instrs[2].source.file_offset, 128);
    assert_eq!(instrs[2].dest.offset, 12);
    assert_eq!(instrs[3].source.file_offset, 64);
    assert_eq!(instrs[3].dest.offset, 20);
    assert_eq!(view.memory.persistent_bytes, 24);
    assert_eq!(view.memory.checkpoint_read_bytes, 24);
    assert_eq!(view.memory.device_write_bytes, 24);

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

#[test]
fn join_and_select_contracts_lower_to_writes_and_view() {
    let file_path = bytes("model.safetensors");
    let a_name = bytes("a.weight");
    let b_name = bytes("b.weight");
    let joined_name = bytes("joined.weight");
    let selected_name = bytes("selected.weight");
    let source_shape_values = [2_i64, 2_i64];
    let joined_shape_values = [4_i64, 2_i64];
    let selected_shape_values = [2_i64, 2_i64];
    let source_shape = PieLoaderI64Slice {
        ptr: source_shape_values.as_ptr(),
        len: source_shape_values.len(),
    };
    let joined_shape = PieLoaderI64Slice {
        ptr: joined_shape_values.as_ptr(),
        len: joined_shape_values.len(),
    };
    let selected_shape = PieLoaderI64Slice {
        ptr: selected_shape_values.as_ptr(),
        len: selected_shape_values.len(),
    };
    let source_ids = [0_u32, 1_u32];
    let files = [PieLoaderCheckpointFileView {
        id: 0,
        path: file_path,
        size_bytes: 256,
        format: PieLoaderCheckpointFormat::Safetensors,
    }];
    let tensors = [
        PieLoaderCheckpointTensorView {
            id: 0,
            name: a_name,
            file_id: 0,
            file_offset: 64,
            span_bytes: 16,
            dtype: PieLoaderDType::F32,
            encoding_kind: PieLoaderEncodingKind::Raw,
            quant_scheme: PieLoaderQuantScheme::None,
            shape: source_shape,
            ..PieLoaderCheckpointTensorView::default()
        },
        PieLoaderCheckpointTensorView {
            id: 1,
            name: b_name,
            file_id: 0,
            file_offset: 80,
            span_bytes: 16,
            dtype: PieLoaderDType::F32,
            encoding_kind: PieLoaderEncodingKind::Raw,
            quant_scheme: PieLoaderQuantScheme::None,
            shape: source_shape,
            ..PieLoaderCheckpointTensorView::default()
        },
    ];
    let contracts = [
        PieLoaderRuntimeTensorContractView {
            output_name: joined_name,
            source_kind: PieLoaderRuntimeSourceKind::Join,
            source_tensor_id: u32::MAX,
            source_tensor_ids: PieLoaderU32Slice {
                ptr: source_ids.as_ptr(),
                len: source_ids.len(),
            },
            source_contract_id: u32::MAX,
            semantic_role: PieLoaderSemanticRole::DirectTensor,
            axis: 0,
            dtype: PieLoaderDType::F32,
            shape: joined_shape,
            alignment: 1,
            ..PieLoaderRuntimeTensorContractView::default()
        },
        PieLoaderRuntimeTensorContractView {
            output_name: selected_name,
            source_kind: PieLoaderRuntimeSourceKind::Select,
            source_tensor_id: u32::MAX,
            source_contract_id: 0,
            semantic_role: PieLoaderSemanticRole::DirectTensor,
            axis: 0,
            start: 1,
            length: 2,
            dtype: PieLoaderDType::F32,
            shape: selected_shape,
            alignment: 1,
            ..PieLoaderRuntimeTensorContractView::default()
        },
    ];
    let input = compile_input(&files, &tensors, &contracts);

    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    let view = unsafe { pie_loader_program_view(handle) };
    let instrs = unsafe { std::slice::from_raw_parts(view.instrs.ptr, view.instrs.len) };
    assert_eq!(view.tensors.len, 2);
    assert_eq!(view.buffers.len, 2);
    assert_eq!(instrs.len(), 6);
    assert_eq!(instrs[0].kind, PieLoaderStorageInstrKind::Allocate);
    assert_eq!(instrs[1].kind, PieLoaderStorageInstrKind::ExtentWrite);
    assert_eq!(instrs[1].source.tensor_id, 0);
    assert_eq!(instrs[1].source.file_offset, 64);
    assert_eq!(instrs[1].dest.offset, 0);
    assert_eq!(instrs[2].kind, PieLoaderStorageInstrKind::ExtentWrite);
    assert_eq!(instrs[2].source.tensor_id, 1);
    assert_eq!(instrs[2].source.file_offset, 80);
    assert_eq!(instrs[2].dest.offset, 16);
    assert_eq!(instrs[3].kind, PieLoaderStorageInstrKind::Finalize);
    assert_eq!(instrs[4].kind, PieLoaderStorageInstrKind::CreateView);
    assert!(instrs[4].has_dest);
    assert_eq!(instrs[4].dest.offset, 8);
    assert_eq!(instrs[5].kind, PieLoaderStorageInstrKind::Finalize);
    assert_eq!(view.memory.persistent_bytes, 32);
    assert_eq!(view.memory.checkpoint_read_bytes, 32);
    assert_eq!(view.memory.device_write_bytes, 32);

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

#[test]
fn empty_runtime_abi_builds_default_tp_row_contracts() {
    let file_path = bytes("model.safetensors");
    let tensor_name = bytes("model.layers.0.self_attn.q_proj.weight");
    let shape_values = [4_i64, 4_i64];
    let shape = PieLoaderI64Slice {
        ptr: shape_values.as_ptr(),
        len: shape_values.len(),
    };
    let files = [PieLoaderCheckpointFileView {
        id: 0,
        path: file_path,
        size_bytes: 256,
        format: PieLoaderCheckpointFormat::Safetensors,
    }];
    let tensors = [PieLoaderCheckpointTensorView {
        id: 0,
        name: tensor_name,
        file_id: 0,
        file_offset: 64,
        span_bytes: 32,
        dtype: PieLoaderDType::BF16,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape,
        ..PieLoaderCheckpointTensorView::default()
    }];
    let mut input = compile_input(&files, &tensors, &[]);
    input.target.tp_rank = 1;
    input.target.tp_size = 2;

    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    let view = unsafe { pie_loader_program_view(handle) };
    assert_eq!(view.tensors.len, 1);
    let tensors_view = unsafe { std::slice::from_raw_parts(view.tensors.ptr, view.tensors.len) };
    let final_shape =
        unsafe { std::slice::from_raw_parts(tensors_view[0].shape.ptr, tensors_view[0].shape.len) };
    assert_eq!(final_shape, &[2, 4]);
    let instrs = unsafe { std::slice::from_raw_parts(view.instrs.ptr, view.instrs.len) };
    assert_eq!(instrs[1].kind, PieLoaderStorageInstrKind::ExtentWrite);
    assert_eq!(instrs[1].source.file_offset, 80);
    assert_eq!(instrs[1].source.span_bytes, 16);

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

#[test]
fn empty_runtime_abi_builds_default_tp_column_contracts_as_strided_extents() {
    let file_path = bytes("model.safetensors");
    let tensor_name = bytes("model.layers.0.self_attn.o_proj.weight");
    let shape_values = [4_i64, 4_i64];
    let shape = PieLoaderI64Slice {
        ptr: shape_values.as_ptr(),
        len: shape_values.len(),
    };
    let files = [PieLoaderCheckpointFileView {
        id: 0,
        path: file_path,
        size_bytes: 256,
        format: PieLoaderCheckpointFormat::Safetensors,
    }];
    let tensors = [PieLoaderCheckpointTensorView {
        id: 0,
        name: tensor_name,
        file_id: 0,
        file_offset: 64,
        span_bytes: 32,
        dtype: PieLoaderDType::BF16,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape,
        ..PieLoaderCheckpointTensorView::default()
    }];
    let mut input = compile_input(&files, &tensors, &[]);
    input.target.tp_rank = 1;
    input.target.tp_size = 2;

    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    let view = unsafe { pie_loader_program_view(handle) };
    let tensors_view = unsafe { std::slice::from_raw_parts(view.tensors.ptr, view.tensors.len) };
    let final_shape =
        unsafe { std::slice::from_raw_parts(tensors_view[0].shape.ptr, tensors_view[0].shape.len) };
    assert_eq!(final_shape, &[4, 2]);
    let instrs = unsafe { std::slice::from_raw_parts(view.instrs.ptr, view.instrs.len) };
    assert_eq!(instrs[1].source.file_offset, 68);
    assert_eq!(instrs[1].source.span_bytes, 16);
    let dims = unsafe {
        std::slice::from_raw_parts(
            instrs[1].source.stride.dims.ptr,
            instrs[1].source.stride.dims.len,
        )
    };
    assert_eq!(dims[0].count, 4);
    assert_eq!(dims[0].src_stride, 8);
    assert_eq!(dims[0].dst_stride, 4);
    assert_eq!(dims[1].count, 2);
    assert_eq!(dims[1].src_stride, 2);
    assert_eq!(dims[1].dst_stride, 2);

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

#[test]
fn empty_runtime_abi_lowers_runtime_quant_to_encode_and_scale_output() {
    let file_path = bytes("model.safetensors");
    let tensor_name = bytes("model.layers.0.self_attn.q_proj.weight");
    let model_type = bytes("qwen3");
    let runtime_quant = bytes("int8");
    let shape_values = [4_i64, 4_i64];
    let shape = PieLoaderI64Slice {
        ptr: shape_values.as_ptr(),
        len: shape_values.len(),
    };
    let files = [PieLoaderCheckpointFileView {
        id: 0,
        path: file_path,
        size_bytes: 256,
        format: PieLoaderCheckpointFormat::Safetensors,
    }];
    let tensors = [PieLoaderCheckpointTensorView {
        id: 0,
        name: tensor_name,
        file_id: 0,
        file_offset: 64,
        span_bytes: 32,
        dtype: PieLoaderDType::BF16,
        encoding_kind: PieLoaderEncodingKind::Raw,
        quant_scheme: PieLoaderQuantScheme::None,
        shape,
        ..PieLoaderCheckpointTensorView::default()
    }];
    let mut input = compile_input(&files, &tensors, &[]);
    input.model.model_type = model_type;
    input.model.runtime_quant = runtime_quant;
    input.target.backend = pie_weight_loader::PieLoaderBackendKind::Cuda;
    input.target.preferred_alignment = 256;

    let mut handle: *mut PieLoaderProgramHandle = ptr::null_mut();
    let mut error = PieLoaderError::default();
    let status = unsafe { pie_loader_compile(&input, &mut handle, &mut error) };
    assert_eq!(status, PieLoaderStatus::Ok, "{}", error_message(&error));
    let view = unsafe { pie_loader_program_view(handle) };
    assert_eq!(view.tensors.len, 2);
    let tensors_view = unsafe { std::slice::from_raw_parts(view.tensors.ptr, view.tensors.len) };
    assert_eq!(tensors_view[0].encoding_kind, PieLoaderEncodingKind::Quant);
    assert_eq!(
        tensors_view[0].quant_scheme,
        PieLoaderQuantScheme::Int8Symmetric
    );
    assert_eq!(tensors_view[1].encoding_kind, PieLoaderEncodingKind::Raw);
    assert_eq!(tensors_view[1].dtype, PieLoaderDType::F32);
    let scale_shape =
        unsafe { std::slice::from_raw_parts(tensors_view[1].shape.ptr, tensors_view[1].shape.len) };
    assert_eq!(scale_shape, &[4]);
    let instrs = unsafe { std::slice::from_raw_parts(view.instrs.ptr, view.instrs.len) };
    assert_eq!(instrs[2].kind, PieLoaderStorageInstrKind::TileMap);
    assert_eq!(instrs[2].tile_kind, PieLoaderTileMapKind::Encode);
    assert_eq!(instrs[2].transform_to, PieLoaderQuantScheme::Int8Symmetric);
    assert_eq!(instrs[2].output_buffers.len, 2);
    assert_eq!(instrs[3].kind, PieLoaderStorageInstrKind::Finalize);
    assert_eq!(instrs[4].kind, PieLoaderStorageInstrKind::Finalize);
    assert!(
        !instrs
            .iter()
            .any(|instr| instr.kind == PieLoaderStorageInstrKind::CreateView)
    );

    unsafe {
        pie_loader_program_free(handle);
        pie_loader_error_free(&mut error);
    }
}

fn bytes(value: &'static str) -> PieLoaderBytes {
    PieLoaderBytes {
        ptr: value.as_ptr(),
        len: value.len(),
    }
}

fn compile_input(
    files: &[PieLoaderCheckpointFileView],
    tensors: &[PieLoaderCheckpointTensorView],
    contracts: &[PieLoaderRuntimeTensorContractView],
) -> PieLoaderCompileInput {
    PieLoaderCompileInput {
        files: PieLoaderCheckpointFileSlice {
            ptr: files.as_ptr(),
            len: files.len(),
        },
        tensors: PieLoaderCheckpointTensorSlice {
            ptr: tensors.as_ptr(),
            len: tensors.len(),
        },
        runtime_abi: PieLoaderRuntimeAbiView {
            name: bytes("pie-cuda-test"),
            version: 1,
            tensors: PieLoaderRuntimeTensorContractSlice {
                ptr: contracts.as_ptr(),
                len: contracts.len(),
            },
        },
        target: PieLoaderBackendTargetView {
            tp_size: 1,
            ..PieLoaderBackendTargetView::default()
        },
        ..PieLoaderCompileInput::default()
    }
}

fn error_message(error: &PieLoaderError) -> String {
    if error.message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(error.message) }
            .to_string_lossy()
            .into_owned()
    }
}
