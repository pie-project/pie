use std::ffi::c_char;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderStatus {
    Ok = 0,
    NullArgument = 1,
    InvalidInput = 2,
    InternalError = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderCheckpointFormat {
    Safetensors = 0,
    Gguf = 1,
    Unknown = 255,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderBackendKind {
    Cuda = 0,
    Portable = 1,
    Unknown = 255,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    F8E4M3 = 3,
    F8E5M2 = 4,
    I32 = 5,
    I16 = 6,
    I8 = 7,
    U32 = 8,
    U16 = 9,
    U8 = 10,
    Bool = 11,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderEncodingKind {
    Raw = 0,
    Quant = 1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderQuantScheme {
    None = 0,
    Fp8E4M3 = 1,
    Fp8E5M2 = 2,
    Int8Symmetric = 3,
    Int8Asymmetric = 4,
    AwqInt4 = 5,
    GptqInt4 = 6,
    Mxfp4E2M1E8M0 = 7,
    GgufQ4_0 = 8,
    GgufQ4K = 9,
    GgufQ5_0 = 10,
    GgufQ5K = 11,
    GgufQ8_0 = 12,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderMxfp4MoePolicy {
    RoutedDecode = 0,
    NativeGemm = 1,
    EagerBf16 = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderSemanticRole {
    DirectTensor = 0,
    TokenEmbedding = 1,
    OutputEmbedding = 2,
    AttentionQ = 3,
    AttentionK = 4,
    AttentionV = 5,
    AttentionO = 6,
    MlpGate = 7,
    MlpUp = 8,
    MlpDown = 9,
    ExpertRouter = 10,
    ExpertGate = 11,
    ExpertUp = 12,
    ExpertDown = 13,
    ExpertBias = 14,
    Norm = 15,
    QuantData = 16,
    QuantScale = 17,
    QuantZeroPoint = 18,
    QuantGroupIndex = 19,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderRuntimeSourceKind {
    DirectTensor = 0,
    Semantic = 1,
    Join = 2,
    Select = 3,
    ByteSpans = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderBytes {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderU32Slice {
    pub ptr: *const u32,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderRuntimeByteSpanView {
    pub source_tensor_id: u32,
    pub source_offset_bytes: u64,
    pub dest_offset_bytes: u64,
    pub span_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderRuntimeByteSpanSlice {
    pub ptr: *const PieLoaderRuntimeByteSpanView,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderStorageInstrKind {
    Allocate = 0,
    ExtentWrite = 1,
    TileMap = 2,
    CreateView = 3,
    Attach = 4,
    Release = 5,
    Finalize = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieLoaderTileMapKind {
    Cast = 0,
    Decode = 1,
    Encode = 2,
    Transcode = 3,
    Reblock = 4,
    Reorder = 5,
    None = 255,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderI64Slice {
    pub ptr: *const i64,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderDimSpecView {
    pub count: i64,
    pub src_stride: i64,
    pub dst_stride: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderDimSpecSlice {
    pub ptr: *const PieLoaderDimSpecView,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderBufferIdSlice {
    pub ptr: *const u32,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderStridedExtentView {
    pub base_offset: u64,
    pub element_bytes: u32,
    pub dims: PieLoaderDimSpecSlice,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderSourceExtentView {
    pub file_id: u32,
    pub tensor_id: u32,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub stride: PieLoaderStridedExtentView,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderDestExtentView {
    pub buffer_id: u32,
    pub offset: u64,
    pub stride: PieLoaderStridedExtentView,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieLoaderCheckpointFileView {
    pub id: u32,
    pub path: PieLoaderBytes,
    pub size_bytes: u64,
    pub format: PieLoaderCheckpointFormat,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderCheckpointFileSlice {
    pub ptr: *const PieLoaderCheckpointFileView,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieLoaderCheckpointTensorView {
    pub id: u32,
    pub name: PieLoaderBytes,
    pub file_id: u32,
    pub file_offset: u64,
    pub span_bytes: u64,
    pub dtype: PieLoaderDType,
    pub encoding_kind: PieLoaderEncodingKind,
    pub quant_scheme: PieLoaderQuantScheme,
    pub shape: PieLoaderI64Slice,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderCheckpointTensorSlice {
    pub ptr: *const PieLoaderCheckpointTensorView,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieLoaderRuntimeTensorContractView {
    pub output_name: PieLoaderBytes,
    pub source_kind: PieLoaderRuntimeSourceKind,
    pub source_tensor_id: u32,
    pub source_tensor_ids: PieLoaderU32Slice,
    pub byte_spans: PieLoaderRuntimeByteSpanSlice,
    pub metadata_tensor_ids: PieLoaderU32Slice,
    pub source_contract_id: u32,
    pub semantic_role: PieLoaderSemanticRole,
    pub layer: u32,
    pub has_layer: bool,
    pub expert: u32,
    pub has_expert: bool,
    pub axis: i32,
    pub start: i64,
    pub length: i64,
    pub dtype: PieLoaderDType,
    pub encoding_kind: PieLoaderEncodingKind,
    pub quant_scheme: PieLoaderQuantScheme,
    pub shape: PieLoaderI64Slice,
    pub alignment: u32,
    pub shard_axis: i32,
}

impl Default for PieLoaderRuntimeTensorContractView {
    fn default() -> Self {
        Self {
            output_name: PieLoaderBytes::default(),
            source_kind: PieLoaderRuntimeSourceKind::DirectTensor,
            source_tensor_id: u32::MAX,
            source_tensor_ids: PieLoaderU32Slice::default(),
            byte_spans: PieLoaderRuntimeByteSpanSlice::default(),
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
            shape: PieLoaderI64Slice::default(),
            alignment: 1,
            shard_axis: -1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderRuntimeTensorContractSlice {
    pub ptr: *const PieLoaderRuntimeTensorContractView,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderModelConfigView {
    pub model_type: PieLoaderBytes,
    pub quant_method: PieLoaderBytes,
    pub num_hidden_layers: u32,
    pub num_experts: u32,
    pub num_experts_per_tok: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderRuntimeAbiView {
    pub name: PieLoaderBytes,
    pub version: u32,
    pub tensors: PieLoaderRuntimeTensorContractSlice,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieLoaderBackendTargetView {
    pub backend: PieLoaderBackendKind,
    pub tp_rank: u32,
    pub tp_size: u32,
    pub max_tile_bytes: u64,
    pub preferred_alignment: u32,
    pub mxfp4_moe: PieLoaderMxfp4MoePolicy,
    pub native_mxfp4_moe: bool,
}

impl Default for PieLoaderBackendTargetView {
    fn default() -> Self {
        Self {
            backend: PieLoaderBackendKind::Unknown,
            tp_rank: 0,
            tp_size: 1,
            max_tile_bytes: 0,
            preferred_alignment: 0,
            mxfp4_moe: PieLoaderMxfp4MoePolicy::RoutedDecode,
            native_mxfp4_moe: false,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderCompileInput {
    pub version: u32,
    pub files: PieLoaderCheckpointFileSlice,
    pub tensors: PieLoaderCheckpointTensorSlice,
    pub model: PieLoaderModelConfigView,
    pub runtime_abi: PieLoaderRuntimeAbiView,
    pub target: PieLoaderBackendTargetView,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieLoaderTensorDeclView {
    pub id: u32,
    pub name: PieLoaderBytes,
    pub dtype: PieLoaderDType,
    pub encoding_kind: PieLoaderEncodingKind,
    pub quant_scheme: PieLoaderQuantScheme,
    pub shape: PieLoaderI64Slice,
    pub alignment: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderTensorDeclSlice {
    pub ptr: *const PieLoaderTensorDeclView,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieLoaderBufferDeclView {
    pub id: u32,
    pub tensor_id: u32,
    pub has_tensor: bool,
    pub bytes: u64,
    pub alignment: u32,
    pub temporary: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderBufferDeclSlice {
    pub ptr: *const PieLoaderBufferDeclView,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PieLoaderStorageInstrView {
    pub id: u32,
    pub kind: PieLoaderStorageInstrKind,
    pub buffer_id: u32,
    pub source: PieLoaderSourceExtentView,
    pub has_source: bool,
    pub dest: PieLoaderDestExtentView,
    pub has_dest: bool,
    pub input_buffers: PieLoaderBufferIdSlice,
    pub output_buffers: PieLoaderBufferIdSlice,
    pub tile_kind: PieLoaderTileMapKind,
    pub transform_from: PieLoaderQuantScheme,
    pub transform_to: PieLoaderQuantScheme,
    pub name: PieLoaderBytes,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderStorageInstrSlice {
    pub ptr: *const PieLoaderStorageInstrView,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderMemoryPlanView {
    pub persistent_bytes: u64,
    pub temporary_peak_bytes: u64,
    pub transform_scratch_peak_bytes: u64,
    pub checkpoint_read_bytes: u64,
    pub device_write_bytes: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PieLoaderStorageProgramView {
    pub version: u32,
    pub tensors: PieLoaderTensorDeclSlice,
    pub buffers: PieLoaderBufferDeclSlice,
    pub instrs: PieLoaderStorageInstrSlice,
    pub schedule: PieLoaderU32Slice,
    pub memory: PieLoaderMemoryPlanView,
}

#[repr(C)]
#[derive(Debug)]
pub struct PieLoaderError {
    pub code: PieLoaderStatus,
    pub message: *mut c_char,
}

impl Default for PieLoaderError {
    fn default() -> Self {
        Self {
            code: PieLoaderStatus::Ok,
            message: std::ptr::null_mut(),
        }
    }
}
