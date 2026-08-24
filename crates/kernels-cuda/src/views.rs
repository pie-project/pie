use core::ffi::c_void;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PagedKvView {
    pub keys: *mut u8,

    pub values: *mut u8,

    pub bf16_keys: *mut u8,

    pub bf16_values: *mut u8,

    pub page_indices: *const i32,

    pub page_indptr: *const i32,

    pub last_page_lens: *const i32,

    pub key_scales: *const c_void,

    pub value_scales: *const c_void,

    pub write_page: *const i32,

    pub write_offset: *const i32,

    pub page_size: i32,

    pub seq_stride: i64,

    pub head_stride: i64,

    pub layout: i32,

    pub storage_dtype: i32,

    pub scheme_byte: i32,

    pub native_bf16: bool,

    pub has_envelopes: bool,

    pub env_min: *const u16,

    pub env_max: *const u16,

    pub block_size: i32,

    pub max_pages_per_request: i32,

    pub pages_in_batch: i32,

    pub qo_indptr: *const i32,

    pub row_valid: *const u8,

    pub requests: i32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RecurrentView {
    pub slab: *mut c_void,

    pub slot_ids: *const i32,

    pub slot_stride_elems: i64,

    pub slots: *const i32,

    pub state: *mut c_void,

    pub conv_state: *mut c_void,

    pub new_conv_state: *mut c_void,

    pub conv_slab: *mut c_void,

    pub conv_stride: i64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MaskView {
    pub mask: *const u8,

    pub indptr: *const i32,

    pub enabled: bool,

    pub stride: i64,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ExpertWeightsView {
    pub ptrs: *const u8,

    pub scale_ptrs: *const u8,

    pub bias_ptrs: *const u8,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct MoeBanksView {
    pub up_weight_ptrs: *const *const c_void,

    pub down_weight_ptrs: *const *const c_void,

    pub expert_up: *mut c_void,

    pub expert_act: *mut c_void,

    pub expert_out: *mut c_void,

    pub aligned_up: *mut c_void,

    pub aligned_act: *mut c_void,

    pub aligned_out: *mut c_void,

    pub a_up_ptrs: *mut *const c_void,

    pub b_up_ptrs: *mut *const c_void,

    pub c_up_ptrs: *mut *mut c_void,

    pub a_down_ptrs: *mut *const c_void,

    pub b_down_ptrs: *mut *const c_void,

    pub c_down_ptrs: *mut *mut c_void,

    pub route_weights: *mut f32,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GemmGroupsView {
    pub act_ptrs: *const *const c_void,

    pub weight_ptrs: *const *const c_void,

    pub out_ptrs: *const *mut c_void,

    pub m_array_host: *const i32,
}

kernels::raise!(

    KvCache = "kv_cache" => PagedKvView
);
kernels::raise!(

    RecurrentState = "recurrent_state" => RecurrentView
);
kernels::raise!(

    AttnMask = "attention_mask" => MaskView
);
kernels::raise!(

    ExpertWeights = "moe.expert_weights" => ExpertWeightsView
);
kernels::raise!(

    MoeBanks = "moe.banks" => MoeBanksView
);
kernels::raise!(

    GemmGroups = "gemm.groups" => GemmGroupsView
);
kernels::raise!(

    Dsv4StateKv = "dsv4.state_kv" => crate::jit::abi::bf16
);
kernels::raise!(

    Dsv4StateScore = "dsv4.state_score" => crate::jit::abi::bf16
);
kernels::raise!(

    Dsv4Ape = "dsv4.ape" => f32
);
kernels::raise!(

    Dsv4CompKvPages = "dsv4.comp_kv_pages" => crate::jit::abi::bf16
);
kernels::raise!(

    MtpPendingHidden = "mtp.pending_hidden" => c_void
);
kernels::raise!(

    QoIndptrHost = "qo_indptr.host" => u32
);
kernels::raise!(

    KvPageIndptrHost = "kv_page_indptr.host" => u32
);

kernels::raise!(

    RowValid = "row_valid" => u8
);

kernels::raise!(

    RequestOfToken = "request_of_token" => i32
);

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ScoreView {
    pub indptr: *const i32,

    pub window: u32,
}

kernels::raise!(

    AttnScore = "attn.score" => ScoreView
);
