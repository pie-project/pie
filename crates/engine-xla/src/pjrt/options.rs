//! The slice of `xla.CompileOptionsProto` this engine sets, declared by hand
//! with the upstream field numbers (`xla/pjrt/proto/compile_options.proto`)
//! so no protoc runs at build time. Fields left out decode as their defaults
//! on the plugin side.

use prost::Message;

#[derive(Clone, PartialEq, Message)]
pub struct ExecutableBuildOptionsProto {
    #[prost(int64, tag = "1")]
    pub device_ordinal: i64,
    #[prost(int64, tag = "4")]
    pub num_replicas: i64,
    #[prost(int64, tag = "5")]
    pub num_partitions: i64,
    #[prost(bool, tag = "6")]
    pub use_spmd_partitioning: bool,
}

/// `xla.OptionOverrideProto`: one compiler flag's value.
#[derive(Clone, PartialEq, Message)]
pub struct OptionOverrideProto {
    #[prost(bool, tag = "2")]
    pub bool_field: bool,
}

#[derive(Clone, PartialEq, Message)]
pub struct CompileOptionsProto {
    #[prost(message, optional, tag = "3")]
    pub executable_build_options: Option<ExecutableBuildOptionsProto>,
    #[prost(map = "string, message", tag = "7")]
    pub env_option_overrides: std::collections::HashMap<String, OptionOverrideProto>,
}

/// The TPU compiler's flags every program builds under (part of a cached
/// program's key). A conditional's arms share their caller's instruction
/// overlay, so entering one does not wait for its code to load (v6e
/// qwen3-0.6b, 2048-token prefill: a ~5 us overlay wait at each layer's kv
/// append and attention case, 250 us a prompt).
pub const TPU_FLAGS: [(&str, bool); 1] = [("xla_tpu_prefer_conditional_same_overlay", true)];

/// One replica, one partition, placed by the client: every executable this
/// engine builds runs on a single device; on a TPU, under [`TPU_FLAGS`].
#[must_use]
pub fn single_device(tpu: bool) -> Vec<u8> {
    let env_option_overrides = if tpu {
        TPU_FLAGS
            .iter()
            .map(|&(k, v)| (k.to_string(), OptionOverrideProto { bool_field: v }))
            .collect()
    } else {
        std::collections::HashMap::new()
    };
    CompileOptionsProto {
        env_option_overrides,
        executable_build_options: Some(ExecutableBuildOptionsProto {
            device_ordinal: -1,
            num_replicas: 1,
            num_partitions: 1,
            use_spmd_partitioning: false,
        }),
    }
    .encode_to_vec()
}
