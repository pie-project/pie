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

#[derive(Clone, PartialEq, Message)]
pub struct CompileOptionsProto {
    #[prost(message, optional, tag = "3")]
    pub executable_build_options: Option<ExecutableBuildOptionsProto>,
}

/// One replica, one partition, placed by the client: every executable this
/// engine builds runs on a single device.
#[must_use]
pub fn single_device() -> Vec<u8> {
    CompileOptionsProto {
        executable_build_options: Some(ExecutableBuildOptionsProto {
            device_ordinal: -1,
            num_replicas: 1,
            num_partitions: 1,
            use_spmd_partitioning: false,
        }),
    }
    .encode_to_vec()
}
