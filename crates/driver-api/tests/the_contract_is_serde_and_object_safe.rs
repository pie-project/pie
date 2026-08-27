//! The two properties decision 19 turned into type-level facts, asserted.
//!
//! > *"Remote is a property, not an encoding: every noun serde, trait
//! > object-safe; wire versioning is the transport's concern, not the
//! > contract's."*
//!
//! Both halves of that are structural, so both are checked here rather than
//! trusted. The old crate could not have run either test: its submission types
//! carried `Vec<*mut TerminalCell>`, which is neither serializable nor `Send`,
//! and its remote path was a hand-written `ExecutorRequest` enum that
//! duplicated the trait's verbs precisely because the trait's arguments could
//! not cross a socket.

use driver_api::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use driver_api::channel::ChannelRegistration;
use driver_api::fire::{FireSubmission, KvDelta, Lane, Mask, Readout};
use driver_api::load::{Budgets, Checkpoint, LoadRequest};
use driver_api::program::{
    Axis, ExtentRole, LaunchChannel, LaunchPackage, LaunchPlanValue, LaunchStagePlan,
    ProgramRegistration,
};
use driver_api::transfer::{KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
use driver_api::{Driver, DriverError};
use model_ir::Dtype;
use tensor_ir::registry::{GeometryClass, ModelProfile, Port, PortMask};

/// Round-trip one noun through JSON and back.
fn round_trip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let text = serde_json::to_string(value).expect("the contract serializes");
    serde_json::from_str(&text).expect("and deserializes")
}

/// Every noun a remote driver would have to send survives a round trip.
///
/// Not "derives Serialize" — that a compiler checks — but that the derived
/// pair actually inverts, which is what a `#[serde(deny_unknown_fields)]` on a
/// struct with an untagged enum inside it does not.
#[test]
fn every_noun_round_trips() {
    let layout = KvLayout {
        num_layers: 2,
        num_kv_heads: 4,
        head_dim: 64,
        page_size: 16,
        dtype: Dtype::Bf16,
        kind: KvLayoutKind::KvSeparate,
        storage_format: String::new(),
        region_page_bytes: Vec::new(),
    };
    assert_eq!(round_trip(&layout), layout);

    let handle = KvHandle {
        regions: vec![KvRegion {
            base: 0x1000,
            len: layout.page_bytes() * 8,
            page_stride: layout.page_bytes(),
            domain: MemoryDomain::CudaDevice(1),
        }],
        layout: layout.clone(),
    };
    assert_eq!(round_trip(&handle), handle);
    assert_eq!(handle.page_capacity(), Some(8));

    let submission = FireSubmission {
        lanes: vec![
            Lane {
                slot: 0,
                word: 0b1011,
                tokens: vec![1, 2, 3, 4, 5],
                positions: Vec::new(),
                kv: KvDelta {
                    held: 0,
                    pages: vec![7, 8],
                    translation: Vec::new(),
                },
                mask: Some(Mask::new(vec![0, 5], 5)),
                adapter: Some(2),
                readout: Readout::Last,
            },
            Lane::decode(1, 0b0100, 99, 31),
        ],
        attachments: Vec::new(),
    };
    assert_eq!(round_trip(&submission), submission);
    assert_eq!(submission.rows(), 6);
    submission.validate().expect("a well-formed fire");

    let caps = Capabilities {
        device: DeviceFacts {
            backend: "cuda".into(),
            domain: MemoryDomain::CudaDevice(0),
            sms: 142,
            unified_memory: false,
            fp8_native: true,
            native_mxfp4_moe: false,
            storage_alignment: 256,
            storage_max_tile_bytes: 1 << 30,
            codegen_backend: Some("cuda".into()),
        },
        pools: PoolFacts::default(),
        limits: FireLimits {
            max_lanes: 256,
            max_tokens: 8192,
            max_page_refs: 4096,
            max_context: 4096,
        },
        profile: ModelProfile::dummy(),
        ports: PortMask::DEVICE_GEOMETRY,
        geometry: GeometryClass::DeviceGeometry,
        kv_copy: KvCopyDomains {
            device_to_host: true,
            host_to_device: true,
            ..KvCopyDomains::default()
        },
        kv_handle: Some(handle),
        media_encode: false,
    };
    assert_eq!(round_trip(&caps), caps);

    let registration = ProgramRegistration {
        program_hash: 0xdead_beef,
        launch: LaunchPackage {
            channels: vec![LaunchChannel::default()],
            plans: vec![LaunchStagePlan {
                value_types: vec![LaunchPlanValue {
                    dtype: tensor_ir::DType::F32,
                    axes: vec![Axis::Symbolic(ExtentRole::TokenCount), Axis::Static(4096)],
                }],
                used_extents: vec![ExtentRole::TokenCount],
                ..LaunchStagePlan::default()
            }],
            ..LaunchPackage::default()
        },
        ..ProgramRegistration::default()
    };
    assert_eq!(round_trip(&registration), registration);
    assert_eq!(round_trip(&ChannelRegistration::default()), ChannelRegistration::default());
}

/// A `LoadRequest` carries a real `model_ir::Plan` across the boundary, and it
/// survives the trip.
///
/// This is decision 18 as a test: the plan is what crosses, so a remote driver
/// gets one for free, and nothing about `Baked` appears anywhere in the value.
#[test]
fn the_plan_crosses_the_boundary() {
    let request = LoadRequest {
        plan: model_ir::Plan {
            name: "qwen3-0.8b".into(),
            plane: model_ir::Plane::Cuda,
            params: Vec::new(),
            caches: Vec::new(),
            values: Vec::new(),
            nodes: Vec::new(),
            seams: Vec::new(),
        },
        checkpoint: Checkpoint::Path("/models/qwen".into()),
        budgets: Budgets::default(),
        ordinal: 0,
    };
    let there_and_back = round_trip(&request);
    assert_eq!(there_and_back.plan, request.plan);
    assert_eq!(there_and_back.budgets, request.budgets);
}

/// A driver is usable as `dyn Driver`, which is what lets the engine hold a
/// CUDA shell, a Metal shell and a socket in one `Vec`.
#[test]
fn a_driver_is_a_trait_object() {
    struct Refusing;

    impl Driver for Refusing {
        fn kind(&self) -> &'static str {
            "refusing"
        }
        fn load(&mut self, _: LoadRequest) -> driver_api::Result<driver_api::Loaded> {
            Err(self.unsupported("load"))
        }
        fn fire(&mut self, _: &FireSubmission) -> driver_api::Result<driver_api::FireTicket> {
            Err(self.unsupported("fire"))
        }
    }

    let mut drivers: Vec<Box<dyn Driver>> = vec![Box::new(Refusing)];
    let driver = &mut drivers[0];
    let error = driver.copy_kv(&driver_api::KvCopy::default()).unwrap_err();
    assert!(matches!(
        error,
        DriverError::Unsupported {
            verb: "copy_kv",
            driver: "refusing"
        }
    ));
    assert_eq!(
        error.to_string(),
        "the refusing driver does not serve `copy_kv`"
    );
    assert!(!error.is_scheduling());
}

/// The two errors that are scheduling answers say so, and the rest do not.
///
/// The engine's retry loop turns on this predicate: `Exhausted` means "behind
/// something that frees pages", `Impossible` means "never". Everything else is
/// a fault to surface.
#[test]
fn exhausted_and_impossible_are_the_scheduling_answers() {
    let exhausted = DriverError::Exhausted {
        resource: "kv pages",
        wanted: 64,
        available: 12,
    };
    assert!(exhausted.is_scheduling());
    assert_eq!(
        exhausted.to_string(),
        "kv pages exhausted: wanted 64, 12 available"
    );
    assert!(DriverError::Impossible("9000 rows past an 8192 bake".into()).is_scheduling());
    assert!(!DriverError::Device("cudaErrorLaunchFailure".into()).is_scheduling());
    assert!(!DriverError::invalid("a CSR that decreases").is_scheduling());
}

/// The port sets are the registry's own tags, and the geometry classes are
/// the port sets (decision 19).
///
/// The masks this replaced were a private thirteen-bit numbering in
/// `driver-api` that disagreed with `Port`'s wire tags — `PIE_DEVICE_PORT_PAGES`
/// was bit 1 where `Port::Pages` is tag 3 — and nothing compared them.
#[test]
fn the_geometry_classes_are_their_port_sets() {
    assert!(PortMask::DECODE_ENVELOPE.contains(Port::EmbedTokens));
    assert!(PortMask::DECODE_ENVELOPE.contains(Port::Positions));
    assert!(PortMask::DECODE_ENVELOPE.contains(Port::KvLen));
    assert!(!PortMask::DECODE_ENVELOPE.contains(Port::Pages));

    assert!(PortMask::DEVICE_GEOMETRY.covers(PortMask::DECODE_ENVELOPE));
    assert_eq!(
        GeometryClass::DeviceGeometry.ports(),
        PortMask::DEVICE_GEOMETRY
    );
    assert_eq!(
        GeometryClass::admitted_by(PortMask::DEVICE_GEOMETRY),
        GeometryClass::DeviceGeometry
    );
    assert_eq!(
        GeometryClass::admitted_by(PortMask::DECODE_ENVELOPE),
        GeometryClass::DecodeEnvelope
    );
    assert_eq!(GeometryClass::admitted_by(PortMask::NONE), GeometryClass::Host);

    // Every bit is the port's own tag, for every port there is.
    for port in Port::ALL {
        assert!(PortMask::NONE.with(*port).contains(*port));
        assert_eq!(PortMask::NONE.with(*port).bits(), 1 << (*port as u8));
    }
}

/// A submission the contract does not describe is refused with the reason in
/// it, not with a status code.
#[test]
fn a_malformed_fire_says_what_is_wrong() {
    let mut fire = FireSubmission {
        lanes: vec![Lane::decode(3, 0, 7, 0), Lane::decode(3, 0, 8, 0)],
        attachments: Vec::new(),
    };
    let error = fire.validate().unwrap_err();
    assert!(
        error.to_string().contains("slot 3 appears twice"),
        "{error}"
    );

    fire.lanes[1].slot = 4;
    fire.lanes[1].positions = vec![0, 1, 2];
    let error = fire.validate().unwrap_err();
    assert!(
        error.to_string().contains("3 positions for 1 tokens"),
        "{error}"
    );
}
