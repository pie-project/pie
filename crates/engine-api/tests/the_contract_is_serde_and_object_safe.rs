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

use engine_api::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine_api::channel::ChannelRegistration;
use engine_api::fire::{
    FoldLen, FrameSubmission, FrameTicket, KvDelta, Lane, Mask, Readout, RsVerb, Step,
};
use engine_api::channel::Ticket;
use engine_api::load::{Budgets, Checkpoint, LoadRequest};
use engine_api::program::{
    Axis, ExtentRole, LaunchChannel, LaunchPackage, LaunchPlanValue, LaunchStagePlan,
    ProgramRegistration,
};
use engine_api::transfer::{KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
use engine_api::{Engine, Error};
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

/// Every noun a remote engine would have to send survives a round trip.
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

    let submission = Step {
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
                drafts: true,
                captures_scores: true,
                rs: RsVerb::Fold,
                channels: Vec::new(),
                readout: Readout::Last,
            },
            Lane::decode(1, 0b0100, 99, 31),
        ],
        attachments: Vec::new(),
    };
    assert_eq!(round_trip(&submission), submission);
    assert_eq!(submission.rows(), 6);
    submission.validate().expect("a well-formed fire");

    // The frame is the unit of work, and a fire is the degenerate one-step
    // frame — both spellings round trip and both validate.
    let frame = FrameSubmission {
        steps: vec![submission.clone(), submission.clone()],
    };
    assert_eq!(round_trip(&frame), frame);
    assert_eq!(frame.rows(), 12);
    frame.validate().expect("a well-formed frame");
    assert_eq!(FrameSubmission::of(submission.clone()).steps.len(), 1);
    let ticket = FrameTicket {
        id: 7,
        steps: vec![engine_api::FireTicket::default()],
    };
    assert_eq!(round_trip(&ticket), ticket);

    // The two F1 shapes with no F1 mechanism: they SERIALIZE (a remote engine
    // must be able to carry a refusal's cause), and they REFUSE.
    let ticketed = Ticket {
        channel: 3,
        expected_head: 11,
        expected_tail: 12,
    };
    assert_eq!(round_trip(&ticketed), ticketed);
    for verb in [
        RsVerb::Fold,
        RsVerb::Buffer {
            pages: engine_api::PageRange {
                page_index: 4,
                page_count: 2,
            },
            fold: FoldLen::Device(Port::KvLen),
        },
        RsVerb::FoldBuffered {
            bound: 8,
            len: FoldLen::Host(5),
        },
    ] {
        assert_eq!(round_trip(&verb), verb);
    }

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
        device_channel_commit: true,
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

    // The correction class's residency verb (palo C2, design §8). Remote is a
    // property and not an encoding, so a bank's rows cross the same way a
    // checkpoint's plan does.
    let adapter = engine_api::AdapterRegistration {
        id: 3,
        planes: vec![
            engine_api::AdapterPlane {
                bank: "layer.0.lora_a".into(),
                bytes: vec![0x00, 0x3c, 0x00, 0xbc],
            },
            engine_api::AdapterPlane {
                bank: "layer.0.lora_b".into(),
                bytes: vec![0x00, 0x00, 0x80, 0x3f],
            },
        ],
    };
    assert_eq!(round_trip(&adapter), adapter);
}

/// A `LoadRequest` carries a real `model_ir::Trace` across the boundary, and it
/// survives the trip.
///
/// This is decision 18 as a test: the plan is what crosses, so a remote engine
/// gets one for free, and nothing about `CompiledModel` appears anywhere in the value.
#[test]
fn the_plan_crosses_the_boundary() {
    let request = LoadRequest {
        trace: model_ir::Trace {
            name: "qwen3-0.8b".into(),
            platform: model_ir::Platform::Cuda,
            params: Vec::new(),
            caches: Vec::new(),
            values: Vec::new(),
            nodes: Vec::new(),
            seams: Vec::new(),
        },
        checkpoint: Checkpoint::Path("/models/qwen".into()),
        budgets: Budgets::default(),
        ordinal: 0,
        frames_in_flight: 2,
    };
    let there_and_back = round_trip(&request);
    assert_eq!(there_and_back.trace, request.trace);
    assert_eq!(there_and_back.budgets, request.budgets);
}

/// An engine is usable as `dyn Engine`, which is what lets the runtime hold a
/// CUDA shell, a Metal shell and a socket in one `Vec`.
#[test]
fn an_engine_is_a_trait_object() {
    struct Refusing;

    impl Engine for Refusing {
        fn kind(&self) -> &'static str {
            "refusing"
        }
        fn load(&mut self, _: LoadRequest) -> engine_api::Result<engine_api::Loaded> {
            Err(self.unsupported("load"))
        }
        fn submit(
            &mut self,
            _: &FrameSubmission,
        ) -> engine_api::Result<engine_api::FrameTicket> {
            Err(self.unsupported("submit"))
        }
    }

    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Refusing)];
    let engine = &mut engines[0];
    let error = engine.copy_kv(&engine_api::KvCopy::default()).unwrap_err();
    assert!(matches!(
        error,
        Error::Unsupported {
            verb: "copy_kv",
            engine: "refusing"
        }
    ));
    assert_eq!(
        error.to_string(),
        "the refusing engine does not serve `copy_kv`"
    );
    assert!(!error.is_scheduling());
}

/// The two errors that are scheduling answers say so, and the rest do not.
///
/// The runtime's retry loop turns on this predicate: `Exhausted` means "behind
/// something that frees pages", `Impossible` means "never". Everything else is
/// a fault to surface.
#[test]
fn exhausted_and_impossible_are_the_scheduling_answers() {
    let exhausted = Error::Exhausted {
        resource: "kv pages",
        wanted: 64,
        available: 12,
    };
    assert!(exhausted.is_scheduling());
    assert_eq!(
        exhausted.to_string(),
        "kv pages exhausted: wanted 64, 12 available"
    );
    assert!(Error::Impossible("9000 rows past an 8192 bake".into()).is_scheduling());
    assert!(!Error::Device("cudaErrorLaunchFailure".into()).is_scheduling());
    assert!(!Error::invalid("a CSR that decreases").is_scheduling());

    // Article 4 splits the two: only `Exhausted` is the same frame submitted
    // again, because only `Exhausted` left nothing behind.
    assert!(exhausted.is_retryable());
    assert!(!Error::Impossible("9000 rows past an 8192 bake".into()).is_retryable());
    assert!(!Error::Device("cudaErrorLaunchFailure".into()).is_retryable());
}

/// The two v2 shapes with no device half refuse loudly rather than being
/// dropped — a stated prediction nobody validated, and a recurrent verb
/// nobody serves, are exactly the two ways a contract lies to its caller.
#[test]
fn the_unbuilt_v2_shapes_are_refused_by_name() {
    let mut lane = Lane::decode(0, 0, 7, 0);
    lane.channels = vec![Ticket {
        channel: 1,
        expected_head: 0,
        expected_tail: 1,
    }];
    let error = lane.validate().unwrap_err();
    assert!(error.to_string().contains("Lane::channels"), "{error}");
    assert!(
        error.to_string().contains("pull-validate"),
        "the refusal names the missing kernels, not a wave number: {error}"
    );
    assert!(!error.is_scheduling(), "a missing kernel is not a full pool");

    let mut lane = Lane::decode(0, 0, 7, 0);
    lane.rs = RsVerb::FoldBuffered {
        bound: 4,
        len: FoldLen::Host(2),
    };
    let error = lane.validate().unwrap_err();
    assert!(error.to_string().contains("RsVerb::Fold"), "{error}");

    // And the frame refuses on behalf of any step that does (article 4: every
    // step is checked before any of them is admitted).
    let frame = FrameSubmission {
        steps: vec![Step::default(), Step {
            lanes: vec![lane],
            attachments: Vec::new(),
        }],
    };
    assert!(frame.validate().is_err());
    assert!(FrameSubmission { steps: Vec::new() }
        .validate()
        .unwrap_err()
        .to_string()
        .contains("no steps"));
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
    let mut fire = Step {
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
