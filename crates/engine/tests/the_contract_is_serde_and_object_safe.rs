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

use engine::caps::{Capabilities, DeviceFacts, FireLimits, KvCopyDomains, PoolFacts};
use engine::channel::ChannelRegistration;
use engine::fire::{
    FoldLen, FrameSubmission, FrameTicket, KvDelta, Lane, Mask, Masking, Readout, RsVerb, Serves,
    Step,
};
use engine::channel::Ticket;
use engine::load::{Budgets, Checkpoint, LoadFacts, LoadRequest, Loaded, Residency};
use engine::program::ProgramRegistration;
// The launch package is the compiler's, and the contract names it rather than
// re-spelling it — so a test of the contract names it the same way.
use eta_compiler::codegen::launch::{
    LaunchChannel, LaunchPackage, LaunchPlanValue, LaunchStagePlan,
};
use eta_compiler::plan::{Dimension, SymbolicExtent};
use engine::transfer::{KvHandle, KvLayout, KvLayoutKind, KvRegion, MemoryDomain};
use engine::{Engine, Error};
use model_ir::Dtype;
use eta_ir::registry::{GeometryClass, ModelProfile, Port, PortMask};

/// Round-trip one noun through JSON and back.
fn round_trip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let text = serde_json::to_string(value).expect("the contract serializes");
    serde_json::from_str(&text).expect("and deserializes")
}

/// A `Capabilities` with every field stated, for the tests that need one.
fn caps_for_round_trip() -> Capabilities {
    Capabilities {
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
        kv_handle: None,
        media_encode: false,
        device_channel_commit: true,
        rs_verbs: false,
    }
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
                    // Non-empty so the round trip covers it: the working
                    // set's flat table, which a device-geometry lane carries
                    // and every other leaves empty.
                    translation: vec![7, 8, 9],
                },
                mask: Some(Masking::Extent(Mask::new(vec![0, 5], 5))),
                adapter: Some(2),
                drafts: true,
                captures_scores: true,
                rs: RsVerb::Fold,
                rs_reset: engine::RsReset::Fresh,
                channels: Vec::new(),
                readout: Readout::Last,
            },
            Lane::decode(1, 0b0100, 99, 31),
        ],
        attachments: Vec::new(),
        media: Vec::new(),
    };
    assert_eq!(round_trip(&submission), submission);
    assert_eq!(submission.rows(), 6);
    submission.validate().expect("a well-formed fire");

    // THE PER-ROW MASK IS THE SAME NOUN AND CROSSES THE SAME WIRE. A windowed
    // prefill states one restriction per query row — row `i` keeps `[i - 1,
    // i]` here — which no `Masking::Extent` is, and a remote engine has to be
    // handed all five of them, not the first.
    let windowed: Vec<Mask> = (0..5u32)
        .map(|row| {
            let front = row.saturating_sub(1);
            Mask::new(vec![front, row + 1 - front], 5)
        })
        .collect();
    let mut per_row = submission.clone();
    per_row.lanes[0].mask = Some(Masking::Rows(windowed.clone()));
    assert_eq!(round_trip(&per_row), per_row);
    per_row.validate().expect("a well-formed windowed prefill");
    assert_eq!(
        per_row.lanes[0].mask.as_ref().and_then(|m| m.of_row(3)),
        Some(&windowed[3]),
        "row 3 reads row 3's mask, and the silent row-zero substitution this          form replaced is exactly what that must not be"
    );
    assert_eq!(
        Masking::Extent(Mask::new(vec![0, 5], 5)).of_row(4),
        Some(&Mask::new(vec![0, 5], 5)),
        "an extent mask describes every row, which is what `over the extent`          means"
    );

    // And a count that is not the lane's row count is the one thing about a
    // per-row mask the LANE can check about itself.
    let mut short = per_row.clone();
    short.lanes[0].mask = Some(Masking::Rows(windowed[..3].to_vec()));
    let refused = short.validate().expect_err("three masks for five rows");
    assert!(
        matches!(&refused, Error::Invalid(why) if why.contains("per-row masks")),
        "a short per-row mask is refused by name, not silently padded: {refused:?}"
    );

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
        steps: vec![engine::FireTicket::default()],
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
        // The page list is a LIST (wave F3-tail): physical buffer-page slot
        // ids in buffer order, `KvDelta::pages`'s shape, and NOT a range —
        // the runtime's recurrent store copies a forked buffer page on write,
        // so a run is contiguous only by luck.
        RsVerb::Buffer {
            pages: vec![4, 9, 5],
            at: 3,
            fold: FoldLen::Device(Port::KvLen),
        },
        // The replay states its ORIGIN too (wave F3b): the buffer head a
        // mid-page fold left behind, so a replay after one starts at the
        // first live token rather than re-folding the absorbed ones.
        RsVerb::FoldBuffered {
            pages: vec![4, 9, 5],
            at: 2,
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
        rs_verbs: false,
    };
    assert_eq!(round_trip(&caps), caps);

    let registration = ProgramRegistration {
        program_hash: 0xdead_beef,
        launch: LaunchPackage {
            channels: vec![LaunchChannel::default()],
            plans: vec![LaunchStagePlan {
                value_types: vec![LaunchPlanValue {
                    dtype: eta_ir::Dtype::F32,
                    axes: vec![
                        Dimension::Symbolic(SymbolicExtent::TokenCount),
                        Dimension::Static(4096),
                    ],
                }],
                used_extents: vec![SymbolicExtent::TokenCount],
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
    let adapter = engine::AdapterRegistration {
        id: 3,
        planes: vec![
            engine::AdapterPlane {
                bank: "layer.0.lora_a".into(),
                bytes: vec![0x00, 0x3c, 0x00, 0xbc],
            },
            engine::AdapterPlane {
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
        residency: Residency {
            device_weight_budget: Some(8 << 30),
            host_weight_budget: None,
        },
        ordinal: 0,
        frames_in_flight: 2,
    };
    let there_and_back = round_trip(&request);
    assert_eq!(there_and_back.trace, request.trace);
    assert_eq!(there_and_back.budgets, request.budgets);
    assert_eq!(there_and_back.residency, request.residency);
}

/// **Residency is two budgets, and the noun round-trips like every other**
/// (alto design §7).
///
/// The three shapes that matter are all here: uncapped (today's load, and the
/// `Default`), one tier capped, and both. There is no fourth, which is the
/// argument for two `Option<u64>`s instead of a mode enum — an enum would
/// have had to enumerate what multiplication already covers.
#[test]
fn residency_is_two_budgets_and_uncapped_is_the_default() {
    assert_eq!(Residency::default(), Residency::uncapped());
    assert!(Residency::default().is_uncapped());

    for residency in [
        Residency::uncapped(),
        Residency {
            device_weight_budget: Some(4 << 30),
            host_weight_budget: None,
        },
        Residency {
            device_weight_budget: None,
            host_weight_budget: Some(64 << 30),
        },
        Residency {
            device_weight_budget: Some(4 << 30),
            host_weight_budget: Some(64 << 30),
        },
    ] {
        assert_eq!(round_trip(&residency), residency);
    }

    // A request written before the field existed still parses, and still
    // means what it meant: uncapped. Built by serializing a real request and
    // deleting the key, so the shape is the contract's own rather than a
    // hand-written guess at it.
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
        residency: Residency {
            device_weight_budget: Some(1 << 30),
            host_weight_budget: None,
        },
        ordinal: 0,
        frames_in_flight: 2,
    };
    let mut older = serde_json::to_value(&request).expect("a request serializes");
    assert!(
        older
            .as_object_mut()
            .expect("a request is a map")
            .remove("residency")
            .is_some(),
        "the field is named `residency` on the wire"
    );
    let parsed: LoadRequest =
        serde_json::from_value(older).expect("the residency field defaults");
    assert!(parsed.residency.is_uncapped());
}

/// **A budget under the checkpoint's resident demand is `Impossible`, and the
/// refusal names both numbers** (alto design §7, F1's doctrine).
///
/// The streaming half — the indirection table, the prefetch schedule, the
/// pinned host tier — is not built. The refusal is what makes stating the
/// budget honest anyway: it is a permanent answer about a tier this build
/// does not have, so it is `Impossible` and not `Exhausted`, and nothing the
/// deployment frees changes it.
#[test]
fn a_budget_under_the_demand_refuses_with_both_numbers() {
    let uncapped = Residency::uncapped();
    assert!(uncapped.admit(u64::MAX, u64::MAX).is_ok(), "uncapped admits everything");

    let capped = Residency {
        device_weight_budget: Some(1_000),
        host_weight_budget: None,
    };
    assert!(capped.admit(1_000, 0).is_ok(), "a demand that fits under the budget lands");

    let error = capped.admit(4_096, 0).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("4096"), "the demand is named: {text}");
    assert!(text.contains("1000"), "the budget is named: {text}");
    assert!(text.contains("device_weight_budget"), "the field is named: {text}");
    assert!(
        matches!(error, Error::Impossible(_)),
        "a tier that does not exist is permanent, not a full pool: {error}"
    );
    assert!(!error.is_retryable(), "resubmitting the same load cannot help");

    // The host tier refuses on the same terms, and today's demand for it is
    // zero — so a host budget only ever refuses a load that asked for a tier
    // nothing serves yet.
    let host = Residency {
        device_weight_budget: None,
        host_weight_budget: Some(16),
    };
    assert!(host.admit(1 << 40, 0).is_ok(), "no engine here holds host-resident weights");
    let error = host.admit(0, 512).unwrap_err();
    assert!(error.to_string().contains("host_weight_budget"), "{error}");
}

/// `Loaded` reports the residency it ACHIEVED, not the one it was asked for.
///
/// `weight_bytes` was always the size; `weights_resident` is the half that
/// says what the size is OF — the whole table, or the part of it that fits.
#[test]
fn a_load_reports_the_residency_it_achieved() {
    let loaded = Loaded {
        facts: LoadFacts {
            trace_name: "qwen3-0.8b".into(),
            weight_bytes: 1 << 30,
            weights_resident: true,
            weights_from_cache: true,
            arena_bytes: 1 << 20,
            pool_bytes: 1 << 24,
            input_bytes: 4096,
            // The ceiling and what is mapped under it are two numbers on an
            // elastic engine (alto design §8, wave C), and both round-trip.
            pool_committed_bytes: 1 << 22,
            pool_high_water_bytes: 1 << 23,
        },
        caps: caps_for_round_trip(),
    };
    let there_and_back = round_trip(&loaded);
    assert_eq!(there_and_back.facts, loaded.facts);
    assert!(
        there_and_back.facts.weights_resident,
        "every load in this workspace is fully resident today, and says so"
    );

    // The default is the honest one for a `LoadFacts` nobody filled: no
    // bytes, and therefore nothing resident and nothing restored.
    assert!(!LoadFacts::default().weights_resident);
    assert!(!LoadFacts::default().weights_from_cache);

    // An engine written before the warm-boot tier existed still deserializes,
    // and reports the truth about itself: it did not restore anything.
    let mut older = serde_json::to_value(&loaded.facts).expect("facts serialize");
    assert!(
        older
            .as_object_mut()
            .expect("facts are a map")
            .remove("weights_from_cache")
            .is_some(),
        "the field is named `weights_from_cache` on the wire"
    );
    let parsed: LoadFacts = serde_json::from_value(older).expect("the field defaults");
    assert!(!parsed.weights_from_cache);
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
        fn load(&mut self, _: LoadRequest) -> engine::Result<engine::Loaded> {
            Err(self.unsupported("load"))
        }
        fn submit(
            &mut self,
            _: &FrameSubmission,
        ) -> engine::Result<engine::FrameTicket> {
            Err(self.unsupported("submit"))
        }
    }

    let mut engines: Vec<Box<dyn Engine>> = vec![Box::new(Refusing)];
    let engine = &mut engines[0];
    let error = engine.copy_kv(&engine::KvCopy::default()).unwrap_err();
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
        pages: vec![0],
        at: 0,
        bound: 4,
        len: FoldLen::Host(2),
    };
    let error = lane.validate().unwrap_err();
    assert!(error.to_string().contains("RsVerb::Fold"), "{error}");

    // **THE MIXED ROW IS SERVED, AND WHAT IS LEFT IS AN ARITHMETIC** (wave
    // F3b's 2R interior split). A `Buffer` whose fold is non-zero folds a
    // prefix of the tokens it is writing; the boundary is a position among
    // THIS FIRE'S rows, so a host-stated one past them is a refusal — and
    // every value inside them, including the whole row, is served.
    let serves = Serves {
        device_channel_commit: false,
        rs_verbs: true,
    };
    let mut mixed = Lane::decode(0, 0, 7, 0);
    mixed.rs = RsVerb::Buffer {
        pages: vec![2, 3],
        at: 4,
        fold: FoldLen::Host(6),
    };
    let error = mixed.validate_for(serves).unwrap_err();
    assert!(error.to_string().contains("folds 6 of the 1 rows"), "{error}");
    for served in [
        FoldLen::Host(0),
        FoldLen::Host(1),
        FoldLen::Device(Port::RsFoldLen),
    ] {
        mixed.rs = RsVerb::Buffer {
            pages: vec![2, 3],
            at: 4,
            fold: served,
        };
        mixed
            .validate_for(serves)
            .expect("a boundary inside the lane's own rows is the mixed row F3b serves");
    }
    // And the lane stays refused against an engine with no device half at
    // all, which is the clause the mixed row does not get to skip.
    mixed
        .validate_for(Serves::NONE)
        .expect_err("an engine that serves no RS verb refuses this one too");
    let mut lane = Lane::decode(0, 0, 7, 0);
    lane.rs = RsVerb::FoldBuffered {
        pages: vec![0],
        at: 0,
        bound: 4,
        len: FoldLen::Host(2),
    };

    // And the frame refuses on behalf of any step that does (article 4: every
    // step is checked before any of them is admitted).
    let frame = FrameSubmission {
        steps: vec![Step::default(), Step {
            lanes: vec![lane],
            attachments: Vec::new(),
            media: Vec::new(),
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
        media: Vec::new(),
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
