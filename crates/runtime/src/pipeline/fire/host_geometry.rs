//! A pass's geometry class, chosen once at bind, and the passes whose
//! geometry the host resolves while their geometry is not host-derivable.
//!
//! A program whose geometry ports the host cannot derive from what it wrote
//! (the tokens, positions, slots, KV extent, fold or mask a guest stage
//! computes) is served as device geometry or as a decode envelope when the
//! engine reads those ports from the device. Otherwise its geometry is
//! host-resolved: the host evaluates the guest's ops on the settled values,
//! so each fire waits for the one before it to settle (its fires
//! serialize). That happens for one of two reasons, each named at bind
//! ([`HostGeometry::bind_message`]) and both counted per fire in the model
//! status: `host_geometry_fires` sums them, and `host_geometry_fires_by`
//! splits them by engine, reason and program ([`counter`], [`by`]):
//!
//! - **An engine limit** ([`HostGeometry::EngineLacks`]): the program is
//!   device geometry or a decode envelope, and the engine's port mask lacks
//!   a port it needs. It serves for now; a bind refusal replaces it once
//!   engine-gpu carries device geometry.
//! - **The program's one class** ([`HostGeometry::Program`]): the program
//!   is neither (not a decode envelope, or its device-dependent geometry is
//!   not one), on any engine. Host-evaluated is its class, not a fallback.
//!
//! Which geometry ports each engine's mask lacks today, as each states it
//! in `EngineCaps::ports` (a decode envelope needs `embed_tokens` plus what
//! the program computes, device geometry `embed_tokens`, `positions`,
//! `slots`, `kv_len`, and `attn_mask` when a channel feeds it;
//! `embed_indptr` no class requires). The mask is the engine's, read where
//! it is stated:
//!
//! | engine          | mask (stated at)                                          | lacks                                 |
//! |-----------------|-----------------------------------------------------------|---------------------------------------|
//! | cuda (and ROCm) | device geometry + attn_mask + fold (engine-cuda `api.rs`)  | embed_indptr                          |
//! | metal           | device geometry + attn_mask + fold (engine-metal `api.rs`) | embed_indptr                          |
//! | xla             | device geometry + attn_mask + fold (engine-xla `api.rs`)   | embed_indptr                          |
//! | cerebras        | device geometry (engine-cerebras `spatial/engine/mod.rs`)  | attn_mask, fold, embed_indptr         |
//! | vulkan          | embed_tokens + fold + attn_mask (engine-vulkan `Host::ports`) | positions, kv_len, slots, embed_indptr |
//! | wgpu            | embed_tokens + fold + attn_mask (engine-wgpu `Host::ports`)   | positions, kv_len, slots, embed_indptr |
//!
//! So on Vulkan and wgpu no device-geometry program (beam search's, for
//! one) binds as device geometry: its fires are host-resolved, an engine
//! limit, until engine-gpu carries device geometry.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use eta_ir::registry::{GeometryClass, Port, PortMask};

/// Why a pass whose geometry the host cannot derive has its geometry
/// resolved by the host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostGeometry {
    /// The engine's port mask lacks `lacks`, ports the program's device
    /// class needs: a stated engine limit.
    EngineLacks {
        engine: &'static str,
        lacks: PortMask,
    },
    /// The program alone decides it: host-evaluated is its class.
    Program { why: String },
}

impl HostGeometry {
    /// Its reason as the model status names it.
    #[must_use]
    pub fn reason(&self) -> &'static str {
        match self {
            HostGeometry::EngineLacks { .. } => "engine_lacks",
            HostGeometry::Program { .. } => "program",
        }
    }

    /// What bind says of the pass, `program` its container hash.
    #[must_use]
    pub fn bind_message(&self, program: u64) -> String {
        match self {
            HostGeometry::EngineLacks { engine, lacks } => format!(
                "engine {engine} lacks geometry port(s) {} that program {program:016x} needs; \
                 its fires serialize (host-resolved geometry)",
                port_names(*lacks)
            ),
            HostGeometry::Program { why } => format!(
                "program {program:016x}'s geometry is host-evaluated ({why}): its fires \
                 serialize"
            ),
        }
    }
}

fn port_names(mask: PortMask) -> String {
    mask.iter().map(Port::name).collect::<Vec<_>>().join(", ")
}

fn minus(needs: PortMask, has: PortMask) -> PortMask {
    PortMask::from_bits(needs.bits() & !has.bits())
}

/// How a pass binds: the device-geometry pass's rows, the ports a decode
/// envelope reads from the device, the class, and why the host resolves it
/// when it does while its geometry is not host-derivable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindGeometry {
    pub devgeo: Option<Vec<u32>>,
    pub decode_envelope: Option<PortMask>,
    pub class: GeometryClass,
    pub host: Option<HostGeometry>,
}

/// What a program's geometry says of it at bind.
pub struct ProgramGeometry {
    /// Whether the host derives every geometry port from what it wrote.
    pub host_derivable: bool,
    /// The pooled device geometry's `qo_indptr`, when it is one.
    pub pooled: Option<Vec<u32>>,
    /// The decode envelope's required ports, `Ok(None)` with `why` when it
    /// is not one, `Err` when its device-dependent geometry is not one.
    pub envelope: Result<Option<PortMask>, String>,
    pub why: String,
    /// Whether a channel feeds its attention mask port.
    pub needs_mask_port: bool,
}

/// The class a program binds as on an engine with port mask `device` named
/// `engine`.
#[must_use]
pub fn classify(program: ProgramGeometry, device: PortMask, engine: &'static str) -> BindGeometry {
    let host = |host| BindGeometry {
        devgeo: None,
        decode_envelope: None,
        class: GeometryClass::Host,
        host,
    };
    if program.host_derivable {
        return host(None);
    }
    let devgeo_needs = if program.needs_mask_port {
        PortMask::DEVICE_GEOMETRY.with(Port::AttnMask)
    } else {
        PortMask::DEVICE_GEOMETRY
    };
    let envelope_covered =
        matches!(program.envelope, Ok(Some(required)) if device.covers(required));
    match program.pooled {
        // A program the decode envelope covers runs ahead: its pages and
        // write slots are the host's, so the next fire is staged while this
        // one runs. Device geometry reads every descriptor port back from
        // the device before it stages, so it is only the pass for programs
        // the envelope declines.
        Some(qo_indptr) if device.covers(devgeo_needs) && !envelope_covered => {
            return BindGeometry {
                devgeo: Some(qo_indptr),
                decode_envelope: None,
                class: GeometryClass::DeviceGeometry,
                host: None,
            };
        }
        Some(_) if !envelope_covered => {
            return host(Some(HostGeometry::EngineLacks {
                engine,
                lacks: minus(devgeo_needs, device),
            }));
        }
        _ => {}
    }
    match program.envelope {
        Ok(Some(required)) if device.covers(required) => BindGeometry {
            devgeo: None,
            decode_envelope: Some(required),
            class: GeometryClass::DecodeEnvelope,
            host: None,
        },
        Ok(Some(required)) => host(Some(HostGeometry::EngineLacks {
            engine,
            lacks: minus(required, device),
        })),
        Ok(None) => host(Some(HostGeometry::Program {
            why: format!("not a decode envelope: {}", program.why),
        })),
        Err(reason) => host(Some(HostGeometry::Program {
            why: format!(
                "device-dependent geometry that is not a decode envelope: {reason}; a fire \
                 blocks on values the host cannot derive"
            ),
        })),
    }
}

/// What a host-resolved fire is counted under: the engine its pass bound
/// on (its id and kind), why the host resolves it
/// ([`HostGeometry::reason`]), and the program (its container hash).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FireKey {
    pub engine: usize,
    pub kind: &'static str,
    pub reason: &'static str,
    pub program: u64,
}

static FIRES: Mutex<BTreeMap<FireKey, Arc<AtomicU64>>> = Mutex::new(BTreeMap::new());

/// The count a pass bound with a [`HostGeometry`] adds each fire to, one a
/// key: passes of one program on one engine for one reason share it.
#[must_use]
pub fn counter(key: FireKey) -> Arc<AtomicU64> {
    Arc::clone(FIRES.lock().unwrap().entry(key).or_default())
}

/// The fires counted so far, each key's.
#[must_use]
pub fn by() -> Vec<(FireKey, u64)> {
    FIRES
        .lock()
        .unwrap()
        .iter()
        .map(|(key, fires)| (key.clone(), fires.load(Ordering::Relaxed)))
        .collect()
}

/// The fires counted so far, summed: the model status's
/// `host_geometry_fires`.
#[must_use]
pub fn fires() -> u64 {
    by().iter().map(|(_, fires)| fires).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program(
        pooled: Option<Vec<u32>>,
        envelope: Result<Option<PortMask>, String>,
    ) -> ProgramGeometry {
        ProgramGeometry {
            host_derivable: false,
            pooled,
            envelope,
            why: "a guest stage computes its positions".to_string(),
            needs_mask_port: false,
        }
    }

    /// engine-gpu's mask, Vulkan's and wgpu's alike (their `Host::ports`).
    const VULKAN: PortMask = PortMask::DECODE_ENVELOPE
        .with(Port::Fold)
        .with(Port::AttnMask);
    const CUDA: PortMask = PortMask::DEVICE_GEOMETRY
        .with(Port::AttnMask)
        .with(Port::Fold);

    #[test]
    fn host_geometry_every_case() {
        a_capability_limited_bind_is_named_and_counted();
        a_program_only_host_class_is_named_and_counted();
        device_classes_and_host_derivable_name_nothing();
    }

    fn a_capability_limited_bind_is_named_and_counted() {
        // A device-geometry program on an engine without device geometry.
        let bound = classify(program(Some(vec![0, 2]), Ok(None)), VULKAN, "vulkan");
        assert_eq!(bound.class, GeometryClass::Host);
        let Some(host @ HostGeometry::EngineLacks { lacks, .. }) = bound.host.clone() else {
            panic!("an engine limit: {bound:?}");
        };
        assert_eq!(
            lacks,
            PortMask::of(&[Port::Positions, Port::KvLen, Port::Slots])
        );
        let message = host.bind_message(0xabc);
        assert!(
            message.contains("engine vulkan lacks geometry port(s)"),
            "{message}"
        );
        assert!(message.contains("positions"), "{message}");
        assert!(
            message.contains("program 0000000000000abc needs"),
            "{message}"
        );
        assert!(message.contains("its fires serialize"), "{message}");
        assert!(!message.contains("falling back"), "{message}");

        // Its mask port is the engine's: a channel-fed mask adds no lack.
        let mut masked = program(Some(vec![0, 2]), Ok(None));
        masked.needs_mask_port = true;
        let bound = classify(masked, VULKAN, "vulkan");
        assert_eq!(
            bound.host,
            Some(HostGeometry::EngineLacks {
                engine: "vulkan",
                lacks: PortMask::of(&[Port::Positions, Port::KvLen, Port::Slots])
            })
        );

        // A decode envelope whose required ports the engine lacks.
        let required = PortMask::DECODE_ENVELOPE.with(Port::Positions);
        let bound = classify(program(None, Ok(Some(required))), VULKAN, "vulkan");
        assert_eq!(
            bound.host,
            Some(HostGeometry::EngineLacks {
                engine: "vulkan",
                lacks: PortMask::of(&[Port::Positions])
            })
        );

        let key = FireKey {
            engine: 0,
            kind: "vulkan",
            reason: host.reason(),
            program: 0xabc,
        };
        let fires_before = fires();
        counter(key.clone()).fetch_add(1, Ordering::Relaxed);
        counter(key.clone()).fetch_add(1, Ordering::Relaxed);
        assert!(fires() >= fires_before + 2);
        let counted = by().into_iter().find(|(at, _)| *at == key).map(|(_, n)| n);
        assert!(counted.is_some_and(|n| n >= 2), "one count a key: {counted:?}");
    }

    fn a_program_only_host_class_is_named_and_counted() {
        for envelope in [
            Ok(None),
            Err("its slots follow a sampled token".to_string()),
        ] {
            // On any engine, one with device geometry among them.
            let bound = classify(program(None, envelope), CUDA, "cuda");
            assert_eq!(bound.class, GeometryClass::Host);
            let Some(host @ HostGeometry::Program { .. }) = bound.host else {
                panic!("the program's class");
            };
            let message = host.bind_message(0xabc);
            assert!(
                message.starts_with("program 0000000000000abc's geometry is host-evaluated"),
                "{message}"
            );
            assert!(message.contains("its fires serialize"), "{message}");
            assert!(!message.contains("falling back"), "{message}");
        }
        let key = FireKey {
            engine: 1,
            kind: "cuda",
            reason: "program",
            program: 0xdef,
        };
        let before = by().into_iter().find(|(at, _)| *at == key).map_or(0, |(_, n)| n);
        counter(key.clone()).fetch_add(1, Ordering::Relaxed);
        let after = by().into_iter().find(|(at, _)| *at == key).map_or(0, |(_, n)| n);
        assert_eq!(after, before + 1, "counted under its own engine, reason and program");
    }

    fn device_classes_and_host_derivable_name_nothing() {
        let bound = classify(program(Some(vec![0, 2]), Ok(None)), CUDA, "cuda");
        assert_eq!(bound.class, GeometryClass::DeviceGeometry);
        assert_eq!(bound.devgeo, Some(vec![0, 2]));
        assert_eq!(bound.host, None);

        // The envelope runs ahead of device geometry where it covers.
        let envelope = Ok(Some(PortMask::DECODE_ENVELOPE));
        let bound = classify(program(Some(vec![0, 1]), envelope), VULKAN, "vulkan");
        assert_eq!(bound.class, GeometryClass::DecodeEnvelope);
        assert_eq!(bound.host, None);

        let mut derivable = program(None, Ok(None));
        derivable.host_derivable = true;
        let bound = classify(derivable, VULKAN, "vulkan");
        assert_eq!((bound.class, bound.host), (GeometryClass::Host, None));

        // A channel-fed mask needs the mask port too.
        let mut masked = program(Some(vec![0, 1]), Ok(None));
        masked.needs_mask_port = true;
        let bound = classify(masked, PortMask::DEVICE_GEOMETRY, "cerebras");
        assert_eq!(
            bound.host,
            Some(HostGeometry::EngineLacks {
                engine: "cerebras",
                lacks: PortMask::of(&[Port::AttnMask])
            })
        );
    }
}
