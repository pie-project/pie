//! A pass's geometry class, chosen once at bind, and the passes whose
//! geometry the host resolves while their geometry is not host-derivable.
//!
//! A program whose geometry ports the host cannot derive from what it wrote
//! (the tokens, positions, slots, KV extent, fold or mask a guest stage
//! computes) is served as device geometry or as a decode envelope when the
//! engine reads those ports from the device. Where the engine's port mask
//! (`EngineCaps::ports`) lacks a port the program's device class needs, the
//! bind is refused by name ([`EngineLacks`]): an engine limit is never
//! served by resolving the geometry on the host instead.
//!
//! A program that is neither (not a decode envelope, or its
//! device-dependent geometry is not one) is host-evaluated on any engine
//! ([`HostGeometry`]): the host evaluates the guest's ops on the settled
//! values, so each fire waits for the one before it to settle (its fires
//! serialize). That is its class, not a fallback; bind names it
//! ([`HostGeometry::bind_message`]) and each fire is counted in the model
//! status's `host_geometry_fires` ([`fires`]).
//!
//! Which geometry ports each engine's mask carries (a decode envelope needs
//! `embed_tokens` plus what the program computes, device geometry
//! `embed_tokens`, `positions`, `slots`, `kv_len`, and `attn_mask` when a
//! channel feeds it; `embed_indptr` no class requires):
//!
//! | engine            | mask                                      | lacks                                     |
//! |-------------------|-------------------------------------------|-------------------------------------------|
//! | cuda (and ROCm)   | device geometry + attn_mask + fold        | embed_indptr                              |
//! | metal             | device geometry + attn_mask + fold        | embed_indptr                              |
//! | xla               | device geometry + attn_mask + fold        | embed_indptr                              |
//! | vulkan, wgpu      | device geometry + attn_mask + fold        | embed_indptr                              |
//! | cerebras          | device geometry                           | attn_mask, fold, embed_indptr             |

use std::sync::atomic::{AtomicU64, Ordering};

use eta_ir::registry::{GeometryClass, Port, PortMask};

/// A pass whose program alone decides that the host evaluates its
/// geometry: host-evaluated is its class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostGeometry {
    pub why: String,
}

impl HostGeometry {
    /// What bind says of the pass, `program` its container hash.
    #[must_use]
    pub fn bind_message(&self, program: u64) -> String {
        format!(
            "program {program:016x}'s geometry is host-evaluated ({}): its fires serialize",
            self.why
        )
    }
}

/// The engine's port mask lacks `lacks`, ports the program's device class
/// needs: the bind is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineLacks {
    pub engine: &'static str,
    pub lacks: PortMask,
}

impl EngineLacks {
    /// Why bind refuses the pass, `program` its container hash.
    #[must_use]
    pub fn refusal(&self, program: u64) -> String {
        format!(
            "pipeline: engine {} does not read geometry port(s) {} from the device, and \
             program {program:016x}'s geometry is computed there; the engine cannot serve it",
            self.engine,
            port_names(self.lacks)
        )
    }
}

fn port_names(mask: PortMask) -> String {
    mask.iter().map(Port::name).collect::<Vec<_>>().join(", ")
}

fn minus(needs: PortMask, has: PortMask) -> PortMask {
    PortMask::from_bits(needs.bits() & !has.bits())
}

/// How a pass binds: the device-geometry pass's rows, the ports a decode
/// envelope reads from the device, the class, and why the host evaluates it
/// when its program is host-evaluated.
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
/// `engine`, or why the engine cannot serve it.
pub fn classify(
    program: ProgramGeometry,
    device: PortMask,
    engine: &'static str,
) -> Result<BindGeometry, EngineLacks> {
    let host = |host| BindGeometry {
        devgeo: None,
        decode_envelope: None,
        class: GeometryClass::Host,
        host,
    };
    if program.host_derivable {
        return Ok(host(None));
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
            return Ok(BindGeometry {
                devgeo: Some(qo_indptr),
                decode_envelope: None,
                class: GeometryClass::DeviceGeometry,
                host: None,
            });
        }
        Some(_) if !envelope_covered => {
            return Err(EngineLacks {
                engine,
                lacks: minus(devgeo_needs, device),
            });
        }
        _ => {}
    }
    match program.envelope {
        Ok(Some(required)) if device.covers(required) => Ok(BindGeometry {
            devgeo: None,
            decode_envelope: Some(required),
            class: GeometryClass::DecodeEnvelope,
            host: None,
        }),
        Ok(Some(required)) => Err(EngineLacks {
            engine,
            lacks: minus(required, device),
        }),
        Ok(None) => Ok(host(Some(HostGeometry {
            why: format!("not a decode envelope: {}", program.why),
        }))),
        Err(reason) => Ok(host(Some(HostGeometry {
            why: format!(
                "device-dependent geometry that is not a decode envelope: {reason}; a fire \
                 blocks on values the host cannot derive"
            ),
        }))),
    }
}

static FIRES: AtomicU64 = AtomicU64::new(0);

/// A fire of a pass bound with a [`HostGeometry`] resolved its geometry on
/// the host.
pub fn count_fire() {
    FIRES.fetch_add(1, Ordering::Relaxed);
}

/// The fires counted so far: the model status's `host_geometry_fires`.
#[must_use]
pub fn fires() -> u64 {
    FIRES.load(Ordering::Relaxed)
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

    const DECODE_ONLY: PortMask = PortMask::DECODE_ENVELOPE.with(Port::Fold);
    const CUDA: PortMask = PortMask::DEVICE_GEOMETRY
        .with(Port::AttnMask)
        .with(Port::Fold);

    #[test]
    fn host_geometry_every_case() {
        a_capability_limited_bind_is_refused_by_name();
        a_program_only_host_class_is_named_and_counted();
        device_classes_and_host_derivable_name_nothing();
    }

    fn a_capability_limited_bind_is_refused_by_name() {
        // A device-geometry program on an engine without device geometry.
        let refused = classify(program(Some(vec![0, 2]), Ok(None)), DECODE_ONLY, "stub")
            .expect_err("an engine limit refuses the bind");
        assert_eq!(
            refused.lacks,
            PortMask::of(&[Port::Positions, Port::KvLen, Port::Slots])
        );
        let message = refused.refusal(0xabc);
        assert!(message.contains("engine stub does not read"), "{message}");
        assert!(message.contains("positions"), "{message}");
        assert!(message.contains("program 0000000000000abc"), "{message}");
        assert!(!message.contains("serialize"), "{message}");

        // A decode envelope whose required ports the engine lacks.
        let required = PortMask::DECODE_ENVELOPE.with(Port::Positions);
        assert_eq!(
            classify(program(None, Ok(Some(required))), DECODE_ONLY, "stub"),
            Err(EngineLacks {
                engine: "stub",
                lacks: PortMask::of(&[Port::Positions])
            })
        );

        // A channel-fed mask needs the mask port too.
        let mut masked = program(Some(vec![0, 1]), Ok(None));
        masked.needs_mask_port = true;
        assert_eq!(
            classify(masked, PortMask::DEVICE_GEOMETRY, "cerebras"),
            Err(EngineLacks {
                engine: "cerebras",
                lacks: PortMask::of(&[Port::AttnMask])
            })
        );
    }

    fn a_program_only_host_class_is_named_and_counted() {
        for envelope in [
            Ok(None),
            Err("its slots follow a sampled token".to_string()),
        ] {
            // On any engine, one with device geometry among them.
            let bound = classify(program(None, envelope), CUDA, "cuda").expect("binds");
            assert_eq!(bound.class, GeometryClass::Host);
            let Some(host) = bound.host else {
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
        let before = fires();
        count_fire();
        assert!(fires() > before);
    }

    fn device_classes_and_host_derivable_name_nothing() {
        let bound = classify(program(Some(vec![0, 2]), Ok(None)), CUDA, "cuda").expect("binds");
        assert_eq!(bound.class, GeometryClass::DeviceGeometry);
        assert_eq!(bound.devgeo, Some(vec![0, 2]));
        assert_eq!(bound.host, None);

        // The envelope runs ahead of device geometry where it covers.
        let envelope = Ok(Some(PortMask::DECODE_ENVELOPE));
        let bound =
            classify(program(Some(vec![0, 1]), envelope), DECODE_ONLY, "stub").expect("binds");
        assert_eq!(bound.class, GeometryClass::DecodeEnvelope);
        assert_eq!(bound.host, None);

        let mut derivable = program(None, Ok(None));
        derivable.host_derivable = true;
        let bound = classify(derivable, DECODE_ONLY, "stub").expect("binds");
        assert_eq!((bound.class, bound.host), (GeometryClass::Host, None));

        // A channel-fed mask on an engine that reads it binds as device geometry.
        let mut masked = program(Some(vec![0, 1]), Ok(None));
        masked.needs_mask_port = true;
        let bound = classify(masked, CUDA, "vulkan").expect("binds");
        assert_eq!(bound.class, GeometryClass::DeviceGeometry);
    }
}
