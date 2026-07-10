//! PTIR instance construction (thrust-3 P2.2) — the host-side `instantiate`
//! logic behind the WIT `pipeline.instantiate`.
//!
//! An **instance** is one binding of a registered program (the trace identity)
//! to its per-instance state: the seed values for its `seeded` channels
//! (`Channel::from` data — D2, never part of registration/identity) and the set
//! of host-facing channels the guest can drive. The device channel arena is
//! allocated driver-side from the declared shapes; the host instance carries the
//! validated seeds to send at instantiation plus the host-channel index map.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use pie_ptir::container::{self, HostRole};

use super::ptir_registry::{RegisteredProgram, lookup};

/// Process-wide monotonic source of instance identities (0 reserved as a null /
/// "no instance" sentinel). Each `instantiate` mints a fresh id: the driver
/// caches one channel arena per id, so distinct instances of the same program
/// (same `hash`) hold independent, persistent channel state.
static NEXT_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

/// Mint the next process-wide instance identity.
pub fn next_instance_id() -> u64 {
    NEXT_INSTANCE_ID.fetch_add(1, Ordering::Relaxed)
}

/// A per-instance channel seed value, by dense channel index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSeed {
    pub channel: u32,
    pub data: Vec<u8>,
}

/// A constructed instance: the registered program + its validated seeds.
#[derive(Debug)]
pub struct PtirInstance {
    pub program: Arc<RegisteredProgram>,
    /// This instance's identity — the driver's channel-arena cache key (stable
    /// across all its fires). Seeds bind on its first fire; the arena persists.
    pub instance_id: u64,
    /// Validated seeds, one per `seeded` channel, in channel order.
    pub seeds: Vec<ChannelSeed>,
}

impl PtirInstance {
    /// The dense indices of host-facing channels (`host-role != none`) — the
    /// only channels the guest may obtain a host endpoint on.
    pub fn host_channels(&self) -> Vec<u32> {
        self.program
            .bound
            .container
            .channels
            .iter()
            .enumerate()
            .filter(|(_, c)| c.host_role != HostRole::None)
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// The host role of channel `index`, or `None` if the index is out of range.
    pub fn host_role(&self, index: u32) -> Option<HostRole> {
        self.program
            .bound
            .container
            .channels
            .get(index as usize)
            .map(|c| c.host_role)
    }

    /// Assemble the host-known per-channel values for a fire's geometry map:
    /// seeded channels carry their seed value; every other channel is
    /// host-unknown (device-derived, working-set, or not-yet-produced) → `None`,
    /// and the driver fills those descriptor ports itself. Sufficient to
    /// host-prefill a single-sequence decode whose token/geometry ports bind
    /// seeded channels (§3).
    pub fn channel_values(&self) -> Vec<Option<Vec<u8>>> {
        let mut v = vec![None; self.program.bound.container.channels.len()];
        for s in &self.seeds {
            v[s.channel as usize] = Some(s.data.clone());
        }
        v
    }

    /// Map this instance's descriptor ports → the fire's [`ReqGeometry`] from the
    /// host-known channel values (thrust-3 P2c-fire). Errors
    /// [`GeometryError::MissingChannelValue`] if a port binds a channel whose
    /// value isn't host-known — the signal that the geometry needs driver / ws /
    /// run-ahead resolution rather than a pure host prefill.
    pub fn fire_geometry(
        &self,
        page_size: u32,
    ) -> Result<super::ptir_geometry::ReqGeometry, super::ptir_geometry::GeometryError> {
        super::ptir_geometry::map_geometry(
            &self.program.bound.container,
            &self.channel_values(),
            page_size,
        )
    }

    /// **Relaxed** geometry map for a device-geometry fire (plan W3.4): a
    /// descriptor port bound to a device-produced channel with no host-known
    /// value leaves its wire field empty (the driver resolves it pre-forward,
    /// W1.1) instead of erroring, while const / host-known ports still prefill.
    /// Never returns `MissingChannelValue`; used to route a device-geometry
    /// pass through the ordinary solo/prebuilt submit.
    pub fn fire_geometry_relaxed(
        &self,
        page_size: u32,
    ) -> Result<super::ptir_geometry::ReqGeometry, super::ptir_geometry::GeometryError> {
        super::ptir_geometry::map_geometry_relaxed(
            &self.program.bound.container,
            &self.channel_values(),
            page_size,
        )
    }

    /// Build this instance's PTIR carrier submission for a fire (thrust-3 P2c).
    /// `first_fire` = the instance's first fire: it ships the container bytes +
    /// PTIB sidecar (so the driver decodes/compiles + caches by hash) AND the
    /// per-instance seeds (so the driver builds + binds this instance's channel
    /// arena, keyed by `instance_id`). Steady-state fires (`first_fire = false`)
    /// carry the hash + instance id only — the driver serves the compiled
    /// program from cache and drives the already-built arena.
    /// `channel_ids` is the dense-index → global-channel-id map (one entry per
    /// declared channel, from the pass's bound cells); it re-keys the seeds and
    /// rides every fire so the driver binds trace-local channels to the global
    /// device registry. `host_puts` are the D1-coalesced host `channel.put`s
    /// (already global-id keyed) staged for this fire.
    pub fn submission(
        &self,
        first_fire: bool,
        channel_ids: Vec<u64>,
        host_puts: Vec<crate::ptir::PtirChannelValue>,
    ) -> crate::ptir::PtirProgramSubmission {
        let global = |dense: u32| -> u64 {
            channel_ids
                .get(dense as usize)
                .copied()
                .unwrap_or(dense as u64)
        };
        crate::ptir::PtirProgramSubmission {
            hash: self.program.hash,
            instance: self.instance_id,
            bytes: first_fire.then(|| self.program.bytes.clone()),
            sidecar: first_fire.then(|| self.program.sidecar.clone()),
            seeds: if first_fire {
                self.seeds
                    .iter()
                    .map(|s| crate::ptir::PtirChannelValue {
                        channel: global(s.channel),
                        bytes: s.data.clone(),
                    })
                    .collect()
            } else {
                Vec::new()
            },
            channel_ids,
            host_puts,
        }
    }
}

/// An instantiation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstantiateError {
    /// No program with this identity hash is registered.
    UnknownProgram(u64),
    /// A seed targets a channel that was not declared `seeded`.
    SeedForNonSeeded { channel: u32 },
    /// A seed targets a channel index outside the container.
    SeedChannelOutOfRange { channel: u32 },
    /// Two seeds target the same channel.
    DuplicateSeed { channel: u32 },
    /// A declared `seeded` channel has no seed value.
    MissingSeed { channel: u32 },
    /// A seed's byte length does not match its channel's shape×dtype.
    SeedShapeMismatch {
        channel: u32,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for InstantiateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InstantiateError::*;
        match self {
            UnknownProgram(h) => write!(f, "no registered program with identity {h:#018x}"),
            SeedForNonSeeded { channel } => {
                write!(
                    f,
                    "channel {channel}: a seed was supplied but the channel is not seeded"
                )
            }
            SeedChannelOutOfRange { channel } => write!(f, "seed channel {channel} out of range"),
            DuplicateSeed { channel } => write!(f, "channel {channel}: duplicate seed"),
            MissingSeed { channel } => write!(f, "channel {channel}: seeded but no seed supplied"),
            SeedShapeMismatch {
                channel,
                expected,
                got,
            } => write!(
                f,
                "channel {channel}: seed is {got} bytes, expected {expected} (shape×dtype)"
            ),
        }
    }
}
impl std::error::Error for InstantiateError {}

/// Instantiate a registered program with per-channel seeds (looks the program up
/// in the process-wide registry). Validates that exactly the `seeded` channels
/// are seeded, each with the right byte length. Seeds are per-instance data (D2)
/// — never part of the program identity.
pub fn instantiate(
    program: u64,
    seeds: Vec<ChannelSeed>,
) -> Result<PtirInstance, InstantiateError> {
    let prog = lookup(program).ok_or(InstantiateError::UnknownProgram(program))?;
    let validated = validate_seeds(&prog, seeds)?;
    Ok(PtirInstance {
        program: prog,
        instance_id: next_instance_id(),
        seeds: validated,
    })
}

/// Validate + order the seeds against a program's channel declarations.
pub fn validate_seeds(
    prog: &RegisteredProgram,
    seeds: Vec<ChannelSeed>,
) -> Result<Vec<ChannelSeed>, InstantiateError> {
    let channels = &prog.bound.container.channels;

    // Index the supplied seeds; reject out-of-range, non-seeded, and duplicates.
    let mut by_channel: Vec<Option<Vec<u8>>> = vec![None; channels.len()];
    for s in seeds {
        let idx = s.channel as usize;
        let decl = channels
            .get(idx)
            .ok_or(InstantiateError::SeedChannelOutOfRange { channel: s.channel })?;
        if !decl.seeded {
            return Err(InstantiateError::SeedForNonSeeded { channel: s.channel });
        }
        if by_channel[idx].is_some() {
            return Err(InstantiateError::DuplicateSeed { channel: s.channel });
        }
        let expected =
            decl.shape.numel() as usize * container::const_elem_size(decl.dtype.program_dtype());
        if s.data.len() != expected {
            return Err(InstantiateError::SeedShapeMismatch {
                channel: s.channel,
                expected,
                got: s.data.len(),
            });
        }
        by_channel[idx] = Some(s.data);
    }

    // Every seeded channel must have a seed; assemble in channel order.
    let mut out = Vec::new();
    for (i, decl) in channels.iter().enumerate() {
        match (decl.seeded, by_channel[i].take()) {
            (true, Some(data)) => out.push(ChannelSeed {
                channel: i as u32,
                data,
            }),
            (true, None) => return Err(InstantiateError::MissingSeed { channel: i as u32 }),
            (false, _) => {}
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptir::ptir_registry::{PtirRegistry, register};
    use pie_ptir::container::{
        ChanDType, ChannelDecl, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use pie_ptir::op::{IntrinsicId, Op};
    use pie_ptir::registry::{ModelProfile, Port, Stage};
    use pie_ptir::types::{DType, Shape};
    use std::num::NonZeroUsize;

    const VOCAB: u32 = 32;

    fn chan(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded,
        }
    }

    /// tok (i32 [1], seeded, device) + out (i32 [1], host-reader).
    fn greedy() -> TraceContainer {
        let ops = vec![
            Op::IntrinsicVal {
                intr: IntrinsicId::Logits,
                shape: Shape::matrix(1, VOCAB),
                dtype: DType::F32,
            },
            Op::Reshape {
                value: 0,
                shape: Shape::vector(VOCAB),
            },
            Op::ReduceArgmax(1),
            Op::Reshape {
                value: 2,
                shape: Shape::vector(1),
            },
            Op::ChanPut { chan: 1, value: 3 },
        ];
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::I32, HostRole::None, true),
                chan(Shape::vector(1), DType::I32, HostRole::Reader, false),
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].iter().flat_map(|w| w.to_le_bytes()).collect(),
                    },
                },
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
        }
    }

    fn registered() -> Arc<RegisteredProgram> {
        let mut r = PtirRegistry::new(NonZeroUsize::new(8).unwrap());
        r.register(
            greedy().encode(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap()
    }

    fn seed(ch: u32, val: i32) -> ChannelSeed {
        ChannelSeed {
            channel: ch,
            data: val.to_le_bytes().to_vec(),
        }
    }

    #[test]
    fn valid_seeds_construct_instance() {
        let prog = registered();
        let inst = PtirInstance {
            program: prog.clone(),
            instance_id: 1,
            seeds: validate_seeds(&prog, vec![seed(0, 1)]).unwrap(),
        };
        assert_eq!(inst.seeds, vec![seed(0, 1)]);
        assert_eq!(inst.host_channels(), vec![1], "only `out` is host-facing");
        assert_eq!(inst.host_role(0), Some(HostRole::None));
        assert_eq!(inst.host_role(1), Some(HostRole::Reader));
    }

    #[test]
    fn missing_seed_for_seeded_channel_fails() {
        let prog = registered();
        let e = validate_seeds(&prog, vec![]).unwrap_err();
        assert_eq!(e, InstantiateError::MissingSeed { channel: 0 });
    }

    #[test]
    fn seed_for_non_seeded_channel_fails() {
        let prog = registered();
        let e = validate_seeds(&prog, vec![seed(0, 1), seed(1, 9)]).unwrap_err();
        assert_eq!(e, InstantiateError::SeedForNonSeeded { channel: 1 });
    }

    #[test]
    fn wrong_seed_length_fails() {
        let prog = registered();
        let bad = ChannelSeed {
            channel: 0,
            data: vec![1, 2],
        }; // 2 bytes, need 4
        let e = validate_seeds(&prog, vec![bad]).unwrap_err();
        assert_eq!(
            e,
            InstantiateError::SeedShapeMismatch {
                channel: 0,
                expected: 4,
                got: 2
            }
        );
    }

    #[test]
    fn duplicate_and_out_of_range_seeds_fail() {
        let prog = registered();
        assert_eq!(
            validate_seeds(&prog, vec![seed(0, 1), seed(0, 2)]).unwrap_err(),
            InstantiateError::DuplicateSeed { channel: 0 }
        );
        assert_eq!(
            validate_seeds(&prog, vec![seed(0, 1), seed(9, 2)]).unwrap_err(),
            InstantiateError::SeedChannelOutOfRange { channel: 9 }
        );
    }

    #[test]
    fn instantiate_unknown_program_fails() {
        let e = instantiate(0xdead_beef, vec![]).unwrap_err();
        assert_eq!(e, InstantiateError::UnknownProgram(0xdead_beef));
    }

    #[test]
    fn instantiate_via_process_registry_roundtrips() {
        let bytes = greedy().encode();
        let prog = register(
            bytes.clone(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap();
        let inst = instantiate(prog.hash, vec![seed(0, 42)]).unwrap();
        assert_eq!(inst.program.hash, prog.hash);
        assert_eq!(inst.seeds, vec![seed(0, 42)]);
    }

    #[test]
    fn fire_geometry_prefills_request_from_seed() {
        // §3-shaped greedy: embed_tokens ← the seeded `tok` channel, embed_indptr
        // const [0,1]. The host-known geometry (token + qo_indptr + default
        // positions/readout) prefills a LaunchPlan — no driver/ws needed.
        let prog = register(
            greedy().encode(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap();
        let inst = instantiate(prog.hash, vec![seed(0, 42)]).unwrap();

        let g = inst.fire_geometry(16).unwrap();
        assert_eq!(g.token_ids, vec![42], "the seeded token embeds");
        assert_eq!(g.qo_indptr, vec![0, 1], "one lane, one token");
        assert_eq!(g.position_ids, vec![0]);
        assert_eq!(g.sampling_indices, vec![0], "read out the only token");

        let mut req = crate::driver::LaunchPlan::default();
        g.apply_to(&mut req);
        assert_eq!(req.token_ids, vec![42]);
        assert_eq!(req.qo_indptr, vec![0, 1]);
        assert_eq!(req.sampling_indices, vec![0]);
    }

    /// P2/P3 exit gate: register → instantiate → run on echo's reference
    /// interpreter (the mock driver) → golden token. Proves the runtime
    /// register/instantiate/submit-round-trip path end to end against the same
    /// oracle the CUDA backend diffs against — the greedy program argmaxes the
    /// supplied logits and publishes the token to the host-reader `out` channel.
    #[test]
    fn submission_carrier_first_fire_vs_steady_state() {
        let prog = register(
            greedy().encode(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap();
        let inst = instantiate(prog.hash, vec![seed(0, 42)]).unwrap();
        assert_ne!(inst.instance_id, 0, "a fresh instance identity is minted");

        // First fire: ships the container bytes + sidecar + the seed (D2), all
        // tagged with this instance's identity.
        let first = inst.submission(true, vec![], vec![]);
        assert_eq!(first.hash, prog.hash);
        assert_eq!(
            first.instance, inst.instance_id,
            "carries the instance identity"
        );
        assert_eq!(
            first.bytes.as_deref(),
            Some(prog.bytes.as_slice()),
            "first fire ships bytes"
        );
        assert_eq!(
            first.sidecar.as_deref(),
            Some(prog.sidecar.as_slice()),
            "first fire ships sidecar"
        );
        assert_eq!(first.seeds.len(), 1);
        assert_eq!(first.seeds[0].channel, 0);
        assert_eq!(first.seeds[0].bytes, 42i32.to_le_bytes().to_vec());

        // Steady state: hash-cache hit + persistent arena — no bytes/sidecar/seeds,
        // but the instance identity still routes the fire to its arena.
        let steady = inst.submission(false, vec![], vec![]);
        assert_eq!(steady.hash, prog.hash);
        assert_eq!(
            steady.instance, inst.instance_id,
            "steady fire still routes by instance"
        );
        assert!(steady.bytes.is_none(), "steady-state fire is hash-only");
        assert!(steady.sidecar.is_none(), "steady-state ships no sidecar");
        assert!(
            steady.seeds.is_empty(),
            "seeds bind once at the instance's first fire"
        );

        // Two instances of the SAME program get DISTINCT identities (independent
        // channel arenas) though they share one compiled `hash`.
        let inst2 = instantiate(prog.hash, vec![seed(0, 7)]).unwrap();
        assert_eq!(
            inst2.program.hash, inst.program.hash,
            "same compiled program"
        );
        assert_ne!(
            inst2.instance_id, inst.instance_id,
            "distinct persistent instances"
        );
    }

    #[test]
    fn register_instantiate_run_on_mock_interp() {
        use pie_ptir::interp::Value;
        use pie_ptir::interp::{Instance as Interp, NoKernels, PassInputs};

        let prog = register(
            greedy().encode(),
            &ModelProfile {
                vocab: VOCAB,
                ..ModelProfile::dummy()
            },
        )
        .unwrap();
        // Instance construction (seed validation) — the WIT `instantiate` core.
        let inst = instantiate(prog.hash, vec![seed(0, 1)]).unwrap();
        assert_eq!(inst.host_channels(), vec![1], "only `out` is host-facing");

        // Run the bound trace on the mock driver (echo's interp) with the same
        // per-instance seeds the runtime would ship.
        let mut mock = Interp::new(&prog.bound, &[(0, Value::I32(vec![1]))]).unwrap();
        let out_i = prog
            .bound
            .container
            .channels
            .iter()
            .position(|c| c.host_role == HostRole::Reader)
            .unwrap() as u32;

        let mut l = vec![0.0f32; VOCAB as usize];
        l[5] = 9.0; // argmax at index 5
        let r = mock
            .step(
                &prog.bound,
                &PassInputs {
                    logits: Some(Value::F32(l)),
                    ..Default::default()
                },
                &mut NoKernels,
            )
            .unwrap();
        assert!(r.committed, "the pass commits");
        assert_eq!(
            mock.host_take(&prog.bound, out_i).unwrap(),
            Value::I32(vec![5]),
            "greedy token = argmax = 5"
        );
    }
}
