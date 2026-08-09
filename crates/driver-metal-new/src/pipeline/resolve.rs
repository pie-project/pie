//! Reading a fire's geometry out of the channels the last fire wrote it into.
//!
//! # What device-resolved geometry is
//!
//! Ordinarily the host decides how many tokens a fire has, which KV pages each
//! request may address, and where the results are read out, and it sends those
//! numbers down with the fire. A program with device-resolved geometry does
//! not: the previous fire's epilogue *computed* them and put them on channels,
//! and the numbers never left the device. This reads those channels before the
//! forward and fills in the same [`Geometry`] the host would have sent, so
//! everything downstream runs on the ordinary batch machinery and needs to
//! know nothing about the program that produced them.
//!
//! # This is a copier, not a policy
//!
//! Nothing here decides anything. It is a port-to-field copy applying exactly
//! two fixed contracts, both of which belong to the port definitions rather
//! than to any program:
//!
//! * **CSR prefix.** A data port and its `indptr` come as a pair, the channel
//!   shapes are fixed, and the program packs the live entries at the front.
//!   The last element of the `indptr` is therefore the valid length of the
//!   data, and everything past it is whatever was there last time.
//! * **`kv_len` to a last-page length.** `((len - 1) % page) + 1`, which is
//!   the port's own semantics.
//!
//! Any beam search, speculative decode or run-ahead logic lives in the program
//! that wrote the channels. If a rule here ever needs to know which program it
//! is serving, the rule is in the wrong place.
//!
//! # Reading here does not consume
//!
//! Every read is a peek at the committed front of the ring. The actual take
//! for the ports that consume happens once, later, in the interpreter's own
//! port loop, which is unchanged. Two takes would drop a cell, and the symptom
//! would be a fire silently reading the fire-after-next's tokens.
//!
//! # Empty is not malformed
//!
//! An empty descriptor channel is [`Resolution::NotReady`] and a wrongly typed
//! one is [`Resolution::Failed`], and they are different because the remedies
//! are opposite. Not-ready means the producer has not run yet and waiting
//! fixes it. Failed means waiting never will. The C++ returns one status enum
//! for this reason and it is preserved; what is dropped is the out-parameter
//! `std::string* err` beside it, because a status that carries no reason and a
//! reason that arrives by side channel are the two halves of one value.

use driver_abi::plan::{LaunchPackage, LaunchPort};
use tensor_ir::registry::Port;

use super::channel::InterpInstance;
use super::plan::{ConstPortValue, ExecPlan};
use super::value::Value;

/// The geometry of one fire.
///
/// Every field is what the host would otherwise have sent. A field left empty
/// means the program did not bind the port that fills it, and the default
/// stated on that field applies.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Geometry {
    /// The token ids to embed, flat across all requests.
    pub token_ids: Vec<u32>,
    /// Row offsets splitting `token_ids` per request; one longer than the
    /// request count. Defaults to one request over all tokens.
    pub qo_indptr: Vec<u32>,
    /// Each token's position in its sequence. Defaults to `0..tokens`.
    pub position_ids: Vec<u32>,
    /// Row offsets splitting `kv_page_indices` per request.
    pub kv_page_indptr: Vec<u32>,
    /// The KV pages each request may address, trimmed to the CSR prefix.
    pub kv_page_indices: Vec<u32>,
    /// Per request, how much of its last KV page is live.
    pub kv_last_page_lens: Vec<u32>,
    /// Whether the KV family was bound at all.
    pub has_kv_family: bool,
    /// Which token rows the epilogue reads out.
    pub sampling_indices: Vec<u32>,
    /// Row offsets splitting `sampling_indices` per request.
    pub sampling_indptr: Vec<u32>,
    /// The adapter slot each token writes to.
    pub w_page: Vec<u32>,
    /// Each token's offset within that slot.
    pub w_off: Vec<u32>,
    /// Whether an explicit write descriptor was bound.
    pub has_write_desc: bool,
    /// A dense attention mask, one byte per lane.
    pub mask: Vec<u8>,
    /// Whether a mask was bound.
    pub has_mask: bool,
    /// The mask's key extent, derived when the descriptor did not carry one.
    pub mask_key_len: u32,
}

/// What resolving a fire's geometry produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// Every bound port had a cell, and here is the geometry.
    Ready(Box<Geometry>),
    /// A channel is empty because its producer has not run yet. Waiting is
    /// the remedy; the channel that was empty is named so a caller can watch
    /// the right one.
    NotReady {
        /// The channel with nothing in it.
        channel: u32,
    },
    /// A channel will never produce something usable. Waiting is not the
    /// remedy.
    Failed {
        /// What was wrong.
        message: String,
    },
}

/// Reinterpret a cell's lanes as `u32`.
///
/// A bit reinterpretation, not a numeric conversion, which matters for
/// `i32`: a negative id keeps its two's-complement pattern rather than
/// saturating or panicking. Token ids, positions and page ids are
/// non-negative in practice, and the CUDA driver reaches the same bytes with
/// a `memcpy`, so anything else here would make the two backends disagree on
/// a value neither of them should ever see.
fn as_u32(value: &Value) -> Option<Vec<u32>> {
    match value {
        Value::U32(lanes) => Some(lanes.clone()),
        Value::I32(lanes) => Some(lanes.iter().map(|&lane| lane as u32).collect()),
        _ => None,
    }
}

/// Where a port's value comes from for this fire.
enum Source<'a> {
    /// A channel, read at its committed front.
    Channel(u32),
    /// A constant folded at adoption time.
    Const(&'a Value),
}

/// How many descriptor ports there are.
///
/// Taken from the tagged enum's own list rather than written as a number, so
/// a port added to `tensor-ir` widens the table here instead of indexing past
/// the end of it.
const PORTS: usize = Port::ALL.len();

/// The bound ports of a package, indexed by port.
struct Bindings<'a> {
    sources: [Option<Source<'a>>; PORTS],
}

impl<'a> Bindings<'a> {
    /// Index the package's port list once, rather than scanning it per port.
    ///
    /// # Errors
    ///
    /// [`Resolution::Failed`] if a port declares itself const and no folded
    /// value was found for it -- which means adoption and resolution disagree
    /// about what the package contains.
    fn of(package: &'a LaunchPackage, constants: &'a [ConstPortValue]) -> Result<Self, Resolution> {
        let mut sources: [Option<Source<'a>>; PORTS] = [const { None }; PORTS];
        for binding in &package.ports {
            let Some(port) = Port::from_u8(binding.port) else {
                // A port this build does not know is not an error: a newer
                // host may bind ports a resolver of this vintage has no field
                // for, and the fields it does have are still correct.
                continue;
            };
            sources[port as usize] = Some(if binding.is_const {
                Source::Const(constant_for(binding, constants)?)
            } else {
                Source::Channel(binding.channel)
            });
        }
        Ok(Self { sources })
    }

    fn get(&self, port: Port) -> Option<&Source<'a>> {
        self.sources[port as usize].as_ref()
    }

    fn bound(&self, port: Port) -> bool {
        self.sources[port as usize].is_some()
    }
}

fn constant_for<'a>(
    binding: &LaunchPort,
    constants: &'a [ConstPortValue],
) -> Result<&'a Value, Resolution> {
    constants
        .iter()
        .find(|folded| folded.port == binding.port)
        .map(|folded| &folded.value)
        .ok_or_else(|| Resolution::Failed {
            message: format!(
                "port {} is declared const but no folded value was kept for it",
                binding.port
            ),
        })
}

/// Peek a port's cell as `u32` lanes.
fn read_u32(
    bindings: &Bindings<'_>,
    instance: &InterpInstance,
    port: Port,
) -> Result<Vec<u32>, Resolution> {
    match bindings.get(port) {
        Some(Source::Channel(channel)) => {
            let ring =
                instance
                    .channels
                    .get(*channel as usize)
                    .ok_or_else(|| Resolution::Failed {
                        message: format!(
                            "port {port:?} names channel {channel}, which is not bound"
                        ),
                    })?;
            if ring.is_empty() {
                return Err(Resolution::NotReady { channel: *channel });
            }
            as_u32(&ring.front()).ok_or_else(|| Resolution::Failed {
                message: format!(
                    "the cell on channel {channel} for port {port:?} is not an integer type, \
                     so it cannot be a geometry index"
                ),
            })
        }
        Some(Source::Const(value)) => as_u32(value).ok_or_else(|| Resolution::Failed {
            message: format!("the folded constant for port {port:?} is not an integer type"),
        }),
        None => Ok(Vec::new()),
    }
}

/// Peek the mask port's cell as unpacked bytes.
fn read_mask(bindings: &Bindings<'_>, instance: &InterpInstance) -> Result<Vec<u8>, Resolution> {
    let port = Port::AttnMask;
    let value = match bindings.get(port) {
        Some(Source::Channel(channel)) => {
            let ring =
                instance
                    .channels
                    .get(*channel as usize)
                    .ok_or_else(|| Resolution::Failed {
                        message: format!(
                            "the mask port names channel {channel}, which is not bound"
                        ),
                    })?;
            if ring.is_empty() {
                return Err(Resolution::NotReady { channel: *channel });
            }
            ring.front()
        }
        Some(Source::Const(value)) => (*value).clone(),
        None => return Ok(Vec::new()),
    };
    match value {
        Value::Bool(lanes) => Ok(lanes),
        other => Err(Resolution::Failed {
            message: format!(
                "the attention mask cell is {:?}, not bool; a dense mask is one byte per lane",
                other.dtype()
            ),
        }),
    }
}

/// How much of the last KV page is live, given a total length.
///
/// A length that is an exact multiple of the page size fills its last page
/// rather than starting an empty one, which is what the `- 1` and `+ 1` are
/// for. A length of zero has no last page; the caller does not ask.
#[must_use]
pub const fn last_page_len(len: u32, page: u32) -> u32 {
    if page == 0 || len == 0 {
        return 0;
    }
    ((len - 1) % page) + 1
}

/// Read a device-resolved program's descriptor channels into a [`Geometry`].
///
/// `page` is the KV page size, used only for the `kv_len` contract.
#[must_use]
pub fn resolve(plan: &ExecPlan, instance: &InterpInstance, page: u32) -> Resolution {
    match resolve_inner(plan, instance, page) {
        Ok(geometry) => Resolution::Ready(Box::new(geometry)),
        Err(resolution) => resolution,
    }
}

#[allow(clippy::too_many_lines)]
fn resolve_inner(
    plan: &ExecPlan,
    instance: &InterpInstance,
    page: u32,
) -> Result<Geometry, Resolution> {
    let bindings = Bindings::of(&plan.package, &plan.const_ports)?;
    // -- the token family --
    let mut out = Geometry {
        token_ids: read_u32(&bindings, instance, Port::EmbedTokens)?,
        ..Geometry::default()
    };
    let tokens = u32::try_from(out.token_ids.len()).unwrap_or(u32::MAX);

    out.qo_indptr = if bindings.bound(Port::EmbedIndptr) {
        read_u32(&bindings, instance, Port::EmbedIndptr)?
    } else {
        // One request over every token: the default a program that does not
        // split its batch is asking for.
        vec![0, tokens]
    };
    let requests = out.qo_indptr.len().saturating_sub(1);

    out.position_ids = if bindings.bound(Port::Positions) {
        read_u32(&bindings, instance, Port::Positions)?
    } else {
        (0..tokens).collect()
    };

    // -- the KV family, under the CSR-prefix contract --
    out.kv_page_indptr = read_u32(&bindings, instance, Port::PageIndptr)?;
    if bindings.bound(Port::Pages) {
        out.kv_page_indices = read_u32(&bindings, instance, Port::Pages)?;
        // The channel's shape is fixed and the program packs live entries at
        // the front, so everything past the indptr's last entry is the
        // previous fire's pages. Reading them would attend over another
        // request's KV -- plausible output, wrong answer.
        if let Some(&live) = out.kv_page_indptr.last() {
            let live = live as usize;
            if live <= out.kv_page_indices.len() {
                out.kv_page_indices.truncate(live);
            }
        }
        out.has_kv_family = true;
    }
    if bindings.bound(Port::KvLen) {
        out.kv_last_page_lens = read_u32(&bindings, instance, Port::KvLen)?
            .into_iter()
            .map(|len| last_page_len(len, page))
            .collect();
    }

    // -- read-out --
    out.sampling_indptr.push(0);
    if bindings.bound(Port::Readout) {
        out.sampling_indices = read_u32(&bindings, instance, Port::Readout)?;
        if requests <= 1 {
            out.sampling_indptr
                .push(u32::try_from(out.sampling_indices.len()).unwrap_or(u32::MAX));
        } else if out.sampling_indices.len() == requests {
            for lane in 1..=requests {
                out.sampling_indptr
                    .push(u32::try_from(lane).unwrap_or(u32::MAX));
            }
        } else {
            return Err(Resolution::Failed {
                message: format!(
                    "the readout port carries {} index(es) for {requests} requests; a \
                     multi-request readout needs exactly one per request",
                    out.sampling_indices.len()
                ),
            });
        }
    } else {
        // No readout port means the last row of each request, which is what
        // a decode step wants and is why the port is usually absent.
        for lane in 0..requests {
            let span = out.qo_indptr[lane + 1].saturating_sub(out.qo_indptr[lane]);
            if span != 0 {
                out.sampling_indices.push(span - 1);
            }
            out.sampling_indptr
                .push(u32::try_from(out.sampling_indices.len()).unwrap_or(u32::MAX));
        }
    }

    // -- the explicit KV write descriptor --
    if bindings.bound(Port::WSlot) {
        out.w_page = read_u32(&bindings, instance, Port::WSlot)?;
        out.has_write_desc = true;
    }
    if bindings.bound(Port::WOff) {
        out.w_off = read_u32(&bindings, instance, Port::WOff)?;
    }

    // -- the dense attention mask --
    if bindings.bound(Port::AttnMask) {
        out.mask = read_mask(&bindings, instance)?;
        out.has_mask = true;
        // The key extent is derivable when the mask divides evenly by the
        // token count, and is left at zero when it does not rather than
        // guessed: a wrong stride reads the mask transposed, which masks the
        // wrong half of the sequence and still produces text.
        if !out.token_ids.is_empty() && out.mask.len().is_multiple_of(out.token_ids.len()) {
            out.mask_key_len =
                u32::try_from(out.mask.len() / out.token_ids.len()).unwrap_or(u32::MAX);
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use driver_abi::plan::{LaunchChannel, LaunchStage, LaunchStagePlan};
    use tensor_ir::registry::Stage;

    use super::super::channel::ChannelState;
    use super::super::plan::adopt_launch_package;
    use super::*;

    fn port(port: Port, channel: u32) -> LaunchPort {
        LaunchPort {
            port: port as u8,
            is_const: false,
            const_dtype: 0,
            channel,
            const_shape: vec![],
            const_data: vec![],
        }
    }

    fn channel(id: u32, lanes: u32, dtype: u8) -> LaunchChannel {
        LaunchChannel {
            id,
            capacity: 1,
            dtype,
            flags: 0,
            extern_dir: -1,
            readiness: 0,
            shape: vec![lanes],
            extern_name: vec![],
        }
    }

    /// A plan with `ports` bound to channels, and an instance whose rings are
    /// seeded from `cells` (index by channel).
    fn fixture(ports: Vec<LaunchPort>, cells: &[Option<Value>]) -> (ExecPlan, InterpInstance) {
        let channels: Vec<LaunchChannel> = cells
            .iter()
            .enumerate()
            .map(|(id, cell)| {
                let (lanes, dtype) = match cell {
                    Some(Value::Bool(lanes)) => (lanes.len() as u32, 3),
                    Some(Value::U32(lanes)) => (lanes.len() as u32, 2),
                    Some(Value::I32(lanes)) => (lanes.len() as u32, 1),
                    Some(Value::F32(lanes)) => (lanes.len() as u32, 0),
                    None => (1, 2),
                };
                channel(id as u32, lanes.max(1), dtype)
            })
            .collect();
        let package = LaunchPackage {
            values: vec![],
            channels: channels.clone(),
            ports,
            names: vec![],
            stages: vec![LaunchStage {
                kind: Stage::Epilogue as u8,
                ops: vec![],
                puts: vec![],
                takes: vec![],
                reads: vec![],
            }],
            plans: vec![LaunchStagePlan::default()],
        };
        let plan = adopt_launch_package(package).expect("well-formed");
        let rings: Vec<Rc<ChannelState>> = channels
            .iter()
            .zip(cells)
            .map(|(declared, cell)| {
                let dtype = super::super::value::concrete_dtype(declared.dtype);
                let lanes = declared.shape.first().copied().unwrap_or(1) as usize;
                let ring = Rc::new(ChannelState::host(dtype, lanes, 1));
                if let Some(value) = cell {
                    assert!(
                        ring.push(value),
                        "the fixture's ring would not take its cell"
                    );
                }
                ring
            })
            .collect();
        let instance = super::super::channel::make_instance(&plan, rings);
        (plan, instance)
    }

    fn ready(resolution: Resolution) -> Geometry {
        match resolution {
            Resolution::Ready(geometry) => *geometry,
            other => panic!("expected a resolved geometry, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_descriptor_channel_is_not_ready_rather_than_malformed() {
        let (plan, instance) = fixture(vec![port(Port::EmbedTokens, 0)], &[None]);
        assert_eq!(
            resolve(&plan, &instance, 16),
            Resolution::NotReady { channel: 0 },
            "an empty channel means the producer has not run yet, and waiting \
             is the remedy; reporting it as malformed sends the caller to fail \
             the fire instead"
        );
    }

    #[test]
    fn a_wrongly_typed_descriptor_channel_fails_because_waiting_will_not_help() {
        let (plan, instance) = fixture(
            vec![port(Port::EmbedTokens, 0)],
            &[Some(Value::F32(vec![1.0, 2.0]))],
        );
        assert!(
            matches!(resolve(&plan, &instance, 16), Resolution::Failed { .. }),
            "a float cell was accepted as token ids"
        );
    }

    #[test]
    fn resolving_does_not_consume_the_cell_it_reads() {
        let (plan, instance) = fixture(
            vec![port(Port::EmbedTokens, 0)],
            &[Some(Value::U32(vec![7, 8, 9]))],
        );
        let first = ready(resolve(&plan, &instance, 16));
        let second = ready(resolve(&plan, &instance, 16));
        assert_eq!(
            first, second,
            "the second resolve saw something different, so the first one took \
             the cell; the interpreter's own port loop would then read the \
             fire-after-next's tokens"
        );
        assert!(
            !instance.channels[0].is_empty(),
            "the ring was drained by a peek"
        );
    }

    #[test]
    fn the_page_list_is_trimmed_to_the_csr_prefix_not_to_its_channel_shape() {
        // A four-slot page channel with two live entries; the last two are
        // whatever the previous fire left there.
        let (plan, instance) = fixture(
            vec![port(Port::Pages, 0), port(Port::PageIndptr, 1)],
            &[
                Some(Value::U32(vec![11, 12, 999, 999])),
                Some(Value::U32(vec![0, 2])),
            ],
        );
        let geometry = ready(resolve(&plan, &instance, 16));
        assert_eq!(
            geometry.kv_page_indices,
            vec![11, 12],
            "the stale tail of the fixed-shape page channel was kept, so the \
             fire attends over pages belonging to a previous request"
        );
        assert!(geometry.has_kv_family);
    }

    #[test]
    fn a_kv_length_that_fills_its_page_exactly_reports_a_full_page_not_an_empty_one() {
        let (plan, instance) = fixture(
            vec![port(Port::KvLen, 0)],
            &[Some(Value::U32(vec![16, 17, 1, 32]))],
        );
        let geometry = ready(resolve(&plan, &instance, 16));
        assert_eq!(
            geometry.kv_last_page_lens,
            vec![16, 1, 1, 16],
            "a length that is an exact multiple of the page size must fill its \
             last page rather than start an empty one; reporting zero there \
             makes the attention read nothing for that request"
        );
    }

    #[test]
    fn an_absent_readout_port_reads_out_the_last_row_of_each_request() {
        let (plan, instance) = fixture(
            vec![port(Port::EmbedTokens, 0), port(Port::EmbedIndptr, 1)],
            &[
                Some(Value::U32(vec![1, 2, 3, 4, 5])),
                Some(Value::U32(vec![0, 3, 5])),
            ],
        );
        let geometry = ready(resolve(&plan, &instance, 16));
        assert_eq!(
            geometry.sampling_indices,
            vec![2, 1],
            "the last row of a three-token request is index 2 and of a \
             two-token request is index 1, both relative to their own request"
        );
        assert_eq!(geometry.sampling_indptr, vec![0, 1, 2]);
    }

    #[test]
    fn a_multi_request_readout_needs_one_index_per_request() {
        let (plan, instance) = fixture(
            vec![
                port(Port::EmbedTokens, 0),
                port(Port::EmbedIndptr, 1),
                port(Port::Readout, 2),
            ],
            &[
                Some(Value::U32(vec![1, 2, 3, 4, 5])),
                Some(Value::U32(vec![0, 3, 5])),
                Some(Value::U32(vec![2])),
            ],
        );
        assert!(
            matches!(resolve(&plan, &instance, 16), Resolution::Failed { .. }),
            "one readout index for two requests was accepted, so the second \
             request's logits come from the first request's row"
        );
    }

    #[test]
    fn the_defaults_stand_in_for_ports_the_program_did_not_bind() {
        let (plan, instance) = fixture(
            vec![port(Port::EmbedTokens, 0)],
            &[Some(Value::U32(vec![5, 6, 7]))],
        );
        let geometry = ready(resolve(&plan, &instance, 16));
        assert_eq!(
            geometry.qo_indptr,
            vec![0, 3],
            "an unbound indptr means one request over all the tokens"
        );
        assert_eq!(
            geometry.position_ids,
            vec![0, 1, 2],
            "unbound positions means each token sits at its own index"
        );
        assert!(!geometry.has_kv_family && !geometry.has_write_desc && !geometry.has_mask);
    }

    #[test]
    fn a_mask_key_extent_is_derived_only_when_it_divides_evenly() {
        let even = fixture(
            vec![port(Port::EmbedTokens, 0), port(Port::AttnMask, 1)],
            &[
                Some(Value::U32(vec![1, 2])),
                Some(Value::Bool(vec![1, 1, 0, 1, 1, 0])),
            ],
        );
        let geometry = ready(resolve(&even.0, &even.1, 16));
        assert_eq!(
            geometry.mask_key_len, 3,
            "six lanes over two tokens is three keys"
        );

        let odd = fixture(
            vec![port(Port::EmbedTokens, 0), port(Port::AttnMask, 1)],
            &[
                Some(Value::U32(vec![1, 2])),
                Some(Value::Bool(vec![1, 1, 0, 1, 1])),
            ],
        );
        let geometry = ready(resolve(&odd.0, &odd.1, 16));
        assert_eq!(
            geometry.mask_key_len, 0,
            "five lanes do not divide by two tokens, and guessing a stride \
             reads the mask transposed -- which masks the wrong half of the \
             sequence and still produces text"
        );
        assert!(
            geometry.has_mask,
            "the mask is still carried; only its stride is unknown"
        );
    }

    #[test]
    fn an_i32_index_keeps_its_bit_pattern_rather_than_saturating() {
        let (plan, instance) = fixture(
            vec![port(Port::EmbedTokens, 0)],
            &[Some(Value::I32(vec![-1, 3]))],
        );
        let geometry = ready(resolve(&plan, &instance, 16));
        assert_eq!(
            geometry.token_ids,
            vec![u32::MAX, 3],
            "the CUDA driver reaches these bytes with a memcpy, so anything \
             but a bit reinterpretation makes the two backends disagree"
        );
    }

    #[test]
    fn a_port_this_build_does_not_know_is_skipped_rather_than_refused() {
        let (plan, instance) = fixture(
            vec![
                port(Port::EmbedTokens, 0),
                LaunchPort {
                    port: 200,
                    is_const: false,
                    const_dtype: 0,
                    channel: 1,
                    const_shape: vec![],
                    const_data: vec![],
                },
            ],
            &[Some(Value::U32(vec![1, 2])), None],
        );
        let geometry = ready(resolve(&plan, &instance, 16));
        assert_eq!(
            geometry.token_ids,
            vec![1, 2],
            "a newer host bound a port this build has no field for, and the \
             whole fire was refused rather than the fields this build does \
             know being filled"
        );
    }
}
