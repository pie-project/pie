//! **THE LORA SINK, READ** (alto adapter §6.1, §6.3, §6.4) — the half of the
//! adapter axis that is pure arithmetic over a launch package and a seed.
//!
//! # The sentence this module exists to finish
//!
//! The correction has been served on this plane since milestone 1: an entry
//! (`kernels_metal::linear::lora::correct`), a dispatch arm
//! (`Linear::LoraCorrect`), a routes seat (`crate::serve`), a bank
//! ([`crate::weights::Weights::register_adapter`]) and two fire refusals
//! ([`Fault::Adapterless`](crate::Fault), `Fault::AdapterWord`). What was
//! missing was the door a GUEST comes through. `ModelProfile::has_lora`
//! answered `false`, `eta_ir::validate` refused any program carrying a `lora`
//! sink at bind, and so the one axis with a kernel under it was reachable
//! only from a control plane calling `Engine::register_adapter` by id.
//!
//! This module is the join, and it is deliberately three small questions
//! rather than one verb:
//!
//! ```text
//! sink_of(package)         which channels does the `lora` sink name?
//! planes_of(sink, seeds)   what does one seed's f32 cell mean as bank bytes?
//! Slots::acquire           which slot, and who gives it back
//! ```
//!
//! # Why the CHANNELS are read here and never at fire time
//!
//! §6.1's ruling: a 12 MiB channel cell is legal but the machinery re-pays it
//! EVERY FIRE — the cell is materialised into per-lane scratch and dragged
//! over the shared ring on every launch. So an adapter channel is a NAMING
//! device. The bytes are taken off the seed ONCE, at instance bind, converted
//! into the banks' own dtype and landed; the cell is never read again.
//! Everything in this file therefore runs between fires, on the host, with no
//! device in sight — which is what makes every claim it makes testable
//! without a GPU, and it is why the deviceless pins at the foot of this file
//! are the axis's arithmetic gate.
//!
//! # The second arm, and the day it arrived
//!
//! This module used to say "THERE IS NO SHARED-ADAPTER MOUNT HERE", and end
//! with: "the day this plane mounts a directory is the day [`Source`] grows
//! an arm, and the residency table underneath it already takes a key rather
//! than an instance id so that day costs it nothing." That day is this one,
//! the promise held, and the table below is unchanged except that its [`Key`]
//! has a second spelling.
//!
//! The two arms are the two things an adapter can BE, and they are keyed
//! differently because they are owned differently:
//!
//! ```text
//! Source::Own    { instance, planes }  a guest's own bytes, off a channel.
//!                                      Keyed by INSTANCE: private, unshared,
//!                                      released when the instance closes.
//! Source::Shared { name }              a file in the deployment's mount.
//!                                      Keyed by BLOB IDENTITY ([`crate::blob::Stamp`]):
//!                                      N instances naming it land on ONE slot
//!                                      and the device sees ONE copy.
//! ```
//!
//! Everything about the FILES — the mount, the `adapter.toml` grammar, the
//! single-flight host byte cache and the per-layer resolver — is
//! [`crate::blob`], next door. What stays here is what was always here: the
//! sink, the seed arithmetic, and the residency.
//!
//! # What is refused, by name
//!
//! * the SCALE form (`adapter_scale`, IA3/DoRA's two-argument spelling): the
//!   eta wire selects the form by sink arity and `model-ir` declares no
//!   `AdapterScale` op, so a shell that accepted it would seat weights no
//!   text can read.
//! * a sink argument that is not a channel read — the closed language builds
//!   its operands from `chan_read`, so anything else is a program this
//!   resolver did not compile against.
//! * a seed whose element count is not `layers x rank x hidden` for the banks
//!   of its role, both numbers named.
//! * a rank wider than the bank seats, both numbers named.
//! * a channel the sink names and the bind seeded nothing into: the weights
//!   would be a cell of zeros, which is the IDENTITY adapter, which is a
//!   silently wrong answer rather than a loud one.
//! * a site the guest asked for and this load's banks do not declare.

use std::collections::BTreeMap;

use eta_compiler::codegen::launch::{LaunchPackage, ValueOrigin};
use eta_ir::op::tags;

use crate::error::{Fault, Result};
use crate::weights::{AdapterPlane, BankSeat};

/// The sink's name in the package's name table. First-party, and the same
/// string `eta-ir`'s registry and `eta-compiler`'s `SINK_CALL` arm agree on.
pub const LORA: &str = "lora";

/// **WHICH PROJECTION AN ADAPTER CORRECTS.**
///
/// **THE SPELLING IS THE CONTRACT AND IT IS WRITTEN IN FOUR PLACES** — the
/// guest's `Site`, the model text's `models::adapter::Site`, the CUDA
/// resolver's `engine_cuda::blob::Site` and this one — for the reason
/// `lora_a` is: a name crosses a crate boundary as a string, and the crates
/// cannot depend on each other in a circle. What keeps them one vocabulary is
/// that a spelling nobody here knows is REFUSED rather than guessed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The mixer's output projection — the site every family text corrects
    /// today, and therefore the one an untagged bank means.
    O,
    /// The fused gate/up projection of the feed-forward sublayer.
    GateUp,
    /// Its down projection.
    Down,
}

impl Site {
    /// The vocabulary, in bit order.
    pub const ALL: [Site; 6] = [Site::Q, Site::K, Site::V, Site::O, Site::GateUp, Site::Down];

    /// How a bank name and a manifest spell it.
    #[must_use]
    pub const fn spelled(self) -> &'static str {
        match self {
            Site::Q => "q",
            Site::K => "k",
            Site::V => "v",
            Site::O => "o",
            Site::GateUp => "gate_up",
            Site::Down => "down",
        }
    }

    /// The guest surface's own bit for it — `inferlet::eta::adapter::Site`'s
    /// `bit()`, which is what rides the `lora` sink's placement constant.
    #[must_use]
    pub const fn bit(self) -> u32 {
        match self {
            Site::Q => 1 << 0,
            Site::K => 1 << 1,
            Site::V => 1 << 2,
            Site::O => 1 << 3,
            Site::GateUp => 1 << 4,
            Site::Down => 1 << 5,
        }
    }

    /// A spelling, or `None` for a word outside the vocabulary.
    #[must_use]
    pub fn parse(word: &str) -> Option<Site> {
        Site::ALL.into_iter().find(|site| site.spelled() == word)
    }

    /// The vocabulary as a message names it.
    #[must_use]
    pub fn vocabulary() -> String {
        Site::ALL
            .iter()
            .map(|site| format!("`{}`", site.spelled()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// How a message spells "at this site", including the absent one.
    #[must_use]
    pub fn stated(site: Option<Site>) -> String {
        match site {
            Some(site) => format!("at site `{}`", site.spelled()),
            None => "at no stated site".to_string(),
        }
    }
}

/// The bank name's role — everything after a `layer.{l}.` prefix and the
/// OPTIONAL site segment that may follow it.
///
/// `layer.7.lora_a` is the seventh layer's `lora_a` and states no site;
/// `layer.7.o.lora_a` is the same role at the same layer, declaring that it
/// corrects [`Site::O`]. A bank named without a numbered component is its own
/// role at layer zero, and a middle segment outside the site vocabulary is
/// NOT a site — `layer.3.mixer.lora_a` is a role called `layer.3.mixer.lora_a`,
/// which is how an unknown site goes on being refused rather than landed at
/// whatever the text's default happens to be.
#[must_use]
pub fn role_of(bank: &str) -> &str {
    parsed(bank).map_or(bank, |(_, _, role)| role)
}

/// Which layer a bank name puts itself at, or zero for an unnumbered one.
#[must_use]
pub fn layer_of(bank: &str) -> u64 {
    parsed(bank).map_or(0, |(layer, _, _)| layer)
}

/// **WHICH SITE A BANK DECLARES IT CORRECTS**, or `None` for a name that
/// declares none.
///
/// `None` is not "no site": it is TODAY'S MEANING — the text's own default
/// site, the one every family text corrects, unstated because a name had no
/// way to state it. What it buys is the byte-compatible half: a load whose
/// banks all answer `None` behaves exactly as it did.
#[must_use]
pub fn site_of(bank: &str) -> Option<Site> {
    parsed(bank).and_then(|(_, site, _)| site)
}

/// `layer.{l}[.{site}].{role}`, read once — the whole grammar in one place.
fn parsed(bank: &str) -> Option<(u64, Option<Site>, &str)> {
    let (head, role) = bank.rsplit_once('.')?;
    let last = head.rsplit('.').next()?;
    // `layer.{l}.{role}` — the spelling the family texts write. Read first,
    // so nothing about the site widening costs it a step.
    if let Some(layer) = numbered(last) {
        return Some((layer, None, role));
    }
    // `layer.{l}.{site}.{role}` — a site of the vocabulary, with the layer
    // right in front of it. Anything else falls through to `None` and the
    // caller reads the whole name as a role.
    let site = Site::parse(last)?;
    let (rest, _) = head.rsplit_once('.')?;
    let layer = numbered(rest.rsplit('.').next()?)?;
    Some((layer, Some(site), role))
}

fn numbered(part: &str) -> Option<u64> {
    match !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()) {
        true => part.parse().ok(),
        false => None,
    }
}

/// Which plane of an adapter a sink argument is.
///
/// The ROLE is positional in the closed language — `lora(a, b, sites)` — and
/// it is spelled as the bank-name suffix [`role_of`] answers, because that is
/// the one place the convention already lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// `A`: `[layers, rank, hidden]`, rank-major.
    A,
    /// `B`: `[layers, hidden, rank]`, out-major — HF's native orientation,
    /// and §6.3's statute.
    B,
}

impl Role {
    /// The bank-name suffix this role fills.
    #[must_use]
    pub const fn bank(self) -> &'static str {
        match self {
            Role::A => "lora_a",
            Role::B => "lora_b",
        }
    }
}

/// One program's `lora` sink, as the resolver reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sink {
    /// Which stage carries it. Always a prologue in the surface's own
    /// lowering (`Pass::adapter` emits into `prologue`), recorded rather than
    /// assumed because the plan is what says so.
    pub stage: usize,
    /// The dense channel index each plane's weights are seeded into, in role
    /// order.
    pub planes: Vec<(Role, u32)>,
    /// The trace-known placement constant — the site bits the guest asked
    /// for, [`Site::bit`]'s own numbering.
    pub sites: u32,
}

impl Sink {
    /// **WHICH SITE THIS GUEST ASKED FOR**, or `None` for a sink that named
    /// no placement at all.
    ///
    /// One sink corrects ONE site: `Pass::adapter(site, …)` emits one `lora`
    /// per site with `Tensor::constant(site.bit())` beside it, so exactly one
    /// bit is the shape this resolver compiled against.
    ///
    /// # Errors
    ///
    /// [`Fault::Adapter`] for a placement constant that is not one site of
    /// the vocabulary, refused with the bits it carried rather than rounded
    /// to the nearest site.
    pub fn site(&self) -> Result<Option<Site>> {
        match self.sites {
            0 => Ok(None),
            bits => Site::ALL
                .into_iter()
                .find(|site| site.bit() == bits)
                .map(Some)
                .ok_or_else(|| Fault::Adapter {
                    bank: LORA.to_string(),
                    why: format!(
                        "carries placement constant {bits:#08b} and one sink corrects \
                         ONE site of {}: the surface emits one `lora` per \
                         `Pass::adapter` call with that site's single bit beside it, so \
                         a constant naming none of them — or several — is a program \
                         this shell did not compile against",
                        Site::vocabulary()
                    ),
                }),
        }
    }
}

/// **DOES THIS PROGRAM CARRY AN ADAPTER, AND WHICH CHANNELS ARE ITS
/// WEIGHTS?** (§6.4: the plan says WHETHER.)
///
/// `Ok(None)` is the ordinary answer — no stage of this package declares the
/// sink, so nothing about the adapter axis touches this program at all.
///
/// # Errors
///
/// [`Fault::Adapter`] for a sink this shell cannot serve: the scale form, an
/// argument that is not a channel read, or a sink whose arity is neither of
/// the closed language's two.
pub fn sink_of(package: &LaunchPackage) -> Result<Option<Sink>> {
    // The PLAN's own answer first, because it is the fact §6.4 names and
    // because it is one bool per stage against a walk of every op.
    let Some(stage) = package.plans.iter().position(|plan| plan.needs.lora) else {
        return Ok(None);
    };
    let refuse = |why: String| Fault::Adapter {
        bank: LORA.to_string(),
        why,
    };
    let body = package.stages.get(stage).ok_or_else(|| {
        refuse(format!(
            "is declared by stage {stage}'s plan and this package has {} stage bodies; \
             the plans and the bodies are parallel by construction, so a package where \
             they are not is one this shell cannot read",
            package.stages.len()
        ))
    })?;
    let call = body
        .ops
        .iter()
        .find(|op| {
            op.tag == tags::SINK_CALL
                && package.names.get(op.name_index as usize).map(String::as_str) == Some(LORA)
        })
        .ok_or_else(|| {
            refuse(format!(
                "is what stage {stage}'s plan declares it needs, and no `sink_call` in \
                 that stage's body names it"
            ))
        })?;
    // **ARITY SELECTS THE FORM** (§2's item 3, and the surface says so at
    // `inferlet::eta::adapter`): three arguments is `lora(a, b, sites)`, two
    // is `adapter_scale(l, sites)`. The last argument is the trace-known
    // placement constant in both.
    let (weights, sites_arg) = match call.args.as_slice() {
        [a, b, sites] => (vec![(Role::A, *a), (Role::B, *b)], *sites),
        [_, _] => {
            return Err(refuse(
                "is the SCALE form (`adapter_scale`, two arguments): IA3 and DoRA's \
                 second half multiply an already-materialised output by a per-channel \
                 vector, and `model-ir` declares no `AdapterScale` op for a bank to be \
                 read by. This shell seats the low-rank form only, so accepting the \
                 scale would land weights nothing reads — refused rather than ignored"
                    .to_string(),
            ));
        }
        args => {
            return Err(refuse(format!(
                "takes {} arguments and the closed language has two spellings: \
                 `lora(a, b, sites)` and `adapter_scale(l, sites)`",
                args.len()
            )));
        }
    };
    let mut planes = Vec::with_capacity(weights.len());
    for (role, value) in weights {
        let source = package
            .values
            .iter()
            .find(|declared| declared.id == value)
            .ok_or_else(|| {
                refuse(format!(
                    "names value {value} as its `{}` plane and the package's value \
                     table does not declare it",
                    role.bank()
                ))
            })?;
        // A READ, not a take. The surface peeks (`a.read()`), which is what
        // keeps the adapter off the decode chain's edge list; a `chan_take`
        // would consume the cell and the second fire of the pass would find
        // an empty ring. Both are accepted here — the bytes are the seed's
        // either way and this resolver never reads the ring again — and
        // anything that is not a channel at all is refused.
        if !matches!(
            source.source,
            ValueOrigin::ChannelRead | ValueOrigin::ChannelTake
        ) {
            return Err(refuse(format!(
                "names a computed value as its `{}` plane, and an adapter's weights \
                 are channel contents: the closed language builds the sink's operands \
                 out of `chan_read`, so a value with any other origin is a program \
                 this shell did not compile against",
                role.bank()
            )));
        }
        planes.push((role, source.channel));
    }
    let sites = package
        .values
        .iter()
        .find(|declared| declared.id == sites_arg)
        .filter(|declared| declared.source == ValueOrigin::Const)
        .map_or(0, |declared| declared.literal_bits);
    Ok(Some(Sink {
        stage,
        planes,
        sites,
    }))
}

/// **ONE SEEDED CELL, AS THE BANKS WANT IT** (§6.3).
///
/// `wire` is the f32 cell the guest seeded — 4 bytes an element, little
/// endian, which is `eta_exec::wire_cell_bytes`'s encoding for every non-bool
/// dtype. `seats` is the whole load's bank table; the banks that carry `role`
/// are found in it, sorted by layer, and the source is cut into one
/// full-capacity plane each.
///
/// **THE CONVERSION IS THE POINT.** A channel's lane dtypes are F32/I32/U32/
/// Bool and a bank's is the model text's (bf16 at every qwen SKU), so the
/// bytes cannot simply be copied: they are rounded to the bank's own element
/// here, once, on the host, at bind.
///
/// **AND THE SITE IS CHECKED HERE.** `site` is what the guest asked for and
/// the banks it lands in are the ones that declare THAT site. A load whose
/// banks declare no site at all takes today's meaning: the text's own default
/// site, whatever the guest named, because there is no fact to check against
/// and inventing one would refuse every program that works.
///
/// # Errors
///
/// [`Fault::Adapter`] for a role this load declares no bank for, a site its
/// banks do not declare, banks of one role that are not one shape, a cell
/// whose length is not `layers x rank x hidden` f32 elements, or a rank the
/// bank cannot seat.
pub fn planes_of(
    role: Role,
    site: Option<Site>,
    wire: &[u8],
    seats: &[BankSeat],
) -> Result<Vec<(String, Vec<u8>)>> {
    let refuse = |why: String| Fault::Adapter {
        bank: role.bank().to_string(),
        why,
    };
    let of_role: Vec<&BankSeat> = seats
        .iter()
        .filter(|seat| role_of(&seat.name) == role.bank())
        .collect();
    // **ABSENT IS A VALUE, NOT A WILDCARD — EXCEPT WHERE IT IS THE ONLY
    //   VALUE.** A load whose every bank of this role declares no site is a
    //   text written before the site tag existed; its one correction site is
    //   the text's own and there is nothing to compare a guest's ask with, so
    //   the ask is carried unchecked. The moment ONE bank names a site, the
    //   load has an opinion and the ask has to match it.
    let sited = of_role.iter().any(|seat| site_of(&seat.name).is_some());
    let want = match sited {
        true => site,
        false => None,
    };
    let mut banks: Vec<&BankSeat> = of_role
        .iter()
        .copied()
        .filter(|seat| site_of(&seat.name) == want)
        .collect();
    banks.sort_by_key(|seat| layer_of(&seat.name));
    let seat = *banks.first().ok_or_else(|| {
        let declared = {
            let mut seen: Vec<String> = Vec::new();
            for seat in &of_role {
                let spelled = match site_of(&seat.name) {
                    Some(site) => format!("`{}`", site.spelled()),
                    None => "the text's own unstated one".to_string(),
                };
                if !seen.contains(&spelled) {
                    seen.push(spelled);
                }
            }
            seen.join(", ")
        };
        match of_role.is_empty() {
            true => refuse(format!(
                "is a plane of the `lora` sink and this load declares no bank by that \
                 role; its banks are {:?}",
                seats.iter().map(|seat| &seat.name).collect::<Vec<_>>()
            )),
            false => refuse(format!(
                "is a plane of the `lora` sink asked for {}, and this load's banks of \
                 that role correct {} — so the correction the guest asked for is not \
                 one this model text carries. A site nobody declared is refused rather \
                 than served from whichever site the text does correct, because an \
                 adapter that answered the wrong projection would answer it silently",
                Site::stated(want),
                declared
            )),
        }
    })?;
    if let Some(odd) = banks
        .iter()
        .find(|bank| bank.slot != seat.slot || bank.rows != seat.rows)
    {
        return Err(refuse(format!(
            "is filled across {} banks and they are not one shape: `{}` seats {} bytes \
             and `{}` seats {}; a `[layers, ...]` source is L contiguous slices of ONE \
             rectangle",
            banks.len(),
            seat.name,
            seat.slot,
            odd.name,
            odd.slot
        )));
    }
    if seat.elem != 2 {
        return Err(refuse(format!(
            "seats {}-byte elements in `{}` and this resolver rounds an f32 cell to a \
             2-byte bf16; a bank of another width needs its own conversion rather than \
             this one silently writing the wrong stride",
            seat.elem, seat.name
        )));
    }
    // The bank's rectangle says which way it runs: an `A` is `[rank, hidden]`
    // (rank leading, the smaller axis first) and a `B` is `[hidden, rank]`.
    // The same statute the CUDA resolver takes: the source has to arrive in
    // the bank's orientation, because the alternative is a repack kernel this
    // shell does not ship.
    let bank_rank = seat.rows.min(seat.cols);
    let hidden = seat.rows.max(seat.cols);
    let layers = banks.len() as u64;
    let elems = wire.len() as u64 / 4;
    if wire.len() as u64 % 4 != 0 || elems % (layers.saturating_mul(hidden)).max(1) != 0 {
        return Err(refuse(format!(
            "was seeded {} bytes, which is not {layers} layers x rank x {hidden} f32 \
             elements; a plane is a whole `[layers, rank, hidden]` cell",
            wire.len()
        )));
    }
    let rank = elems / layers / hidden;
    if rank == 0 || rank > bank_rank {
        return Err(refuse(format!(
            "was seeded at rank {rank} and bank `{}` seats rank {bank_rank}; the bank's \
             capacity is a shape the model text declared, so the fix is a bank that \
             seats it and not a retry",
            seat.name
        )));
    }
    let slot = usize::try_from(seat.slot).unwrap_or(usize::MAX);
    let hidden = usize::try_from(hidden).unwrap_or(usize::MAX);
    let rank = usize::try_from(rank).unwrap_or(usize::MAX);
    let bank_rank = usize::try_from(bank_rank).unwrap_or(usize::MAX);
    let stride = rank * hidden;
    let mut out = Vec::with_capacity(banks.len());
    for (layer, bank) in banks.iter().enumerate() {
        let source = &wire[layer * stride * 4..(layer + 1) * stride * 4];
        // **ZERO-PADDED PER ORIENTATION**, exactly as [`AdapterPlane`]'s own
        // doc says: `A`'s unused ranks are trailing ROWS and `B`'s are a
        // stride inside every row. A zero row of `A` contributes zero to the
        // waist and a zero column of `B` contributes zero to the sum, so both
        // paddings are exact.
        let mut plane = vec![0u8; slot];
        match role {
            Role::A => {
                for at in 0..stride {
                    let value = f32_at(source, at);
                    plane[at * 2..at * 2 + 2].copy_from_slice(&bf16_bits(value).to_le_bytes());
                }
            }
            Role::B => {
                for row in 0..hidden {
                    for at in 0..rank {
                        let value = f32_at(source, row * rank + at);
                        let to = (row * bank_rank + at) * 2;
                        plane[to..to + 2].copy_from_slice(&bf16_bits(value).to_le_bytes());
                    }
                }
            }
        }
        out.push((bank.name.clone(), plane));
    }
    Ok(out)
}

/// One f32 out of a wire cell.
fn f32_at(wire: &[u8], at: usize) -> f32 {
    let bytes = [
        wire[at * 4],
        wire[at * 4 + 1],
        wire[at * 4 + 2],
        wire[at * 4 + 3],
    ];
    f32::from_le_bytes(bytes)
}

/// f32 to bf16, ROUND TO NEAREST EVEN.
///
/// The same conversion the weight loader performs, and stated rather than
/// truncated: a truncating resolver would land a slightly different adapter
/// than the one the guest described, and every parity claim below it would be
/// about the wrong numbers.
#[must_use]
pub fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// Where one bind's bytes came from.
#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    /// An instance's own full-capacity planes — the private-adapter path, and
    /// the shape [`crate::weights::Weights::register_adapter`] already took.
    Own {
        /// Which instance. Its slot is its own and is never shared.
        instance: u64,
        /// The planes, exactly as the existing verb takes them.
        planes: &'a [AdapterPlane<'a>],
    },
    /// **A DIRECTORY IN THE DEPLOYMENT'S MOUNT** (alto adapter §3.3), named
    /// as the guest spells it. Keyed by its [`crate::blob::Stamp`], so every
    /// instance that names it lands on one slot and pays one copy.
    Shared {
        /// The adapter's name, resolved against
        /// [`crate::serve::Shell::mount_adapters`]'s root.
        name: &'a str,
    },
}

/// What a bind answers (§6.4).
///
/// **NOT `Copy`, AND THAT IS THE SHARED ARM'S DOING.** A binding carries the
/// [`Key`] it holds, because a release has to give back exactly what was
/// taken and a shared key is a stamp over a `Vec` of files rather than a
/// number. The alternative — releasing by SLOT and searching the table for
/// whoever holds it — would work and would be a second way to name one thing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    /// The slot every lane of this instance routes to — what the fire path
    /// stamps onto `Lane::adapter`.
    pub slot: u32,
    /// Did this bind name a file in the mount?
    pub shared: bool,
    /// Did THIS bind pay the landing, or did it join one already resident?
    ///
    /// The sharing observable: the second instance of one blob answers
    /// `false` with the first one's slot.
    pub landed: bool,
    /// What the slot is held under — [`crate::serve::Shell::release_adapter`]'s
    /// argument, and never anyone else's business.
    pub key: Key,
}

/// The key a slot is held under.
///
/// **TWO SPELLINGS, AND THE DIFFERENCE IS THE WHOLE SHARING CLAIM.** An
/// instance id is private by construction — no two instances are one — and a
/// blob stamp is shared by construction, because two instances that name one
/// file compute one stamp. Which arm a bind takes is decided at
/// [`crate::serve::Shell::bind_adapter`] and nowhere else.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Key {
    /// One instance's private adapter.
    Instance(u64),
    /// One file in the mount, by identity (§3.3: path + the files' stamp).
    Shared(crate::blob::Stamp),
}

/// **THE RESIDENCY TABLE** — which of the banks' slots are pinned, by what.
///
/// Pure host state, deliberately: with the residency apart from the write,
/// every claim about pinning, reuse and exhaustion is checkable on a machine
/// with no GPU in it. The landing arrives as a closure at the call site
/// ([`crate::serve::Shell::bind_adapter`]) for exactly that reason.
#[derive(Debug, Default)]
pub struct Slots {
    /// How many slots the banks seat — the SMALLEST capacity any declared
    /// bank states, because an adapter occupies one slot of every bank.
    seats: u32,
    /// Key -> (slot, how many live binds hold it).
    held: BTreeMap<Key, (u32, u32)>,
}

/// What an acquire answered.
#[derive(Debug, Clone, Copy)]
pub struct Grant {
    /// The slot.
    pub slot: u32,
    /// Did this acquire seat it, or join one already held?
    pub fresh: bool,
}

impl Slots {
    /// A table over `seats` slots, holding none.
    #[must_use]
    pub fn new(seats: u32) -> Slots {
        Slots {
            seats,
            held: BTreeMap::new(),
        }
    }

    /// How many slots the banks seat.
    #[must_use]
    pub fn seats(&self) -> u32 {
        self.seats
    }

    /// How many are pinned right now.
    #[must_use]
    pub fn live(&self) -> usize {
        self.held.len()
    }

    /// **PIN A SLOT FOR `key`.**
    ///
    /// A key already held takes its own slot back and its refcount rises —
    /// which is what makes the shared arm free the day it lands. A fresh key
    /// takes the lowest slot nobody holds.
    ///
    /// # Errors
    ///
    /// [`Fault::AdapterSlots`] when every slot is pinned by a live bind. It
    /// is a refusal and not an eviction because a slot's contents are read by
    /// a fire that may be in flight: evicting under a live bind would answer
    /// somebody else's adapter under this one's name.
    pub fn acquire(&mut self, key: Key) -> Result<Grant> {
        if let Some((slot, count)) = self.held.get_mut(&key) {
            *count += 1;
            return Ok(Grant {
                slot: *slot,
                fresh: false,
            });
        }
        let taken: Vec<u32> = self.held.values().map(|(slot, _)| *slot).collect();
        let slot = (0..self.seats)
            .find(|slot| !taken.contains(slot))
            .ok_or(Fault::AdapterSlots { seats: self.seats })?;
        self.held.insert(key, (slot, 1));
        Ok(Grant { slot, fresh: true })
    }

    /// Undo an acquire that could not be landed.
    ///
    /// **A REFUSED LANDING HOLDS NO SLOT.** Anything that goes wrong after
    /// the slot is seated abandons it, so no key is ever left pointing at
    /// bytes that did not arrive.
    pub fn abandon(&mut self, key: &Key) {
        self.release_key(key);
    }

    /// Give a bind back. At the LAST release the slot is free again.
    ///
    /// **AND ITS CONTENTS ARE THEN NOBODY'S**, which is where this table and
    /// the CUDA twin's part company on purpose. `engine_cuda::blob::Slots`
    /// keeps a released occupant seated and reclaims LRU under pressure, so a
    /// shared adapter with intermittent traffic does not re-pay its H2D; this
    /// one frees the seat at zero and a re-bind re-lands. The trade is
    /// deliberate: on this plane the landing is a memcpy into a slot that is
    /// already resident in unified memory — there is no transfer to amortise
    /// — and a table that frees at zero cannot refuse a bind while holding
    /// adapters nobody is using. When that stops being true the LRU arm goes
    /// here, and nothing above it changes.
    pub fn release(&mut self, key: &Key) {
        self.release_key(key);
    }

    fn release_key(&mut self, key: &Key) {
        let drop = match self.held.get_mut(key) {
            Some((_, count)) => {
                *count = count.saturating_sub(1);
                *count == 0
            }
            None => false,
        };
        if drop {
            self.held.remove(key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eta_compiler::codegen::launch::{LaunchOp, LaunchStage, LaunchStagePlan, LaunchValue};

    fn seat(name: &str, rows: u64, cols: u64) -> BankSeat {
        BankSeat {
            name: name.to_string(),
            adapters: 8,
            slot: rows * cols * 2,
            rows,
            cols,
            elem: 2,
        }
    }

    /// Two layers of an `A` bank at rank 2, hidden 3.
    fn a_seats() -> Vec<BankSeat> {
        vec![seat("layer.0.lora_a", 2, 3), seat("layer.1.lora_a", 2, 3)]
    }

    fn b_seats() -> Vec<BankSeat> {
        vec![seat("layer.0.lora_b", 3, 2), seat("layer.1.lora_b", 3, 2)]
    }

    fn wire(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bf16(value: f32) -> [u8; 2] {
        bf16_bits(value).to_le_bytes()
    }

    /// Read a bf16 plane back as f32, so a pin can state the ARITHMETIC
    /// rather than a byte pattern.
    fn as_f32(plane: &[u8]) -> Vec<f32> {
        plane
            .chunks_exact(2)
            .map(|two| f32::from_bits(u32::from(u16::from_le_bytes([two[0], two[1]])) << 16))
            .collect()
    }

    /// **THE SLICE IS PER LAYER** (§6.3): a `[layers, rank, hidden]` cell is L
    /// contiguous rectangles and the L banks of the role take one each, in
    /// layer order.
    #[test]
    fn a_layered_cell_becomes_one_plane_per_layer_bank() {
        let cell = wire(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let planes = planes_of(Role::A, None, &cell, &a_seats()).expect("a full-rank A");
        assert_eq!(planes.len(), 2, "one plane per layer bank");
        assert_eq!(planes[0].0, "layer.0.lora_a");
        assert_eq!(planes[1].0, "layer.1.lora_a");
        assert_eq!(&planes[0].1[..2], &bf16(1.0), "layer 0 starts the cell");
        assert_eq!(&planes[1].1[..2], &bf16(7.0), "layer 1 starts halfway");
        assert_eq!(planes[0].1.len(), 12, "a plane is one whole slot");
    }

    /// **A SHORT RANK PADS WHERE ITS ORIENTATION SAYS** — `A`'s unused ranks
    /// are trailing rows and `B`'s are a stride inside every row. The two are
    /// different placements of the same zeros, and getting them the same way
    /// round is the whole of the orientation statute.
    #[test]
    fn a_short_rank_pads_where_its_orientation_says() {
        // Rank 1 into a rank-2 bank: A's second ROW is zero.
        let cell = wire(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let planes = planes_of(Role::A, None, &cell, &a_seats()).expect("a rank-1 A");
        assert_eq!(&planes[0].1[..6], [bf16(1.0), bf16(2.0), bf16(3.0)].concat());
        assert_eq!(&planes[0].1[6..], &[0u8; 6], "the unused rank is a zero row");
        // Rank 1 into a rank-2 B: every ROW's second column is zero.
        let planes = planes_of(Role::B, None, &cell, &b_seats()).expect("a rank-1 B");
        assert_eq!(&planes[0].1[..2], &bf16(1.0));
        assert_eq!(&planes[0].1[2..4], &[0u8; 2], "the unused rank is a stride");
        assert_eq!(&planes[0].1[4..6], &bf16(2.0));
    }

    /// **THE RANK-r CORRECTION, AS ARITHMETIC** — the deviceless pin the
    /// device gate is checked against.
    ///
    /// This is what `linear/lora.metal` computes, in f32, over the planes
    /// this resolver lands: `y += B·(A·x)`. Stated here rather than only on
    /// the device because the landing's ORIENTATION is the half that can be
    /// silently wrong — an `A` read as `[hidden, rank]` and a `B` read as
    /// `[rank, hidden]` both produce numbers, and only one of them is the
    /// adapter the guest described.
    #[test]
    fn the_correction_is_b_times_a_times_x_at_the_landed_orientation() {
        // rank 2, hidden 3, one layer.
        let seats_a = vec![seat("layer.0.lora_a", 2, 3)];
        let seats_b = vec![seat("layer.0.lora_b", 3, 2)];
        // A is [rank, hidden] = [[1,2,3],[4,5,6]]
        let a_cell = wire(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // B is [hidden, rank] = [[1,0],[0,1],[1,1]]
        let b_cell = wire(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let a = as_f32(&planes_of(Role::A, None, &a_cell, &seats_a).expect("A").remove(0).1);
        let b = as_f32(&planes_of(Role::B, None, &b_cell, &seats_b).expect("B").remove(0).1);
        let x = [1.0f32, 1.0, 1.0];
        // The waist: A·x, rank-major.
        let waist: Vec<f32> = (0..2)
            .map(|r| (0..3).map(|h| a[r * 3 + h] * x[h]).sum::<f32>())
            .collect();
        assert_eq!(waist, vec![6.0, 15.0], "A·x over the rank rows");
        // The correction: B·waist, out-major.
        let delta: Vec<f32> = (0..3)
            .map(|o| (0..2).map(|r| b[o * 2 + r] * waist[r]).sum::<f32>())
            .collect();
        assert_eq!(delta, vec![6.0, 15.0, 21.0], "B·(A·x) over the output rows");
    }

    /// **A ZERO `B` IS EXACTLY THE IDENTITY ADAPTER**, and that is what makes
    /// `lora-probe`'s `adapter_scale = 0.0` a byte-identity claim rather than
    /// a tolerance. The scale folds into `B` at the guest, so every element
    /// of the landed plane is a positive zero and `B(Ax)` adds exactly 0.0.
    #[test]
    fn a_zero_up_bank_lands_exact_zero_and_corrects_by_nothing() {
        let seats_b = vec![seat("layer.0.lora_b", 3, 2)];
        let cell = wire(&[0.0; 6]);
        let plane = planes_of(Role::B, None, &cell, &seats_b).expect("a zero B").remove(0).1;
        assert!(plane.iter().all(|&byte| byte == 0), "a zero cell is zero bytes");
        for value in as_f32(&plane) {
            assert_eq!(value, 0.0);
            assert!(value.is_sign_positive(), "positive zero, so the add is exact");
        }
    }

    /// **THE SCALE RIDES `B` AND THE RESOLVER IS LINEAR IN IT.** The guest
    /// folds `alpha/r` into `B` before it seeds, so a cell scaled by `k`
    /// lands a correction scaled by `k` — checked at a power of two, where
    /// the bf16 rounding is exact and the claim is an identity.
    #[test]
    fn a_scaled_up_bank_scales_the_correction_exactly() {
        let seats_b = vec![seat("layer.0.lora_b", 3, 2)];
        let base = [1.5f32, -2.25, 0.75, 3.0, -0.5, 1.0];
        let one = as_f32(&planes_of(Role::B, None, &wire(&base), &seats_b).expect("B").remove(0).1);
        let scaled: Vec<f32> = base.iter().map(|v| v * 0.25).collect();
        let quarter =
            as_f32(&planes_of(Role::B, None, &wire(&scaled), &seats_b).expect("B").remove(0).1);
        for (at, (whole, part)) in one.iter().zip(&quarter).enumerate() {
            assert_eq!(whole * 0.25, *part, "element {at} scales exactly at a power of two");
        }
    }

    /// **A CELL THAT IS NOT A WHOLE `[layers, rank, hidden]` IS REFUSED WITH
    /// BOTH NUMBERS**, because the alternative is a plane that is a layer out
    /// of step with the bank it lands in.
    #[test]
    fn a_cell_the_banks_cannot_seat_is_refused_by_name() {
        let cell = wire(&[1.0; 7]);
        let why = planes_of(Role::A, None, &cell, &a_seats()).expect_err("7 is not 2 x r x 3");
        let said = why.to_string();
        assert!(said.contains("28"), "names the bytes it was handed: {said}");
        assert!(said.contains('3'), "and the rectangle it wanted: {said}");
        let cell = wire(&[1.0; 18]);
        let why = planes_of(Role::A, None, &cell, &a_seats()).expect_err("rank 3 into rank 2");
        assert!(why.to_string().contains("seats rank 2"), "names both ranks: {why}");
    }

    fn package_with(args: Vec<u32>) -> LaunchPackage {
        let mut package = LaunchPackage {
            names: vec![LORA.to_string()],
            plans: vec![LaunchStagePlan {
                needs: eta_compiler::codegen::launch::StageNeeds {
                    lora: true,
                    ..Default::default()
                },
                ..Default::default()
            }],
            ..LaunchPackage::default()
        };
        package.values = vec![
            LaunchValue {
                id: 0,
                source: ValueOrigin::ChannelRead,
                channel: 4,
                ..LaunchValue::default()
            },
            LaunchValue {
                id: 1,
                source: ValueOrigin::ChannelRead,
                channel: 5,
                ..LaunchValue::default()
            },
            LaunchValue {
                id: 2,
                source: ValueOrigin::Const,
                literal_bits: 0b1000,
                ..LaunchValue::default()
            },
        ];
        package.stages = vec![LaunchStage {
            ops: vec![LaunchOp {
                tag: tags::SINK_CALL,
                name_index: 0,
                args,
                ..LaunchOp::default()
            }],
            ..LaunchStage::default()
        }];
        package
    }

    /// **THE PLAN SAYS WHETHER AND THE BODY SAYS WHICH** (§6.4): the two
    /// channels come back in role order, with the placement constant beside
    /// them.
    #[test]
    fn the_sink_answers_its_two_channels_in_role_order() {
        let sink = sink_of(&package_with(vec![0, 1, 2]))
            .expect("a readable sink")
            .expect("a package that declares one");
        assert_eq!(sink.stage, 0);
        assert_eq!(sink.planes, vec![(Role::A, 4), (Role::B, 5)]);
        assert_eq!(sink.sites, 0b1000, "the trace-known placement constant");
    }

    /// **THE PLACEMENT CONSTANT READS BACK AS ONE SITE**, and a constant
    /// naming none of them is a program outside the closed surface.
    #[test]
    fn the_placement_constant_is_one_site_or_a_refusal() {
        let sink = sink_of(&package_with(vec![0, 1, 2]))
            .expect("a readable sink")
            .expect("a package that declares one");
        assert_eq!(
            sink.site().expect("0b1000 is a site"),
            Some(Site::O),
            "bit 3 is the mixer output, the site every family text corrects"
        );
        let none = Sink {
            stage: 0,
            planes: vec![],
            sites: 0,
        };
        assert_eq!(none.site().expect("no constant is no ask"), None);
        let many = Sink {
            stage: 0,
            planes: vec![],
            sites: 0b1001,
        };
        let why = many.site().expect_err("two sites is not one sink");
        assert!(why.to_string().contains("ONE site"), "{why}");
    }

    /// A package with no sink is the ordinary answer and costs nothing.
    #[test]
    fn a_package_with_no_sink_answers_none() {
        assert_eq!(
            sink_of(&LaunchPackage::default()).expect("no sink is not an error"),
            None
        );
    }

    /// **THE SCALE FORM IS REFUSED BY NAME** (§2's item 3). `model-ir`
    /// declares no `AdapterScale`, so a shell that seated one would be
    /// landing weights no text can read — and `lora-probe`'s `form = "scale"`
    /// is the guest that makes this refusal reachable.
    #[test]
    fn the_scale_form_is_refused_and_says_why() {
        let why = sink_of(&package_with(vec![0, 2])).expect_err("two arguments is the scale");
        let said = why.to_string();
        assert!(said.contains("SCALE"), "{said}");
        assert!(said.contains("AdapterScale"), "names the missing op: {said}");
    }

    /// **A GUEST'S SITE, AGAINST A TEXT THAT NAMES ITS OWN.**
    #[test]
    fn a_site_the_banks_do_not_declare_is_refused_by_name() {
        let sited = vec![seat("layer.0.o.lora_a", 2, 3), seat("layer.1.o.lora_a", 2, 3)];
        let cell = wire(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let planes = planes_of(Role::A, Some(Site::O), &cell, &sited).expect("the named site");
        assert_eq!(planes[0].0, "layer.0.o.lora_a");
        let why = planes_of(Role::A, Some(Site::Q), &cell, &sited).expect_err("q is not o");
        let said = why.to_string();
        assert!(said.contains("`q`"), "names the site asked for: {said}");
        assert!(said.contains("`o`"), "and the site declared: {said}");
    }

    /// **AND AN UNTAGGED LOAD MEANS WHAT IT ALWAYS MEANT** — every family
    /// text names `layer.{l}.lora_a`, so every guest site it is asked for
    /// lands exactly the bytes it landed before the widening.
    #[test]
    fn an_untagged_load_answers_every_site_the_bytes_it_always_did() {
        let cell = wire(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let before = planes_of(Role::A, None, &cell, &a_seats()).expect("the untagged call");
        for site in Site::ALL {
            let after = planes_of(Role::A, Some(site), &cell, &a_seats())
                .unwrap_or_else(|why| panic!("an untagged load serves {site:?}: {why}"));
            assert_eq!(after, before, "{site:?} lands the same bytes in the same banks");
        }
    }

    /// **A SLOT IS PINNED BY A LIVE BIND AND RECLAIMED BY A RELEASE**, and a
    /// table with every slot pinned refuses by name rather than evicting
    /// somebody's adapter out from under a fire.
    #[test]
    fn the_residency_pins_releases_and_refuses_by_name() {
        let mut slots = Slots::new(2);
        let first = slots.acquire(Key::Instance(1)).expect("a free slot");
        assert!(first.fresh);
        let second = slots.acquire(Key::Instance(2)).expect("the other one");
        assert_ne!(first.slot, second.slot, "two instances are two slots");
        let why = slots.acquire(Key::Instance(3)).expect_err("both are pinned");
        assert!(why.to_string().contains('2'), "names the capacity: {why}");
        // The same key comes back to its own slot without seating a new one.
        let again = slots.acquire(Key::Instance(1)).expect("its own slot");
        assert_eq!(again.slot, first.slot);
        assert!(!again.fresh, "a held key does not re-land");
        slots.release(&Key::Instance(1));
        assert_eq!(slots.live(), 2, "the second hold keeps it pinned");
        slots.release(&Key::Instance(1));
        assert_eq!(slots.live(), 1, "the last release frees it");
        let reused = slots.acquire(Key::Instance(3)).expect("the freed slot");
        assert_eq!(reused.slot, first.slot, "the lowest free slot is taken");
    }

    /// One stamp, as the residency table sees it — the shape
    /// [`crate::blob::Store::stamp`] answers, built by hand so this pin needs
    /// no filesystem.
    fn stamp(at: &str, bytes: u64) -> crate::blob::Stamp {
        crate::blob::Stamp {
            at: at.to_string(),
            files: vec![("adapter.toml".to_string(), bytes, 7)],
        }
    }

    /// **N INSTANCES OF ONE BLOB OCCUPY ONE SLOT, AND A PRIVATE ADAPTER IS
    /// NOBODY'S NEIGHBOUR** — the whole sharing claim, as a host table.
    ///
    /// This is what keying by identity buys and it is decided with no device
    /// in the machine: the second and third binds of one stamp answer the
    /// first one's slot with `fresh: false`, so the caller lands nothing; a
    /// byte-seeded instance beside them takes a seat of its own; and a
    /// rewritten file is a different stamp, therefore a different slot.
    #[test]
    fn one_blob_is_one_slot_however_many_instances_name_it() {
        let mut slots = Slots::new(2);
        let alice = Key::Shared(stamp("/mnt/alice-v2", 128));

        let first = slots.acquire(alice.clone()).expect("a free slot");
        assert!(first.fresh, "the first bind is the one that pays");
        let second = slots.acquire(alice.clone()).expect("the same identity");
        assert_eq!(second.slot, first.slot, "one blob, one slot");
        assert!(!second.fresh, "the second joins what is already there");
        let third = slots.acquire(alice.clone()).expect("and a third");
        assert_eq!(third.slot, first.slot);
        assert_eq!(slots.live(), 1, "three binds, one resident identity");

        // A private adapter shares nothing: content-hash dedup across
        // byte-seeded instances is a later optimization, and sharing here
        // would put one tenant's fine-tune under another tenant's rows.
        let private = slots.acquire(Key::Instance(77)).expect("the other slot");
        assert_ne!(private.slot, first.slot);
        assert!(private.fresh);

        // A rewritten file is a new identity, and the table is full.
        let rewritten = Key::Shared(stamp("/mnt/alice-v2", 129));
        let why = slots
            .acquire(rewritten)
            .expect_err("a new identity wants a seat and both are pinned");
        assert!(why.to_string().contains('2'), "names the capacity: {why}");

        // Three binds hold the shared slot, so it takes three releases.
        slots.release(&alice);
        slots.release(&alice);
        assert_eq!(slots.live(), 2, "two holds are gone and one remains");
        slots.release(&alice);
        assert_eq!(slots.live(), 1, "the last release frees the shared slot");
        let after = slots
            .acquire(Key::Shared(stamp("/mnt/bob-v1", 64)))
            .expect("the freed slot seats the next identity");
        assert_eq!(after.slot, first.slot);
    }

    /// **AN ABANDONED ACQUIRE HOLDS NOTHING**, which is what keeps a refused
    /// landing from pinning a slot pointing at bytes that never arrived.
    #[test]
    fn an_abandoned_acquire_leaves_the_slot_free() {
        let mut slots = Slots::new(1);
        let grant = slots.acquire(Key::Instance(7)).expect("the only slot");
        slots.abandon(&Key::Instance(7));
        assert_eq!(slots.live(), 0);
        let after = slots.acquire(Key::Instance(8)).expect("it is free again");
        assert_eq!(after.slot, grant.slot);
    }

    /// The name grammar, including the two ways a middle segment is NOT a
    /// site.
    #[test]
    fn the_bank_name_grammar_reads_role_layer_and_site() {
        assert_eq!(role_of("layer.7.lora_a"), "lora_a");
        assert_eq!(layer_of("layer.7.lora_a"), 7);
        assert_eq!(site_of("layer.7.lora_a"), None);
        assert_eq!(role_of("layer.7.o.lora_b"), "lora_b");
        assert_eq!(layer_of("layer.7.o.lora_b"), 7);
        assert_eq!(site_of("layer.7.o.lora_b"), Some(Site::O));
        // An unknown middle segment is not a site, so the WHOLE name is the
        // role — which is how it goes on matching no bank.
        assert_eq!(role_of("layer.3.mixer.lora_a"), "layer.3.mixer.lora_a");
        assert_eq!(site_of("layer.3.mixer.lora_a"), None);
    }
}
