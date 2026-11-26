use std::collections::BTreeMap;

use engine::program::InstanceId;
use eta_compiler::codegen::launch::{LaunchPackage, ValueOrigin};
use eta_ir::op::tags;

use crate::error::{Fault, Result};
use crate::weights::BankSeat;

pub const LORA: &str = "lora";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    Q,

    K,

    V,

    O,

    GateUp,

    Down,
}

impl Site {
    pub const ALL: [Site; 6] = [Site::Q, Site::K, Site::V, Site::O, Site::GateUp, Site::Down];

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

    #[must_use]
    pub fn parse(word: &str) -> Option<Site> {
        Site::ALL.into_iter().find(|site| site.spelled() == word)
    }

    #[must_use]
    pub fn vocabulary() -> String {
        Site::ALL
            .iter()
            .map(|site| format!("`{}`", site.spelled()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    #[must_use]
    pub fn stated(site: Option<Site>) -> String {
        match site {
            Some(site) => format!("at site `{}`", site.spelled()),
            None => "at no stated site".to_string(),
        }
    }
}

#[must_use]
pub fn role_of(bank: &str) -> &str {
    parsed(bank).map_or(bank, |(_, _, role)| role)
}

#[must_use]
pub fn layer_of(bank: &str) -> u64 {
    parsed(bank).map_or(0, |(layer, _, _)| layer)
}

#[must_use]
pub fn site_of(bank: &str) -> Option<Site> {
    parsed(bank).and_then(|(_, site, _)| site)
}

fn parsed(bank: &str) -> Option<(u64, Option<Site>, &str)> {
    let (head, role) = bank.rsplit_once('.')?;
    let last = head.rsplit('.').next()?;

    if let Some(layer) = numbered(last) {
        return Some((layer, None, role));
    }

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    A,

    B,
}

impl Role {
    #[must_use]
    pub const fn bank(self) -> &'static str {
        match self {
            Role::A => "lora_a",
            Role::B => "lora_b",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sink {
    pub stage: usize,

    pub planes: Vec<(Role, u32)>,

    pub sites: u32,
}

impl Sink {
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

pub fn sink_of(package: &LaunchPackage) -> Result<Option<Sink>> {
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
                && package
                    .names
                    .get(op.name_index as usize)
                    .map(String::as_str)
                    == Some(LORA)
        })
        .ok_or_else(|| {
            refuse(format!(
                "is what stage {stage}'s plan declares it needs, and no `sink_call` in \
                 that stage's body names it"
            ))
        })?;

    let (weights, sites_arg) = match call.args.as_slice() {
        [a, b, sites] => (vec![(Role::A, *a), (Role::B, *b)], *sites),
        [_, _] => {
            return Err(refuse(
                "is the SCALE form (`adapter_scale`, two arguments): IA3 and DoRA's \
                 second half multiply an already-materialised output by a per-channel \
                 vector, and `poem-ir` declares no `AdapterScale` op for a bank to be \
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

/// The banks a sink's plane of `role` at `site` fills, one a layer in layer
/// order: each holds one plane of an adapter.
pub fn banks_of(role: Role, site: Option<Site>, seats: &[BankSeat]) -> Result<Vec<&BankSeat>> {
    let refuse = |why: String| Fault::Adapter {
        bank: role.bank().to_string(),
        why,
    };
    let of_role: Vec<&BankSeat> = seats
        .iter()
        .filter(|seat| role_of(&seat.name) == role.bank())
        .collect();

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
    Ok(banks)
}

/// Every bank a sink's planes fill, in the order its planes land: the
/// sink's planes in order, each's banks in layer order. An instance's
/// `Adapter` slots are in this order (`ProgramFacts::adapter`).
pub fn plane_banks(sink: &Sink, seats: &[BankSeat]) -> Result<Vec<BankSeat>> {
    let site = sink.site()?;
    let mut out = Vec::new();
    for (role, _) in &sink.planes {
        out.extend(banks_of(*role, site, seats)?.into_iter().cloned());
    }
    Ok(out)
}

/// The content key of an adapter (`Engine::adapter_key`): two independent
/// 64-bit digests (FNV-1a high, SipHash low) of the sink's site word and, in
/// the sink's plane order, each plane's bank role, its seed's length and its
/// seed, word for word as CUDA's and Metal's, so one adapter keys the same
/// on every engine.
#[must_use]
pub fn adapter_digest<'a>(sites: u32, planes: impl Iterator<Item = (&'a str, &'a [u8])>) -> u128 {
    let mut fnv: u64 = 0xcbf2_9ce4_8422_2325;
    let mut sip = std::collections::hash_map::DefaultHasher::new();
    let mut eat = |bytes: &[u8]| {
        use std::hash::Hasher;
        for byte in bytes {
            fnv = (fnv ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        sip.write(bytes);
    };
    eat(&sites.to_le_bytes());
    for (bank, wire) in planes {
        eat(bank.as_bytes());
        eat(&(wire.len() as u64).to_le_bytes());
        eat(wire);
    }
    let sip = std::hash::Hasher::finish(&sip);
    (u128::from(fnv) << 64) | u128::from(sip)
}

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
    let banks = banks_of(role, site, seats)?;
    let seat = banks[0];

    let bank_rank = seat.rows.min(seat.cols);
    let hidden = seat.rows.max(seat.cols);
    let layers = banks.len() as u64;
    let elems = wire.len() as u64 / 4;
    if !(wire.len() as u64).is_multiple_of(4)
        || !elems.is_multiple_of((layers.saturating_mul(hidden)).max(1))
    {
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

fn f32_at(wire: &[u8], at: usize) -> f32 {
    let bytes = [
        wire[at * 4],
        wire[at * 4 + 1],
        wire[at * 4 + 2],
        wire[at * 4 + 3],
    ];
    f32::from_le_bytes(bytes)
}

#[must_use]
pub fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// The adapters landed in the pool's `Adapter` slots, each named by its
/// first slot ref (what its lanes route by) and held by the instances that
/// bound it: the runtime keys an adapter by its content
/// (`Engine::adapter_key`), so instances that carry the same adapter bind
/// the same slots and it lands once.
#[derive(Debug, Default)]
pub struct Landings {
    held: BTreeMap<u32, Landing>,
}

#[derive(Debug)]
struct Landing {
    /// Its slots, one a plane in [`plane_banks`] order.
    slots: Vec<u32>,
    /// The bank each slot's plane fills.
    banks: Vec<String>,
    holders: std::collections::BTreeSet<InstanceId>,
}

impl Landings {
    /// Whether `slots` already hold a landed adapter; `instance` joins it.
    pub fn join(&mut self, instance: InstanceId, slots: &[u32]) -> Option<u32> {
        let first = *slots.first()?;
        let landing = self.held.get_mut(&first)?;
        landing.holders.insert(instance);
        Some(first)
    }

    /// Records `slots` as landed for `instance`; its id, the first slot.
    pub fn land(&mut self, instance: InstanceId, slots: &[u32], banks: Vec<String>) -> u32 {
        let first = slots[0];
        self.held.insert(
            first,
            Landing {
                slots: slots.to_vec(),
                banks,
                holders: std::iter::once(instance).collect(),
            },
        );
        first
    }

    /// `instance` lets go of adapter `id`; the last holder's release frees
    /// the landing (the runtime frees its slots).
    pub fn release(&mut self, instance: InstanceId, id: u32) {
        if let Some(landing) = self.held.get_mut(&id) {
            landing.holders.remove(&instance);
            if landing.holders.is_empty() {
                self.held.remove(&id);
            }
        }
    }

    /// Adapter `id` now lies in `slots` (the runtime moved it).
    pub fn rebind(&mut self, id: u32, slots: &[u32]) -> Option<u32> {
        let mut landing = self.held.remove(&id)?;
        landing.slots = slots.to_vec();
        let first = *slots.first()?;
        self.held.insert(first, landing);
        Some(first)
    }

    /// Adapter `id`'s slots, one a plane in [`plane_banks`] order.
    #[must_use]
    pub fn slots(&self, id: u32) -> Option<&[u32]> {
        self.held.get(&id).map(|landing| landing.slots.as_slice())
    }

    /// The slot adapter `id` holds bank `bank`'s plane in.
    #[must_use]
    pub fn slot_of(&self, id: u32, bank: &str) -> Option<u32> {
        let landing = self.held.get(&id)?;
        let at = landing.banks.iter().position(|b| b == bank)?;
        landing.slots.get(at).copied()
    }

    #[must_use]
    pub fn live(&self) -> usize {
        self.held.len()
    }
}

#[cfg(test)]
mod digest_tests {}
