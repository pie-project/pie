//! The `lora` sink: resolving a launch package's adapter channels into bank
//! bytes, and the host-side slot residency table that pins them.
//!
//! Runs entirely between fires, on the host: channel bytes are converted to
//! the bank dtype once, at bind, and never read again on the device.
//!
//! Refuses: the scale form (`adapter_scale`, no `AdapterScale` op exists), a
//! sink argument that is not a channel read, a seed whose element count is
//! not `layers x rank x hidden`, a rank wider than the bank seats, an unseeded
//! channel, and a site the load's banks do not declare.

use std::collections::BTreeMap;

use eta_compiler::codegen::launch::{LaunchPackage, ValueOrigin};
use eta_ir::op::tags;

use crate::error::{Fault, Result};
use crate::weights::{AdapterPlane, BankSeat};

/// The sink's name in the package's name table.
pub const LORA: &str = "lora";

/// Which projection an adapter corrects.
///
/// The spelling is the contract, duplicated in four places (guest `Site`,
/// model text, CUDA resolver, this one) since they cannot share a type
/// across the crate boundary; an unknown spelling is refused, not guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Site {
    /// The query projection.
    Q,
    /// The key projection.
    K,
    /// The value projection.
    V,
    /// The mixer's output projection; the site an untagged bank means.
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

    /// Matches `inferlet::eta::adapter::Site::bit()`; rides the `lora` sink's
    /// placement constant.
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

/// The bank name's role: everything after an optional `layer.{l}[.{site}].`
/// prefix. A middle segment outside the site vocabulary is not a site, so
/// e.g. `layer.3.mixer.lora_a`'s role is the whole string.
#[must_use]
pub fn role_of(bank: &str) -> &str {
    parsed(bank).map_or(bank, |(_, _, role)| role)
}

/// Which layer a bank name puts itself at, or zero for an unnumbered one.
#[must_use]
pub fn layer_of(bank: &str) -> u64 {
    parsed(bank).map_or(0, |(layer, _, _)| layer)
}

/// Which site a bank declares it corrects, or `None` for the text's own
/// default site (unstated, not "no site" — keeps untagged loads unchanged).
#[must_use]
pub fn site_of(bank: &str) -> Option<Site> {
    parsed(bank).and_then(|(_, site, _)| site)
}

/// `layer.{l}[.{site}].{role}`, read once — the whole grammar in one place.
fn parsed(bank: &str) -> Option<(u64, Option<Site>, &str)> {
    let (head, role) = bank.rsplit_once('.')?;
    let last = head.rsplit('.').next()?;
    // `layer.{l}.{role}`, tried first.
    if let Some(layer) = numbered(last) {
        return Some((layer, None, role));
    }
    // `layer.{l}.{site}.{role}`; anything else falls through to `None`.
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

/// Which plane of an adapter a sink argument is. Positional in the closed
/// language (`lora(a, b, sites)`), spelled as [`role_of`]'s bank-name suffix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// `A`: `[layers, rank, hidden]`, rank-major.
    A,
    /// `B`: `[layers, hidden, rank]`, out-major (HF's native orientation).
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
    /// Which stage carries it.
    pub stage: usize,
    /// The dense channel index each plane's weights are seeded into, in role
    /// order.
    pub planes: Vec<(Role, u32)>,
    /// The trace-known placement constant: the site bits the guest asked
    /// for, [`Site::bit`]'s numbering.
    pub sites: u32,
}

impl Sink {
    /// Which site this guest asked for, or `None` for no placement stated.
    /// One sink corrects one site; more than one bit set is refused.
    ///
    /// # Errors
    ///
    /// [`Fault::Adapter`] for a nonzero placement constant not equal to
    /// exactly one site's bit.
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

/// Does this program carry an adapter, and which channels are its weights?
/// `Ok(None)` means no stage declares the sink.
///
/// # Errors
///
/// [`Fault::Adapter`] for a sink this shell cannot serve: the scale form, an
/// argument that is not a channel read, or an arity outside the closed
/// language's two forms.
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
                && package.names.get(op.name_index as usize).map(String::as_str) == Some(LORA)
        })
        .ok_or_else(|| {
            refuse(format!(
                "is what stage {stage}'s plan declares it needs, and no `sink_call` in \
                 that stage's body names it"
            ))
        })?;
    // Arity selects the form: three args is `lora(a, b, sites)`, two is
    // `adapter_scale(l, sites)`. The last arg is the placement constant in both.
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
        // A channel read or take is accepted; a computed value is not.
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

/// Converts one seeded cell into per-layer bank bytes.
///
/// `wire` is the f32 cell the guest seeded. `seats` is the load's bank
/// table; banks carrying `role` are sorted by layer and each gets one
/// full-capacity plane, rounded from f32 to the bank's own element width.
/// `site` is checked only against banks that declare one.
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
    // If no bank of this role names a site, the ask is unchecked; once one
    // bank names a site, the load has an opinion and the ask must match it.
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
    // `A` is `[rank, hidden]` (rank leading), `B` is `[hidden, rank]`.
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
        // Zero-padded per orientation: `A`'s unused ranks are trailing rows,
        // `B`'s are a stride inside every row (see `AdapterPlane`).
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

/// f32 to bf16, round to nearest even (matches the weight loader's conversion).
#[must_use]
pub fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// Where one bind's bytes came from.
#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    /// An instance's own full-capacity planes (private-adapter path).
    Own {
        /// Which instance; its slot is never shared.
        instance: u64,
        /// The planes.
        planes: &'a [AdapterPlane<'a>],
    },
    /// A directory in the deployment's mount, named as the guest spells it.
    /// Keyed by [`crate::blob::Stamp`], so every instance naming it lands on
    /// one slot and pays one copy.
    Shared {
        /// The adapter's name, resolved against
        /// [`crate::serve::Shell::mount_adapters`]'s root.
        name: &'a str,
    },
}

/// What a bind answers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    /// The slot every lane of this instance routes to.
    pub slot: u32,
    /// Did this bind name a file in the mount?
    pub shared: bool,
    /// Did this bind pay the landing, or join one already resident?
    pub landed: bool,
    /// What the slot is held under (see [`crate::serve::Shell::release_adapter`]).
    pub key: Key,
}

/// The key a slot is held under. An instance id is private (never two
/// instances share one); a blob stamp is shared (two instances naming one
/// file compute one stamp).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Key {
    /// One instance's private adapter.
    Instance(u64),
    /// One file in the mount, by identity (path + files' stamp).
    Shared(crate::blob::Stamp),
}

/// The residency table: which of the banks' slots are pinned, by what. Pure
/// host state, checkable with no GPU; the landing happens in a closure at
/// the call site ([`crate::serve::Shell::bind_adapter`]).
#[derive(Debug, Default)]
pub struct Slots {
    /// How many slots the banks seat: the smallest capacity any declared
    /// bank states, since an adapter occupies one slot of every bank.
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

    /// Pin a slot for `key`. A key already held takes its own slot back and
    /// its refcount rises; a fresh key takes the lowest free slot.
    ///
    /// # Errors
    ///
    /// [`Fault::AdapterSlots`] when every slot is pinned by a live bind
    /// (refused rather than evicted, since a pinned slot may be mid-fire).
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

    /// Undo an acquire that could not be landed, so no key is left pointing
    /// at bytes that never arrived.
    pub fn abandon(&mut self, key: &Key) {
        self.release_key(key);
    }

    /// Give a bind back; at the last release the slot is free again. Unlike
    /// the CUDA twin, this frees at zero holds rather than keeping an LRU
    /// occupant seated (landing here is a unified-memory memcpy, not a
    /// transfer to amortise).
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
    //! Pins the lora sink resolution and slot residency arithmetic.
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

    // Two layers of an `A` bank at rank 2, hidden 3.
    fn a_seats() -> Vec<BankSeat> {
        vec![seat("layer.0.lora_a", 2, 3), seat("layer.1.lora_a", 2, 3)]
    }

    fn wire(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bf16(value: f32) -> [u8; 2] {
        bf16_bits(value).to_le_bytes()
    }

    /// Read a bf16 plane back as f32.
    fn as_f32(plane: &[u8]) -> Vec<f32> {
        plane
            .chunks_exact(2)
            .map(|two| f32::from_bits(u32::from(u16::from_le_bytes([two[0], two[1]])) << 16))
            .collect()
    }

    /// A `[layers, rank, hidden]` cell is L contiguous rectangles; the L
    /// banks of the role take one each, in layer order.
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

    /// The rank-r correction as arithmetic: `y += B·(A·x)`, matching what
    /// `linear/lora.metal` computes over the planes this resolver lands.
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

    /// The two channels come back in role order, with the placement
    /// constant beside them.
    #[test]
    fn the_sink_answers_its_two_channels_in_role_order() {
        let sink = sink_of(&package_with(vec![0, 1, 2]))
            .expect("a readable sink")
            .expect("a package that declares one");
        assert_eq!(sink.stage, 0);
        assert_eq!(sink.planes, vec![(Role::A, 4), (Role::B, 5)]);
        assert_eq!(sink.sites, 0b1000, "the trace-known placement constant");
    }

    /// The placement constant reads back as one site.
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

    /// A slot is pinned by a live bind and reclaimed by a release; a full
    /// table refuses by name rather than evicting.
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

    /// One stamp, built by hand so this pin needs no filesystem.
    fn stamp(at: &str, bytes: u64) -> crate::blob::Stamp {
        crate::blob::Stamp {
            at: at.to_string(),
            files: vec![("adapter.toml".to_string(), bytes, 7)],
        }
    }

    /// N instances of one blob occupy one slot; a private adapter shares
    /// with nobody.
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

    /// An abandoned acquire holds nothing.
    #[test]
    fn an_abandoned_acquire_leaves_the_slot_free() {
        let mut slots = Slots::new(1);
        let grant = slots.acquire(Key::Instance(7)).expect("the only slot");
        slots.abandon(&Key::Instance(7));
        assert_eq!(slots.live(), 0);
        let after = slots.acquire(Key::Instance(8)).expect("it is free again");
        assert_eq!(after.slot, grant.slot);
    }

}
