//! **THE LORA SINK, READ** (alto adapter §6.1, §6.3, §6.4) — the half of the
//! resolver that is pure arithmetic over a launch package and a seed.
//!
//! # The sentence this module exists to finish
//!
//! `plan.needs.lora` had zero readers. The compiler has always said WHETHER a
//! pass carries an adapter (`eta-compiler/src/codegen/launch.rs`, the
//! `SINK_CALL` arm), the engine has always had a bank to put one in
//! ([`crate::weights::Weights::register_adapter`]), and nothing joined the
//! two. This module is the join, and it is deliberately three small
//! questions rather than one verb:
//!
//! ```text
//! sink_of(package)         which channels does the `lora` sink name?
//! planes_of(sink, seeds)   what does one seed's f32 cell mean as bank bytes?
//! (the caller)             which slot, and who releases it
//! ```
//!
//! # Why the CHANNELS are read here and never at fire time
//!
//! §6.1's ruling: a 12 MiB channel cell is legal but the machinery re-pays it
//! EVERY FIRE — `CHAN_READ` materialises the cell into per-lane scratch and
//! `pull_validate` re-drags the mirror over mapped-pinned PCIe. So an adapter
//! channel is a NAMING device. The bytes are taken off the seed ONCE, at
//! instance bind, converted into the banks' own dtype and landed; the cell is
//! never read again, and [`crate::blob`]'s residency table decides which slot
//! they land in. Everything in this file therefore runs between fires, on the
//! host, with no device in sight — which is what makes every claim it makes
//! testable without a GPU.
//!
//! # What is refused, by name
//!
//! * the SCALE form (`adapter_scale`, IA3/DoRA's two-argument spelling): the
//!   eta wire selects the form by sink arity and `model-ir` declares no
//!   `AdapterScale` op, so a shell that accepted it would seat weights no
//!   text can read. Refused with the arity in the message (§2's item 3).
//! * a sink argument that is not a channel read — the closed language builds
//!   its operands from `chan_read`, so anything else is a program this
//!   resolver did not compile against.
//! * a seed whose element count is not `layers x rank x hidden` for the banks
//!   of its role, both numbers named.
//! * a rank wider than the bank seats, both numbers named.
//! * a channel the sink names and the bind seeded nothing into: the weights
//!   would be a cell of zeros, which is the identity adapter, which is a
//!   silently wrong answer rather than a loud one.
//! * **a site the guest asked for and this load's banks do not declare**
//!   (alto next B3): until a bank name could carry a site, a guest's
//!   `Site::Q` took whichever site the text corrected and nobody was told.
//!   Now a text may name it ([`crate::site_of`]) and a mismatch is refused
//!   with both sites in the message. A load whose banks name NO site is the
//!   pre-B3 load and keeps its meaning exactly — the widening costs the six
//!   family texts nothing.

use eta_compiler::codegen::launch::{LaunchPackage, ValueOrigin};
use eta_ir::op::tags;

use crate::blob::Site;
use crate::error::{Fault, Result};
use crate::weights::BankSeat;

/// The sink's name in the package's name table. First-party, and the same
/// string `eta-ir`'s registry and `eta-compiler`'s `SINK_CALL` arm agree on.
pub const LORA: &str = "lora";

/// Which plane of an adapter a sink argument is.
///
/// The ROLE is positional in the closed language — `lora(a, b, sites)` — and
/// it is spelled as the bank-name suffix [`crate::role_of`] answers, because
/// that is the one place the convention already lives (§6.3: "the resolver
/// slices, per layer", and the role is how a `[layers, ...]` source finds its
/// L banks).
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
    /// for, [`crate::blob::Site::bit`]'s own numbering.
    ///
    /// **AND IT IS MATCHED AGAINST THE MODEL TEXT NOW** (alto next B3). It
    /// used not to be, and could not be: a bank was named `layer.{l}.lora_a`
    /// and carried no site of its own, so a guest's `Site::Q` took whichever
    /// site the text corrected, unchecked. A name may now declare its site
    /// ([`crate::site_of`]) and [`Sink::site`] reads this constant back into
    /// one — so a load whose banks name a site refuses a guest that asked for
    /// a different one, and a load whose banks name none goes on meaning what
    /// it meant.
    pub sites: u32,
}

impl Sink {
    /// **WHICH SITE THIS GUEST ASKED FOR**, or `None` for a sink that named
    /// no placement at all.
    ///
    /// One sink corrects ONE site: `Pass::adapter(site, …)` emits one `lora`
    /// per site with `Tensor::constant(site.bit())` beside it
    /// (`crates/inferlet/src/eta.rs`), so exactly one bit is the shape this
    /// resolver compiled against. `None` — the constant the package did not
    /// declare — keeps today's meaning and lets an untagged load serve it.
    ///
    /// # Errors
    ///
    /// [`Fault::Adapter`] for a placement constant that is not one site of
    /// the vocabulary: a program built outside the closed surface, refused
    /// with the bits it carried rather than rounded to the nearest site.
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
/// endian, which is [`eta_exec::wire_cell_bytes`]'s encoding for every
/// non-bool dtype. `seats` is the whole load's bank table; the banks that
/// carry `role` are found in it, sorted by layer, and the source is cut into
/// one full-capacity plane each.
///
/// **THE CONVERSION IS THE POINT.** A channel's lane dtypes are F32/I32/U32/
/// Bool and a bank's is the model text's (bf16 at every qwen SKU), so the
/// bytes cannot simply be copied: they are rounded to the bank's own element
/// here, once, on the host, at bind. §6.3 gives that as one of the reasons
/// the blob path exists at all — a file carries the bank's dtype and a
/// channel never could — and it is why a channel-seeded adapter pays a
/// conversion the file path does not.
///
/// **AND THE SITE IS CHECKED HERE** (alto next B3). `site` is what the guest
/// asked for — [`Sink::site`]'s reading of the placement constant — and the
/// banks it lands in are the ones that declare THAT site. A load whose banks
/// declare no site at all is the pre-B3 load and takes today's meaning: the
/// text's own default site, whatever the guest named, because there is no
/// fact to check against and inventing one would refuse every program that
/// works. A load whose banks DO name sites refuses a guest that asked for one
/// they do not, by name — which is the silent-wrongness A-2 found, closed.
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
        .filter(|seat| crate::role_of(&seat.name) == role.bank())
        .collect();
    // **ABSENT IS A VALUE, NOT A WILDCARD — EXCEPT WHERE IT IS THE ONLY
    //   VALUE.** A load whose every bank of this role declares no site is a
    //   text written before the site tag existed; its one correction site is
    //   the text's own and there is nothing to compare a guest's ask with, so
    //   the ask is carried unchecked exactly as it was. The moment ONE bank
    //   names a site, the load has an opinion and the ask has to match it.
    let sited = of_role.iter().any(|seat| crate::site_of(&seat.name).is_some());
    let want = match sited {
        true => site,
        false => None,
    };
    let mut banks: Vec<&BankSeat> = of_role
        .iter()
        .copied()
        .filter(|seat| crate::site_of(&seat.name) == want)
        .collect();
    banks.sort_by_key(|seat| crate::layer_of(&seat.name));
    let seat = *banks.first().ok_or_else(|| {
        let declared = {
            let mut seen: Vec<String> = Vec::new();
            for seat in &of_role {
                let spelled = match crate::site_of(&seat.name) {
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
    // The same reading `crate::blob::Layout::of_bank` takes, and the same
    // statute: the source has to arrive in the bank's orientation, because
    // the alternative is a repack kernel this shell does not ship.
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
        // **ZERO-PADDED PER ORIENTATION**, exactly as `AdapterPlane`'s own doc
        // and `crate::blob::Adapters::planes` say: `A`'s unused ranks are
        // trailing ROWS and `B`'s are a stride inside every row. A zero row of
        // `A` contributes zero to the waist and a zero column of `B`
        // contributes zero to the sum, so both paddings are exact.
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
    let bytes = [wire[at * 4], wire[at * 4 + 1], wire[at * 4 + 2], wire[at * 4 + 3]];
    f32::from_le_bytes(bytes)
}

/// f32 to bf16, ROUND TO NEAREST EVEN.
///
/// The same conversion the weight loader performs, and stated rather than
/// truncated for the reason `adapter_banks.rs` gives where it does the same:
/// a truncating resolver would land a slightly different adapter than the one
/// the guest described, and every parity claim below it would be about the
/// wrong numbers.
#[must_use]
pub fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
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

    /// **A SHORT RANK PADS AS TRAILING ROWS FOR `A`** and as a stride inside
    /// every row for `B` — the two are different placements of the same
    /// zeros, and getting them the same way round is the whole of the
    /// orientation statute.
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
        // And a rank past the bank's.
        let cell = wire(&[1.0; 18]);
        let why = planes_of(Role::A, None, &cell, &a_seats()).expect_err("rank 3 into rank 2");
        assert!(
            why.to_string().contains("seats rank 2"),
            "names both ranks: {why}"
        );
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

    /// **A GUEST'S SITE, AGAINST A TEXT THAT NAMES ITS OWN** (alto next B3).
    ///
    /// This is A-2's silent wrongness, closed: a load whose banks declare
    /// `o` and a guest that asked for `q` used to be indistinguishable from
    /// a match, because the bank name could not carry the site to compare
    /// with. Now it is a refusal, and the refusal says both sites.
    #[test]
    fn a_site_the_banks_do_not_declare_is_refused_by_name() {
        let sited = vec![seat("layer.0.o.lora_a", 2, 3), seat("layer.1.o.lora_a", 2, 3)];
        let cell = wire(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        // The site the text names lands, into the sited banks by name.
        let planes = planes_of(Role::A, Some(Site::O), &cell, &sited).expect("the named site");
        assert_eq!(planes[0].0, "layer.0.o.lora_a");
        assert_eq!(planes[1].0, "layer.1.o.lora_a");
        // Another one does not.
        let why = planes_of(Role::A, Some(Site::Q), &cell, &sited).expect_err("q is not o");
        let said = why.to_string();
        assert!(said.contains("`q`"), "names the site asked for: {said}");
        assert!(said.contains("`o`"), "and the site declared: {said}");
        // And so does a guest that named none against a load that named one:
        // the ask is unstated, the text's is not, and unstated is a value.
        let why = planes_of(Role::A, None, &cell, &sited).expect_err("none is not o");
        assert!(
            why.to_string().contains("at no stated site"),
            "says what was asked: {why}"
        );
    }

    /// **AND AN UNTAGGED LOAD MEANS WHAT IT ALWAYS MEANT** — the
    /// byte-compatible half. Every A-6 family text names `layer.{l}.lora_a`,
    /// so every guest site it is asked for lands exactly the bytes it landed
    /// before the widening.
    #[test]
    fn an_untagged_load_answers_every_site_the_bytes_it_always_did() {
        let cell = wire(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let before = planes_of(Role::A, None, &cell, &a_seats()).expect("the pre-B3 call");
        for site in Site::ALL {
            let after = planes_of(Role::A, Some(site), &cell, &a_seats())
                .unwrap_or_else(|why| panic!("an untagged load serves {site:?}: {why}"));
            assert_eq!(after, before, "{site:?} lands the same bytes in the same banks");
        }
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
    /// landing weights no text can read.
    #[test]
    fn the_scale_form_is_refused_and_says_why() {
        let why = sink_of(&package_with(vec![0, 2])).expect_err("two arguments is the scale");
        let said = why.to_string();
        assert!(said.contains("SCALE"), "{said}");
        assert!(said.contains("AdapterScale"), "names the missing op: {said}");
    }
}
