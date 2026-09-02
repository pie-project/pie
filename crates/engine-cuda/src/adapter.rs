//! Resolves the `lora` sink: which channels carry adapter weights
//! (`sink_of`) and how one seeded f32 cell becomes bank bytes (`planes_of`).
//! Runs between fires, on the host; the scale form (`adapter_scale`) is refused.

use eta_compiler::codegen::launch::{LaunchPackage, ValueOrigin};
use eta_ir::op::tags;

use crate::blob::Site;
use crate::error::{Fault, Result};
use crate::weights::BankSeat;

/// The sink's name in the package's name table.
pub const LORA: &str = "lora";

/// Which plane of an adapter a sink argument is. Positional in `lora(a, b, sites)`.
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
    /// The trace-known placement constant: site bits the guest asked for,
    /// [`crate::blob::Site::bit`]'s numbering. Checked against banks that
    /// declare a site; a load whose banks name none is unchecked.
    pub sites: u32,
}

impl Sink {
    /// Which site this guest asked for, or `None` if the sink named no
    /// placement at all.
    ///
    /// # Errors
    ///
    /// [`Fault::Adapter`] if the placement constant is not one site of the
    /// vocabulary.
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

/// Whether this program carries an adapter, and which channels are its
/// weights. `Ok(None)` if no stage declares the sink.
///
/// # Errors
///
/// [`Fault::Adapter`] for a sink this shell cannot serve: the scale form, an
/// argument that is not a channel read, or an arity that is neither of the
/// closed language's two.
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
    // arity selects the form: 3 args is `lora(a, b, sites)`, 2 is
    // `adapter_scale(l, sites)`; the last arg is the placement constant.
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
        // both channel read and take are accepted; anything else is refused.
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

/// One seeded cell, as the banks want it. `wire` is the f32 cell the guest
/// seeded (little-endian, 4 bytes/element); banks carrying `role` are found,
/// sorted by layer, and cut into one full-capacity plane each, rounded to
/// the bank's own element. `site` selects banks declaring that site.
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
    // if no bank of this role names a site, the ask is unchecked; the
    // moment one bank names a site, the load has an opinion and it must match.
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
    // A is [rank, hidden] (rank leading), B is [hidden, rank].
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
        // zero-padded per orientation: A's unused ranks are trailing rows,
        // B's are a stride inside every row.
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

/// f32 to bf16, round to nearest even (matches the weight loader).
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

    fn wire(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bf16(value: f32) -> [u8; 2] {
        bf16_bits(value).to_le_bytes()
    }

    /// A `[layers, rank, hidden]` cell is L contiguous rectangles; each
    /// layer bank takes one, in order.
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

    /// A guest's requested site is checked against banks that declare one.
    #[test]
    fn a_site_the_banks_do_not_declare_is_refused_by_name() {
        let sited = vec![seat("layer.0.o.lora_a", 2, 3), seat("layer.1.o.lora_a", 2, 3)];
        let cell = wire(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        // the site the text names lands, into the sited banks by name.
        let planes = planes_of(Role::A, Some(Site::O), &cell, &sited).expect("the named site");
        assert_eq!(planes[0].0, "layer.0.o.lora_a");
        assert_eq!(planes[1].0, "layer.1.o.lora_a");
        // another one does not.
        let why = planes_of(Role::A, Some(Site::Q), &cell, &sited).expect_err("q is not o");
        let said = why.to_string();
        assert!(said.contains("`q`"), "names the site asked for: {said}");
        assert!(said.contains("`o`"), "and the site declared: {said}");
        // a guest that named none against a load that named one is also refused.
        let why = planes_of(Role::A, None, &cell, &sited).expect_err("none is not o");
        assert!(
            why.to_string().contains("at no stated site"),
            "says what was asked: {why}"
        );
    }

}
