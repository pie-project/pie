use super::extent::ValueDesc;
pub use eta_compiler::codegen::temporary::TemporaryNeed;

mod lifetime;

pub use lifetime::{
    Keeps, Launch, Launches, describe_launched, launch_areas, lay_out, read_only_unlaunched,
    region_rows, value_lifetimes,
};

pub const ALIGN: u64 = 256;

/// A lane's first bytes, no value's: where a value that takes no bytes is
/// stated to lie.
pub const DUMMY_BYTES: u64 = ALIGN;

/// What an engine built to fill test slots writes over [`DUMMY_BYTES`]
/// before a launch,
/// so a kernel that reads a value stated to take no bytes answers wrongly
/// where a test can see it: one byte repeated, which a device's byte fill
/// writes; a float above any logit, which decides a maximum, an argmax or a
/// sum that reads it; an index past any table; a state no draw starts from.
pub const POISON: u32 = 0x7f7f_7f7f;

pub const MAX_BYTES: u64 = (4 << 30) - ALIGN;

const TEMPORARIES_PER_ELEMENT: u64 = 4;

/// The WGSL runtime's one-row select keeps its state just below the
/// temporary area, out of reach of the sort ladder that reuses all of it.
pub const SELECT_RESERVE: u64 = eta_compiler::codegen::wgsl::SELECT_WORDS as u64 * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TooLarge {
    Bound { bytes: u64, limit: u64 },

    Overflow,
}

impl core::fmt::Display for TooLarge {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bound { bytes, limit } => {
                write!(
                    f,
                    "a scratch of {bytes} bytes, past the {limit} a lane addresses"
                )
            }
            Self::Overflow => write!(f, "a scratch past a u64"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub values: Vec<u64>,

    pub temporary: u64,

    pub temporary_bytes: u64,

    /// Each launch's own temporary area ([`Area`]), in the order stated:
    /// its first byte, its reserve right below.
    pub areas: Vec<u64>,

    pub total: u64,
}

/// A temporary area one launch keeps for itself ([`Keeps`]): alive over
/// its steps, `reserve` bytes clear below its `bytes`. It shares bytes
/// with a value as another value would, by their lifetimes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Area {
    pub life: Lifetime,
    pub reserve: u64,
    pub bytes: u64,
}

fn align_up(value: u64) -> Option<u64> {
    value.checked_next_multiple_of(ALIGN)
}

pub fn layout(descriptors: &[ValueDesc]) -> Result<Layout, TooLarge> {
    let mut values = Vec::with_capacity(descriptors.len());
    let mut at = DUMMY_BYTES;

    for descriptor in descriptors {
        at = align_up(at).ok_or(TooLarge::Overflow)?;
        values.push(at);
        let span = align_up(descriptor.device_bytes()).ok_or(TooLarge::Overflow)?;
        at = at.checked_add(span).ok_or(TooLarge::Overflow)?;
    }

    let temporary = at
        .checked_add(SELECT_RESERVE)
        .and_then(align_up)
        .ok_or(TooLarge::Overflow)?;
    let temporary_bytes = whole_value_temporaries(descriptors)
        .and_then(align_up)
        .ok_or(TooLarge::Overflow)?;
    let total = temporary
        .checked_add(temporary_bytes)
        .ok_or(TooLarge::Overflow)?;

    if total > MAX_BYTES {
        return Err(TooLarge::Bound {
            bytes: total,
            limit: MAX_BYTES,
        });
    }
    Ok(Layout {
        values,
        temporary,
        temporary_bytes,
        areas: Vec::new(),
        total,
    })
}

/// The temporaries a launch that sorts or reduces a whole value takes:
/// [`TEMPORARIES_PER_ELEMENT`] words an element of the widest of
/// `descriptors`. `None` past a `u64`.
#[must_use]
pub fn whole_value_temporaries(descriptors: &[ValueDesc]) -> Option<u64> {
    descriptors
        .iter()
        .map(|descriptor| u64::from(descriptor.len))
        .fold(1, u64::max)
        .checked_mul(size_of::<u32>() as u64)
        .and_then(|bytes| bytes.checked_mul(TEMPORARIES_PER_ELEMENT))
}

/// A stage's temporary area: the widest of the whole-value rule
/// ([`whole_value_temporaries`]) and what each region's emitter publishes
/// it takes (`needs`, [`TemporaryNeed`]), evaluated against `descriptors`.
/// An emitter that publishes none passes none. Refused by name where a
/// need names a value the stage does not hold, or the bytes pass a `u64`.
pub fn stage_temporaries(
    descriptors: &[ValueDesc],
    needs: &[TemporaryNeed],
) -> Result<u64, TemporaryRefused> {
    let whole = whole_value_temporaries(descriptors).ok_or(TemporaryRefused::Overflow)?;
    needs.iter().try_fold(whole, |widest, need| {
        let rows = match need.per_row {
            Some((value, bytes)) => u64::from(
                descriptors
                    .get(value as usize)
                    .ok_or(TemporaryRefused::NoValue { value })?
                    .rows,
            )
            .checked_mul(bytes)
            .ok_or(TemporaryRefused::Overflow)?,
            None => 0,
        };
        let below = if need.above_whole { whole } else { 0 };
        let bytes = need
            .fixed
            .checked_add(rows)
            .and_then(|bytes| bytes.checked_add(below))
            .ok_or(TemporaryRefused::Overflow)?;
        Ok(widest.max(bytes))
    })
}

/// Why [`stage_temporaries`] refuses a stage's temporary area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TemporaryRefused {
    /// A region's need is a row count of a value the stage does not hold.
    NoValue { value: u32 },
    /// The bytes pass a `u64`.
    Overflow,
}

impl core::fmt::Display for TemporaryRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoValue { value } => write!(
                f,
                "a region's temporary need counts the rows of value {value}, which the stage does not hold"
            ),
            Self::Overflow => write!(f, "a stage's temporary area passes a u64"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lifetime {
    pub def: u32,
    pub last: u32,
    pub dead: bool,
    pub reusable: bool,
    pub launch_def: u32,
    pub launch_last: u32,
    pub class_def: u64,
    pub class_last: u64,
}

impl Lifetime {
    pub const SEQUENTIAL: u64 = u64::MAX;
}

/// Whether value `later`'s bytes may lie over value `earlier`'s, given the
/// two lifetimes, or `None` where they may lie anywhere over each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Share {
    /// Anywhere over each other: one is last read a launch before the
    /// other is written, or a step before it in a launch of one block.
    Anywhere,
    /// Only at one offset: within a launch across blocks, block `b` owns
    /// row `b` of every value of a class, so a value that takes another's
    /// bytes there lies at the other's offset, or a block's row would land
    /// on another block's, which may still be reading it.
    AtItsOffset,
    /// Nowhere: their lifetimes overlap, the later may go unwritten in
    /// part (it keeps bytes no value before it held), or they are rows of
    /// two classes in one launch.
    Apart,
}

fn share(a: Lifetime, b: Lifetime) -> Share {
    let (early, late) = if a.last < b.def {
        (a, b)
    } else if b.last < a.def {
        (b, a)
    } else {
        return Share::Apart;
    };
    if !late.reusable {
        Share::Apart
    } else if early.launch_last < late.launch_def {
        Share::Anywhere
    } else if late.class_def == 0 || early.class_last != late.class_def {
        Share::Apart
    } else if late.class_def == Lifetime::SEQUENTIAL {
        Share::Anywhere
    } else {
        Share::AtItsOffset
    }
}

/// `descriptors` laid by `lifetimes`, largest first, with each of `areas`
/// an item of its own: each at the lowest offset where it shares bytes only
/// with items it may share them with ([`share`]). Two items share bytes
/// only where one is last read before the other is written, so the bytes
/// laid are what the items alive together at once take, short of the gaps
/// that sizes leave. `reserve` bytes lie right below the stage's temporary
/// area, which holds `temporary` bytes, past every item: the area alive
/// across the stage.
fn place(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    areas: &[Area],
) -> Result<Layout, TooLarge> {
    let count = descriptors.len();
    let mut spans = Vec::with_capacity(count + areas.len());
    for descriptor in descriptors {
        spans.push(align_up(descriptor.device_bytes()).ok_or(TooLarge::Overflow)?);
    }
    // An area's item: its reserve, aligned, then its bytes.
    let mut leads = Vec::with_capacity(areas.len());
    for area in areas {
        let lead = align_up(area.reserve).ok_or(TooLarge::Overflow)?;
        let span = align_up(area.bytes)
            .and_then(|bytes| bytes.checked_add(lead))
            .ok_or(TooLarge::Overflow)?;
        leads.push(lead);
        spans.push(span);
    }
    let life_of = |i: usize| -> Lifetime {
        if i < count {
            lifetimes[i]
        } else {
            areas[i - count].life
        }
    };
    let mut order: Vec<usize> = (0..count)
        .filter(|&i| !lifetimes[i].dead)
        .chain(count..count + areas.len())
        .collect();
    order.sort_by_key(|&i| (core::cmp::Reverse(spans[i]), life_of(i).def, i));

    let mut starts = vec![0u64; spans.len()];
    let mut laid: Vec<usize> = Vec::new();
    let mut end = DUMMY_BYTES;
    let mut near: Vec<(u64, u64, bool)> = Vec::new();
    let mut candidates: Vec<u64> = Vec::new();
    for &i in &order {
        let span = spans[i];
        near.clear();
        candidates.clear();
        candidates.push(DUMMY_BYTES);
        for &j in &laid {
            let at = starts[j];
            let one = match share(life_of(i), life_of(j)) {
                Share::Anywhere => continue,
                Share::AtItsOffset => true,
                Share::Apart => false,
            };
            near.push((at, at + spans[j], one));
            candidates.push(at + spans[j]);
            if one {
                candidates.push(at);
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        let fits = |at: u64| {
            near.iter()
                .all(|&(start, end, one)| end <= at || at + span <= start || (one && start == at))
        };
        let offset = candidates
            .iter()
            .copied()
            .find(|&at| at.checked_add(span).is_some() && fits(at))
            .ok_or(TooLarge::Overflow)?;
        laid.push(i);
        end = end.max(offset + span);
        starts[i] = offset;
    }
    let temporary_bytes = align_up(temporary).ok_or(TooLarge::Overflow)?;
    let at = end
        .checked_add(reserve)
        .and_then(align_up)
        .ok_or(TooLarge::Overflow)?;
    let total = at.checked_add(temporary_bytes).ok_or(TooLarge::Overflow)?;
    if total > MAX_BYTES {
        return Err(TooLarge::Bound {
            bytes: total,
            limit: MAX_BYTES,
        });
    }
    Ok(Layout {
        // A dead value's offset is the dummy bytes', 0.
        values: starts[..count].to_vec(),
        temporary: at,
        temporary_bytes,
        areas: leads
            .iter()
            .zip(&starts[count..])
            .map(|(&lead, &start)| start + lead)
            .collect(),
        total,
    })
}

/// The most bytes `descriptors`' values alive together take at any step of
/// `lifetimes` (each value's bytes aligned as a layout lays them): the
/// least any layout of their values holds.
#[must_use]
pub fn most_alive(descriptors: &[ValueDesc], lifetimes: &[Lifetime]) -> u64 {
    let mut edges: Vec<(u32, bool, u64)> = descriptors
        .iter()
        .zip(lifetimes)
        .filter(|(_, life)| !life.dead)
        .flat_map(|(descriptor, life)| {
            let span = align_up(descriptor.device_bytes()).unwrap_or(u64::MAX);
            // A value is alive from its step through its last: born at
            // `def`, gone after `last`; at one step births count first.
            [
                (life.def, false, span),
                (life.last.saturating_add(1), true, span),
            ]
        })
        .collect();
    edges.sort_unstable_by_key(|&(step, gone, _)| (step, !gone));
    let (mut alive, mut most) = (0u64, 0u64);
    for (_, gone, span) in edges {
        if gone {
            alive -= span;
        } else {
            alive = alive.saturating_add(span);
            most = most.max(alive);
        }
    }
    most
}

/// `descriptors` laid by `lifetimes`: a value takes the bytes of one whose
/// last reader ran before it is written. `reserve` bytes lie clear below
/// the temporary area, which holds `temporary` bytes: what the kernels keep
/// there, as the engine that runs them states it.
pub fn layout_reusing(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
) -> Result<Layout, TooLarge> {
    reusing_with(descriptors, lifetimes, reserve, temporary, &[])
}

/// [`layout_reusing`] with each of `areas` laid as an item of its own
/// ([`place`]). Where `lifetimes` are not the values', each value holds
/// its own bytes and each area its own past them.
pub(crate) fn reusing_with(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    reserve: u64,
    temporary: u64,
    areas: &[Area],
) -> Result<Layout, TooLarge> {
    if lifetimes.len() != descriptors.len() {
        return layout_with_areas(descriptors, areas);
    }
    place(descriptors, lifetimes, reserve, temporary, areas)
}

/// Each value its own bytes ([`layout`]), and each of `areas` its own
/// past them, its reserve below it.
fn layout_with_areas(descriptors: &[ValueDesc], areas: &[Area]) -> Result<Layout, TooLarge> {
    let mut laid = layout(descriptors)?;
    for area in areas {
        let at = laid
            .total
            .checked_add(area.reserve)
            .and_then(align_up)
            .ok_or(TooLarge::Overflow)?;
        laid.total = align_up(area.bytes)
            .and_then(|bytes| at.checked_add(bytes))
            .ok_or(TooLarge::Overflow)?;
        laid.areas.push(at);
    }
    if laid.total > MAX_BYTES {
        return Err(TooLarge::Bound {
            bytes: laid.total,
            limit: MAX_BYTES,
        });
    }
    Ok(laid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eta_ir::types::MAX_RANK;

    fn desc(len: u32) -> ValueDesc {
        ValueDesc {
            len,
            rows: 1,
            last: len,
            rank: 1,
            dtype: 0,
            dims: [0; MAX_RANK],
        }
    }

    /// The stage's temporary area is the widest of the whole-value rule and
    /// what each region's emitter publishes: a short vector's select keeps
    /// its fixed state below the rule's bytes, a top-k takes so much a row
    /// of its input, an emitter that publishes none takes the rule alone,
    /// and a need naming a value the stage lacks, or past a u64, is refused
    /// by name.
    #[test]
    fn a_stage_s_temporaries_take_the_widest_published_need() {
        let short = [desc(16)];
        let whole = whole_value_temporaries(&short).unwrap();
        assert_eq!(whole, 16 * 16);
        assert_eq!(
            stage_temporaries(&short, &[]),
            Ok(whole),
            "none published: the rule"
        );
        let select = TemporaryNeed {
            fixed: 51_200,
            per_row: None,
            above_whole: true,
        };
        assert_eq!(stage_temporaries(&short, &[select]), Ok(51_200 + whole));
        let mut rows = desc(16);
        rows.rows = 4;
        let topk = TemporaryNeed {
            fixed: 15,
            per_row: Some((0, 14_360)),
            above_whole: false,
        };
        assert_eq!(stage_temporaries(&[rows], &[topk]), Ok(15 + 4 * 14_360));
        assert_eq!(
            stage_temporaries(&short, &[select, topk]),
            Ok((51_200 + whole).max(15 + 14_360)),
            "the widest region's"
        );
        // A wide value's rule covers a small need.
        let wide = [desc(1 << 20)];
        assert_eq!(
            stage_temporaries(&wide, &[topk]),
            Ok(whole_value_temporaries(&wide).unwrap())
        );
        let stray = TemporaryNeed {
            per_row: Some((3, 8)),
            ..TemporaryNeed::default()
        };
        assert_eq!(
            stage_temporaries(&short, &[stray]),
            Err(TemporaryRefused::NoValue { value: 3 })
        );
        let past = TemporaryNeed {
            fixed: u64::MAX,
            above_whole: true,
            ..TemporaryNeed::default()
        };
        assert_eq!(
            stage_temporaries(&short, &[past]),
            Err(TemporaryRefused::Overflow)
        );
    }

    fn life(def: u32, last: u32) -> Lifetime {
        Lifetime {
            dead: false,
            def,
            last,
            reusable: true,
            launch_def: def,
            launch_last: last,
            class_def: Lifetime::SEQUENTIAL,
            class_last: Lifetime::SEQUENTIAL,
        }
    }

    fn rowed(def: u32, last: u32, launch: u32, class: u64) -> Lifetime {
        Lifetime {
            dead: false,
            def,
            last,
            reusable: true,
            launch_def: launch,
            launch_last: launch,
            class_def: class,
            class_last: class,
        }
    }

    #[test]
    fn scratch_every_case() {
        a_value_dead_before_the_next_region_hands_its_slot_on();
        a_value_last_read_where_another_is_defined_does_not_share();
        within_a_many_block_launch_only_one_class_shares();
        a_result_that_may_go_unwritten_keeps_a_fresh_slot();
        a_wider_taker_gets_a_coalesced_pair_of_slots();
        the_temporary_area_holds_what_is_stated();
        mismatched_lifetimes_fall_back_to_the_naive_layout();
        a_dead_value_takes_no_slot();
    }

    /// Within a launch across blocks, block `b` owns row `b` of every value
    /// of a row class: a value that takes another's bytes there lies at the
    /// other's offset, so no block's row lands on another block's. Values
    /// that share across launches may lie anywhere over each other.
    #[test]
    fn rows_of_one_class_share_only_at_one_offset() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        for _ in 0..20_000 {
            let count = 2 + next(10) as usize;
            let mut descriptors = Vec::new();
            let mut lifetimes = Vec::new();
            for _ in 0..count {
                let rows = 1 + next(4) as u32;
                let width = [64u32, 64, 64, 96][next(4) as usize];
                descriptors.push(ValueDesc {
                    len: rows * width,
                    rows,
                    last: width,
                    rank: 2,
                    dtype: 0,
                    dims: [0; MAX_RANK],
                });
                let def = next(8) as u32;
                let last = def + next(4) as u32;
                // Steps 0..3 one launch across blocks, 4.. another.
                let launch = |step: u32| u32::from(step >= 4);
                let class = if next(5) == 0 {
                    0
                } else {
                    (1u64 << 40) | (u64::from(width) * 4)
                };
                lifetimes.push(Lifetime {
                    dead: false,
                    def,
                    last,
                    reusable: next(6) != 0,
                    launch_def: launch(def),
                    launch_last: launch(last),
                    class_def: class,
                    class_last: class,
                });
            }
            let places = layout_reusing(&descriptors, &lifetimes, 0, 0)
                .unwrap()
                .values;
            let span = |i: usize| align_up(descriptors[i].device_bytes()).unwrap();
            for i in 0..count {
                for j in 0..count {
                    let (a, b) = (lifetimes[i], lifetimes[j]);
                    let (oi, oj) = (places[i], places[j]);
                    let overlap = oi < oj + span(j) && oj < oi + span(i);
                    if i == j || !overlap {
                        continue;
                    }
                    assert!(
                        a.last < b.def || b.last < a.def,
                        "{i} and {j} live together"
                    );
                    let (early, late) = if a.last < b.def { (a, b) } else { (b, a) };
                    if early.launch_last >= late.launch_def {
                        assert_eq!(oi, oj, "{i} and {j} share a launch's rows at two offsets");
                    }
                }
            }
        }
    }

    fn area(def: u32, last: u32, reserve: u64, bytes: u64) -> Area {
        Area {
            life: life(def, last),
            reserve,
            bytes,
        }
    }

    /// Random values and areas: no value alive with an area shares a byte
    /// of it or of its reserve, nor two areas alive together, nor two
    /// values alive together, and every item lies inside the scratch.
    #[test]
    fn no_item_alive_with_another_shares_its_bytes() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let overlap = |a: Lifetime, b: Lifetime| a.last >= b.def && b.last >= a.def;
        for _ in 0..20_000 {
            let count = 1 + next(10) as usize;
            let mut descriptors = Vec::new();
            let mut lifetimes = Vec::new();
            for _ in 0..count {
                descriptors.push(desc(1 + next(4096) as u32));
                let def = next(8) as u32;
                let mut value = life(def, def + next(4) as u32);
                value.reusable = next(6) != 0;
                value.dead = next(10) == 0;
                lifetimes.push(value);
            }
            let areas: Vec<Area> = (0..next(4))
                .map(|_| {
                    let def = next(8) as u32;
                    let reserve = [0, SELECT_RESERVE, 100][next(3) as usize];
                    area(def, def, reserve, next(40_000))
                })
                .collect();
            let laid = reusing_with(&descriptors, &lifetimes, 0, 0, &areas).unwrap();
            let mut items: Vec<(u64, u64, Lifetime, bool)> = (0..count)
                .filter(|&i| !lifetimes[i].dead)
                .map(|i| {
                    let at = laid.values[i];
                    (
                        at,
                        at + descriptors[i].device_bytes(),
                        lifetimes[i],
                        lifetimes[i].reusable,
                    )
                })
                .collect();
            items.extend(
                areas.iter().zip(&laid.areas).map(|(area, &first)| {
                    (first - area.reserve, first + area.bytes, area.life, true)
                }),
            );
            for (x, &(start, end, life, ours)) in items.iter().enumerate() {
                assert!(start >= DUMMY_BYTES && end <= laid.total, "{laid:?}");
                for &(s, e, them, theirs) in &items[x + 1..] {
                    // A value that may go unwritten in part takes no bytes
                    // any item held before it.
                    let fresh =
                        (!theirs && them.def > life.last) || (!ours && life.def > them.last);
                    if overlap(life, them) || fresh {
                        assert!(end <= s || e <= start, "{items:?}");
                    }
                }
            }
        }
    }

    fn a_value_dead_before_the_next_region_hands_its_slot_on() {
        let descriptors = [desc(1024), desc(1024), desc(1024)];
        let lifetimes = [life(0, 1), life(1, 2), life(2, 2)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        let naive = layout(&descriptors).unwrap();
        assert_eq!(reused.values[2], reused.values[0]);
        assert_ne!(reused.values[1], reused.values[0]);
        assert!(reused.total < naive.total);
    }

    fn a_value_last_read_where_another_is_defined_does_not_share() {
        let descriptors = [desc(1024), desc(1024)];
        let lifetimes = [life(0, 1), life(1, 1)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        assert_ne!(reused.values[0], reused.values[1]);
    }

    fn within_a_many_block_launch_only_one_class_shares() {
        let descriptors = [desc(1024), desc(1024), desc(1024), desc(1024), desc(1024)];
        let lifetimes = [
            rowed(0, 0, 0, 7),
            rowed(1, 1, 0, 7),
            rowed(2, 2, 0, 3),
            rowed(3, 3, 0, 0),
            rowed(4, 4, 1, 0),
        ];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        assert_eq!(
            reused.values[1], reused.values[0],
            "same class, same launch"
        );
        assert_ne!(reused.values[2], reused.values[0], "another class");
        assert_ne!(reused.values[2], reused.values[1]);
        assert_ne!(
            reused.values[3], reused.values[0],
            "a whole value shares with nothing"
        );
        assert_eq!(
            reused.values[4], reused.values[0],
            "the next launch may take it"
        );
    }

    fn a_result_that_may_go_unwritten_keeps_a_fresh_slot() {
        let descriptors = [desc(1024), desc(1024)];
        let lifetimes = [
            life(0, 0),
            Lifetime {
                reusable: false,
                ..life(1, 1)
            },
        ];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        assert_ne!(reused.values[0], reused.values[1]);
    }

    fn a_wider_taker_gets_a_coalesced_pair_of_slots() {
        let descriptors = [desc(1024), desc(1024), desc(2048)];
        let lifetimes = [life(0, 0), life(0, 0), life(1, 1)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0, 0).unwrap();
        assert_eq!(reused.values[2], reused.values[0].min(reused.values[1]));
    }

    fn the_temporary_area_holds_what_is_stated() {
        let wide = ValueDesc {
            len: 4 * 1024,
            rows: 4,
            last: 1024,
            rank: 2,
            dtype: 0,
            dims: [0; MAX_RANK],
        };
        let none = layout_reusing(&[wide], &[life(0, 0)], 0, 0).unwrap();
        assert_eq!(none.temporary_bytes, 0, "nothing stated, nothing held");
        assert_eq!(none.total, none.temporary);
        let stated = layout_reusing(&[wide], &[life(0, 0)], 0, 1000).unwrap();
        assert_eq!(stated.values, none.values);
        assert_eq!(stated.temporary, none.temporary);
        assert_eq!(stated.temporary_bytes, align_up(1000).unwrap());
        assert_eq!(
            layout(&[wide]).unwrap().temporary_bytes,
            4 * 1024 * 4 * TEMPORARIES_PER_ELEMENT,
            "each value its own bytes holds the widest value's words"
        );
        assert_eq!(
            layout_reusing(&[wide], &[life(0, 0)], 0, u64::MAX),
            Err(TooLarge::Overflow)
        );
        // Past what a lane addresses: refused, the bytes and the bound said.
        let refused = layout_reusing(&[wide], &[life(0, 0)], 0, MAX_BYTES).unwrap_err();
        let asked = align_up(DUMMY_BYTES + wide.device_bytes()).unwrap() + MAX_BYTES;
        assert_eq!(
            refused.to_string(),
            format!("a scratch of {asked} bytes, past the {MAX_BYTES} a lane addresses")
        );
    }

    fn mismatched_lifetimes_fall_back_to_the_naive_layout() {
        let descriptors = [desc(1024), desc(1024)];
        assert_eq!(
            layout_reusing(&descriptors, &[life(0, 0)], 0, 0).unwrap(),
            layout(&descriptors).unwrap()
        );
    }

    fn a_dead_value_takes_no_slot() {
        let descriptors = [desc(1024), desc(1024), desc(1024)];
        let mut lifetimes = [life(0, 2), life(1, 2), life(2, 2)];
        let with = layout_reusing(&descriptors, &lifetimes, 0, 0).expect("fits");
        lifetimes[1].dead = true;
        let without = layout_reusing(&descriptors, &lifetimes, 0, 0).expect("fits");
        assert_eq!(
            without.values[1], 0,
            "a dead value's offset is the dummy region's"
        );
        assert_eq!(
            without.total + align_up(4096).unwrap(),
            with.total,
            "the dead value's slot is the whole difference"
        );
        assert_eq!(
            without.values[2],
            without.values[0] + align_up(4096).unwrap()
        );
    }
}
