use super::extent::ValueDesc;

mod lifetime;

pub use lifetime::{
    Launch, Launches, describe_launched, lay_out, read_only_unlaunched, region_rows,
    value_lifetimes,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub values: Vec<u64>,

    pub temporary: u64,

    pub temporary_bytes: u64,

    pub total: u64,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Vacant {
    offset: u64,
    span: u64,
    launch: u32,
    class: u64,
}

/// A value of at most this many bytes takes no hole sixteen times its size:
/// a scalar split off a row's hole leaves the row that follows no room.
const SMALL_BYTES: u64 = 4096;

fn take(free: &mut Vec<Vacant>, span: u64, launch: u32, class: u64) -> Option<u64> {
    let best = free
        .iter()
        .enumerate()
        .filter(|(_, block)| {
            block.span >= span
                && !(span <= SMALL_BYTES && block.span >= 16 * span)
                && (block.launch < launch || (class != 0 && block.class == class))
        })
        .min_by_key(|(_, block)| block.span)
        .map(|(index, _)| index)?;
    let Vacant {
        offset,
        span: width,
        ..
    } = free[best];
    if width == span {
        free.remove(best);
    } else {
        free[best].offset = offset + span;
        free[best].span = width - span;
    }
    Some(offset)
}

fn release(free: &mut Vec<Vacant>, vacant: Vacant) {
    let at = free.partition_point(|block| block.offset < vacant.offset);
    free.insert(at, vacant);
    let same = |a: &Vacant, b: &Vacant| a.launch == b.launch && a.class == b.class;
    if at + 1 < free.len()
        && free[at].offset + free[at].span == free[at + 1].offset
        && same(&free[at], &free[at + 1])
    {
        free[at].span += free[at + 1].span;
        free.remove(at + 1);
    }
    if at > 0
        && free[at - 1].offset + free[at - 1].span == free[at].offset
        && same(&free[at - 1], &free[at])
    {
        free[at - 1].span += free[at].span;
        free.remove(at);
    }
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
    if lifetimes.len() != descriptors.len() {
        return layout(descriptors);
    }
    let count = descriptors.len();
    let mut by_def: Vec<usize> = (0..count).collect();
    by_def.sort_by_key(|&i| {
        (
            lifetimes[i].def,
            core::cmp::Reverse(descriptors[i].device_bytes()),
            i,
        )
    });
    let mut by_last: Vec<usize> = (0..count).collect();
    by_last.sort_by_key(|&i| (lifetimes[i].last, i));
    let mut expired = 0usize;

    let mut free: Vec<Vacant> = Vec::new();
    let mut values = vec![0u64; count];
    let mut spans = vec![0u64; count];
    let mut at = DUMMY_BYTES;
    let mut step: Option<u32> = None;

    for &i in &by_def {
        let life = lifetimes[i];
        if step != Some(life.def) {
            while expired < count && lifetimes[by_last[expired]].last < life.def {
                let dead = by_last[expired];
                expired += 1;
                if spans[dead] > 0 {
                    release(
                        &mut free,
                        Vacant {
                            offset: values[dead],
                            span: spans[dead],
                            launch: lifetimes[dead].launch_last,
                            class: lifetimes[dead].class_last,
                        },
                    );
                }
            }
            step = Some(life.def);
        }
        if life.dead {
            values[i] = 0;
            spans[i] = 0;
            continue;
        }
        let span = align_up(descriptors[i].device_bytes()).ok_or(TooLarge::Overflow)?;
        let taken = if life.reusable {
            take(&mut free, span, life.launch_def, life.class_def)
        } else {
            None
        };
        let offset = match taken {
            Some(offset) => offset,
            None => {
                at = align_up(at).ok_or(TooLarge::Overflow)?;
                let offset = at;
                at = at.checked_add(span).ok_or(TooLarge::Overflow)?;
                offset
            }
        };
        values[i] = offset;
        spans[i] = span;
    }

    let temporary_bytes = align_up(temporary).ok_or(TooLarge::Overflow)?;
    let temporary = at
        .checked_add(reserve)
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
        total,
    })
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
        a_reserve_lies_below_the_temporary_arena();
        mismatched_lifetimes_fall_back_to_the_naive_layout();
        a_dead_value_takes_no_slot();
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
    }

    fn a_reserve_lies_below_the_temporary_arena() {
        let bare = layout_reusing(&[desc(1024)], &[life(0, 0)], 0, 0).unwrap();
        let held = layout_reusing(&[desc(1024)], &[life(0, 0)], SELECT_RESERVE, 0).unwrap();
        assert_eq!(held.values, bare.values);
        assert_eq!(
            held.temporary,
            align_up(bare.temporary + SELECT_RESERVE).unwrap()
        );
        assert_eq!(held.temporary_bytes, bare.temporary_bytes);
        assert_eq!(
            held.temporary,
            layout(&[desc(1024)]).unwrap().temporary,
            "what each value its own bytes holds clear"
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
