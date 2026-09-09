use super::extent::ValueDesc;

pub const ALIGN: u64 = 256;

pub const DUMMY_BYTES: u64 = ALIGN;

pub const MAX_BYTES: u64 = (4 << 30) - ALIGN;

const TEMPORARIES_PER_ELEMENT: u64 = 4;

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

    let mut widest: u64 = 1;

    for descriptor in descriptors {
        at = align_up(at).ok_or(TooLarge::Overflow)?;
        values.push(at);
        let span = align_up(descriptor.device_bytes()).ok_or(TooLarge::Overflow)?;
        at = at.checked_add(span).ok_or(TooLarge::Overflow)?;
        widest = widest.max(u64::from(descriptor.len));
    }

    let temporary = align_up(at).ok_or(TooLarge::Overflow)?;
    let temporary_bytes = widest
        .checked_mul(size_of::<u32>() as u64)
        .and_then(|bytes| bytes.checked_mul(TEMPORARIES_PER_ELEMENT))
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

fn take(free: &mut Vec<Vacant>, span: u64, launch: u32, class: u64) -> Option<u64> {
    let best = free
        .iter()
        .enumerate()
        .filter(|(_, block)| {
            block.span >= span && (block.launch < launch || (class != 0 && block.class == class))
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

pub fn layout_reusing(
    descriptors: &[ValueDesc],
    lifetimes: &[Lifetime],
    temporary_floor: u64,
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
    let mut widest: u64 = 1;
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
        let descriptor = &descriptors[i];
        widest = widest.max(u64::from(descriptor.last.max(1)));
        let span = align_up(descriptor.device_bytes()).ok_or(TooLarge::Overflow)?;
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

    let temporary = align_up(at).ok_or(TooLarge::Overflow)?;
    let temporary_bytes = widest
        .checked_mul(size_of::<u32>() as u64)
        .and_then(|bytes| bytes.checked_mul(TEMPORARIES_PER_ELEMENT))
        .map(|bytes| bytes.max(temporary_floor))
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
        the_temporary_arena_is_sized_by_the_widest_row();
        the_temporary_floor_lifts_the_arena();
        mismatched_lifetimes_fall_back_to_the_naive_layout();
        a_dead_value_takes_no_slot();
    }

    fn a_value_dead_before_the_next_region_hands_its_slot_on() {
        let descriptors = [desc(1024), desc(1024), desc(1024)];
        let lifetimes = [life(0, 1), life(1, 2), life(2, 2)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0).unwrap();
        let naive = layout(&descriptors).unwrap();
        assert_eq!(reused.values[2], reused.values[0]);
        assert_ne!(reused.values[1], reused.values[0]);
        assert!(reused.total < naive.total);
    }

    fn a_value_last_read_where_another_is_defined_does_not_share() {
        let descriptors = [desc(1024), desc(1024)];
        let lifetimes = [life(0, 1), life(1, 1)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0).unwrap();
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
        let reused = layout_reusing(&descriptors, &lifetimes, 0).unwrap();
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
        let reused = layout_reusing(&descriptors, &lifetimes, 0).unwrap();
        assert_ne!(reused.values[0], reused.values[1]);
    }

    fn a_wider_taker_gets_a_coalesced_pair_of_slots() {
        let descriptors = [desc(1024), desc(1024), desc(2048)];
        let lifetimes = [life(0, 0), life(0, 0), life(1, 1)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0).unwrap();
        assert_eq!(reused.values[2], reused.values[0].min(reused.values[1]));
    }

    fn the_temporary_arena_is_sized_by_the_widest_row() {
        let wide = ValueDesc {
            len: 4 * 1024,
            rows: 4,
            last: 1024,
            rank: 2,
            dtype: 0,
            dims: [0; MAX_RANK],
        };
        let reused = layout_reusing(&[wide], &[life(0, 0)], 0).unwrap();
        assert_eq!(reused.temporary_bytes, 1024 * 4 * TEMPORARIES_PER_ELEMENT);
        assert_eq!(
            layout(&[wide]).unwrap().temporary_bytes,
            4 * 1024 * 4 * TEMPORARIES_PER_ELEMENT
        );
    }

    fn the_temporary_floor_lifts_the_arena() {
        let reused = layout_reusing(&[desc(1024)], &[life(0, 0)], 1 << 20).unwrap();
        assert_eq!(reused.temporary_bytes, 1 << 20);
    }

    fn mismatched_lifetimes_fall_back_to_the_naive_layout() {
        let descriptors = [desc(1024), desc(1024)];
        assert_eq!(
            layout_reusing(&descriptors, &[life(0, 0)], 0).unwrap(),
            layout(&descriptors).unwrap()
        );
    }

    fn a_dead_value_takes_no_slot() {
        let descriptors = [desc(1024), desc(1024), desc(1024)];
        let mut lifetimes = [life(0, 2), life(1, 2), life(2, 2)];
        let with = layout_reusing(&descriptors, &lifetimes, 0).expect("fits");
        lifetimes[1].dead = true;
        let without = layout_reusing(&descriptors, &lifetimes, 0).expect("fits");
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
