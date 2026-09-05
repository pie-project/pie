use super::extent::ValueDesc;

pub const ALIGN: u64 = 256;

pub const DUMMY_BYTES: u64 = ALIGN;

/// The most scratch one lane's program may carve. Was 512 MiB, which a
/// single-row sampler never approaches; a block-diffusion denoiser reads
/// out 256 rows of a 262 144-wide vocabulary, and its epilogue names a
/// dozen `[256, vocab]` f32 rectangles (3.5 GB laid out one slot each).
/// The stride is a `u32` downstream, so this stops just short of 4 GiB.
/// [`layout_reusing`] brings such a program back under it by handing a
/// dead value's slot on; [`layout`] is the one-slot-each form.
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

/// One value's life on a launch clock: defined at step `def`, last read at
/// step `last` (inclusive). A step is whatever the shell can order — a
/// region launch, or one node of a region whose nodes run one block per
/// lane with a barrier between them. A value nothing reads has
/// `last == def`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lifetime {
    pub def: u32,
    pub last: u32,
    /// Whether this value may take an offset an earlier value vacated. Only
    /// a result its op always writes in full qualifies: a never-reused slot
    /// reads back as zero (the per-fire clear), a reused one as whatever
    /// died there, so a result that may go unwritten (a predicated pivot, a
    /// channel materialisation) keeps a fresh slot.
    pub reusable: bool,
    /// The launch (a kernel) in which the value is defined, and the one in
    /// which it is last read. Launches are ordered; steps within one may
    /// run on many blocks at once.
    pub launch_def: u32,
    pub launch_last: u32,
    /// How the blocks of a many-block launch touch the value — in the
    /// launch that defines it (`class_def`) and the one that last reads it
    /// (`class_last`). Two values of one launch may share bytes only when
    /// their classes match and are non-zero: then every block touches the
    /// same disjoint slice of both (its row, its element). Class 0 is a
    /// value every block reads whole — a scalar, a constant — and shares
    /// with nothing in its launch: a fast block would overwrite what a slow
    /// one still reads. A one-block launch runs its steps in order, so all
    /// its values are one class ([`Lifetime::SEQUENTIAL`]).
    pub class_def: u64,
    pub class_last: u64,
}

impl Lifetime {
    /// The class of every value in a one-block launch.
    pub const SEQUENTIAL: u64 = u64::MAX;
}

/// A vacated span: where, how wide, the launch that last read its value
/// and that value's class there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Vacant {
    offset: u64,
    span: u64,
    launch: u32,
    class: u64,
}

/// Best-fit vacant span of at least `span` bytes a value defined in
/// `launch` with `class` may take: one vacated by an earlier launch, or by
/// this launch's own values of the same non-zero class. Split if wider,
/// the remainder keeping the span's provenance; `None` when nothing fits.
fn take(free: &mut Vec<Vacant>, span: u64, launch: u32, class: u64) -> Option<u64> {
    let best = free
        .iter()
        .enumerate()
        .filter(|(_, block)| {
            block.span >= span
                && (block.launch < launch || (class != 0 && block.class == class))
        })
        .min_by_key(|(_, block)| block.span)
        .map(|(index, _)| index)?;
    let Vacant { offset, span: width, .. } = free[best];
    if width == span {
        free.remove(best);
    } else {
        free[best].offset = offset + span;
        free[best].span = width - span;
    }
    Some(offset)
}

/// Return `[offset, offset + span)` to the free list, coalescing with a
/// neighbour of the same provenance; the list stays sorted by offset.
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

/// [`layout`], reusing a value's bytes once the last step that reads it
/// has run. `temporary_floor` is the least the temporary arena may be — the
/// shell's own sum for the blocks that share it (a row-parallel region's
/// blocks each take `temporary_bytes / rows` of it). Steps run in order, so a value whose `last` step precedes
/// another's `def` step is dead before the other is written; nothing
/// defined at the same step as a value's last read may share with it.
/// Widest-first within a step keeps the free list packed. Falls back to
/// [`layout`] when `lifetimes` does not cover every descriptor.
///
/// # Errors
///
/// As [`layout`].
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
            // Entering `life.def`: whatever was last read strictly before it
            // is vacant.
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
        let descriptor = &descriptors[i];
        // Every CUDA consumer of the temporary arena works one row at a
        // time (`m1_reduce_*`, `ptir_parallel_reduce_f32`, the order
        // kernels: `work[i]` for `i < last`), so it is sized by the widest
        // ROW, not the widest value as [`layout`] does; a `[256, 262144]`
        // f32 asks 4 MiB here, not 1 GiB.
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

    /// One launch per step, every value sequential.
    fn life(def: u32, last: u32) -> Lifetime {
        Lifetime {
            def,
            last,
            reusable: true,
            launch_def: def,
            launch_last: last,
            class_def: Lifetime::SEQUENTIAL,
            class_last: Lifetime::SEQUENTIAL,
        }
    }

    /// A value of one many-block launch (steps `def..=last` inside it).
    fn rowed(def: u32, last: u32, launch: u32, class: u64) -> Lifetime {
        Lifetime {
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
    fn a_value_dead_before_the_next_region_hands_its_slot_on() {
        // a: r0..r1, b: r1..r2, c: r2 — c can take a's slot, not b's.
        let descriptors = [desc(1024), desc(1024), desc(1024)];
        let lifetimes = [life(0, 1), life(1, 2), life(2, 2)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0).unwrap();
        let naive = layout(&descriptors).unwrap();
        assert_eq!(reused.values[2], reused.values[0]);
        assert_ne!(reused.values[1], reused.values[0]);
        assert!(reused.total < naive.total);
    }

    #[test]
    fn a_value_last_read_where_another_is_defined_does_not_share() {
        let descriptors = [desc(1024), desc(1024)];
        let lifetimes = [life(0, 1), life(1, 1)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0).unwrap();
        assert_ne!(reused.values[0], reused.values[1]);
    }

    #[test]
    fn within_a_many_block_launch_only_one_class_shares() {
        // Same launch: a row-sliced value (class 7) dead at step 0 hands its
        // slot to a class-7 value at step 1, not to a class-3 one, and never
        // to a whole-value (class 0) one; the next launch takes anything.
        let descriptors = [desc(1024), desc(1024), desc(1024), desc(1024), desc(1024)];
        let lifetimes = [
            rowed(0, 0, 0, 7),
            rowed(1, 1, 0, 7),
            rowed(2, 2, 0, 3),
            rowed(3, 3, 0, 0),
            rowed(4, 4, 1, 0),
        ];
        let reused = layout_reusing(&descriptors, &lifetimes, 0).unwrap();
        assert_eq!(reused.values[1], reused.values[0], "same class, same launch");
        assert_ne!(reused.values[2], reused.values[0], "another class");
        assert_ne!(reused.values[2], reused.values[1]);
        assert_ne!(reused.values[3], reused.values[0], "a whole value shares with nothing");
        assert_eq!(reused.values[4], reused.values[0], "the next launch may take it");
    }

    #[test]
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

    #[test]
    fn a_wider_taker_gets_a_coalesced_pair_of_slots() {
        // a and b (adjacent, both dead after r0) merge to fit c's double width.
        let descriptors = [desc(1024), desc(1024), desc(2048)];
        let lifetimes = [life(0, 0), life(0, 0), life(1, 1)];
        let reused = layout_reusing(&descriptors, &lifetimes, 0).unwrap();
        assert_eq!(reused.values[2], reused.values[0].min(reused.values[1]));
    }

    #[test]
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

    #[test]
    fn the_temporary_floor_lifts_the_arena() {
        let reused = layout_reusing(&[desc(1024)], &[life(0, 0)], 1 << 20).unwrap();
        assert_eq!(reused.temporary_bytes, 1 << 20);
    }

    #[test]
    fn mismatched_lifetimes_fall_back_to_the_naive_layout() {
        let descriptors = [desc(1024), desc(1024)];
        assert_eq!(
            layout_reusing(&descriptors, &[life(0, 0)], 0).unwrap(),
            layout(&descriptors).unwrap()
        );
    }
}
