use serde::{Deserialize, Serialize};

use crate::guard::Guard;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ValueId(pub u32);

pub use dtype::Dtype;
pub use dtype::{BIASES, SCALES, TILED_BAND, TILED_STEP};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    Const(u64),
    Tokens,
    TokensTimes(u32),
    Lanes,
    LanesPlus(u32),
    Readouts,
    Patches,
    Images,
    ImagesPlus(u32),
    Voxels,
    VoxelsTimes(u32),
    Clips,
    ClipsPlus(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RowAxis {
    Tokens,
    Patches,
    Voxels,
}

impl RowAxis {
    pub const PRIMARY: RowAxis = RowAxis::Tokens;

    pub const ALL: [RowAxis; 3] = [RowAxis::Tokens, RowAxis::Patches, RowAxis::Voxels];

    pub const COUNT: usize = RowAxis::ALL.len();

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            RowAxis::Tokens => "tokens",
            RowAxis::Patches => "patches",
            RowAxis::Voxels => "voxels",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PerAxis<T>([T; RowAxis::COUNT]);

impl<T> PerAxis<T> {
    pub const fn new(values: [T; RowAxis::COUNT]) -> PerAxis<T> {
        PerAxis(values)
    }

    pub fn from_fn(mut f: impl FnMut(RowAxis) -> T) -> PerAxis<T> {
        PerAxis(RowAxis::ALL.map(&mut f))
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> PerAxis<U> {
        PerAxis(self.0.map(&mut f))
    }

    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn iter(&self) -> impl Iterator<Item = (RowAxis, &T)> {
        RowAxis::ALL.into_iter().zip(self.0.iter())
    }
}

impl<T: Clone> PerAxis<T> {
    #[must_use]
    pub fn splat(value: T) -> PerAxis<T> {
        PerAxis::from_fn(|_| value.clone())
    }
}

impl<T> core::ops::Index<RowAxis> for PerAxis<T> {
    type Output = T;

    fn index(&self, axis: RowAxis) -> &T {
        &self.0[axis as usize]
    }
}

impl<T> core::ops::IndexMut<RowAxis> for PerAxis<T> {
    fn index_mut(&mut self, axis: RowAxis) -> &mut T {
        &mut self.0[axis as usize]
    }
}

impl Dim {
    #[must_use]
    pub fn axis(self) -> Option<RowAxis> {
        match self {
            Dim::Const(_) => None,
            Dim::Tokens | Dim::TokensTimes(_) | Dim::Lanes | Dim::LanesPlus(_) | Dim::Readouts => {
                Some(RowAxis::Tokens)
            }
            Dim::Patches | Dim::Images | Dim::ImagesPlus(_) => Some(RowAxis::Patches),
            Dim::Voxels | Dim::VoxelsTimes(_) | Dim::Clips | Dim::ClipsPlus(_) => {
                Some(RowAxis::Voxels)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StructKind {
    AttnDecodePlan,
    AttnPrefillPlan,
    AttnPrefillPlanSm90,
    MlaPlan,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeomKind {
    Indptr,
    Indices,
    SeqLens,
    LastPageLen,
    KvLen,
    RowValid,
    RequestOfToken,
    WritePage,
    WriteOffset,
    GroupOfLane,
    GroupIndptr { select: Selection },
    LaneIndptr { select: Selection },
    ReferenceTag { select: Selection },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Selection {
    pub mask: u32,
    pub value: u32,
}

impl Selection {
    pub const ALL: Selection = Selection { mask: 0, value: 0 };

    #[must_use]
    pub fn of(guard: &Guard) -> Option<Selection> {
        let mut select = Selection::ALL;
        select.gather(guard).then_some(select)
    }

    fn gather(&mut self, guard: &Guard) -> bool {
        match guard {
            Guard::Always => true,
            Guard::Fact(bit) => self.literal(*bit, true),
            Guard::Not(inner) => match inner.as_ref() {
                Guard::Fact(bit) => self.literal(*bit, false),
                _ => false,
            },
            Guard::And(a, b) => self.gather(a) && self.gather(b),
            Guard::Or(..) => false,
        }
    }

    fn literal(&mut self, bit: u8, set: bool) -> bool {
        if bit >= 32 {
            return false;
        }
        let one = 1u32 << bit;
        let want = if set { one } else { 0 };
        if self.mask & one != 0 && self.value & one != want {
            return false;
        }
        self.mask |= one;
        self.value |= want;
        true
    }

    #[must_use]
    pub fn holds(self, word: u64) -> bool {
        (word as u32) & self.mask == self.value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuntimeInput {
    Tokens,
    Positions,
    MropePositions,
    Mask { space: u32 },
    Geometry { space: u32, kind: GeomKind },
    ReadoutRows,
    AdapterRoutes,
    Patches,
    PatchSegments,
    PatchRoutes,
    PatchPositions,
    PatchEmbedRows,
    PatchEmbedWeights,
    SelfCondRows,
    SelfCondWeights,
    RowPermutation { select: Selection },
    Latents { port: u8, width: u32 },
    LaneVector { port: u8, width: u32 },
    Context { port: u8, width: u32 },
    AxisPositions { port: u8, axes: u8 },
    Grid,
    Voxels { port: u8, channels: u32 },
    TokenGrid { p: [u32; 3] },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ty {
    Tensor { shape: Vec<Dim>, dtype: Dtype },
    Struct(StructKind),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Def {
    Input(RuntimeInput),
    Weight(u32),
    Cache(u32),
    Op(u32),
    Merge(Vec<(ValueId, Guard)>),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueDecl {
    pub def: Def,
    pub ty: Ty,
}

#[cfg(test)]
mod tests {
    use super::{Dim, Guard, PerAxis, RowAxis, Selection};

    #[test]
    fn value_every_case() {
        a_selection_is_a_split_arms_guard_as_two_words();
        a_per_axis_reads_back_what_each_axis_was_filled_with();
    }

    fn a_selection_is_a_split_arms_guard_as_two_words() {
        let arm = Guard::and(Guard::Fact(3), Guard::not(Guard::Fact(5)));
        let select = Selection::of(&arm).expect("a conjunction of literals");
        assert_eq!(
            select,
            Selection {
                mask: 0b101000,
                value: 0b001000
            }
        );
        for word in 0..64u64 {
            assert_eq!(select.holds(word), arm.holds(word), "word {word:#b}");
        }
        assert_eq!(Selection::of(&Guard::Always), Some(Selection::ALL));
        assert!(Selection::ALL.holds(u64::MAX));
        assert_eq!(
            Selection::of(&Guard::or(Guard::Fact(0), Guard::Fact(1))),
            None
        );
        assert_eq!(
            Selection::of(&Guard::and(Guard::Fact(0), Guard::not(Guard::Fact(0)))),
            None,
            "a contradiction admits no lane and is not a selection"
        );
    }

    fn a_per_axis_reads_back_what_each_axis_was_filled_with() {
        let mut table = PerAxis::new(["tokens", "patches", "voxels"]);
        assert_eq!(table[RowAxis::Tokens], "tokens");
        assert_eq!(table[RowAxis::Patches], "patches");
        assert_eq!(table.as_slice().len(), RowAxis::COUNT);

        for (at, axis) in RowAxis::ALL.into_iter().enumerate() {
            assert_eq!(table.as_slice()[at], axis.name());
        }

        table[RowAxis::Patches] = "second";
        assert_eq!(table[RowAxis::Patches], "second");
        assert_eq!(
            table[RowAxis::Tokens],
            "tokens",
            "the other axis stood still"
        );

        let named = PerAxis::from_fn(RowAxis::name);
        assert_eq!(named[RowAxis::Tokens], "tokens");
        assert_eq!(named[RowAxis::Patches], "patches");
        assert_eq!(named[RowAxis::Voxels], "voxels");

        let cut = PerAxis::new([10u32, 20, 30]);
        for (dim, want) in [
            (Dim::Tokens, 10),
            (Dim::TokensTimes(2), 10),
            (Dim::Lanes, 10),
            (Dim::LanesPlus(1), 10),
            (Dim::Patches, 20),
            (Dim::Images, 20),
            (Dim::ImagesPlus(1), 20),
            (Dim::Voxels, 30),
            (Dim::VoxelsTimes(8), 30),
            (Dim::Clips, 30),
            (Dim::ClipsPlus(1), 30),
        ] {
            assert_eq!(
                cut[dim.axis().expect("a symbolic dim names a row space")],
                want
            );
        }
        assert_eq!(
            Dim::Const(8).axis(),
            None,
            "a fixed block belongs to no axis"
        );
    }
}
