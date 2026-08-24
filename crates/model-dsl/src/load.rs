//! Import: checkpoint bytes to canonical zt, one production table per
//! flavor. Import may rewrite bytes; load may only view — the rank cut
//! happens at load, from the plan's shard column, never here.

pub trait Base {
    const NAME: &'static str;
}

pub trait SfBase: Base {}

pub enum SfBf16 {}

impl Base for SfBf16 {
    const NAME: &'static str = "safetensors-bf16";
}

impl SfBase for SfBf16 {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Import {
    pub base: &'static str,
    pub rows: Vec<Row>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub target: String,
    pub source: Source,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Copy(String),
    /// Concatenate along the output axis.
    Pack(Vec<Source>),
    /// Stack under a new leading axis.
    Stack(Vec<Source>),
    /// Ungroup `groups`-way interleaving ALONG ONE AXIS into contiguous
    /// segments: `g0 g1 g0 g1 ...` becomes `g0 g0 ... g1 g1 ...`, once for
    /// every position of whatever axes precede it.
    ///
    /// THE AXIS IS SPELLED, and it was a silent bug when it was not. This
    /// verb used to permute the LEADING axis and nothing else; gpt-oss's
    /// fused gate/up bank is `[experts, 2*inter, ..]` with the interleaving
    /// on axis 1, so the two rows that stated it were permuting the EXPERTS —
    /// a shuffle of which expert is which, on a tensor whose leading axis
    /// happens to divide by two. An axis that must be stated is an axis a
    /// reader can check.
    Deinterleave(String, u32, u32),
    /// Drop one extent-1 axis. A depthwise conv ships as
    /// `[channels, 1, width]` and the canonical weight is
    /// `[channels, width]` -- the kernel point states the width in the
    /// `[channels, width]` weight and nowhere else. The axis is spelled so
    /// the drop is checkable: an axis whose extent is not 1 is refused,
    /// which is what keeps this from becoming a blanket squeeze that would
    /// eat the leading `1` a `[1, hidden]` gate genuinely has.
    Squeeze(String, u32),
}

impl Import {
    #[must_use]
    pub fn new<B: Base>() -> Import {
        Import {
            base: B::NAME,
            rows: Vec::new(),
        }
    }

    pub fn write(&mut self, target: impl Into<String>, source: Source) {
        let target = target.into();
        assert!(
            self.rows.iter().all(|r| r.target != target),
            "`{target}` is produced twice"
        );
        self.rows.push(Row { target, source });
    }

    /// Every stored plane of one QUANTISED bank, in the repr's own order:
    /// mxfp4's packed codes under `target`, its block exponents under
    /// `<target>.scales`.
    ///
    /// A WRITE VERB AND NOT A `Source`, and the arithmetic of the import is
    /// why. A `Source` answers with ONE tensor; a bank at a quantised repr is
    /// two parameters with two names, two shapes and two allocations — so a
    /// paired `Source` would have to concatenate the planes into one row,
    /// which throws away both shapes, defeats the join's shape check and buys
    /// a split point somebody has to recompute on the device. What the
    /// pairing actually has to say is that these rows are ONE weight and that
    /// the suffix is not a convention each family gets to invent, and a write
    /// verb says exactly that.
    ///
    /// THE SUFFIXES COME OFF THE REPR, and this is the third reader of the
    /// same list rather than a third spelling of it: `Dtype::PLANE_SUFFIXES`
    /// states them, `Stmt::bank` records a statement's columns from them, and
    /// this names the production rows. A family's import table spells none.
    ///
    /// # Panics
    ///
    /// If the sources do not match the repr's plane count — an mxfp4 bank
    /// built from one tensor is a bank with no scales, which would load as
    /// silently un-dequantised bytes rather than as a refusal.
    pub fn bank<W: crate::axes::Dtype>(
        &mut self,
        target: impl Into<String>,
        planes: impl IntoIterator<Item = Source>,
    ) {
        let target = target.into();
        let planes: Vec<Source> = planes.into_iter().collect();
        let suffixes = W::PLANE_SUFFIXES;
        assert!(
            planes.len() == suffixes.len(),
            "`{target}` is a {} bank of {} plane(s) and this states {}",
            W::NAME,
            suffixes.len(),
            planes.len()
        );
        for (suffix, source) in suffixes.iter().zip(planes) {
            self.write(format!("{target}{suffix}"), source);
        }
    }
}

pub fn copy(name: impl Into<String>) -> Source {
    Source::Copy(name.into())
}

impl From<String> for Source {
    fn from(name: String) -> Source {
        Source::Copy(name)
    }
}

impl From<&str> for Source {
    fn from(name: &str) -> Source {
        Source::Copy(name.to_string())
    }
}

pub fn pack<I: IntoIterator>(sources: I) -> Source
where
    I::Item: Into<Source>,
{
    Source::Pack(sources.into_iter().map(Into::into).collect())
}

pub fn stack<I: IntoIterator>(sources: I) -> Source
where
    I::Item: Into<Source>,
{
    Source::Stack(sources.into_iter().map(Into::into).collect())
}

pub fn deinterleave(name: impl Into<String>, axis: u32, groups: u32) -> Source {
    Source::Deinterleave(name.into(), axis, groups)
}

pub fn squeeze(name: impl Into<String>, axis: u32) -> Source {
    Source::Squeeze(name.into(), axis)
}

/// One shipping import point: which SKU it produces, which checkpoint flavor
/// it reads, and the production run the CLI may execute for it.
///
/// `base` IS A COLUMN and not merely a field of whatever `make` returns: a
/// lookup that had to run the closure to learn which flavor it just built
/// would be choosing after the work, which is the wrong order. Every row
/// says `safetensors-bf16` today — gemma used to file each SKU twice, once
/// from GGUF, and that leg went because nothing could run it — but the
/// column is what a second flavor arrives through, and it is what all three
/// drivers filter on (`READABLE_BASE`) to say which they can read.
#[derive(Clone, Copy)]
pub struct ImportRow {
    pub sku: &'static str,
    pub base: &'static str,
    pub make: fn() -> Import,
}

/// State one shipping import point per row. The first type argument of the
/// production fn is its [`Base`], which is what puts the flavor in the key.
///
/// # THE DEGREE IS NOT SPELLED HERE, and the trailing `_` is why
///
/// A production fn's last generic parameter is the rank-cut degree its model
/// carries (`const TP: usize`), and this expands to `_` for it — inferred
/// from the model value in the same row. That is not a shortcut: a `-tp2`
/// import point differs from its sibling's by the MODEL it is given and by
/// nothing else, because a checkpoint holds the same bytes however a
/// deployment cuts them. Writing the degree in the turbofish as well would
/// state it twice in one line, with the two free to disagree — the fault the
/// whole tensor-parallel column was rebuilt to make impossible.
#[macro_export]
macro_rules! allow_import {
    ($( $f:ident::<$b:ty $(, $t:ty)* $(,)?> => ($sku:literal, $m:expr $(,)?) ),+ $(,)?) => {
        pub const IMPORTS: &[$crate::load::ImportRow] = &[ $(
            $crate::load::ImportRow {
                sku: $sku,
                base: <$b as $crate::load::Base>::NAME,
                make: || $f::<$b $(, $t)*, _>(&$m),
            }
        ),+ ];
    };
}
