//! Import: checkpoint bytes to canonical zt, one production table per
//! flavor. Import may rewrite bytes; load may only view — the rank cut
//! happens at load, from the plan's shard column, never here.

pub trait Base {
    const NAME: &'static str;
}

pub trait SfBase: Base {}

pub trait GgufBase: Base {}

pub enum SfBf16 {}

impl Base for SfBf16 {
    const NAME: &'static str = "safetensors-bf16";
}

impl SfBase for SfBf16 {}

pub enum GgufBf16 {}

impl Base for GgufBf16 {
    const NAME: &'static str = "gguf-bf16";
}

impl GgufBase for GgufBf16 {}

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
    /// The (1 + w) norm fold; the canonical weight is plain.
    PlusOne(String),
    /// Concatenate along the output axis.
    Pack(Vec<Source>),
    /// Stack under a new leading axis.
    Stack(Vec<Source>),
    /// A `[1]` tensor from a scalar the checkpoint stores beside `name`.
    ScalarOf(String),
    /// Ungroup `groups`-way row interleaving into contiguous segments.
    Deinterleave(String, u32),
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
}

pub fn copy(name: impl Into<String>) -> Source {
    Source::Copy(name.into())
}

pub fn plus_one(name: impl Into<String>) -> Source {
    Source::PlusOne(name.into())
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

pub fn scalar_of(name: impl Into<String>) -> Source {
    Source::ScalarOf(name.into())
}

pub fn deinterleave(name: impl Into<String>, groups: u32) -> Source {
    Source::Deinterleave(name.into(), groups)
}

/// One shipping import point: the SKU, and the production run the CLI may
/// execute for it. The produced table names its own flavor.
#[macro_export]
macro_rules! allow_import {
    ($( $f:path => ($sku:literal, $m:expr $(,)?) ),+ $(,)?) => {
        pub const IMPORTS: &[(&str, fn() -> $crate::load::Import)] =
            &[ $( ($sku, || $f(&$m)) ),+ ];
    };
}
