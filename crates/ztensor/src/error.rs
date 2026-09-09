use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Rule {
    FileTooSmall,
    HeaderMagic,
    FooterMagic,
    Version,
    ManifestBounds,
    ManifestTooLarge,
    ManifestHash,
    CborSyntax,
    CborDeterminism,
    CborDuplicateKey,
    CborDepth,
    Schema,
    Name,
    Shape,
    BlobAlignment,
    BlobBounds,
    BlobOverlap,
    ShardRef,
    ShardName,
    Type,
    LayoutRule,
    LayoutData,
    Size,
    Encoding,
    Digest,
    ShardIdentity,
    NameCollision,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Reject { rule: Rule, detail: String },
    NotFound(String),
    Unsupported(String),
    InvalidInput(String),
    Io(std::io::Error),
}

impl Error {
    pub fn reject(rule: Rule, detail: impl Into<String>) -> Self {
        Error::Reject {
            rule,
            detail: detail.into(),
        }
    }

    pub fn rule(&self) -> Option<Rule> {
        match self {
            Error::Reject { rule, .. } => Some(*rule),
            _ => None,
        }
    }

    pub(crate) fn at(self, name: &str) -> Error {
        match self {
            Error::Reject { rule, detail } => Error::Reject {
                rule,
                detail: format!("{name:?}: {detail}"),
            },
            other => other,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Reject { rule, detail } => write!(f, "rejected ({rule:?}): {detail}"),
            Error::NotFound(what) => write!(f, "not found: {what}"),
            Error::Unsupported(what) => write!(f, "unsupported: {what}"),
            Error::InvalidInput(what) => write!(f, "invalid input: {what}"),
            Error::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
