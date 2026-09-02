use std::fmt;

/// The spec rule a rejected file violated.
///
/// Tags correspond to sections of `spec/ztensor-v3-spec.md` so the
/// conformance corpus can assert exact rejection reasons. The set grows with
/// the spec, so it is `#[non_exhaustive]`: matching on one rule stays
/// source-compatible when the next is added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Rule {
    /// Shorter than a header plus a footer, so there is no container (§2.1).
    FileTooSmall,
    /// The first eight bytes are not the magic (§2.2).
    HeaderMagic,
    /// The last eight bytes are not the magic (§2.2).
    FooterMagic,
    /// The footer's version integer is one this build does not implement.
    Version,
    /// The manifest blob falls outside the data region, or a data shard's
    /// footer does not zero the manifest fields (§2.3, §7.2).
    ManifestBounds,
    /// The declared manifest length exceeds the 1 GiB cap (§3.1).
    ManifestTooLarge,
    /// The manifest bytes do not hash to what the footer claims (§2.3).
    ManifestHash,
    /// The manifest is not well-formed CBOR.
    CborSyntax,
    /// Well-formed CBOR, but not the deterministic encoding the format
    /// requires: a non-shortest head, an indefinite length, or unsorted map
    /// keys (§3.1).
    CborDeterminism,
    /// A CBOR map repeats a key (§3.1).
    CborDuplicateKey,
    /// CBOR nesting beyond the depth limit, the bound that keeps a hostile
    /// manifest from exhausting the stack.
    CborDepth,
    /// The manifest parses but is not the shape the schema describes: a
    /// missing field, a field of the wrong type, a value out of range (§3).
    Schema,
    /// An object or attribute name violates §3.5.
    Name,
    /// A shape exceeds the rank limit, or its element product overflows
    /// `u64` (§3.3).
    Shape,
    /// A blob offset is not on the alignment floor (§2.4).
    BlobAlignment,
    /// A blob runs past the end of the file's data region (§2.4).
    BlobBounds,
    /// Two blobs in one file overlap partially. Sharing is identical-or
    /// -disjoint, never in between (§2.4).
    BlobOverlap,
    /// A blob names a shard the manifest's table does not declare (§7.1).
    ShardRef,
    /// A shard name is outside the character set of §7.1.
    ShardName,
    /// An object's `type` is not a well-formed term, names an unknown leaf,
    /// or has group forms its shape cannot satisfy (§4).
    Type,
    /// Object metadata violates its layout's rules (§5.2).
    LayoutRule,
    /// Data-level violation of a layout or a leaf's content rules (bool
    /// bytes, packed tail bits, CSR index rules).
    LayoutData,
    /// A blob's decoded size is not what its shape and type (or its layout's
    /// size equation) require (§5.1).
    Size,
    /// Stored bytes violate their declared encoding profile.
    Encoding,
    /// The bytes do not hash to their digest (§6.1).
    Digest,
    /// A resolved shard does not match the identity in the root's table
    /// (§7.1, §7.3).
    ShardIdentity,
    /// Two sources of one composite claim the same tensor name.
    NameCollision,
}

#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The file violates a MUST of the spec and was rejected.
    Reject {
        rule: Rule,
        detail: String,
    },
    /// The named tensor does not exist.
    NotFound(String),
    /// The file is valid, but uses vocabulary or requires a capability this
    /// implementation does not support (an unregistered layout or encoding,
    /// a payload that cannot be addressed, ...). Refusal, never
    /// reinterpretation.
    Unsupported(String),
    /// Caller error on the write path.
    InvalidInput(String),
    Io(std::io::Error),
}

impl Error {
    /// Rejects a file under a spec rule.
    ///
    /// Public because a profile registered from another crate has to be able
    /// to refuse a file exactly as a built-in one does. A validator that can
    /// only say `InvalidInput` is a second-class validator.
    pub fn reject(rule: Rule, detail: impl Into<String>) -> Self {
        Error::Reject {
            rule,
            detail: detail.into(),
        }
    }

    /// The rule this error rejected under, if it is a rejection. Lets a
    /// consumer ask the question without matching the struct variant.
    pub fn rule(&self) -> Option<Rule> {
        match self {
            Error::Reject { rule, .. } => Some(*rule),
            _ => None,
        }
    }

    /// Prefixes a rejection's detail with the object it concerns.
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
