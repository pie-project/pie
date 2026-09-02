use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

/// A launch program this host half cannot interpret. A struct rather than a
/// one-variant enum: no discriminant nobody reads, no forced `match` arm,
/// and room to grow a second field without changing kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "launch program cannot be interpreted: {}", self.message)
    }
}

impl std::error::Error for Error {}
