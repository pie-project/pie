use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_SYMBOL_BYTES: usize = 128;

/// A versioned open-vocabulary name resolved to a compact handle at attachment.
///
/// Symbols use the form `namespace.name@version`, for example
/// `pie.schedule.token-budget@1`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Symbol(String);

impl Symbol {
    pub fn new(value: impl Into<String>) -> Result<Self, SymbolError> {
        let symbol = Self(value.into());
        symbol.validate()?;
        Ok(symbol)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn validate(&self) -> Result<(), SymbolError> {
        if self.0.is_empty() {
            return Err(SymbolError::Empty);
        }
        if self.0.len() > MAX_SYMBOL_BYTES {
            return Err(SymbolError::TooLong {
                actual: self.0.len(),
                maximum: MAX_SYMBOL_BYTES,
            });
        }

        let (name, version) = self.0.rsplit_once('@').ok_or(SymbolError::MissingVersion)?;
        if name.is_empty() {
            return Err(SymbolError::EmptyName);
        }
        if !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b'/'))
        {
            return Err(SymbolError::InvalidName);
        }
        if version.is_empty() || !version.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err(SymbolError::InvalidVersion);
        }
        Ok(())
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SymbolError {
    #[error("symbol is empty")]
    Empty,
    #[error("symbol is {actual} bytes; maximum is {maximum}")]
    TooLong { actual: usize, maximum: usize },
    #[error("symbol must end with @<numeric-version>")]
    MissingVersion,
    #[error("symbol name is empty")]
    EmptyName,
    #[error("symbol name contains unsupported characters")]
    InvalidName,
    #[error("symbol version must be a non-empty unsigned integer")]
    InvalidVersion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DependencyRequirement {
    Required,
    Optional,
}

/// Invocation mode supported by the first contract version.
///
/// Candidate-local standing indexes are intentionally deferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InvocationMode {
    SetDependent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityDeclaration {
    pub name: Symbol,
    pub requirement: DependencyRequirement,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EventDeclaration {
    pub name: Symbol,
    pub requirement: DependencyRequirement,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_versioned_symbols() {
        assert!(Symbol::new("acme.workflow-id@1").is_ok());
        assert!(Symbol::new("pie/schedule.token_budget@12").is_ok());
    }

    #[test]
    fn rejects_unversioned_or_invalid_symbols() {
        assert_eq!(
            Symbol::new("acme.workflow-id").unwrap_err(),
            SymbolError::MissingVersion
        );
        assert_eq!(
            Symbol::new("acme workflow@1").unwrap_err(),
            SymbolError::InvalidName
        );
        assert_eq!(
            Symbol::new("acme.workflow@v1").unwrap_err(),
            SymbolError::InvalidVersion
        );
    }
}
