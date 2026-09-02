//! The demand half of a tokenizer: what a serving row reads from the
//! vocabulary an artifact supplies. A [`Contract`] gathers a family's token
//! citations (stop markers, media delimiters, reserved ids) into one
//! declaration per serving row, checked once at serve boot rather than
//! panicking wherever a string is first resolved.
//!
//! Layered like the load contract: `checkpoint::contract` owns weight
//! demands (declared in `model/<family>/import.rs`), this module owns token
//! demands (declared in `model/<family>/tokenizer.rs`).
//!
//! Deliberately does not cover a grammar's own markers (e.g. ChatML's
//! `<|im_start|>`) — those already fail loudly at the same boot, and
//! restating them here would be two spellings of one dependency.

use crate::Tokenizer;

/// One serving row's demands on the vocabulary it is paired with. Built
/// entirely of `&'static` data so a family can declare it `const`, citing
/// the same constants its template and media code spell.
#[derive(Clone, Copy, Debug)]
pub struct Contract {
    /// Marker strings that must each spell as a single whole token. A slice
    /// of slices (not flat) so a declaration can cite the lists where they
    /// live, since `const` cannot concatenate slices.
    pub markers: &'static [&'static [&'static str]],
    /// Markers that must sit at exactly this id — for a row whose identity
    /// hangs on it. Checks existence at the id, not upstream "special"
    /// metadata this crate hasn't verified.
    pub pinned: &'static [(&'static str, u32)],
}

/// How a pairing of row and artifact breaks its tokenizer contract. Blame
/// points at the pairing, not the tokenizer: this serving row reads
/// something this artifact never carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The row reads a marker the vocabulary does not spell as one token.
    Missing { marker: &'static str },
    /// The row pins a marker at one id and the vocabulary holds it at another.
    Displaced {
        marker: &'static str,
        want: u32,
        found: u32,
    },
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing { marker } => write!(
                f,
                "the serving row reads a `{marker}` token and this artifact's \
                 tokenizer spells no such token; the row and the checkpoint \
                 are not each other's"
            ),
            Self::Displaced {
                marker,
                want,
                found,
            } => write!(
                f,
                "the serving row pins `{marker}` at id {want} and this \
                 artifact's tokenizer holds it at {found}; a pinned marker \
                 selects a reading of the artifact, and this artifact answers \
                 a different one"
            ),
        }
    }
}

impl std::error::Error for Fault {}

impl Contract {
    /// Check every demand against `tokenizer`, refusing at the first breach
    /// (not collecting all of them), so a boot refusal names one concrete
    /// disagreement to fix.
    pub fn verify(&self, tokenizer: &Tokenizer) -> Result<(), Fault> {
        for group in self.markers {
            for &marker in *group {
                if tokenizer.token_to_id(marker).is_none() {
                    return Err(Fault::Missing { marker });
                }
            }
        }
        for &(marker, want) in self.pinned {
            match tokenizer.token_to_id(marker) {
                None => return Err(Fault::Missing { marker }),
                Some(found) if found != want => {
                    return Err(Fault::Displaced {
                        marker,
                        want,
                        found,
                    });
                }
                Some(_) => {}
            }
        }
        Ok(())
    }
}

