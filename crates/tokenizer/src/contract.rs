//! **THE DEMAND HALF OF A TOKENIZER** — what a serving row reads from the
//! vocabulary an artifact supplies.
//!
//! The supply side is settled: an artifact carries its compiled tokenizer
//! whole ([`canonical`](crate::canonical), `pie.tokenizer/1`), and the same
//! artifact tokenizes the same way forever, whatever pie release replays it.
//! What had no statement was the demand: a family's code cites tokens by
//! name — a stop marker in a chat template, a media delimiter, a reserved id
//! a future door will key on — and every citation was a string resolved where
//! it was used, panicking at whatever moment first touched it. A [`Contract`]
//! is those citations gathered into one declaration per serving row, checked
//! once at serve boot, refusing by name before the first token.
//!
//! The layering is the load contract's exactly: `checkpoint::contract` owns
//! the language weights are demanded in and `model/<family>/import.rs`
//! declares in it; THIS module owns the language tokens are demanded in and
//! `model/<family>/tokenizer.rs` declares in it. The crate that holds the
//! [`Tokenizer`] holds the verifier, so checking a demand never widens the
//! public vocabulary API.
//!
//! **WHAT A CONTRACT DELIBERATELY DOES NOT COVER: a grammar's own markers.**
//! `ChatMLInstruct::new` resolves `<|im_start|>` because ChatML's grammar
//! needs it; that demand is the grammar's and it already fails loudly at the
//! same boot ([`chat-template`'s `special`]). A contract restating it would
//! be two spellings of one dependency, free to disagree. A contract carries
//! what the FAMILY spells: its stop list, its media delimiters, its pins.

use crate::Tokenizer;

/// One serving row's demands on the vocabulary it is paired with.
///
/// Built entirely of `&'static` data so a family can declare it `const`,
/// citing the same constants its template and media code spell — one
/// spelling per marker, cited everywhere it is read.
#[derive(Clone, Copy, Debug)]
pub struct Contract {
    /// Marker strings that must each spell as a SINGLE whole token —
    /// the property every citation site actually relies on: a stop list
    /// interrupts on one id, a placeholder run is scanned as one id.
    ///
    /// A slice of slices, not a flat list, so a declaration can cite the
    /// lists where they live (`STOP_TOKENS`, a delimiter triple) instead of
    /// restating their contents; `const` cannot concatenate slices.
    pub markers: &'static [&'static [&'static str]],
    /// Markers that must sit at exactly this id.
    ///
    /// The strong claim, for the row whose IDENTITY hangs on it: qwen3.8's
    /// audio specials are the one artifact-visible fact that tells a 3.8
    /// tokenizer from its 3.6 twin, so the 3.8 rows pin them. Existence is
    /// checked, not the added-token `special` flag — the flag is upstream
    /// metadata this crate has not verified for these tokens, and a pin that
    /// demanded more than was proven would refuse real artifacts.
    pub pinned: &'static [(&'static str, u32)],
}

/// How a pairing of row and artifact breaks its tokenizer contract.
///
/// Blame points at the PAIRING, not the tokenizer: every tokenizer here was
/// compiled faithfully from its own provenance, and the fault is that this
/// serving row reads something this artifact never carried.
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
    /// Check every demand against `tokenizer`, refusing at the first breach.
    ///
    /// First-fault and not a collection, following `checkpoint_dsl`'s own
    /// reads: a boot refusal names one concrete disagreement to fix.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn vocabulary(tokens: &[&str]) -> Tokenizer {
        Tokenizer::from_vocab(&tokens.iter().map(|t| (*t).to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn a_spelled_marker_and_a_true_pin_verify() {
        let tokenizer = vocabulary(&["<|im_end|>", "<|audio_start|>"]);
        let contract = Contract {
            markers: &[&["<|im_end|>"]],
            pinned: &[("<|audio_start|>", 1)],
        };
        assert_eq!(contract.verify(&tokenizer), Ok(()));
    }

    #[test]
    fn an_unspelled_marker_is_missing() {
        let tokenizer = vocabulary(&["<|im_end|>"]);
        let contract = Contract {
            markers: &[&["<|im_end|>"], &["<|vision_start|>"]],
            pinned: &[],
        };
        assert_eq!(
            contract.verify(&tokenizer),
            Err(Fault::Missing {
                marker: "<|vision_start|>"
            })
        );
    }

    #[test]
    fn a_marker_at_the_wrong_id_is_displaced() {
        let tokenizer = vocabulary(&["<|im_end|>", "<|audio_start|>"]);
        let contract = Contract {
            markers: &[],
            pinned: &[("<|audio_start|>", 248_070)],
        };
        assert_eq!(
            contract.verify(&tokenizer),
            Err(Fault::Displaced {
                marker: "<|audio_start|>",
                want: 248_070,
                found: 1
            })
        );
    }

    #[test]
    fn an_absent_pin_is_missing_not_displaced() {
        let tokenizer = vocabulary(&["<|im_end|>"]);
        let contract = Contract {
            markers: &[],
            pinned: &[("<|audio_start|>", 248_070)],
        };
        assert_eq!(
            contract.verify(&tokenizer),
            Err(Fault::Missing {
                marker: "<|audio_start|>"
            })
        );
    }
}
