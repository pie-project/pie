//! THE FACT VOCABULARY: the closed set of names a text may state, and the one
//! derivation that answers them for a fire.
//!
//! # Why the word is derived HERE and not by the text that declared it
//!
//! A text says what its facts mean in Rust — `qo_one: r.query_len() == 1` —
//! and that sentence cannot reach an executor, because what crosses is a
//! [`crate::plan::Plan`] and a plan carries fact NAMES and not a type. So
//! the declaration was never the live derivation: four executors each kept a
//! hand match on the name string, and they disagreed with the declaration
//! about what the fact even ranges over (the text's is per REQUEST, an
//! executor's is per FIRE — "this is a decode", one answer for the whole
//! batch).
//!
//! One vocabulary and one [`word_of`] is what that measurement leaves. A name
//! not on this list is a REFUSAL and not a clear bit: a text stating a fact
//! nothing can answer would have its lanes picked by a guess, and the lane is
//! the whole program.
//!
//! # The census
//!
//! Two names, and both are on the list because a shipping text states them:
//! [`QO_ONE`] on all sixteen catalog rows, [`MASKED`] on gemma-4's three.

use crate::plan::{FireClass, Plan};

/// EVERY REQUEST'S QUERY IS ONE TOKEN — which is what a decode fire is.
///
/// The fact ranges over the FIRE and not over a request: a driver picks one
/// lane for the batch, so the bit is the class the batch was assembled as,
/// and the class is what named it.
pub const QO_ONE: &str = "qo_one";

/// THE FIRE CARRIES A CUSTOM ATTENTION MASK the caller staged.
///
/// Nothing derives this one — the frame states it — and a text that does not
/// declare it has ONE attention arm and it is causal, so a masked frame
/// reaching such a text is refused rather than answered.
pub const MASKED: &str = "masked";

/// The closed vocabulary, in no particular order: a plan's `facts` column is
/// its own bit ordering and this is only the set of spellings.
pub const NAMES: &[&str] = &[QO_ONE, MASKED];

/// The fact word a fire of `class` sets on `plan`, computed off `plan.facts`
/// rather than assumed: bit `i` is `plan.facts[i]`.
///
/// # Errors
///
/// A plan over more than 64 facts, a fact outside [`NAMES`], or a masked
/// frame against a text that states no [`MASKED`] fact. All three are the
/// same refusal in different clothes — the lane a word picks IS the program,
/// so a bit this cannot answer is never a bit it clears.
pub fn word_of(plan: &Plan, class: FireClass, masked: bool) -> Result<u64, String> {
    let mut word = 0u64;
    for (bit, fact) in plan.facts.iter().enumerate() {
        if bit >= 64 {
            return Err(format!(
                "`{}` declares {} facts; a fact word is 64 bits",
                plan.name,
                plan.facts.len()
            ));
        }
        let holds = match fact.as_str() {
            QO_ONE => class == FireClass::Decode,
            MASKED => masked,
            other => {
                return Err(format!(
                    "`{}` states `{other}`, which is not a fact this floor can \
                     answer for a {} fire; name it in `model_ir::facts` or the \
                     lane is a guess",
                    plan.name,
                    class.suffix()
                ));
            }
        };
        if holds {
            word |= 1 << bit;
        }
    }
    if masked && !plan.facts.iter().any(|f| f == MASKED) {
        return Err(format!(
            "this frame carries a user attention mask and `{}` states no \
             `masked` fact, so every lane it has attends causally: the mask \
             would be staged and IGNORED, and the request answered as though \
             it had asked nothing",
            plan.name
        ));
    }
    Ok(word)
}
