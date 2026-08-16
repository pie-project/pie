//! One family's tensor names, in every vocabulary that spells them.
//!
//! A weight has as many names as there are people who wrote it down.
//! llama.cpp calls it `blk.7.attn_q.weight`, HuggingFace calls it
//! `model.layers.7.self_attn.q_proj.weight`, and pie's artifact calls it --
//! today -- the same thing HuggingFace does. This module is the type a
//! generation states all of them in at once, as columns of one table.
//!
//! # Why the columns are in one table and not in one map per direction
//!
//! They were two maps, and only one of them existed. `<generation>/import.rs`
//! held a GGUF-to-HuggingFace table for the four families llama.cpp ships,
//! and the HuggingFace direction held nothing at all, because pie's artifact
//! is spelled HuggingFace's way and a map from a vocabulary to itself has no
//! rows to write.
//!
//! That absence was load-bearing in the wrong direction. "pie's names are
//! HuggingFace's" is a CHOICE, and it was recorded in four places at once:
//! the empty HuggingFace arm, the target of every GGUF map (a function
//! literally named `hf_name`), the bytes of the artifact, and the prefix
//! rules in [`crate::manifest::Observed::logical`]. Four places is what a
//! choice looks like when nobody owns it.
//!
//! One table with a `pie` column owns it. The column is equal to `hf` in
//! every row this build ships, and it says so rather than being silent about
//! it -- which is the difference between a fact and a coincidence.
//!
//! # The strictness follows the data, not the format
//!
//! An importer that meets a name its table does not have must either pass it
//! through or refuse it, and which one is right is not a property of the
//! FILE FORMAT. It is a property of the table:
//!
//! * If [`Vocab::respells`] is false -- every `pie` equals its `hf` -- then a
//!   name the table missed is ALREADY in pie's vocabulary, and passing it
//!   through loses nothing. This is every family today, which is why this
//!   whole module changes no byte of any artifact.
//! * The moment one `pie` column is edited, that family's table starts
//!   refusing what it cannot place, by name. It has to: an artifact where
//!   `q_proj` was respelled and `up_proj` was passed through is half in each
//!   vocabulary, and nothing downstream can tell which half.
//!
//! So the table is not a formality that a future rename would have to be
//! remembered alongside. It is the switch that rename throws, and it throws
//! it in the same edit. Nobody has to remember.
//!
//! GGUF refuses either way, and for a reason that is about the data too: a
//! `blk.3.attn_q` passed through is a tensor under a name no row can ever
//! match, so there is no version of "lose nothing" available.
//!
//! # Why the entries are whole names
//!
//! The four tables this replaces stored member STEMS -- `attn_q` against
//! `self_attn.q_proj` -- and rebuilt the surroundings in code:
//! `format!("model.layers.{layer}.{member}")`, with `output` carved out by
//! hand because `lm_head` does not sit under `model.`.
//!
//! Those surroundings are vocabulary too, and two of the nineteen row-bearing
//! generations do not share them: `nemotron_h` publishes
//! `backbone.layers.{}.mixer.q_proj` and `csm` publishes five towers with
//! five prefixes. A stem table cannot hold either without a second knob for
//! the prefix, and a knob whose value differs per family is a column that has
//! not admitted it is one.
//!
//! Whole names cost repetition -- `model.layers.{layer}.` on every row -- and
//! buy that no part of a name is stated anywhere but in the table. The `{}`
//! carve-out disappears with them: `lm_head` is a row like any other.

/// One tensor of one family, in every vocabulary that names it.
///
/// The strings are whole names with `{layer}` standing for the decoder index
/// and `{expert}` for a routed expert's, and with the `.weight` / `.bias`
/// suffix left off -- a suffix means the same thing in all three vocabularies
/// and rides along untouched, which is why it is not worth a column.
#[derive(Clone, Copy, Debug)]
pub struct Member {
    /// pie's own name for it, as the artifact holds it.
    ///
    /// The column that makes the artifact's spelling something pie decides
    /// rather than something it inherited. Equal to [`Self::hf`] in every row
    /// shipped today.
    pub pie: &'static str,
    /// HuggingFace's name for it, as the checkpoint holds it.
    pub hf: &'static str,
    /// llama.cpp's, or `None` for a member no GGUF this build ingests carries.
    ///
    /// `None` is most rows of most families: llama.cpp ships four of the
    /// nineteen generations here, and the rest have no GGUF column to fill.
    /// It is not a gap to be filled in later by guessing -- an unmapped GGUF
    /// name is refused by name, which is the answer that can be acted on.
    pub gguf: Option<&'static str>,
}

impl Member {
    /// A member both foreign vocabularies spell the same way pie does.
    #[must_use]
    pub const fn same(name: &'static str) -> Self {
        Self {
            pie: name,
            hf: name,
            gguf: None,
        }
    }

    /// The same, and llama.cpp's spelling of it.
    #[must_use]
    pub const fn gguf(name: &'static str, gguf: &'static str) -> Self {
        Self {
            pie: name,
            hf: name,
            gguf: Some(gguf),
        }
    }
}

/// One generation's whole naming table.
///
/// Ordered, scanned, and small: the widest family here is 28 rows, and a
/// `HashMap` would cost a build and an allocation per import to beat a scan
/// over that. It also reads as a table this way, which is the point of
/// writing it out instead of deriving it.
#[derive(Clone, Copy, Debug)]
pub struct Vocab(pub &'static [Member]);

impl Vocab {
    /// Whether any row spells pie's name differently from HuggingFace's.
    ///
    /// False for every generation this build ships, and the whole of why
    /// [`Vocab::from_hf`] may pass an unknown name through. See the module
    /// doc: this is the switch, not a formality beside it.
    #[must_use]
    pub fn respells(&self) -> bool {
        self.0.iter().any(|m| m.pie != m.hf)
    }

    /// pie's name for a tensor a HuggingFace checkpoint spells `name`.
    ///
    /// `None` means the table has no row for it, which the caller turns into
    /// a pass-through or a refusal depending on [`Vocab::respells`].
    #[must_use]
    pub fn from_hf(&self, name: &str) -> Option<String> {
        self.translate(name, Suffix::Optional, |m| Some(m.hf))
    }

    /// pie's name for a tensor a GGUF spells `name`.
    ///
    /// `None` is a refusal in every caller: see the module doc for why this
    /// direction has no pass-through available to it.
    #[must_use]
    pub fn from_gguf(&self, name: &str) -> Option<String> {
        self.translate(name, Suffix::Required, |m| m.gguf)
    }

    /// The shared half: strip the suffix, find the row whose `from` column
    /// matches, fill pie's pattern with what the match captured.
    fn translate(
        &self,
        name: &str,
        policy: Suffix,
        from: fn(&Member) -> Option<&'static str>,
    ) -> Option<String> {
        let (stem, suffix) = match name.rsplit_once('.') {
            Some((stem, tail @ ("weight" | "bias"))) => (stem, Some(tail)),
            _ if policy == Suffix::Required => return None,
            _ => (name, None),
        };
        let member = self.0.iter().find_map(|m| {
            from(m)
                .and_then(|pattern| capture(pattern, stem))
                .map(|layer| (m, layer))
        })?;
        let (m, layer) = member;
        let mut pie = match layer {
            Some(index) => m.pie.replace("{layer}", &index.to_string()),
            None => m.pie.to_string(),
        };
        // `{expert}` survives the substitution on purpose: a stacked expert
        // tensor becomes MANY artifact tensors, and the count is the source's
        // leading extent -- a fact the file states and this table must not
        // duplicate. `Ingest::Unstack` takes the template from here and fills
        // it, and it spells the hole `{}`.
        pie = pie.replace("{expert}", "{}");
        Some(match suffix {
            Some(suffix) => format!("{pie}.{suffix}"),
            None => pie,
        })
    }
}

/// `blk.7.attn_q.weight` as `(7, "attn_q")` -- llama.cpp's layer split.
///
/// Shared rather than per-family because `TENSOR_NAMES` in llama.cpp takes no
/// architecture argument: the `blk.{bid}.` split is one fact about one file
/// format, and four generations had a private copy of it. The index is
/// PARSED, so a malformed `blk.x.attn_q` is no member rather than layer 0.
///
/// This answers the hooks that key on a MEMBER -- which tensors the converter
/// derived, which regroup, which fold a constant -- and not the naming, which
/// is [`Vocab::from_gguf`]'s.
#[must_use]
pub fn gguf_member(name: &str) -> Option<(u32, &str)> {
    let (stem, _) = name.rsplit_once('.')?;
    let rest = stem.strip_prefix("blk.")?;
    let (index, member) = rest.split_once('.')?;
    Some((index.parse().ok()?, member))
}

/// Whether a vocabulary suffixes every tensor it names.
///
/// Not a knob and not a formatting preference -- a measured fact about each
/// vocabulary, which is why it is stated here and not passed in by callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Suffix {
    /// llama.cpp's. Every one of the 291 tensors of
    /// `Qwen2.5-0.5B-Instruct-Q4_0.gguf` ends in `.weight` or `.bias`, and
    /// nothing else does. So a `blk.3.attn_q` with no suffix, or a
    /// `blk.3.attn_q.scales`, is not a tensor this map is merely missing --
    /// it is a file this map predates, and answering it with
    /// `self_attn.q_proj.scales` would invent a plane llama.cpp does not
    /// publish. Refusing sends the operator to the map.
    Required,
    /// HuggingFace's, which suffixes most tensors and not all. pie's own
    /// `openai--gpt-oss-20b.zt` carries `..._blocks` and `..._scales` as
    /// whole names, and reading `_blocks` as a suffix would leave the stem
    /// short by a segment and match nothing.
    Optional,
}

/// Match `stem` against a pattern holding at most one `{layer}`.
///
/// `Some(None)` is a pattern with no hole that matched exactly;
/// `Some(Some(i))` is a hole that captured layer `i`; `None` is no match.
///
/// The index is PARSED rather than accepted as any text, so that a malformed
/// `blk.x.attn_q` falls through to the unmapped case instead of matching with
/// a garbage capture -- the same care the four tables this replaces each took
/// separately in their own `split_layer`.
fn capture(pattern: &str, stem: &str) -> Option<Option<u32>> {
    let Some((head, tail)) = pattern.split_once("{layer}") else {
        return (pattern == stem).then_some(None);
    };
    let rest = stem.strip_prefix(head)?;
    let digits = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits == 0 || rest[digits..] != *tail {
        return None;
    }
    Some(Some(rest[..digits].parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::{Member, Vocab};

    const TABLE: Vocab = Vocab(&[
        Member::gguf(
            "model.layers.{layer}.self_attn.q_proj",
            "blk.{layer}.attn_q",
        ),
        Member::gguf("model.embed_tokens", "token_embd"),
        Member::gguf("lm_head", "output"),
        Member::same("model.layers.{layer}.mlp.experts.{expert}.gate_proj"),
    ]);

    /// Both directions of a layer member, index and suffix carried across.
    #[test]
    fn a_layer_member_keeps_its_index_and_its_suffix() {
        assert_eq!(
            TABLE.from_gguf("blk.7.attn_q.weight").as_deref(),
            Some("model.layers.7.self_attn.q_proj.weight")
        );
        assert_eq!(
            TABLE.from_gguf("blk.7.attn_q.bias").as_deref(),
            Some("model.layers.7.self_attn.q_proj.bias")
        );
        assert_eq!(
            TABLE
                .from_hf("model.layers.7.self_attn.q_proj.weight")
                .as_deref(),
            Some("model.layers.7.self_attn.q_proj.weight")
        );
    }

    /// A name with no layer index, and one that does not sit under `model.`.
    ///
    /// `lm_head` is the row the four stem tables each carved out by hand.
    /// Here it is a row, which is the whole argument for whole names.
    #[test]
    fn the_head_is_a_row_and_not_a_carve_out() {
        assert_eq!(
            TABLE.from_gguf("output.weight").as_deref(),
            Some("lm_head.weight")
        );
        assert_eq!(
            TABLE.from_gguf("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
    }

    /// A malformed index does not match with a garbage capture.
    #[test]
    fn a_layer_index_that_is_not_a_number_matches_nothing() {
        assert_eq!(TABLE.from_gguf("blk.x.attn_q.weight"), None);
        assert_eq!(TABLE.from_gguf("blk..attn_q.weight"), None);
    }

    /// llama.cpp suffixes everything, so a name that does not is refused.
    ///
    /// Both halves matter. `blk.3.attn_q` must not match the row it is a
    /// prefix of, and `.scales` must not be carried across as though
    /// llama.cpp published a plane -- see `Suffix::Required`.
    #[test]
    fn a_gguf_name_without_a_suffix_is_refused() {
        assert_eq!(TABLE.from_gguf("blk.3.attn_q"), None);
        assert_eq!(TABLE.from_gguf("token_embd"), None);
        assert_eq!(TABLE.from_gguf("blk.3.attn_q.scales"), None);
    }

    /// A tensor whose last segment is not a suffix is matched whole.
    ///
    /// pie's own gpt-oss artifact holds `..._blocks` and `..._scales`, and
    /// reading `_blocks` as a suffix would leave the stem short by a segment.
    #[test]
    fn a_plane_is_a_name_and_not_a_name_plus_a_suffix() {
        const PLANES: Vocab = Vocab(&[Member::same(
            "model.layers.{layer}.mlp.experts.gate_up_proj_blocks",
        )]);
        assert_eq!(
            PLANES
                .from_hf("model.layers.3.mlp.experts.gate_up_proj_blocks")
                .as_deref(),
            Some("model.layers.3.mlp.experts.gate_up_proj_blocks")
        );
    }

    /// The expert hole survives as `Ingest::Unstack`'s template.
    #[test]
    fn an_expert_index_is_left_for_the_unstacker() {
        assert_eq!(
            TABLE
                .from_hf("model.layers.2.mlp.experts.{expert}.gate_proj.weight")
                .as_deref(),
            Some("model.layers.2.mlp.experts.{}.gate_proj.weight")
        );
    }

    /// No shipped table respells, and the identity is stated rather than
    /// assumed.
    #[test]
    fn a_table_that_does_not_respell_says_so() {
        assert!(!TABLE.respells());
        const RESPELT: Vocab = Vocab(&[Member {
            pie: "layer.{layer}.attn.q",
            hf: "model.layers.{layer}.self_attn.q_proj",
            gguf: None,
        }]);
        assert!(RESPELT.respells());
        assert_eq!(
            RESPELT
                .from_hf("model.layers.4.self_attn.q_proj.weight")
                .as_deref(),
            Some("layer.4.attn.q.weight")
        );
    }
}
