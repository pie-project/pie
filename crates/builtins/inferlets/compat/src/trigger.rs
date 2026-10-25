//! A grammar that holds only after a trigger: structural tags, as vLLM and
//! SGLang serve them for tool calling.
//!
//! Generation runs free until the text it makes contains one of the
//! triggers. The structures whose `begin` starts with that trigger are the
//! candidates: the rest of the `begin` is forced, then the structure's JSON
//! Schema holds, then its `end` is forced, and generation is free again,
//! watching for the next trigger. A structure's schema lets go as soon as
//! its grammar could end (an object at its closing brace), so a schema
//! whose documents can grow past a complete one (a bare number) is cut at
//! the first.
//!
//! The grammars only see tokens, never text. A trigger whose token runs on
//! past the whole `begin` into the schema leaves that occurrence free: what
//! it ran on with cannot be fed to the schema's matcher.

use crate::generate::Guide;
use crate::request::StructuralTag;

/// The host side a trigger needs: grammars, and the text tokens spell.
pub trait Grammars {
    type Matcher: Guide;
    /// A matcher over exactly one of `alternatives`.
    fn literal(&self, alternatives: &[&str]) -> Result<Self::Matcher, String>;
    /// A matcher over the documents `schema` (a JSON Schema) admits.
    fn schema(&self, schema: &str) -> Result<Self::Matcher, String>;
    fn decode(&self, tokens: &[u32]) -> String;
    /// The tokens a matcher allows exactly where its grammar could end.
    fn stops(&self) -> &[u32];
}

enum Phase<M> {
    /// Unconstrained; `tail` is the text since the last structure, as much
    /// of it as could still hold the start of a trigger.
    Free {
        tail: String,
    },
    /// The rest of a `begin`: `candidates` are the structures with their
    /// rests, `tokens` what was sampled toward them.
    Begin {
        candidates: Vec<(usize, String)>,
        matcher: M,
        tokens: Vec<u32>,
    },
    Schema {
        structure: usize,
        matcher: M,
    },
    /// A structure's `end`.
    End {
        matcher: M,
    },
}

/// The guide of a structural tag: free, then each triggered structure's
/// grammar, then free again.
pub struct Triggered<G: Grammars> {
    tag: StructuralTag,
    grammars: G,
    phase: Phase<G::Matcher>,
}

impl<G: Grammars> Triggered<G> {
    pub fn new(tag: StructuralTag, grammars: G) -> Self {
        Self {
            tag,
            grammars,
            phase: Phase::Free {
                tail: String::new(),
            },
        }
    }

    /// Whether a grammar holds the next token.
    pub fn constrained(&self) -> bool {
        !matches!(self.phase, Phase::Free { .. })
    }

    fn longest_trigger(&self) -> usize {
        self.tag.triggers.iter().map(String::len).max().unwrap_or(0)
    }

    /// `text` freshly generated in the free phase: a trigger in it opens
    /// a structure.
    fn free(&self, mut tail: String, text: &str) -> Result<Phase<G::Matcher>, String> {
        tail.push_str(text);
        let found = self
            .tag
            .triggers
            .iter()
            .filter_map(|trigger| tail.find(trigger.as_str()).map(|at| (at, trigger)))
            .min_by_key(|&(at, trigger)| (at, std::cmp::Reverse(trigger.len())));
        let Some((at, trigger)) = found else {
            // Only the end can still be the start of a trigger.
            let keep = self.longest_trigger().saturating_sub(1);
            let mut cut = tail.len().saturating_sub(keep);
            while !tail.is_char_boundary(cut) {
                cut -= 1;
            }
            return Ok(Phase::Free {
                tail: tail[cut..].to_string(),
            });
        };
        let opened = &tail[at..];
        let candidates: Vec<(usize, String)> = self
            .tag
            .structures
            .iter()
            .enumerate()
            .filter(|(_, s)| s.begin.starts_with(trigger.as_str()))
            .filter_map(|(i, s)| {
                s.begin
                    .strip_prefix(opened)
                    .map(|rest| (i, rest.to_string()))
            })
            .collect();
        self.begin(candidates)
    }

    /// Toward the rest of a candidate `begin`: a structure whose `begin`
    /// the trigger already spelled opens its schema; none left leaves the
    /// occurrence free.
    fn begin(&self, candidates: Vec<(usize, String)>) -> Result<Phase<G::Matcher>, String> {
        if let Some(&(structure, _)) = candidates.iter().find(|(_, rest)| rest.is_empty()) {
            return self.schema(structure);
        }
        if candidates.is_empty() {
            return Ok(Phase::Free {
                tail: String::new(),
            });
        }
        let rests: Vec<&str> = candidates.iter().map(|(_, rest)| rest.as_str()).collect();
        Ok(Phase::Begin {
            matcher: self.grammars.literal(&rests)?,
            candidates,
            tokens: Vec::new(),
        })
    }

    fn schema(&self, structure: usize) -> Result<Phase<G::Matcher>, String> {
        let matcher = self
            .grammars
            .schema(&self.tag.structures[structure].schema)?;
        Ok(Phase::Schema { structure, matcher })
    }

    /// Past a structure's schema: its `end`, or free where it has none.
    fn end(&self, structure: usize) -> Result<Phase<G::Matcher>, String> {
        let end = &self.tag.structures[structure].end;
        if end.is_empty() {
            return Ok(Phase::Free {
                tail: String::new(),
            });
        }
        Ok(Phase::End {
            matcher: self.grammars.literal(&[end.as_str()])?,
        })
    }

    /// Whether `matcher`'s grammar could end here.
    fn complete(&self, matcher: &mut G::Matcher) -> bool {
        let mask = matcher.mask();
        self.grammars
            .stops()
            .iter()
            .any(|&stop| inferlet::mask::bit_allowed(&mask, stop as usize))
    }
}

impl<G: Grammars> Guide for Triggered<G> {
    fn mask(&mut self) -> Vec<u32> {
        match &mut self.phase {
            Phase::Free { .. } => Vec::new(),
            Phase::Begin { matcher, .. }
            | Phase::Schema { matcher, .. }
            | Phase::End { matcher, .. } => matcher.mask(),
        }
    }

    fn accept(&mut self, token: u32) -> Result<(), String> {
        let phase = std::mem::replace(
            &mut self.phase,
            Phase::Free {
                tail: String::new(),
            },
        );
        self.phase = match phase {
            Phase::Free { tail } => {
                let text = self.grammars.decode(&[token]);
                self.free(tail, &text)?
            }
            Phase::Begin {
                candidates,
                mut matcher,
                mut tokens,
            } => {
                matcher.accept(token)?;
                tokens.push(token);
                let spelled = self.grammars.decode(&tokens);
                match candidates.iter().find(|(_, rest)| *rest == spelled) {
                    Some(&(structure, _)) => self.schema(structure)?,
                    None => {
                        let candidates = candidates
                            .into_iter()
                            .filter(|(_, rest)| rest.starts_with(spelled.as_str()))
                            .collect();
                        Phase::Begin {
                            candidates,
                            matcher,
                            tokens,
                        }
                    }
                }
            }
            Phase::Schema {
                structure,
                mut matcher,
            } => {
                matcher.accept(token)?;
                if self.complete(&mut matcher) {
                    self.end(structure)?
                } else {
                    Phase::Schema { structure, matcher }
                }
            }
            Phase::End { mut matcher } => {
                matcher.accept(token)?;
                if self.complete(&mut matcher) {
                    Phase::Free {
                        tail: String::new(),
                    }
                } else {
                    Phase::End { matcher }
                }
            }
        };
        Ok(())
    }

    fn is_terminated(&self) -> bool {
        false
    }
}

/// The host's grammars, over the loaded model's tokenizer.
pub struct Host {
    stops: Vec<u32>,
}

impl Host {
    pub fn new() -> Self {
        let mut stops = inferlet::chat::stop_tokens();
        stops.extend(inferlet::chat::seal());
        Self { stops }
    }
}

impl Default for Host {
    fn default() -> Self {
        Self::new()
    }
}

/// `text` as an EBNF string literal.
pub fn ebnf_literal(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl Grammars for Host {
    type Matcher = inferlet::grammar::Matcher;

    fn literal(&self, alternatives: &[&str]) -> Result<Self::Matcher, String> {
        let body: Vec<String> = alternatives.iter().map(|a| ebnf_literal(a)).collect();
        let grammar =
            inferlet::grammar::Grammar::from_ebnf(&format!("root ::= {}", body.join(" | ")))
                .map_err(|e| format!("structural tag literal: {e:?}"))?;
        Ok(inferlet::grammar::Matcher::new(&grammar))
    }

    fn schema(&self, schema: &str) -> Result<Self::Matcher, String> {
        let grammar = inferlet::grammar::Grammar::from_json_schema(schema)
            .map_err(|e| format!("structural tag schema: {e:?}"))?;
        Ok(inferlet::grammar::Matcher::new(&grammar))
    }

    fn decode(&self, tokens: &[u32]) -> String {
        inferlet::model::decode(tokens).unwrap_or_default()
    }

    fn stops(&self) -> &[u32] {
        &self.stops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::Structure;
    use inferlet::mask::{bit_allowed, pack_allowed};

    /// Bytes are their own tokens; this one ends the turn.
    const STOP: u32 = 256;
    /// Tokens of several bytes, past the bytes and the stop.
    const WORDS: &[(u32, &str)] = &[
        (300, "<function=get_weather>{"),
        (301, "<function=get"),
        (302, "</function>"),
    ];
    const VOCAB: usize = 400;

    fn spell(token: u32) -> String {
        match WORDS.iter().find(|(id, _)| *id == token) {
            Some((_, word)) => word.to_string(),
            None if token < 256 => (token as u8 as char).to_string(),
            None => String::new(),
        }
    }

    /// Every token whose text keeps `done` a prefix of something `ok`
    /// admits, and the stop where `end` says the grammar could end.
    fn allowed(done: &str, ok: impl Fn(&str) -> bool, end: bool) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..VOCAB as u32)
            .filter(|&t| t != STOP && !spell(t).is_empty() && ok(&format!("{done}{}", spell(t))))
            .collect();
        if end {
            ids.push(STOP);
        }
        pack_allowed(VOCAB, &ids)
    }

    /// One of some strings, or a flat JSON-ish object: `{`, anything but
    /// braces, `}`.
    enum Fake {
        Literal(Vec<String>, String),
        Object(String),
    }

    impl Fake {
        fn admits(&self, text: &str) -> bool {
            match self {
                Fake::Literal(alts, _) => alts.iter().any(|a| a.starts_with(text)),
                Fake::Object(_) => match text.strip_prefix('{') {
                    None => text.is_empty(),
                    Some(rest) => match rest.find('}') {
                        None => !rest.contains('{'),
                        Some(at) => at + 1 == rest.len() && !rest.contains('{'),
                    },
                },
            }
        }

        fn done(&self) -> &str {
            match self {
                Fake::Literal(_, done) | Fake::Object(done) => done,
            }
        }

        fn complete(&self) -> bool {
            match self {
                Fake::Literal(alts, done) => alts.contains(done),
                Fake::Object(done) => done.ends_with('}'),
            }
        }
    }

    impl Guide for Fake {
        fn mask(&mut self) -> Vec<u32> {
            allowed(self.done(), |t| self.admits(t), self.complete())
        }

        fn accept(&mut self, token: u32) -> Result<(), String> {
            let next = format!("{}{}", self.done(), spell(token));
            if !self.admits(&next) {
                return Err(format!("{next:?} is not admitted"));
            }
            match self {
                Fake::Literal(_, done) | Fake::Object(done) => *done = next,
            }
            Ok(())
        }

        fn is_terminated(&self) -> bool {
            false
        }
    }

    struct FakeGrammars;

    impl Grammars for FakeGrammars {
        type Matcher = Fake;

        fn literal(&self, alternatives: &[&str]) -> Result<Fake, String> {
            Ok(Fake::Literal(
                alternatives.iter().map(|a| a.to_string()).collect(),
                String::new(),
            ))
        }

        fn schema(&self, _schema: &str) -> Result<Fake, String> {
            Ok(Fake::Object(String::new()))
        }

        fn decode(&self, tokens: &[u32]) -> String {
            tokens.iter().map(|&t| spell(t)).collect()
        }

        fn stops(&self) -> &[u32] {
            &[STOP]
        }
    }

    fn tag(begins: &[&str], triggers: &[&str]) -> Triggered<FakeGrammars> {
        let structures = begins
            .iter()
            .map(|begin| Structure {
                begin: begin.to_string(),
                schema: "{}".into(),
                end: "</function>".into(),
            })
            .collect();
        let triggers = triggers.iter().map(|t| t.to_string()).collect();
        Triggered::new(
            StructuralTag {
                structures,
                triggers,
            },
            FakeGrammars,
        )
    }

    /// Feed `text` a byte a token, each through the mask it must pass.
    fn feed(guide: &mut Triggered<FakeGrammars>, text: &str) {
        for b in text.bytes() {
            feed_token(guide, b as u32);
        }
    }

    fn feed_token(guide: &mut Triggered<FakeGrammars>, token: u32) {
        let mask = guide.mask();
        assert!(
            mask.is_empty() || bit_allowed(&mask, token as usize),
            "{:?} is masked out",
            spell(token)
        );
        guide.accept(token).unwrap();
    }

    #[test]
    fn text_without_a_trigger_runs_free() {
        let mut guide = tag(&["<function=get_weather>"], &["<function="]);
        feed(&mut guide, "Let me think <func about it.");
        assert!(!guide.constrained());
        assert!(guide.mask().is_empty());
    }

    #[test]
    fn a_trigger_forces_the_begin_then_the_schema_then_the_end_then_frees_again() {
        let mut guide = tag(&["<function=get_weather>"], &["<function="]);
        feed(&mut guide, "Checking. <function=");
        assert!(guide.constrained());
        let mask = guide.mask();
        assert!(bit_allowed(&mask, b'g' as usize));
        assert!(!bit_allowed(&mask, b'x' as usize));
        assert!(!bit_allowed(&mask, STOP as usize));
        feed(&mut guide, "get_weather>");
        // The schema now: only an object opens.
        let mask = guide.mask();
        assert!(bit_allowed(&mask, b'{' as usize) && !bit_allowed(&mask, b'"' as usize));
        feed(&mut guide, "{\"city\": \"Paris\"}");
        // The end is forced, not the stop.
        let mask = guide.mask();
        assert!(bit_allowed(&mask, b'<' as usize) && !bit_allowed(&mask, STOP as usize));
        feed(&mut guide, "</function>");
        assert!(!guide.constrained());
        feed(&mut guide, " done. <function=get_weather>{}");
        assert!(
            guide.constrained(),
            "a second trigger opens a second structure"
        );
    }

    #[test]
    fn the_rest_after_a_trigger_is_any_structure_it_begins() {
        let mut guide = tag(
            &["<function=a>", "<function=b>", "<tool>"],
            &["<function=", "<tool>"],
        );
        feed(&mut guide, "<function=");
        let mask = guide.mask();
        assert!(bit_allowed(&mask, b'a' as usize) && bit_allowed(&mask, b'b' as usize));
        assert!(!bit_allowed(&mask, b'c' as usize));
        feed(&mut guide, "b>{}</function>");
        assert!(!guide.constrained());
    }

    #[test]
    fn a_trigger_that_is_a_whole_begin_opens_the_schema_at_once() {
        let mut guide = tag(&["<tool>"], &["<tool>"]);
        feed(&mut guide, "x<tool>");
        let mask = guide.mask();
        assert!(bit_allowed(&mask, b'{' as usize) && !bit_allowed(&mask, b'<' as usize));
    }

    #[test]
    fn a_token_carrying_the_trigger_and_part_of_the_begin_forces_only_the_rest() {
        let mut guide = tag(&["<function=get_weather>"], &["<function="]);
        feed_token(&mut guide, 301);
        let mask = guide.mask();
        assert!(bit_allowed(&mask, b'_' as usize) && !bit_allowed(&mask, b'g' as usize));
        feed(&mut guide, "_weather>{}");
        feed_token(&mut guide, 302);
        assert!(!guide.constrained());
    }

    #[test]
    fn a_token_running_past_the_begin_into_the_schema_leaves_that_call_free() {
        let mut guide = tag(&["<function=get_weather>"], &["<function="]);
        feed_token(&mut guide, 300);
        assert!(!guide.constrained());
        assert!(guide.mask().is_empty());
    }

    #[test]
    fn a_trigger_split_across_tokens_still_opens() {
        let mut guide = tag(&["<function=get_weather>"], &["<function="]);
        feed(&mut guide, &"word ".repeat(40));
        feed(&mut guide, "<func");
        assert!(!guide.constrained());
        feed(&mut guide, "tion=");
        assert!(guide.constrained());
    }

    #[test]
    fn a_tag_whose_triggers_and_structures_do_not_meet_is_refused() {
        let structure = |begin: &str| Structure {
            begin: begin.into(),
            schema: "{}".into(),
            end: String::new(),
        };
        let refused = |structures: Vec<Structure>, triggers: &[&str]| {
            StructuralTag {
                structures,
                triggers: triggers.iter().map(|t| t.to_string()).collect(),
            }
            .validate()
            .is_err()
        };
        assert!(refused(vec![structure("<a>")], &[]));
        assert!(refused(vec![structure("<a>")], &[""]));
        assert!(refused(vec![structure("<a>")], &["<b"]));
        assert!(refused(vec![structure("<a>")], &["<a", "<c"]));
        assert!(!refused(vec![structure("<a>"), structure("<ab>")], &["<a"]));
    }

    #[test]
    fn a_literal_is_quoted_for_ebnf() {
        assert_eq!(ebnf_literal("a\"b\\c\nd\u{1}"), r#""a\"b\\c\nd\x01""#);
        assert_eq!(ebnf_literal("<função>"), "\"<função>\"");
    }
}
