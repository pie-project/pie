//! CSM's turn protocol.
//!
//! Not ChatML, and not anything else in this crate. CSM's template
//! validates that a message's role is a **stringified integer** — a
//! SPEAKER ID — and raises if it is not:
//!
//! ```text
//! {%- if not message['role'] is string or not message['role'].isdigit() %}
//!     {{- raise_exception("The role must be an integer ...") }}
//! ```
//!
//! and then writes each turn as
//!
//! ```text
//! <|begin_of_text|>[{speaker}]{text}<|end_of_text|>
//! ```
//!
//! with `<|AUDIO|><|audio_eos|>` appended when the turn carries audio.
//! There is no `system` role, no assistant marker, and no generation
//! prompt in the HuggingFace sense — a turn is opened by naming a
//! speaker.
//!
//! # Why this file exists at all
//!
//! [`crate::catalog::Variant::chat`] has no default body, so a row must
//! answer it, and CSM is the row that proves why that is the right
//! design. Under the registry this replaced, `architectures[0]` was
//! `CsmForConditionalGeneration`, no arm matched it, and the `_ =>` arm
//! handed back ChatML — so a CSM would have been prompted with
//! `<|im_start|>user`, tokens its vocabulary does not contain, and
//! stopped on an `<|im_end|>` it can never emit. A conversation that
//! never ends is what a wrong template looks like from outside.
//!
//! The mapping from pie's two conversational roles to CSM's speaker ids
//! is this file's one editorial decision: **user is speaker 0 and
//! assistant is speaker 1**, which is the convention the reference
//! implementation's own examples use. It is stated here, once, rather
//! than assumed at three call sites.

use std::sync::Arc;
use tokenizer::Tokenizer;

use crate::decoders::{GenericChatDecoder, NoopReasoningDecoder, NoopToolDecoder};
use crate::instruct::{ChatDecoder, Instruct, ReasoningDecoder, ToolDecoder};

/// The speaker a user's turn is attributed to.
pub const SPEAKER_USER: u32 = 0;

/// The speaker the model's own turn is attributed to.
pub const SPEAKER_ASSISTANT: u32 = 1;

/// CSM's turn protocol, bound to one tokenizer.
pub struct CsmInstruct {
    tokenizer: Arc<Tokenizer>,
    bos: Vec<u32>,
    eos: Vec<u32>,
    stop_ids: Vec<u32>,
}

impl CsmInstruct {
    #[must_use]
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        // Llama-3's vocabulary, which is what CSM's text side is: the
        // markers are `<|begin_of_text|>` and `<|end_of_text|>`, not
        // `<bos>`/`<eos>`. Spelling them the gemma way would encode as
        // ordinary text and put four visible tokens into every turn.
        let bos = tokenizer.encode("<|begin_of_text|>");
        let eos = tokenizer.encode("<|end_of_text|>");
        // `<|audio_eos|>` ends a spoken turn and `<|end_of_text|>` ends
        // a written one. Both are terminal, and a decode loop that
        // watched only the second would run past the end of an
        // utterance into the next speaker's frame.
        let stop_ids = ["<|end_of_text|>", "<|audio_eos|>"]
            .iter()
            .filter_map(|s| tokenizer.token_to_id(s))
            .collect();
        Self { tokenizer, bos, eos, stop_ids }
    }

    /// One turn: `<bos>[speaker]text<eos>`.
    fn turn(&self, speaker: u32, message: &str) -> Vec<u32> {
        let mut v = self.bos.clone();
        v.extend(self.tokenizer.encode(&format!("[{speaker}]")));
        v.extend(self.tokenizer.encode(message.trim()));
        v.extend(&self.eos);
        v
    }
}

impl Instruct for CsmInstruct {
    /// Nothing. CSM has no system role, and the template REJECTS one.
    ///
    /// The alternative — folding the system prompt into speaker 0's
    /// text — is worse than dropping it: this model's output is speech,
    /// so an instruction placed in a turn is an instruction the model
    /// reads aloud. An empty prefix loses the instruction; a folded one
    /// broadcasts it.
    fn system(&self, _message: &str) -> Vec<u32> {
        Vec::new()
    }

    fn user(&self, message: &str) -> Vec<u32> {
        self.turn(SPEAKER_USER, message)
    }

    /// No first-turn special case: every turn carries its own
    /// `<|begin_of_text|>`, which is unusual and is what the template
    /// says — `{{- bos_token }}` sits INSIDE the per-message loop.
    fn first_user(&self, message: &str) -> Vec<u32> {
        self.user(message)
    }

    /// The system prompt is dropped, so this is the user turn alone.
    fn system_user(&self, _system: &str, user: &str) -> Vec<u32> {
        self.user(user)
    }

    fn assistant(&self, message: &str) -> Vec<u32> {
        self.turn(SPEAKER_ASSISTANT, message)
    }

    /// Open speaker 1's turn and stop.
    ///
    /// `<bos>[1]` and no role word, because CSM has none — the bracket
    /// IS the role marker.
    fn cue(&self) -> Vec<u32> {
        let mut v = self.bos.clone();
        v.extend(self.tokenizer.encode(&format!("[{SPEAKER_ASSISTANT}]")));
        v
    }

    fn seal(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    /// CSM has no tool protocol. It emits audio codes.
    fn equip(&self, _tools: &[String]) -> Vec<u32> {
        Vec::new()
    }

    fn answer(&self, _name: &str, _value: &str) -> Vec<u32> {
        Vec::new()
    }

    fn chat_decoder(&self) -> Box<dyn ChatDecoder> {
        Box::new(GenericChatDecoder::new(self.tokenizer.clone(), self.stop_ids.clone()))
    }

    /// No reasoning block: this model's continuation is speech.
    fn reasoning_decoder(&self) -> Box<dyn ReasoningDecoder> {
        Box::new(NoopReasoningDecoder)
    }

    fn tool_decoder(&self) -> Box<dyn ToolDecoder> {
        Box::new(NoopToolDecoder)
    }
}

#[cfg(test)]
mod tests {
    use super::{CsmInstruct, SPEAKER_ASSISTANT, SPEAKER_USER};
    use crate::instruct::Instruct;
    use std::sync::Arc;
    use tokenizer::Tokenizer;

    fn csm() -> CsmInstruct {
        let vocab: Vec<String> = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|audio_eos|>",
            "<|AUDIO|>",
            "[0]",
            "[1]",
            "Hello",
            "Hi",
            "Be brief",
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        CsmInstruct::new(Arc::new(Tokenizer::from_vocab(&vocab)))
    }

    fn text(inst: &CsmInstruct, ids: &[u32]) -> String {
        inst.tokenizer.decode(ids, false)
    }

    /// A user turn is `<bos>[0]text<eos>`, which is the template's own
    /// line read literally.
    #[test]
    fn a_user_turn_names_speaker_zero() {
        let inst = csm();
        assert_eq!(text(&inst, &inst.user("Hello")), "<|begin_of_text|>[0]Hello<|end_of_text|>");
    }

    /// And the model's turn names speaker 1.
    #[test]
    fn the_models_turn_names_speaker_one() {
        let inst = csm();
        assert_eq!(text(&inst, &inst.assistant("Hi")), "<|begin_of_text|>[1]Hi<|end_of_text|>");
        assert_ne!(SPEAKER_USER, SPEAKER_ASSISTANT, "one speaker is not a conversation");
    }

    /// EVERY turn carries `<|begin_of_text|>`, including the later ones.
    ///
    /// This is where CSM differs from every other template in the
    /// crate. Llama and gemma emit BOS once, at the head of the
    /// conversation, and their `first_user` exists to say so. CSM's
    /// template puts `{{- bos_token }}` inside the per-message loop, so
    /// dropping it after the first turn would silently change the
    /// framing of every later one.
    #[test]
    fn every_turn_repeats_the_beginning_of_text_marker() {
        let inst = csm();
        assert_eq!(inst.first_user("Hello"), inst.user("Hello"));
        assert!(text(&inst, &inst.user("Hello")).starts_with("<|begin_of_text|>"));
        assert!(text(&inst, &inst.assistant("Hi")).starts_with("<|begin_of_text|>"));
        assert!(text(&inst, &inst.cue()).starts_with("<|begin_of_text|>"));
    }

    /// The cue opens speaker 1's turn and stops there.
    ///
    /// No role word and no trailing newline, because the bracket is the
    /// whole marker — an extra token here is a token the model has to
    /// account for before it can start speaking.
    #[test]
    fn the_cue_opens_the_models_turn_and_nothing_more() {
        let inst = csm();
        assert_eq!(text(&inst, &inst.cue()), "<|begin_of_text|>[1]");
    }

    /// A system prompt has nowhere to go, and is dropped rather than
    /// spoken.
    ///
    /// The template raises on a non-numeric role, so there is no
    /// spelling for a system turn. Folding it into speaker 0's text
    /// would make the model READ THE INSTRUCTION ALOUD, which is a
    /// failure a listener notices and a log does not.
    #[test]
    fn a_system_prompt_is_dropped_rather_than_spoken() {
        let inst = csm();
        assert!(inst.system("Be brief").is_empty());
        assert_eq!(inst.system_user("Be brief", "Hello"), inst.user("Hello"));
        assert!(
            !text(&inst, &inst.system_user("Be brief", "Hello")).contains("Be brief"),
            "an instruction inside a turn is an instruction the model speaks"
        );
    }

    /// Both terminals are in the stop set.
    ///
    /// `<|audio_eos|>` ends a spoken turn and `<|end_of_text|>` ends a
    /// written one; watching only the latter runs a decode loop past
    /// the end of an utterance.
    #[test]
    fn both_ways_a_turn_can_end_are_stop_tokens() {
        let inst = csm();
        let tok = &inst.tokenizer;
        let seal = inst.seal();
        for marker in ["<|end_of_text|>", "<|audio_eos|>"] {
            let id = tok.token_to_id(marker).expect("in the fixture vocabulary");
            assert!(seal.contains(&id), "{marker} does not stop generation");
        }
        assert_eq!(seal.len(), 2);
    }

    /// A vocabulary without the markers yields an empty stop set rather
    /// than a panic.
    ///
    /// `token_to_id` returns `None` for a token a tokenizer has never
    /// heard of, and a CSM served with a mismatched tokenizer is a
    /// misconfiguration to report, not a crash inside a constructor.
    #[test]
    fn a_tokenizer_without_the_markers_still_constructs() {
        let vocab: Vec<String> = ["a", "b"].iter().map(ToString::to_string).collect();
        let inst = CsmInstruct::new(Arc::new(Tokenizer::from_vocab(&vocab)));
        assert!(inst.seal().is_empty());
    }

    /// No tools, and the two tool methods say so by returning nothing.
    #[test]
    fn a_speech_model_equips_no_tools() {
        let inst = csm();
        assert!(inst.equip(&["f".to_string()]).is_empty());
        assert!(inst.answer("f", "{}").is_empty());
    }

    /// The decoders exist and are the plain ones.
    ///
    /// A reasoning decoder that looked for `<think>` in a stream of
    /// audio codebook indices would occasionally find one.
    #[test]
    fn the_decoders_look_for_nothing_that_is_not_there() {
        use crate::instruct::{ReasoningEvent, ToolEvent};
        let inst = csm();
        let _ = inst.chat_decoder();
        let mut r = inst.reasoning_decoder();
        assert!(matches!(r.feed(&[0, 1, 2]), ReasoningEvent::Delta(d) if d.is_empty()));
        let mut t = inst.tool_decoder();
        assert!(matches!(t.feed(&[0, 1, 2]), ToolEvent::Start));
    }

    /// Leading and trailing whitespace is trimmed out of a turn.
    ///
    /// A stray newline before `<|end_of_text|>` is a token the model
    /// learns to expect and a caller does not know to send.
    #[test]
    fn a_turns_text_is_trimmed() {
        let inst = csm();
        assert_eq!(inst.user("  Hello  "), inst.user("Hello"));
    }
}
