//! What a CSM checkpoint IS.
//!
//! CSM is three stacks in one package, and that is the whole reason
//! this generation is written differently from every other one in the
//! catalog:
//!
//! * a **backbone**, an ordinary llama-like decoder that reads
//!   interleaved text and audio-code frames;
//! * a **depth decoder**, a second, narrower decoder that runs once per
//!   frame and emits the 32 residual codebooks in order;
//! * a **Mimi codec**, a convolutional encoder/decoder with a
//!   transformer in the middle and a residual vector quantizer, which
//!   turns waveforms into those codes and back.
//!
//! Nothing else in the catalog has more than one stack. `Deployment`
//! describes ONE — one layer count, one geometry, one KV store — so the
//! numbers below are stated in full here and [`super::project`] refuses
//! rather than picking the backbone's and calling it the model's. The
//! refusal is the point: it says what is missing by name instead of
//! answering with a stack that would load, page and fire, and produce
//! text where audio was asked for.
//!
//! Every number below is a measurement of `sesame/csm-1b` — its
//! `config.json` and its `model.safetensors` header, which is 537
//! tensors and agrees with the config field for field.

/// The backbone: the decoder that reads the interleaved frame stream.
///
/// It is llama-like — GQA, SwiGLU, RMSNorm, llama3 rope scaling — and
/// that resemblance is a trap this row exists to spring. A llama-like
/// derivation reading this config finds every key it wants and produces
/// a servable 16-layer stack; what it cannot find is the depth decoder
/// that turns the backbone's one hidden row into 32 codes, so the
/// "working" model emits a first codebook and nothing that plays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsmBackboneFacts {
    pub hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    /// `head_dim`, stated: 64 here, which is `hidden / q_heads` — but
    /// the depth decoder's is NOT its own quotient, so this generation
    /// has already proved that the quotient is not a rule.
    pub head_dim: u32,
    pub intermediate: u32,
    /// `text_vocab_size` — the size of the TEXT embedding table.
    pub text_vocab: u32,
    /// `vocab_size` — the size of ONE audio codebook, and the width of
    /// `lm_head`. 2051 = 2048 Mimi codes + 3 control ids.
    pub audio_vocab: u32,
    /// How many residual codebooks make one audio frame.
    pub codebooks: u32,
}

impl CsmBackboneFacts {
    /// The query projection's output width.
    #[must_use]
    pub const fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }
    /// One of the key/value projections' output width.
    #[must_use]
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

/// The depth decoder: 32 steps per frame, over its own hidden width.
///
/// `backbone_hidden` is the field that makes this struct load-bearing.
/// The depth decoder's embedding table is indexed by
/// `codebooks * audio_vocab` and its rows are `backbone_hidden` wide,
/// not `hidden` — because the SAME table is what the backbone reads its
/// audio codes through, and `inputs_embeds_projector` is the
/// `[hidden, backbone_hidden]` matrix that narrows a backbone row into
/// this stack. Two stacks sharing one table is why `tie_codebooks` is a
/// fact of the package rather than of either stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsmDepthFacts {
    pub hidden: u32,
    /// The width of the stack this one hangs off — 2048 against a
    /// 1024-wide depth decoder on `csm-1b`.
    pub backbone_hidden: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    /// 128 on `csm-1b`, against `hidden / q_heads == 128` — equal here
    /// by coincidence of 1024/8, and stated rather than derived because
    /// the coincidence is not a rule.
    pub head_dim: u32,
    pub intermediate: u32,
    /// One codebook's alphabet, the same 2051 the backbone's head emits.
    pub vocab: u32,
    /// How many codebooks this stack walks. `codebooks_head` has
    /// `codebooks - 1` slices, because the FIRST codebook is the
    /// backbone's `lm_head` output and this stack predicts the rest.
    pub codebooks: u32,
}

impl CsmDepthFacts {
    /// The rows of the shared code-embedding table:
    /// every codebook's alphabet, concatenated.
    ///
    /// 32 × 2051 = 65 632 on `csm-1b`, and that is the extent the
    /// checkpoint ships — so a row that got the product wrong would be
    /// caught by the manifest rather than by a wrong sound.
    #[must_use]
    pub const fn code_table_rows(&self) -> u32 {
        self.codebooks * self.vocab
    }
    /// How many codebooks the depth decoder's own head predicts.
    #[must_use]
    pub const fn head_slices(&self) -> u32 {
        self.codebooks - 1
    }
    /// The query projection's output width.
    #[must_use]
    pub const fn q_width(&self) -> u32 {
        self.q_heads * self.head_dim
    }
    /// One of the key/value projections' output width.
    #[must_use]
    pub const fn kv_width(&self) -> u32 {
        self.kv_heads * self.head_dim
    }
}

/// The Mimi codec: convolutions, a transformer, and a residual VQ.
///
/// It is in the package and therefore in the manifest. It is not in any
/// deployment, because a `Deployment` has no vocabulary for a
/// convolutional codec — no layer count that means anything, no paged
/// store, no rope. Naming its tensors is still worth doing: the codec
/// is 350 of the checkpoint's 537 tensors, and a row that ignored them
/// would happily match a bare backbone with no codec at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsmCodecFacts {
    /// The codec transformer's width, and the channel count the
    /// quantizer's projections work in.
    pub hidden: u32,
    /// The quantizer's LATENT width — 256 against a 512-wide codec, so
    /// `input_proj` narrows and `output_proj` widens.
    pub codebook_dim: u32,
    /// One codebook's alphabet: 2048, which with three control ids is
    /// the backbone's 2051.
    pub codebook_size: u32,
    /// Total quantizers, semantic plus acoustic.
    pub quantizers: u32,
    /// How many of them are SEMANTIC. The remainder are acoustic, and
    /// the two live in separately-named modules, so this split is what
    /// says how many `layers.{}` each of the two carries.
    pub semantic_quantizers: u32,
    /// The convolutional stack's base channel count.
    pub filters: u32,
}

impl CsmCodecFacts {
    /// The acoustic quantizer's layer count: everything that is not
    /// semantic.
    #[must_use]
    pub const fn acoustic_quantizers(&self) -> u32 {
        self.quantizers - self.semantic_quantizers
    }
}

/// The whole package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsmFacts {
    pub backbone: CsmBackboneFacts,
    pub depth: CsmDepthFacts,
    pub codec: CsmCodecFacts,
    /// `tie_codebooks_embeddings`: whether the backbone reads its audio
    /// codes through the depth decoder's table instead of shipping one.
    ///
    /// `true` on `csm-1b`, and therefore an ABSENCE in the manifest:
    /// `backbone_model.embed_tokens.embed_audio_tokens.weight` is not
    /// in the checkpoint. Stating the absence is what stops an untied
    /// build from matching this row.
    pub tied_codebooks: bool,
    /// `tie_word_embeddings` — `false`, so `lm_head` is shipped.
    pub tied_embeddings: bool,
}

impl CsmFacts {
    /// `sesame/csm-1b`, and the `unsloth/csm-1b` mirror of it.
    ///
    /// The only published CSM. Every number is from its `config.json`;
    /// every extent the manifest derives from them was checked against
    /// the safetensors header rather than reasoned about.
    #[must_use]
    pub const fn csm_1b() -> Self {
        Self {
            backbone: CsmBackboneFacts {
                hidden: 2048,
                layers: 16,
                q_heads: 32,
                kv_heads: 8,
                head_dim: 64,
                intermediate: 8192,
                text_vocab: 128_256,
                audio_vocab: 2051,
                codebooks: 32,
            },
            depth: CsmDepthFacts {
                hidden: 1024,
                backbone_hidden: 2048,
                layers: 4,
                q_heads: 8,
                kv_heads: 2,
                head_dim: 128,
                intermediate: 8192,
                vocab: 2051,
                codebooks: 32,
            },
            codec: CsmCodecFacts {
                hidden: 512,
                codebook_dim: 256,
                codebook_size: 2048,
                quantizers: 32,
                semantic_quantizers: 1,
                filters: 64,
            },
            tied_codebooks: true,
            tied_embeddings: false,
        }
    }

    /// The corpus's `synthetic--csm.json`, at the width it states.
    ///
    /// Not a row — nothing has ever shipped this stack — but it is the
    /// only CSM config checked into this repository, and a fixture that
    /// transcribes it is what keeps the reading of that file honest.
    /// The keys it omits are the ones HuggingFace derives: it states no
    /// `num_key_value_heads` and no `head_dim` for the backbone, so
    /// those are `q_heads` and `hidden / q_heads` here, which is what a
    /// loader would compute.
    #[must_use]
    pub const fn csm_synthetic() -> Self {
        Self {
            backbone: CsmBackboneFacts {
                hidden: 128,
                layers: 4,
                q_heads: 8,
                kv_heads: 8,
                head_dim: 16,
                intermediate: 128,
                text_vocab: 1000,
                audio_vocab: 1000,
                codebooks: 8,
            },
            depth: CsmDepthFacts {
                hidden: 64,
                backbone_hidden: 128,
                layers: 2,
                q_heads: 4,
                kv_heads: 2,
                head_dim: 16,
                intermediate: 128,
                vocab: 2048,
                codebooks: 8,
            },
            codec: CsmCodecFacts {
                hidden: 64,
                codebook_dim: 32,
                codebook_size: 1024,
                quantizers: 8,
                semantic_quantizers: 1,
                filters: 16,
            },
            tied_codebooks: true,
            tied_embeddings: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CsmFacts;

    fn both() -> [CsmFacts; 2] {
        [CsmFacts::csm_1b(), CsmFacts::csm_synthetic()]
    }

    /// Every fixture states a stack that could exist.
    ///
    /// A zero anywhere here is a stack with a projection of no width or
    /// a table of no rows, and every extent the manifest states is a
    /// product of these — so a zero would travel into the manifest as a
    /// `[0, n]` that matches nothing and reports itself as a mismatched
    /// EXTENT rather than as the missing measurement it is.
    #[test]
    fn every_fixture_states_a_stack_that_could_exist() {
        for f in both() {
            let b = f.backbone;
            assert!(b.hidden > 0 && b.layers > 0 && b.intermediate > 0);
            assert!(b.q_heads > 0 && b.kv_heads > 0 && b.head_dim > 0);
            assert!(b.text_vocab > 0 && b.audio_vocab > 0 && b.codebooks > 0);
            let d = f.depth;
            assert!(d.hidden > 0 && d.layers > 0 && d.intermediate > 0);
            assert!(d.q_heads > 0 && d.kv_heads > 0 && d.head_dim > 0);
            assert!(d.vocab > 0 && d.codebooks > 0 && d.backbone_hidden > 0);
            let c = f.codec;
            assert!(c.hidden > 0 && c.codebook_dim > 0 && c.codebook_size > 0);
            assert!(c.quantizers > 0 && c.filters > 0);
        }
    }

    /// Both stacks' heads group evenly.
    ///
    /// GQA needs `q_heads % kv_heads == 0` or a query head has no key
    /// head to read — and the two stacks answer it separately, 32/8 in
    /// the backbone and 8/2 in the depth decoder, which is the reason
    /// they are two structs rather than one with a width field.
    #[test]
    fn both_stacks_group_their_query_heads_evenly() {
        for f in both() {
            assert_eq!(
                f.backbone.q_heads % f.backbone.kv_heads,
                0,
                "a backbone query head with no key head to read"
            );
            assert_eq!(
                f.depth.q_heads % f.depth.kv_heads,
                0,
                "a depth-decoder query head with no key head to read"
            );
        }
    }

    /// The two stacks agree about the alphabet they pass between them.
    ///
    /// The backbone's `lm_head` emits codebook 0 and the depth decoder
    /// emits the other 31, so both must be over the SAME alphabet — and
    /// both must walk the same number of codebooks or a frame ends
    /// short. `csm-1b` states 2051 and 32 in three separate places in
    /// its config and this is where they are held together.
    #[test]
    fn the_two_stacks_agree_about_one_frame() {
        let f = CsmFacts::csm_1b();
        assert_eq!(f.backbone.audio_vocab, f.depth.vocab);
        assert_eq!(f.backbone.codebooks, f.depth.codebooks);
        assert_eq!(f.depth.backbone_hidden, f.backbone.hidden);
    }

    /// The shared code table is one alphabet per codebook, concatenated.
    ///
    /// 32 × 2051 = 65 632, which is the extent
    /// `depth_decoder.model.embed_tokens.weight` actually ships. This
    /// is the arithmetic a tie makes load-bearing: get the product
    /// wrong and the backbone reads audio codes out of the wrong band
    /// of one table, which produces sound rather than an error.
    #[test]
    fn the_shared_code_table_is_one_alphabet_per_codebook() {
        assert_eq!(CsmFacts::csm_1b().depth.code_table_rows(), 65_632);
        assert_eq!(CsmFacts::csm_synthetic().depth.code_table_rows(), 8 * 2048);
    }

    /// The depth head predicts every codebook but the first.
    ///
    /// `codebooks_head` is `[31, 1024, 2051]` in the checkpoint: 31
    /// slices, not 32, because codebook 0 is the BACKBONE's output. An
    /// off-by-one here is 32 slices against 31 shipped, which the
    /// manifest catches — and the reason to catch it is that the same
    /// off-by-one in a decode loop drops the last codebook of every
    /// frame and quietly dulls the audio.
    #[test]
    fn the_depth_head_predicts_every_codebook_but_the_backbones() {
        assert_eq!(CsmFacts::csm_1b().depth.head_slices(), 31);
        assert_eq!(CsmFacts::csm_synthetic().depth.head_slices(), 7);
    }

    /// Projection widths are heads times head dim, and for the backbone
    /// that is NOT `hidden`.
    ///
    /// 32 × 64 = 2048 happens to be `hidden` here, and 8 × 64 = 512
    /// does not — which is the ordinary GQA asymmetry. The depth
    /// decoder's 8 × 128 = 1024 is its `hidden` and its 2 × 128 = 256
    /// is not.
    #[test]
    fn projection_widths_are_heads_times_head_dim() {
        let f = CsmFacts::csm_1b();
        assert_eq!(f.backbone.q_width(), 2048);
        assert_eq!(f.backbone.kv_width(), 512);
        assert_eq!(f.depth.q_width(), 1024);
        assert_eq!(f.depth.kv_width(), 256);
        let s = CsmFacts::csm_synthetic();
        assert_eq!(s.backbone.q_width(), 128);
        assert_eq!(s.backbone.kv_width(), 128, "the synthetic states no GQA at all");
        assert_eq!(s.depth.q_width(), 64);
        assert_eq!(s.depth.kv_width(), 32);
    }

    /// The quantizer splits into a semantic part and an acoustic one,
    /// and the split is exact.
    ///
    /// The checkpoint ships them as two separately-named modules with
    /// their own `layers.{}` — 1 semantic and 31 acoustic on `csm-1b` —
    /// so the manifest has to know both counts. Deriving one from the
    /// other is what this states.
    #[test]
    fn the_quantizer_splits_exactly_into_semantic_and_acoustic() {
        assert_eq!(CsmFacts::csm_1b().codec.acoustic_quantizers(), 31);
        assert_eq!(CsmFacts::csm_synthetic().codec.acoustic_quantizers(), 7);
        for f in both() {
            assert!(
                f.codec.semantic_quantizers < f.codec.quantizers,
                "a codec with no acoustic quantizer is a semantic tokenizer, not a codec"
            );
        }
    }

    /// The depth decoder's head dim is not its quotient, and that is
    /// why it is a field.
    ///
    /// `csm-1b`'s backbone is 2048/32 = 64, which the quotient rule
    /// gets right. The synthetic's depth decoder is 64/4 = 16, which it
    /// also gets right. But the published depth decoder is 1024 wide
    /// over 8 heads of 128 — and `8 × 128 = 1024` only because the
    /// config states `head_dim` explicitly; a derivation that assumed
    /// `hidden / q_heads` gets 128 here by luck and would not on a
    /// stack whose q projection is not square.
    #[test]
    fn head_dim_is_stated_because_it_is_not_always_the_quotient() {
        let f = CsmFacts::csm_1b();
        assert_eq!(f.backbone.head_dim * f.backbone.q_heads, f.backbone.hidden);
        assert_eq!(f.depth.head_dim * f.depth.q_heads, f.depth.hidden);
        // The synthetic's depth decoder is where the two part company:
        // 4 heads of 16 is 64, and its BACKBONE hidden is 128.
        let s = CsmFacts::csm_synthetic();
        assert_ne!(
            s.depth.head_dim * s.depth.q_heads,
            s.depth.backbone_hidden,
            "the depth decoder is narrower than the stack it hangs off"
        );
    }

    /// The tie is a fact of the package, and both fixtures state it.
    ///
    /// `tie_codebooks_embeddings: true` with `tie_word_embeddings:
    /// false` is an unusual pair — the audio table is shared and the
    /// text head is not — and stating them separately is what lets the
    /// manifest expect `lm_head` AND expect the audio embedding to be
    /// missing.
    #[test]
    fn the_audio_table_is_tied_and_the_text_head_is_not() {
        for f in both() {
            assert!(f.tied_codebooks);
            assert!(!f.tied_embeddings);
        }
    }

    /// The fixtures are distinguishable.
    ///
    /// A fixture that equals another is a transcription that was copied
    /// rather than measured.
    #[test]
    fn the_two_fixtures_are_different_stacks() {
        assert_ne!(CsmFacts::csm_1b(), CsmFacts::csm_synthetic());
    }
}
