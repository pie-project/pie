//! **THE WHOLE PIPE, PINNED — real bytes through the real codec into the
//! family's own arithmetic.**
//!
//! ```text
//! cargo test -p runtime --test media_pipe_is_the_pinned_preprocessing
//! ```
//!
//! The arithmetic's goldens live beside the arithmetic
//! (`model`'s `qwen3_5_media_is_the_pinned_arithmetic` /
//! `gemma4_media_is_the_pinned_arithmetic`); THESE claims need the codec —
//! decode and the Catmull-Rom resample, which are the host's
//! (`runtime::inferlet::host::media::decode`, per `model::media`'s dependency
//! rule) — so they run here, composed exactly the way `image.from-bytes`
//! composes them in service. The digest claims ride here too, because the
//! digest does ([`span_digest`]'s own doc says why).

use model::media::{Budget, EncodedSpan, Fault, Grid, VisionFrontEnd};
use runtime::inferlet::media_codec as decode;
use runtime::inferlet::span_digest;

/// The pipe as `image.from-bytes` runs it: the host decodes, the front-end
/// does its family's arithmetic with the resample lent.
fn encode_png(
    fe: &dyn VisionFrontEnd,
    bytes: &[u8],
    budget: Budget,
) -> model::media::Result<EncodedSpan> {
    fe.encode(&decode::decode(bytes)?, budget, decode::resize_exact)
}

mod png {
    //! **A REAL PNG, WRITTEN BY HAND** — and the reason it is written by hand.
    //!
    //! Both front-end gates end with a whole-pipe claim: real encoded bytes in,
    //! the right shapes out. Encoding those bytes with the same crate that decodes
    //! them would make the claim circular — an encoder and a decoder from one
    //! library agree with each other by construction, and a gate that only proves
    //! that has proved nothing about the file format. So this module emits PNG
    //! from the specification: IHDR, one IDAT of STORED (uncompressed) deflate
    //! blocks under a zlib wrapper, IEND, with CRC-32 per chunk and Adler-32 over
    //! the raw stream.
    //!
    //! Stored blocks rather than a compressor for the same reason the whole crate
    //! prefers transcription to cleverness: there is exactly one byte sequence this
    //! can emit for a given image, so two runs and two machines produce identical
    //! bytes and the digest gate downstream means what it says.

    #![allow(dead_code)]

    /// CRC-32 (IEEE), the polynomial PNG's chunk checksum names.
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                let mask = 0u32.wrapping_sub(crc & 1);
                crc = (crc >> 1) ^ (0xedb8_8320 & mask);
            }
        }
        !crc
    }

    /// Adler-32, zlib's own checksum over the UNCOMPRESSED stream.
    fn adler32(bytes: &[u8]) -> u32 {
        let (mut a, mut b) = (1u32, 0u32);
        for &x in bytes {
            a = (a + u32::from(x)) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }

    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
        #[allow(clippy::cast_possible_truncation)]
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        let mut crc_over = Vec::with_capacity(4 + body.len());
        crc_over.extend_from_slice(kind);
        crc_over.extend_from_slice(body);
        out.extend_from_slice(&crc32(&crc_over).to_be_bytes());
    }

    /// **A DETERMINISTIC `w × h` 8-BIT RGB PNG.**
    ///
    /// `pixel(x, y)` names the colour; the caller picks a rule it can also assert
    /// against, so a test can follow one source pixel all the way to a patch lane.
    pub fn png_rgb(w: u32, h: u32, pixel: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        // Raw scanlines: PNG prefixes each with a filter byte, and 0 is "None".
        let mut raw = Vec::with_capacity((h * (1 + w * 3)) as usize);
        for y in 0..h {
            raw.push(0u8);
            for x in 0..w {
                raw.extend_from_slice(&pixel(x, y));
            }
        }

        // zlib: CMF/FLG, then stored deflate blocks of at most 65535 bytes, then
        // Adler-32 of the raw stream.
        let mut z = vec![0x78u8, 0x01];
        let mut at = 0usize;
        while at < raw.len() {
            let take = (raw.len() - at).min(0xffff);
            let last = u8::from(at + take == raw.len());
            z.push(last);
            #[allow(clippy::cast_possible_truncation)]
            let len = take as u16;
            z.extend_from_slice(&len.to_le_bytes());
            z.extend_from_slice(&(!len).to_le_bytes());
            z.extend_from_slice(&raw[at..at + take]);
            at += take;
        }
        z.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit, colour type 2 (RGB)
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &z);
        chunk(&mut out, b"IEND", &[]);
        out
    }

    /// A ramp that makes every pixel of a small image distinct, so a golden can
    /// name which source pixel it expects in which patch lane.
    #[must_use]
    pub fn ramp(x: u32, y: u32) -> [u8; 3] {
        [
            ((x * 7 + y * 13) % 251) as u8,
            ((x * 31 + y * 3) % 251) as u8,
            ((x + y * 97) % 251) as u8,
        ]
    }
}

mod qwen {
    use super::*;
    use model::qwen_3::media::Qwen35Vision;

    /// **THE WHOLE PIPE, ON A REAL PNG.** Bytes this test wrote from the PNG
    /// specification (not from `image`'s encoder — see `common`), decoded,
    /// resized, patchified, and every stream's length checked against the geometry
    /// the same span reports.
    #[test]
    fn a_real_png_goes_through_the_whole_pipe() {
        let fe = Qwen35Vision::new();
        let c = fe.config;
        let bytes = png::png_rgb(200, 120, png::ramp);
        let span = encode_png(&fe, &bytes, Budget::Still)
            .expect("a well-formed PNG encodes");

        let (gh, gw) = c.patch_grid(120, 200).expect("servable");
        assert_eq!(
            c.smart_resize(120, 200).expect("servable"),
            (224, 352),
            "the resize policy"
        );
        assert_eq!((gh, gw), (14, 22));

        assert_eq!(span.rows, gh * gw, "one payload row per pre-merge patch");
        assert_eq!(span.patch_grid, Grid::still(gh, gw));
        assert_eq!(
            span.grid,
            Grid::still(gh / 2, gw / 2),
            "the merged grid is what the token rectangle sees"
        );
        assert_eq!(span.token_count, gh * gw / 4);
        assert_eq!(span.position_span, (gw / 2).max(gh / 2));
        assert!(span.uses_mrope, "qwen's trunk rotates on the triple");

        assert_eq!(
            span.payload.len(),
            span.rows as usize * c.patch_width(),
            "the payload is `rows · C·T·P²`"
        );
        assert_eq!(span.positions.len(), span.rows as usize * 2);
        assert_eq!(span.embed_rows.len(), span.rows as usize * 4);
        assert_eq!(span.embed_weights.len(), span.rows as usize * 4);
        assert!(
            span.payload.iter().all(|v| (-1.0..=1.0).contains(v)),
            "normalized pixels live in [-1, 1]"
        );
        // Not a flat image: the ramp survived decode and resize.
        let first = span.payload[0];
        assert!(
            span.payload.iter().any(|v| (v - first).abs() > 1e-3),
            "the decoded image is uniform, so nothing downstream was exercised"
        );
    }

    /// The delimiters are NAMED, not numbered, and `tokens()` is
    /// prefix + pad × token_count + suffix in whatever ids the bound tokenizer
    /// hands back (media-door §0, §2).
    #[test]
    fn the_span_spells_itself_out_of_the_tokenizers_own_ids() {
        let fe = Qwen35Vision::new();
        let d = fe.delimiters();
        assert_eq!(d.prefix, "<|vision_start|>");
        assert_eq!(d.placeholder, "<|image_pad|>");
        assert_eq!(d.suffix, "<|vision_end|>");

        let bytes = png::png_rgb(64, 64, png::ramp);
        let mut span = encode_png(&fe, &bytes, Budget::Still).expect("encodes");
        // What the runtime does with `tokenizer.token_to_id` — two arbitrary
        // checkpoint numberings, one of which is not the other.
        span.spell_with(vec![151_652], 151_655, vec![151_653]);
        let toks = span.tokens();
        assert_eq!(toks.len(), 1 + span.token_count as usize + 1);
        assert_eq!(toks[0], 151_652);
        assert_eq!(*toks.last().expect("non-empty"), 151_653);
        assert!(toks[1..toks.len() - 1].iter().all(|&t| t == 151_655));

        let mut renumbered = span.clone();
        renumbered.spell_with(vec![7], 8, vec![9]);
        assert_ne!(span.tokens(), renumbered.tokens(), "the ids moved");
        assert_eq!(
            span_digest(&span),
            span_digest(&renumbered),
            "and the span did not — a digest is over the preprocessed span, never over its spelling"
        );
    }

    /// **THE CACHE STATUTE'S KEY** (media-door §5). Two different images produce
    /// identical token lists; the digest is what tells them apart, and it is
    /// stable across runs because everything it hashes is deterministic
    /// arithmetic over deterministic bytes in a fixed byte order.
    #[test]
    fn the_digest_is_stable_and_separates_two_images_one_run_cannot() {
        let fe = Qwen35Vision::new();
        let one = encode_png(&fe, &png::png_rgb(96, 96, png::ramp), Budget::Still)
            .expect("one");
        let again = encode_png(&fe, &png::png_rgb(96, 96, png::ramp), Budget::Still)
            .expect("again");
        let other = encode_png(
            &fe,
                &png::png_rgb(96, 96, |x, y| {
                    let mut p = png::ramp(x, y);
                    // One pixel of one channel, moved by one.
                    if x == 5 && y == 7 {
                        p[1] = p[1].wrapping_add(1);
                    }
                    p
                }),
                Budget::Still,
            )
            .expect("other");

        assert_eq!(span_digest(&one).len(), 32, "blake3, the workspace's own hash");
        assert_eq!(
            span_digest(&one),
            span_digest(&again),
            "two encodings of one image must collide, or a correct cache hit looks like a bug"
        );
        assert_eq!(one.token_count, other.token_count);
        let mut a = one.clone();
        let mut b = other.clone();
        a.spell_with(vec![1], 2, vec![3]);
        b.spell_with(vec![1], 2, vec![3]);
        assert_eq!(
            a.tokens(),
            b.tokens(),
            "the ledger cannot tell two images apart"
        );
        assert_ne!(span_digest(&a), span_digest(&b), "and the statute's key must");
    }

    /// The refusals, by name.
    #[test]
    fn the_refusals_fire_by_name() {
        let fe = Qwen35Vision::new();
        let empty = encode_png(&fe, &[], Budget::Still)
            .expect_err("zero bytes are refused");
        assert_eq!(empty.name(), "Decode", "{empty}");

        let garbage = encode_png(&fe, b"this is not a picture, it is a sentence", Budget::Still)
            .expect_err("prose is refused");
        assert_eq!(garbage.name(), "Decode", "{garbage}");
        assert!(matches!(garbage, Fault::Decode(_)));
    }

    /// `budget` is ignored here, and that is a fact about qwen rather than an
    /// omission: its ceiling is `max_pixels`, and `Processor::for_arch_video`
    /// already answered one config for both.
    #[test]
    fn a_video_frame_is_the_same_preprocessing_as_a_still() {
        let fe = Qwen35Vision::new();
        let bytes = png::png_rgb(80, 60, png::ramp);
        let still: EncodedSpan = encode_png(&fe, &bytes, Budget::Still).expect("still");
        let frame: EncodedSpan = encode_png(&fe, &bytes, Budget::VideoFrame)
            .expect("a frame");
        assert_eq!(still, frame);
    }
}

mod gemma {
    use super::*;
    use model::gemma_4::media::Gemma4Vision;

    /// **THE WHOLE PIPE, ON A REAL PNG** written from the PNG specification rather
    /// than by the decoder's own library.
    #[test]
    fn a_real_png_goes_through_the_whole_pipe() {
        let fe = Gemma4Vision::new();
        let c = fe.config;
        let bytes = png::png_rgb(200, 120, png::ramp);
        let span = encode_png(&fe, &bytes, Budget::Still)
            .expect("a well-formed PNG encodes");

        let (th, tw) = c
            .aspect_ratio_preserving_size(120, 200, Budget::Still)
            .expect("resizes");
        let (gh, gw) = (th / c.patch_size, tw / c.patch_size);
        assert_eq!((th, tw), (576, 1008));
        assert_eq!((gh, gw), (36, 63));

        assert_eq!(span.rows, gh * gw, "one payload row per patch, and no padding");
        assert_eq!(span.patch_grid, Grid::still(gh, gw));
        assert_eq!(span.token_count, gh * gw / 9);
        assert_eq!(
            span.grid,
            Grid::still(1, span.token_count),
            "gemma's merged extent is a run, not a rectangle"
        );
        assert_eq!(
            span.position_span, span.token_count,
            "1-D rope advances by the rows the span occupies"
        );
        assert!(!span.uses_mrope, "gemma's trunk rotates scalar");

        assert_eq!(span.payload.len(), span.rows as usize * c.patch_width());
        assert_eq!(span.positions.len(), span.rows as usize * 2);
        assert_eq!(span.embed_rows.len(), span.rows as usize * 2);
        assert_eq!(span.embed_weights.len(), span.rows as usize * 2);
        let first = span.payload[0];
        assert!(
            span.payload.iter().any(|v| (v - first).abs() > 1e-3),
            "the decoded image is uniform, so nothing downstream was exercised"
        );
    }

    /// A frame of a clip gets the smaller ceiling, which is the whole reason
    /// [`Budget`] is an argument rather than a second trait method.
    #[test]
    fn a_video_frame_gets_the_frame_budget() {
        let fe = Gemma4Vision::new();
        let bytes = png::png_rgb(200, 120, png::ramp);
        let still = encode_png(&fe, &bytes, Budget::Still).expect("still");
        let frame = encode_png(&fe, &bytes, Budget::VideoFrame)
            .expect("a frame");
        assert!(
            frame.token_count < still.token_count,
            "a frame occupied {} rows and a still {}",
            frame.token_count,
            still.token_count
        );
        assert!(frame.token_count <= fe.config.video_soft_tokens);
        assert!(still.token_count <= fe.config.max_soft_tokens);
        assert_ne!(
            span_digest(&still),
            span_digest(&frame),
            "two preprocessings of one image are two spans"
        );
    }

    /// **GEMMA-4'S OWN DELIMITERS**, which are not gemma-3's. This vocabulary
    /// spells markers `<|x>` … `<x|>` with `<|x|>` standalone — the same family as
    /// the `<|turn>` / `<turn|>` pair `chat_template::gemma` already reads by name,
    /// and the `<|audio>` / `<audio|>` pair the campaign pinned. Nothing here is a
    /// number: the runtime resolves all three through `tokenizer.token_to_id`.
    #[test]
    fn the_span_spells_itself_out_of_the_tokenizers_own_ids() {
        let fe = Gemma4Vision::new();
        let d = fe.delimiters();
        assert_eq!(d.prefix, "<|image>");
        assert_eq!(d.placeholder, "<|image|>");
        assert_eq!(d.suffix, "<image|>");
        assert!(
            !d.placeholder.is_empty(),
            "the run scan finds a span by its pad, so an architecture must name one"
        );

        let bytes = png::png_rgb(96, 96, png::ramp);
        let mut span = encode_png(&fe, &bytes, Budget::Still).expect("encodes");
        span.spell_with(vec![262_144], 262_145, vec![262_146]);
        let toks = span.tokens();
        assert_eq!(toks.len(), 1 + span.token_count as usize + 1);
        assert_eq!(toks[0], 262_144);
        assert_eq!(*toks.last().expect("non-empty"), 262_146);
        assert!(toks[1..toks.len() - 1].iter().all(|&t| t == 262_145));
    }

    /// The digest is stable across readings and content-addressed, and two images
    /// the ledger cannot tell apart it can.
    #[test]
    fn the_digest_is_stable_and_separates_two_images_one_run_cannot() {
        let fe = Gemma4Vision::new();
        let one = encode_png(&fe, &png::png_rgb(96, 96, png::ramp), Budget::Still)
            .expect("one");
        let again = encode_png(&fe, &png::png_rgb(96, 96, png::ramp), Budget::Still)
            .expect("again");
        let other = encode_png(
            &fe,
                &png::png_rgb(96, 96, |x, y| {
                    let mut p = png::ramp(x, y);
                    if x == 11 && y == 3 {
                        p[2] = p[2].wrapping_add(1);
                    }
                    p
                }),
                Budget::Still,
            )
            .expect("other");

        assert_eq!(span_digest(&one), span_digest(&again));
        assert_eq!(span_digest(&one), span_digest(&one), "and stable across two readings");
        assert_eq!(one.token_count, other.token_count);
        assert_ne!(span_digest(&one), span_digest(&other));
    }

    /// The refusals, by name.
    #[test]
    fn the_refusals_fire_by_name() {
        let fe = Gemma4Vision::new();
        assert_eq!(
            encode_png(&fe, &[], Budget::Still)
                .expect_err("zero bytes")
                .name(),
            "Decode"
        );
        assert_eq!(
            encode_png(&fe, b"<html>not a picture</html>", Budget::Still)
                .expect_err("markup")
                .name(),
            "Decode"
        );
        assert_eq!(fe.arch(), "gemma4");
    }
}
