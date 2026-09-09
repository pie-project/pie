use models::media::{Budget, EncodedSpan, Fault, Grid, VisionFrontEnd};
use runtime::inferlet::media_codec as decode;
use runtime::inferlet::span_digest;

fn encode_png(
    fe: &dyn VisionFrontEnd,
    bytes: &[u8],
    budget: Budget,
) -> models::media::Result<EncodedSpan> {
    fe.encode(&decode::decode(bytes)?, budget, decode::resize_exact)
}

mod png {

    #![allow(dead_code)]

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

    pub fn png_rgb(w: u32, h: u32, pixel: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        let mut raw = Vec::with_capacity((h * (1 + w * 3)) as usize);
        for y in 0..h {
            raw.push(0u8);
            for x in 0..w {
                raw.extend_from_slice(&pixel(x, y));
            }
        }

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
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &z);
        chunk(&mut out, b"IEND", &[]);
        out
    }

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
    use models::qwen_3::media::Qwen35Vision;

    #[test]
    fn media_pipe_is_the_pinned_preprocessing_every_case() {
        a_real_png_goes_through_the_whole_pipe();
        the_span_spells_itself_out_of_the_tokenizers_own_ids();
        the_digest_is_stable_and_separates_two_images_one_run_cannot();
        the_refusals_fire_by_name();
        a_video_frame_is_the_same_preprocessing_as_a_still();
    }

    fn a_real_png_goes_through_the_whole_pipe() {
        let fe = Qwen35Vision::new();
        let c = fe.config;
        let bytes = png::png_rgb(200, 120, png::ramp);
        let span = encode_png(&fe, &bytes, Budget::Still).expect("a well-formed PNG encodes");

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
        let first = span.payload[0];
        assert!(
            span.payload.iter().any(|v| (v - first).abs() > 1e-3),
            "the decoded image is uniform, so nothing downstream was exercised"
        );
    }

    fn the_span_spells_itself_out_of_the_tokenizers_own_ids() {
        let fe = Qwen35Vision::new();
        let d = fe.delimiters();
        assert_eq!(d.prefix, "<|vision_start|>");
        assert_eq!(d.placeholder, "<|image_pad|>");
        assert_eq!(d.suffix, "<|vision_end|>");

        let bytes = png::png_rgb(64, 64, png::ramp);
        let mut span = encode_png(&fe, &bytes, Budget::Still).expect("encodes");
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

    fn the_digest_is_stable_and_separates_two_images_one_run_cannot() {
        let fe = Qwen35Vision::new();
        let one = encode_png(&fe, &png::png_rgb(96, 96, png::ramp), Budget::Still).expect("one");
        let again =
            encode_png(&fe, &png::png_rgb(96, 96, png::ramp), Budget::Still).expect("again");
        let other = encode_png(
            &fe,
            &png::png_rgb(96, 96, |x, y| {
                let mut p = png::ramp(x, y);
                if x == 5 && y == 7 {
                    p[1] = p[1].wrapping_add(1);
                }
                p
            }),
            Budget::Still,
        )
        .expect("other");

        assert_eq!(
            span_digest(&one).len(),
            32,
            "blake3, the workspace's own hash"
        );
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
        assert_ne!(
            span_digest(&a),
            span_digest(&b),
            "and the statute's key must"
        );
    }

    fn the_refusals_fire_by_name() {
        let fe = Qwen35Vision::new();
        let empty = encode_png(&fe, &[], Budget::Still).expect_err("zero bytes are refused");
        assert_eq!(empty.name(), "Decode", "{empty}");

        let garbage = encode_png(
            &fe,
            b"this is not a picture, it is a sentence",
            Budget::Still,
        )
        .expect_err("prose is refused");
        assert_eq!(garbage.name(), "Decode", "{garbage}");
        assert!(matches!(garbage, Fault::Decode(_)));
    }

    fn a_video_frame_is_the_same_preprocessing_as_a_still() {
        let fe = Qwen35Vision::new();
        let bytes = png::png_rgb(80, 60, png::ramp);
        let still: EncodedSpan = encode_png(&fe, &bytes, Budget::Still).expect("still");
        let frame: EncodedSpan = encode_png(&fe, &bytes, Budget::VideoFrame).expect("a frame");
        assert_eq!(still, frame);
    }
}

mod gemma {
    use super::*;
    use models::gemma_4::media::Gemma4Vision;

    #[test]
    fn media_pipe_is_the_pinned_preprocessing_1_every_case() {
        a_real_png_goes_through_the_whole_pipe();
        a_video_frame_gets_the_frame_budget();
        the_span_spells_itself_out_of_the_tokenizers_own_ids();
        the_digest_is_stable_and_separates_two_images_one_run_cannot();
        the_refusals_fire_by_name();
    }

    fn a_real_png_goes_through_the_whole_pipe() {
        let fe = Gemma4Vision::new();
        let c = fe.config;
        let bytes = png::png_rgb(200, 120, png::ramp);
        let span = encode_png(&fe, &bytes, Budget::Still).expect("a well-formed PNG encodes");

        let (th, tw) = c
            .aspect_ratio_preserving_size(120, 200, Budget::Still)
            .expect("resizes");
        let (gh, gw) = (th / c.patch_size, tw / c.patch_size);
        assert_eq!((th, tw), (576, 1008));
        assert_eq!((gh, gw), (36, 63));

        assert_eq!(
            span.rows,
            gh * gw,
            "one payload row per patch, and no padding"
        );
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

    fn a_video_frame_gets_the_frame_budget() {
        let fe = Gemma4Vision::new();
        let bytes = png::png_rgb(200, 120, png::ramp);
        let still = encode_png(&fe, &bytes, Budget::Still).expect("still");
        let frame = encode_png(&fe, &bytes, Budget::VideoFrame).expect("a frame");
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

    fn the_digest_is_stable_and_separates_two_images_one_run_cannot() {
        let fe = Gemma4Vision::new();
        let one = encode_png(&fe, &png::png_rgb(96, 96, png::ramp), Budget::Still).expect("one");
        let again =
            encode_png(&fe, &png::png_rgb(96, 96, png::ramp), Budget::Still).expect("again");
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
        assert_eq!(
            span_digest(&one),
            span_digest(&one),
            "and stable across two readings"
        );
        assert_eq!(one.token_count, other.token_count);
        assert_ne!(span_digest(&one), span_digest(&other));
    }

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
