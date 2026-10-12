//! A still goes through the package's frame and the host's cut exactly as
//! the models' own processors would have it: Qwen's smart resize and 2 x 2
//! merge blocks, Gemma's soft-token budget and 3 x 3 pooling blocks, each
//! read off the repository's packages, not off anything in Rust.

use std::sync::Arc;

use media::front::{Budget, EncodedSpan, Fault, Grid, Image};
use poem::star::{Deploy, ImageSpec, Package};
use runtime::inferlet::media_decode as decode;
use runtime::inferlet::span_digest;

/// A deployment's still front end, as the runtime assembles it.
struct Front {
    package: Arc<Package>,
    id: String,
    deploy: Deploy,
    spec: ImageSpec,
}

impl Front {
    fn of(deployment: &str) -> Front {
        let d = poem_compiler::catalog::deployment(deployment)
            .unwrap_or_else(|| panic!("the catalog lists {deployment}"));
        let spec = d
            .package
            .image(&d.model.id, &d.deploy)
            .expect("the package states its media")
            .expect("a vision deployment reads stills");
        Front {
            package: Arc::clone(&d.package),
            id: d.model.id.clone(),
            deploy: d.deploy.clone(),
            spec,
        }
    }

    fn frame(&self, h: u32, w: u32, budget: Budget) -> (u32, u32) {
        self.package
            .frame(&self.id, &self.deploy, h, w, budget.word())
            .expect("the package frames the still")
    }

    fn encode(&self, bytes: &[u8], budget: Budget) -> media::front::Result<EncodedSpan> {
        let rgb = decode::decode(bytes)?;
        let framed = self.frame(rgb.h, rgb.w, budget);
        Image {
            patch: self.spec.patch,
            block: self.spec.block,
            mrope: self.spec.mrope,
        }
        .encode(&rgb, framed, decode::resize_exact)
    }
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

    const DEPLOYMENT: &str = "qwen35-d0.8b-vision-u4g64-kv-bf16";

    #[test]
    fn media_pipe_is_the_pinned_preprocessing_every_case() {
        a_real_png_goes_through_the_whole_pipe();
        the_span_spells_itself_out_of_the_tokenizers_own_ids();
        the_digest_is_stable_and_separates_two_images_one_run_cannot();
        the_refusals_fire_by_name();
        a_video_frame_is_the_same_preprocessing_as_a_still();
    }

    fn a_real_png_goes_through_the_whole_pipe() {
        let fe = Front::of(DEPLOYMENT);
        assert_eq!((fe.spec.patch, fe.spec.block), (16, 2));
        let bytes = png::png_rgb(200, 120, png::ramp);
        let span = fe
            .encode(&bytes, Budget::Still)
            .expect("a well-formed PNG encodes");

        assert_eq!(
            fe.frame(120, 200, Budget::Still),
            (224, 352),
            "the resize policy"
        );
        let (gh, gw) = (14, 22);
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
            span.rows as usize * 3 * 16 * 16,
            "the payload is `rows · 3 · P²` bytes"
        );
        assert_eq!(span.positions.len(), span.rows as usize * 2);
        let first = span.payload[0];
        assert!(
            span.payload.iter().any(|&v| v != first),
            "the decoded image is uniform, so nothing downstream was exercised"
        );
    }

    fn the_span_spells_itself_out_of_the_tokenizers_own_ids() {
        let fe = Front::of(DEPLOYMENT);
        assert_eq!(fe.spec.prefix, "<|vision_start|>");
        assert_eq!(fe.spec.placeholder, "<|image_pad|>");
        assert_eq!(fe.spec.suffix, "<|vision_end|>");

        let bytes = png::png_rgb(64, 64, png::ramp);
        let mut span = fe.encode(&bytes, Budget::Still).expect("encodes");
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
            "and the span did not: a digest is over the preprocessed span, never its spelling"
        );
    }

    fn the_digest_is_stable_and_separates_two_images_one_run_cannot() {
        let fe = Front::of(DEPLOYMENT);
        let one = fe
            .encode(&png::png_rgb(96, 96, png::ramp), Budget::Still)
            .expect("one");
        let again = fe
            .encode(&png::png_rgb(96, 96, png::ramp), Budget::Still)
            .expect("again");
        let other = fe
            .encode(
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

        assert_eq!(span_digest(&one).len(), 32, "blake3");
        assert_eq!(span_digest(&one), span_digest(&again));
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
        assert_ne!(span_digest(&a), span_digest(&b), "and the digest must");
    }

    fn the_refusals_fire_by_name() {
        let fe = Front::of(DEPLOYMENT);
        let empty = fe
            .encode(&[], Budget::Still)
            .expect_err("zero bytes are refused");
        assert_eq!(empty.name(), "Decode", "{empty}");
        let garbage = fe
            .encode(b"this is not a picture, it is a sentence", Budget::Still)
            .expect_err("prose is refused");
        assert!(matches!(garbage, Fault::Decode(_)));
    }

    fn a_video_frame_is_the_same_preprocessing_as_a_still() {
        let fe = Front::of(DEPLOYMENT);
        let bytes = png::png_rgb(80, 60, png::ramp);
        let still = fe.encode(&bytes, Budget::Still).expect("still");
        let frame = fe.encode(&bytes, Budget::VideoFrame).expect("a frame");
        assert_eq!(still, frame);
    }
}

mod gemma {
    use super::*;

    const DEPLOYMENT: &str = "gemma4-e4b-vision-bf16-kv-bf16";

    #[test]
    fn media_pipe_is_the_pinned_preprocessing_1_every_case() {
        a_real_png_goes_through_the_whole_pipe();
        a_video_frame_gets_the_frame_budget();
        the_span_spells_itself_out_of_the_tokenizers_own_ids();
    }

    fn a_real_png_goes_through_the_whole_pipe() {
        let fe = Front::of(DEPLOYMENT);
        assert_eq!((fe.spec.patch, fe.spec.block), (16, 3));
        let bytes = png::png_rgb(200, 120, png::ramp);
        let span = fe
            .encode(&bytes, Budget::Still)
            .expect("a well-formed PNG encodes");

        assert_eq!(fe.frame(120, 200, Budget::Still), (576, 1008));
        let (gh, gw) = (36, 63);
        assert_eq!(
            span.rows,
            gh * gw,
            "one payload row per patch, and no padding"
        );
        assert_eq!(span.patch_grid, Grid::still(gh, gw));
        assert_eq!(span.token_count, gh * gw / 9);
        assert_eq!(span.grid, Grid::still(gh / 3, gw / 3));
        assert_eq!(
            span.position_span, span.token_count,
            "1-D rope advances by the rows the span occupies"
        );
        assert!(!span.uses_mrope, "gemma's trunk rotates scalar");
        assert_eq!(span.payload.len(), span.rows as usize * 3 * 16 * 16);
        assert_eq!(span.positions.len(), span.rows as usize * 2);
    }

    fn a_video_frame_gets_the_frame_budget() {
        let fe = Front::of(DEPLOYMENT);
        let bytes = png::png_rgb(200, 120, png::ramp);
        let still = fe.encode(&bytes, Budget::Still).expect("still");
        let frame = fe.encode(&bytes, Budget::VideoFrame).expect("a frame");
        assert!(frame.token_count < still.token_count);
        assert!(frame.token_count <= 70);
        assert!(still.token_count <= 280);
        assert_ne!(span_digest(&still), span_digest(&frame));
    }

    fn the_span_spells_itself_out_of_the_tokenizers_own_ids() {
        let fe = Front::of(DEPLOYMENT);
        assert_eq!(fe.spec.prefix, "<|image>");
        assert_eq!(fe.spec.placeholder, "<|image|>");
        assert_eq!(fe.spec.suffix, "<image|>");
        let bytes = png::png_rgb(96, 96, png::ramp);
        let mut span = fe.encode(&bytes, Budget::Still).expect("encodes");
        span.spell_with(vec![262_144], 262_145, vec![262_146]);
        let toks = span.tokens();
        assert_eq!(toks.len(), 1 + span.token_count as usize + 1);
        assert!(toks[1..toks.len() - 1].iter().all(|&t| t == 262_145));
    }
}
