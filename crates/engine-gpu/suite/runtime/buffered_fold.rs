//! A buffered fold is the fold it replaces (Metal's
//! `a_buffered_fold_is_the_fold_it_replaces`, on this shell): a
//! `RsVerb::Buffer` fire leaves every recurrent state as it was and lands the
//! rows it would have folded in the lane's rs buffer; a later fold replaying
//! them, truncated at any length, leaves the state a plain fold of as many
//! leaves. Every state a fire advances takes part: the gated-delta layers'
//! convs and scans, a conv no recurrence reads (the PLE conv of the qwen_4
//! package, `poem_compiler::rs::Feeds::Alone`), whose state a buffered fire
//! would otherwise advance past rows a partial fold never takes, and the
//! PLE's n-gram id window. qwen38-flash-next-micro (gated delta and the PLE)
//! and qwen35-tiny (gated delta alone), each over its seeded checkpoint.
//!
//! A state is the fold it must be when its distance from it is under a
//! tenth of the distance between two different folds of the window (Metal's
//! and CUDA's bar): a fold that took rows it should not have moves the state
//! by about that distance, rounding carried through the layers by a small
//! fraction of it. Integer planes (the id window) are exact.
//!
//! A lane's logits answer what the fold's answer, its rows fired as the
//! lane fired them, within one bf16 gap of each logit (the rounding of a
//! value the head reads) and what the fold's own logits move by when the
//! same rows are fired in another composition (the claim's: a fold of the
//! window in one fire, a lane alone): a reference computed both ways. A
//! fire's dense kernel is its rung's (`engine_gpu::dense`: a vector kernel
//! to a few rows, a tiled one past them, picked by measured rates), and
//! kernels of other rungs accumulate in other orders, so the same rows
//! folded in one fire of nine and in fires of six, one and two answer apart
//! by what those orders carry through the layers (a few hundredths of a
//! logit on qwen35-tiny), however the junk row is handled. A claim whose
//! fold fires its rows one way only states no such spread.

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_gpu::store::layout::FIXED;

use crate::backend::{Lane, Seated, Shell};
use crate::model_logits::{CONTEXT, argmax, fixture, load, word};

const WINDOW: usize = 6;

/// The discard rounds of claim 9.
const ROUNDS: u32 = 8;

/// A state row as the record lays it: its name, its bytes, and whether it
/// holds integers (the n-gram id window) rather than f32 (this shell keeps
/// every float state in f32).
struct Plane {
    name: String,
    bytes: usize,
    integer: bool,
}

struct Model {
    shell: Shell,
    trace: poem_ir::Trace,
    /// The record's state planes, in record order.
    planes: Vec<Plane>,
    /// Tokens an rs page.
    per_page: u32,
}

impl Model {
    fn load(name: &str) -> Option<Model> {
        let fixture = fixture(name, None);
        let shell = load(&fixture, CONTEXT)?;
        assert!(shell.serves_rs_verbs(), "{name}: the shell buffers rs rows");
        assert!(
            shell.state_slot_bytes() > 0,
            "{name}: a recurrent plan has state"
        );
        let rs = shell.pools().rs_rows().expect("the pool lays an rs list");
        eprintln!(
            "{name}: {} rs bytes a slot of {} tokens, {} state bytes a record",
            rs.bytes,
            rs.tokens,
            shell.state_slot_bytes()
        );
        let per_page = rs.tokens;
        let planes: Vec<Plane> = fixture
            .trace
            .caches
            .iter()
            .filter_map(|row| match row {
                poem_ir::CacheRow::State {
                    name, slab, dtype, ..
                } => Some(Plane {
                    name: name.clone(),
                    bytes: 4 * slab.iter().product::<u64>() as usize,
                    integer: matches!(
                        dtype,
                        poem_ir::Dtype::I32 | poem_ir::Dtype::U32 | poem_ir::Dtype::I64
                    ),
                }),
                _ => None,
            })
            .collect();
        assert_eq!(
            planes.iter().map(|plane| plane.bytes as u64).sum::<u64>(),
            shell.state_slot_bytes(),
            "{name}: the state rows, four bytes an element, are the record"
        );
        Some(Model {
            shell,
            trace: fixture.trace,
            planes,
            per_page,
        })
    }

    /// The state rows of seat `slot`'s record.
    fn state(&mut self, slot: u32) -> Vec<u8> {
        let pieces = self.shell.pools().pieces();
        let fixed = *self
            .shell
            .pool_layout()
            .shape(FIXED)
            .expect("a recurrent pool lays out records");
        let refs: Vec<u32> = (0..pieces)
            .map(|p| {
                fixed
                    .seat_ref(slot * pieces + p)
                    .expect("the seat's record")
            })
            .collect();
        self.shell
            .state_bytes(&refs)
            .expect("the record reads back")
    }

    fn open(&mut self, slot: u32) {
        self.shell.open(slot).expect("the slot opens");
    }

    /// Fires `lanes` (`(slot, tokens, verb, reset, held)`), seat-addressed.
    fn fire(&mut self, lanes: &[(u32, &[u32], &RsVerb, RsReset, u32)]) -> Vec<Vec<f32>> {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .map(|&(slot, tokens, rs, rs_reset, held)| Seated {
                rs,
                rs_reset,
                held: Some(held),
                ..Seated::of(Lane {
                    slot,
                    word: word(&self.trace, tokens.len() as u32),
                    tokens,
                })
            })
            .collect();
        self.shell.fire_seated(&seated).expect("the fire runs")
    }

    fn one(
        &mut self,
        slot: u32,
        tokens: &[u32],
        rs: &RsVerb,
        reset: RsReset,
        held: u32,
    ) -> Vec<f32> {
        self.fire(&[(slot, tokens, rs, reset, held)]).remove(0)
    }

    /// A buffering verb over the seat's own rs run (a seat-addressed lane's
    /// pages are its seat's, so only their count is read) reaching `tokens`
    /// past the base: its state holds the slots below `folded`, it replays
    /// the buffered slots from there to where its rows land, then folds
    /// `fold` of them.
    fn buffer_over(&self, tokens: u32, folded: u32, fold: u32) -> RsVerb {
        RsVerb::Buffer {
            pages: vec![0; tokens.div_ceil(self.per_page.max(1)) as usize],
            base: 0,
            folded,
            fold: FoldLen::Host(fold),
            lineage: engine::fire::Lineage::Set,
        }
    }

    fn buffer(&self, folded: u32, fold: u32) -> RsVerb {
        self.buffer_over(2 * WINDOW as u32, folded, fold)
    }

    /// Where `got` and `want` differ, plane by plane: how many elements, the
    /// largest difference and the plane's largest value.
    fn tell_planes(&self, got: &[u8], want: &[u8]) {
        let mut at = 0usize;
        for plane in &self.planes {
            let (g, w) = (&got[at..at + plane.bytes], &want[at..at + plane.bytes]);
            at += plane.bytes;
            if g == w {
                continue;
            }
            let (gv, wv) = (words(g), words(w));
            let apart = gv.iter().zip(&wv).filter(|(x, y)| x != y).count();
            if plane.integer {
                eprintln!(
                    "plane {} (integers): {apart} of {} apart",
                    plane.name,
                    gv.len()
                );
                continue;
            }
            let (gv, wv) = (f32s(&gv), f32s(&wv));
            let worst = gv
                .iter()
                .zip(&wv)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            let largest = wv.iter().fold(0f32, |m, v| m.max(v.abs()));
            eprintln!(
                "plane {}: {apart} of {} apart, max |diff| {worst:e}, max |value| {largest:e}",
                plane.name,
                gv.len()
            );
        }
    }

    /// The distance between two states: the L2 norm of their difference
    /// over every float plane. With `exact`, `Err` names an integer plane
    /// that differs at all; without, integer planes are not counted (two
    /// different folds carry different ids).
    fn distance(&self, a: &[u8], b: &[u8], exact: bool) -> Result<f64, String> {
        assert_eq!(a.len(), b.len(), "two records of one load");
        let mut at = 0usize;
        let mut sum = 0f64;
        for plane in &self.planes {
            let (x, y) = (&a[at..at + plane.bytes], &b[at..at + plane.bytes]);
            at += plane.bytes;
            if plane.integer {
                if exact && x != y {
                    return Err(format!("integer plane {} differs", plane.name));
                }
                continue;
            }
            sum += f32s(&words(x))
                .iter()
                .zip(&f32s(&words(y)))
                .map(|(p, q)| (f64::from(*p) - f64::from(*q)).powi(2))
                .sum::<f64>();
        }
        assert_eq!(at, a.len(), "the planes cover the record");
        Ok(sum.sqrt())
    }

    /// `got` is the state `want` is: under a tenth of `scale`, the distance
    /// between two different folds of the window, from it; integer planes
    /// exact.
    fn assert_same_state(&self, got: &[u8], want: &[u8], what: &str, scale: f64) {
        let d = match self.distance(got, want, true) {
            Ok(d) => d,
            Err(why) => {
                self.tell_planes(got, want);
                panic!("{what}: {why}");
            }
        };
        eprintln!(
            "{what}: {d:e} from the fold, against {scale:e} between two folds ({:.1e} of it)",
            d / scale
        );
        if d >= scale / 10.0 {
            self.tell_planes(got, want);
            panic!(
                "{what}: the state is {d:e} from the fold, past a tenth of the {scale:e} \
                 between two folds"
            );
        }
    }

    /// `got` is `want` bit for bit (a state a buffering fire must leave).
    fn assert_untouched(&self, got: &[u8], want: &[u8], what: &str) {
        if got != want {
            self.tell_planes(got, want);
            panic!("{what}");
        }
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

fn f32s(words: &[u32]) -> Vec<f32> {
    words.iter().map(|&w| f32::from_bits(w)).collect()
}

fn window(seed: u32) -> Vec<u32> {
    (0..WINDOW as u32)
        .map(|at| (seed + at * 37) % 256)
        .collect()
}

fn spread(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The gap between `x` and the next bf16 away from zero: what one rounding
/// of a bf16 value moves it by.
fn bf16_gap(x: f32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent.max(-126) - 7)
}

/// `got` answers what the fold answers, `want`: each logit within one bf16
/// gap of the larger of the two, and the spread of the fold's own logits
/// between `want` and `other`, the same rows folded in another composition
/// (`want` itself where the lane and the fold fire their rows alike).
fn assert_answers(got: &[f32], want: &[f32], other: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: one vocabulary");
    assert_eq!(other.len(), want.len(), "{what}: one vocabulary");
    // The logit nearest its room: (index, apart, room).
    let mut nearest = (0usize, 0f32, f32::INFINITY);
    for (at, ((&g, &w), &o)) in got.iter().zip(want).zip(other).enumerate() {
        assert!(
            g.is_finite() && w.is_finite() && o.is_finite(),
            "{what}: logit {at} is not finite"
        );
        let room = bf16_gap(g.abs().max(w.abs())) + (w - o).abs();
        let apart = (g - w).abs();
        if apart > 0.0 && apart / room > nearest.1 / nearest.2 {
            nearest = (at, apart, room);
        }
    }
    let (at, apart, room) = nearest;
    eprintln!(
        "{what}: logit spread {:.4}, the fold's own two ways {:.4}; nearest its room: logit {at} \
         {apart:.4} apart of {room:.4} ({:.2} of it); argmax {} vs {}",
        spread(got, want),
        spread(want, other),
        apart / room,
        argmax(got),
        argmax(want)
    );
    assert!(
        apart <= room,
        "{what}: logit {at} answers {apart} apart ({} against {}), past its room {room}: one bf16 \
         gap of the two and the fold's own {} between its two compositions",
        got[at],
        want[at],
        (want[at] - other[at]).abs()
    );
}

/// The same contract over a gated-delta model with no PLE (Qwen3.5).
#[test]
fn a_gdn_buffered_fold_is_the_fold_it_replaces() {
    if let Some(model) = Model::load("qwen35-tiny") {
        holds_the_contract(model, "qwen35-tiny");
    }
}

/// A buffering fire leaves the state as it was; a fold replaying it is the
/// fold it deferred, at any length; a folding lane beside a buffering one
/// answers as alone; and rounds of a two-row window with a rejected row
/// leave the state a plain fold of the survivors leaves. Four seats: each
/// claim reopens the ones it takes.
fn holds_the_contract(mut m: Model, name: &str) {
    let fold = RsVerb::Fold;
    let tokens = window(11);
    let k = WINDOW as u32;
    let (fresh, held) = (RsReset::Fresh, RsReset::Held);

    m.open(0);
    let zeroed = m.state(0);
    assert!(
        zeroed.iter().all(|byte| *byte == 0),
        "an opened slot's state is zero"
    );
    let reference = m.one(0, &tokens, &fold, fresh, 0);
    let folded = m.state(0);
    assert_ne!(
        folded, zeroed,
        "a fold over {WINDOW} tokens left the state untouched"
    );

    // Claim 1: a buffering fire leaves the state, and answers the fold's logits.
    m.open(1);
    let verb = m.buffer(0, 0);
    let buffered = m.one(1, &tokens, &verb, fresh, 0);
    let state = m.state(1);
    m.assert_untouched(
        &state,
        &zeroed,
        &format!("{name} claim 1: a `RsVerb::Buffer` fire perturbed the state"),
    );
    assert_answers(
        &buffered,
        &reference,
        &reference,
        &format!("{name} claim 1: a buffered window against the folding window"),
    );

    // Claim 3: a replay truncated at j is a fold of j.
    let second = window(101);
    let j = 3u32;
    m.open(2);
    m.one(2, &tokens[..j as usize], &fold, fresh, 0);
    let folded_j = m.state(2);
    assert_ne!(
        folded_j, folded,
        "folds of {j} and {k} leave the same state"
    );
    let scale = m
        .distance(&folded_j, &folded, false)
        .expect("integer planes are not counted");
    assert!(
        scale > 0.0,
        "the folds of {j} and {k} differ in their float planes"
    );
    m.open(3);
    let (a, b) = (m.buffer(0, 0), m.buffer(0, j));
    m.one(3, &tokens, &a, fresh, 0);
    m.one(3, &second, &b, held, k);
    let state = m.state(3);
    m.assert_same_state(
        &state,
        &folded_j,
        &format!("{name} claim 3: a replay truncated at {j} against a fold of {j}"),
        scale,
    );

    // Claim 4: a fire buffering its own rows and folding j of them.
    m.open(1);
    let mixed = m.one(1, &tokens, &b, fresh, 0);
    let state = m.state(1);
    m.assert_same_state(
        &state,
        &folded_j,
        &format!("{name} claim 4: the mixed row against a fold of {j}"),
        scale,
    );
    assert_answers(
        &mixed,
        &reference,
        &reference,
        &format!("{name} claim 4: the mixed row against the folding window"),
    );

    // Claim 5: a replay of A ahead of B is Fold(A), and B reads as after it.
    m.open(2);
    let all = m.buffer(0, k);
    m.one(2, &tokens, &a, fresh, 0);
    let round = m.one(2, &second, &all, held, k);
    let state = m.state(2);
    m.assert_same_state(
        &state,
        &folded,
        &format!("{name} claim 5: replay(A) ahead of B against Fold(A)"),
        scale,
    );
    m.open(3);
    m.one(3, &tokens, &fold, fresh, 0);
    let plain = m.one(3, &second, &fold, held, k);
    assert_answers(
        &round,
        &plain,
        &plain,
        &format!("{name} claim 5: B after replay(A) against B after Fold(A)"),
    );

    // Claim 6: a folding lane beside a buffering one answers as beside a
    // folding one, and as alone: its rows fired beside a peer's are a fire
    // of twice the rows, which a rung's kernel past the alone fire's may
    // serve, so alone is the fold's other way.
    let third = window(203);
    m.open(3);
    let alone = m.one(3, &third, &fold, fresh, 0);
    let alone_state = m.state(3);
    m.open(3);
    m.open(1);
    let paired = m.fire(&[(3, &third, &fold, fresh, 0), (1, &tokens, &fold, fresh, 0)]);
    m.open(3);
    m.open(1);
    let beside = m.fire(&[(3, &third, &fold, fresh, 0), (1, &tokens, &a, fresh, 0)]);
    let state = m.state(3);
    m.assert_same_state(
        &state,
        &alone_state,
        &format!("{name} claim 6: a folding lane beside a buffering peer"),
        scale,
    );
    let state = m.state(1);
    m.assert_untouched(
        &state,
        &zeroed,
        &format!("{name} claim 6: the buffering peer folded"),
    );
    assert_answers(
        &beside[0],
        &paired[0],
        &alone,
        &format!("{name} claim 6: a folding lane beside a buffering peer against a folding one"),
    );

    // Claim 9: rounds of a two-row window whose second row is rejected: each
    // round's fold keeps the survivor of the round before, and the state is
    // a plain fold of the survivors.
    let steps: Vec<u32> = (0..12u32).map(|i| (503 + i * 37) % 256).collect();
    let junk = 7u32;
    m.open(0);
    m.one(0, &tokens, &fold, fresh, 0);
    m.open(2);
    m.one(2, &tokens, &fold, fresh, 0);
    for r in 0..ROUNDS {
        let rows = [steps[r as usize], junk];
        let reach = k + ROUNDS + 1;
        let verb = if r == 0 {
            m.buffer_over(reach, k, 0)
        } else {
            m.buffer_over(reach, k + r - 1, 1)
        };
        let win = m.one(0, &rows, &verb, held, k + r);
        // The fold two ways: fired as the window's lane fired its rows (the
        // window, each survivor before this round's in a fire of its own,
        // then the round's two rows); and the window and the survivors
        // folded in one fire, the junk row alone after them.
        let mut prefix = tokens.clone();
        prefix.extend_from_slice(&steps[..=r as usize]);
        m.open(3);
        m.one(3, &prefix, &fold, fresh, 0);
        let ref1 = m.one(3, &[junk], &fold, held, prefix.len() as u32);
        m.open(3);
        m.one(3, &tokens, &fold, fresh, 0);
        for i in 0..r {
            m.one(3, &steps[i as usize..=i as usize], &fold, held, k + i);
        }
        let ref2 = m.one(3, &rows, &fold, held, k + r);
        assert_answers(
            &win,
            &ref2,
            &ref1,
            &format!("{name} claim 9: round {r}, the window's junk row against its fold"),
        );
        if r > 0 {
            m.one(
                2,
                &steps[r as usize - 1..r as usize],
                &fold,
                held,
                k + r - 1,
            );
            let (got, want) = (m.state(0), m.state(2));
            m.assert_same_state(
                &got,
                &want,
                &format!(
                    "{name} claim 9: round {r}, s_{} folded through the discard path",
                    r - 1
                ),
                scale,
            );
        }
    }
    eprintln!(
        "{name}: every claim holds over {} state bytes",
        folded.len()
    );
    // Every fire above (buffering lanes, replays, folds beside them, the
    // windows of an MTP verify) took only pipelines the load's walks named:
    // the committed recurrences and convs and the buffered rows' copies
    // among them.
    let built = m.shell.built_serving();
    assert!(
        built.is_empty(),
        "{name}: a buffering fire built pipelines no walk of the load named: {built:?}"
    );
}
