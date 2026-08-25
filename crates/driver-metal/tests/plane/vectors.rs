//! Reference vectors computed by the REAL CUDA kernels, carried here.
//!
//! # Why a second reference exists at all
//!
//! Every other comparison in this sweep is against a model written in Rust
//! from the Metal shader's own body. That is a strong reference against an
//! index arithmetic and a weak one against a TRANSCRIPTION: a model written
//! by reading the shader agrees with the shader about anything the shader's
//! author got wrong on purpose. `kernels-metal` was written last week from
//! `kernels-cuda`, and the question a model cannot answer is whether the
//! reading was right.
//!
//! There is no CUDA on an Apple machine, so the two cannot be fired side by
//! side the way the wgpu and Vulkan planes fire theirs. What can be done is
//! what `crates/driver-cuda/tests/oracle/` already does for the cuBLAS
//! service: run the real kernel on a real card, record what went in and what
//! came out, and carry the transcript. `tests/fixtures/cuda_reference.cu` is
//! that program and `tests/fixtures/run.sh` regenerates it; the four `.txt`
//! files beside them are what it produced on an L40S.
//!
//! # The format, and why it is text
//!
//! One `case` per launch, one `array` per operand, every element written as
//! the eight hex digits of its little-endian bit pattern. Hex rather than a
//! decimal literal because a reference vector that does not round-trip is not
//! a reference; text rather than a binary blob because a fixture nobody can
//! read is a magic number with a checksum, and this tree does not keep those.
//!
//! A bf16 operand is recorded as the f32 its sixteen bits widen to, so
//! [`super::alloc_bf16`] narrows it back to exactly what the CUDA kernel was
//! handed.

use std::collections::HashMap;
use std::path::PathBuf;

/// One `array` row: four-byte elements, and which kind they are.
enum Array {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

/// One `case`: the operands of a single launch, by name.
pub struct Case {
    name: String,
    arrays: HashMap<String, Array>,
}

impl Case {
    /// The f32 array `name`, or a panic naming what is missing.
    pub fn f32s(&self, name: &str) -> &[f32] {
        match self.arrays.get(name) {
            Some(Array::F32(v)) => v,
            Some(Array::I32(_)) => panic!("`{}.{name}` is an i32 array", self.name),
            None => panic!("`{}` has no array `{name}`", self.name),
        }
    }

    /// The i32 array `name`.
    pub fn i32s(&self, name: &str) -> &[i32] {
        match self.arrays.get(name) {
            Some(Array::I32(v)) => v,
            Some(Array::F32(_)) => panic!("`{}.{name}` is an f32 array", self.name),
            None => panic!("`{}` has no array `{name}`", self.name),
        }
    }

    /// A one-element f32 array, which is how the format spells a scalar.
    pub fn f32(&self, name: &str) -> f32 {
        let a = self.f32s(name);
        assert_eq!(a.len(), 1, "`{}.{name}` is not a scalar", self.name);
        a[0]
    }

    /// A one-element i32 array.
    pub fn i32(&self, name: &str) -> i32 {
        let a = self.i32s(name);
        assert_eq!(a.len(), 1, "`{}.{name}` is not a scalar", self.name);
        a[0]
    }

    /// A one-element i32 array, as the extent it states.
    pub fn at(&self, name: &str) -> usize {
        let n = self.i32(name);
        usize::try_from(n).unwrap_or_else(|_| panic!("`{}.{name}` is {n}", self.name))
    }
}

/// One fixture file.
pub struct Vectors {
    file: String,
    cases: HashMap<String, Case>,
}

impl Vectors {
    /// Read `tests/fixtures/<file>`.
    ///
    /// # Panics
    ///
    /// On anything the format does not allow, naming the line. A fixture is
    /// data a test's whole claim rests on, so a malformed one has to stop the
    /// run rather than be skipped past.
    pub fn read(file: &str) -> Self {
        let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|why| panic!("the fixture at {path:?} reads: {why}"));

        let mut cases: HashMap<String, Case> = HashMap::new();
        let mut open: Option<String> = None;
        let mut pending: Option<(String, bool, usize)> = None;
        let mut words: Vec<u32> = Vec::new();

        let close = |cases: &mut HashMap<String, Case>,
                     open: &Option<String>,
                     pending: &mut Option<(String, bool, usize)>,
                     words: &mut Vec<u32>| {
            let Some((name, floats, len)) = pending.take() else {
                return;
            };
            assert_eq!(
                words.len(),
                len,
                "`{name}` states {len} elements and carries {}",
                words.len()
            );
            let case = open.as_ref().expect("an array opens inside a case");
            let array = if floats {
                Array::F32(words.iter().copied().map(f32::from_bits).collect())
            } else {
                Array::I32(words.iter().map(|w| *w as i32).collect())
            };
            let held = cases.get_mut(case).expect("the case was opened");
            assert!(
                held.arrays.insert(name.clone(), array).is_none(),
                "`{case}` names `{name}` twice"
            );
            words.clear();
        };

        for (n, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(name) = line.strip_prefix("case ") {
                close(&mut cases, &open, &mut pending, &mut words);
                let name = name.trim().to_string();
                assert!(
                    cases
                        .insert(
                            name.clone(),
                            Case {
                                name: name.clone(),
                                arrays: HashMap::new(),
                            },
                        )
                        .is_none(),
                    "{file}:{} names case `{name}` twice",
                    n + 1
                );
                open = Some(name);
            } else if let Some(rest) = line.strip_prefix("array ") {
                close(&mut cases, &open, &mut pending, &mut words);
                let mut parts = rest.split_whitespace();
                let name = parts.next().expect("an array is named").to_string();
                let kind = parts.next().expect("an array states its kind");
                let len: usize = parts
                    .next()
                    .expect("an array states its length")
                    .parse()
                    .expect("an array's length is a number");
                let floats = match kind {
                    "f32" => true,
                    "i32" => false,
                    other => panic!("{file}:{} states kind `{other}`", n + 1),
                };
                pending = Some((name, floats, len));
            } else {
                assert!(
                    pending.is_some(),
                    "{file}:{} carries elements outside an array",
                    n + 1
                );
                for token in line.split_whitespace() {
                    words.push(u32::from_str_radix(token, 16).unwrap_or_else(|_| {
                        panic!(
                            "{file}:{} holds `{token}`, which is not eight hex digits",
                            n + 1
                        )
                    }));
                }
            }
        }
        close(&mut cases, &open, &mut pending, &mut words);

        Self {
            file: file.to_string(),
            cases,
        }
    }

    /// The case `name`, or a panic saying which file does not hold it.
    pub fn case(&self, name: &str) -> &Case {
        self.cases
            .get(name)
            .unwrap_or_else(|| panic!("{} has no case `{name}`", self.file))
    }

    /// How many cases the file holds, so a test can say it read them all.
    pub fn len(&self) -> usize {
        self.cases.len()
    }
}
