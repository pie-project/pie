//! Python bindings.
//!
//! Exposes exactly what a serving backend needs: compile a grammar against a
//! vocabulary once, then run a cheap matcher per request. The matcher shares
//! the compiled artifact, so admitting a new request costs an integer stack
//! rather than a rebuilt automaton.

// Compiling a schema makes and frees hundreds of thousands of small vectors -
// one per group of tokens, one per reading, one per reading's terminals - and
// freeing them was measured at 186 ms of a 2.45 s run over thirty schemas, or
// 286 ns a group. That is the allocator, not the code, so this replaces the
// allocator rather than restructuring what it is asked to do.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::sync::Arc;

use engrain_ir::grammar::Grammar;
use engrain_ir::regex::regex_to_grammar;
use engrain_tables::pipeline::{
    Failure, Limits, compile_grammar_within, compile_json_schema as compile_schema,
};
use engrain_run::Matcher as RunMatcher;
use engrain_tables::Artifact;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use pyo3::create_exception;
use pyo3::types::{PyByteArray, PyBytes, PyDict};

create_exception!(
    _engrain,
    CompileError,
    PyValueError,
    "A grammar the compiler refused, carrying the stage that refused it."
);

/// The stage as a value rather than as prose. A serving engine has to decide
/// whether to fall back to another backend or to reject the request, and
/// those are different answers for different stages: a budget may be raised,
/// a lowering failure will not be.
fn refusal(failure: Failure) -> PyErr {
    let stage = match failure {
        Failure::Lowering => "lowering",
        Failure::Lexer => "lexer",
        Failure::Productions => "productions",
        Failure::Conflict => "conflict",
        Failure::Emit => "emit",
    };
    Python::attach(|python| {
        let error = CompileError::new_err(format!("{failure}"));
        // A failure here would mean the exception object refused an attribute,
        // which cannot happen for a Python-level class; the message still
        // carries the stage either way, so this must not itself raise.
        let _ = error.value(python).setattr("stage", stage);
        error
    })
}

fn compile(
    vocabulary: &[Vec<u8>],
    grammar: Grammar,
    lexer_states: Option<usize>,
) -> PyResult<Artifact> {
    let limits = Limits {
        lexer_states: lexer_states.unwrap_or(Limits::default().lexer_states),
        ..Default::default()
    };
    // Released for the same reason the schema path releases it: this holds no
    // Python object, takes tens of milliseconds across every core, and a
    // serving engine compiles on a thread pool while a decode loop runs.
    Python::attach(|python| {
        python.detach(|| compile_grammar_within(&grammar, vocabulary, limits))
    })
    .map_err(refusal)
}

/// A vocabulary-bound compiler. Build one per model.
#[pyclass]
pub struct Compiler {
    vocabulary: Vec<Vec<u8>>,
    /// A digest of the vocabulary, stamped onto everything this compiler
    /// produces. Two tokenizers of the same size are not the same tokenizer,
    /// and a grammar used against the wrong one gives a mask that is wrong
    /// token by token with nothing to notice it.
    digest: u64,
}

#[pymethods]
impl Compiler {
    #[new]
    fn new(vocabulary: Vec<Vec<u8>>) -> Self {
        let digest = vocabulary_digest(&vocabulary);
        Self { vocabulary, digest }
    }

    /// The digest of the vocabulary this compiler was built for.
    #[getter]
    fn vocabulary_digest(&self) -> u64 {
        self.digest
    }

    /// Compile a JSON Schema, searching the lowerings for one that is LALR(1).
    #[pyo3(signature = (
        schema,
        lexer_states = None,
        exact = false,
        max_digits = None,
        max_string = None,
        max_whitespace = None,
    ))]
    fn compile_json_schema(
        &self,
        python: Python<'_>,
        schema: &str,
        lexer_states: Option<usize>,
        exact: bool,
        max_digits: Option<u32>,
        max_string: Option<u32>,
        max_whitespace: Option<u32>,
    ) -> PyResult<CompiledGrammar> {
        let limits = Limits {
            lexer_states: lexer_states.unwrap_or(Limits::default().lexer_states),
            exact,
            max_digits,
            max_string,
            max_whitespace,
            ..Default::default()
        };
        // Compiling holds no Python object and takes tens of milliseconds
        // across every core, so holding the interpreter lock for it stops the
        // caller's other threads and makes rayon's workers contend with
        // whatever else the process is doing.
        let compiled = python
            .detach(|| compile_schema(schema, &self.vocabulary, limits))
            .map_err(refusal)?;
        Ok(CompiledGrammar {
            artifact: Arc::new(compiled.artifact),
            precision: format!("{:?}", compiled.precision),
            relaxations: compiled.relaxations,
            digest: self.digest,
        })
    }

    #[pyo3(signature = (source, root, lexer_states = None))]
    fn compile_ebnf(
        &self,
        source: &str,
        root: &str,
        lexer_states: Option<usize>,
    ) -> PyResult<CompiledGrammar> {
        let grammar = Grammar::from_ebnf(source, root)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(CompiledGrammar {
            artifact: Arc::new(compile(&self.vocabulary, grammar, lexer_states)?),
            precision: "n/a".to_string(),
            relaxations: Vec::new(),
            digest: self.digest,
        })
    }

    #[pyo3(signature = (pattern, lexer_states = None))]
    fn compile_regex(
        &self,
        pattern: &str,
        lexer_states: Option<usize>,
    ) -> PyResult<CompiledGrammar> {
        let grammar =
            regex_to_grammar(pattern).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(CompiledGrammar {
            artifact: Arc::new(compile(&self.vocabulary, grammar, lexer_states)?),
            precision: "n/a".to_string(),
            relaxations: Vec::new(),
            digest: self.digest,
        })
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// A compiled grammar, shareable across requests.
#[pyclass]
#[derive(Clone)]
pub struct CompiledGrammar {
    artifact: Arc<Artifact>,
    /// Which lowering the pipeline settled on. A benchmark that reports
    /// acceptance has to be able to separate the schemas that got the exact
    /// treatment from those that had to be relaxed.
    precision: String,
    /// What the grammar does not enforce. A mask may admit more than the
    /// schema allows - that is the direction it must err in - so a caller that
    /// needs the schema itself has to check the finished document against
    /// these, and cannot do that without being told which they are.
    relaxations: Vec<engrain_tables::pipeline::Relaxation>,
    /// The vocabulary this was compiled against, as a digest.
    digest: u64,
}

/// Order-sensitive, because a vocabulary is a mapping from token id to bytes
/// and a permutation of it is a different mapping.
fn vocabulary_digest(vocabulary: &[Vec<u8>]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for (index, token) in vocabulary.iter().enumerate() {
        for byte in index.to_le_bytes().iter().chain(token.iter()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[pymethods]
impl CompiledGrammar {
    /// A digest of the vocabulary this grammar was compiled against.
    #[getter]
    fn vocabulary_digest(&self) -> u64 {
        self.digest
    }
    #[pyo3(signature = (max_rollback = 8))]
    fn matcher(&self, max_rollback: usize) -> Matcher {
        Matcher {
            inner: RunMatcher::new(self.artifact.clone(), max_rollback),
            words: self.artifact.bitset_words as usize,
            vocabulary_size: self.artifact.vocab_size as usize,
        }
    }

    #[getter]
    fn num_groups(&self) -> usize {
        self.artifact.groups.len()
    }

    #[getter]
    fn num_parser_states(&self) -> usize {
        self.artifact.num_parser_states as usize
    }

    #[getter]
    fn num_lexer_states(&self) -> usize {
        self.artifact.num_lexer_states as usize
    }

    #[getter]
    fn num_terminals(&self) -> usize {
        self.artifact.num_terminals as usize
    }

    #[getter]
    fn bitset_words(&self) -> usize {
        self.artifact.bitset_words as usize
    }

    #[getter]
    fn precision(&self) -> &str {
        &self.precision
    }

    #[getter]
    /// What this grammar does not enforce, where, and what to change.
    ///
    /// One dictionary per finding: the keyword responsible, a JSON pointer to
    /// the place, what the mask now admits that the schema does not, and the
    /// edit that would enforce it. A sentence saying "this schema is relaxed"
    /// sends an author looking; a pointer sends them to the object.
    fn relaxations<'py>(&self, python: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.relaxations
            .iter()
            .map(|note| {
                let entry = PyDict::new(python);
                entry.set_item("keyword", &note.keyword)?;
                entry.set_item("at", &note.at)?;
                entry.set_item("effect", &note.effect)?;
                entry.set_item("remedy", &note.remedy)?;
                Ok(entry)
            })
            .collect()
    }

    #[getter]
    fn resident_bytes(&self) -> usize {
        self.artifact.resident_bytes()
    }

    /// `transitions[state * 256 + byte]`, `0xffffffff` where impossible.
    ///
    /// This is what a device-side token walk needs instead of the masks.
    fn lexer_transitions<'py>(&self, python: Python<'py>) -> Bound<'py, PyBytes> {
        let mut bytes = Vec::with_capacity(self.artifact.lexer_transitions.len() * 4);
        for word in &self.artifact.lexer_transitions {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        PyBytes::new(python, &bytes)
    }

    /// One bit per lexer state: does a lexeme may end here?
    fn lexer_accepting<'py>(&self, python: Python<'py>) -> Bound<'py, PyBytes> {
        let states = self.artifact.num_lexer_states as usize;
        let mut flags = vec![0u8; states];
        for state in 0..states {
            let from = self.artifact.accepting_offsets[state];
            let to = self.artifact.accepting_offsets[state + 1];
            flags[state] = u8::from(to > from);
        }
        PyBytes::new(python, &flags)
    }

    /// Every array a device-side mask fill needs, as one dict of bytes.
    ///
    /// This is the artifact as the GPU sees it. Nothing here is per-request:
    /// the tables are a pure function of the grammar and the vocabulary, so a
    /// decode step reads them and writes only its own stack and mask.
    fn device_arrays<'py>(&self, python: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let artifact = &self.artifact;
        let out = PyDict::new(python);

        fn words<'py>(python: Python<'py>, values: &[u32]) -> Bound<'py, PyBytes> {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            PyBytes::new(python, &bytes)
        }
        fn signed<'py>(python: Python<'py>, values: &[i32]) -> Bound<'py, PyBytes> {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            PyBytes::new(python, &bytes)
        }

        // Groups, flattened. A reading is a run of terminals plus the lexer
        // state it leaves, and a group is a run of readings, so both are CSR.
        let mut group_state = Vec::new();
        let mut group_set_kind = Vec::new();
        let mut group_set_offset = Vec::new();
        let mut group_set_length = Vec::new();
        // Readings, interned.
        //
        // A group carries its own copy of every way its tokens can be read,
        // and within one lexer state those copies repeat heavily - 40% to 72%
        // of a state's readings are a terminal sequence some other group in
        // the same state also has. Replaying each copy separately is most of
        // what the mask fill costs, so identical readings collapse to one
        // entry here and a group points at the entries it uses. What the
        // kernel can then share is the replay, not merely the bytes.
        // One entry per group, naming a length-prefixed block. A group's list
        // of ways to read its tokens is the same list in state after state -
        // measured 4.77x over the corpus - so the lists are shared here, and
        // the prefix is what lets a shared block say how long it is when it is
        // no longer followed by the next one.
        let mut reading_offsets = Vec::new();
        let mut reading_index = Vec::new();
        let mut interned_lists: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
        let mut list = Vec::new();
        let mut reading_next_state = Vec::new();
        let mut reading_term_offsets = vec![0u32];
        let mut reading_terminals = Vec::new();
        let mut interned_readings: FxHashMap<(Vec<u32>, u32), u32> = FxHashMap::default();
        for group in &artifact.groups {
            group_state.push(group.lexer_state);
            group_set_kind.push(match group.set.kind {
                engrain_tables::SetKind::Sparse => 0u32,
                engrain_tables::SetKind::Complement => 1,
                engrain_tables::SetKind::Dense => 2,
            });
            group_set_offset.push(group.set.offset);
            group_set_length.push(group.set.length);
            for reading in &group.readings {
                let key = (reading.terminals.clone(), reading.next_lexer_state);
                let index = match interned_readings.get(&key) {
                    Some(&existing) => existing,
                    None => {
                        let fresh = reading_next_state.len() as u32;
                        reading_next_state.push(reading.next_lexer_state);
                        reading_terminals.extend(reading.terminals.iter().copied());
                        reading_term_offsets.push(reading_terminals.len() as u32);
                        interned_readings.insert(key, fresh);
                        fresh
                    }
                };
                list.push(index);
            }
            let at = match interned_lists.get(&list) {
                Some(&existing) => existing,
                None => {
                    let fresh = reading_index.len() as u32;
                    reading_index.push(list.len() as u32);
                    reading_index.extend(list.iter().copied());
                    interned_lists.insert(list.clone(), fresh);
                    fresh
                }
            };
            reading_offsets.push(at);
            list.clear();
        }

        out.set_item("group_offsets", words(python, &artifact.group_offsets))?;
        out.set_item("group_state", words(python, &group_state))?;
        out.set_item("group_set_kind", words(python, &group_set_kind))?;
        out.set_item("group_set_offset", words(python, &group_set_offset))?;
        out.set_item("group_set_length", words(python, &group_set_length))?;
        out.set_item("set_payload", words(python, &artifact.set_payload))?;
        out.set_item("reading_offsets", words(python, &reading_offsets))?;
        out.set_item("reading_index", words(python, &reading_index))?;
        out.set_item("reading_next_state", words(python, &reading_next_state))?;
        out.set_item("reading_term_offsets", words(python, &reading_term_offsets))?;
        out.set_item("reading_terminals", words(python, &reading_terminals))?;
        out.set_item("action_offsets", words(python, &artifact.action_offsets))?;
        out.set_item(
            "action_terminals",
            words(python, &artifact.action_terminals),
        )?;
        out.set_item("action_values", signed(python, &artifact.action_values))?;
        out.set_item(
            "action_extra_offsets",
            words(python, &artifact.action_extra_offsets),
        )?;
        out.set_item("action_extra", signed(python, &artifact.action_extra))?;
        out.set_item("goto_offsets", words(python, &artifact.goto_offsets))?;
        out.set_item(
            "goto_nonterminals",
            words(python, &artifact.goto_nonterminals),
        )?;
        out.set_item("goto_targets", words(python, &artifact.goto_targets))?;
        out.set_item("production_lhs", words(python, &artifact.production_lhs))?;
        out.set_item(
            "production_arity",
            words(python, &artifact.production_arity),
        )?;
        out.set_item("verdict_offsets", words(python, &artifact.verdict_offsets))?;
        out.set_item("verdicts", words(python, &artifact.verdicts))?;
        out.set_item("verdict_stride", words(python, &artifact.verdict_stride))?;
        out.set_item("pending_offsets", words(python, &artifact.pending_offsets))?;
        out.set_item(
            "pending_terminals",
            words(python, &artifact.pending_terminals),
        )?;
        out.set_item("eof_terminal", artifact.eof_terminal)?;
        out.set_item("start_parser_state", artifact.start_parser_state)?;
        out.set_item("vocab_size", artifact.vocab_size)?;
        out.set_item("bitset_words", artifact.bitset_words)?;
        // How many actions the widest ACTION cell holds. One unless the grammar
        // is ambiguous, and the device compiles a replay that does not fork at
        // all when it is one - which is every grammar that used to compile.
        out.set_item(
            "max_actions",
            artifact
                .action_extra_offsets
                .windows(2)
                .map(|pair| 1 + pair[1] - pair[0])
                .max()
                .unwrap_or(1),
        )?;
        Ok(out)
    }

    /// Every group's token set as a dense bitmask, for callers that want to
    /// inspect the sets rather than the storage. The artifact itself keeps them
    /// in whichever exact form is smallest.
    fn group_bitsets<'py>(&self, python: Python<'py>) -> Bound<'py, PyBytes> {
        let words = self.artifact.bitset_words as usize;
        let vocab = self.artifact.vocab_size as usize;
        let mut bytes = Vec::with_capacity(self.artifact.groups.len() * words * 4);
        for group in &self.artifact.groups {
            let set = group.set;
            let from = set.offset as usize;
            let body = &self.artifact.set_payload[from..from + set.length as usize];
            let mut bits = match set.kind {
                engrain_tables::SetKind::Complement => vec![u32::MAX; words],
                _ => vec![0u32; words],
            };
            match set.kind {
                engrain_tables::SetKind::Sparse => {
                    for token in body {
                        bits[*token as usize / 32] |= 1u32 << (*token % 32);
                    }
                }
                engrain_tables::SetKind::Complement => {
                    for token in body {
                        bits[*token as usize / 32] &= !(1u32 << (*token % 32));
                    }
                    let spare = words * 32 - vocab;
                    if spare > 0 {
                        bits[words - 1] &= u32::MAX >> spare;
                    }
                }
                engrain_tables::SetKind::Dense => bits.copy_from_slice(body),
            }
            for word in &bits {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        PyBytes::new(python, &bytes)
    }
}

/// One request's parse state.
#[pyclass]
pub struct Matcher {
    inner: RunMatcher,
    words: usize,
    vocabulary_size: usize,
}

#[pymethods]
impl Matcher {
    /// Write the allowed-token bitmask into `buffer`, which must be a
    /// contiguous 32-bit CPU tensor of at least `bitset_words` elements.
    ///
    /// Everything in that sentence is checked. This writes through a raw
    /// pointer the caller supplies, so a CUDA tensor would have the host
    /// dereference a device address and an 8-bit tensor of the same length
    /// would be a quarter of the bytes needed - both reachable from Python,
    /// and neither one a Python exception without these checks.
    fn fill_bitmask(&self, buffer: Bound<'_, PyAny>) -> PyResult<()> {
        let device: String = buffer
            .getattr("device")?
            .getattr("type")?
            .extract()
            .unwrap_or_default();
        if device != "cpu" {
            return Err(PyValueError::new_err(format!(
                "the bitmask buffer must be on the host, not on {device}"
            )));
        }
        if !buffer.getattr("is_contiguous")?.call0()?.extract::<bool>()? {
            return Err(PyValueError::new_err(
                "the bitmask buffer must be contiguous",
            ));
        }
        let width: usize = buffer
            .getattr("dtype")?
            .getattr("itemsize")?
            .extract()
            .unwrap_or(0);
        if width != 4 {
            return Err(PyValueError::new_err(format!(
                "the bitmask buffer must hold 32-bit words, not {width}-byte ones"
            )));
        }
        let pointer: usize = buffer.getattr("data_ptr")?.call0()?.extract()?;
        let length: usize = buffer.getattr("numel")?.call0()?.extract()?;
        if length < self.words {
            return Err(PyValueError::new_err(format!(
                "buffer holds {length} words but {} are needed",
                self.words
            )));
        }
        // Safety: checked above to be a contiguous 32-bit host buffer of at
        // least `self.words` elements.
        let slice = unsafe { std::slice::from_raw_parts_mut(pointer as *mut u32, self.words) };
        self.inner.fill_bitmask(slice);
        Ok(())
    }

    fn accept_token(&mut self, token: u32) -> bool {
        self.inner.accept_token(token).is_ok()
    }

    /// Every token the matcher admits right now.
    ///
    /// The bitmask path is what serving uses; this is for tests and
    /// benchmarks, which need the set itself rather than a buffer to apply.
    fn allowed_tokens(&self) -> Vec<u32> {
        let mut words = vec![0u32; self.words];
        self.inner.fill_bitmask(&mut words);
        let mut allowed = Vec::new();
        for (index, word) in words.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                let token = index as u32 * 32 + bit;
                if (token as usize) < self.vocabulary_size {
                    allowed.push(token);
                }
            }
        }
        allowed
    }

    fn rollback(&mut self, tokens: usize) {
        self.inner.rollback(tokens);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn is_terminated(&self) -> bool {
        self.inner.is_terminated()
    }

    fn can_terminate(&self) -> bool {
        self.inner.can_terminate()
    }

    fn terminate(&mut self) {
        self.inner.terminate();
    }

    #[getter]
    fn lexer_state(&self) -> u32 {
        self.inner.lexer_state()
    }

    /// How many readings of the input are still alive.
    #[getter]
    fn num_configs(&self) -> usize {
        self.inner.num_configs()
    }

    /// The parser stack of the first live configuration, so a caller can put a
    /// device matcher into the same state.
    fn stack(&self) -> Vec<u32> {
        self.inner.stack().to_vec()
    }

    /// Every live configuration as `(lexer_state, stack)`, which is the whole
    /// parse state a device matcher has to be given.
    fn configurations(&self) -> Vec<(u32, Vec<u32>)> {
        self.inner.configurations()
    }

    /// How deep the deepest live configuration's stack is.
    ///
    /// The cheap form of `max(len(stack) for _, stack in configurations())`,
    /// which a serving batch runs once per row on any step that met the
    /// device's depth ceiling. Going through `configurations` clones every
    /// stack of every row to read one integer from each.
    fn max_stack_depth(&self) -> usize {
        self.inner.max_stack_depth()
    }

    #[getter]
    fn parser_state(&self) -> u32 {
        self.inner.parser_state()
    }

    /// Groups the parser admits right now, for the device path.
    fn admissible_groups(&self) -> Vec<usize> {
        self.inner.admissible_groups()
    }
}

/// Pack many matchers' parse states into flat buffers, ready for one upload.
///
/// The device path needs every sequence's configuration set on the accelerator
/// each step, and building that a row at a time in Python costs more than every
/// kernel it feeds: 2.3 ms at batch 512 against 84 us for the fill. Most of that
/// is not the copy but the conversion - `configurations()` builds a list of
/// tuples of lists per sequence, and then Python writes each one into an array.
///
/// Returns `(lexer, depths, stacks, counts, width, deep)`, where `width` is the
/// widest configuration set and `deep` the deepest stack, so the caller uploads
/// what is in use rather than what the ceilings allow.
#[pyfunction]
fn pack_configurations<'py>(
    python: Python<'py>,
    matchers: Vec<PyRef<'py, Matcher>>,
    limit: usize,
) -> PyResult<(
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    usize,
    usize,
)> {
    // Two passes over borrowed state rather than one over a cloned copy. The
    // clone was the cost: at batch 512 with a 520-deep stack it is a megabyte
    // of `Vec<u32>` built and dropped every step, to be read once.
    let mut width = 1usize;
    let mut deep = 1usize;
    for matcher in matchers.iter() {
        width = width.max(matcher.inner.configuration_count());
        deep = deep.max(matcher.inner.max_stack_depth());
    }
    if width > limit {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{width} configurations exceeds the batch's limit of {limit}"
        )));
    }

    let rows = matchers.len();
    let mut lexer = vec![0i32; rows * width];
    let mut depths = vec![1i32; rows * width];
    let mut stacks = vec![0i32; rows * width * deep];
    let mut counts = vec![1i32; rows];
    for (row, matcher) in matchers.iter().enumerate() {
        counts[row] = matcher.inner.configuration_count().max(1) as i32;
        let mut index = 0usize;
        matcher.inner.each_configuration(|state, stack| {
            lexer[row * width + index] = state as i32;
            depths[row * width + index] = stack.len() as i32;
            let at = (row * width + index) * deep;
            for (offset, entry) in stack.iter().enumerate() {
                stacks[at + offset] = *entry as i32;
            }
            index += 1;
        });
    }

    /// The words as bytes, in one copy rather than four per word.
    ///
    /// A `bytearray` rather than `bytes` because the caller hands it to
    /// `torch.frombuffer`, which insists on a writable buffer and made a whole
    /// second copy of every array to get one.
    fn bytes<'py>(python: Python<'py>, values: &[i32]) -> PyResult<Bound<'py, PyByteArray>> {
        #[cfg(target_endian = "little")]
        // Sound: `i32` has no padding and no invalid bit patterns, the slice
        // outlives the call, and `PyByteArray::new` copies before returning.
        let raw = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
        };
        #[cfg(not(target_endian = "little"))]
        let raw = &{
            let mut out = Vec::with_capacity(values.len() * 4);
            for value in values {
                out.extend_from_slice(&value.to_le_bytes());
            }
            out
        }[..];
        Ok(PyByteArray::new(python, raw))
    }
    Ok((
        bytes(python, &lexer)?,
        bytes(python, &depths)?,
        bytes(python, &stacks)?,
        bytes(python, &counts)?,
        width,
        deep,
    ))
}

/// The device runtime, exposed as functions rather than as a class: it holds
/// no state a caller owns - the module is loaded once per process - and the
/// tensors it works on belong to PyTorch.
///
/// Device pointers arrive as integers from `tensor.data_ptr()` and the stream
/// as one from `torch.cuda.current_stream().cuda_stream`. That is the whole
/// interface, and it is what lets a launch be recorded by PyTorch's own graph
/// capture rather than run - which is the property this engine exists for.
#[pyfunction]
fn cuda_available() -> bool {
    engrain_cuda::available()
}

#[pyfunction]
fn cuda_fatbin_bytes() -> usize {
    engrain_cuda::fatbin_bytes()
}

/// Launch one kernel by name. The bring-up path, and the shape every later
/// launch site will take.
#[pyfunction]
#[pyo3(signature = (name, grid, block, stream, pointers, scalars, shared_bytes = 0, grid_y = 1))]
fn cuda_launch(
    name: &str,
    grid: u32,
    block: u32,
    stream: u64,
    pointers: Vec<u64>,
    scalars: Vec<i32>,
    shared_bytes: u32,
    // A copy wants the sequence in a second dimension: one row is up to 19 KiB
    // and a grid of one block per sequence leaves a batch of 32 running 32
    // blocks on 108 multiprocessors.
    grid_y: u32,
) -> PyResult<()> {
    let kernel = engrain_cuda::Kernel::named(name).map_err(PyRuntimeError::new_err)?;
    // The driver takes an array of pointers *to* the arguments, so the values
    // have to outlive the launch call - hence binding them before taking
    // addresses rather than building the array inline.
    let mut addresses: Vec<u64> = pointers;
    let mut values: Vec<i32> = scalars;
    let mut arguments: Vec<*mut std::ffi::c_void> = Vec::with_capacity(
        addresses.len() + values.len(),
    );
    for address in &mut addresses {
        arguments.push(std::ptr::from_mut(address).cast());
    }
    for value in &mut values {
        arguments.push(std::ptr::from_mut(value).cast());
    }
    // Safety: the caller supplies the argument list in the order the named
    // kernel declares. Checked by the tests that launch each kernel, not by
    // the type system - see the note on `Kernel::launch`.
    unsafe {
        kernel
            .launch((grid, grid_y, 1), (block, 1, 1), shared_bytes, stream, &mut arguments)
            .map_err(PyRuntimeError::new_err)
    }
}

/// The compiled front end, imported by the `engrain` package rather than
/// directly: the public library is Python over this, and keeping the extension
/// under a private name is what lets the package own the name users type.
#[pymodule]
fn _engrain(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Compiler>()?;
    module.add_class::<CompiledGrammar>()?;
    module.add_class::<Matcher>()?;
    module.add_function(wrap_pyfunction!(pack_configurations, module)?)?;
    module.add("CompileError", module.py().get_type::<CompileError>())?;
    module.add_function(wrap_pyfunction!(cuda_available, module)?)?;
    module.add_function(wrap_pyfunction!(cuda_fatbin_bytes, module)?)?;
    module.add_function(wrap_pyfunction!(cuda_launch, module)?)?;
    Ok(())
}
