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

use gpugrammar_ir::grammar::Grammar;
use gpugrammar_ir::regex::regex_to_grammar;
use gpugrammar_tables::pipeline::{Limits, compile_json_schema as compile_schema};
use gpugrammar_lex::lexicon::{extract, terminal_automata};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{build_lexer, group_vocabulary};
use gpugrammar_lr::cfg::flatten;
use gpugrammar_lr::tables::build;
use gpugrammar_run::Matcher as RunMatcher;
use gpugrammar_tables::{Artifact, emit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use pyo3::types::{PyBytes, PyDict};

fn compile(vocabulary: &[Vec<u8>], grammar: Grammar) -> PyResult<Artifact> {
    let lexicon = extract(&grammar, &analyze(&grammar));
    let lexer = build_lexer(terminal_automata(&grammar, &lexicon));
    let groups = group_vocabulary(&lexer, vocabulary);
    let cfg = flatten(&lexicon);
    let tables = build(&cfg).map_err(|error| PyValueError::new_err(error.to_string()))?;
    emit(&lexicon, &lexer, &groups, &cfg, &tables, vocabulary.len())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))
}

/// A vocabulary-bound compiler. Build one per model.
#[pyclass]
pub struct Compiler {
    vocabulary: Vec<Vec<u8>>,
}

#[pymethods]
impl Compiler {
    #[new]
    fn new(vocabulary: Vec<Vec<u8>>) -> Self {
        Self { vocabulary }
    }

    /// Compile a JSON Schema, searching the lowerings for one that is LALR(1).
    #[pyo3(signature = (schema, lexer_states = None, exact = false))]
    fn compile_json_schema(
        &self,
        python: Python<'_>,
        schema: &str,
        lexer_states: Option<usize>,
        exact: bool,
    ) -> PyResult<CompiledGrammar> {
        let limits = Limits {
            lexer_states: lexer_states.unwrap_or(Limits::default().lexer_states),
            exact,
            ..Default::default()
        };
        // Compiling holds no Python object and takes tens of milliseconds
        // across every core, so holding the interpreter lock for it stops the
        // caller's other threads and makes rayon's workers contend with
        // whatever else the process is doing.
        let compiled = python
            .detach(|| compile_schema(schema, &self.vocabulary, limits))
            .map_err(|failure| PyValueError::new_err(failure.to_string()))?;
        Ok(CompiledGrammar {
            artifact: Arc::new(compiled.artifact),
            precision: format!("{:?}", compiled.precision),
            approximations: compiled.approximations,
        })
    }

    fn compile_ebnf(&self, source: &str, root: &str) -> PyResult<CompiledGrammar> {
        let grammar = Grammar::from_ebnf(source, root)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(CompiledGrammar {
            artifact: Arc::new(compile(&self.vocabulary, grammar)?),
            precision: "n/a".to_string(),
            approximations: Vec::new(),
        })
    }

    fn compile_regex(&self, pattern: &str) -> PyResult<CompiledGrammar> {
        let grammar =
            regex_to_grammar(pattern).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(CompiledGrammar {
            artifact: Arc::new(compile(&self.vocabulary, grammar)?),
            precision: "n/a".to_string(),
            approximations: Vec::new(),
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
    approximations: Vec<String>,
}

#[pymethods]
impl CompiledGrammar {
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
    fn approximations(&self) -> Vec<String> {
        self.approximations.clone()
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
        let mut reading_offsets = vec![0u32];
        let mut reading_index = Vec::new();
        let mut reading_next_state = Vec::new();
        let mut reading_term_offsets = vec![0u32];
        let mut reading_terminals = Vec::new();
        let mut interned_readings: FxHashMap<(Vec<u32>, u32), u32> = FxHashMap::default();
        for group in &artifact.groups {
            group_state.push(group.lexer_state);
            group_set_kind.push(match group.set.kind {
                gpugrammar_tables::SetKind::Sparse => 0u32,
                gpugrammar_tables::SetKind::Complement => 1,
                gpugrammar_tables::SetKind::Dense => 2,
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
                reading_index.push(index);
            }
            reading_offsets.push(reading_index.len() as u32);
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
                gpugrammar_tables::SetKind::Complement => vec![u32::MAX; words],
                _ => vec![0u32; words],
            };
            match set.kind {
                gpugrammar_tables::SetKind::Sparse => {
                    for token in body {
                        bits[*token as usize / 32] |= 1u32 << (*token % 32);
                    }
                }
                gpugrammar_tables::SetKind::Complement => {
                    for token in body {
                        bits[*token as usize / 32] &= !(1u32 << (*token % 32));
                    }
                    let spare = words * 32 - vocab;
                    if spare > 0 {
                        bits[words - 1] &= u32::MAX >> spare;
                    }
                }
                gpugrammar_tables::SetKind::Dense => bits.copy_from_slice(body),
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
    /// Write the allowed-token bitmask into `buffer`, which must hold at least
    /// `bitset_words` 32-bit words.
    fn fill_bitmask(&self, buffer: Bound<'_, PyAny>) -> PyResult<()> {
        let pointer: usize = buffer.getattr("data_ptr")?.call0()?.extract()?;
        let length: usize = buffer.getattr("numel")?.call0()?.extract()?;
        if length < self.words {
            return Err(PyValueError::new_err(format!(
                "buffer holds {length} words but {} are needed",
                self.words
            )));
        }
        // Safety: the caller supplies a contiguous int32 tensor and the length
        // has just been checked against the artifact's word count.
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
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    usize,
    usize,
)> {
    let states: Vec<Vec<(u32, Vec<u32>)>> = matchers
        .iter()
        .map(|matcher| matcher.inner.configurations())
        .collect();
    let mut width = 1usize;
    let mut deep = 1usize;
    for set in &states {
        width = width.max(set.len());
        for (_, stack) in set {
            deep = deep.max(stack.len());
        }
    }
    if width > limit {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{width} configurations exceeds the batch's limit of {limit}"
        )));
    }

    let rows = states.len();
    let mut lexer = vec![0i32; rows * width];
    let mut depths = vec![1i32; rows * width];
    let mut stacks = vec![0i32; rows * width * deep];
    let mut counts = vec![1i32; rows];
    for (row, set) in states.iter().enumerate() {
        counts[row] = set.len().max(1) as i32;
        for (index, (state, stack)) in set.iter().enumerate() {
            lexer[row * width + index] = *state as i32;
            depths[row * width + index] = stack.len() as i32;
            let at = (row * width + index) * deep;
            for (offset, entry) in stack.iter().enumerate() {
                stacks[at + offset] = *entry as i32;
            }
        }
    }

    fn bytes<'py>(python: Python<'py>, values: &[i32]) -> Bound<'py, PyBytes> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        PyBytes::new(python, &out)
    }
    Ok((
        bytes(python, &lexer),
        bytes(python, &depths),
        bytes(python, &stacks),
        bytes(python, &counts),
        width,
        deep,
    ))
}

/// The compiled front end, imported by the `gpugrammar` package rather than
/// directly: the public library is Python over this, and keeping the extension
/// under a private name is what lets the package own the name users type.
#[pymodule]
fn _gpugrammar(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Compiler>()?;
    module.add_class::<CompiledGrammar>()?;
    module.add_class::<Matcher>()?;
    module.add_function(wrap_pyfunction!(pack_configurations, module)?)?;
    Ok(())
}
