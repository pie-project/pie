//! Python bindings.
//!
//! Exposes exactly what a serving backend needs: compile a grammar against a
//! vocabulary once, then run a cheap matcher per request. The matcher shares
//! the compiled artifact, so admitting a new request costs an integer stack
//! rather than a rebuilt automaton.

use std::sync::Arc;

use gpugrammar_ir::grammar::Grammar;
use gpugrammar_ir::json_schema::{JsonSchemaOptions, json_schema_to_grammar};
use gpugrammar_ir::regex::regex_to_grammar;
use gpugrammar_lex::lexicon::{extract, terminal_automata};
use gpugrammar_lex::regular::analyze;
use gpugrammar_lex::{build_lexer, group_vocabulary};
use gpugrammar_lr::cfg::flatten;
use gpugrammar_lr::tables::build;
use gpugrammar_run::Matcher as RunMatcher;
use gpugrammar_tables::{Artifact, emit};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

    fn compile_json_schema(&self, schema: &str) -> PyResult<CompiledGrammar> {
        let grammar = json_schema_to_grammar(schema, &JsonSchemaOptions::default())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(CompiledGrammar {
            artifact: Arc::new(compile(&self.vocabulary, grammar)?),
        })
    }

    fn compile_ebnf(&self, source: &str, root: &str) -> PyResult<CompiledGrammar> {
        let grammar = Grammar::from_ebnf(source, root)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(CompiledGrammar {
            artifact: Arc::new(compile(&self.vocabulary, grammar)?),
        })
    }

    fn compile_regex(&self, pattern: &str) -> PyResult<CompiledGrammar> {
        let grammar =
            regex_to_grammar(pattern).map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(CompiledGrammar {
            artifact: Arc::new(compile(&self.vocabulary, grammar)?),
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
}

#[pymethods]
impl CompiledGrammar {
    #[pyo3(signature = (max_rollback = 8))]
    fn matcher(&self, max_rollback: usize) -> Matcher {
        Matcher {
            inner: RunMatcher::new(self.artifact.clone(), max_rollback),
            words: self.artifact.bitset_words as usize,
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

    #[getter]
    fn parser_state(&self) -> u32 {
        self.inner.parser_state()
    }

    /// Groups the parser admits right now, for the device path.
    fn admissible_groups(&self) -> Vec<usize> {
        self.inner.admissible_groups()
    }
}

#[pymodule]
fn gpugrammar(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Compiler>()?;
    module.add_class::<CompiledGrammar>()?;
    module.add_class::<Matcher>()?;
    Ok(())
}
