/// A trait for defining stopping conditions during token generation.
pub trait StopCondition {
    /// Checks if the generation should stop based on the sequence of token IDs.
    fn check(&self, token_ids: &[u32]) -> bool;

    /// Combines this condition with another using a logical OR.
    ///
    /// This allows for creating complex conditions by chaining calls.
    /// Example: `max_len(100).or(ends_with(&[50256]))`
    fn or<Other: StopCondition>(self, other: Other) -> Or<Self, Other>
    where
        Self: Sized,
    {
        Or::new(self, other)
    }
}

// --- Concrete Conditions ---

/// Stops generation if the sequence ends with a specific sub-sequence of tokens.
#[derive(Debug, Clone)]
pub struct EndsWith {
    token_ids: Vec<u32>,
}

impl StopCondition for EndsWith {
    fn check(&self, token_ids: &[u32]) -> bool {
        token_ids.ends_with(&self.token_ids)
    }
}

/// Stops generation if the sequence reaches a maximum length.
#[derive(Debug, Clone, Copy)]
pub struct MaxLen {
    max_tokens: usize,
}

impl StopCondition for MaxLen {
    fn check(&self, token_ids: &[u32]) -> bool {
        token_ids.len() >= self.max_tokens
    }
}

// --- Combinators ---

/// A combinator that stops if *any* of its inner conditions are met.
/// This version is specialized for `EndsWith` to handle a dynamic list efficiently.
#[derive(Debug, Clone)]
pub struct AnyEndsWith {
    conditions: Vec<EndsWith>,
}

impl StopCondition for AnyEndsWith {
    fn check(&self, token_ids: &[u32]) -> bool {
        self.conditions.iter().any(|c| c.check(token_ids))
    }
}

/// A generic combinator that stops if either of its two conditions (`A` or `B`) is met.
/// This is the backbone of the `.or()` chaining method.
#[derive(Debug, Clone, Copy)]
pub struct Or<A, B> {
    first: A,
    second: B,
}

impl<A, B> Or<A, B> {
    /// Creates a new `Or` condition.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A, B> StopCondition for Or<A, B>
where
    A: StopCondition,
    B: StopCondition,
{
    fn check(&self, token_ids: &[u32]) -> bool {
        self.first.check(token_ids) || self.second.check(token_ids)
    }
}

// --- Constructor Functions ---

/// Creates a condition that stops when the generated sequence reaches `max_tokens`.
pub fn max_len(max_tokens: usize) -> MaxLen {
    MaxLen { max_tokens }
}

/// Creates a condition that stops if the sequence ends with any of the provided token sequences.
pub fn ends_with_any(stop_sequences: Vec<Vec<u32>>) -> AnyEndsWith {
    let conditions = stop_sequences
        .into_iter()
        .map(|token_ids| EndsWith { token_ids })
        .collect();
    AnyEndsWith { conditions }
}

/// Creates a condition that stops if the sequence ends with a single provided token sequence.
pub fn ends_with(token_ids: Vec<u32>) -> EndsWith {
    EndsWith { token_ids }
}
