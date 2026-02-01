/// A trait for implementing draft token generation in speculative decoding.
///
/// Speculative decoding accelerates text generation by using a small, fast "drafter"
/// model to propose candidate tokens. These draft tokens are then verified in a single,
/// parallel forward pass by the main model, allowing multiple tokens to be generated
/// for the cost of one forward pass when the drafter's predictions are correct.
///
/// The drafter produces a **forest of Tries** (prefix trees) representing multiple
/// speculative token sequences.
///
/// # Workflow
///
/// 1. Before generation starts, `update` is called with the full token history to
///    synchronize the drafter with the main model's state.
/// 2. In each iteration:
///    - `update` is called with the newly accepted tokens from the previous step.
///    - `draft` is called to produce candidate token sequences.
pub trait Drafter {
    /// Updates the drafter's internal state with new context tokens.
    ///
    /// This method is called to keep the drafter synchronized with the main model's
    /// token history. It is invoked:
    /// - Once at the start of generation with the full token history.
    /// - After each verification step with the tokens that were accepted.
    ///
    /// # Arguments
    ///
    /// * `context` - A slice of token IDs to append to the drafter's internal context.
    ///   These are tokens that have been confirmed by the main model.
    fn update(&mut self, context: &[u32]);

    /// Generates a forest of draft token sequences for speculative verification.
    ///
    /// Returns a Trie forest encoded as two parallel vectors in **depth-first search (DFS)
    /// order**. This encoding allows the verifier to efficiently traverse and verify
    /// multiple speculative paths.
    ///
    /// # Returns
    ///
    /// A tuple `(draft_tokens, draft_pos_ids)` where:
    ///
    /// - `draft_tokens`: The proposed token IDs in DFS traversal order.
    /// - `draft_pos_ids`: The depth of each token in the trie, starting at 1 for root nodes.
    ///   These positions are **relative to the last context token**.
    ///
    /// # Invariants
    ///
    /// The returned vectors must satisfy the following invariants:
    ///
    /// 1. **Equal length**: Both vectors must have the same length.
    /// 2. **Valid depths**: All position IDs must >= 1 (depth 1 = immediate continuation
    ///    of the context).
    /// 3. **Valid DFS order**: The sequence must represent a valid DFS traversal of a Trie
    ///    forest. Specifically, for consecutive tokens at indices `i` and `i+1`:
    ///    - If `pos_ids[i+1] == pos_ids[i] + 1`: token `i+1` is a child of token `i`.
    ///    - If `pos_ids[i+1] <= pos_ids[i]`: token `i+1` is a sibling or belongs to a
    ///      different subtree (backtracking in DFS).
    ///
    /// # Example: Trie Forest Structure
    ///
    /// A draft representing multiple alternative continuations with branching:
    ///
    /// ```text
    /// Context: [  ...  tokens  ...  ]
    ///                               |
    ///           +-------------------+--------------+
    ///           |                   |              |
    ///         [A] (1)            [F] (1)        [J] (1)
    ///           |                   |
    ///     +-----+-----+          +--+--+
    ///     |     |     |          |     |
    ///   [B](2) [D](2) [E](2)   [G](2) [I](2)
    ///     |                      |
    ///   [C](3)                 [H](3)
    ///
    /// draft_tokens:  [A, B, C, D, E, F, G, H, I, J]
    /// draft_pos_ids: [1, 2, 3, 2, 2, 1, 2, 3, 2, 1]
    /// ```
    fn draft(&mut self) -> (Vec<u32>, Vec<u32>);
}

pub struct Empty {}

impl Drafter for Empty {
    fn update(&mut self, _context: &[u32]) {}

    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        (vec![], vec![])
    }
}
