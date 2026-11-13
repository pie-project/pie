//! How a load holds its weights. Every dense plane lives in device memory
//! for the load's life; the weights a fire reads a few atoms of at a time
//! (routed experts, token-id-gathered table rows) are the weight cache's
//! (`store::streamed`), which the runtime holds or streams through pool
//! slots.

/// A load's residency choices: the weight cache's knobs a test or the
/// runtime states.
#[derive(Debug, Clone, Default)]
pub struct Plan {
    /// The weight cache states no weight shape: every table and bank is
    /// read whole, as before the cache (a diagnostic, for comparing).
    whole_weights: bool,
    /// The chunk the pool's recipe aims near, where not the on-device
    /// Metal profile's (a diagnostic: records cut into pieces).
    chunk_bytes: Option<u64>,
    /// The runtime holds the weight cache's atoms for the load's life
    /// (`engine::load::Residency::weights_held`): the load seats them in
    /// pool slots itself and states where (`WeightShape::seated`).
    weights_held: bool,
}

impl Plan {
    /// This plan stating no weight shape to the runtime (see
    /// `whole_weights`).
    #[must_use]
    pub fn with_whole_weights(mut self) -> Plan {
        self.whole_weights = true;
        self
    }

    #[must_use]
    pub fn whole_weights(&self) -> bool {
        self.whole_weights
    }

    /// This plan told the runtime holds the weights (see `weights_held`).
    #[must_use]
    pub fn with_weights_held(mut self, held: bool) -> Plan {
        self.weights_held = held;
        self
    }

    #[must_use]
    pub fn weights_held(&self) -> bool {
        self.weights_held
    }

    /// Whether the token-id tables (PLE rows) go through the weight cache:
    /// only where the runtime may stream them. A held table is read in
    /// place: memory is unified, so the table's own pages (the mapped
    /// serving file, or its allocation on a cold load) are device memory
    /// already, and a pool copy of it would cost its bytes twice and its
    /// seating at load. The table is then a dense weight, counted with
    /// the weights, outside the pool (see [`rows_streamed`]).
    #[must_use]
    pub fn rows_streamed(&self) -> bool {
        rows_streamed(self.weights_held)
    }

    /// This plan's pool aiming its chunk near `bytes` (see `chunk_bytes`).
    #[must_use]
    pub fn with_chunk_bytes(mut self, bytes: u64) -> Plan {
        self.chunk_bytes = Some(bytes);
        self
    }

    #[must_use]
    pub fn chunk_bytes(&self) -> Option<u64> {
        self.chunk_bytes
    }
}

/// Whether a load whose runtime holds the weights (`held`) states its
/// token-id tables as weight shapes ([`Plan::rows_streamed`]).
#[must_use]
pub fn rows_streamed(held: bool) -> bool {
    !held
}
