pub mod prelude {
    pub use super::{entropy_bound_accept, linear_temperature, stable_and_confident};
    pub use crate::eta::prelude::*;
}

use super::{
    Tensor, and, cast, cumsum, dtype, eq, iota, le, lt, reduce_sum, scatter_set, sort_desc,
};

pub fn linear_temperature(remaining: u32, max_steps: u32, t_max: f32, t_min: f32) -> f32 {
    t_min + (t_max - t_min) * (remaining as f32 / max_steps.max(1) as f32)
}

pub fn entropy_bound_accept(entropy: &Tensor, bound: f32) -> Tensor {
    let n = entropy.shape().dims()[0];
    let (neg_sorted, order) = sort_desc(-entropy);
    let sorted = -&neg_sorted;
    let below = le(&(&cumsum(&sorted) - &sorted), bound);
    let none = lt(iota(n), 0u32);
    scatter_set(&none, &order, &below)
}

pub fn stable_and_confident(
    argmax: &Tensor,
    previous: &Tensor,
    entropy: &Tensor,
    threshold: f32,
) -> Tensor {
    let n = argmax.shape().dims()[0];
    let unchanged = reduce_sum(cast(eq(argmax, previous), dtype::i32));
    let stable = eq(&unchanged, n as i32);
    let mean = &reduce_sum(entropy) / (n as f32);
    and(&stable, lt(&mean, threshold))
}
