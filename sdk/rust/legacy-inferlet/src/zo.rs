use crate::forward::ForwardPass;
use crate::{Queue, api};

pub trait Evolve {
    fn initialize_adapter(
        &self,
        adapter_ptr: u32,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    );
    fn update_adapter(&self, adapter_ptr: u32, scores: Vec<f32>, seeds: Vec<i64>, max_sigma: f32);
}

pub trait SetAdapterSeed {
    fn set_adapter_seed(&self, seed: i64);
}

impl Evolve for Queue {
    fn initialize_adapter(
        &self,
        adapter_ptr: u32,
        rank: u32,
        alpha: f32,
        population_size: u32,
        mu_fraction: f32,
        initial_sigma: f32,
    ) {
        api::zo::evolve::initialize_adapter(
            &self.inner,
            adapter_ptr,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        )
    }

    fn update_adapter(&self, adapter_ptr: u32, scores: Vec<f32>, seeds: Vec<i64>, max_sigma: f32) {
        api::zo::evolve::update_adapter(&self.inner, adapter_ptr, &scores, &seeds, max_sigma)
    }
}

impl SetAdapterSeed for ForwardPass {
    fn set_adapter_seed(&self, seed: i64) {
        api::zo::evolve::set_adapter_seed(&self.inner, seed)
    }
}
