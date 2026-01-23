use crate::api;
use crate::brle::Brle;
use crate::{Queue, Resource};
use std::rc::Rc;
use wstd::io::AsyncPollable;

#[derive(Debug, Clone)]
pub struct ForwardPass {
    pub(crate) inner: Rc<api::forward::ForwardPass>,
}

#[derive(Debug, Clone)]
pub struct ForwardPassResult {
    pub distributions: Option<Vec<Distribution>>,
    pub tokens: Option<Vec<u32>>,
}

/// Represents a probability distribution over a set of tokens.
#[derive(Clone, Debug)]
pub struct Distribution {
    /// A vector of token IDs.
    pub ids: Vec<u32>,
    /// A vector of probabilities corresponding to the token IDs.
    pub probs: Vec<f32>,
}

// "Smart" kv page
#[derive(Debug, Clone)]
pub struct KvPage {
    queue: Queue,
    rc: Rc<()>,
    ptr: u32,
}

impl KvPage {
    pub fn new(queue: &Queue, ptr: u32) -> Self {
        KvPage {
            queue: queue.clone(),
            rc: Rc::new(()),
            ptr,
        }
    }

    pub fn ptr(&self) -> u32 {
        self.ptr
    }
}

impl Drop for KvPage {
    fn drop(&mut self) {
        if Rc::strong_count(&self.rc) == 1 {
            self.queue.deallocate_kv_page_ptr(self.ptr);
        }
    }
}

pub fn causal_mask(num_total_tokens: u32, num_input_tokens: u32) -> Vec<Brle> {
    let mut mask = Vec::new();
    let offset = num_total_tokens - num_input_tokens;
    for i in 0..num_input_tokens {
        mask.push(Brle::new((offset + i + 1) as usize));
    }
    mask
}

/// Defines the interface for executing forward passes in a neural network model,
/// including support for PEFT adapters and KV cache manipulation.
pub trait Forward {
    fn new_kv_page(&self) -> KvPage;
    fn new_kv_pages(&self, count: usize) -> Vec<KvPage>;

    fn export_kv_pages(&self, ptrs: &[KvPage], name: &str);
    fn import_kv_pages(&self, name: &str) -> Vec<KvPage>;
    fn allocate_kv_page_ptr(&self) -> u32;
    fn allocate_kv_page_ptrs(&self, count: usize) -> Vec<u32>;
    fn deallocate_kv_page_ptr(&self, ptr: u32);
    fn deallocate_kv_page_ptrs(&self, ptrs: &[u32]);
    fn export_kv_page_ptrs(&self, ptrs: &[u32], name: &str);
    fn import_kv_page_ptrs(&self, name: &str) -> Vec<u32>;

    fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)>;
    fn release_exported_kv_pages(&self, name: &str);

    fn allocate_embed_ptr(&self) -> u32;
    fn allocate_embed_ptrs(&self, count: usize) -> Vec<u32>;
    fn deallocate_embed_ptr(&self, ptr: u32);
    fn deallocate_embed_ptrs(&self, ptrs: &[u32]);
    fn export_embed_ptrs(&self, ptrs: &[u32], name: &str);
    fn import_embed_ptrs(&self, name: &str) -> Vec<u32>;
    fn get_all_exported_embeds(&self) -> Vec<(String, u32)>;
    fn release_exported_embeds(&self, name: &str);

    fn create_forward_pass(&self) -> ForwardPass;
}

impl Forward for Queue {
    fn new_kv_page(&self) -> KvPage {
        let ptr = self.allocate_kv_page_ptr();
        KvPage::new(self, ptr)
    }

    fn new_kv_pages(&self, count: usize) -> Vec<KvPage> {
        self.allocate_kv_page_ptrs(count)
            .into_iter()
            .map(|ptr| KvPage::new(self, ptr))
            .collect()
    }

    fn export_kv_pages(&self, kv_pages: &[KvPage], name: &str) {
        let ptrs = kv_pages.iter().map(|kv| kv.ptr()).collect::<Vec<_>>();
        self.export_resource(Resource::KvPage, &ptrs, name)
    }

    fn import_kv_pages(&self, name: &str) -> Vec<KvPage> {
        let ptrs = self.import_resource(Resource::KvPage, name);
        ptrs.into_iter().map(|ptr| KvPage::new(self, ptr)).collect()
    }

    fn allocate_kv_page_ptr(&self) -> u32 {
        self.allocate_resources(Resource::KvPage, 1)
            .into_iter()
            .next()
            .unwrap()
    }

    fn allocate_kv_page_ptrs(&self, count: usize) -> Vec<u32> {
        self.allocate_resources(Resource::KvPage, count as u32)
    }

    fn deallocate_kv_page_ptr(&self, ptr: u32) {
        self.deallocate_resources(Resource::KvPage, &[ptr])
    }

    fn deallocate_kv_page_ptrs(&self, ptrs: &[u32]) {
        self.deallocate_resources(Resource::KvPage, ptrs)
    }

    fn export_kv_page_ptrs(&self, ptrs: &[u32], name: &str) {
        self.export_resource(Resource::KvPage, ptrs, name)
    }

    fn import_kv_page_ptrs(&self, name: &str) -> Vec<u32> {
        self.import_resource(Resource::KvPage, name)
    }

    fn get_all_exported_kv_pages(&self) -> Vec<(String, u32)> {
        self.get_all_exported_resources(Resource::KvPage)
    }

    fn release_exported_kv_pages(&self, name: &str) {
        self.release_exported_resources(Resource::KvPage, name)
    }

    fn allocate_embed_ptr(&self) -> u32 {
        self.allocate_resources(Resource::Embed, 1)
            .into_iter()
            .next()
            .unwrap()
    }

    fn allocate_embed_ptrs(&self, count: usize) -> Vec<u32> {
        self.allocate_resources(Resource::Embed, count as u32)
    }

    fn deallocate_embed_ptr(&self, ptr: u32) {
        self.deallocate_resources(Resource::Embed, &[ptr])
    }

    fn deallocate_embed_ptrs(&self, ptrs: &[u32]) {
        self.deallocate_resources(Resource::Embed, ptrs)
    }

    fn export_embed_ptrs(&self, ptrs: &[u32], name: &str) {
        self.export_resource(Resource::Embed, ptrs, name)
    }

    fn import_embed_ptrs(&self, name: &str) -> Vec<u32> {
        self.import_resource(Resource::Embed, name)
    }

    fn get_all_exported_embeds(&self) -> Vec<(String, u32)> {
        self.get_all_exported_resources(Resource::Embed)
    }

    fn release_exported_embeds(&self, name: &str) {
        self.release_exported_resources(Resource::Embed, name)
    }

    fn create_forward_pass(&self) -> ForwardPass {
        ForwardPass {
            inner: Rc::new(api::forward::create_forward_pass(&self.inner)),
        }
    }
}

impl ForwardPass {
    pub async fn execute(&self) -> ForwardPassResult {
        if let Some(future) = self.inner.execute() {
            let pollable = future.pollable();
            AsyncPollable::new(pollable).wait_for().await;

            let mut dists = Vec::new();
            if let Some(distributions) = future.get_distributions() {
                for (ids, probs) in distributions {
                    dists.push(Distribution { ids, probs });
                }
            }
            let distributions = if dists.is_empty() { None } else { Some(dists) };

            ForwardPassResult {
                distributions,
                tokens: future.get_tokens(),
            }
        } else {
            ForwardPassResult {
                distributions: None,
                tokens: None,
            }
        }
    }

    pub fn input_embed_ptrs(&self, embed_ptrs: &[u32], positions: &[u32]) {
        api::forward::input_embeddings(&self.inner, embed_ptrs, positions);
    }

    pub fn input_tokens(&self, input_tokens: &[u32], positions: &[u32]) {
        api::forward::input_tokens(&self.inner, input_tokens, positions);
    }

    pub fn output_embed_ptrs(&self, embed_ptrs: &[u32], indices: &[u32]) {
        api::forward::output_embeddings(&self.inner, embed_ptrs, indices);
    }

    pub fn output_distributions(&self, indices: &[u32], temperature: f32, top_k: Option<u32>) {
        api::forward::output_distributions(&self.inner, indices, temperature, top_k);
    }

    pub fn output_tokens(&self, indices: &[u32], temperature: f32) {
        api::forward::output_tokens(&self.inner, indices, temperature);
    }

    pub fn output_tokens_top_p(&self, indices: &[u32], temperature: f32, top_p: f32) {
        api::forward::output_tokens_top_p(&self.inner, indices, temperature, top_p);
    }

    pub fn output_tokens_top_k(&self, indices: &[u32], temperature: f32, top_k: u32) {
        api::forward::output_tokens_top_k(&self.inner, indices, temperature, top_k);
    }

    pub fn output_tokens_min_p(&self, indices: &[u32], temperature: f32, min_p: f32) {
        api::forward::output_tokens_min_p(&self.inner, indices, temperature, min_p);
    }

    pub fn output_tokens_top_k_top_p(
        &self,
        indices: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) {
        api::forward::output_tokens_top_k_top_p(&self.inner, indices, temperature, top_k, top_p);
    }

    pub fn attention_mask(&self, mask: &[Vec<u32>]) {
        api::forward::attention_mask(&self.inner, mask);
    }

    pub fn sampling_mask(&self, mask: &[u32]) {
        api::forward::sampling_mask(&self.inner, mask);
    }

    pub fn kv_cache(&self, kv_pages: &[KvPage], last_kv_page_len: usize) {
        let ptrs = kv_pages.iter().map(|kv| kv.ptr()).collect::<Vec<_>>();
        api::forward::kv_cache(&self.inner, &ptrs, last_kv_page_len as u32);
    }

    pub fn kv_cache_ptrs(&self, kv_page_ptrs: &[u32], last_kv_page_len: usize) {
        api::forward::kv_cache(&self.inner, kv_page_ptrs, last_kv_page_len as u32);
    }
}
