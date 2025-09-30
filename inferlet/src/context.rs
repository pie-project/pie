use crate::adapter::SetAdapter;
use crate::brle::Brle;
use crate::drafter::Drafter;
use crate::forward::{Distribution, Forward, KvPage};
use crate::sampler::Sample;
use crate::stop_condition::StopCondition;
use crate::zo::SetAdapterSeed;
use crate::{ChatFormatter, Model, Queue, Sampler, Tokenizer};
use futures::future::join_all;
use std::cmp::Ordering;
use std::mem;

#[derive(Debug)]
pub struct Context {
    pub queue: Queue,
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub formatter: ChatFormatter,

    pub token_ids: Vec<u32>,
    pub token_ids_pending: Vec<u32>,

    pub token_mask_pending: Vec<Brle>,
    pub token_mask_current: Brle,

    pub position_ids: Vec<u32>,

    pub kv_pages: Vec<KvPage>,
    pub kv_page_last_len: usize,
    pub kv_page_size: usize,

    pub adapter_ptr: Option<u32>,
    pub adapter_random_seed: Option<i64>,

    pub begin_of_sequence: bool,
}

impl Context {
    pub fn new(model: &Model) -> Self {
        let queue = model.create_queue();
        let kv_page_size = model.get_kv_page_size() as usize;
        let tokenizer = model.get_tokenizer();
        Context {
            queue,
            model: model.clone(),
            tokenizer,
            formatter: ChatFormatter::new(),
            token_ids: Vec::new(),
            token_ids_pending: Vec::new(),
            token_mask_pending: Vec::new(),
            token_mask_current: Brle::new(0),
            position_ids: Vec::new(),
            kv_pages: Vec::new(),
            kv_page_last_len: 0,
            kv_page_size,
            adapter_ptr: None,
            adapter_random_seed: None,
            begin_of_sequence: true,
        }
    }

    pub fn set_adapter(&mut self, adapter_ptr: u32) {
        self.adapter_ptr = Some(adapter_ptr);
    }

    pub fn remove_adapter(&mut self) {
        self.adapter_ptr = None;
    }

    pub fn set_adapter_random_seed(&mut self, seed: i64) {
        self.adapter_random_seed = Some(seed);
    }

    /// Creates a new Context from previously exported and now imported KV pages.
    /// This is used to restore a context's state from a cache.
    pub fn from_imported_state(
        model: &Model,
        kv_pages: Vec<KvPage>,
        prefix_tokens: Vec<u32>,
        kv_page_last_len: usize,
    ) -> Self {
        let queue = model.create_queue();
        let kv_page_size = model.get_kv_page_size() as usize;
        let tokenizer = model.get_tokenizer();

        assert_eq!(
            prefix_tokens.len(),
            (kv_pages.len() - 1) * kv_page_size + kv_page_last_len,
        );

        let num_tokens = prefix_tokens.len();
        // The new context takes ownership of the imported pages.
        // It's assumed the state in these pages corresponds exactly
        // to the provided prefix_tokens and kv_page_last_len.
        Context {
            queue,
            model: model.clone(),
            tokenizer,
            formatter: ChatFormatter::new(),
            token_ids: prefix_tokens,
            token_ids_pending: Vec::new(),
            token_mask_pending: Vec::new(),
            token_mask_current: Brle::new(num_tokens),
            position_ids: (0..num_tokens as u32).collect(),
            kv_pages,
            kv_page_last_len,
            kv_page_size,
            adapter_ptr: None,
            adapter_random_seed: None,
            begin_of_sequence: false,
        }
    }

    pub fn model(&self) -> &Model {
        &self.model
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn get_token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn get_text(&self) -> String {
        self.tokenizer.detokenize(&self.token_ids)
    }

    /// Returns the unique IDs of the KV cache pages currently in use.
    pub fn get_kv_page_ptrs(&self) -> Vec<u32> {
        self.kv_pages.iter().map(|p| p.ptr()).collect()
    }

    /// Returns the number of tokens stored in the last KV cache page.
    pub fn get_kv_page_last_len(&self) -> usize {
        self.kv_page_last_len
    }

    /// Creates a safe, copy-on-write fork of the context.
    ///
    /// This method creates a new context that shares the immutable history of the current
    /// one. If the last KV-cache page is not full, its tokens are moved to the
    /// `token_ids_pending` buffer of the new context to be recomputed, ensuring state isolation.
    ///
    /// This function will flush any pending tokens in the current context before forking.
    pub fn fork(&self) -> Self {
        let (
            new_tokens,
            new_pending,
            new_kv_page_ptrs,
            new_kv_page_last_len,
            new_pos_ids,
            new_mask_pending,
        ) = if self.kv_page_last_len == self.kv_page_size {
            // Easy case: the last page is full, we can share everything.
            (
                self.token_ids.clone(),
                self.token_ids_pending.clone(),
                self.kv_pages.clone(),
                self.kv_page_last_len,
                self.position_ids.clone(), // Clone position_ids
                self.token_mask_pending.clone(),
            )
        } else {
            // Hard case: the last page is partially full and must be recomputed.
            let kept_kv_page_len = self.kv_pages.len().saturating_sub(1);
            let kept_tokens_len = kept_kv_page_len * self.kv_page_size;

            let forked_token_ids = self.token_ids[..kept_tokens_len].to_vec();
            let forked_kv_page_ptrs = self.kv_pages[..kept_kv_page_len].to_vec();
            let forked_pos_ids = self.position_ids[..kept_tokens_len].to_vec();

            let forked_pending_token_ids = [
                &self.token_ids[kept_tokens_len..],
                &self.token_ids_pending[..],
            ]
            .concat();

            let forked_last_kv_page_len = if !forked_kv_page_ptrs.is_empty() {
                self.kv_page_size
            } else {
                0
            };

            let mut mask_builder = self.token_mask_current.clone();
            let parent_total_mask_len = self.token_ids.len() + self.token_ids_pending.len();
            mask_builder.remove_range(kept_tokens_len, parent_total_mask_len);

            // 2. Iteratively build the pending masks, appending `false` for each new
            // pending token, which mimics the `fill_tokens` behavior.
            let mut forked_mask_pending = Vec::with_capacity(forked_pending_token_ids.len());
            for _ in 0..forked_pending_token_ids.len() {
                mask_builder.append(false);
                forked_mask_pending.push(mask_builder.clone());
            }

            (
                forked_token_ids,
                forked_pending_token_ids,
                forked_kv_page_ptrs,
                forked_last_kv_page_len,
                forked_pos_ids,
                forked_mask_pending,
            )
        };

        Context {
            queue: self.model.create_queue(),
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            formatter: self.formatter.clone(),
            token_ids: new_tokens,
            token_ids_pending: new_pending,
            token_mask_pending: new_mask_pending,
            token_mask_current: self.token_mask_current.clone(),
            position_ids: new_pos_ids,
            kv_pages: new_kv_page_ptrs,
            kv_page_last_len: new_kv_page_last_len,
            kv_page_size: self.kv_page_size,
            adapter_ptr: self.adapter_ptr,
            adapter_random_seed: self.adapter_random_seed,
            begin_of_sequence: self.begin_of_sequence,
        }
    }

    pub fn fill(&mut self, text: &str) {
        let new_token_ids = self.tokenizer.tokenize(text);
        self.fill_tokens(new_token_ids);
    }

    pub fn fill_tokens(&mut self, new_token_ids: Vec<u32>) {
        let n = new_token_ids.len();
        self.token_ids_pending.extend(new_token_ids);

        for _ in 0..n {
            // always fill with false - we don't mask tokens to mask itself.
            self.token_mask_current.append(false);
            self.token_mask_pending
                .push(self.token_mask_current.clone())
        }
        self.begin_of_sequence = false;
    }

    pub fn fill_token(&mut self, new_token_id: u32) {
        self.token_ids_pending.push(new_token_id);
        self.token_mask_current.append(false);
        self.token_mask_pending
            .push(self.token_mask_current.clone());
        self.begin_of_sequence = false;
    }

    pub fn fill_system(&mut self, text: &str) {
        self.formatter.system(text);
        self.flush_chat_messages2(false);
    }

    pub fn fill_user(&mut self, text: &str) {
        self.formatter.user(text);
        self.flush_chat_messages2(true);
    }

    pub fn fill_user_only(&mut self, text: &str) {
        self.formatter.user(text);
        self.flush_chat_messages2(false);
    }

    pub fn fill_assistant(&mut self, text: &str) {
        self.formatter.assistant(text);
        self.flush_chat_messages2(false);
    }

    pub fn mask_tokens(&mut self, indices: &[usize], mask: bool) {
        self.token_mask_current.mask(indices, mask)
    }

    pub fn mask_token_range(&mut self, start: usize, end: usize, mask: bool) {
        self.token_mask_current.mask_range(start, end, mask)
    }

    pub fn mask_token(&mut self, index: usize, mask: bool) {
        self.token_mask_current.mask(&[index], mask)
    }

    /// Drops fully masked KV pages to save memory, supporting non-contiguous
    /// dropping for optimizations like attention sink.
    ///
    /// The function iterates through all committed pages and checks if the tokens
    /// corresponding to a page are all masked as `true`. If so, it deallocates
    /// the page and removes the corresponding token ranges from the context's state.
    ///
    /// # Warning
    ///
    /// This operation modifies the token history non-contiguously, which can
    /// break the assumptions of a standard causal attention model. It should
    /// only be used with models and systems (like StreamingLLM) designed to
    /// handle a KV cache with logical gaps.
    pub fn drop_masked_kv_pages(&mut self) {
        let num_committed_pages = self.token_ids.len() / self.kv_page_size;

        // Iterate backwards to safely remove elements from vectors by index.
        // We only consider dropping full pages, not the last (potentially partial) page.
        for i in (0..num_committed_pages).rev() {
            let page_start_token_idx = i * self.kv_page_size;
            let page_end_token_idx = (i + 1) * self.kv_page_size;

            if self.token_mask_current.is_range_all_value(
                page_start_token_idx,
                page_end_token_idx,
                true,
            ) {
                // This page is fully masked and can be dropped.

                // 1. Remove the page ID and deallocate the physical page.
                self.kv_pages.remove(i);

                // 2. Remove the corresponding token range from the main token list.
                self.token_ids
                    .drain(page_start_token_idx..page_end_token_idx);

                self.position_ids
                    .drain(page_start_token_idx..page_end_token_idx);

                // 3. Remove the same range from the current mask.
                self.token_mask_current
                    .remove_range(page_start_token_idx, page_end_token_idx);

                // 4. Remove the range from all historical pending masks.
                for mask in &mut self.token_mask_pending {
                    mask.remove_range(page_start_token_idx, page_end_token_idx);
                }
            }
        }

        // After removing tokens, the total count has changed, so we must
        // recalculate the number of tokens stored in the last page.
        let new_total_tokens = self.token_ids.len();
        let last_page_len = new_total_tokens % self.kv_page_size;

        self.kv_page_last_len = if last_page_len == 0 && new_total_tokens > 0 {
            self.kv_page_size
        } else {
            last_page_len
        };
    }

    /// Adjusts the number of KV pages to match the required number of tokens.
    ///
    /// This function handles both allocating new pages (growing) and deallocating
    /// unused pages (shrinking).
    ///
    /// # Arguments
    ///
    /// * `num_tokens`: The number of tokens to add or remove. A positive value
    ///   grows the KV cache, while a negative value shrinks it.
    fn adjust_kv_pages(&mut self, num_tokens: isize) {
        if num_tokens == 0 {
            return;
        }

        let current_tokens = if self.kv_pages.is_empty() {
            self.kv_page_last_len
        } else {
            (self.kv_pages.len() - 1) * self.kv_page_size + self.kv_page_last_len
        };

        // Safely calculate the new total number of tokens after the adjustment.
        let new_total_tokens = match current_tokens.checked_add_signed(num_tokens) {
            Some(n) => n,
            None => panic!("Token count adjustment resulted in underflow"),
        };

        let current_pages = self.kv_pages.len();
        let required_pages = new_total_tokens.div_ceil(self.kv_page_size);

        match required_pages.cmp(&current_pages) {
            Ordering::Greater => {
                // Grow: Allocate new pages if more are needed.
                let new_pages_needed = required_pages - current_pages;
                let new_kv_page_ids = self.queue.new_kv_pages(new_pages_needed);
                self.kv_pages.extend(new_kv_page_ids);
            }
            Ordering::Less => {
                // Shrink: Deallocate pages that are no longer needed.
                let _ = self.kv_pages.split_off(required_pages);
            }
            Ordering::Equal => {
                // No change in the number of pages is required.
            }
        }

        // Finally, update the length of the last page based on the new total.
        let last_page_len = new_total_tokens % self.kv_page_size;
        self.kv_page_last_len = if last_page_len == 0 && new_total_tokens > 0 {
            self.kv_page_size
        } else {
            last_page_len
        };
    }

    pub fn grow_kv_pages(&mut self, num_tokens: usize) {
        self.adjust_kv_pages(num_tokens as isize);
    }

    pub fn shrink_kv_pages(&mut self, num_tokens: usize) {
        // Convert the number of tokens to a negative adjustment for shrinking.
        self.adjust_kv_pages(-(num_tokens as isize));
    }

    fn flush_chat_messages2(&mut self, add_generation_prompt: bool) {
        if self.formatter.has_messages() {
            let p = self.formatter.render(
                &self.model.get_prompt_template(),
                add_generation_prompt,
                self.begin_of_sequence,
            );
            self.begin_of_sequence = false;
            self.formatter.clear();
            self.fill(&p);
        }
    }

    /// Processes a batch of pending tokens to update the model's internal state.
    pub async fn flush(&mut self) {
        if self.token_ids_pending.is_empty() {
            return;
        }
        let process_count = self.token_ids_pending.len();

        // Process all but the last pending token, leaving it for the next generation step.
        let pending_token_ids = self
            .token_ids_pending
            .drain(..process_count)
            .collect::<Vec<u32>>();

        let mask = self
            .token_mask_pending
            .drain(..process_count)
            .map(|b| b.buffer)
            .collect::<Vec<Vec<u32>>>();

        let last_pos = self.position_ids.last().map(|&p| p + 1).unwrap_or(0);
        let position_ids =
            (last_pos..(last_pos + pending_token_ids.len() as u32)).collect::<Vec<u32>>();

        self.grow_kv_pages(pending_token_ids.len());

        // println!("pending token ids: {:?}", &pending_token_ids);
        // println!("mask: {:?}", &mask);
        // println!("position ids: {:?}", &position_ids);

        let p = self.queue.create_forward_pass();
        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&self.kv_pages, self.kv_page_last_len);
        p.attention_mask(&mask);

        let _ = p.execute().await;

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(&position_ids);
        // self.queue.deallocate_embeds(&embed_ids);
    }

    /// Performs a single, atomic autoregressive decoding step.
    ///
    /// This function is the core of the generation process. It takes the last token
    /// from the pending buffer (`token_ids_pending`), runs a forward pass through the model,
    /// and returns the resulting probability distribution for the next token.
    ///
    /// This operation is stateful and modifies the context:
    /// 1.  The pending token is consumed and moved to the main `token_ids` history.
    /// 2.  The model's internal state (KV cache) is updated to reflect the new token.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `Distribution` over the next possible tokens,
    /// or an error if the generation step could not be performed.
    pub async fn decode_step(&mut self, sampler: &Sampler) -> u32 {
        assert!(
            !self.token_ids_pending.is_empty(),
            "Must have at least one seed token"
        );

        let pending_token_ids = mem::take(&mut self.token_ids_pending);
        let last_pos_id = self.position_ids.last().map(|&p| p + 1).unwrap_or(0);
        let position_ids =
            (last_pos_id..(last_pos_id + pending_token_ids.len() as u32)).collect::<Vec<u32>>();

        self.grow_kv_pages(pending_token_ids.len());

        // println!("next token id: {}", next_token_id);
        // println!("next pos id: {}", next_pos_id);
        // println!("kv page last len: {}", self.kv_page_last_len);
        // println!("kv page ids: {:?}", &self.kv_page_ids);
        // println!("token ids: {:?}", &self.token_ids);
        // println!("token ids pending: {:?}", &self.token_ids_pending);

        let mask = mem::take(&mut self.token_mask_pending)
            .into_iter()
            .map(|brie| brie.buffer)
            .collect::<Vec<Vec<u32>>>();

        let p = self.queue.create_forward_pass();

        if let Some(adapter_ptr) = self.adapter_ptr {
            p.set_adapter(adapter_ptr);

            if let Some(adapter_random_seed) = self.adapter_random_seed {
                p.set_adapter_seed(adapter_random_seed);
            }
        }

        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&self.kv_pages, self.kv_page_last_len);
        p.attention_mask(&mask);

        let output_idx = pending_token_ids.len() as u32 - 1;
        match sampler {
            Sampler::Custom {
                temperature,
                sampler: _sampler,
            } => {
                p.output_distributions(&[output_idx], *temperature, None);
            }
            Sampler::Multinomial { temperature } => {
                p.output_tokens(&[output_idx], *temperature);
            }
            Sampler::TopP { temperature, top_p } => {
                p.output_tokens_top_p(&[output_idx], *temperature, *top_p);
            }
            Sampler::TopK { temperature, top_k } => {
                p.output_tokens_top_k(&[output_idx], *temperature, *top_k);
            }
            Sampler::MinP { temperature, min_p } => {
                p.output_tokens_min_p(&[output_idx], *temperature, *min_p);
            }
            Sampler::TopKTopP {
                temperature,
                top_k,
                top_p,
            } => {
                p.output_tokens_top_k_top_p(&[output_idx], *temperature, *top_k, *top_p);
            }
        }

        let res = p.execute().await;

        let sampled = match sampler {
            Sampler::Custom {
                temperature: _temperature,
                sampler,
            } => {
                let dist = res.distributions.unwrap().into_iter().next().unwrap();
                let sampled = sampler.sample(&dist.ids, &dist.probs);
                sampled
            }
            _ => {
                let sampled = res.tokens.unwrap().into_iter().next().unwrap();
                sampled
            }
        };

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(position_ids);

        sampled
    }

    /// Performs a single, atomic autoregressive decoding step.
    ///
    /// This function is the core of the generation process. It takes the last token
    /// from the pending buffer (`token_ids_pending`), runs a forward pass through the model,
    /// and returns the resulting probability distribution for the next token.
    ///
    /// This operation is stateful and modifies the context:
    /// 1.  The pending token is consumed and moved to the main `token_ids` history.
    /// 2.  The model's internal state (KV cache) is updated to reflect the new token.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `Distribution` over the next possible tokens,
    /// or an error if the generation step could not be performed.
    pub async fn decode_step_dist(&mut self) -> Distribution {
        assert!(
            !self.token_ids_pending.is_empty(),
            "Must have at least one seed token"
        );

        let pending_token_ids = mem::take(&mut self.token_ids_pending);
        let last_pos_id = self.position_ids.last().map(|&p| p + 1).unwrap_or(0);
        let position_ids =
            (last_pos_id..(last_pos_id + pending_token_ids.len() as u32)).collect::<Vec<u32>>();

        self.grow_kv_pages(pending_token_ids.len());

        // println!("next token id: {}", next_token_id);
        // println!("next pos id: {}", next_pos_id);
        // println!("kv page last len: {}", self.kv_page_last_len);
        // println!("kv page ids: {:?}", &self.kv_page_ids);
        // println!("token ids: {:?}", &self.token_ids);
        // println!("token ids pending: {:?}", &self.token_ids_pending);

        let mask = mem::take(&mut self.token_mask_pending)
            .into_iter()
            .map(|brie| brie.buffer)
            .collect::<Vec<Vec<u32>>>();

        let p = self.queue.create_forward_pass();

        if let Some(adapter_ptr) = self.adapter_ptr {
            p.set_adapter(adapter_ptr);

            if let Some(adapter_random_seed) = self.adapter_random_seed {
                p.set_adapter_seed(adapter_random_seed);
            }
        }

        p.input_tokens(&pending_token_ids, &position_ids);
        p.kv_cache(&self.kv_pages, self.kv_page_last_len);
        p.attention_mask(&mask);

        let output_idx = pending_token_ids.len() as u32 - 1;
        p.output_distributions(&[output_idx], 1.0, None);

        let res = p.execute().await;

        let dist = res.distributions.unwrap().into_iter().next().unwrap();

        self.token_ids.extend(pending_token_ids);
        self.position_ids.extend(position_ids);

        dist
    }

    /// Generates text autoregressively until a stop condition is met.
    ///
    /// This function drives the text generation loop. In each iteration, it calls
    /// `decode_step()` to get a probability distribution, uses the provided `sampler`
    /// to choose the next token, and adds it to the context. The loop continues
    /// until the `stop_condition` signals that generation should end.
    ///
    /// # Arguments
    ///
    /// * `sampler`: A mutable reference to a struct implementing the `Sampler` trait,
    ///   which will be used to sample a token from the model's output distribution at each step.
    /// * `stop_condition`: A mutable reference to a struct implementing the `StopCondition`
    ///   trait, which determines when to halt the generation process.
    ///
    /// # Returns
    ///
    /// A `Result` containing the generated `String` upon successful completion.
    pub async fn generate<S: StopCondition>(
        &mut self,
        sampler: Sampler,
        stop_condition: S,
    ) -> String {
        let mut generated_token_ids = Vec::new();

        // The autoregressive generation loop
        loop {
            // start time
            //let start_time = Instant::now();
            let next_token_id = self.decode_step(&sampler).await;

            self.fill_token(next_token_id);

            generated_token_ids.push(next_token_id);

            if stop_condition.check(&generated_token_ids) {
                break;
            }
        }

        self.tokenizer.detokenize(&generated_token_ids)
    }

    /// Generates text using beam search decoding until a stop condition is met.
    ///
    /// Beam search is an autoregressive decoding algorithm that explores multiple
    /// potential sequences (hypotheses) simultaneously. At each step, it maintains
    /// a set of `beam_size` candidate sequences, called "beams."
    ///
    /// # Algorithm
    /// 1.  **Expansion**: For each existing beam, the model predicts the probability
    ///     distribution for the next token. The top `beam_size` most likely next
    ///     tokens are used to create new, longer candidate beams.
    /// 2.  **Scoring**: Each new beam's score is calculated by adding the
    ///     log probability of the new token to the score of its parent beam.
    /// 3.  **Pruning**: All new candidate beams from all parents are gathered,
    ///     sorted by their score in descending order, and only the top `beam_size`
    ///     are kept for the next generation step.
    /// 4.  **Termination**: The generation loop continues until the highest-scoring
    ///     beam in the current set satisfies the `stop_condition`. The text from
    ///     this winning beam is then returned. Note that this heuristic stops at the
    ///     first valid complete beam and may not be the global highest-scoring sequence.
    ///
    /// # Side Effects
    /// Upon completion, this method **modifies the `Context` (`self`)** to adopt the
    /// full state (token history and KV cache) of the winning beam.
    ///
    /// # Arguments
    ///
    /// * `stop_condition`: A mutable reference to a struct implementing the
    ///   `StopCondition` trait. The generation process will halt once the condition
    ///   is met by the highest-scoring beam.
    /// * `beam_size`: The number of candidate sequences to maintain at each step.
    ///   A larger `beam_size` increases the search space and potential quality at the
    ///   cost of computational resources.
    ///
    /// # Returns
    ///
    /// A `String` containing the generated text from the winning beam.
    pub async fn generate_with_beam<C: StopCondition>(
        &mut self,
        stop_condition: &mut C,
        beam_size: usize,
    ) -> String {
        let mut beams = Vec::new();
        // The score is the cumulative probability, starting at 1.0.
        beams.push((self.fork(), vec![], 0.0f32));

        loop {
            // Beams are sorted by score, so the first match is the best valid one found so far.
            if let Some((beam, generated_tokens, _)) =
                beams.iter().find(|(_, g, _)| stop_condition.check(g))
            {
                // Deallocate the pages previously held by `self`.
                let _ = mem::take(&mut self.kv_pages);

                // Adopt the state from the winning beam.
                self.kv_page_last_len = beam.kv_page_last_len;
                self.token_ids = beam.token_ids.clone();
                self.token_ids_pending = beam.token_ids_pending.clone();
                self.kv_pages = beam.kv_pages.clone();

                // Increment the ref count for the newly adopted pages, as `self` is a new owner.
                //self.queue.increase_ref_count(&self.kv_page_ptrs);

                return self.tokenizer.detokenize(generated_tokens);
            }

            // Progress the beams in parallel.
            let mut next_dist_futures = Vec::with_capacity(beams.len());
            for (beam, _, _) in beams.iter_mut() {
                let next_dist = beam.decode_step_dist();
                next_dist_futures.push(next_dist);
            }

            // Wait for all forward passes to complete.
            let next_dists = join_all(next_dist_futures).await;

            let mut next_beams = Vec::new();
            for ((beam, generated, score), next_dist) in beams.into_iter().zip(next_dists) {
                // print out the distributions
                //println!("dist: {:?}", &next_dist.ids);
                //println!("probs: {:?}", &next_dist.probs);

                // Expand this beam with the top candidates.
                for i in 0..beam_size.min(next_dist.ids.len()) {
                    let mut next_beam = beam.fork();
                    // We assume the distribution is sorted by probability in descending order.
                    next_beam.fill_token(next_dist.ids[i]);

                    let mut next_generated = generated.clone();
                    next_generated.push(next_dist.ids[i]);

                    // Update score by multiplying probabilities.
                    let next_score = score + next_dist.probs[i].ln();

                    next_beams.push((next_beam, next_generated, next_score));
                }
            }

            // Prune: Sort all new candidates by score and keep only the top `beam_size`.
            next_beams.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
            next_beams.truncate(beam_size);
            beams = next_beams;

            //println!("beam size: {}", beams.len());
        }
    }

    /// Generates text using speculative decoding for faster inference.
    ///
    /// This function accelerates text generation by using a small, fast "drafter"
    /// model to propose a sequence of candidate tokens. These draft tokens are then
    /// verified in a single, parallel forward pass by the main model.
    ///
    ///
    /// This process allows the model to generate multiple tokens for the cost of a
    /// single forward pass when the drafter's predictions are correct, significantly
    /// improving throughput.
    ///
    /// # Arguments
    ///
    /// * `drafter`: A mutable reference to an object implementing the `Drafter` trait,
    ///   used to generate speculative token sequences.
    /// * `sampler`: A mutable reference to an object implementing the `Sampler` trait,
    ///   used to sample from the main model's distributions during verification.
    /// * `stop_condition`: A mutable reference to an object implementing the `StopCondition`
    ///   trait, which determines when to halt generation.
    ///
    /// # Returns
    ///
    /// A `String` containing the full sequence of generated text.
    ///
    /// # Panics
    ///
    /// This function will panic if the context's pending token buffer does not contain
    /// exactly one token before the call, as this token is required to seed the
    /// speculative decoding process.
    pub async fn generate_with_drafter<D: Drafter, S: Sample, C: StopCondition>(
        &mut self,
        drafter: &mut D,
        sampler: &mut S,
        stop_condition: &mut C,
    ) -> String {
        // Synchronize the drafter with the main model's history before starting.
        drafter.update(&self.token_ids);

        let mut all_generated_tokens = Vec::new();

        // --- Generation Loop ---
        // Each iteration performs one step of speculative decoding, which may accept
        // multiple tokens at once.
        loop {
            let token_ids_pending = mem::take(&mut self.token_ids_pending);

            // Update the drafter with the seed and generate a sequence of draft tokens.
            drafter.update(&all_generated_tokens);
            let (draft_tokens, draft_pos_ids) = drafter.draft();

            // Combine the seed and draft tokens into a single batch for the main model.
            let batch_tokens = [token_ids_pending.as_slice(), draft_tokens.as_slice()].concat();
            //println!("pending len: {:?}", &token_ids_pending);

            let batch_positions = {
                let pos_offset = self.position_ids.last().map(|&p| p + 1).unwrap_or(0);
                let pending_len = token_ids_pending.len() as u32;
                let mut positions = Vec::with_capacity(batch_tokens.len());

                // Extend with base position to base + pending_len
                positions.extend(pos_offset..pos_offset + pending_len);
                //println!("pending positions: {:?}", &positions);

                // Extend with adjusted draft positions
                positions.extend(
                    draft_pos_ids
                        .iter()
                        .map(|&pos| pos_offset + pending_len - 1 + pos),
                );

                positions
            };

            //println!("verification batch tokens: {:?}", &batch_tokens);
            //println!("verification batch positions: {:?}", &batch_positions);

            // Allocate resources and expand the KV cache to accommodate the entire batch.
            self.grow_kv_pages(batch_tokens.len());

            let out_range = token_ids_pending.len() - 1..batch_tokens.len();

            let p = self.queue.create_forward_pass();
            p.input_tokens(&batch_tokens, &batch_positions);
            p.kv_cache(&self.kv_pages, self.kv_page_last_len);
            p.output_distributions(&out_range.map(|x| x as u32).collect::<Vec<_>>(), 1.0, None);

            let res = p.execute().await;
            let output_distributions = res.distributions.unwrap();
            // The speculation "Trie" is a tree of possibilities. The first token is always correct.
            // R[n] (P[n+1] P[n+2] P[n+3] P[n+2] P[n+3]) (P[n+1] P[n+2]) (P[n+1] P[n+2]) ...

            let mut accepted_tokens: Vec<u32> = Vec::new();
            let mut draft_token_idx = 0; // Current index in the verification batch.

            let mut num_retained_draft_tokens = 0;
            let mut contiguous_flag = true;

            // accept the first token
            let first_accepted_token =
                sampler.sample(&output_distributions[0].ids, &output_distributions[0].probs);

            accepted_tokens.push(first_accepted_token);

            while draft_token_idx < draft_tokens.len() {
                //let next_token = *accepted_draft_tokens.last().unwrap();
                let last_accepted_token = *accepted_tokens.last().unwrap();

                let draft_token = draft_tokens[draft_token_idx];
                let draft_next_dist = &output_distributions[draft_token_idx + 1];
                let draft_next_token = sampler.sample(&draft_next_dist.ids, &draft_next_dist.probs);

                if last_accepted_token == draft_token {
                    // MATCH: Accept token and continue down the current path.
                    accepted_tokens.push(draft_next_token);

                    if contiguous_flag {
                        num_retained_draft_tokens += 1;
                    }

                    // Check if this path has a child in the drafted Trie.
                    let has_child = draft_token_idx + 1 < draft_tokens.len()
                        && draft_pos_ids[draft_token_idx] + 1 == draft_pos_ids[draft_token_idx + 1];
                    if has_child {
                        draft_token_idx += 1; // Move to verify the child.
                    } else {
                        // Nothing left to verify, so we're done.
                        break;
                    }
                } else {
                    contiguous_flag = false;

                    // Attempt to find a sibling branch in the draft to jump to.
                    let current_level = draft_pos_ids[draft_token_idx];
                    let next_sibling_draft_idx = (draft_token_idx + 1..draft_tokens.len())
                        .find(|&idx| draft_pos_ids[idx] == current_level);

                    if let Some(sibling_idx) = next_sibling_draft_idx {
                        // Jump the batch index to the sibling's position.
                        draft_token_idx = sibling_idx;
                    } else {
                        // No more valid paths to check.
                        break;
                    }
                }
            }

            // Rewind the KV cache to discard the states of any rejected draft tokens.
            let redundant_count =
                batch_tokens.len() - token_ids_pending.len() - num_retained_draft_tokens;

            let num_tokens_to_keep = token_ids_pending.len() + num_retained_draft_tokens;
            self.shrink_kv_pages(redundant_count);

            self.token_ids.extend(&batch_tokens[..num_tokens_to_keep]);
            self.position_ids
                .extend(&batch_positions[..num_tokens_to_keep]);
            //drafter.update(&batch_tokens[..num_tokens_to_keep]);

            all_generated_tokens.extend_from_slice(&accepted_tokens);
            self.fill_tokens(accepted_tokens[num_retained_draft_tokens..].to_owned());

            // Check if the stop condition has been met after the step is complete.
            if stop_condition.check(&all_generated_tokens) {
                break;
            }
        }

        self.tokenizer.detokenize(&all_generated_tokens)
    }
}
