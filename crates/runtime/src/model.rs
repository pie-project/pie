use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use tokenizer::Specials;

use anyhow::{Result, anyhow};

use chat_template::Instruct;
use tokenizer::Tokenizer;

static MODEL: OnceLock<Arc<Model>> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub tokenizer: Option<Vec<(String, Vec<u8>)>>,
    pub config: Vec<u8>,
}

/// The deployment `model_id` names: of the package an artifact's stamp
/// names, or of this build's catalog for a raw checkpoint.
pub fn deployment_of(model_id: &str, artifact: &Path) -> Result<crate::catalog::Deployment> {
    if checkpoint::file::serve::stamp_of(artifact)?.is_none() {
        return crate::catalog::catalog().parse(model_id).ok_or_else(|| {
            anyhow!("the engine loaded {model_id:?}, which names no deployment this build serves")
        });
    }
    let package = crate::engine::load::carried(artifact)?;
    let (model, deploy) = package.manifest().parse(model_id).ok_or_else(|| {
        anyhow!(
            "the engine loaded {model_id:?}, which the package `{}` names no deployment of",
            package.name()
        )
    })?;
    Ok(crate::catalog::Deployment::of(
        Arc::clone(&package),
        model,
        deploy,
    ))
}

fn compiled_tokenizer(metadata: &ModelMetadata) -> Option<Result<Tokenizer>> {
    let objects = metadata.tokenizer.as_ref()?;
    Some((|| {
        let canonical = tokenizer::canonical::CanonicalTokenizer::from_objects(|name| {
            objects
                .iter()
                .find(|(have, _)| have == name)
                .map(|(_, bytes)| bytes.clone())
        })?;
        Tokenizer::from_canonical(&canonical)
    })())
}

#[allow(clippy::too_many_arguments)]
pub fn register(
    name: String,
    model_id: &str,
    facts: poem_ir::Facts,
    inputs: Vec<poem_ir::InputDecl>,
    kv_page_size: u32,
    rs: RsCaps,
    eta: EtaCaps,
    tokenizer_path: PathBuf,
    metadata: &ModelMetadata,
) -> Result<()> {
    let deployment = deployment_of(model_id, &tokenizer_path)?;
    let tokenizer = match compiled_tokenizer(metadata) {
        Some(compiled) => compiled?,
        None => Tokenizer::from_file(&tokenizer_path)?,
    };
    let identity = identity_of(&deployment, metadata, &tokenizer_path)?;
    let tokenizer = Arc::new(tokenizer);
    let contract = tokenizer::contract::Contract {
        markers: deployment.markers(),
        pinned: deployment.model.tokenizer.pinned.clone(),
    };
    contract
        .verify(&tokenizer)
        .map_err(|fault| anyhow!("`{model_id}` refuses this artifact's tokenizer: {fault}"))?;
    let instruct = chat_template::build(&template_of(&deployment.model), tokenizer.clone())
        .map_err(|why| anyhow!("`{model_id}` states a template this runtime refuses: {why}"))?;
    let diffusion = deployment
        .try_diffusion()
        .map_err(|why| anyhow!("`{model_id}` states a canvas this runtime refuses: {why}"))?;
    let generative = deployment.try_generative().map_err(|why| {
        anyhow!("`{model_id}` states generative facts this runtime refuses: {why}")
    })?;
    if let Some(generative) = &generative {
        validate_generative(generative).map_err(|fault| {
            anyhow!("`{model_id}` states generative facts this runtime refuses: {fault}")
        })?;
    }

    let model = Arc::new(Model {
        name,
        image: deployment
            .package
            .image(&deployment.model.id, &deployment.deploy)
            .map_err(|why| {
                anyhow!("`{model_id}` states a still front end this runtime refuses: {why:#}")
            })?,
        deployment: deployment.clone(),
        instruct,
        facts,
        inputs,
        kv_page_size,
        rs_caps: rs,
        eta_caps: eta,
        tokenizer,
        vocab: OnceLock::new(),
        vocab_size: deployment.model.vocab,
        num_layers: deployment.model.layers,
        diffusion,
        generative,
        identity,
    });
    MODEL.set(model).map_err(|_| {
        anyhow!("a model is already registered; the runtime serves exactly one model")
    })?;
    Ok(())
}

/// What the model is, apart from how this worker serves it. `weights` is the
/// package's model id until artifacts carry a digest of their source weights.
fn identity_of(
    deployment: &crate::catalog::Deployment,
    metadata: &ModelMetadata,
    tokenizer_path: &Path,
) -> Result<state::ModelIdentity> {
    let mut hasher = blake3::Hasher::new();
    match &metadata.tokenizer {
        Some(objects) => {
            let mut objects: Vec<&(String, Vec<u8>)> = objects.iter().collect();
            objects.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, bytes) in objects {
                hasher.update(&(name.len() as u64).to_le_bytes());
                hasher.update(name.as_bytes());
                hasher.update(&(bytes.len() as u64).to_le_bytes());
                hasher.update(bytes);
            }
        }
        None => {
            let bytes = std::fs::read(tokenizer_path)
                .map_err(|e| anyhow!("read the tokenizer at {}: {e}", tokenizer_path.display()))?;
            hasher.update(&bytes);
        }
    }
    Ok(state::ModelIdentity {
        model: deployment.model.id.clone(),
        weights: deployment.model.id.clone(),
        tokenizer: hasher.finalize().to_hex().to_string(),
    })
}

/// The template a model's package states, as the template crate reads it.
pub(crate) fn template_of(model: &poem::star::Model) -> chat_template::Spec {
    let t = &model.template;
    chat_template::Spec {
        format: t.format.clone(),
        thinking: t.thinking,
        preserve_thinking: t.preserve_thinking,
        reasoning_effort: t.reasoning_effort,
        tools: t.tools,
        xml_tools: t.xml_tools,
        generation_suffix: t.generation_suffix.clone(),
        identity: t.identity.clone(),
        stop: t.stop.clone(),
        bos: t.bos.clone(),
        eos: t.eos.clone(),
    }
}

pub fn validate_generative(generative: &poem::generative::Generative) -> Result<(), String> {
    let mut velocity_width = None;
    for (at, reading) in generative.readings.iter().enumerate() {
        if usize::from(reading.index) != at {
            return Err(format!(
                "reading `{}` sits at position {at} but states index {}; readings are \
                 listed in index order, dense from 0",
                reading.name, reading.index
            ));
        }
        if reading.name.is_empty() {
            return Err(format!("reading {at} has an empty name"));
        }
        if generative.readings[..at]
            .iter()
            .any(|r| r.name == reading.name)
        {
            return Err(format!("reading `{}` is declared twice", reading.name));
        }
        if reading.readout_width == 0 {
            return Err(format!(
                "reading `{}` states a zero-width readout",
                reading.name
            ));
        }
        if reading.readout == poem::generative::ReadoutKind::Velocity {
            match velocity_width {
                None => velocity_width = Some(reading.readout_width),
                Some(width) if width != reading.readout_width => {
                    return Err(format!(
                        "reading `{}` reads a velocity {} wide beside another reading's {width}; \
                         the eta profile carries one velocity width",
                        reading.name, reading.readout_width
                    ));
                }
                Some(_) => {}
            }
        }
    }
    Ok(())
}

pub fn velocity_facts(readings: &[poem::generative::ReadingFact]) -> (bool, u32) {
    readings
        .iter()
        .find(|reading| reading.readout == poem::generative::ReadoutKind::Velocity)
        .map_or((false, 0), |reading| (true, reading.readout_width))
}

pub fn pixels_facts(readings: &[poem::generative::ReadingFact]) -> (bool, u32) {
    let mut widths = readings
        .iter()
        .filter(|reading| reading.readout == poem::generative::ReadoutKind::Pixels)
        .map(|reading| reading.readout_width);
    let Some(first) = widths.next() else {
        return (false, 0);
    };
    (
        true,
        if widths.all(|width| width == first) {
            first
        } else {
            0
        },
    )
}

pub fn model() -> &'static Arc<Model> {
    MODEL.get().expect("model accessed before registration")
}

/// So a key means the same state across restarts and every worker serving
/// the model, whatever its engine, variant or page size.
pub fn identity() -> Option<&'static state::ModelIdentity> {
    MODEL.get().map(|m| &m.identity)
}

pub fn media_pad() -> Option<u32> {
    static PAD: OnceLock<Option<u32>> = OnceLock::new();
    *PAD.get_or_init(|| {
        let m = model();
        let spelling = m.image()?.placeholder.clone();
        match m.tokenize(&spelling)[..] {
            [id] => Some(id),
            _ => None,
        }
    })
}

pub struct Model {
    name: String,
    deployment: crate::catalog::Deployment,
    image: Option<poem::star::ImageSpec>,
    instruct: Arc<dyn Instruct>,
    facts: poem_ir::Facts,
    inputs: Vec<poem_ir::InputDecl>,
    kv_page_size: u32,
    rs_caps: RsCaps,
    eta_caps: EtaCaps,
    tokenizer: Arc<Tokenizer>,
    vocab: OnceLock<(Vec<u32>, Vec<Vec<u8>>)>,
    vocab_size: u32,
    num_layers: u32,
    diffusion: Option<poem::generative::Diffusion>,
    generative: Option<poem::generative::Generative>,
    identity: state::ModelIdentity,
}

#[derive(Debug, Clone, Copy)]
pub struct RsCaps {
    pub state_size: u64,
    pub buffer_page_size: u32,
    /// The window the model's state is read through, when it is one:
    /// its bounded half is then a ring of windowed kv pages, not a slot.
    pub window_tokens: u32,
}

impl RsCaps {
    /// The model has a bounded half beside its kv: a folded state or a window.
    pub fn has_state(&self) -> bool {
        self.state_size > 0 || self.window_tokens > 0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EtaCaps {
    pub has_mtp_logits: bool,
    pub mtp_depth: u32,
    pub draft_block: u32,
    pub draft_mask_token: u32,
    pub draft_proposals_from: u32,
    pub has_value_head: bool,
    pub has_kv_envelopes: bool,
    pub has_attn_score: bool,
    pub has_attn_page_mask: bool,
    pub has_lora: bool,
    pub adapter_attachments: bool,
    /// [`::engine::Capabilities::rs_snapshots`], the least of the engines'.
    pub rs_snapshots: u32,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").field("name", &self.name).finish()
    }
}

impl Model {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The architecture word the package states, for inferlets that ask.
    pub fn arch(&self) -> &str {
        &self.deployment.model.arch
    }

    /// How this model reads a still, if it does.
    pub fn image(&self) -> Option<&poem::star::ImageSpec> {
        self.image.as_ref()
    }

    /// The size a `h` × `w` still is framed to before the tower reads it.
    pub fn frame(&self, h: u32, w: u32, budget: &str) -> Result<(u32, u32)> {
        let d = &self.deployment;
        d.package.frame(&d.model.id, &d.deploy, h, w, budget)
    }

    pub fn instruct(&self) -> &dyn Instruct {
        &*self.instruct
    }

    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    pub fn detokenize(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, Specials::Keep)
    }

    /// Every token the vocabulary spells, each with its bytes.
    pub fn vocabs(&self) -> impl Iterator<Item = (u32, &[u8])> {
        let (ids, bytes) = self.vocab();
        ids.iter().copied().zip(bytes.iter().map(Vec::as_slice))
    }

    fn vocab(&self) -> &(Vec<u32>, Vec<Vec<u8>>) {
        self.vocab.get_or_init(|| {
            let size = self.tokenizer.vocab_size();
            let mut ids = Vec::with_capacity(size);
            let mut bytes = Vec::with_capacity(size);
            for id in 0..size as u32 {
                if let Some(tok_bytes) = self.tokenizer.id_to_token(id) {
                    ids.push(id);
                    bytes.push(tok_bytes);
                }
            }
            (ids, bytes)
        })
    }

    pub fn token_bytes(&self, tokens: &[u32]) -> Vec<Vec<u8>> {
        tokens
            .iter()
            .map(|id| self.tokenizer.id_to_token(*id).unwrap_or_default())
            .collect()
    }

    pub fn tokens_with_prefix(&self, prefix: &[u8]) -> Vec<u32> {
        self.tokenizer.ids_with_prefix(prefix)
    }

    pub fn split_regex(&self) -> Option<&str> {
        self.tokenizer.split_regex()
    }

    pub fn special_tokens(&self) -> impl Iterator<Item = (u32, &[u8])> {
        self.tokenizer.special_tokens()
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn word(
        &self,
        query_len: u32,
        custom_mask: bool,
        adapter: bool,
        drafts: bool,
        captures_scores: bool,
        media: bool,
        block_draft: bool,
        denoise: bool,
        stream: poem::Stream,
        reading: u8,
    ) -> u64 {
        let request = poem::Request::new(query_len, custom_mask)
            .adapted(adapter)
            .drafting(drafts)
            .capturing_scores(captures_scores)
            .with_media(media)
            .drafting_a_block(block_draft)
            .denoising(denoise)
            .on_stream(stream);
        let request = match self.readings().get(usize::from(reading)) {
            Some(fact) => request.in_reading(&fact.name),
            None => request,
        };
        self.facts.word(&request)
    }

    /// The inputs the model reads from its guest; an input's id is its
    /// place here.
    pub fn inputs(&self) -> &[poem_ir::InputDecl] {
        &self.inputs
    }

    /// The input `name` and its id.
    pub fn input(&self, name: &str) -> Option<(u32, &poem_ir::InputDecl)> {
        self.inputs
            .iter()
            .enumerate()
            .find(|(_, input)| input.name == name)
            .map(|(id, input)| (id as u32, input))
    }

    /// The ids of the inputs a lane of these guest switches reads: those a
    /// class it can land in reads, under any value of the switches the
    /// runtime derives (query length, mask, adapter, drafts, scores, media,
    /// encode).
    pub fn lane_inputs(
        &self,
        reading: Option<&str>,
        stream: poem::Stream,
        denoise: bool,
        block_draft: bool,
    ) -> Vec<u32> {
        const RUNTIME_SWITCHES: u32 = 7;
        let words: Vec<u64> = (0..1u32 << RUNTIME_SWITCHES)
            .map(|bits| {
                let request = poem::Request::new(if bits & 1 == 0 { 1 } else { 2 }, bits & 2 != 0)
                    .adapted(bits & 4 != 0)
                    .drafting(bits & 8 != 0)
                    .capturing_scores(bits & 16 != 0)
                    .with_media(bits & 32 != 0)
                    .encoding(bits & 64 != 0)
                    .denoising(denoise)
                    .drafting_a_block(block_draft)
                    .on_stream(stream);
                let request = match reading {
                    Some(name) => request.in_reading(name),
                    None => request,
                };
                self.facts.word(&request)
            })
            .collect();
        (0u32..)
            .zip(&self.inputs)
            .filter(|(_, input)| words.iter().any(|&word| input.read_by(word)))
            .map(|(id, _)| id)
            .collect()
    }

    pub fn diffusion(&self) -> Option<poem::generative::Diffusion> {
        self.diffusion
    }

    pub fn generative(&self) -> Option<&poem::generative::Generative> {
        self.generative.as_ref()
    }

    pub fn readings(&self) -> &[poem::generative::ReadingFact] {
        self.generative
            .as_ref()
            .map_or(&[], |generative| generative.readings.as_slice())
    }

    pub fn sole_reading(&self) -> Option<&poem::generative::ReadingFact> {
        match self.readings() {
            [only] => Some(only),
            _ => None,
        }
    }

    pub fn kv_page_size(&self) -> u32 {
        self.kv_page_size
    }

    pub fn num_layers(&self) -> u32 {
        self.num_layers
    }

    pub fn rs_caps(&self) -> RsCaps {
        self.rs_caps
    }

    pub fn pass_kind(&self) -> crate::pipeline::instance::PassKind {
        use crate::pipeline::instance::PassKind;
        if self.diffusion().is_some() {
            PassKind::Diffusion
        } else {
            PassKind::Text
        }
    }

    pub fn eta_caps(&self) -> EtaCaps {
        self.eta_caps
    }
}
