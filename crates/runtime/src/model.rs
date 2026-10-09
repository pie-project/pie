use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};

use models::template::Instruct;
use tokenizer::Tokenizer;

static MODEL: OnceLock<Arc<Model>> = OnceLock::new();

#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub tokenizer: Option<Vec<(String, Vec<u8>)>>,
    pub config: Vec<u8>,
}

/// The deployment `model_id` names: one of the package the artifact at
/// `artifact` carries, or else one of this build's catalog.
pub fn deployment_of(model_id: &str, artifact: &Path) -> Result<models::Deployment> {
    let Some(package) = crate::engine::load::package_of(artifact)? else {
        return models::Deployment::parse(model_id).ok_or_else(|| {
            anyhow!(
                "the engine loaded {model_id:?}, which names no deployment of this \
                 build's catalog; nearest: {:?}",
                nearest(model_id, 3)
            )
        });
    };
    // The runtime serves one model for as long as it runs.
    let package: &'static poem::star::Package = Box::leak(Box::new(package));
    let (model, deploy) = package.manifest().parse(model_id).ok_or_else(|| {
        anyhow!(
            "the engine loaded {model_id:?}, which the package `{}` names no deployment of",
            package.name()
        )
    })?;
    let entries = Box::leak(models::star::family(package).into_boxed_slice());
    let entry = entries
        .iter()
        .find(|entry| entry.id == model.id)
        .expect("a package's entries are its models");
    let deploy = models::star::catalog(&deploy).map_err(|why| anyhow!("{model_id}: {}", why.0))?;
    Ok(models::Deployment::of(entry, deploy))
}

fn nearest(name: &str, take: usize) -> Vec<&'static str> {
    let mut scored: Vec<(usize, &'static str)> = models::deployments()
        .map(|d| (edit_distance(name, &d.name), d.name.as_str()))
        .collect();
    scored.sort();
    scored
        .into_iter()
        .take(take)
        .map(|(_, name)| name)
        .collect()
}

fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
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
    let tokenizer = Arc::new(tokenizer);
    deployment
        .tokenizer
        .verify(&tokenizer)
        .map_err(|fault| anyhow!("`{model_id}` refuses this artifact's tokenizer: {fault}"))?;
    let instruct = (deployment.template)(tokenizer.clone());
    let diffusion = deployment.diffusion;
    let generative = deployment.generative.clone();
    if let Some(generative) = &generative {
        validate_generative(generative).map_err(|fault| {
            anyhow!("`{model_id}` states generative facts this runtime refuses: {fault}")
        })?;
    }

    let model = Arc::new(Model {
        name,
        arch_name: deployment.entry.arch,
        instruct,
        facts,
        kv_page_size,
        rs_caps: rs,
        eta_caps: eta,
        tokenizer,
        vocab: OnceLock::new(),
        vocab_size: deployment.entry.vocab,
        num_layers: deployment.entry.layers,
        diffusion,
        generative,
    });
    MODEL.set(model).map_err(|_| {
        anyhow!("a model is already registered; the runtime serves exactly one model")
    })?;
    Ok(())
}

pub fn validate_generative(generative: &models::Generative) -> Result<(), String> {
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
        if reading.readout == models::ReadoutKind::Velocity {
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
        for (i, port) in reading.ports.iter().enumerate() {
            if port.name.is_empty() {
                return Err(format!(
                    "reading `{}` port {i} has an empty name",
                    reading.name
                ));
            }
            if reading.ports[..i].iter().any(|p| p.name == port.name) {
                return Err(format!(
                    "reading `{}` declares port `{}` twice",
                    reading.name, port.name
                ));
            }
            if port.width == 0 {
                return Err(format!(
                    "reading `{}` port `{}` states a zero width",
                    reading.name, port.name
                ));
            }
            if port.kind == models::PortKind::AxisPositions && port.width > 4 {
                return Err(format!(
                    "reading `{}` port `{}` states {} axes; a positions port carries 1..=4",
                    reading.name, port.name, port.width
                ));
            }
        }
        if let Some(convention) = &reading.positions {
            let axes = reading
                .ports
                .iter()
                .find(|port| port.kind == models::PortKind::AxisPositions)
                .map(|port| port.width);
            let Some(axes) = axes else {
                return Err(format!(
                    "reading `{}` states a position convention but declares no \
                     axis-positions port",
                    reading.name
                ));
            };
            if convention.axes.len() != axes as usize {
                return Err(format!(
                    "reading `{}` states {} axis roles for a {axes}-wide positions port",
                    reading.name,
                    convention.axes.len()
                ));
            }
            if convention.text_axis >= axes {
                return Err(format!(
                    "reading `{}` numbers its text rows on axis {} of a {axes}-axis \
                     positions port",
                    reading.name, convention.text_axis
                ));
            }
        }
        if !reading.takes_tokens
            && !reading.ports.iter().any(|port| {
                matches!(
                    port.kind,
                    models::PortKind::Latents
                        | models::PortKind::Voxels
                        | models::PortKind::Context
                )
            })
        {
            return Err(format!(
                "reading `{}` embeds no tokens and declares no latents, context or voxels port; \
                 nothing states its lane's row count",
                reading.name
            ));
        }
    }
    Ok(())
}

pub fn velocity_facts(readings: &[models::ReadingFact]) -> (bool, u32) {
    readings
        .iter()
        .find(|reading| reading.readout == models::ReadoutKind::Velocity)
        .map_or((false, 0), |reading| (true, reading.readout_width))
}

pub fn pixels_facts(readings: &[models::ReadingFact]) -> (bool, u32) {
    let mut widths = readings
        .iter()
        .filter(|reading| reading.readout == models::ReadoutKind::Pixels)
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

pub fn media_pad() -> Option<u32> {
    static PAD: OnceLock<Option<u32>> = OnceLock::new();
    *PAD.get_or_init(|| {
        use crate::inferlet::host::media::multimodal;
        let m = model();
        let arch = m.arch_name();
        let spelling = models::media::vision_front_end(arch)
            .map(|fe| fe.delimiters().placeholder)
            .or_else(|| {
                multimodal::audio_arch_supported(arch).then(multimodal::audio_placeholder)
            })?;
        match m.tokenize(spelling)[..] {
            [id] => Some(id),
            _ => None,
        }
    })
}

pub struct Model {
    name: String,
    arch_name: &'static str,
    instruct: Arc<dyn Instruct>,
    facts: poem_ir::Facts,
    kv_page_size: u32,
    rs_caps: RsCaps,
    eta_caps: EtaCaps,
    tokenizer: Arc<Tokenizer>,
    vocab: OnceLock<(Vec<u32>, Vec<Vec<u8>>)>,
    vocab_size: u32,
    num_layers: u32,
    diffusion: Option<models::Diffusion>,
    generative: Option<models::Generative>,
}

#[derive(Debug, Clone, Copy)]
pub struct RsCaps {
    pub state_size: u64,
    pub buffer_page_size: u32,
    pub fold_granularity: u32,
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
    pub draft_bidirectional: bool,
    pub draft_proposals_from: u32,
    pub has_value_head: bool,
    pub has_kv_envelopes: bool,
    pub has_attn_score: bool,
    pub has_attn_page_mask: bool,
    pub has_lora: bool,
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

    pub fn arch_name(&self) -> &'static str {
        self.arch_name
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
        self.tokenizer.decode(tokens, false)
    }

    pub fn get_vocabs(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.vocab().clone()
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

    pub fn get_split_regex(&self) -> String {
        self.tokenizer.get_split_regex()
    }

    pub fn get_special_tokens(&self) -> (Vec<u32>, Vec<Vec<u8>>) {
        self.tokenizer.get_special_tokens()
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
        stream: models::Stream,
        reading: u8,
    ) -> u64 {
        let request = models::Request::new(query_len, custom_mask)
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

    pub fn diffusion(&self) -> Option<models::Diffusion> {
        self.diffusion
    }

    pub fn generative(&self) -> Option<&models::Generative> {
        self.generative.as_ref()
    }

    pub fn readings(&self) -> &[models::ReadingFact] {
        self.generative
            .as_ref()
            .map_or(&[], |generative| generative.readings.as_slice())
    }

    pub fn sole_reading(&self) -> Option<&models::ReadingFact> {
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

    /// What a forward pass of the model is: kv, a bounded state, both, or
    /// a canvas.
    pub fn pass_kind(&self) -> crate::pipeline::instance::PassKind {
        use crate::pipeline::instance::PassKind;
        if self.diffusion().is_some() {
            return PassKind::Diffusion;
        }
        match (self.kv_page_size > 0, self.rs_caps.has_state()) {
            (_, false) => PassKind::Attention,
            (true, true) => PassKind::Hybrid,
            (false, true) => PassKind::Recurrent,
        }
    }

    pub fn eta_caps(&self) -> EtaCaps {
        self.eta_caps
    }
}
