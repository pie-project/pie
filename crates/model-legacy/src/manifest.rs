use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Presence {
    Required,

    Absent,

    Optional,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,

    pub extents: Vec<u64>,
    pub presence: Presence,

    pub instead: Vec<Vec<(String, Vec<u64>)>>,

    pub tied_copy: Vec<u64>,
}

impl TensorSpec {
    #[must_use]
    pub fn required(name: impl Into<String>, extents: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            extents: extents.into(),
            presence: Presence::Required,
            instead: Vec::new(),
            tied_copy: Vec::new(),
        }
    }

    #[must_use]
    pub fn present(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            extents: Vec::new(),
            presence: Presence::Required,
            instead: Vec::new(),
            tied_copy: Vec::new(),
        }
    }

    #[must_use]
    pub fn absent(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            extents: Vec::new(),
            presence: Presence::Absent,
            instead: Vec::new(),
            tied_copy: Vec::new(),
        }
    }

    #[must_use]
    pub fn tied(name: impl Into<String>, extents: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            extents: Vec::new(),
            presence: Presence::Absent,
            instead: Vec::new(),
            tied_copy: extents.into(),
        }
    }

    #[must_use]
    pub fn optional(name: impl Into<String>, extents: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            extents: extents.into(),
            presence: Presence::Optional,
            instead: Vec::new(),
            tied_copy: Vec::new(),
        }
    }

    #[must_use]
    pub fn or_published_as<N, S>(mut self, layout: impl IntoIterator<Item = (N, S)>) -> Self
    where
        N: Into<String>,
        S: Into<Vec<u64>>,
    {
        self.instead.push(
            layout
                .into_iter()
                .map(|(n, e)| (n.into(), e.into()))
                .collect(),
        );
        self
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Manifest {
    pub layers: u32,
    pub tensors: Vec<TensorSpec>,

    pub proj_repr: Option<model_dsl::WeightRepr>,

    pub expert_repr: Option<model_dsl::WeightRepr>,
}

impl Manifest {
    #[must_use]
    pub fn new(layers: u32) -> Self {
        Self {
            layers,
            tensors: Vec::new(),
            proj_repr: None,
            expert_repr: None,
        }
    }

    #[must_use]
    pub fn holds_projections_as(mut self, repr: model_dsl::WeightRepr) -> Self {
        self.proj_repr = Some(repr);
        self
    }

    #[must_use]
    pub fn holds_experts_as(mut self, repr: model_dsl::WeightRepr) -> Self {
        self.expert_repr = Some(repr);
        self
    }

    #[must_use]
    pub fn with(mut self, spec: TensorSpec) -> Self {
        self.tensors.push(spec);
        self
    }

    #[must_use]
    pub fn with_if(self, when: bool, spec: TensorSpec) -> Self {
        if when { self.with(spec) } else { self }
    }

    #[must_use]
    pub fn either(self, when: bool, name: &str, extents: impl Into<Vec<u64>>) -> Self {
        if when {
            self.with(TensorSpec::required(name, extents))
        } else {
            self.with(TensorSpec::absent(name))
        }
    }

    #[must_use]
    pub fn tie(self, tied: bool, name: &str, extents: impl Into<Vec<u64>>) -> Self {
        if tied {
            self.with(TensorSpec::tied(name, extents))
        } else {
            self.with(TensorSpec::required(name, extents))
        }
    }

    fn rows(&self) -> impl Iterator<Item = (String, &TensorSpec)> {
        self.tensors
            .iter()
            .map(|spec| (Observed::logical(&spec.name), spec))
    }

    pub fn check(&self, observed: &Observed) -> Result<(), Mismatch> {
        let mut faults = Vec::new();
        for (name, spec) in self.rows() {
            match (spec.presence, observed.extents(&name)) {
                (Presence::Required, None) if applies(&spec.instead, observed) => {}
                (Presence::Required, None) => faults.push(Fault::Missing(name)),

                (Presence::Absent, Some(seen))
                    if !spec.tied_copy.is_empty()
                        && extents_agree(
                            &spec.tied_copy,
                            seen,
                            observed.has(&format!("{name}.scales")),
                        ) => {}
                (Presence::Absent, Some(_)) => faults.push(Fault::Unexpected(name)),
                (Presence::Required | Presence::Optional, Some(seen))
                    if !spec.extents.is_empty()
                        && !extents_agree(
                            &spec.extents,
                            seen,
                            observed.has(&format!("{name}.scales")),
                        ) =>
                {
                    faults.push(Fault::Extent {
                        name,
                        want: spec.extents.clone(),
                        got: seen.to_vec(),
                    });
                }
                _ => {}
            }
        }
        if faults.is_empty() {
            Ok(())
        } else {
            Err(Mismatch { faults })
        }
    }
}

fn applies(layouts: &[Vec<(String, Vec<u64>)>], observed: &Observed) -> bool {
    layouts.iter().any(|layout| {
        !layout.is_empty()
            && layout.iter().all(|(name, want)| {
                let name = Observed::logical(name);
                observed.extents(&name).is_some_and(|seen| {
                    want.is_empty()
                        || extents_agree(want, seen, observed.has(&format!("{name}.scales")))
                })
            })
    })
}

fn extents_agree(want: &[u64], got: &[u64], packed: bool) -> bool {
    let squeeze = |d: &[u64]| -> Vec<u64> {
        let mut v: Vec<u64> = d.iter().copied().filter(|&x| x != 1).collect();
        if v.is_empty() && !d.is_empty() {
            v.push(1);
        }
        v
    };
    let (want, got) = (squeeze(want), squeeze(got));
    if want == got {
        return true;
    }
    if !packed || want.len() != got.len() || want.is_empty() {
        return false;
    }
    let split = want.len() - 1;
    if want[..split] != got[..split] {
        return false;
    }
    let (w, g) = (want[split], got[split]);

    g != 0 && w > g && w.is_multiple_of(g) && (w / g).is_power_of_two()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mismatch {
    pub faults: Vec<Fault>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    Missing(String),

    Unexpected(String),

    Extent {
        name: String,
        want: Vec<u64>,
        got: Vec<u64>,
    },
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(name) => write!(f, "missing {name}"),
            Self::Unexpected(name) => write!(f, "unexpected {name}"),
            Self::Extent { name, want, got } => {
                write!(f, "{name} is {got:?}, this variant implies {want:?}")
            }
        }
    }
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let listed: Vec<String> = self
            .faults
            .iter()
            .take(4)
            .map(ToString::to_string)
            .collect();
        write!(f, "{}", listed.join("; "))?;
        if self.faults.len() > listed.len() {
            write!(f, " (+{} more)", self.faults.len() - listed.len())?;
        }
        Ok(())
    }
}

impl std::error::Error for Mismatch {}

#[derive(Clone, Debug, Default)]
pub struct Observed {
    by_name: BTreeMap<String, Vec<u64>>,
}

impl Observed {
    pub fn from_pairs<N, S>(pairs: impl IntoIterator<Item = (N, S)>) -> Self
    where
        N: AsRef<str>,
        S: AsRef<[u64]>,
    {
        let mut by_name: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        let mut at: BTreeMap<String, u64> = BTreeMap::new();
        for (name, extents) in pairs {
            let raw = name.as_ref();
            let key = Self::logical(raw);
            let index = Self::layer_index(raw).unwrap_or(0);
            if at.get(&key).is_some_and(|&seen| seen <= index) {
                continue;
            }
            at.insert(key.clone(), index);
            by_name.insert(key, extents.as_ref().to_vec());
        }
        Self { by_name }
    }

    fn layer_index(raw: &str) -> Option<u64> {
        let (at, token) = ["layers.", "layer."]
            .into_iter()
            .filter_map(|token| raw.find(token).map(|at| (at, token)))
            .min_by_key(|(at, _)| *at)?;
        let rest = &raw[at + token.len()..];
        let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    }

    #[must_use]
    pub fn extents(&self, logical: &str) -> Option<&[u64]> {
        self.by_name.get(logical).map(Vec::as_slice)
    }

    #[must_use]
    pub fn without<N: AsRef<str>>(mut self, names: impl IntoIterator<Item = N>) -> Self {
        for name in names {
            self.by_name.remove(&Self::logical(name.as_ref()));
        }
        self
    }

    #[must_use]
    pub fn renamed<K: AsRef<str>, V: AsRef<str>>(
        mut self,
        pairs: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        for (from, to) in pairs {
            let from = Self::logical(from.as_ref());
            let to = Self::logical(to.as_ref());
            if from == to {
                continue;
            }
            if let Some(extents) = self.by_name.remove(&from) {
                self.by_name.insert(to, extents);
            }
        }
        self
    }

    #[must_use]
    pub fn unstacked<K: AsRef<str>, V: AsRef<str>>(
        mut self,
        pairs: impl IntoIterator<Item = (K, V)>,
    ) -> Self {
        for (from, template) in pairs {
            let Some(extents) = self.by_name.remove(&Self::logical(from.as_ref())) else {
                continue;
            };
            let Some((&count, instance)) = extents.split_first() else {
                continue;
            };
            for index in 0..count {
                let name = template.as_ref().replace("{}", &index.to_string());
                self.by_name.insert(Self::logical(&name), instance.to_vec());
            }
        }
        self
    }

    #[must_use]
    pub fn has(&self, logical: &str) -> bool {
        self.by_name.contains_key(logical)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_name.keys().map(String::as_str)
    }

    #[must_use]
    pub fn logical(raw: &str) -> String {
        let mut name = raw;
        loop {
            let before = name.len();
            for prefix in ["language_model.", "text_model.", "model.", "transformer."] {
                if let Some(rest) = name.strip_prefix(prefix) {
                    name = rest;
                }
            }

            if name.len() == before {
                break;
            }
        }
        let name = name.strip_suffix(".weight").unwrap_or(name);
        let mut out = String::with_capacity(name.len() + 4);
        let mut rest = name;

        while let Some((at, token)) = ["layers.", "layer."]
            .into_iter()
            .filter_map(|token| rest.find(token).map(|at| (at, token)))
            .min_by_key(|(at, _)| *at)
        {
            out.push_str(&rest[..at]);
            let tail = &rest[at + token.len()..];
            let digits = tail
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(tail.len());
            if digits == 0 {
                out.push_str(token);
                rest = tail;
                continue;
            }
            out.push_str("layer.{}");
            rest = &tail[digits..];
        }
        out.push_str(rest);
        Self::global_spelling(out)
    }

    fn global_spelling(name: String) -> String {
        for (lowered, checkpoint) in [("shared_embedding", "embed_tokens"), ("final_norm", "norm")]
        {
            if name == lowered {
                return checkpoint.to_string();
            }
            if let Some(rest) = name.strip_prefix(lowered)
                && rest.starts_with('.')
            {
                return format!("{checkpoint}{rest}");
            }
        }
        name
    }
}

#[cfg(feature = "contract")]
mod from_checkpoint {
    use super::Observed;
    use model_loader::checkpoint::CheckpointMetadata;
    use model_loader::types::Encoding;

    impl Observed {
        #[must_use]
        pub fn of(metadata: &CheckpointMetadata) -> Self {
            Self::from_pairs(metadata.weights().map(|tensor| {
                let extents = logical_extents(&tensor.shape, &tensor.encoding, tensor.span_bytes);
                (tensor.name.clone(), extents)
            }))
        }
    }

    fn logical_extents(shape: &[i64], encoding: &Encoding, span_bytes: u64) -> Vec<u64> {
        let mut stored: Vec<u64> = shape
            .iter()
            .map(|&d| u64::try_from(d).unwrap_or(0))
            .collect();
        let Encoding::Quant(spec) = encoding else {
            return stored;
        };
        let stored_elems: u64 = stored.iter().product();
        let logical = logical_element_count(spec, span_bytes);
        if stored_elems == 0 || logical <= stored_elems || !logical.is_multiple_of(stored_elems) {
            return stored;
        }
        if let Some(last) = stored.last_mut() {
            *last *= logical / stored_elems;
        }
        stored
    }

    fn logical_element_count(spec: &model_loader::types::QuantSpec, span_bytes: u64) -> u64 {
        let spec = spec.clone().normalized();
        if let Some((elems, bytes)) = spec.block_layout() {
            return span_bytes
                .checked_div(bytes)
                .map_or(0, |blocks| blocks * elems);
        }
        let bits = u64::from(spec.normalized_bits());
        (span_bytes * 8).checked_div(bits).unwrap_or(0)
    }
}
