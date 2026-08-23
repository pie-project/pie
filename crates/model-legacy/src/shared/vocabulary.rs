#[derive(Clone, Copy, Debug)]
pub struct Member {
    pub pie: &'static str,

    pub hf: &'static str,

    pub gguf: Option<&'static str>,
}

impl Member {
    #[must_use]
    pub const fn same(name: &'static str) -> Self {
        Self {
            pie: name,
            hf: name,
            gguf: None,
        }
    }

    #[must_use]
    pub const fn gguf(name: &'static str, gguf: &'static str) -> Self {
        Self {
            pie: name,
            hf: name,
            gguf: Some(gguf),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Vocab(pub &'static [Member]);

impl Vocab {
    #[must_use]
    pub fn respells(&self) -> bool {
        self.0.iter().any(|m| m.pie != m.hf)
    }

    #[must_use]
    pub fn from_hf(&self, name: &str) -> Option<String> {
        self.translate(name, Suffix::Optional, |m| Some(m.hf))
    }

    #[must_use]
    pub fn from_gguf(&self, name: &str) -> Option<String> {
        self.translate(name, Suffix::Required, |m| m.gguf)
    }

    fn translate(
        &self,
        name: &str,
        policy: Suffix,
        from: fn(&Member) -> Option<&'static str>,
    ) -> Option<String> {
        let (stem, suffix) = match name.rsplit_once('.') {
            Some((stem, tail @ ("weight" | "bias"))) => (stem, Some(tail)),
            _ if policy == Suffix::Required => return None,
            _ => (name, None),
        };
        let member = self.0.iter().find_map(|m| {
            from(m)
                .and_then(|pattern| capture(pattern, stem))
                .map(|layer| (m, layer))
        })?;
        let (m, layer) = member;
        let mut pie = match layer {
            Some(index) => m.pie.replace("{layer}", &index.to_string()),
            None => m.pie.to_string(),
        };

        pie = pie.replace("{expert}", "{}");
        Some(match suffix {
            Some(suffix) => format!("{pie}.{suffix}"),
            None => pie,
        })
    }
}

#[must_use]
pub fn gguf_member(name: &str) -> Option<(u32, &str)> {
    let (stem, _) = name.rsplit_once('.')?;
    let rest = stem.strip_prefix("blk.")?;
    let (index, member) = rest.split_once('.')?;
    Some((index.parse().ok()?, member))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Suffix {
    Required,

    Optional,
}

fn capture(pattern: &str, stem: &str) -> Option<Option<u32>> {
    let Some((head, tail)) = pattern.split_once("{layer}") else {
        return (pattern == stem).then_some(None);
    };
    let rest = stem.strip_prefix(head)?;
    let digits = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits == 0 || rest[digits..] != *tail {
        return None;
    }
    Some(Some(rest[..digits].parse().ok()?))
}
