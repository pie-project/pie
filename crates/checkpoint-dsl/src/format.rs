//! The ways a family's checkpoints are laid out, and the one a checkpoint is.
//!
//! A container (safetensors, GGUF, a diffusers pipeline) says nothing of how a
//! model's tensors are named or which configuration keys state its shape; a
//! family does. It lists its formats, each with what only its checkpoints hold
//! and the configuration a checkpoint of it states, and [`read_one`] reads a
//! checkpoint by the one format that recognizes it: none recognizing it, or
//! two, is a refusal, not a first-that-works.

use ztensor::format::cbor::Value;

use crate::Error;

/// The attribute a snapshot's configuration rides under
/// (`checkpoint::file::snapshot::CONFIG`).
pub const CONFIG: &str = "config";

/// One way a family's checkpoints are laid out.
pub struct Format<'a, T> {
    name: &'static str,
    recognizes: Option<Box<dyn Fn(&ztensor::Source) -> bool + 'a>>,
    states: Vec<Stated>,
    read: Box<dyn FnOnce() -> Result<T, Error> + 'a>,
}

impl<'a, T> Format<'a, T> {
    /// The format `name`: a checkpoint `recognizes` holds is one of it, and
    /// `read` reads it.
    pub fn new(
        name: &'static str,
        recognizes: impl Fn(&ztensor::Source) -> bool + 'a,
        read: impl FnOnce() -> Result<T, Error> + 'a,
    ) -> Format<'a, T> {
        Format {
            name,
            recognizes: Some(Box::new(recognizes)),
            states: Vec::new(),
            read: Box::new(read),
        }
    }

    /// The format `name`, recognized by `read` reading a checkpoint: for a
    /// family whose formats share their names, so that reading is the only
    /// test of one. Two readings of a checkpoint still refuse it.
    pub fn reading(
        name: &'static str,
        read: impl FnOnce() -> Result<T, Error> + 'a,
    ) -> Format<'a, T> {
        Format {
            name,
            recognizes: None,
            states: Vec::new(),
            read: Box::new(read),
        }
    }

    /// What a checkpoint of this format states of the model it holds: a
    /// checkpoint stating another value is another model's.
    #[must_use]
    pub fn stating(mut self, states: impl IntoIterator<Item = Stated>) -> Format<'a, T> {
        self.states.extend(states);
        self
    }
}

/// A configuration value a checkpoint states, and the value the model is
/// built with.
#[derive(Clone, Debug, PartialEq)]
pub struct Stated {
    pub key: Key,
    pub want: f64,
    pub at_least: bool,
}

/// Where a checkpoint states a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Key {
    /// A dotted path into the snapshot's configuration (`config.json`, or a
    /// pipeline's per-component configurations under their prefix).
    Config(String),
    /// A top-level attribute of the container (a GGUF metadata key).
    Attribute(String),
}

/// The configuration value at `path` must be `want`.
pub fn config(path: impl Into<String>, want: impl Into<f64>) -> Stated {
    Stated {
        key: Key::Config(path.into()),
        want: want.into(),
        at_least: false,
    }
}

/// The container attribute `key` must be `want`.
pub fn attribute(key: impl Into<String>, want: impl Into<f64>) -> Stated {
    Stated {
        key: Key::Attribute(key.into()),
        want: want.into(),
        at_least: false,
    }
}

impl Stated {
    /// A depth or a bank's breadth: the checkpoint may hold more than the
    /// model reads, since a miniature reads a prefix of its whole model's layers
    /// or experts; fewer is another model's.
    #[must_use]
    pub fn or_deeper(mut self) -> Stated {
        self.at_least = true;
        self
    }
}

/// Reads `src` as `family` by the one of `formats` that recognizes it, once the
/// configuration it states agrees with the model's.
pub fn read_one<T>(
    family: &str,
    src: &ztensor::Source,
    formats: Vec<Format<'_, T>>,
) -> Result<T, Error> {
    let annotate = |format: &str, why: Error| match why {
        Error::Illegible { name, detail } => Error::Illegible {
            name,
            detail: format!("as {format}, {detail}"),
        },
        other => other,
    };
    let names: Vec<&str> = formats.iter().map(|f| f.name).collect();
    let mut refusals: Vec<String> = Vec::new();
    let mut recognized: Vec<(
        &'static str,
        Option<Result<T, Error>>,
        Vec<Stated>,
        Option<Format<'_, T>>,
    )> = Vec::new();
    for format in formats {
        match &format.recognizes {
            Some(recognizes) => {
                if recognizes(src) {
                    let (name, states) = (format.name, format.states.clone());
                    recognized.push((name, None, states, Some(format)));
                }
            }
            None => {
                let Format {
                    name, states, read, ..
                } = format;
                match agrees(src, &states).map_err(|detail| Error::Illegible {
                    name: family.to_string(),
                    detail,
                }) {
                    Err(why) => refusals.push(format!("as {name}, {why}")),
                    Ok(()) => match read() {
                        Ok(read) => recognized.push((name, Some(Ok(read)), states, None)),
                        Err(why) => refusals.push(format!("as {name}, {why}")),
                    },
                }
            }
        }
    }
    match recognized.len() {
        0 => Err(Error::Illegible {
            name: family.to_string(),
            detail: if refusals.is_empty() {
                format!(
                    "no format of `{family}` recognizes this checkpoint; it reads {}",
                    names.join(", ")
                )
            } else {
                format!(
                    "no format of `{family}` reads this checkpoint — {}",
                    refusals.join("; ")
                )
            },
        }),
        1 => {
            let (name, read, states, format) = recognized.remove(0);
            if let Some(read) = read {
                return read;
            }
            agrees(src, &states).map_err(|detail| Error::Illegible {
                name: family.to_string(),
                detail: format!("as {name}, {detail}"),
            })?;
            let format = format.expect("a recognized format that has not read is held");
            (format.read)().map_err(|why| annotate(name, why))
        }
        _ => Err(Error::Illegible {
            name: family.to_string(),
            detail: format!(
                "{} all recognize this checkpoint; a format recognizes what only its \
                 checkpoints hold",
                recognized
                    .iter()
                    .map(|r| r.0)
                    .collect::<Vec<_>>()
                    .join(" and ")
            ),
        }),
    }
}

/// Whether every value `states` names that `src` states is the one wanted; a
/// value the checkpoint does not state is not held against it.
fn agrees(src: &ztensor::Source, states: &[Stated]) -> Result<(), String> {
    for stated in states {
        let (found, spelled) = match &stated.key {
            Key::Config(path) => (
                src.attributes()
                    .and_then(|a| a.get(CONFIG))
                    .and_then(|c| at_path(c, path)),
                format!("its configuration's `{path}`"),
            ),
            Key::Attribute(key) => (
                src.attributes().and_then(|a| a.get(key)),
                format!("its `{key}`"),
            ),
        };
        let Some(found) = found.and_then(number) else {
            continue;
        };
        let fits = if stated.at_least {
            found >= stated.want
        } else {
            same(found, stated.want)
        };
        if !fits {
            return Err(format!(
                "the checkpoint states {spelled} as {found} and this model is built for {}{}",
                stated.want,
                if stated.at_least { " or more" } else { "" }
            ));
        }
    }
    Ok(())
}

fn at_path<'v>(value: &'v Value, path: &str) -> Option<&'v Value> {
    path.split('.').try_fold(value, |at, key| at.get(key))
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Uint(n) => Some(*n as f64),
        Value::Nint(n) => Some(-1.0 - *n as f64),
        Value::Float(x) => Some(*x),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        _ => None,
    }
}

fn same(found: f64, want: f64) -> bool {
    found == want || (found - want).abs() <= 1e-6 * want.abs().max(found.abs())
}

/// Whether `src` holds a tensor named `name`.
#[must_use]
pub fn has(src: &ztensor::Source, name: &str) -> bool {
    src.get(name).is_some()
}

/// Whether `src` holds a tensor whose name ends in `suffix`.
#[must_use]
pub fn has_suffix(src: &ztensor::Source, suffix: &str) -> bool {
    src.names().any(|name| name.ends_with(suffix))
}

/// The container attribute `key`, as text.
#[must_use]
pub fn attribute_text<'s>(src: &'s ztensor::Source, key: &str) -> Option<&'s str> {
    src.attributes()?.get(key)?.as_text()
}
