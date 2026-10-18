//! Props: what a client says about its own work, for the scheduling policy
//! (PLEX 1.0 §9).
//!
//! Props travel as JSON text: with serde_json's `arbitrary_precision` on, as in
//! pie's build, a `Value` number serializes as a private struct only JSON reads
//! back.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::ops::{Deref, DerefMut};

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Props(pub BTreeMap<String, Value>);

/// Props under this prefix are the cluster policy's, passed on with a
/// placement; a client cannot set them.
pub const CLUSTER_PREFIX: &str = "cluster/";

pub const MAX_PROPS: usize = 64;
pub const MAX_NAME_BYTES: usize = 128;
/// Names and JSON-encoded values together.
pub const MAX_PROPS_BYTES: usize = 16 * 1024;

pub const MAX_SESSION_BYTES: usize = 128;

/// The prompt text, so the host can estimate `cached-tokens` before the
/// inferlet tokenizes.
pub const PROMPT: &str = "prompt";

/// The request's output limit, so the router has it before the inferlet runs.
pub const MAX_TOKENS: &str = "max-tokens";

pub const MAX_PROMPT_BYTES: usize = 16 * 1024 * 1024;

pub const PROMPT_BLOCK_BYTES: usize = 64;

/// Chained FNV-1a hashes of whole blocks: texts sharing a prefix share its
/// hashes.
pub fn prompt_blocks(text: &str) -> Vec<u64> {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut chain = OFFSET;
    let (blocks, _) = text.as_bytes().as_chunks::<PROMPT_BLOCK_BYTES>();
    blocks
        .iter()
        .map(|block| {
            for &byte in chain.to_le_bytes().iter().chain(block) {
                chain = (chain ^ u64::from(byte)).wrapping_mul(PRIME);
            }
            chain
        })
        .collect()
}

pub fn prompt_of<'a>(props: impl IntoIterator<Item = (&'a String, &'a Value)>) -> Option<&'a str> {
    props
        .into_iter()
        .find(|(name, _)| name.as_str() == PROMPT)
        .and_then(|(_, value)| value.as_str())
}

/// The host scopes a session by tenant.
pub fn check_session(session: Option<&str>) -> Result<(), String> {
    match session {
        Some(s) if s.is_empty() || s.len() > MAX_SESSION_BYTES => Err(format!(
            "a session id must be 1 to {MAX_SESSION_BYTES} bytes"
        )),
        _ => Ok(()),
    }
}

pub fn from_client(props: &mut Props) -> Result<(), String> {
    props.retain(|name, value| !name.starts_with(CLUSTER_PREFIX) && !value.is_null());
    if props.len() > MAX_PROPS {
        return Err(format!("{} props, limit {MAX_PROPS}", props.len()));
    }
    let mut bytes = 0;
    for (name, value) in props.iter() {
        if name.is_empty() || name.len() > MAX_NAME_BYTES {
            return Err(format!(
                "prop name '{name}' must be 1 to {MAX_NAME_BYTES} bytes"
            ));
        }
        if let (PROMPT, Some(text)) = (name.as_str(), value.as_str()) {
            if text.len() > MAX_PROMPT_BYTES {
                return Err(format!(
                    "prop '{PROMPT}' is {} bytes, limit {MAX_PROMPT_BYTES}",
                    text.len()
                ));
            }
            continue;
        }
        bytes += name.len() + value.to_string().len();
    }
    if bytes > MAX_PROPS_BYTES {
        return Err(format!("props are {bytes} bytes, limit {MAX_PROPS_BYTES}"));
    }
    Ok(())
}

/// English runs about four bytes a token.
pub const TOKENS_PER_BYTE: f64 = 0.25;

/// Approximate by design: a text block is not a page, and text is not tokens.
#[derive(Debug, Clone, Default)]
pub struct PromptIndex {
    blocks: HashMap<u64, (u32, u64)>,
    holders: HashMap<String, (u64, Vec<u64>)>,
    order: BTreeMap<u64, String>,
    stamp: u64,
}

impl PromptIndex {
    pub fn put(&mut self, holder: &str, blocks: &[u64], tokens_per_block: f64) {
        self.remove(holder);
        self.stamp += 1;
        for (at, &block) in blocks.iter().enumerate() {
            let tokens = ((at + 1) as f64 * tokens_per_block) as u64;
            let held = self.blocks.entry(block).or_insert((0, 0));
            held.0 += 1;
            held.1 = held.1.max(tokens);
        }
        self.holders
            .insert(holder.to_string(), (self.stamp, blocks.to_vec()));
        self.order.insert(self.stamp, holder.to_string());
    }

    pub fn remove(&mut self, holder: &str) {
        let Some((stamp, blocks)) = self.holders.remove(holder) else {
            return;
        };
        self.order.remove(&stamp);
        for block in blocks {
            if let Some(held) = self.blocks.get_mut(&block) {
                held.0 -= 1;
                if held.0 == 0 {
                    self.blocks.remove(&block);
                }
            }
        }
    }

    pub fn trim(&mut self, blocks: usize) {
        while self.blocks.len() > blocks {
            let Some((_, holder)) = self.order.pop_first() else {
                break;
            };
            self.remove(&holder);
        }
    }

    pub fn held_tokens(&self, blocks: &[u64]) -> u64 {
        blocks
            .iter()
            .map_while(|block| self.blocks.get(block))
            .last()
            .map_or(0, |&(_, tokens)| tokens)
    }
}

pub fn merge(current: &Props, changes: Props) -> Result<(Props, Vec<String>), String> {
    let mut merged = current.clone();
    for (name, value) in changes {
        if value.is_null() {
            merged.remove(&name);
        } else {
            merged.insert(name, value);
        }
    }
    from_client(&mut merged)?;
    let changed: BTreeSet<&String> = current
        .keys()
        .chain(merged.keys())
        .filter(|name| current.get(*name) != merged.get(*name))
        .collect();
    let changed = changed.into_iter().cloned().collect();
    Ok((merged, changed))
}

impl Props {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Deref for Props {
    type Target = BTreeMap<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Props {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl FromIterator<(String, Value)> for Props {
    fn from_iter<I: IntoIterator<Item = (String, Value)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for Props {
    type Item = (String, Value);
    type IntoIter = std::collections::btree_map::IntoIter<String, Value>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Serialize for Props {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let text = serde_json::to_string(&self.0).map_err(serde::ser::Error::custom)?;
        serializer.serialize_str(&text)
    }
}

impl<'de> Deserialize<'de> for Props {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PropsVisitor;

        impl<'de> Visitor<'de> for PropsVisitor {
            type Value = Props;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("props: a JSON object, or its JSON text")
            }

            fn visit_str<E: de::Error>(self, text: &str) -> Result<Props, E> {
                serde_json::from_str(text).map(Props).map_err(E::custom)
            }

            fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<Props, A::Error> {
                // serde_json's own map reader, which knows its number form.
                BTreeMap::deserialize(de::value::MapAccessDeserializer::new(map)).map(Props)
            }
        }

        deserializer.deserialize_any(PropsVisitor)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::ClientMessage;

    fn props(value: Value) -> Props {
        serde_json::from_value::<BTreeMap<String, Value>>(value)
            .unwrap()
            .into_iter()
            .collect()
    }

    #[test]
    fn a_client_sets_any_name_but_the_routers_within_the_limits() {
        let mut given = props(json!({
            "priority": 3,
            "cluster/priority": 9,
            "deadline": null,
            "tags": ["a", "b"],
        }));
        from_client(&mut given).unwrap();
        assert_eq!(given, props(json!({"priority": 3, "tags": ["a", "b"]})));

        let mut many: Props = (0..=MAX_PROPS)
            .map(|i| (format!("p{i}"), json!(i)))
            .collect();
        assert!(from_client(&mut many).is_err());
        let mut long: Props = [("x".repeat(MAX_NAME_BYTES + 1), json!(1))]
            .into_iter()
            .collect();
        assert!(from_client(&mut long).is_err());
        let mut large: Props = [("blob".to_string(), json!("x".repeat(MAX_PROPS_BYTES)))]
            .into_iter()
            .collect();
        assert!(from_client(&mut large).is_err());
    }

    #[test]
    fn the_prompt_is_outside_the_byte_limit_and_shares_its_blocks() {
        let long = "x".repeat(MAX_PROPS_BYTES * 4);
        let mut set = props(json!({ "prompt": long }));
        assert!(from_client(&mut set).is_ok(), "under its own limit");
        let mut set = props(json!({ "note": "x".repeat(MAX_PROPS_BYTES) }));
        assert!(from_client(&mut set).is_err());

        let first = "system: be brief\nuser: what is the capital of france? ".repeat(4);
        let second = format!("{first}and of spain?");
        let (a, b) = (prompt_blocks(&first), prompt_blocks(&second));
        assert_eq!(a.len(), first.len() / PROMPT_BLOCK_BYTES);
        assert_eq!(b[..a.len()], a[..], "a shared start, shared blocks");
        let other = prompt_blocks(&first.replacen('s', "S", 1));
        assert!(
            other.iter().zip(&a).all(|(x, y)| x != y),
            "chained: one byte moves them all"
        );
    }

    #[test]
    fn an_index_holds_what_its_holders_put_and_keeps_the_newest() {
        let first = "a".repeat(PROMPT_BLOCK_BYTES * 4);
        let longer = format!("{first}{}", "b".repeat(PROMPT_BLOCK_BYTES * 2));
        let (a, b) = (prompt_blocks(&first), prompt_blocks(&longer));
        let mut index = PromptIndex::default();
        assert_eq!(index.held_tokens(&b), 0);
        index.put("p1", &a, 16.0);
        assert_eq!(index.held_tokens(&b), 64, "the four blocks they share");
        index.put("p2", &b, 16.0);
        assert_eq!(index.held_tokens(&b), 96);
        index.remove("p2");
        assert_eq!(index.held_tokens(&b), 64, "p1 still holds its own");
        let other = prompt_blocks(&"c".repeat(PROMPT_BLOCK_BYTES * 3));
        index.put("p3", &other, 16.0);
        index.trim(4);
        assert_eq!(index.held_tokens(&a), 0, "the oldest went");
        assert_eq!(index.held_tokens(&other), 48, "the newest stays");
    }

    #[test]
    fn a_merge_names_what_it_changed() {
        let current = props(json!({"priority": 1, "tier": "gold", "keep": true}));
        let changes = props(json!({
            "priority": 5,
            "tier": null,
            "keep": true,
            "new": [1],
            "cluster/priority": 9,
        }));
        let (merged, mut changed) = merge(&current, changes).unwrap();
        assert_eq!(
            merged,
            props(json!({"priority": 5, "keep": true, "new": [1]}))
        );
        changed.sort();
        assert_eq!(changed, ["new", "priority", "tier"]);

        let many: Props = (0..MAX_PROPS)
            .map(|i| (format!("p{i}"), json!(i)))
            .collect();
        let (full, _) = merge(&Props::new(), many).unwrap();
        let one_more: Props = [("x".to_string(), json!(1))].into_iter().collect();
        assert!(
            merge(&full, one_more).is_err(),
            "the limits hold for the merge"
        );
    }

    /// The dev-dependency turns serde_json's `arbitrary_precision` on here,
    /// as pie's build does, so this is the number form that broke.
    #[test]
    fn props_survive_json_and_msgpack_with_any_number() {
        let sent = r#"{"type":"launch_process","corr_id":1,"inferlet":"chat","input":"",
            "capture_outputs":true,"props":{"priority":10,"w":0.5,"big":1e300,
            "tags":["a",2],"tier":"gold"}}"#;
        let expected =
            json!({"priority": 10, "w": 0.5, "big": 1e300, "tags": ["a", 2], "tier": "gold"});
        let launch: ClientMessage = serde_json::from_str(sent).unwrap();
        for bytes in [
            rmp_serde::to_vec(&launch).unwrap(),
            rmp_serde::to_vec_named(&launch).unwrap(),
        ] {
            let ClientMessage::LaunchProcess { props, .. } = rmp_serde::from_slice(&bytes).unwrap()
            else {
                panic!("still a launch");
            };
            assert_eq!(serde_json::to_value(&props.0).unwrap(), expected);
            assert_eq!(props["priority"].as_i64(), Some(10));
            assert_eq!(props["w"].as_f64(), Some(0.5));
        }
        let bare = ClientMessage::LaunchProcess {
            corr_id: 1,
            inferlet: "chat".into(),
            input: String::new(),
            capture_outputs: true,
            props: Props::new(),
            session: Some("conv-1".into()),
            cluster_props: Props::new(),
        };
        for bytes in [
            rmp_serde::to_vec(&bare).unwrap(),
            rmp_serde::to_vec_named(&bare).unwrap(),
        ] {
            let ClientMessage::LaunchProcess { props, session, .. } =
                rmp_serde::from_slice(&bytes).unwrap()
            else {
                panic!("still a launch");
            };
            assert!(props.is_empty());
            assert_eq!(session.as_deref(), Some("conv-1"));
        }
    }
}
