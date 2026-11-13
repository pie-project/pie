//! What a session names, and each name's state.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use client_api::{
    GrammarSource, HandleErrorCode, HandleKind, HandleName, HandleOp, Handles, Props,
};
use serde_json::{Value, json};
use uuid::Uuid;

use super::held::Held;
use super::lease::{Bound, Lease};
use super::{HandleError, MAX_HANDLES, MAX_MEDIA_BYTES};
use crate::inferlet::ProcessId;
use crate::inferlet::host::frames::{Frames, Pcm};
use crate::inferlet::host::grammar::Matcher;
use crate::policy::{self, By, Event, Resting};
use crate::server::ClientId;

#[derive(Debug)]
pub(crate) enum Slot {
    /// Named, and nothing else yet: a process makes it, on its own engine.
    Unborn(HandleKind),
    Parked(Box<Held>),
    /// Host memory and nothing else: it moves as it is. The lock is for
    /// `Sync`, which the actor holding the table needs and it alone reaches.
    Matcher(Box<Mutex<Matcher>>),
    /// Host memory never written once made: a fork shares it.
    Frames(Box<Frames>),
    Pcm(Box<Pcm>),
    /// In a process, or promised by one: frames naming it wait.
    Lent,
    /// Gone for a reason the next use hears.
    Dead(String),
}

impl Slot {
    /// What tells a buffer shared between forks from another.
    fn media_key(&self) -> Option<usize> {
        match self {
            Slot::Frames(frames) => frames.buffer(),
            Slot::Pcm(pcm) => Some(pcm.samples.as_ptr() as usize),
            _ => None,
        }
    }

    fn media_bytes(&self) -> usize {
        match self {
            Slot::Unborn(HandleKind::Frames(bytes)) => bytes.len(),
            Slot::Unborn(HandleKind::Matcher(
                GrammarSource::JsonSchema(text)
                | GrammarSource::Regex(text)
                | GrammarSource::Ebnf(text),
            )) => text.len(),
            Slot::Frames(frames) => frames.rgb8().map_or(0, <[u8]>::len),
            Slot::Pcm(pcm) => pcm.samples.len() * size_of::<f32>(),
            _ => 0,
        }
    }
}

/// What a name keeps through every move of what it holds.
#[derive(Debug)]
struct Meta {
    /// How the worker policy knows it, for its whole life.
    id: ProcessId,
    props: Props,
    /// The session of the launch that last had it.
    session: Option<String>,
}

impl Default for Meta {
    fn default() -> Self {
        Meta {
            id: Uuid::new_v4(),
            props: Props::new(),
            session: None,
        }
    }
}

/// Where a name stands once a process has ended.
#[derive(Debug)]
pub(crate) enum Landing {
    /// The process took it and gave nothing back under it.
    Gone,
    At(Slot),
}

/// Encoding is slow and the session has other frames to serve.
pub(crate) type Encode = Box<dyn FnOnce() -> Result<Vec<u8>, String> + Send>;

/// A session's handles. An actor owns it, so names resolve in the order the
/// client sent them.
#[derive(Debug)]
pub(crate) struct Table {
    client: ClientId,
    tenant: String,
    slots: HashMap<HandleName, Slot>,
    meta: HashMap<HandleName, Meta>,
    /// What each lent handle held: in a process, it is held all the same.
    lent_media: HashMap<HandleName, usize>,
    max_media_bytes: usize,
}

impl Default for Table {
    fn default() -> Self {
        Table::new(0, String::new())
    }
}

impl Table {
    pub(crate) fn new(client: ClientId, tenant: String) -> Self {
        Table {
            client,
            tenant,
            slots: HashMap::new(),
            meta: HashMap::new(),
            lent_media: HashMap::new(),
            max_media_bytes: MAX_MEDIA_BYTES,
        }
    }

    /// Its session has ended: what rests here is gone with it.
    pub(crate) fn closed(&self) {
        for meta in self.meta.values() {
            policy::report(Event::HandleGone(meta.id, By::Client));
        }
    }

    /// Tells the worker policy a working set under `name` has come to rest.
    fn rested(&self, name: HandleName) {
        let (Some(Slot::Parked(held)), Some(meta)) = (self.slots.get(&name), self.meta.get(&name))
        else {
            return;
        };
        let (tokens, folded, slots) = held.measure();
        policy::report(Event::HandleRested(
            meta.id,
            Resting {
                client: self.client,
                tenant: self.tenant.clone(),
                session: meta.session.clone(),
                props: meta.props.clone(),
                tokens,
                folded,
                slots,
            },
        ));
    }

    /// The worker policy dropped the working set it knows as `id`. One a
    /// process has by now is out of its reach: a policy ends that process.
    pub(crate) fn discard(&mut self, id: ProcessId, why: String) {
        let found = self.meta.iter().find_map(|(&name, meta)| {
            (meta.id == id && matches!(self.slots.get(&name), Some(Slot::Parked(_))))
                .then_some(name)
        });
        if let Some(name) = found {
            self.slots.insert(name, Slot::Dead(why));
            self.meta.remove(&name);
            policy::report(Event::HandleGone(id, By::Policy));
        }
    }

    pub(crate) fn lent(&self, name: HandleName) -> bool {
        matches!(self.slots.get(&name), Some(Slot::Lent))
    }

    pub(crate) async fn apply(&mut self, ops: Vec<HandleOp>) -> Vec<Value> {
        let mut answers = Vec::with_capacity(ops.len());
        for op in ops {
            answers.push(match self.op(op).await {
                Ok(answer) => answer,
                Err(error) => json!({ "error": error.to_string(), "code": error.code }),
            });
        }
        answers
    }

    async fn op(&mut self, op: HandleOp) -> Result<Value, HandleError> {
        match op {
            HandleOp::New { name, kind } => {
                self.claim(name)?;
                let slot = Slot::Unborn(kind);
                self.room_for(&slot)?;
                self.slots.insert(name, slot);
                self.meta.insert(name, Meta::default());
                Ok(json!({ "ok": true }))
            }
            HandleOp::Fork { name, from } => {
                self.claim(name)?;
                let meta = Meta::default();
                let child = match self.slots.get(&from) {
                    Some(Slot::Unborn(kind)) => {
                        let child = Slot::Unborn(kind.clone());
                        self.room_for(&child)?;
                        child
                    }
                    Some(Slot::Parked(held)) => {
                        let child = held.fork(meta.id).await.map_err(HandleError::moving)?;
                        Slot::Parked(Box::new(child))
                    }
                    Some(Slot::Matcher(matcher)) => {
                        Slot::Matcher(Box::new(Mutex::new(matcher.lock().unwrap().fork())))
                    }
                    Some(Slot::Frames(frames)) => Slot::Frames(frames.clone()),
                    Some(Slot::Pcm(pcm)) => Slot::Pcm(pcm.clone()),
                    other => return Err(refusal(from, other)),
                };
                self.slots.insert(name, child);
                self.meta.insert(name, meta);
                self.rested(name);
                Ok(json!({ "ok": true }))
            }
            HandleOp::Close { name } => match self.slots.remove(&name) {
                Some(_) => {
                    if let Some(meta) = self.meta.remove(&name) {
                        policy::report(Event::HandleGone(meta.id, By::Client));
                    }
                    Ok(json!({ "ok": true }))
                }
                None => Err(refusal(name, None)),
            },
            #[cfg(not(target_arch = "wasm32"))]
            HandleOp::Save { name, snapshot } => match self.slots.get(&name) {
                Some(Slot::Parked(held)) => {
                    held.claim().await.map_err(HandleError::moving)?;
                    let saved = super::snapshot::save(held, &self.tenant, &snapshot).await;
                    held.unclaim();
                    saved
                }
                Some(Slot::Unborn(_)) => Err(HandleError::invalid(format!(
                    "handle {name} holds nothing yet: a process has not bound it"
                ))),
                Some(Slot::Matcher(_) | Slot::Frames(_) | Slot::Pcm(_)) => Err(
                    HandleError::invalid(format!("handle {name} is not a working set")),
                ),
                other => Err(refusal(name, other)),
            },
            #[cfg(not(target_arch = "wasm32"))]
            HandleOp::Load { name, snapshot } => {
                self.claim(name)?;
                let meta = Meta::default();
                let held = super::snapshot::load(meta.id, &self.tenant, &snapshot).await?;
                self.slots.insert(name, Slot::Parked(Box::new(held)));
                self.meta.insert(name, meta);
                self.rested(name);
                Ok(json!({ "ok": true }))
            }
            #[cfg(not(target_arch = "wasm32"))]
            HandleOp::DeleteSnapshot { snapshot } => {
                super::snapshot::delete(&self.tenant, &snapshot).await
            }
            #[cfg(not(target_arch = "wasm32"))]
            HandleOp::ListSnapshots { prefix } => {
                super::snapshot::list(&self.tenant, &prefix).await
            }
            #[cfg(target_arch = "wasm32")]
            HandleOp::Save { .. }
            | HandleOp::Load { .. }
            | HandleOp::DeleteSnapshot { .. }
            | HandleOp::ListSnapshots { .. } => {
                Err(HandleError::invalid("this build keeps no snapshots"))
            }
            HandleOp::SetProps { name, props } => {
                let slot = self.slots.get(&name);
                if matches!(slot, None | Some(Slot::Dead(_) | Slot::Lent)) {
                    return Err(refusal(name, slot));
                }
                let parked = matches!(slot, Some(Slot::Parked(_)));
                let meta = self.meta.entry(name).or_default();
                let (merged, names) =
                    client_api::props::merge(&meta.props, props).map_err(HandleError::invalid)?;
                meta.props = merged.clone();
                if parked && !names.is_empty() {
                    policy::report(Event::HandleProps(meta.id, merged, names));
                }
                Ok(json!({ "ok": true }))
            }
            HandleOp::Stat { name } => match self.slots.get(&name) {
                Some(Slot::Unborn(HandleKind::WorkingSet)) => {
                    Ok(json!({ "state": "unborn", "tokens": 0, "folded": 0, "engine": null }))
                }
                Some(Slot::Unborn(_)) => Ok(json!({ "state": "unborn" })),
                Some(Slot::Frames(frames)) => Ok(json!({
                    "state": "parked",
                    "width": frames.width,
                    "height": frames.height,
                    "count": frames.count,
                    "fps": frames.fps,
                    "bytes": frames.rgb8().map_or(0, <[u8]>::len),
                })),
                Some(Slot::Pcm(pcm)) => Ok(json!({
                    "state": "parked",
                    "rate": pcm.rate,
                    "channels": pcm.channels,
                    "samples": pcm.samples.len(),
                })),
                Some(Slot::Parked(held)) => Ok(held.stat()),
                Some(Slot::Matcher(matcher)) => {
                    let terminated = matcher.lock().unwrap().is_terminated();
                    Ok(json!({ "state": "parked", "terminated": terminated }))
                }
                other => Err(refusal(name, other)),
            },
        }
    }

    /// The media `slot` would add within what the session may hold.
    fn room_for(&self, slot: &Slot) -> Result<(), HandleError> {
        // A fork shares what it forked: what is shared counts once.
        let mut seen = HashSet::new();
        let held: usize = self
            .slots
            .values()
            .chain(std::iter::once(slot))
            .filter(|slot| slot.media_key().is_none_or(|key| seen.insert(key)))
            .map(Slot::media_bytes)
            .sum::<usize>()
            + self.lent_media.values().sum::<usize>();
        if held > self.max_media_bytes {
            return Err(HandleError::quota(format!(
                "this would hold {held} bytes of media; close a handle"
            )));
        }
        Ok(())
    }

    /// What encodes a handle for a client, and the name of the file. It owns
    /// what it reads, which is never written, so it runs on while the table
    /// moves on.
    pub(crate) fn read(
        &self,
        name: HandleName,
        format: &str,
    ) -> Result<(Encode, String), HandleError> {
        use crate::inferlet::host::frames::{
            audio_extension, audio_format, image_extension, image_format,
        };
        match self.slots.get(&name) {
            Some(Slot::Frames(frames)) => {
                let format = image_format(format)
                    .ok_or_else(|| HandleError::invalid(format!("no image format `{format}`")))?;
                let frames = frames.clone();
                Ok((
                    Box::new(move || frames.encode(format)),
                    format!("{name}.{}", image_extension(format)),
                ))
            }
            Some(Slot::Pcm(pcm)) => {
                let format = audio_format(format)
                    .ok_or_else(|| HandleError::invalid(format!("no audio format `{format}`")))?;
                let pcm = pcm.clone();
                Ok((
                    Box::new(move || Ok(pcm.encode(format))),
                    format!("{name}.{}", audio_extension(format)),
                ))
            }
            other => Err(refusal(name, other)),
        }
    }

    fn claim(&self, name: HandleName) -> Result<(), HandleError> {
        self.claim_beside(name, 0)
    }

    /// `claim` for a launch that has claimed `also` other new names already.
    fn claim_beside(&self, name: HandleName, also: usize) -> Result<(), HandleError> {
        match self.slots.get(&name) {
            None if self.slots.len() + also >= MAX_HANDLES => Err(HandleError::quota(format!(
                "this session names {MAX_HANDLES} handles; close one"
            ))),
            None | Some(Slot::Dead(_)) => Ok(()),
            Some(_) => Err(HandleError::new(
                HandleErrorCode::InUse,
                format!("handle {name} is already in use"),
            )),
        }
    }

    /// Moves what a launch names out of the table. Nothing moves if it fails.
    pub(crate) fn lend(
        &mut self,
        session: ClientId,
        handles: Handles,
        returns: Handles,
        launch_session: Option<&str>,
    ) -> Result<Option<Lease>, HandleError> {
        if handles.is_empty() && returns.is_empty() {
            return Ok(None);
        }
        let mut seen = HashSet::new();
        for (slot, &name) in &handles {
            if !seen.insert(name) {
                return Err(HandleError::invalid(format!(
                    "handle {name} is bound twice, the second as `{slot}`"
                )));
            }
            let slot = self.slots.get(&name);
            if matches!(slot, None | Some(Slot::Dead(_) | Slot::Lent)) {
                return Err(refusal(name, slot));
            }
        }
        let mut engines = handles
            .values()
            .filter_map(|name| match self.slots.get(name) {
                Some(Slot::Parked(held)) => Some(held.parts().kv.engine),
                _ => None,
            });
        if let Some(first) = engines.next()
            && engines.any(|engine| engine != first)
        {
            return Err(HandleError::invalid("its handles are on different engines"));
        }
        let mut landing = HashSet::new();
        for (slot, &name) in &returns {
            if !landing.insert(name) {
                return Err(HandleError::invalid(format!(
                    "handle {name} is returned twice, the second as `{slot}`"
                )));
            }
            if !seen.contains(&name) {
                self.claim_beside(name, landing.len() - 1)?;
            }
        }
        let mut ids = HashMap::new();
        for &name in handles.values().chain(returns.values()) {
            let meta = self.meta.entry(name).or_default();
            meta.session = launch_session.map(str::to_string);
            ids.insert(name, meta.id);
        }
        let bound = handles
            .into_iter()
            .map(|(slot, name)| {
                let handle = self.slots.insert(name, Slot::Lent).expect("checked above");
                self.lent_media.insert(name, handle.media_bytes());
                policy::report(Event::HandleLent(ids[&name]));
                Bound { slot, name, handle }
            })
            .collect();
        for &name in returns.values() {
            self.slots.insert(name, Slot::Lent);
        }
        Ok(Some(Lease {
            session,
            bound,
            returns,
            ids,
        }))
    }

    /// What a process's end settled.
    pub(crate) fn settle(&mut self, settled: Vec<(HandleName, Landing)>) {
        for (name, landing) in settled {
            self.slots.remove(&name);
            self.lent_media.remove(&name);
            let Landing::At(slot) = landing else {
                self.meta.remove(&name);
                continue;
            };
            let slot = match self.room_for(&slot) {
                Ok(()) => slot,
                Err(why) => Slot::Dead(why.to_string()),
            };
            self.slots.insert(name, slot);
            self.rested(name);
        }
    }
}

fn refusal(name: HandleName, slot: Option<&Slot>) -> HandleError {
    match slot {
        None => HandleError::unknown(name),
        Some(Slot::Dead(why)) => {
            HandleError::new(HandleErrorCode::Dead, format!("handle {name}: {why}"))
        }
        Some(Slot::Lent) => HandleError::new(
            HandleErrorCode::InUse,
            format!("handle {name} is in a process"),
        ),
        Some(_) => HandleError::invalid(format!("handle {name} cannot do that")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inferlet::host::frames::Frames;

    fn names(pairs: &[(&str, HandleName)]) -> Handles {
        pairs
            .iter()
            .map(|&(slot, name)| (slot.into(), name))
            .collect()
    }

    fn new(name: HandleName) -> HandleOp {
        HandleOp::New {
            name,
            kind: HandleKind::WorkingSet,
        }
    }

    async fn answer(table: &mut Table, op: HandleOp) -> Value {
        table.apply(vec![op]).await.remove(0)
    }

    /// A process that ended: nobody hears, so these tests read the table.
    fn spent(lease: Lease) {
        let mut lease = lease;
        lease.bound.clear();
        lease.returns.clear();
    }

    #[tokio::test]
    async fn handle_every_case() {
        an_unborn_handle_is_a_name_and_nothing_else().await;
        a_session_names_only_so_many().await;
        media_past_the_session_limit_is_refused_or_dead().await;
        a_fork_counts_against_the_limit_as_what_it_shares().await;
        a_name_is_taken_once().await;
        one_bad_op_does_not_stop_the_rest().await;
        a_launch_moves_all_it_names_or_none().await;
        a_name_a_launch_promises_waits_like_one_it_took().await;
        a_settled_end_lands_each_name().await;
    }

    async fn a_session_names_only_so_many() {
        let mut table = Table::default();
        for name in 0..MAX_HANDLES as HandleName {
            table
                .slots
                .insert(name, Slot::Unborn(HandleKind::WorkingSet));
        }
        let over = answer(&mut table, new(MAX_HANDLES as HandleName)).await;
        assert!(over["error"].as_str().unwrap().starts_with("quota"));
        let fork = HandleOp::Fork {
            name: MAX_HANDLES as HandleName,
            from: 0,
        };
        assert!(
            answer(&mut table, fork).await["error"]
                .as_str()
                .unwrap()
                .starts_with("quota")
        );
        answer(&mut table, HandleOp::Close { name: 0 }).await;
        assert_eq!(
            answer(&mut table, new(MAX_HANDLES as HandleName)).await["ok"],
            true
        );
    }

    async fn media_past_the_session_limit_is_refused_or_dead() {
        let mut table = Table {
            max_media_bytes: 8,
            ..Table::default()
        };
        let upload = |name, len| HandleOp::New {
            name,
            kind: HandleKind::Frames(vec![0u8; len].into()),
        };
        assert_eq!(answer(&mut table, upload(1, 6)).await["ok"], true);
        let over = answer(&mut table, upload(2, 6)).await;
        assert!(over["error"].as_str().unwrap().starts_with("quota"));

        let frames = Frames::from_rgb8(vec![0; 12], 2, 2, 1, 0.0).unwrap();
        table.settle(vec![(3, Landing::At(Slot::Frames(Box::new(frames))))]);
        let landed = answer(&mut table, HandleOp::Stat { name: 3 }).await;
        assert!(
            landed["error"].as_str().unwrap().contains("quota"),
            "what a process gave past the limit is lost, and says so: {landed}"
        );
    }

    async fn a_fork_counts_against_the_limit_as_what_it_shares() {
        let mut table = Table {
            max_media_bytes: 16,
            ..Table::default()
        };
        let upload = HandleOp::New {
            name: 1,
            kind: HandleKind::Frames(vec![0u8; 10].into()),
        };
        assert_eq!(answer(&mut table, upload).await["ok"], true);
        let copy = answer(&mut table, HandleOp::Fork { name: 2, from: 1 }).await;
        assert!(
            copy["error"].as_str().unwrap().starts_with("quota"),
            "an upload's fork copies its bytes: {copy}"
        );

        assert_eq!(
            answer(&mut table, HandleOp::Close { name: 1 }).await["ok"],
            true
        );
        let decoded = Frames::from_rgb8(vec![0; 12], 2, 2, 1, 0.0).unwrap();
        table.settle(vec![(3, Landing::At(Slot::Frames(Box::new(decoded))))]);
        let fork = answer(&mut table, HandleOp::Fork { name: 4, from: 3 }).await;
        assert_eq!(fork["ok"], true, "decoded pixels are shared: {fork}");
        let held = Frames::from_rgb8(vec![0; 3], 1, 1, 1, 0.0).unwrap();
        table.settle(vec![(5, Landing::At(Slot::Frames(Box::new(held))))]);
        assert_eq!(
            answer(&mut table, HandleOp::Stat { name: 5 }).await["state"],
            "parked",
            "12 shared twice and 3 is 15, not 27"
        );
    }

    async fn an_unborn_handle_is_a_name_and_nothing_else() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let stat = answer(&mut table, HandleOp::Stat { name: 1 }).await;
        assert_eq!(stat["state"], "unborn");
        assert_eq!(stat["engine"], Value::Null);
        answer(&mut table, HandleOp::Fork { name: 2, from: 1 }).await;
        let child = answer(&mut table, HandleOp::Stat { name: 2 }).await;
        assert_eq!(child["state"], "unborn", "there is nothing to share yet");
        answer(&mut table, HandleOp::Close { name: 1 }).await;
        let gone = answer(&mut table, HandleOp::Stat { name: 1 }).await;
        assert!(gone["error"].as_str().unwrap().starts_with("unknown"));
    }

    async fn a_name_is_taken_once() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let again = answer(&mut table, new(1)).await;
        assert!(again["error"].as_str().unwrap().contains("in use"));
        let fork = answer(&mut table, HandleOp::Fork { name: 1, from: 1 }).await;
        assert!(fork["error"].as_str().unwrap().contains("in use"));
    }

    async fn one_bad_op_does_not_stop_the_rest() {
        let mut table = Table::default();
        let answers = table.apply(vec![HandleOp::Close { name: 9 }, new(1)]).await;
        assert!(answers[0].get("error").is_some());
        assert_eq!(answers[1]["ok"], true);
    }

    async fn a_launch_moves_all_it_names_or_none() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let refused = table.lend(7, names(&[("a", 1), ("b", 2)]), Handles::new(), None);
        assert!(refused.unwrap_err().to_string().starts_with("unknown"));
        assert!(!table.lent(1), "the known one stays");
        let twice = table.lend(7, names(&[("a", 1), ("b", 1)]), Handles::new(), None);
        assert!(twice.unwrap_err().to_string().contains("bound twice"));
        assert!(!table.lent(1));
        assert!(
            table
                .lend(7, Handles::new(), Handles::new(), None)
                .unwrap()
                .is_none()
        );
    }

    async fn a_name_a_launch_promises_waits_like_one_it_took() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let lease = table
            .lend(
                7,
                names(&[("ws", 1)]),
                names(&[("ws", 1), ("child", 2)]),
                None,
            )
            .unwrap()
            .unwrap();
        assert!(table.lent(1) && table.lent(2));
        assert_eq!(lease.engine(), None, "an unborn handle is on no engine");
        let in_use = table.lend(7, names(&[("ws", 1)]), Handles::new(), None);
        assert!(in_use.unwrap_err().to_string().contains("in a process"));
        spent(lease);
    }

    async fn a_settled_end_lands_each_name() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        answer(&mut table, new(2)).await;
        table.settle(vec![
            (1, Landing::Gone),
            (
                2,
                Landing::At(Slot::Dead("the process did not give `ws` back".into())),
            ),
        ]);
        let gone = answer(&mut table, HandleOp::Stat { name: 1 }).await;
        assert!(gone["error"].as_str().unwrap().starts_with("unknown"));
        let dead = answer(&mut table, HandleOp::Stat { name: 2 }).await;
        assert!(dead["error"].as_str().unwrap().contains("did not give"));
        let refused = table.lend(7, names(&[("ws", 2)]), Handles::new(), None);
        assert!(
            refused.unwrap_err().to_string().starts_with("dead"),
            "the reason stays"
        );
        answer(&mut table, new(2)).await;
        assert!(
            table
                .lend(7, names(&[("ws", 2)]), Handles::new(), None)
                .is_ok(),
            "a dead name is free"
        );
    }

    #[tokio::test]
    async fn props_merge_until_the_name_is_closed_and_a_fork_starts_with_none() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let set = |name, props| HandleOp::SetProps {
            name,
            props: serde_json::from_value(props).unwrap(),
        };
        answer(&mut table, set(1, json!({"a": 1, "b": 2}))).await;
        answer(&mut table, set(1, json!({"a": null}))).await;
        assert_eq!(
            serde_json::to_value(&table.meta[&1].props.0).unwrap(),
            json!({"b": 2})
        );

        answer(&mut table, HandleOp::Fork { name: 2, from: 1 }).await;
        assert!(table.meta[&2].props.is_empty());
        assert_ne!(table.meta[&1].id, table.meta[&2].id);

        answer(&mut table, HandleOp::Close { name: 1 }).await;
        assert!(!table.meta.contains_key(&1));
        let gone = answer(&mut table, set(1, json!({"a": 1}))).await;
        assert!(
            gone["error"].as_str().unwrap().starts_with("unknown"),
            "{gone}"
        );
    }

    #[tokio::test]
    async fn a_policy_reaches_a_handle_only_at_rest() {
        let mut table = Table::default();
        answer(&mut table, new(1)).await;
        let id = table.meta[&1].id;
        table.discard(id, "gone".to_string());
        assert!(
            matches!(table.slots.get(&1), Some(Slot::Unborn(_))),
            "nothing was there to drop"
        );
    }
}
