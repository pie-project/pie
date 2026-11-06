//! What a session names, and each name's state.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use client_api::{HandleErrorCode, HandleKind, HandleName, HandleOp, Handles};
use serde_json::{Value, json};

use super::held::Held;
use super::lease::{Bound, Lease};
use super::{HandleError, MAX_HANDLES, MAX_MEDIA_BYTES};
use crate::inferlet::host::frames::{Frames, Pcm};
use crate::inferlet::host::grammar::Matcher;
use crate::server::ClientId;
use crate::store::registry as store_registry;

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
            Slot::Frames(frames) => frames.rgb8().map_or(0, <[u8]>::len),
            Slot::Pcm(pcm) => pcm.samples.len() * size_of::<f32>(),
            _ => 0,
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
    slots: HashMap<HandleName, Slot>,
    max_media_bytes: usize,
}

impl Default for Table {
    fn default() -> Self {
        Table {
            slots: HashMap::new(),
            max_media_bytes: MAX_MEDIA_BYTES,
        }
    }
}

impl Table {
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
                Ok(json!({ "ok": true }))
            }
            HandleOp::Fork { name, from } => {
                self.claim(name)?;
                let child = match self.slots.get(&from) {
                    Some(Slot::Unborn(kind)) => {
                        let child = Slot::Unborn(kind.clone());
                        self.room_for(&child)?;
                        child
                    }
                    Some(Slot::Parked(held)) => {
                        let child = held.fork().await.map_err(HandleError::moving)?;
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
                Ok(json!({ "ok": true }))
            }
            HandleOp::Close { name } => match self.slots.remove(&name) {
                Some(_) => Ok(json!({ "ok": true })),
                None => Err(refusal(name, None)),
            },
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
            .sum();
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
        match self.slots.get(&name) {
            None if self.slots.len() >= MAX_HANDLES => Err(HandleError::quota(format!(
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
        state_room: Option<u32>,
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
        let growing = handles
            .values()
            .filter(|name| match self.slots.get(name) {
                Some(Slot::Unborn(HandleKind::WorkingSet)) => true,
                Some(Slot::Parked(held)) => held.shares_state(),
                _ => false,
            })
            .count();
        if state_room.is_some_and(|room| (room as usize) < growing) {
            return Err(HandleError::quota(format!(
                "{growing} working set(s) taking new state would leave more recurrent state at \
                 rest than the host holds; close a handle, or raise `[engine] max_state_slots`"
            )));
        }
        let mut landing = HashSet::new();
        for (slot, &name) in &returns {
            if !landing.insert(name) {
                return Err(HandleError::invalid(format!(
                    "handle {name} is returned twice, the second as `{slot}`"
                )));
            }
            if !seen.contains(&name) {
                self.claim(name)?;
            }
        }
        let bound = handles
            .into_iter()
            .map(|(slot, name)| Bound {
                slot,
                name,
                handle: self.slots.insert(name, Slot::Lent).expect("checked above"),
            })
            .collect();
        for &name in returns.values() {
            self.slots.insert(name, Slot::Lent);
        }
        Ok(Some(Lease {
            session,
            bound,
            returns,
        }))
    }

    /// What a process's end settled.
    pub(crate) fn settle(&mut self, settled: Vec<(HandleName, Landing)>) {
        for (name, landing) in settled {
            self.slots.remove(&name);
            let Landing::At(slot) = landing else {
                continue;
            };
            let slot = match self.room_for(&slot) {
                Ok(()) => slot,
                Err(why) => Slot::Dead(why.to_string()),
            };
            self.slots.insert(name, slot);
        }
    }
}

/// How many more sets can come to rest on the most open engine, where it has
/// recurrent state to hold. That state has no disk tier and a running lane
/// may want every device slot, so what rests must fit the host: past that, a
/// fire would starve on state nothing can move.
pub(crate) fn state_room() -> Option<u32> {
    store_registry::all_for_model(0)
        .iter()
        .filter_map(|stores| {
            let rs = stores.rs.lock().unwrap();
            let device = rs.capacity().slots;
            (device > 0).then(|| {
                rs.available()
                    .slots
                    .saturating_add(rs.host_available().slots)
                    .saturating_sub(device)
            })
        })
        .max()
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
        let full = table.lend(7, names(&[("a", 1)]), Handles::new(), Some(0));
        assert!(full.unwrap_err().to_string().starts_with("quota"));
        assert!(!table.lent(1), "a full host moves nothing");
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
}
