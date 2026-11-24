use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Default)]
pub struct Airborne {
    counts: Arc<Counts>,
}

#[derive(Debug, Default)]
struct Counts {
    issued: AtomicU64,
    settled: AtomicU64,
}

impl Airborne {
    #[must_use]
    pub fn new() -> Airborne {
        Airborne::default()
    }

    pub fn enter(&self) -> u64 {
        self.counts.issued.fetch_add(1, Ordering::AcqRel)
    }

    /// The steps entered so far: the sequence the next one takes.
    #[must_use]
    pub fn entered(&self) -> u64 {
        self.counts.issued.load(Ordering::Acquire)
    }

    pub fn leave(&self) {
        self.counts.settled.fetch_add(1, Ordering::Release);
    }

    pub fn abandon(&self) {
        self.leave();
    }

    #[must_use]
    pub fn count(&self) -> u64 {
        self.counts
            .issued
            .load(Ordering::Acquire)
            .saturating_sub(self.counts.settled.load(Ordering::Acquire))
    }

    #[must_use]
    pub fn settled_past(&self, seq: u64) -> bool {
        self.counts.settled.load(Ordering::Acquire) > seq
    }
}

pub struct Done {
    pub at: engine::StepDone,

    pub sink: engine::CompletionSink,

    /// What the step's lanes report when it settles: the slots each fold
    /// froze, on a fire whose fold length is the device's.
    pub wrote: Vec<engine::KvWrites>,

    /// The step's lanes: every one faulted where the step is lost whole.
    pub lanes: u32,

    /// The lanes whose guest passes faulted ahead of the forward (a
    /// prologue, an envelope): told with the step however it lands.
    pub faulted: Vec<engine::LaneFault>,

    /// Where the step's guest passes run on the host behind its forward:
    /// the step is told once its forward landed and they ran, with the
    /// lanes they faulted.
    pub meet: Option<Meet>,

    /// The step's device-fed token ids, checked once its forward landed.
    pub fed: Option<FedCheck>,
}

impl Done {
    /// The outcome of a step whose forward landed: what its lanes report,
    /// and the lanes its guests faulted ahead of it.
    #[must_use]
    pub fn settled(&self) -> engine::StepOutcome {
        let outcome = engine::StepOutcome {
            beam_common: Vec::new(),
            wrote: self.wrote.clone(),
            faulted: self.faulted.clone(),
        };
        match &self.fed {
            Some(fed) => engine::StepOutcome {
                beam_common: fed.commons(),
                ..merged(outcome, fed.faults())
            },
            None => outcome,
        }
    }

    /// Hands `then` the step and its settled outcome once the forward's
    /// device-fed ids are checked: at once where it has none or they were,
    /// else from the forward's landing. Unlike [`Done::settled`] it never
    /// waits, so a landing that may run before the forward's (a deferred
    /// guest frame's, on the thread that polls the device) asks this.
    pub fn when_settled(self, then: impl FnOnce(Done, engine::StepOutcome) + Send + 'static) {
        let outcome = engine::StepOutcome {
            wrote: self.wrote.clone(),
            faulted: self.faulted.clone(),
        };
        match self.fed.clone() {
            None => then(self, outcome),
            Some(fed) => fed.then(move |faults| then(self, merged(outcome, faults))),
        }
    }

    /// Tells the sink the step's forward landed: whole, or lost for
    /// `refused`, `what` naming what was refused. Where its guest passes
    /// run behind it on the host, the step is told once they ran too.
    pub fn tell(&self, refused: Option<String>, what: &str) {
        let outcome = match refused {
            None => self.settled(),
            Some(why) => engine::StepOutcome::lost(
                self.lanes,
                format!(
                    "{what} for frame {} step {}: {why}",
                    self.at.frame, self.at.step
                ),
            ),
        };
        match &self.meet {
            None => (self.sink)(self.at, outcome),
            Some(meet) => meet.forward_landed(self, outcome),
        }
    }
}

impl std::fmt::Debug for Done {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Done").field("at", &self.at).finish()
    }
}

/// Where a step's forward landing and its host guest passes meet: whichever
/// comes second tells the sink the step, its lanes the guests faulted
/// beside what the forward says.
#[derive(Clone, Default)]
pub struct Meet {
    inner: Arc<std::sync::Mutex<Met>>,
}

#[derive(Default)]
struct Met {
    forward: Option<(engine::StepDone, engine::CompletionSink, engine::StepOutcome)>,
    guests: Option<Vec<engine::LaneFault>>,
}

impl Meet {
    #[must_use]
    pub fn new() -> Meet {
        Meet::default()
    }

    /// The step's guest passes ran on the host, faulting `faulted`.
    pub fn guests_ran(&self, faulted: Vec<engine::LaneFault>) {
        let told = {
            let mut met = self.inner.lock().expect("meet lock poisoned");
            match met.forward.take() {
                Some(forward) => Some((forward, faulted)),
                None => {
                    met.guests = Some(faulted);
                    None
                }
            }
        };
        if let Some(((at, sink, outcome), faulted)) = told {
            sink(at, merged(outcome, faulted));
        }
    }

    fn forward_landed(&self, done: &Done, outcome: engine::StepOutcome) {
        let told = {
            let mut met = self.inner.lock().expect("meet lock poisoned");
            match met.guests.take() {
                Some(faulted) => Some(merged(outcome, faulted)),
                None => {
                    met.forward = Some((done.at, done.sink.clone(), outcome));
                    None
                }
            }
        };
        if let Some(outcome) = told {
            (done.sink)(done.at, outcome);
        }
    }
}

/// A step's lanes fed their one token on the device (a cell a guest pass
/// still on the queue wrote): the ids the forward embedded are read back
/// from the arm's token seat once the forward landed, before the arm is
/// given back, and a lane whose id is past the vocabulary faults by name,
/// as one the host resolved does ahead of the forward.
#[derive(Clone)]
pub struct FedCheck {
    inner: Arc<Fed>,
}

/// A device-fed lane: its row in the fire's token seat, and the lanes and
/// instance of the attachment it is of.
#[derive(Debug, Clone)]
pub struct FedLane {
    pub row: u32,
    pub lanes: std::ops::Range<u32>,
    pub instance: Option<engine::InstanceId>,
}

/// Reads the id at token seat row `row`.
pub type ReadRow = Box<dyn Fn(u32) -> Option<u32> + Send + Sync>;

/// A beam's fault word (`crate::beam`): its u32 word in the fire's input
/// store, the lane it faults and its instance.
#[derive(Debug, Clone)]
pub struct BeamWord {
    pub word: u32,
    pub lane: u32,
    pub instance: engine::InstanceId,
}

struct Fed {
    read: ReadRow,
    lanes: Vec<FedLane>,
    vocab: u32,
    /// Reads the store's u32 word, and each beam's fault word.
    read_word: Option<ReadRow>,
    beams: Vec<BeamWord>,
    /// Each beam group's common slot word (§11.3) and its instance.
    commons: Vec<(u32, engine::InstanceId)>,
    /// The store's words from a first one as the fire's readout brought
    /// them back ([`FedCheck::supply`]): read in place of the store.
    supplied: std::sync::Mutex<Option<(u32, Vec<u32>)>>,
    /// The groups' common slots the landing read.
    read_commons: std::sync::Mutex<Vec<(engine::InstanceId, u32)>>,
    faults: std::sync::Mutex<FedState>,
    ready: std::sync::Condvar,
}

/// What a fed check knows: the lanes it faulted once the forward landed,
/// and, until then, who is told them when it does.
#[derive(Default)]
struct FedState {
    faults: Option<Vec<engine::LaneFault>>,
    told: Vec<Told>,
}

/// One told the fed check's faults once the forward landed.
type Told = Box<dyn FnOnce(Vec<engine::LaneFault>) + Send>;

impl FedCheck {
    #[must_use]
    pub fn new(read: ReadRow, lanes: Vec<FedLane>, vocab: u32) -> FedCheck {
        FedCheck {
            inner: Arc::new(Fed {
                read,
                lanes,
                vocab,
                read_word: None,
                beams: Vec::new(),
                commons: Vec::new(),
                supplied: std::sync::Mutex::new(None),
                read_commons: std::sync::Mutex::new(Vec::new()),
                faults: std::sync::Mutex::new(FedState::default()),
                ready: std::sync::Condvar::new(),
            }),
        }
    }

    /// A check of `lanes`' fed ids, as [`FedCheck::new`], and of each of
    /// `beams`' fault words, read by `read_word`: a nonzero word faults its
    /// lane by name (`crate::beam::fault_reason`).
    #[must_use]
    pub fn with_beams(
        read: ReadRow,
        lanes: Vec<FedLane>,
        vocab: u32,
        read_word: ReadRow,
        beams: Vec<BeamWord>,
        commons: Vec<(u32, engine::InstanceId)>,
    ) -> FedCheck {
        FedCheck {
            inner: Arc::new(Fed {
                read,
                lanes,
                vocab,
                read_word: Some(read_word),
                beams,
                commons,
                supplied: std::sync::Mutex::new(None),
                read_commons: std::sync::Mutex::new(Vec::new()),
                faults: std::sync::Mutex::new(FedState::default()),
                ready: std::sync::Condvar::new(),
            }),
        }
    }

    /// The token seat rows the check reads: up to the last fed lane's.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.inner
            .lanes
            .iter()
            .map(|lane| lane.row + 1)
            .max()
            .unwrap_or(0)
    }

    /// The forward landed (`whole`), or was lost: reads the fed ids back,
    /// once. A lost step faults every lane by its own loss, not here.
    pub fn landed(&self, whole: bool) {
        let fed = &*self.inner;
        let mut state = fed.faults.lock().expect("fed lock poisoned");
        if state.faults.is_some() {
            return;
        }
        let mut outcome = engine::StepOutcome::committed();
        for lane in fed.lanes.iter().filter(|_| whole) {
            let Some(id) = (fed.read)(lane.row) else {
                continue;
            };
            if id >= fed.vocab {
                let why = match lane.instance {
                    Some(instance) => format!(
                        "instance {instance}: token {id} is out of range for a vocabulary of {}",
                        fed.vocab
                    ),
                    None => format!(
                        "token {id} is out of range for a vocabulary of {}",
                        fed.vocab
                    ),
                };
                tracing::warn!(row = lane.row, %why, "a device-fed token is out of range; its lanes fault");
                outcome.fault_lanes(lane.lanes.clone(), lane.instance, &why);
            }
        }
        let supplied = fed.supplied.lock().expect("fed lock poisoned").take();
        if let Some(read_word) = fed.read_word.as_ref().filter(|_| whole) {
            let word = |at: u32| match &supplied {
                Some((first, words)) => at
                    .checked_sub(*first)
                    .and_then(|k| words.get(k as usize).copied()),
                None => read_word(at),
            };
            *fed.read_commons.lock().expect("fed lock poisoned") = fed
                .commons
                .iter()
                .filter_map(|&(at, instance)| Some((instance, word(at)?)))
                .collect();
            for beam in &fed.beams {
                let code = word(beam.word).unwrap_or(0);
                if code == 0 {
                    continue;
                }
                let why = crate::beam::fault_reason(code, beam.lane as usize);
                tracing::warn!(lane = beam.lane, instance = beam.instance.0, %why, "a beam lane faults");
                outcome.fault_lanes(beam.lane..beam.lane + 1, Some(beam.instance), &why);
            }
        }
        state.faults = Some(outcome.faulted.clone());
        let told = std::mem::take(&mut state.told);
        drop(state);
        fed.ready.notify_all();
        for tell in told {
            tell(outcome.faulted.clone());
        }
    }

    /// Tells `then` the lanes the fed ids faulted: now where the forward
    /// landed, else from its landing. Nothing waits here, so a landing
    /// callback (a guest frame's, on the thread that polls the device) may
    /// ask it without parking the thread the forward's own landing needs.
    pub fn then(&self, then: impl FnOnce(Vec<engine::LaneFault>) + Send + 'static) {
        let mut state = self.inner.faults.lock().expect("fed lock poisoned");
        match state.faults.clone() {
            Some(faults) => {
                drop(state);
                then(faults);
            }
            None => state.told.push(Box::new(then)),
        }
    }

    /// The store's words from `first` as the fire's readout brought them
    /// back: [`FedCheck::landed`] reads the beams' words there in place of
    /// the store.
    pub fn supply(&self, first: u32, words: Vec<u32>) {
        *self.inner.supplied.lock().expect("fed lock poisoned") = Some((first, words));
    }

    /// Each beam group's common slot (§11.3) as the landing read it, once
    /// the forward landed.
    #[must_use]
    pub fn commons(&self) -> Vec<(engine::InstanceId, u32)> {
        let _ = self.faults();
        self.inner
            .read_commons
            .lock()
            .expect("fed lock poisoned")
            .clone()
    }

    /// Whether the check reads beams' fault words.
    #[must_use]
    pub fn reads_beams(&self) -> bool {
        !self.inner.beams.is_empty()
    }

    /// The lanes the fed ids faulted, once the forward landed.
    #[must_use]
    pub fn faults(&self) -> Vec<engine::LaneFault> {
        let fed = &*self.inner;
        let mut state = fed.faults.lock().expect("fed lock poisoned");
        while state.faults.is_none() {
            state = fed.ready.wait(state).expect("fed lock poisoned");
        }
        state.faults.clone().unwrap_or_default()
    }

    /// Waits until the forward's landing read the fed ids back: the arm's
    /// token seat may be written again only then.
    pub fn wait(&self) {
        let _ = self.faults();
    }
}

impl std::fmt::Debug for FedCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FedCheck").field("lanes", &self.inner.lanes).finish()
    }
}

/// `outcome` with the lanes `faulted` faulted beside its own; a lane it
/// faulted already keeps its own reason.
#[must_use]
pub fn merged(
    mut outcome: engine::StepOutcome,
    faulted: Vec<engine::LaneFault>,
) -> engine::StepOutcome {
    for fault in faulted {
        if outcome.faulted.iter().all(|had| had.lane != fault.lane) {
            outcome.faulted.push(fault);
        }
    }
    outcome
}

#[derive(Debug, Clone)]
pub struct Arms {
    taken: Vec<bool>,
}

impl Arms {
    #[must_use]
    pub fn of(depth: usize) -> Arms {
        Arms {
            taken: vec![false; depth.max(1)],
        }
    }

    #[must_use]
    pub fn depth(&self) -> usize {
        self.taken.len()
    }

    #[must_use]
    pub fn free(&self) -> Option<usize> {
        self.taken.iter().position(|held| !held)
    }

    pub fn take(&mut self, at: usize) {
        if let Some(held) = self.taken.get_mut(at) {
            *held = true;
        }
    }

    pub fn give(&mut self, at: usize) {
        if let Some(held) = self.taken.get_mut(at) {
            *held = false;
        }
    }

    #[must_use]
    pub fn held(&self) -> usize {
        self.taken.iter().filter(|held| **held).count()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{Done, Meet};

    type Heard = Arc<Mutex<Vec<engine::StepOutcome>>>;

    fn done(meet: &Meet, heard: &Heard) -> Done {
        let into = Arc::clone(heard);
        Done {
            at: engine::StepDone {
                frame: engine::FrameId(1),
                step: 0,
            },
            sink: Arc::new(move |_, outcome| into.lock().unwrap().push(outcome)),
            wrote: vec![engine::KvWrites {
                tag: 7,
                ..Default::default()
            }],
            lanes: 4,
            faulted: Vec::new(),
            meet: Some(meet.clone()),
            fed: None,
        }
    }

    fn guest_fault(lane: u32) -> Vec<engine::LaneFault> {
        let mut outcome = engine::StepOutcome::committed();
        outcome.fault_lanes(
            lane..lane + 1,
            Some(engine::InstanceId(9)),
            "instance 9 (program 1) faulted: guard",
        );
        outcome.faulted
    }

    /// A step whose guest pass faulted one lane of four on the host is
    /// told once, after both its forward landed and its guests ran,
    /// whichever comes first: that lane faulted, the rest committed with
    /// what they wrote.
    #[test]
    fn a_guest_fault_faults_its_own_lane_whichever_lands_first() {
        for guests_first in [false, true] {
            let heard = Heard::default();
            let meet = Meet::new();
            let done = done(&meet, &heard);
            if guests_first {
                meet.guests_ran(guest_fault(2));
                assert!(heard.lock().unwrap().is_empty(), "the forward is still out");
                done.tell(None, "test buffer");
            } else {
                done.tell(None, "test buffer");
                assert!(heard.lock().unwrap().is_empty(), "the guests have not run");
                meet.guests_ran(guest_fault(2));
            }
            let heard = heard.lock().unwrap();
            assert_eq!(heard.len(), 1, "told once");
            let outcome = &heard[0];
            assert_eq!(outcome.wrote.len(), 1, "the committed lanes' reports stand");
            assert_eq!(outcome.faulted.len(), 1, "one lane faulted: {outcome:?}");
            assert_eq!(outcome.faulted[0].lane, 2);
            assert_eq!(outcome.faulted[0].instance, Some(engine::InstanceId(9)));
            assert!(outcome.faulted[0].reason.contains("guard"));
        }
    }

    /// A device-fed lane whose embedded id is past the vocabulary faults its
    /// own attachment's lanes by name once the forward landed; the other
    /// fed lanes, in range, fault nothing.
    #[test]
    fn a_device_fed_id_past_the_vocabulary_faults_its_lanes_alone() {
        use super::{FedCheck, FedLane};
        let seat = [5u32, 7, 151_954, 9];
        let lanes = (0..4u32)
            .map(|row| FedLane {
                row,
                lanes: row..row + 1,
                instance: Some(engine::InstanceId(100 + u64::from(row))),
            })
            .collect();
        let fed = FedCheck::new(
            Box::new(move |row| seat.get(row as usize).copied()),
            lanes,
            151_936,
        );
        let heard = Heard::default();
        let meet = Meet::new();
        let mut done = done(&meet, &heard);
        done.meet = None;
        done.fed = Some(fed.clone());
        let landing = std::thread::spawn(move || fed.landed(true));
        landing.join().expect("the landing");
        done.tell(None, "test buffer");
        let heard = heard.lock().unwrap();
        let faulted: Vec<(u32, Option<engine::InstanceId>)> = heard[0]
            .faulted
            .iter()
            .map(|fault| (fault.lane, fault.instance))
            .collect();
        assert_eq!(faulted, [(2, Some(engine::InstanceId(102)))]);
        assert_eq!(
            heard[0].faulted[0].reason,
            "instance 102: token 151954 is out of range for a vocabulary of 151936"
        );
        assert_eq!(heard[0].wrote.len(), 1, "the other lanes' reports stand");
    }

    /// A step told from a landing that may come before its forward's (a
    /// deferred guest frame's, on the thread that polls the device) is told
    /// once the forward's fed ids are read, by that read, never by parking
    /// for it: the thread asking is the one the read needs. Asked after the
    /// read, it is told at once.
    #[test]
    fn a_step_waits_for_its_fed_ids_without_parking_the_asker() {
        use super::{FedCheck, FedLane};
        for fed_first in [false, true] {
            let fed = FedCheck::new(
                Box::new(|row| Some(if row == 1 { 99 } else { 3 })),
                (0..2u32)
                    .map(|row| FedLane {
                        row,
                        lanes: row..row + 1,
                        instance: None,
                    })
                    .collect(),
                16,
            );
            let heard = Heard::default();
            let meet = Meet::new();
            let mut done = done(&meet, &heard);
            done.meet = None;
            done.fed = Some(fed.clone());
            if fed_first {
                fed.landed(true);
            }
            let into = Arc::clone(&heard);
            done.when_settled(move |done, outcome| {
                assert_eq!(done.at.step, 0);
                into.lock().unwrap().push(outcome);
            });
            if !fed_first {
                assert!(heard.lock().unwrap().is_empty(), "the fed ids are unread");
                fed.landed(true);
            }
            let heard = heard.lock().unwrap();
            assert_eq!(heard.len(), 1, "told once");
            let lanes: Vec<u32> = heard[0].faulted.iter().map(|fault| fault.lane).collect();
            assert_eq!(lanes, [1], "the out-of-range id faults its lane");
            assert_eq!(heard[0].wrote.len(), 1, "the step's reports stand");
        }
    }

    /// A forward the queue refused loses the whole step: every lane
    /// faulted, a guest's own fault on a lane not counted twice.
    #[test]
    fn a_refused_forward_faults_every_lane_once() {
        let heard = Heard::default();
        let meet = Meet::new();
        let done = done(&meet, &heard);
        done.tell(Some("device lost".into()), "test buffer");
        meet.guests_ran(guest_fault(1));
        let heard = heard.lock().unwrap();
        let lanes: Vec<u32> = heard[0].faulted.iter().map(|fault| fault.lane).collect();
        assert_eq!(lanes, [0, 1, 2, 3]);
        assert!(heard[0].faulted[1].reason.contains("device lost"));
    }
}
