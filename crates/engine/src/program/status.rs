#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Status {
    pub state: u32,

    pub fault: u32,

    pub reserved0: u32,

    pub reserved1: u32,
}

pub const STATUS_BYTES: usize = 16;

impl Status {
    #[must_use]
    pub fn read(bytes: &[u8]) -> Option<Status> {
        if bytes.len() < STATUS_BYTES {
            return None;
        }
        let word = |at: usize| -> u32 {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        Some(Status {
            state: word(0),
            fault: word(4),
            reserved0: word(8),
            reserved1: word(12),
        })
    }

    #[must_use]
    pub fn state(self) -> Option<State> {
        match self.state {
            0 => Some(State::Unset),
            1 => Some(State::Running),
            2 => Some(State::Retry),
            3 => Some(State::Fault),
            4 => Some(State::Committed),
            _ => None,
        }
    }

    #[must_use]
    pub fn site(self) -> Site {
        match self.reserved1 >> 24 {
            1 => Site::SinkNarrowerThanValue,
            2 => Site::MtpDraftsZeroRowWidth,
            3 => Site::NoArmClaimedTheTag,
            other => Site::Unknown(other),
        }
    }

    #[must_use]
    pub fn immediate(self) -> u32 {
        self.reserved1 & 0x00ff_ffff
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Unset,

    Running,

    Retry,

    Fault,

    Committed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    SinkNarrowerThanValue,

    MtpDraftsZeroRowWidth,

    NoArmClaimedTheTag,

    Unknown(u32),
}

impl Site {
    #[must_use]
    pub fn describe(self) -> Option<&'static str> {
        match self {
            Site::SinkNarrowerThanValue => Some("channel sink narrower than the value"),
            Site::MtpDraftsZeroRowWidth => Some("MtpDrafts with a zero row width"),
            Site::NoArmClaimedTheTag => Some("no arm claimed this op tag"),
            Site::Unknown(_) => None,
        }
    }
}

pub use tensor_compiler::codegen::fault::{CLASSES as FAULT_CLASSES, FaultClass};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    pub class: Option<&'static str>,

    pub channel: Option<u32>,

    pub ambiguous_with_op_tag: bool,
}

#[must_use]
pub fn describe_fault(fault: u32, max_channel: u32) -> Fault {
    let mut found: Option<(&'static FaultClass, u32)> = None;
    for class in FAULT_CLASSES {
        let span = if class.per_channel { max_channel } else { 0 };
        if fault >= class.base && fault - class.base <= span {
            found = Some((class, fault - class.base));
        }
    }
    match found {
        Some((class, offset)) => Fault {
            class: Some(class.name),
            channel: class.per_channel.then_some(offset),
            ambiguous_with_op_tag: class.base < 0x100,
        },
        None => Fault {
            class: None,
            channel: None,

            ambiguous_with_op_tag: false,
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Committed,

    Retry,

    Failed,
}

impl Outcome {
    #[must_use]
    pub fn of(status: Status, dispatched: bool) -> (Outcome, Diagnosis) {
        if !dispatched {
            return (Outcome::Failed, Diagnosis::NeverDispatched);
        }
        match status.state() {
            Some(State::Committed) => (Outcome::Committed, Diagnosis::Committed),
            Some(State::Retry) => (Outcome::Retry, Diagnosis::ReadinessUnmet),
            Some(State::Fault) => (Outcome::Failed, Diagnosis::Faulted),
            Some(State::Unset) => (Outcome::Failed, Diagnosis::NeverWritten),
            Some(State::Running) => (Outcome::Failed, Diagnosis::NeverFinished),
            None => (Outcome::Failed, Diagnosis::UnknownState(status.state)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diagnosis {
    Committed,

    ReadinessUnmet,

    Faulted,

    NeverDispatched,

    NeverWritten,

    NeverFinished,

    UnknownState(u32),
}

impl Diagnosis {
    #[must_use]
    pub fn describe(self) -> &'static str {
        match self {
            Diagnosis::Committed => "committed",
            Diagnosis::ReadinessUnmet => "a readiness guard in the kernel was unmet",
            Diagnosis::Faulted => "a guard inside the kernel refused",
            Diagnosis::NeverDispatched => {
                "the command was prepared but never encoded, so no kernel ran"
            }
            Diagnosis::NeverWritten => {
                "the command was submitted and the kernel wrote no status at all"
            }
            Diagnosis::NeverFinished => "the kernel started and did not reach its end",
            Diagnosis::UnknownState(_) => "the kernel wrote a status this engine cannot read",
        }
    }
}

#[must_use]
pub fn report(status: Status, dispatched: bool, max_channel: u32) -> String {
    let (_, diagnosis) = Outcome::of(status, dispatched);
    if !dispatched {
        return diagnosis.describe().to_string();
    }
    if matches!(diagnosis, Diagnosis::Committed) {
        return "committed".to_string();
    }
    let fault = describe_fault(status.fault, max_channel);
    let mut line = format!("{}: fault 0x{:x}", diagnosis.describe(), status.fault);
    if let Some(class) = fault.class {
        line.push_str(" (");
        line.push_str(class);
        if let Some(channel) = fault.channel {
            line.push_str(&format!(" channel {channel}"));
        }
        if fault.ambiguous_with_op_tag {
            line.push_str(", or the op tag of the same value");
        }
        line.push(')');
    }
    if let Some(site) = status.site().describe() {
        line.push_str(&format!(
            " intr={} imm={} guard={site}",
            status.reserved0,
            status.immediate()
        ));
    }
    line
}
