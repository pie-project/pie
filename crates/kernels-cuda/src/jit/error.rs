#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    NoDevice,
    Compile {
        unit: &'static str,
        why: String,
    },
    Driver {
        what: &'static str,
        code: i32,
        why: String,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => {
                write!(
                    f,
                    "no CUDA device is current, so no architecture could be discovered"
                )
            }
            Self::Compile { unit, why } => write!(f, "`{unit}` would not compile: {why}"),
            Self::Driver { what, code, why } => write!(f, "{what} failed with {code} ({why})"),
        }
    }
}

impl std::error::Error for Error {}
