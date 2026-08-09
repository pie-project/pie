/// Why a fire did not happen.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// No unit hosts the symbol.
    Unknown {
        /// The symbol that was asked for. Owned, because it came from a
        symbol: String,
    },
    /// No CUDA device is current, so no architecture could be discovered.
    NoDevice,
    /// The unit would not compile.
    Compile {
        /// The unit that failed, by the name NVRTC calls it in diagnostics.
        unit: &'static str,
        /// What the compiler said.
        why: String,
    },
    /// The cubin loaded but the row's entry is not in it.
    Missing {
        /// The unit whose image was searched.
        unit: &'static str,
        /// The row that named the entry.
        symbol: &'static str,
    },
    /// The row's launch rule could not be evaluated over these dims.
    Geometry {
        /// The row whose rule was evaluated.
        symbol: &'static str,
        /// Which way it did not work out.
        why: crate::runtime::Ungeometric,
    },
    /// The values did not match the row.
    Args(crate::runtime::ArgError),
    /// The row's arm could not be chosen or could not be filled.
    Specialise {
        /// The row whose specialisation did not work out — the base symbol
        symbol: &'static str,
        /// What did not line up.
        why: String,
    },
    /// The driver refused.
    Driver {
        /// The call that failed, spelled as CUDA spells it.
        what: &'static str,
        /// Its `CUresult`, as an integer.
        code: i32,
        /// Its `CUresult`, as a name.
        why: String,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unknown { symbol } => write!(f, "no unit hosts `{symbol}`"),
            Error::NoDevice => {
                write!(f, "no CUDA device is current, so no architecture could be discovered")
            }
            Error::Compile { unit, why } => write!(f, "`{unit}` would not compile: {why}"),
            Error::Missing { unit, symbol } => {
                write!(f, "`{unit}` compiled and its image has no entry for row `{symbol}`")
            }
            Error::Geometry { symbol, why } => {
                write!(f, "`{symbol}` states a launch these dims cannot satisfy: {why:?}")
            }
            Error::Args(why) => write!(f, "{why}"),
            Error::Specialise { symbol, why } => {
                write!(f, "`{symbol}` states a specialisation this fire could not take: {why}")
            }
            Error::Driver { what, code, why } => write!(f, "{what} failed with {code} ({why})"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Args(why) => Some(why),
            _ => None,
        }
    }
}

impl From<crate::runtime::ArgError> for Error {
    fn from(why: crate::runtime::ArgError) -> Self {
        Error::Args(why)
    }
}
