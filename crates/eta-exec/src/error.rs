use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

/// A launch program this host half cannot interpret.
///
/// **A STRUCT, AND NOT A ONE-VARIANT ENUM.** What stood here was
/// `enum Error { Program { message }, Fire(fire::Fault), Kernel(KernelError) }`,
/// shared with the model forward plane; the other two variants were
/// constructed only under `fire/` and left with it. Thirty-three sites
/// construct this one and all thirty-three are in this crate.
///
/// A single-variant enum would keep a discriminant nobody reads and would make
/// every consumer write a `match` arm to reach a field it already knows is
/// there. The struct keeps the `{ message: … }` construction syntax those
/// thirty-three sites already used — the diff was two words per site — and it
/// can grow a second field (a site, an instance id) without turning into a
/// different kind of type. A newtype would have cost the field name, which is
/// the only thing here worth reading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "launch program cannot be interpreted: {}", self.message)
    }
}

impl std::error::Error for Error {}
