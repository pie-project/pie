use std::fmt;

pub mod arena;
pub mod check;
pub mod kv;

pub use kv::{
    Geometry, Paging, Reader, Seat, SpaceFacts, Windowed, geometry, geometry_with, indptr,
    window_table,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    Ceiling {
        what: &'static str,
        need: u64,
        have: u64,
    },

    Unbound {
        what: String,
    },

    Straddled {
        value: u32,
        node: u32,
        planned: String,
        consumed: String,
    },
}

pub type Result<T> = std::result::Result<T, Fault>;

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ceiling { what, need, have } => {
                write!(f, "this fire wants {need} {what} and {have} was reserved")
            }
            Self::Unbound { what } => write!(f, "this plan names {what}, which nothing binds"),
            Self::Straddled {
                value,
                node,
                planned,
                consumed,
            } => write!(
                f,
                "the schedule in value {value} is planned over {planned} and read by node \
                 {node} over {consumed}"
            ),
        }
    }
}

impl std::error::Error for Fault {}
