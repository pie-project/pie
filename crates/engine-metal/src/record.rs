use std::cell::RefCell;

use kernels_metal::{ArgValue, Encode, Error, Fire};
use poem_exec::fire::At;

use crate::device::{Handles, handles::NIL};
use crate::window::Windows;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arg {
    Buffer {
        slab: u64,
        offset: u64,
        mutable: bool,
    },
    Absent,
    I32(i32),
    U32(u32),
    F32(u32),
    Usize(u64),
}

impl Arg {
    #[must_use]
    pub fn scalar(self) -> Option<i128> {
        match self {
            Arg::Buffer { offset, .. } => Some(i128::from(offset)),
            Arg::Absent => None,
            Arg::I32(v) => Some(i128::from(v)),
            Arg::U32(v) => Some(i128::from(v)),
            Arg::F32(_) => None,
            Arg::Usize(v) => Some(i128::from(v)),
        }
    }

    #[must_use]
    pub fn shape(self) -> (u8, u64) {
        match self {
            Arg::Buffer { slab, mutable, .. } => (u8::from(mutable), slab),
            Arg::Absent => (2, 0),
            Arg::I32(_) => (3, 0),
            Arg::U32(_) => (4, 0),
            Arg::F32(bits) => (5, u64::from(bits)),
            Arg::Usize(_) => (6, 0),
        }
    }

    #[must_use]
    pub fn kind(self) -> &'static str {
        match self {
            Arg::Buffer { mutable: false, .. } => "a buffer offset",
            Arg::Buffer { mutable: true, .. } => "a writable buffer offset",
            Arg::Absent => "an absent binding",
            Arg::I32(_) => "an i32",
            Arg::U32(_) => "a u32",
            Arg::F32(_) => "an f32",
            Arg::Usize(_) => "a usize",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    pub point: Point,
    pub lanes: [u32; 3],
    pub group: [u32; 3],
    pub args: Vec<Arg>,
    pub region: u32,
    pub run: u32,
    pub window_rows: u32,
    pub window_lanes: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Point {
    pub file: &'static str,
    pub entrypoint: &'static str,
    pub stamp: &'static str,
}

impl Point {
    #[must_use]
    fn of(fire: Fire) -> Point {
        Point {
            file: fire.file,
            entrypoint: fire.entrypoint,
            stamp: fire.stamp,
        }
    }
}

impl std::fmt::Display for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.stamp.is_empty() {
            write!(f, "{}::{}", self.file, self.entrypoint)
        } else {
            write!(f, "{}::{}[{}]", self.file, self.entrypoint, self.stamp)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Recording {
    pub slots: Vec<Slot>,
    pub classes: Vec<(u32, u32)>,
    pub coords: Vec<i128>,
}

impl Recording {
    #[must_use]
    pub fn at(mut self, coords: Vec<i128>) -> Recording {
        self.coords = coords;
        self
    }

    #[must_use]
    pub fn rows(&self) -> u32 {
        self.classes.iter().map(|(rows, _)| rows).sum()
    }
}

pub struct Tape<'a> {
    handles: &'a Handles,
    place: &'a At,
    windows: &'a Windows,
    slots: RefCell<Vec<Slot>>,
}

impl<'a> Tape<'a> {
    #[must_use]
    pub fn new(handles: &'a Handles, place: &'a At, windows: &'a Windows) -> Tape<'a> {
        Tape {
            handles,
            place,
            windows,
            slots: RefCell::new(Vec::new()),
        }
    }

    #[must_use]
    pub fn finish(self, classes: Vec<(u32, u32)>) -> Recording {
        Recording {
            slots: self.slots.into_inner(),
            classes,
            coords: Vec::new(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.borrow().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.borrow().is_empty()
    }

    fn resolve(&self, fire: Fire, at: usize, arg: ArgValue) -> Result<Arg, Error> {
        let (handle, mutable) = match arg {
            ArgValue::Buffer(handle) => (handle, false),
            ArgValue::BufferMut(handle) => (handle, true),
            ArgValue::I32(v) => return Ok(Arg::I32(v)),
            ArgValue::U32(v) => return Ok(Arg::U32(v)),
            ArgValue::F32(v) => return Ok(Arg::F32(v.to_bits())),
            ArgValue::Usize(v) => return Ok(Arg::Usize(v)),
        };
        if handle == NIL {
            return Ok(Arg::Absent);
        }
        let binding = self.handles.get(handle).ok_or_else(|| Error::Backend {
            op: fire.entrypoint,
            detail: format!("handle {handle} at argument {at}, which this fire minted no row for"),
        })?;
        Ok(Arg::Buffer {
            slab: crate::device::alloc::slab_id(binding.slab()),
            offset: binding.offset(),
            mutable,
        })
    }
}

impl Encode for Tape<'_> {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        let mut resolved = Vec::with_capacity(args.len());
        for (at, arg) in args.iter().enumerate() {
            resolved.push(self.resolve(fire, at, *arg)?);
        }
        let region = self.place.region.get();
        let run = self.place.run.get();
        let window = self.windows.at(region, run).span;
        self.slots.borrow_mut().push(Slot {
            point: Point::of(fire),
            lanes: fire.lanes,
            group: fire.group,
            args: resolved,
            region,
            run,
            window_rows: window.rows,
            window_lanes: window.lanes,
        });
        Ok(())
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(NIL))
    }
}
