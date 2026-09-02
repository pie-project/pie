//! The differential recorder: the walk, written down instead of encoded. One
//! more implementation of `kernels_metal::Encode`, standing where
//! [`Sink`](crate::Sink) stands. A buffer argument is recorded as the
//! resolved (reservation, offset) pair rather than the handle, since two
//! walks of one template mint the same handle number at different offsets.
//! One `Encode::fire` is one Metal dispatch is one [`Slot`], with its
//! region/run recorded beside it so a refusal can say where.

use std::cell::RefCell;

use kernels_metal::{ArgValue, Encode, Error, Fire};

use crate::device::{Handles, handles::NIL};
use crate::window::{At, Windows};

/// One recorded argument: what the encoder would have bound, not what the
/// entry said.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arg {
    /// A buffer binding: which reservation (by its retained pointer, stable
    /// for the life of a load) and the byte offset into it.
    Buffer {
        /// The reservation's identity — an address, used only for equality.
        slab: u64,
        /// Bytes from the reservation's base.
        offset: u64,
        /// Whether the entry declared write intent.
        mutable: bool,
    },
    /// A nil binding at an index the shader does not dereference on this arm.
    Absent,
    /// A scalar bound by value.
    I32(i32),
    /// A scalar bound by value.
    U32(u32),
    /// A scalar bound by value — recorded as bits, so the fit compares an
    /// f32 the way the bytes compare.
    F32(u32),
    /// A 64-bit scalar (`size_t` in MSL).
    Usize(u64),
}

impl Arg {
    /// The one number a component of this argument varies in, when it has
    /// one. `None` for an argument whose kind cannot be fitted at all.
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

    /// What stays the same across two recordings for the two to be the same
    /// argument at all — the kind, and for a buffer the reservation.
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

    /// How a refusal names it.
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

/// One dispatch, as the ICB would hold it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slot {
    /// The shader point: file, entrypoint, jit stamp.
    pub point: Point,
    /// Total threads per axis.
    pub lanes: [u32; 3],
    /// Threadgroup extent per axis, `[0,0,0]` where the entry left it to the
    /// pipeline's own occupancy answer.
    pub group: [u32; 3],
    /// Every argument at its own index, gaps included.
    pub args: Vec<Arg>,
    /// Which region of the template this dispatch stood in.
    pub region: u32,
    /// Which run of that region's window.
    pub run: u32,
    /// How many token rows that run's window covered. Not a component of the
    /// dispatch itself, but needed by a tiling law's ceiling
    /// (`abi::Law::Ceil`) and an arm picked off the window (`abi::Pick::Rows`).
    pub window_rows: u32,
    /// How many lanes that run's window covered.
    pub window_lanes: u32,
}

/// A shader point, owned rather than borrowed so a recording outlives the
/// walk that made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Point {
    /// The `.metal` file.
    pub file: &'static str,
    /// The entrypoint.
    pub entrypoint: &'static str,
    /// The jit instantiation stamp; empty for a stamped-in-source point.
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

/// One walk, written down: the slots in dispatch order, the composition it
/// was walked at (`classes`), and where that composition sits in the fit's
/// probe-basis coordinates (`coords`) — a basis of reachable directions,
/// since a class's rows and lanes aren't always independently reachable
/// (e.g. a decode class's word ties them together).
#[derive(Clone, Debug)]
pub struct Recording {
    /// Every dispatch, in walk order.
    pub slots: Vec<Slot>,
    /// The per-class `(rows, lanes)` vector this walk was recorded at.
    pub classes: Vec<(u32, u32)>,
    /// This point's coordinates in the probe basis, one per direction.
    pub coords: Vec<i128>,
}

impl Recording {
    /// The same recording, placed in the probe basis.
    #[must_use]
    pub fn at(mut self, coords: Vec<i128>) -> Recording {
        self.coords = coords;
        self
    }

    /// How many token rows this composition carried, over every class.
    #[must_use]
    pub fn rows(&self) -> u32 {
        self.classes.iter().map(|(rows, _)| rows).sum()
    }
}

/// The recording `Encode`: everything a [`Sink`](crate::Sink) does, except
/// the dispatch.
pub struct Tape<'a> {
    handles: &'a Handles,
    place: &'a At,
    /// This fire's resolved windows, read at the cursor — the one number a
    /// row carries that the dispatch itself does not ([`Slot::window_rows`]).
    windows: &'a Windows,
    slots: RefCell<Vec<Slot>>,
}

impl<'a> Tape<'a> {
    /// An empty tape over one load's handle table, one walk's cursor and
    /// this fire's windows.
    #[must_use]
    pub fn new(handles: &'a Handles, place: &'a At, windows: &'a Windows) -> Tape<'a> {
        Tape {
            handles,
            place,
            windows,
            slots: RefCell::new(Vec::new()),
        }
    }

    /// The slots, in dispatch order, at the composition `classes`. The probe
    /// basis is attached afterwards by whoever chose it
    /// ([`Recording::at`]).
    #[must_use]
    pub fn finish(self, classes: Vec<(u32, u32)>) -> Recording {
        Recording {
            slots: self.slots.into_inner(),
            classes,
            coords: Vec::new(),
        }
    }

    /// How many dispatches have been written down.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.borrow().len()
    }

    /// Whether nothing has been written down.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.borrow().is_empty()
    }

    /// Resolve one argument the way the sink would bind it.
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
    /// Write the dispatch down. Every argument is resolved here, not at fit
    /// time, since the handle table is rewound at the end of the fire.
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
