use std::sync::{Arc, Barrier};

use crate::device::{Buffer, Context};
use crate::error::{Fault, Result};

const SLOT_ALIGN: u64 = 256;

#[derive(Debug)]
pub struct Group {
    band: Arc<Buffer>,
    gate: Arc<Barrier>,
    world: u32,
    slot_bytes: u64,
}

impl Group {
    pub fn open(device: &Context, world: u32, slot_bytes: u64) -> Result<Group> {
        if world == 0 {
            return Err(Fault::Ceiling {
                what: "ranks in a communicator",
                need: 0,
                have: 1,
            });
        }
        let slot_bytes = slot_bytes.next_multiple_of(SLOT_ALIGN).max(SLOT_ALIGN);
        let band = Buffer::zeroed(device, slot_bytes * u64::from(world))?;
        Ok(Group {
            band: Arc::new(band),

            gate: Arc::new(Barrier::new(world as usize)),
            world,
            slot_bytes,
        })
    }

    pub fn rank(&self, rank: u32) -> Result<Comm> {
        if rank >= self.world {
            return Err(Fault::Ceiling {
                what: "a rank in this communicator",
                need: u64::from(rank),
                have: u64::from(self.world),
            });
        }
        Ok(Comm {
            band: Arc::clone(&self.band),
            gate: Arc::clone(&self.gate),
            rank,
            world: self.world,
            slot_bytes: self.slot_bytes,
        })
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.band.bytes()
    }
}

#[derive(Clone, Debug)]
pub struct Comm {
    band: Arc<Buffer>,
    gate: Arc<Barrier>,
    rank: u32,
    world: u32,
    slot_bytes: u64,
}

impl Comm {
    #[must_use]
    pub fn rank(&self) -> u32 {
        self.rank
    }

    #[must_use]
    pub fn world(&self) -> u32 {
        self.world
    }

    #[must_use]
    pub fn band(&self) -> &Buffer {
        &self.band
    }

    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    pub fn slot_bytes_u32(&self) -> Result<u32> {
        u32::try_from(self.slot_bytes).map_err(|_| Fault::Ceiling {
            what: "bytes in one communicator slot",
            need: self.slot_bytes,
            have: u64::from(u32::MAX),
        })
    }

    pub fn wait(&self) {
        self.gate.wait();
    }
}
