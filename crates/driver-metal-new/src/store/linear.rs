//! Per-slot step counters for the GDN recurrent state, whose parity is the
//! conv-state ping-pong.
//!
//! `ConvState` (read) and `ConvStateOut` (written) are DISTINCT buffers,
//! advanced token to token by swapping their binds: step `i` reads what
//! `i - 1` wrote. Which buffer currently holds the latest data is therefore
//! `steps % 2` — and it is a function of how many steps THIS SLOT has
//! taken. Not the absolute position, which can start non-zero; and not a
//! single decoder-wide counter, which would silently corrupt a slot's
//! parity whenever a DIFFERENT slot had stepped in between. Each slot
//! tracks its own count so switching between slots between forwards resumes
//! each slot's ping-pong correctly.
//!
//! [`copy`](LinearStateSlots::copy) moves the count with the state, and the
//! why is worth keeping: a state copy moves both ping-pong buffers VERBATIM
//! (A stays A, B stays B, never swapped), so the destination must inherit
//! the source's exact count too — otherwise a later step on the
//! destination could read the STALE half instead of the one holding the
//! copied-in latest data, silently correct only when the two slots happened
//! to share a parity by coincidence.
//!
//! What did not survive the port: the C++ `at()` returns slot 0's counter
//! for ANY out-of-range slot. A wild slot id — and slot ids arrive from the
//! ABI — did not fail; it read and wrote slot 0's parity, corrupting a slot
//! that was never named. Every accessor here refuses instead.

/// Which ping-pong buffer holds the latest conv state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parity {
    /// An even step count: the slot reads the same buffer it started with.
    Even,
    /// An odd step count: the binds are swapped.
    Odd,
}

/// A slot outside the pool. Slot ids are ABI inputs — data, not caller
/// bugs — so a wild one is refused, never aliased onto slot 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WildSlot {
    /// The slot asked for.
    pub slot: u32,
    /// How many slots exist.
    pub slots: u32,
}

impl std::fmt::Display for WildSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "linear state: slot {} outside the {} configured",
            self.slot, self.slots
        )
    }
}

impl std::error::Error for WildSlot {}

/// The counters, one per recurrent-state slot.
#[derive(Clone, Debug)]
pub struct LinearStateSlots {
    steps: Vec<u64>,
}

impl LinearStateSlots {
    /// `count` zeroed counters.
    #[must_use]
    pub fn new(count: u32) -> Self {
        LinearStateSlots {
            steps: vec![0; count as usize],
        }
    }

    /// How many slots exist.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn len(&self) -> u32 {
        self.steps.len() as u32
    }

    /// Whether there are no slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Zero every counter (a pool reset).
    pub fn reset_all(&mut self) {
        self.steps.fill(0);
    }

    /// Zero one slot's counter (its state was reinitialised).
    ///
    /// # Errors
    ///
    /// [`WildSlot`]; see the type.
    pub fn reset(&mut self, slot: u32) -> Result<(), WildSlot> {
        *self.checked_mut(slot)? = 0;
        Ok(())
    }

    /// Advance `slot` by one step. Wrapping, which preserves parity: the
    /// wrap modulus is even, so `steps % 2` continues correctly across it.
    ///
    /// # Errors
    ///
    /// [`WildSlot`]; see the type.
    pub fn step(&mut self, slot: u32) -> Result<(), WildSlot> {
        let counter = self.checked_mut(slot)?;
        *counter = counter.wrapping_add(1);
        Ok(())
    }

    /// The steps `slot` has taken.
    ///
    /// # Errors
    ///
    /// [`WildSlot`]; see the type.
    pub fn count(&self, slot: u32) -> Result<u64, WildSlot> {
        self.checked(slot).map(|at| self.steps[at])
    }

    /// Which ping-pong buffer holds `slot`'s latest conv state.
    ///
    /// # Errors
    ///
    /// [`WildSlot`]; see the type.
    pub fn parity(&self, slot: u32) -> Result<Parity, WildSlot> {
        Ok(if self.count(slot)? % 2 == 0 {
            Parity::Even
        } else {
            Parity::Odd
        })
    }

    /// `dst` inherits `src`'s exact count, because its state buffers were
    /// just copied verbatim; see the module docs for the stale-half failure
    /// this prevents.
    ///
    /// # Errors
    ///
    /// [`WildSlot`] naming whichever side is out of range.
    pub fn copy(&mut self, src: u32, dst: u32) -> Result<(), WildSlot> {
        let value = self.steps[self.checked(src)?];
        *self.checked_mut(dst)? = value;
        Ok(())
    }

    fn checked(&self, slot: u32) -> Result<usize, WildSlot> {
        if (slot as usize) < self.steps.len() {
            Ok(slot as usize)
        } else {
            Err(WildSlot {
                slot,
                slots: self.len(),
            })
        }
    }

    fn checked_mut(&mut self, slot: u32) -> Result<&mut u64, WildSlot> {
        let at = self.checked(slot)?;
        Ok(&mut self.steps[at])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_is_a_slots_own_and_survives_other_slots_stepping() {
        let mut slots = LinearStateSlots::new(3);
        // Slot 1 steps three times while slot 2 steps once in between: a
        // decoder-wide counter would give slot 1 the wrong parity here.
        slots.step(1).unwrap();
        slots.step(2).unwrap();
        slots.step(1).unwrap();
        slots.step(1).unwrap();
        assert_eq!(slots.parity(1), Ok(Parity::Odd));
        assert_eq!(slots.parity(2), Ok(Parity::Odd));
        assert_eq!(slots.parity(0), Ok(Parity::Even), "an untouched slot");
        assert_eq!(slots.count(1), Ok(3));
    }

    #[test]
    fn a_state_copy_inherits_the_exact_count_not_just_the_parity() {
        let mut slots = LinearStateSlots::new(2);
        for _ in 0..5 {
            slots.step(0).unwrap();
        }
        slots.copy(0, 1).unwrap();
        // The buffers were copied verbatim, never swapped: dst must read
        // the same half src would have.
        assert_eq!(slots.count(1), Ok(5));
        assert_eq!(slots.parity(1), Ok(Parity::Odd));
    }

    #[test]
    fn a_wild_slot_is_refused_not_aliased_onto_slot_zero() {
        let mut slots = LinearStateSlots::new(2);
        slots.step(0).unwrap();
        // The C++ handed slot 0's counter to any out-of-range id; a wild
        // ABI slot silently flipped slot 0's parity. Here it refuses.
        assert_eq!(slots.step(7), Err(WildSlot { slot: 7, slots: 2 }));
        assert_eq!(slots.count(0), Ok(1), "slot 0 was not touched");
        assert_eq!(slots.reset(2), Err(WildSlot { slot: 2, slots: 2 }));
        assert_eq!(slots.copy(0, 9), Err(WildSlot { slot: 9, slots: 2 }));
    }

    #[test]
    fn resets_zero_what_they_name_and_only_that() {
        let mut slots = LinearStateSlots::new(2);
        slots.step(0).unwrap();
        slots.step(1).unwrap();
        slots.reset(0).unwrap();
        assert_eq!(slots.count(0), Ok(0));
        assert_eq!(slots.count(1), Ok(1));
        slots.reset_all();
        assert_eq!(slots.count(1), Ok(0));
    }
}
