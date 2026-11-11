#![allow(unsafe_code)]
use kernels_cuda::Tensor;
use poem_compiler::ArenaMap;
use poem_exec::store::arena::rect;
use poem_ir::ValueId;

use crate::error::{Fault, Result};
use crate::run::SlotTable;
use crate::store::Pools;

/// A fire's activation: the span under the state pool's end
/// ([`Pools::activation_base`]), backed as far as the load's fires reach.
#[derive(Debug)]
pub struct Arena {
    committed: u64,
    reserved: u64,
    /// The activation rungs (SPEC decision 6): for fires of at most so many
    /// tokens (and lanes), the carve laid out for them, smallest first.
    /// Each lies at the reservation's end, so a rung's fires touch only the
    /// top of the span and state may take what lies under it.
    rungs: Vec<(u32, ArenaMap)>,
    /// For every carve laid at the reservation's end, its bytes and the
    /// bytes before its `attn.scores` planes ([`Arena::ensure_rungs`]).
    prefixes: Vec<(u64, u64)>,
}

impl Arena {
    /// `bytes`, the most any arena map this load carves asks for: what the
    /// pool was reserved its activation chunks for.
    #[must_use]
    pub fn reserve(bytes: u64) -> Arena {
        Arena {
            committed: 0,
            reserved: bytes,
            rungs: Vec::new(),
            prefixes: Vec::new(),
        }
    }

    /// With `rungs`, each a carve no larger than the reservation, smallest
    /// first: a fire carves the smallest that holds it, at the
    /// reservation's end ([`Arena::fire_slots`]).
    ///
    /// `prefixes`: each carve that may be laid at the end, its bytes and
    /// the bytes before its `attn.scores` planes.
    #[must_use]
    pub fn with_rungs(
        mut self,
        mut rungs: Vec<(u32, ArenaMap)>,
        prefixes: Vec<(u64, u64)>,
    ) -> Arena {
        rungs.retain(|(_, map)| map.bytes <= self.reserved);
        rungs.sort_by_key(|&(tokens, _)| tokens);
        self.rungs = rungs;
        self.prefixes = prefixes;
        self
    }

    /// Whether this load lays carves at the reservation's end.
    #[must_use]
    pub fn has_rungs(&self) -> bool {
        !self.rungs.is_empty()
    }

    /// Each rung's tokens and the bytes of the span under what a fire of
    /// it takes at the top (its carve, or `forked`'s where a fire that forks
    /// is wider) that state may take at that rung: no more than the prefix
    /// backed from the span's start ([`Arena::ensure`]), so state never
    /// lands where nothing is backed.
    #[must_use]
    pub fn rungs(&self, forked: Option<&ArenaMap>) -> Vec<(u32, u64)> {
        self.rungs
            .iter()
            .map(|(tokens, map)| {
                let bytes = map.bytes.max(forked.map_or(0, |forked| forked.bytes));
                (
                    *tokens,
                    self.reserved.saturating_sub(bytes).min(self.committed),
                )
            })
            .collect()
    }

    /// Backs, for every carve laid at the reservation's end, the part a
    /// fire reads before its scores. Nothing where no rung is stated: every
    /// carve then lies at the span's start, under [`Arena::ensure`]'s
    /// prefix. After the fit: a range reserved anew backs again only the
    /// prefix from the span's start.
    pub fn ensure_rungs(&self, pools: &mut Pools) -> Result<()> {
        if self.rungs.is_empty() {
            return Ok(());
        }
        for &(bytes, unscored) in &self.prefixes {
            let at = self.reserved.saturating_sub(bytes);
            pools.hold_activation_at(at, unscored.min(bytes), false)?;
        }
        Ok(())
    }

    /// No smaller rung: every fire's arena at the span's start.
    pub fn drop_rungs(&mut self) {
        self.rungs.clear();
    }

    /// The reservation's bytes: the most any carve this load lays asks for.
    #[must_use]
    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    /// Backs the whole reservation: the `attn.scores` planes over the prefix
    /// a load commits, once a program reads them.
    pub fn ensure_whole(&mut self, pools: &mut Pools) -> Result<()> {
        if self.committed >= self.reserved {
            return Ok(());
        }
        pools.hold_activation(self.reserved, true)?;
        self.committed = self.reserved;
        Ok(())
    }

    pub fn ensure(&mut self, pools: &mut Pools, bytes: u64) -> Result<()> {
        if bytes <= self.committed {
            return Ok(());
        }
        pools.hold_activation(bytes, false)?;
        self.committed = bytes;
        Ok(())
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.committed
    }

    #[must_use]
    pub fn slots(&self, pools: &Pools, map: &ArenaMap, rows: poem_compiler::FireRows) -> SlotTable {
        carve(pools.activation_base(), map, rows)
    }

    /// A fire's slots: `map` (the arena in order, or the forked one) as
    /// laid at the span's start, or, where this load states rungs, the
    /// smallest rung's carve that holds `rows` (a fire that forks keeps its
    /// own) at the reservation's end. The place each carve takes is fixed,
    /// so a body recorded at any rung stays good at every rung its fire
    /// fits.
    #[must_use]
    pub fn fire_slots(
        &self,
        pools: &Pools,
        map: &ArenaMap,
        forked: bool,
        rows: poem_compiler::FireRows,
    ) -> SlotTable {
        if self.rungs.is_empty() {
            return self.slots(pools, map, rows);
        }
        let chosen = if forked {
            map
        } else {
            self.rungs
                .iter()
                .find(|(tokens, _)| {
                    u64::from(*tokens) >= rows.tokens && u64::from(*tokens) >= rows.lanes
                })
                .map_or(map, |(_, rung)| rung)
        };
        let base = pools.activation_base() + self.reserved.saturating_sub(chosen.bytes);
        carve(base, chosen, rows)
    }

    /// Reads `into.len()` bytes at `at` on `stream` (the shell's own), and
    /// waits for it. Not on the legacy stream: that waits on every blocking
    /// stream of the context, a peer rank's on the same device too, and a
    /// peer spinning in its next step's collective waits on this rank's
    /// launch, which queues behind the legacy copy.
    pub fn read(
        &self,
        pools: &Pools,
        stream: *mut core::ffi::c_void,
        at: u64,
        into: &mut [u8],
    ) -> Result<()> {
        let base = pools.activation_base();
        let offset = at.checked_sub(base).ok_or(Fault::Ceiling {
            what: "bytes into the arena",
            need: at,
            have: base,
        })?;
        // A carve laid at the reservation's end reads out past the prefix
        // backed from the start, inside what `ensure_rungs` backed for it.
        let backed = if self.rungs.is_empty() {
            self.committed
        } else {
            self.reserved
        };
        if offset.saturating_add(into.len() as u64) > backed {
            return Err(Fault::Ceiling {
                what: "bytes into the arena's committed prefix",
                need: offset.saturating_add(into.len() as u64),
                have: backed,
            });
        }
        #[cfg(feature = "device")]
        {
            use crate::device::rt;
            // SAFETY: `at..at + len` lies inside the committed prefix, and
            // `stream` is the caller's live stream of this device.
            unsafe {
                crate::device::ctx::check(
                    "cudaMemcpyAsync",
                    rt::cudaMemcpyAsync(
                        into.as_mut_ptr().cast(),
                        at as *const core::ffi::c_void,
                        into.len(),
                        rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                        stream.cast(),
                    ),
                )?;
            }
            crate::device::ctx::sync(stream)
        }
        #[cfg(not(feature = "device"))]
        {
            let _ = (stream, into);
            Err(Fault::Runtimeless)
        }
    }
}

#[must_use]
pub fn carve(base: u64, map: &ArenaMap, rows: poem_compiler::FireRows) -> SlotTable {
    let mut cells: Vec<Option<Tensor>> = Vec::with_capacity(map.placements.len());
    for value in 0..map.placements.len() {
        let value = ValueId(value as u32);
        cells.push(
            rect(map, value, rows)
                .map(|rect| Tensor::new(base + rect.offset, rect.rows, rect.width, rect.dtype)),
        );
    }
    SlotTable(cells)
}

#[cfg(test)]
mod tests {
    use poem::Platform;
    use poem_compiler::{Budget, DeviceProfile, compile};

    use super::*;

    const DEPLOYMENT: &str = "qwen35-d0.8b-bf16-kv-bf16";

    fn compiled() -> (poem_ir::Trace, poem_compiler::CompiledModel) {
        let trace = poem_compiler::catalog::deployment(DEPLOYMENT)
            .expect("the catalog ships the smoke's deployment")
            .trace(Platform::Cuda);
        let compiled = compile(&trace, &Budget::new(4, 64), &DeviceProfile::default())
            .expect("the smoke's deployment bakes");
        (trace, compiled)
    }

    #[test]
    fn the_carve_fits_the_allocation_it_asks_for() {
        let (_, compiled) = compiled();
        let slots = carve(
            0,
            &compiled.arena,
            poem_compiler::FireRows::text_only(64, 4),
        );
        for handle in slots.0.iter().flatten() {
            let end = handle.ptr
                + handle.elements() * poem_compiler::arena::elem_bytes(handle.dtype).unwrap_or(0);
            assert!(
                end <= compiled.arena.bytes,
                "a rectangle ending at {end} in an arena of {} bytes",
                compiled.arena.bytes
            );
        }
    }
}
