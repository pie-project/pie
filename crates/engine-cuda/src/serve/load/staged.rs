//! A load staged and finished: its workspaces laid, its bodies armed, its
//! pool fitted to the card.

#[allow(clippy::wildcard_imports)]
use super::*;

/// A shell whose load stopped short of arming its bodies ([`Shell::stage`]).
pub struct Staged {
    pub shell: Shell,
    pub(super) rank: u32,
    pub(super) arm_trace: bool,
    /// The load was told its weights are held (`Knobs::weights_held`).
    pub(super) weights_held: bool,
    pub(super) device_total: u64,
    pub(super) free_before: u64,
    /// Where the device's user profile lies (the product times a load
    /// measures are saved there).
    pub(super) profiles: Option<std::path::PathBuf>,
}

impl Staged {
    /// Arms the bodies and fits the cache rows to what the card has left.
    pub fn finish(self) -> Result<Shell> {
        let Staged {
            mut shell,
            rank,
            arm_trace,
            weights_held,
            device_total,
            free_before,
            profiles,
        } = self;
        if arm_trace {
            eprintln!(
                "[arm-trace] device free {} MiB before arming",
                crate::device::free_bytes().unwrap_or(0) >> 20
            );
        }
        // Weights left in the serving file are on no tier a fire can read:
        // they are seated in the pool before the lattice is armed over them.
        // So are those a load told they are held keeps a host copy of
        // (planes laid at load, which the file does not hold as served).
        let filed = shell
            .weights
            .experts()
            .is_some_and(|tier| tier.plan().files());
        let held_copied = weights_held
            && shell
                .weights
                .experts()
                .is_some_and(|tier| !tier.plan().cached_experts().is_empty());
        if filed || held_copied {
            shell.preseat_weights(filed, held_copied)?;
        }
        // A load whose weights the pool does not hold whole streams them:
        // armed now, its bodies would read every expert from the host copy,
        // over the bus, fire after fire. It arms at the starting weight
        // residency instead, over the slots the cache then holds, and the
        // fit keeps the bodies' allowance out of the cache rows till then,
        // when it settles ([`Shell::settle_fit`]).
        let deferred =
            shell.ep.is_none() && shell.records_bodies() && !shell.pools.weights_seated_whole();
        if deferred {
            shell.arm_deferred = true;
            tracing::info!(
                "this load streams its weights; its bodies arm at the starting \
                 weight residency, over the slots the cache holds then"
            );
        } else {
            shell.arming_at_load = true;
            let armed = shell.arm_bodies();
            shell.arming_at_load = false;
            armed?;
            // What arming's picks read an estimate of, before the products'
            // measuring lays a table of its own.
            let noted = shell
                .device
                .ctx()
                .routes()
                .map(|routes| routes.behind())
                .unwrap_or_default();
            shell.routes_measured(profiles.as_deref())?;
            say_stale_banks(&noted);
            if shell.measuring {
                shell.measured_refit(device_total, free_before)?;
            }
            shell.settle_outside()?;
        }
        let resident = footprint(&shell, device_total, free_before);
        // Whatever armed a body, the range stays where it lies: the bodies
        // name addresses in it, and a fit that reserved it anew (a flat
        // range's does) would leave them naming the old one.
        if shell.cache.body_stats().census.bodies > 0 {
            shell.pools.pin();
        }
        // No fire of the load's own is still in flight over the chunks the
        // last fit lets go and lays again.
        shell.device.synchronize()?;
        let fitted = if deferred && shell.arena.has_rungs() {
            // The bodies arm after the runtime has read the pool, so the
            // fit that follows them cannot be the one it is stated: it is
            // stated the range whole, what that fit may yet give the cache
            // rows with it, and keeps the pool under it by what this fit
            // holds back for the bodies' allowance and the activation no
            // fire has backed yet (`Shell::settle_fit` lifts it). The
            // state's chunks then end where the activation starts, so what
            // a rung lends and what lies under it are one span.
            shell
                .pools
                .fit_to_state_whole(shell.bodies_hold(), shell.held_for_fires, &resident)?
        } else {
            // Everything the load puts on the device is there, its bodies
            // among it: the pool is what the card has left, and the range
            // is stated as exactly the chunks backed below. Arming fired
            // every composition, so the activation is at its high water; a
            // load that records no bodies has fired none, and keeps the
            // whole span for its fires; one that arms later and lends no
            // rung keeps the bodies' allowance out as well.
            let bodies = if deferred { shell.bodies_hold() } else { 0 };
            let settled = shell.records_bodies() && !deferred;
            shell
                .pools
                .fit_and_state(bodies, shell.held_for_fires, settled, &resident)?
        };
        // A load that arms later may have had its range reserved anew by
        // its fit: the workspaces are laid where it now lies.
        shell.lay_workspace()?;
        // The load's last fit backs what it took, here and not later: every
        // page the load serves is backed before anything else can take the
        // card's room, so serving maps nothing (a dying process holding the
        // driver's lock stalls no fire of this one's), and the pool the
        // runtime is stated is the pool the card backed.
        shell.back_whole()?;
        if let Some(kv) = shell.pools.stated_kv() {
            tracing::info!(
                "kv_device_tokens = {}: the device pool holds {} kv pages of {} \
                 tokens ({} asked), {} chunks of {} MiB: {} for the kv pages and the shapes \
                 beside them, {} for the weights held whole",
                kv.tokens,
                shell.pools.paging().pages(),
                kv.page_tokens,
                kv.pages,
                kv.chunks(),
                kv.chunk_bytes >> 20,
                kv.kv_chunks,
                kv.weight_chunks,
            );
        }
        if arm_trace {
            eprintln!(
                "[arm-trace] cache rows sized to {} pages ({} MiB, {} sequences at the declared \
                 context) of the {} MiB the card hands out for them ({resident})",
                fitted.fit,
                shell.pools.declared_bytes() >> 20,
                fitted.fit / u64::from(shell.pools.paging().pages_per_slot),
                fitted.room >> 20,
            );
            if let Some(free) = crate::device::alloc::free_bytes() {
                let ledger = ledger(&shell, device_total, free_before);
                let audit = ledger.audit(device_total.saturating_sub(free));
                eprintln!(
                    "[arm-trace] ledger: {ledger}; the device reports {} MiB in use: {} MiB no \
                     owner entered",
                    audit.device >> 20,
                    audit.gap() >> 20,
                );
            }
            eprintln!(
                "[arm-trace] after arming: {} MiB asked past the laid workspaces",
                kernels_cuda::jit::Slabs::unplaced_bytes() >> 20,
            );
        }
        shell.card = (device_total, free_before);
        if rank != 0 {
            shell.programs.set_shadow(true);
        }
        // Only a warm fire waits on the device, and a load whose bodies were
        // held to nothing fires none: a fault raised by what the load put
        // down asynchronously would otherwise be the first request's to
        // find, as a sticky error from whichever call touches the context
        // next, under a banner that said ready.
        shell.device.synchronize().map_err(|why| {
            Fault::program(
                "serve::load",
                format!(
                    "the device faulted under this load's own work ({why}); a context that \
                     has faulted serves nothing, so the load is refused rather than reported \
                     ready"
                ),
            )
        })?;
        Ok(shell)
    }
}
