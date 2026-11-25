//! What only tests ask of the roles: audits of the slots they hold, and
//! a bind or channel registration in one call.

use super::*;

impl Roles {
    /// Every live part's slot on the device, by item: what the pool must
    /// count held while the item names it (for tests: a holder naming a
    /// slot the pool counts free reads and writes another's state).
    pub(crate) fn device_slots_named(&self) -> Vec<(String, super::super::slots::Slot)> {
        let device = self.device.lock().unwrap();
        let mut out = Vec::new();
        for (owner, item) in &self.items {
            if item.tier != Tier::Device {
                continue;
            }
            for part in item.parts.iter().filter(|part| !part.dead()) {
                let Some(shape) = self.shapes.of(part.role) else {
                    continue;
                };
                for &slot_ref in &part.refs {
                    out.push((
                        format!("{owner:?} {:?} shed={}", part.role, item.shed),
                        device.slot(shape.pool, slot_ref),
                    ));
                }
            }
        }
        out
    }

    /// Every device slot the items hold in slots of their shape (tables,
    /// rings, channels, adapters: not guest runs of chunks), checked
    /// against the pool and each other: each held by one item, and handed
    /// out in the pool. What is said is a slot two items hold, or one an
    /// item holds that the pool counts free: the store's double give that
    /// handed one ref to a restore and a fresh bind together.
    #[must_use]
    pub fn audit_device_slots(&self) -> Vec<String> {
        let device = self.device.lock().unwrap();
        let mut seen: HashMap<(SlotRole, u32), Owner> = HashMap::new();
        let mut said = Vec::new();
        for (owner, item) in &self.items {
            if item.tier != Tier::Device {
                continue;
            }
            for part in item.parts.iter().filter(|part| !part.shaped()) {
                let Some(shape) = self.shapes.of(part.role) else {
                    continue;
                };
                for &slot_ref in &part.refs {
                    if let Some(other) = seen.insert((part.role, slot_ref), *owner) {
                        said.push(format!(
                            "{:?} slot {slot_ref} held by {other:?} and {owner:?}",
                            part.role
                        ));
                    }
                    if !device
                        .pool()
                        .is_held(shape.pool, device.slot(shape.pool, slot_ref))
                    {
                        said.push(format!(
                            "{:?} slot {slot_ref} held by {owner:?} is free in the pool",
                            part.role
                        ));
                    }
                }
            }
        }
        drop(device);
        // Nothing moving and no bind on its way: every slot the pool holds
        // out of a role's shape is one an item names (a fresh pair taken by
        // a bind no item recorded held its chunk for good).
        let settled = self.on_way.1 == 0
            && self.binding.is_empty()
            && self.items.values().all(|item| item.tier != Tier::Moving);
        if settled {
            for (role, used, named) in self.audit_held() {
                if used != named {
                    said.push(format!(
                        "{role:?}: the pool holds {used} slots out, the items name {named}"
                    ));
                }
            }
        }
        said
    }

    /// The device slots of each role's shape the pool holds out, beside the
    /// slots the items on the device (and moving) name: `(role, pool,
    /// items)`. Equal, nothing is held that no item names.
    #[must_use]
    pub fn audit_held(&self) -> Vec<(SlotRole, u64, u64)> {
        let device = self.device.lock().unwrap();
        let pool = device.pool();
        let mut out = Vec::new();
        for role in [SlotRole::Instance, SlotRole::Channel, SlotRole::Adapter] {
            let Some(shape) = self.shapes.of(role) else {
                continue;
            };
            let used: u64 = (0..pool.total_chunks())
                .filter(|&chunk| pool.chunk_shape(chunk) == Some(shape.pool))
                .map(|chunk| u64::from(pool.chunk_used(chunk)))
                .sum();
            let named: u64 = self
                .items
                .values()
                .filter(|item| matches!(item.tier, Tier::Device | Tier::Moving))
                .flat_map(|item| &item.parts)
                .filter(|part| part.role == role)
                .map(|part| part.refs.len() as u64)
                .sum();
            out.push((role, used, named));
        }
        out
    }

}

/// Registers channel `plan` by `register`, over the `Channel` slots
/// [`Channels::ring_facts`] states its ring takes ([`ChannelRegistration::slots`]),
/// held for `holder` until it closes. Without `roles`, `plan` as it is.
///
/// [`Channels::ring_facts`]: ::engine::Channels::ring_facts
/// [`ChannelRegistration::slots`]: ::engine::ChannelRegistration::slots
pub fn register_channel<T>(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plan: &ChannelRegistration,
    holder: Option<Holder>,
    register: impl FnOnce(&mut dyn Engine, &ChannelRegistration) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let taken = take_ring(roles, engine, plan)?;
    register_taken(roles, engine, plan, holder, taken, register)
}

/// Binds `binding` on `engine` over the `Instance` and `Channel` slots its
/// program's facts state, and the whole chunks of its guest state
/// ([`Programs::guest_state`], asked at every bind), held for `holder` until
/// the instance closes; a ring of a channel `registered` names is the
/// channel's. A pool short of them refuses the bind by name. Without
/// `roles`, `binding` as it is: the engine keeps it all itself.
///
/// [`Programs::guest_state`]: ::engine::Programs::guest_state
pub fn bind_instance(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    binding: &InstanceBinding,
    holder: Option<Holder>,
    registered: impl Fn(ChannelId) -> bool,
) -> anyhow::Result<BoundInstance> {
    let taken = take_bind(
        roles,
        engine,
        binding,
        registered,
        holder,
        |_| true,
    )?;
    bind_taken(roles, engine, binding, holder, taken)
}
