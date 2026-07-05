//! Minimal generational slot map for typed store ids.
//!
//! Keys are generation-tagged so a recycled slot never aliases a stale handle.
//! `M` is a zero-sized marker type; keys of maps with different markers are
//! mutually untypable.

use std::marker::PhantomData;

/// Generation-tagged key into a [`GenMap`].
pub struct GenKey<M> {
    index: u32,
    generation: u32,
    _marker: PhantomData<M>,
}

impl<M> GenKey<M> {
    fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }
}

impl<M> Clone for GenKey<M> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<M> Copy for GenKey<M> {}
impl<M> PartialEq for GenKey<M> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<M> Eq for GenKey<M> {}
impl<M> std::hash::Hash for GenKey<M> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}
impl<M> std::fmt::Debug for GenKey<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}v{}", self.index, self.generation)
    }
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Generational slot map.
pub struct GenMap<M, T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    len: usize,
    _marker: PhantomData<M>,
}

impl<M, T> Default for GenMap<M, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<M, T> GenMap<M, T> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
            _marker: PhantomData,
        }
    }

    pub fn insert(&mut self, value: T) -> GenKey<M> {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(slot.value.is_none());
            slot.value = Some(value);
            GenKey::new(index, slot.generation)
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot {
                generation: 0,
                value: Some(value),
            });
            GenKey::new(index, 0)
        }
    }

    pub fn get(&self, key: GenKey<M>) -> Option<&T> {
        let slot = self.slots.get(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_ref()
    }

    pub fn get_mut(&mut self, key: GenKey<M>) -> Option<&mut T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_mut()
    }

    pub fn contains(&self, key: GenKey<M>) -> bool {
        self.get(key).is_some()
    }

    pub fn remove(&mut self, key: GenKey<M>) -> Option<T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation || slot.value.is_none() {
            return None;
        }
        let value = slot.value.take();
        // Bump the generation on removal so stale keys can never resolve.
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(key.index);
        self.len -= 1;
        value
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = (GenKey<M>, &T)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.value
                .as_ref()
                .map(|value| (GenKey::new(index as u32, slot.generation), value))
        })
    }

    pub fn keys(&self) -> impl Iterator<Item = GenKey<M>> + '_ {
        self.iter().map(|(key, _)| key)
    }
}
