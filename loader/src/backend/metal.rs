//! Metal lowering rules.
//!
//! Metal's load executor binds tensors into a heap and runs no tile-map
//! transforms at all, which is what `TILE_MAP_MASK == 0` says. There is
//! therefore no tiling or fusion decision to make: the loader will not emit a
//! transform this backend cannot run, so every `TileMap` that reaches here is
//! already a contradiction the planner should have rejected.
//!
//! This module exists as a named home for that fact rather than as a stub. When
//! Metal grows transform kernels, the mask and the rules below are the two
//! things that change, and they change together.

use super::{Backend, TileLowering, TileMapFacts};
use crate::load_plan::StorageTarget;

pub const TILE_MAP_MASK: u32 = 0;

pub struct Metal;

impl Backend for Metal {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn tile_map_mask(&self) -> u32 {
        TILE_MAP_MASK
    }

    fn lower_tile_map(&self, _facts: &TileMapFacts, _target: &StorageTarget) -> TileLowering {
        TileLowering::default()
    }
}
