//! Which kernel a prefill attention fire takes: the tiled kernel (a tile of
//! rows a workgroup a head, every row of the tile walking the tile's keys
//! in step) or the per-row walk (a row a workgroup a head), whichever the
//! device's roofline prices cheaper at the fire's shape. No row threshold,
//! no knob: each kernel costs its work over its rate, scaled by its wave
//! fill (`engine::roofline::Constants::wave_ns`, `[rates]
//! workgroups_per_core`):
//!
//! - the tiled kernel is compute-bound: its tiles' rows (`ceil(rows /
//!   tile) · tile`) each walk a key at `4 · q_heads · head_dim` flops, at
//!   the attention curve's compute-bound rate (`[rates]
//!   attn_tflops_by_keys` at its longest keys, timed on the tiled kernel;
//!   the curve counts a causal walk's whole rectangle, of which the kernel
//!   computed half, so its flops count twice: [`tiled_flops`]), over
//!   `tiles · q_heads` workgroups;
//! - the per-row walk is bound by its reads: each row reads its keys' K and
//!   V (`kv_heads · head_dim` bf16 each) on its own, at `[rates]
//!   hbm_gbps`, over `rows · q_heads` workgroups.
//!
//! The keys scale both costs alike, so the choice is the fire's rows and
//! heads alone, the same at record time and at every replay. A profile
//! that states none of a rate the choice reads is refused by its key: no
//! choice is made without it.

use engine::roofline::Constants;

/// A prefill attention fire's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire {
    /// The fire's query rows, every request's.
    pub rows: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// The rows a tile of the tiled kernel holds.
    pub tile_rows: u32,
}

/// Each kernel's cost a key, in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Costs {
    pub tiled: f64,
    pub walked: f64,
}

impl Costs {
    /// Whether the tiled kernel is the cheaper.
    #[must_use]
    pub fn tiles(&self) -> bool {
        self.tiled < self.walked
    }
}

/// The work of `rows` rows each walking `keys` keys of the tiled kernel,
/// as the attention curve's rate counts it: `4 · rows · keys · heads ·
/// head_dim` flops computed, twice over (the curve's rate counts a causal
/// walk's whole rectangle, of which its kernel computed half).
#[must_use]
pub fn tiled_flops(rows: u32, keys: u32, heads: u32, head_dim: u32) -> f64 {
    2.0 * 4.0 * f64::from(rows) * f64::from(keys) * f64::from(heads) * f64::from(head_dim)
}

/// The two kernels' costs a key at `fire` on a device of `cores` compute
/// units, as `rates` price them; the key of the first rate they need and
/// the profile leaves unstated.
pub fn costs(fire: Fire, rates: &Constants, cores: u32) -> Result<Costs, &'static str> {
    let rate = rates
        .attn_by_keys
        .last()
        .map(|&(_, r)| r)
        .ok_or("attn_tflops_by_keys")?;
    let tile = fire.tile_rows.max(1);
    let tiles = fire.rows.div_ceil(tile).max(1);
    let heads = fire.q_heads.max(1);
    let tiled = rates.wave_ns(
        tiled_flops(tiles * tile, 1, heads, fire.head_dim),
        rate,
        tiles.saturating_mul(heads),
        cores,
    )?;
    let bytes = f64::from(fire.rows) * 2.0 * 2.0 * f64::from(fire.kv_heads * fire.head_dim);
    let walked = rates.wave_ns(
        bytes,
        rates.bytes_per_ns,
        fire.rows.max(1).saturating_mul(heads),
        cores,
    )?;
    Ok(Costs { tiled, walked })
}

/// Whether `fire` takes the tiled kernel: the cheaper of the two at
/// [`costs`], or the key of the rate the choice lacks.
pub fn tiles(fire: Fire, rates: &Constants, cores: u32) -> Result<bool, &'static str> {
    costs(fire, rates, cores).map(|c| c.tiles())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rates an RTX PRO 4500 (82 cores) measured on Vulkan: the tiled
    /// attention's curve, the copy's bandwidth and its occupancy.
    fn measured() -> Constants {
        Constants::of(&profile::fire::Rates {
            launch_ns: Some(5000),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 40.0)])]),
            attn_tflops_by_keys: Some(vec![(2048, 7.2), (8192, 7.0)]),
            hbm_gbps: Some(737.0),
            elementwise_gbps: Some(500.0),
            scan_gelems: Some(100.0),
            few_row_tflops: Some(4.0),
            workgroups_per_core: Some(3.3),
        })
        .unwrap()
    }

    fn fire(rows: u32) -> Fire {
        Fire {
            rows,
            q_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            tile_rows: 16,
        }
    }

    /// A few rows walk, a full prompt tiles, with no row threshold: where
    /// the tiled kernel's tiles fill the device and its rate passes the
    /// rows' own reads.
    #[test]
    fn a_few_rows_walk_and_a_prompt_tiles() {
        let r = measured();
        for rows in [1, 8, 16, 48, 64, 80, 96] {
            assert_eq!(tiles(fire(rows), &r, 82), Ok(false), "{rows} rows walk");
        }
        for rows in [128, 256, 512, 2048] {
            assert_eq!(tiles(fire(rows), &r, 82), Ok(true), "{rows} rows tile");
        }
    }

    /// A choice the profile cannot price is refused by the key it lacks,
    /// never guessed.
    #[test]
    fn an_unpriced_fire_is_refused_by_the_key_it_lacks() {
        let no_occupancy = Constants {
            workgroups_per_core: None,
            ..measured()
        };
        assert_eq!(
            tiles(fire(64), &no_occupancy, 82),
            Err("workgroups_per_core")
        );
        let no_curve = Constants {
            attn_by_keys: Vec::new(),
            ..measured()
        };
        assert_eq!(tiles(fire(64), &no_curve, 82), Err("attn_tflops_by_keys"));
    }
}
