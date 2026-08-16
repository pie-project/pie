//! Per-family decode smokes for the Metal driver.
//!
//! Empty on purpose: every gate here was keyed on a `PIE_METAL_SMOKE_CHECKPOINT`
//! whose generation this crate does not model, so none of them ran in CI. What
//! they covered runs unconditionally in `device_text_fire`,
//! `device_real_weights` and `text_conformance`.
