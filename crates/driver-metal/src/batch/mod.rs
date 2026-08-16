//! The one shape a checkpoint decides, and how a catalog row states it.
//!
//! [`geometry`] holds only the METAL-side numbers (affine point, GDN
//! strides, pool capacities), derived from a `model::deployment::Deployment`.

pub(crate) mod geometry;

pub use geometry::{
    AffineFormat, DecodeGeometry, GeometryRefused, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K,
    geometry_from_deployment,
};
