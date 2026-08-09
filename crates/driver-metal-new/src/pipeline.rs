//! The PTIR channel plane — re-exported, not owned.
//!
//! This was `src/pipeline/`, twenty files and ten thousand lines, until the
//! CUDA shell reached the same layer. Not one of those lines named a Metal
//! type: the directory built and tested on Linux while the crate around it did
//! not, which is the mechanical form of the claim `PARITY-M1.md` opens with —
//! *"the portable half — everything that is a function of the plan and the
//! fire's numbers rather than of the device — goes first, because it is the
//! half that can be tested without a GPU."*
//!
//! `PARITY-INTERP.md` named the destination before the move happened:
//! *"`driver-cuda-new` shares `driver-abi` and `tensor-ir` with this crate, so
//! `src/pipeline/` is the natural single home for both device copies when that
//! port reaches this file."* It has, so this file is the re-export and
//! [`driver_pipeline`] is the home.
//!
//! The path stays `driver_metal_new::pipeline::*` on purpose. Every call site
//! in `src/metal/` and `src/batch/` names it, the names did not change, and a
//! rename would have made a move that changes no behaviour look like one that
//! does.

pub use driver_pipeline::*;
