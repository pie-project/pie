//! Writing a `ModelContract` into the POD form C++ owns in production.
//!
//! Test support, and only that. The loader *reads* this format: a driver
//! authors the contract in C++ and `ffi::contract::read_contract` borrows it.
//! Nothing on the load path constructs one from a Rust `ModelContract`.
//!
//! It exists so the round-trip is assertable — every golden plan checks
//! `read_contract(write_contract(c)) == c`, which is what makes moving a
//! family's authorship into C++ a *checkable* refactor rather than a rewrite
//! that is only as good as its reviewer. Contract equality implies plan
//! equality, so it is a stronger claim than comparing plans.
//!
//! It lives outside `ffi/` because `ffi/` is the crate's C surface and this is
//! not part of it: no `extern "C"` function reaches this code, and cbindgen
//! emits nothing from it.

use crate::contract::{Expr, ModelContract, ScaleFactor, Visibility};
use crate::ffi::contract::{
    PieLoaderExprKind, PieLoaderExprNode, PieLoaderExprNodeSlice, PieLoaderGroupContractSlice,
    PieLoaderGroupContractView, PieLoaderModelContractView, PieLoaderScalesView,
    PieLoaderTensorContractSlice, PieLoaderTensorContractView, write_encoding,
};
use crate::ffi::types::*;

/// A contract flattened into the POD form, with its backing storage.
///
/// Handles are indices, so nothing here points into anything that moves; the
/// `Box`ed backing stores exist because the POD views point into *them*.
pub struct OwnedContract {
    alignment: u32,
    nodes: Vec<PieLoaderExprNode>,
    tensors: Vec<PieLoaderTensorContractView>,
    names: Vec<Box<str>>,
    shapes: Vec<Box<[i64]>>,
    parts: Vec<Box<[u32]>>,
    groups: Vec<PieLoaderGroupContractView>,
    /// Each group's declarations, boxed so the views above point at storage
    /// that does not move when a later group is appended.
    group_tensors: Vec<Box<[PieLoaderTensorContractView]>>,
}

impl OwnedContract {
    /// A borrowed POD view, exactly as a C++ `ModelContract::view()` produces.
    pub fn view(&self) -> PieLoaderModelContractView {
        PieLoaderModelContractView {
            alignment: self.alignment,
            nodes: PieLoaderExprNodeSlice {
                ptr: self.nodes.as_ptr(),
                len: self.nodes.len(),
            },
            tensors: PieLoaderTensorContractSlice {
                ptr: self.tensors.as_ptr(),
                len: self.tensors.len(),
            },
            groups: PieLoaderGroupContractSlice {
                ptr: self.groups.as_ptr(),
                len: self.groups.len(),
            },
        }
    }

    /// How many nodes the flattening produced. Shared subexpressions are *not*
    /// deduplicated: the tree is written as a tree, so a round-trip compares
    /// exactly what was written.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Write raw integers into a declared tensor's `scales` enum codes.
    ///
    /// `granularity` and `form` are `u32` on the ABI because they are *input*
    /// fields: the value is whatever the driver wrote, and a Rust enum holding
    /// something outside its variants is undefined behaviour before any check
    /// could run. A well-typed writer cannot express a value a driver could
    /// nonetheless produce, so a test that wants to prove the loader *rejects*
    /// one reaches past the typed builder here.
    pub fn set_raw_scale_codes(&mut self, tensor: usize, granularity: u32, form: u32) {
        let scales = &mut self.tensors[tensor].scales;
        scales.granularity = granularity;
        scales.form = form;
    }

    /// Position of the first declaration that carries scales.
    pub fn first_scaled(&self) -> Option<usize> {
        self.tensors
            .iter()
            .position(|tensor| tensor.scales.of.len != 0)
    }

    /// Write a raw integer into a node's `repack_layout`, for the same reason
    /// [`set_raw_scale_codes`](Self::set_raw_scale_codes) exists.
    ///
    /// [`RepackLayout`](crate::types::RepackLayout) has no member for "no
    /// layout", so a typed builder cannot express the zero an all-zero node
    /// carries — which is exactly the value the boundary has to reject.
    pub fn set_raw_repack_layout(&mut self, node: usize, layout: u32) {
        self.nodes[node].repack_layout = layout;
    }

    /// Position of the first `Repack` node.
    pub fn first_repack(&self) -> Option<usize> {
        self.nodes
            .iter()
            .position(|node| node.kind == PieLoaderExprKind::Repack as u32)
    }

    fn name(&mut self, value: &str) -> PieLoaderBytes {
        self.names.push(value.into());
        let stored = self.names.last().unwrap();
        PieLoaderBytes {
            ptr: stored.as_ptr(),
            len: stored.len(),
        }
    }

    fn shape(&mut self, value: &[i64]) -> PieLoaderI64Slice {
        self.shapes.push(value.into());
        let stored = self.shapes.last().unwrap();
        PieLoaderI64Slice {
            ptr: stored.as_ptr(),
            len: stored.len(),
        }
    }

    fn part_list(&mut self, value: &[u32]) -> PieLoaderU32Slice {
        self.parts.push(value.into());
        let stored = self.parts.last().unwrap();
        PieLoaderU32Slice {
            ptr: stored.as_ptr(),
            len: stored.len(),
        }
    }

    fn push(&mut self, node: PieLoaderExprNode) -> u32 {
        self.nodes.push(node);
        (self.nodes.len() - 1) as u32
    }

    /// Flatten one expression, returning the index of its root.
    ///
    /// Post-order, so every operand is already in the array when its parent is
    /// pushed — which is the index discipline `read_expr` relies on, established
    /// here by construction rather than checked afterwards.
    fn write_expr(&mut self, expr: &Expr) -> u32 {
        let mut node = PieLoaderExprNode::default();
        match expr {
            Expr::Src(name) => {
                node.kind = PieLoaderExprKind::Src as u32;
                node.name = self.name(name);
            }
            Expr::Out(name) => {
                node.kind = PieLoaderExprKind::Out as u32;
                node.name = self.name(name);
            }
            Expr::SrcIndexed(template) => {
                node.kind = PieLoaderExprKind::SrcIndexed as u32;
                node.name = self.name(template);
            }
            Expr::Select {
                src,
                axis,
                stride,
                len,
            } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Select as u32;
                node.src = src;
                node.axis = axis.0;
                // The stride rides in `start`; see `read_expr`.
                node.start = *stride;
                node.len = *len;
            }
            Expr::Slice {
                src,
                axis,
                start,
                len,
            } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Slice as u32;
                node.src = src;
                node.axis = axis.0;
                node.start = *start;
                node.len = *len;
            }
            Expr::Stride {
                src,
                axis,
                start,
                len,
                step,
            } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Stride as u32;
                node.src = src;
                node.axis = axis.0;
                node.start = *start;
                node.len = *len;
                node.step = *step;
            }
            Expr::Gather { src, axis, indices } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Gather as u32;
                node.src = src;
                node.axis = axis.0;
                node.indices = self.shape(indices);
            }
            Expr::Concat { axis, parts } => {
                let indices: Vec<u32> = parts.iter().map(|part| self.write_expr(part)).collect();
                node.kind = PieLoaderExprKind::Concat as u32;
                node.axis = axis.0;
                node.parts = self.part_list(&indices);
            }
            Expr::Transmute { src, to } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Transmute as u32;
                node.src = src;
                node.out_shape = self.shape(&to.shape);
                node.out_encoding = write_encoding(&to.encoding);
            }
            Expr::Fill { value, ty } => {
                node.kind = PieLoaderExprKind::Fill as u32;
                node.fill_bits = *value;
                node.out_shape = self.shape(&ty.shape);
                node.out_encoding = write_encoding(&ty.encoding);
            }
            Expr::Shard { src, axis } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Shard as u32;
                node.src = src;
                node.axis = axis.0;
            }
            Expr::Repack { src, layout, to } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Repack as u32;
                node.src = src;
                node.repack_layout = PieLoaderRepackLayout::from(*layout) as u32;
                node.out_shape = self.shape(&to.shape);
                node.out_encoding = write_encoding(&to.encoding);
            }
            Expr::Cast { src, to } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Cast as u32;
                node.src = src;
                node.out_encoding = write_encoding(to);
            }
            Expr::Scale { src, factor } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Scale as u32;
                node.src = src;
                match factor {
                    ScaleFactor::Uniform(bits) => node.scale_factor_bits = *bits,
                    ScaleFactor::PerBlock { by } => {
                        node.scale_by = self.write_expr(by);
                    }
                }
            }
        }
        self.push(node)
    }
}

/// Flatten a contract into the POD form a driver would have written by hand.
pub fn write_contract(contract: &ModelContract) -> OwnedContract {
    let mut owned = OwnedContract {
        alignment: contract.alignment,
        nodes: Vec::new(),
        tensors: Vec::new(),
        names: Vec::new(),
        shapes: Vec::new(),
        parts: Vec::new(),
        groups: Vec::new(),
        group_tensors: Vec::new(),
    };
    // Two passes over each tensor, because `name`, `shape` and the node array
    // all borrow from `owned`: the borrow checker is enforcing the same rule
    // the C++ side has to follow by hand, which is that nothing may point into
    // a store while it is being appended to.
    for tensor in &contract.tensors {
        let view = owned.write_tensor(tensor);
        owned.tensors.push(view);
    }
    for group in &contract.groups {
        let mut tensors = Vec::with_capacity(group.tensors.len());
        for tensor in &group.tensors {
            let view = owned.write_tensor(tensor);
            tensors.push(view);
        }
        let boxed = tensors.into_boxed_slice();
        let view = PieLoaderGroupContractView {
            name: owned.name(&group.name),
            arity: group.arity,
            tensors: PieLoaderTensorContractSlice {
                ptr: boxed.as_ptr(),
                len: boxed.len(),
            },
        };
        owned.group_tensors.push(boxed);
        owned.groups.push(view);
    }
    owned
}

impl OwnedContract {
    /// Flatten one declaration. Shared by the contract's tensors and a group's.
    fn write_tensor(
        &mut self,
        tensor: &crate::contract::TensorContract,
    ) -> PieLoaderTensorContractView {
        let owned = self;
        let root = owned.write_expr(&tensor.expr);
        let name = owned.name(&tensor.name);
        let shape = match &tensor.shape {
            Some(shape) => owned.shape(shape),
            None => PieLoaderI64Slice::default(),
        };
        let encoding = write_encoding(&tensor.encoding);
        let scales = match &tensor.scales {
            Some(scales) => PieLoaderScalesView {
                of: owned.name(&scales.of),
                granularity: PieLoaderQuantGranularity::from(scales.granularity) as u32,
                group_size: scales.group_size,
                channel_axis: scales.channel_axis,
                form: PieLoaderScaleForm::from(scales.form) as u32,
            },
            None => PieLoaderScalesView::default(),
        };
        PieLoaderTensorContractView {
            name,
            root,
            shape,
            encoding,
            scales,
            visibility: match tensor.visibility {
                Visibility::Public => PieLoaderVisibility::Public,
                Visibility::Internal => PieLoaderVisibility::Internal,
            },
        }
    }
}
