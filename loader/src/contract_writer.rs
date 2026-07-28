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

use crate::contract::{Expr, ModelContract};
use crate::ffi::contract::{
    PieLoaderExprKind, PieLoaderExprNode, PieLoaderExprNodeSlice, PieLoaderModelContractView,
    PieLoaderRepackSpecView, PieLoaderTensorContractSlice, PieLoaderTensorContractView,
    write_encoding, write_quant,
};
use crate::ffi::types::*;
use crate::types::RepackSpec;

/// A contract flattened into the POD form, with its backing storage.
///
/// Handles are indices, so nothing here points into anything that moves; the
/// `Box`ed backing stores exist because the POD views point into *them*.
pub struct OwnedContract {
    abi_version: u32,
    alignment: u32,
    nodes: Vec<PieLoaderExprNode>,
    tensors: Vec<PieLoaderTensorContractView>,
    names: Vec<Box<str>>,
    shapes: Vec<Box<[i64]>>,
    parts: Vec<Box<[u32]>>,
}

impl OwnedContract {
    /// A borrowed POD view, exactly as a C++ `ModelContract::view()` produces.
    pub fn view(&self) -> PieLoaderModelContractView {
        PieLoaderModelContractView {
            abi_version: self.abi_version,
            alignment: self.alignment,
            nodes: PieLoaderExprNodeSlice {
                ptr: self.nodes.as_ptr(),
                len: self.nodes.len(),
            },
            tensors: PieLoaderTensorContractSlice {
                ptr: self.tensors.as_ptr(),
                len: self.tensors.len(),
            },
        }
    }

    /// How many nodes the flattening produced. Shared subexpressions are *not*
    /// deduplicated: the tree is written as a tree, so a round-trip compares
    /// exactly what was written.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
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
            Expr::Slice {
                src,
                axis,
                start,
                len,
                step,
            } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Slice as u32;
                node.src = src;
                node.axis = axis.0;
                node.start = *start;
                node.len = *len;
                node.step = *step;
            }
            Expr::Cat { axis, parts } => {
                let indices: Vec<u32> = parts.iter().map(|part| self.write_expr(part)).collect();
                node.kind = PieLoaderExprKind::Cat as u32;
                node.axis = axis.0;
                node.parts = self.part_list(&indices);
            }
            Expr::Reshape { src, shape } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Reshape as u32;
                node.src = src;
                node.shape = self.shape(shape);
            }
            Expr::Pad {
                src,
                axis,
                before,
                after,
            } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Pad as u32;
                node.src = src;
                node.axis = axis.0;
                node.before = *before;
                node.after = *after;
            }
            Expr::Shard { src, axis } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Shard as u32;
                node.src = src;
                node.axis = axis.0;
            }
            Expr::Repack { src, spec, out } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Repack as u32;
                node.src = src;
                node.repack = write_repack(spec);
                node.out_shape = self.shape(&out.shape);
                node.out_encoding = write_encoding(&out.encoding);
            }
            Expr::Quantize { src, spec } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Quantize as u32;
                node.src = src;
                node.quant = write_quant(spec);
            }
            Expr::Bitcast { src, out } => {
                let src = self.write_expr(src);
                node.kind = PieLoaderExprKind::Bitcast as u32;
                node.src = src;
                node.out_shape = self.shape(&out.shape);
                node.out_encoding = write_encoding(&out.encoding);
            }
        }
        self.push(node)
    }
}

fn write_repack(spec: &RepackSpec) -> PieLoaderRepackSpecView {
    PieLoaderRepackSpecView {
        layout: PieLoaderRepackLayout::from(spec.layout) as u32,
        row_map: PieLoaderRowMap::from(spec.row_map) as u32,
        batch: spec.batch,
        source_rows: spec.source_rows,
        source_row_offset: spec.source_row_offset,
        target_rows: spec.target_rows,
        valid_rows: spec.valid_rows,
        source_stride_cols: spec.source_stride_cols,
        source_col_offset: spec.source_col_offset,
        source_cols: spec.source_cols,
        target_cols: spec.target_cols,
    }
}

/// Flatten a contract into the POD form a driver would have written by hand.
pub fn write_contract(contract: &ModelContract) -> OwnedContract {
    let mut owned = OwnedContract {
        abi_version: contract.abi_version,
        alignment: contract.alignment,
        nodes: Vec::new(),
        tensors: Vec::new(),
        names: Vec::new(),
        shapes: Vec::new(),
        parts: Vec::new(),
    };
    // Two passes over each tensor, because `name`, `shape` and the node array
    // all borrow from `owned`: the borrow checker is enforcing the same rule
    // the C++ side has to follow by hand, which is that nothing may point into
    // a store while it is being appended to.
    for tensor in &contract.tensors {
        let root = owned.write_expr(&tensor.expr);
        let name = owned.name(&tensor.name);
        let shape = match &tensor.shape {
            Some(shape) => owned.shape(shape),
            None => PieLoaderI64Slice::default(),
        };
        let encoding = write_encoding(&tensor.encoding);
        owned.tensors.push(PieLoaderTensorContractView {
            name,
            root,
            shape,
            encoding,
        });
    }
    owned
}
