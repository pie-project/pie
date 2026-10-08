//! A contract's constants as the source tensors a plan reads: each `Const`
//! is mounted in memory and stated in the metadata beside the checkpoint's own
//! tensors, so lowering, the plan and its execution read it as they read a
//! plane, wherever the plan compiles.

use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use crate::contract::{Expr, ModelContract};
use crate::error::{Error, Result};
use crate::file::{File, Metadata, RawTensor};
use crate::types::{CheckpointFormat, FileId, TensorId};

/// The prefix a mounted constant's tensor is named under.
pub const CONST_PREFIX: &str = "__const__/";

/// `contract` with every constant read as a mounted source tensor, and
/// `metadata` stating those tensors.
pub fn mounted<'a>(
    metadata: &'a Metadata,
    contract: &'a ModelContract,
) -> Result<(Cow<'a, Metadata>, Cow<'a, ModelContract>)> {
    let mut any = false;
    for tensor in &contract.tensors {
        tensor
            .expr
            .visit(&mut |expr| any |= matches!(expr, Expr::Const { .. }));
    }
    if !any {
        return Ok((Cow::Borrowed(metadata), Cow::Borrowed(contract)));
    }
    let mut metadata = metadata.clone();
    let mut contract = contract.clone();
    for tensor in &mut contract.tensors {
        let expr = std::mem::replace(&mut tensor.expr, Expr::Src(String::new()));
        tensor.expr = mount(expr, &mut metadata)?;
    }
    Ok((Cow::Owned(metadata), Cow::Owned(contract)))
}

fn mount(expr: Expr, metadata: &mut Metadata) -> Result<Expr> {
    let Expr::Const { ty, bytes } = expr else {
        return expr.map_children(|child| mount(child, metadata));
    };
    let want = ty.byte_size()?;
    if bytes.len() as u64 != want {
        return Err(Error::Contract(format!(
            "Const of shape {:?} as {:?} is {want} bytes and carries {}",
            ty.shape,
            ty.encoding,
            bytes.len()
        )));
    }
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    ty.shape.hash(&mut hasher);
    format!("{:?}", ty.encoding).hash(&mut hasher);
    let key = format!("{:016x}-{}", hasher.finish(), bytes.len());
    let name = format!("{CONST_PREFIX}{key}");
    if !metadata.tensors.iter().any(|tensor| tensor.name == name) {
        // Absolute, so the plan's file root joins onto nothing.
        let path = format!("/__pie_const__/{key}");
        let size_bytes = bytes.len() as u64;
        ztensor::memfs::mount(&path, Arc::from(bytes));
        let file = FileId(metadata.files.len() as u32);
        metadata.files.push(File {
            id: file,
            path,
            size_bytes,
            format: CheckpointFormat::Unknown,
        });
        metadata.tensors.push(RawTensor {
            id: TensorId(metadata.tensors.len() as u32),
            name: name.clone(),
            file_id: file,
            file_offset: 0,
            span_bytes: size_bytes,
            shape: ty.shape.clone(),
            encoding: ty.encoding.clone(),
        });
    }
    Ok(Expr::Src(name))
}
