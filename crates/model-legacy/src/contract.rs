use model_loader::checkpoint::CheckpointMetadata;
use model_loader::contract::ModelContract;
use model_loader::error::Error;
use model_loader::plan::StorageTarget;

use crate::shared::builder::Builder;
use crate::shared::policy::{Mxfp4MoePolicy, Policy};

pub type Author = fn(&mut Builder<'_>) -> Result<(), Error>;

pub fn author_with_policy(
    row: &dyn crate::catalog::Variant,
    encoding: &crate::encoding::Encoding,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    policy: &Policy,
) -> Result<(ModelContract, Mxfp4MoePolicy), Error> {
    let mut builder = Builder::new(
        metadata,
        row.id(),
        row.load_shape(),
        encoding,
        target,
        policy,
    );
    row.author(&mut builder)?;
    let resolved = builder.mxfp4_moe();
    builder.finish().map(|contract| (contract, resolved))
}

pub fn author(
    row: &dyn crate::catalog::Variant,
    encoding: &crate::encoding::Encoding,
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    policy: &Policy,
) -> Result<ModelContract, Error> {
    author_with_policy(row, encoding, metadata, target, policy).map(|(contract, _)| contract)
}
