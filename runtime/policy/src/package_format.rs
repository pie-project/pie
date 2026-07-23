use pie_plex::Manifest;
use thiserror::Error;

const MAGIC: &[u8; 8] = b"PLEXPKG\0";
const FORMAT_VERSION: u16 = 6;
const HEADER_BYTES: usize = 56;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackageLimits {
    pub max_package_bytes: usize,
    pub max_manifest_bytes: usize,
    pub max_component_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyPackage {
    manifest: Manifest,
    component: Vec<u8>,
    digest: [u8; 32],
}

impl PolicyPackage {
    pub fn new(manifest: Manifest, component: Vec<u8>) -> Result<Self, PackageError> {
        manifest.validate().map_err(PackageError::Manifest)?;
        let manifest_bytes = serde_json::to_vec(&manifest).map_err(PackageError::EncodeManifest)?;
        Ok(Self {
            digest: digest(&manifest_bytes, &component),
            manifest,
            component,
        })
    }

    pub fn decode(bytes: &[u8], limits: PackageLimits) -> Result<Self, PackageError> {
        if bytes.len() > limits.max_package_bytes {
            return Err(PackageError::PackageTooLarge {
                actual: bytes.len(),
                maximum: limits.max_package_bytes,
            });
        }
        if bytes.len() < HEADER_BYTES {
            return Err(PackageError::TruncatedHeader);
        }
        if &bytes[..8] != MAGIC {
            return Err(PackageError::InvalidMagic);
        }
        let version = u16::from_le_bytes(bytes[8..10].try_into().expect("version bytes"));
        if version != FORMAT_VERSION {
            return Err(PackageError::UnsupportedFormat(version));
        }
        let flags = u16::from_le_bytes(bytes[10..12].try_into().expect("flags bytes"));
        if flags != 0 {
            return Err(PackageError::UnsupportedFlags(flags));
        }
        let manifest_len = usize::try_from(u32::from_le_bytes(
            bytes[12..16].try_into().expect("manifest length bytes"),
        ))
        .map_err(|_| PackageError::LengthOverflow)?;
        let component_len = usize::try_from(u64::from_le_bytes(
            bytes[16..24].try_into().expect("component length bytes"),
        ))
        .map_err(|_| PackageError::LengthOverflow)?;
        if manifest_len > limits.max_manifest_bytes {
            return Err(PackageError::ManifestTooLarge {
                actual: manifest_len,
                maximum: limits.max_manifest_bytes,
            });
        }
        if component_len > limits.max_component_bytes {
            return Err(PackageError::ComponentTooLarge {
                actual: component_len,
                maximum: limits.max_component_bytes,
            });
        }
        let expected_len = HEADER_BYTES
            .checked_add(manifest_len)
            .and_then(|length| length.checked_add(component_len))
            .ok_or(PackageError::LengthOverflow)?;
        if expected_len != bytes.len() {
            return Err(PackageError::LengthMismatch {
                declared: expected_len,
                actual: bytes.len(),
            });
        }

        let expected_digest: [u8; 32] = bytes[24..56].try_into().expect("digest bytes");
        let manifest_end = HEADER_BYTES + manifest_len;
        let manifest_bytes = &bytes[HEADER_BYTES..manifest_end];
        let component = &bytes[manifest_end..];
        let actual_digest = digest(manifest_bytes, component);
        if expected_digest != actual_digest {
            return Err(PackageError::DigestMismatch);
        }
        let manifest: Manifest =
            serde_json::from_slice(manifest_bytes).map_err(PackageError::DecodeManifest)?;
        manifest.validate().map_err(PackageError::Manifest)?;
        Ok(Self {
            manifest,
            component: component.to_vec(),
            digest: actual_digest,
        })
    }

    pub fn encode(&self) -> Result<Vec<u8>, PackageError> {
        let manifest = serde_json::to_vec(&self.manifest).map_err(PackageError::EncodeManifest)?;
        let manifest_len =
            u32::try_from(manifest.len()).map_err(|_| PackageError::LengthOverflow)?;
        let component_len =
            u64::try_from(self.component.len()).map_err(|_| PackageError::LengthOverflow)?;
        let mut encoded = Vec::with_capacity(
            HEADER_BYTES
                .checked_add(manifest.len())
                .and_then(|length| length.checked_add(self.component.len()))
                .ok_or(PackageError::LengthOverflow)?,
        );
        encoded.extend_from_slice(MAGIC);
        encoded.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        encoded.extend_from_slice(&0u16.to_le_bytes());
        encoded.extend_from_slice(&manifest_len.to_le_bytes());
        encoded.extend_from_slice(&component_len.to_le_bytes());
        encoded.extend_from_slice(&digest(&manifest, &self.component));
        encoded.extend_from_slice(&manifest);
        encoded.extend_from_slice(&self.component);
        Ok(encoded)
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn component(&self) -> &[u8] {
        &self.component
    }

    pub fn into_parts(self) -> (Manifest, Vec<u8>) {
        (self.manifest, self.component)
    }
}

fn digest(manifest: &[u8], component: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(manifest.len() as u64).to_le_bytes());
    hasher.update(manifest);
    hasher.update(&(component.len() as u64).to_le_bytes());
    hasher.update(component);
    *hasher.finalize().as_bytes()
}

#[derive(Debug, Error)]
pub enum PackageError {
    #[error("package contains {actual} bytes; maximum is {maximum}")]
    PackageTooLarge { actual: usize, maximum: usize },
    #[error("manifest contains {actual} bytes; maximum is {maximum}")]
    ManifestTooLarge { actual: usize, maximum: usize },
    #[error("component contains {actual} bytes; maximum is {maximum}")]
    ComponentTooLarge { actual: usize, maximum: usize },
    #[error("package header is truncated")]
    TruncatedHeader,
    #[error("package magic is invalid")]
    InvalidMagic,
    #[error("package format version {0} is unsupported")]
    UnsupportedFormat(u16),
    #[error("package flags {0:#x} are unsupported")]
    UnsupportedFlags(u16),
    #[error("package length overflows the host address space")]
    LengthOverflow,
    #[error("package declares {declared} bytes but contains {actual}")]
    LengthMismatch { declared: usize, actual: usize },
    #[error("package digest does not match its manifest and component")]
    DigestMismatch,
    #[error("failed to encode package manifest")]
    EncodeManifest(#[source] serde_json::Error),
    #[error("failed to decode package manifest")]
    DecodeManifest(#[source] serde_json::Error),
    #[error("package manifest is invalid")]
    Manifest(#[source] pie_plex::ManifestValidationError),
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pie_plex::{ContractVersion, Operation, PolicyLimits};

    use super::*;

    fn manifest() -> Manifest {
        Manifest {
            contract: ContractVersion::V0_5,
            package_name: "json-package".into(),
            package_version: "0.5.0".into(),
            operations: BTreeSet::from([Operation::Route]),
            limits: PolicyLimits {
                memory_bytes: 1,
                deadline_ms: 1,
                input_bytes: 1,
                output_bytes: 1,
            },
        }
    }

    #[test]
    fn deterministic_round_trip_and_corruption_rejection() {
        let package = PolicyPackage::new(manifest(), vec![0, 1, 2]).unwrap();
        let encoded = package.encode().unwrap();
        let decoded = PolicyPackage::decode(
            &encoded,
            PackageLimits {
                max_package_bytes: 4096,
                max_manifest_bytes: 2048,
                max_component_bytes: 1024,
            },
        )
        .unwrap();
        assert_eq!(decoded, package);

        let mut corrupted = encoded;
        *corrupted.last_mut().unwrap() ^= 1;
        assert!(matches!(
            PolicyPackage::decode(
                &corrupted,
                PackageLimits {
                    max_package_bytes: 4096,
                    max_manifest_bytes: 2048,
                    max_component_bytes: 1024,
                }
            ),
            Err(PackageError::DigestMismatch)
        ));
    }
}
