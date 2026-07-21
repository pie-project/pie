use pie_plex::Manifest;
use thiserror::Error;

const MAGIC: &[u8; 8] = b"PLEXPKG\0";
const FORMAT_VERSION: u16 = 1;
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
        let digest = digest(&manifest_bytes, &component);
        Ok(Self {
            manifest,
            component,
            digest,
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
        let version = u16::from_le_bytes(bytes[8..10].try_into().expect("two-byte version"));
        if version != FORMAT_VERSION {
            return Err(PackageError::UnsupportedFormat(version));
        }
        let flags = u16::from_le_bytes(bytes[10..12].try_into().expect("two-byte flags"));
        if flags != 0 {
            return Err(PackageError::UnsupportedFlags(flags));
        }
        let manifest_len = usize::try_from(u32::from_le_bytes(
            bytes[12..16].try_into().expect("four-byte manifest length"),
        ))
        .map_err(|_| PackageError::LengthOverflow)?;
        let component_len = usize::try_from(u64::from_le_bytes(
            bytes[16..24]
                .try_into()
                .expect("eight-byte component length"),
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
            .and_then(|value| value.checked_add(component_len))
            .ok_or(PackageError::LengthOverflow)?;
        if bytes.len() != expected_len {
            return Err(PackageError::LengthMismatch {
                declared: expected_len,
                actual: bytes.len(),
            });
        }

        let expected_digest: [u8; 32] = bytes[24..56].try_into().expect("fixed digest length");
        let manifest_end = HEADER_BYTES + manifest_len;
        let manifest_bytes = &bytes[HEADER_BYTES..manifest_end];
        let component = &bytes[manifest_end..];
        let actual_digest = digest(manifest_bytes, component);
        if actual_digest != expected_digest {
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
        let total = HEADER_BYTES
            .checked_add(manifest.len())
            .and_then(|value| value.checked_add(self.component.len()))
            .ok_or(PackageError::LengthOverflow)?;
        let mut encoded = Vec::with_capacity(total);
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

    pub fn digest(&self) -> [u8; 32] {
        self.digest
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

    use pie_plex::{ContractVersion, InvocationMode, Operation, PolicyLimits};

    use super::*;

    fn manifest() -> Manifest {
        Manifest {
            contract: ContractVersion::V0_1,
            package_name: "package-test".into(),
            package_version: "0.1.0".into(),
            operations: BTreeSet::from([Operation::Schedule]),
            invocation_mode: InvocationMode::SetDependent,
            capabilities: Vec::new(),
            facts: Vec::new(),
            metadata: Vec::new(),
            events: Vec::new(),
            maps: Vec::new(),
            limits: PolicyLimits {
                memory_bytes: 1,
                fuel: 1,
                deadline_ms: 1,
                input_bytes: 1,
                output_bytes: 1,
                map_calls: 1,
                map_bytes: 1,
                staged_mutations: 1,
                feedback_records: 1,
                telemetry_records: 0,
                telemetry_bytes: 0,
            },
        }
    }

    fn limits() -> PackageLimits {
        PackageLimits {
            max_package_bytes: 4096,
            max_manifest_bytes: 2048,
            max_component_bytes: 1024,
        }
    }

    #[test]
    fn deterministic_round_trip() {
        let package = PolicyPackage::new(manifest(), vec![0, 1, 2, 3]).unwrap();
        let first = package.encode().unwrap();
        let second = package.encode().unwrap();
        assert_eq!(first, second);
        let decoded = PolicyPackage::decode(&first, limits()).unwrap();
        assert_eq!(decoded, package);
    }

    #[test]
    fn rejects_corruption_and_trailing_bytes() {
        let package = PolicyPackage::new(manifest(), vec![0, 1, 2, 3]).unwrap();
        let mut corrupted = package.encode().unwrap();
        *corrupted.last_mut().unwrap() ^= 1;
        assert!(matches!(
            PolicyPackage::decode(&corrupted, limits()),
            Err(PackageError::DigestMismatch)
        ));

        let mut trailing = package.encode().unwrap();
        trailing.push(0);
        assert!(matches!(
            PolicyPackage::decode(&trailing, limits()),
            Err(PackageError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn enforces_declared_limits_before_allocation() {
        let encoded = PolicyPackage::new(manifest(), vec![0; 16])
            .unwrap()
            .encode()
            .unwrap();
        let mut limits = limits();
        limits.max_component_bytes = 8;
        assert!(matches!(
            PolicyPackage::decode(&encoded, limits),
            Err(PackageError::ComponentTooLarge {
                actual: 16,
                maximum: 8
            })
        ));
    }
}
