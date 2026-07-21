use std::collections::{BTreeMap, BTreeSet};

use pie_plex::{
    CapabilityHandle, DependencyRequirement, EventHandle, FactHandle, FieldLocation, FieldSchema,
    FieldUse, LinkSet, LinkedRecordSchema, Manifest, MapClass, MapDeclaration, MapHandle,
    MapSchema, MetadataHandle, Operation, Symbol, ValueType,
};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
struct FactCapability {
    value_type: ValueType,
    max_value_bytes: u32,
}

#[derive(Debug, Clone, Default)]
pub struct CapabilityCatalog {
    facts: BTreeMap<Symbol, FactCapability>,
    events: BTreeSet<Symbol>,
    capabilities: BTreeSet<Symbol>,
    external_maps: BTreeMap<Symbol, MapSchema>,
}

impl CapabilityCatalog {
    pub fn add_fact(
        &mut self,
        name: Symbol,
        value_type: ValueType,
        max_value_bytes: u32,
    ) -> Result<(), CatalogError> {
        if max_value_bytes == 0 || max_value_bytes < value_type.minimum_payload_bytes() {
            return Err(CatalogError::InvalidFactBounds(name));
        }
        match self.facts.entry(name.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(FactCapability {
                    value_type,
                    max_value_bytes,
                });
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(_) => {
                Err(CatalogError::Duplicate { kind: "fact", name })
            }
        }
    }

    pub fn add_event(&mut self, name: Symbol) -> Result<(), CatalogError> {
        insert_symbol(&mut self.events, "event", name)
    }

    pub fn add_capability(&mut self, name: Symbol) -> Result<(), CatalogError> {
        insert_symbol(&mut self.capabilities, "capability", name)
    }

    pub fn add_external_map(
        &mut self,
        name: Symbol,
        schema: MapSchema,
    ) -> Result<(), CatalogError> {
        match self.external_maps.entry(name.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(schema);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(_) => Err(CatalogError::Duplicate {
                kind: "external map",
                name,
            }),
        }
    }
}

fn insert_symbol(
    set: &mut BTreeSet<Symbol>,
    kind: &'static str,
    name: Symbol,
) -> Result<(), CatalogError> {
    if set.insert(name.clone()) {
        Ok(())
    } else {
        Err(CatalogError::Duplicate { kind, name })
    }
}

#[derive(Debug, Clone)]
pub struct AttachmentResolution {
    links: LinkSet,
    record_schemas: BTreeMap<FieldUse, LinkedRecordSchema>,
    maps: BTreeMap<MapHandle, MapDeclaration>,
    capabilities: BTreeSet<Symbol>,
}

impl AttachmentResolution {
    pub fn links(&self) -> &LinkSet {
        &self.links
    }

    pub fn record_schema(
        &self,
        operation: Operation,
        location: FieldLocation,
    ) -> &LinkedRecordSchema {
        self.record_schemas
            .get(&FieldUse {
                operation,
                location,
            })
            .expect("validated manifest has a schema for every operation input")
    }

    pub fn maps(&self) -> &BTreeMap<MapHandle, MapDeclaration> {
        &self.maps
    }

    pub(crate) fn has_capability(&self, name: &str) -> bool {
        self.capabilities
            .iter()
            .any(|capability| capability.as_str() == name)
    }
}

pub(crate) fn link_manifest(
    manifest: &Manifest,
    catalog: &CapabilityCatalog,
) -> Result<AttachmentResolution, LinkError> {
    let mut links = LinkSet::default();
    let mut record_schemas = operation_schemas(&manifest.operations);
    let mut maps = BTreeMap::new();
    let mut linked_capabilities = BTreeSet::new();

    let mut fact_count = 0usize;
    for declaration in &manifest.facts {
        let Some(capability) = catalog.facts.get(&declaration.name) else {
            if declaration.requirement == DependencyRequirement::Required {
                return Err(LinkError::MissingRequired {
                    kind: "fact",
                    name: declaration.name.clone(),
                });
            }
            links.facts.push(None);
            continue;
        };
        if capability.value_type != declaration.value_type {
            return Err(LinkError::FactType {
                name: declaration.name.clone(),
                expected: declaration.value_type,
                actual: capability.value_type,
            });
        }
        let handle = FactHandle::new(next_handle(fact_count)?);
        fact_count += 1;
        links.facts.push(Some(handle));
        for field_use in &declaration.uses {
            record_schemas
                .get_mut(field_use)
                .expect("manifest field use was validated")
                .facts
                .insert(
                    handle,
                    FieldSchema {
                        value_type: declaration.value_type,
                        required_column: declaration.requirement == DependencyRequirement::Required,
                        required_values: declaration.requirement == DependencyRequirement::Required,
                        max_value_bytes: declaration
                            .max_value_bytes
                            .min(capability.max_value_bytes),
                    },
                );
        }
    }

    for (metadata_count, declaration) in manifest.metadata.iter().enumerate() {
        let handle = MetadataHandle::new(next_handle(metadata_count)?);
        links.metadata.push(Some(handle));
        for field_use in &declaration.uses {
            record_schemas
                .get_mut(field_use)
                .expect("manifest field use was validated")
                .metadata
                .insert(
                    handle,
                    FieldSchema {
                        value_type: declaration.value_type,
                        required_column: declaration.requirement == DependencyRequirement::Required,
                        required_values: declaration.requirement == DependencyRequirement::Required,
                        max_value_bytes: declaration.max_value_bytes,
                    },
                );
        }
    }

    let mut event_count = 0usize;
    for declaration in &manifest.events {
        if !catalog.events.contains(&declaration.name) {
            if declaration.requirement == DependencyRequirement::Required {
                return Err(LinkError::MissingRequired {
                    kind: "event",
                    name: declaration.name.clone(),
                });
            }
            links.events.push(None);
            continue;
        }
        links
            .events
            .push(Some(EventHandle::new(next_handle(event_count)?)));
        event_count += 1;
    }

    let mut capability_count = 0usize;
    for declaration in &manifest.capabilities {
        if !catalog.capabilities.contains(&declaration.name) {
            if declaration.requirement == DependencyRequirement::Required {
                return Err(LinkError::MissingRequired {
                    kind: "capability",
                    name: declaration.name.clone(),
                });
            }
            links.capabilities.push(None);
            continue;
        }
        links
            .capabilities
            .push(Some(CapabilityHandle::new(next_handle(capability_count)?)));
        capability_count += 1;
        linked_capabilities.insert(declaration.name.clone());
    }

    for declaration in &manifest.maps {
        match declaration.class {
            MapClass::External { requirement } => {
                let Some(schema) = catalog.external_maps.get(&declaration.name) else {
                    if requirement == DependencyRequirement::Required {
                        return Err(LinkError::MissingRequired {
                            kind: "external map",
                            name: declaration.name.clone(),
                        });
                    }
                    links.maps.push(None);
                    continue;
                };
                if schema != &declaration.schema {
                    return Err(LinkError::MapSchema(declaration.name.clone()));
                }
            }
            MapClass::PolicyOwned { .. } => {}
        }
        let handle = MapHandle::new(next_handle(maps.len())?);
        links.maps.push(Some(handle));
        maps.insert(handle, declaration.clone());
    }

    Ok(AttachmentResolution {
        links,
        record_schemas,
        maps,
        capabilities: linked_capabilities,
    })
}

fn operation_schemas(operations: &BTreeSet<Operation>) -> BTreeMap<FieldUse, LinkedRecordSchema> {
    let mut schemas = BTreeMap::new();
    for operation in operations {
        let locations: &[FieldLocation] = match operation {
            Operation::Admit => &[FieldLocation::Request],
            Operation::Route => &[FieldLocation::Request, FieldLocation::Candidate],
            Operation::Schedule | Operation::Evict => &[FieldLocation::Candidate],
            Operation::Feedback => &[FieldLocation::Feedback],
        };
        for location in locations {
            schemas.insert(
                FieldUse {
                    operation: *operation,
                    location: *location,
                },
                LinkedRecordSchema::default(),
            );
        }
    }
    schemas
}

fn next_handle(count: usize) -> Result<u32, LinkError> {
    u32::try_from(count).map_err(|_| LinkError::TooManyHandles)
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CatalogError {
    #[error("duplicate {kind} declaration {name}")]
    Duplicate { kind: &'static str, name: Symbol },
    #[error("fact {0} has impossible byte bounds")]
    InvalidFactBounds(Symbol),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinkError {
    #[error("missing required {kind} {name}")]
    MissingRequired { kind: &'static str, name: Symbol },
    #[error("fact {name} has type {actual:?}; policy requires {expected:?}")]
    FactType {
        name: Symbol,
        expected: ValueType,
        actual: ValueType,
    },
    #[error("external map {0} schema does not match the policy declaration")]
    MapSchema(Symbol),
    #[error("attachment contains more than u32::MAX linked handles")]
    TooManyHandles,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pie_plex::{
        CapabilityDeclaration, FactDeclaration, InvocationMode, MapKeyType, MapPersistence,
        MetadataDeclaration, MetadataScope, Operation, PolicyLimits,
    };

    use super::*;

    fn limits() -> PolicyLimits {
        PolicyLimits {
            memory_bytes: 2 << 20,
            fuel: 100_000,
            deadline_ms: 10,
            input_bytes: 1 << 16,
            output_bytes: 1 << 16,
            map_calls: 16,
            map_bytes: 1 << 12,
            staged_mutations: 8,
            feedback_records: 32,
            telemetry_records: 0,
            telemetry_bytes: 0,
        }
    }

    fn manifest() -> Manifest {
        Manifest {
            contract: pie_plex::ContractVersion::V0_1,
            package_name: "linked-policy".into(),
            package_version: "0.1.0".into(),
            operations: BTreeSet::from([Operation::Schedule]),
            invocation_mode: InvocationMode::SetDependent,
            capabilities: vec![CapabilityDeclaration {
                name: Symbol::new("pie.schedule.token-budget@1").unwrap(),
                requirement: DependencyRequirement::Optional,
            }],
            facts: vec![FactDeclaration {
                name: Symbol::new("pie.attained-service@1").unwrap(),
                value_type: ValueType::U64,
                requirement: DependencyRequirement::Required,
                max_value_bytes: 8,
                uses: BTreeSet::from([FieldUse {
                    operation: Operation::Schedule,
                    location: FieldLocation::Candidate,
                }]),
            }],
            metadata: vec![MetadataDeclaration {
                name: Symbol::new("acme.expected-tokens@1").unwrap(),
                value_type: ValueType::U64,
                scope: MetadataScope::Generation,
                requirement: DependencyRequirement::Optional,
                max_value_bytes: 8,
                uses: BTreeSet::from([FieldUse {
                    operation: Operation::Schedule,
                    location: FieldLocation::Candidate,
                }]),
            }],
            events: Vec::new(),
            maps: vec![MapDeclaration {
                name: Symbol::new("policy.accounting@1").unwrap(),
                class: MapClass::PolicyOwned {
                    persistence: MapPersistence::Attachment,
                },
                schema: MapSchema {
                    key_type: MapKeyType::Bytes,
                    value_type: ValueType::U64,
                    max_entries: 8,
                    max_key_bytes: 16,
                    max_value_bytes: 8,
                    default_ttl_ms: None,
                    max_ttl_ms: None,
                },
            }],
            limits: limits(),
        }
    }

    #[test]
    fn links_exact_required_and_absent_optional_dependencies() {
        let mut catalog = CapabilityCatalog::default();
        catalog
            .add_fact(
                Symbol::new("pie.attained-service@1").unwrap(),
                ValueType::U64,
                8,
            )
            .unwrap();
        let linked = link_manifest(&manifest(), &catalog).unwrap();
        assert_eq!(linked.links.facts, vec![Some(FactHandle::new(0))]);
        assert_eq!(linked.links.metadata, vec![Some(MetadataHandle::new(0))]);
        assert_eq!(linked.links.capabilities, vec![None]);
        assert_eq!(linked.links.maps, vec![Some(MapHandle::new(0))]);
    }

    #[test]
    fn rejects_missing_required_fact() {
        assert!(matches!(
            link_manifest(&manifest(), &CapabilityCatalog::default()),
            Err(LinkError::MissingRequired { kind: "fact", .. })
        ));
    }

    #[test]
    fn rejected_catalog_duplicate_preserves_original_definition() {
        let mut catalog = CapabilityCatalog::default();
        let name = Symbol::new("pie.attained-service@1").unwrap();
        catalog.add_fact(name.clone(), ValueType::U64, 8).unwrap();
        assert!(matches!(
            catalog.add_fact(name.clone(), ValueType::String, 32),
            Err(CatalogError::Duplicate { kind: "fact", .. })
        ));
        assert_eq!(catalog.facts[&name].value_type, ValueType::U64);
    }
}
