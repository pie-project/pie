use plex::types::{AdmissionDecision, AdmissionInput, AdmissionOutput, PolicyError};

struct DeferAll;

impl plex::Policy for DeferAll {
    fn admit(_input: AdmissionInput) -> Result<AdmissionOutput, PolicyError> {
        Ok(AdmissionOutput {
            decision: AdmissionDecision::Defer,
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(DeferAll);
