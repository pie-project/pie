use plex::types::{AdmissionDecision, AdmissionInput, AdmissionOutput, PolicyError};

struct AcceptAll;

impl plex::Policy for AcceptAll {
    fn admit(_input: AdmissionInput) -> Result<AdmissionOutput, PolicyError> {
        Ok(AdmissionOutput {
            decision: AdmissionDecision::Accept,
            mutations: Vec::new(),
        })
    }
}

plex::export_policy!(AcceptAll);
