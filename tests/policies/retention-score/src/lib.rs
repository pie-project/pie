use plex::serde_json::json;
use plex::{Document, Policy};

struct RetentionScore;

impl Policy for RetentionScore {
    fn evict(input: &mut Document) -> Result<Document, String> {
        let resident = input["resident"]
            .as_array_mut()
            .ok_or("resident must be an array")?;
        let scores = resident
            .iter_mut()
            .map(|unit| {
                let reload = unit["facts"]["reload_cost"].as_f64().unwrap_or(0.0);
                let retention = unit["request"]["state"]["retention_bonus"]
                    .as_f64()
                    .unwrap_or(0.0);
                if unit["request"].is_object() {
                    unit["request"]["state"]["eviction_checks"] = json!(
                        unit["request"]["state"]["eviction_checks"]
                            .as_u64()
                            .unwrap_or(0)
                            + 1
                    );
                }
                reload + retention
            })
            .collect::<Vec<_>>();
        Ok(json!({"scores": scores}))
    }
}

plex::export_policy!(RetentionScore);
