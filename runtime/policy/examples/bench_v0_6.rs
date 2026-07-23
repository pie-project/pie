use std::collections::BTreeMap;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use pie_plex::v0_6::*;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let budget_path = parse_budget_path()?;
    let samples = 31;
    let mut results = BTreeMap::new();

    let singleton_state = state(1, RequestStatus::Pending);
    let singleton_admit = admit_context(1);
    results.insert(
        "admit_singleton_ns",
        median(samples, || {
            validate_admit_context(&singleton_state, &singleton_admit).unwrap();
            validate_admit_plan(
                &singleton_admit,
                &AdmitPlan {
                    decisions: vec![AdmissionDecision::Accept],
                },
            )
            .unwrap();
        }),
    );

    let batch_state = state(64, RequestStatus::Pending);
    let batch_admit = admit_context(64);
    results.insert(
        "admit_batch_64_ns",
        median(samples, || {
            validate_admit_context(&batch_state, &batch_admit).unwrap();
            validate_admit_plan(
                &batch_admit,
                &AdmitPlan {
                    decisions: vec![AdmissionDecision::Accept; 64],
                },
            )
            .unwrap();
        }),
    );

    let route_state = state(64, RequestStatus::Admitted);
    let route = route_context(64, 8);
    results.insert(
        "route_64x8_ns",
        median(samples, || {
            validate_route_context(&route_state, &route).unwrap();
            validate_route_plan(
                &route,
                &RoutePlan {
                    decisions: (0..64)
                        .map(|request| RouteDecision::Assign(request * 8))
                        .collect(),
                },
            )
            .unwrap();
        }),
    );

    let schedule_state = state(128, RequestStatus::Active);
    let schedule = schedule_context(128);
    results.insert(
        "schedule_128_ns",
        median(samples, || {
            validate_schedule_context(&schedule_state, &schedule).unwrap();
            validate_schedule_plan(
                &schedule,
                &SchedulePlan {
                    selections: (0..128)
                        .map(|index| ScheduleSelection {
                            requests: vec![index],
                            token_budgets: vec![1],
                        })
                        .collect(),
                },
            )
            .unwrap();
        }),
    );

    let cache = cache_context(1024);
    results.insert(
        "cache_1024_ns",
        median(samples, || {
            validate_cache_context(&cache).unwrap();
            validate_cache_plan(
                &cache,
                &CachePlan {
                    admissions: Vec::new(),
                    reclaim: Vec::new(),
                },
            )
            .unwrap();
        }),
    );

    if let Some(path) = budget_path {
        let budgets: BTreeMap<String, u64> = serde_json::from_slice(&std::fs::read(path)?)?;
        for (name, actual) in &results {
            let maximum = budgets
                .get(*name)
                .ok_or_else(|| format!("missing performance budget for {name}"))?;
            if actual > maximum {
                return Err(format!("{name} median {actual}ns exceeds budget {maximum}ns").into());
            }
        }
    }

    let output = json!({
        "contract": {"major": 0, "minor": 6},
        "profile": "release",
        "samples": samples,
        "architecture": std::env::consts::ARCH,
        "operating_system": std::env::consts::OS,
        "median_ns": results,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn parse_budget_path() -> Result<Option<PathBuf>, String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        None => Ok(None),
        Some("--check") => {
            let path = args.next().ok_or("--check requires a budget path")?;
            if args.next().is_some() {
                return Err("too many arguments".into());
            }
            Ok(Some(path.into()))
        }
        Some(argument) => Err(format!("unknown argument {argument:?}")),
    }
}

fn median(samples: usize, mut operation: impl FnMut()) -> u64 {
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        black_box(operation());
        timings.push(start.elapsed().as_nanos() as u64);
    }
    timings.sort_unstable();
    timings[timings.len() / 2]
}

fn request_ref(index: u32) -> RequestRef {
    RequestRef {
        request_id: format!("request-{index}").into(),
        generation_id: 0,
        group_id: None,
        principal_id: "tenant".into(),
    }
}

fn state(count: u32, status: RequestStatus) -> PolicyState {
    PolicyState {
        shared: json!({}),
        groups: Vec::new(),
        requests: (0..count)
            .map(|index| {
                let request = request_ref(index);
                RequestState {
                    facts: json!({
                        "request_id": request.request_id.as_str(),
                        "generation_id": 0,
                        "group_id": null,
                        "principal_id": "tenant"
                    }),
                    request,
                    status,
                    fields: json!({}),
                    scratch: json!({}),
                }
            })
            .collect(),
    }
}

fn meta(id: &str) -> DecisionMeta {
    DecisionMeta {
        opportunity_id: id.into(),
        snapshot: SnapshotRef {
            id: "benchmark".into(),
            revision: 0,
        },
        attempt: 0,
        mechanics: Vec::new(),
    }
}

fn admit_context(count: u32) -> AdmitContext {
    AdmitContext {
        meta: meta("admit"),
        cause: AdmitCause::Arrival,
        candidates: (0..count)
            .map(|index| AdmissionCandidate {
                request: request_ref(index),
                demand: vec![ResourceAmount {
                    name: "kv".into(),
                    unit: "bytes".into(),
                    amount: 1,
                }],
                facts: json!({}),
            })
            .collect(),
        capacity: AdmissionCapacity {
            max_accepted: count,
            limits: vec![ResourceLimit {
                name: "kv".into(),
                unit: "bytes".into(),
                maximum: u64::from(count),
            }],
            facts: json!({}),
        },
    }
}

fn route_context(requests: u32, targets: u32) -> RouteContext {
    RouteContext {
        meta: meta("route"),
        cause: RouteCause::Admission,
        requests: (0..requests)
            .map(|index| RouteRequest {
                request: request_ref(index),
                facts: json!({}),
            })
            .collect(),
        targets: (0..targets)
            .map(|index| RouteTarget {
                target_id: format!("target-{index}").into(),
                max_assignments: requests,
                capacity: Vec::new(),
                revision: 0,
                facts: json!({}),
            })
            .collect(),
        feasible_edges: (0..requests)
            .flat_map(|request_index| {
                (0..targets).map(move |target_index| RouteEdge {
                    request_index,
                    target_index,
                    demand: Vec::new(),
                    facts: json!({}),
                })
            })
            .collect(),
    }
}

fn schedule_context(count: u32) -> ScheduleContext {
    ScheduleContext {
        meta: meta("schedule"),
        cause: ScheduleCause::CapacityChanged,
        runnable: (0..count)
            .map(|index| ScheduleCandidate {
                request: request_ref(index),
                max_token_budget: 1,
                facts: json!({}),
            })
            .collect(),
        capacity: ScheduleCapacity {
            max_selections: count,
            max_requests: count,
            max_total_tokens: u64::from(count),
            facts: json!({}),
        },
    }
}

fn cache_context(count: u32) -> CacheContext {
    CacheContext {
        meta: meta("cache"),
        cause: CacheCause::Pressure,
        resident: (0..count)
            .map(|index| ResidentCacheObject {
                object: CacheObject {
                    object_id: format!("object-{index}").into(),
                    size_bytes: 1,
                    beneficiaries: Vec::new(),
                    beneficiary_count: 0,
                    facts: json!({}),
                },
                reclaimable: true,
            })
            .collect(),
        prospective: Vec::new(),
        capacity: CacheCapacity {
            max_bytes: u64::from(count),
            fixed_bytes: 0,
            facts: json!({}),
        },
        episode: None,
    }
}
