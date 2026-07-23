use std::collections::BTreeSet;

use pie_plex::v0_6::*;
use serde_json::json;

fn meta() -> DecisionMeta {
    DecisionMeta {
        opportunity_id: "op-1".into(),
        snapshot: SnapshotRef {
            id: "snapshot-1".into(),
            revision: 7,
        },
        attempt: 0,
        mechanics: Vec::new(),
    }
}

fn request_ref(id: &str) -> RequestRef {
    RequestRef {
        request_id: id.into(),
        generation_id: 0,
        group_id: None,
        principal_id: "tenant-a".into(),
    }
}

fn request_state(id: &str, status: RequestStatus) -> RequestState {
    RequestState {
        request: request_ref(id),
        status,
        facts: json!({
            "request_id": id,
            "generation_id": 0,
            "group_id": null,
            "principal_id": "tenant-a"
        }),
        fields: json!({}),
        scratch: json!({}),
    }
}

fn state(requests: Vec<RequestState>) -> PolicyState {
    PolicyState {
        shared: json!({}),
        groups: Vec::new(),
        requests,
    }
}

#[test]
fn validates_v0_6_manifest_and_mechanic_negotiation() {
    let manifest = Manifest {
        contract: ContractVersion::V0_6,
        package_name: "joint-policy".into(),
        package_version: "0.6.0".into(),
        implements: BTreeSet::from([Operation::Route, Operation::Schedule]),
        requires: BTreeSet::from([MechanicId::from("request.cancel@1")]),
        optional: BTreeSet::from([MechanicId::from("cache.prefetch@1")]),
        schemas: BTreeSet::from([SchemaRequirement {
            kind: SchemaKind::Fact,
            id: "pie.example.queue-facts@1".into(),
            required: true,
        }]),
        limits: PolicyLimits {
            memory_bytes: 1 << 20,
            deadline_ms: 20,
            input_bytes: 1 << 16,
            output_bytes: 1 << 16,
            host_calls: 8,
            host_call_bytes: 1 << 14,
        },
    };
    manifest.validate().unwrap();
    assert_eq!(
        standard_mechanic("cache.prefetch@1").unwrap().method,
        Some("pie.cache.prefetch@1")
    );

    let mut overlapping = manifest;
    overlapping
        .optional
        .insert(MechanicId::from("request.cancel@1"));
    assert!(matches!(
        overlapping.validate(),
        Err(ManifestValidationError::MechanicOverlap(_))
    ));

    let mut duplicate_schema = overlapping;
    duplicate_schema
        .optional
        .remove(&MechanicId::from("request.cancel@1"));
    duplicate_schema.schemas.insert(SchemaRequirement {
        kind: SchemaKind::Fact,
        id: "pie.example.queue-facts@1".into(),
        required: false,
    });
    assert!(matches!(
        duplicate_schema.validate(),
        Err(ManifestValidationError::DuplicateSchemaRequirement { .. })
    ));
}

#[test]
fn lifecycle_transitions_and_continuations_are_explicit() {
    validate_group_transition(None, GroupStatus::Open).unwrap();
    validate_group_transition(Some(GroupStatus::Open), GroupStatus::Closed).unwrap();
    assert!(matches!(
        validate_group_transition(Some(GroupStatus::Closed), GroupStatus::Open),
        Err(ContractValidationError::InvalidGroupTransition { .. })
    ));

    validate_request_transition(None, RequestStatus::Pending).unwrap();
    validate_request_transition(Some(RequestStatus::Pending), RequestStatus::Admitted).unwrap();
    validate_request_transition(Some(RequestStatus::Active), RequestStatus::Paused).unwrap();
    assert!(matches!(
        validate_request_transition(Some(RequestStatus::Completed), RequestStatus::Active),
        Err(ContractValidationError::InvalidRequestTransition { .. })
    ));

    let previous = request_ref("a");
    let mut next = previous.clone();
    next.generation_id = 1;
    validate_request_continuation(&previous, &next).unwrap();
    next.generation_id = 2;
    assert!(matches!(
        validate_request_continuation(&previous, &next),
        Err(ContractValidationError::InvalidContinuation { .. })
    ));
}

#[test]
fn group_identity_is_typed_and_outlives_child_requests() {
    let group_id = GroupId::from("group-1");
    let state = PolicyState {
        shared: json!({}),
        groups: vec![GroupState {
            group_id: group_id.clone(),
            principal_id: "tenant-a".into(),
            status: GroupStatus::Closed,
            limits: GroupLimits {
                max_members: 4,
                max_scratch_bytes: 1024,
            },
            member_count: 1,
            facts: json!({"group_id": "group-1", "principal_id": "tenant-a"}),
            scratch: json!({"service": 10}),
        }],
        requests: vec![RequestState {
            request: RequestRef {
                request_id: "request-1".into(),
                generation_id: 0,
                group_id: Some(group_id),
                principal_id: "tenant-a".into(),
            },
            status: RequestStatus::Completed,
            facts: json!({
                "request_id": "request-1",
                "generation_id": 0,
                "group_id": "group-1",
                "principal_id": "tenant-a"
            }),
            fields: json!({}),
            scratch: json!({}),
        }],
    };
    validate_policy_state(&state).unwrap();

    let mut cross_principal = state.clone();
    cross_principal.requests[0].request.principal_id = "tenant-b".into();
    cross_principal.requests[0].facts["principal_id"] = json!("tenant-b");
    assert!(matches!(
        validate_policy_state(&cross_principal),
        Err(ContractValidationError::PrincipalMismatch { .. })
    ));

    assert!(matches!(
        validate_state_update(
            &state,
            &StateUpdate {
                shared: None,
                groups: vec![GroupStateUpdate {
                    group_id: "group-1".into(),
                    scratch: json!({"value": "x".repeat(2048)}),
                }],
                requests: Vec::new(),
            }
        ),
        Err(ContractValidationError::GroupScratchLimit { .. })
    ));

    let mut forged = state;
    forged.requests[0].request.group_id = Some("other-group".into());
    assert!(matches!(
        validate_policy_state(&forged),
        Err(ContractValidationError::MissingScope { kind: "group", .. })
    ));
}

#[test]
fn terminal_groups_cannot_admit_new_requests() {
    let group_id = GroupId::from("group-1");
    let request = RequestState {
        request: RequestRef {
            request_id: "request-1".into(),
            generation_id: 0,
            group_id: Some(group_id.clone()),
            principal_id: "tenant-a".into(),
        },
        status: RequestStatus::Pending,
        facts: json!({
            "request_id": "request-1",
            "generation_id": 0,
            "group_id": "group-1",
            "principal_id": "tenant-a"
        }),
        fields: json!({}),
        scratch: json!({}),
    };
    let state = PolicyState {
        shared: json!({}),
        groups: vec![GroupState {
            group_id,
            principal_id: "tenant-a".into(),
            status: GroupStatus::Closed,
            limits: GroupLimits {
                max_members: 4,
                max_scratch_bytes: 1024,
            },
            member_count: 1,
            facts: json!({"group_id": "group-1", "principal_id": "tenant-a"}),
            scratch: json!({}),
        }],
        requests: vec![request.clone()],
    };
    let context = AdmitContext {
        meta: meta(),
        cause: AdmitCause::Arrival,
        candidates: vec![AdmissionCandidate {
            request: request.request,
            demand: Vec::new(),
            facts: json!({}),
        }],
        capacity: AdmissionCapacity {
            max_accepted: 1,
            limits: Vec::new(),
            facts: json!({}),
        },
    };
    assert!(matches!(
        validate_admit_context(&state, &context),
        Err(ContractValidationError::AdmissionIntoTerminalGroup { .. })
    ));
}

#[test]
fn batch_admission_enforces_dense_output_and_resources() {
    let state = state(vec![
        request_state("a", RequestStatus::Pending),
        request_state("b", RequestStatus::Pending),
    ]);
    let context = AdmitContext {
        meta: meta(),
        cause: AdmitCause::Arrival,
        candidates: vec![
            AdmissionCandidate {
                request: request_ref("a"),
                demand: vec![ResourceAmount {
                    name: "kv".into(),
                    unit: "bytes".into(),
                    amount: 6,
                }],
                facts: json!({}),
            },
            AdmissionCandidate {
                request: request_ref("b"),
                demand: vec![ResourceAmount {
                    name: "kv".into(),
                    unit: "bytes".into(),
                    amount: 5,
                }],
                facts: json!({}),
            },
        ],
        capacity: AdmissionCapacity {
            max_accepted: 2,
            limits: vec![ResourceLimit {
                name: "kv".into(),
                unit: "bytes".into(),
                maximum: 10,
            }],
            facts: json!({}),
        },
    };
    validate_admit_context(&state, &context).unwrap();
    validate_admit_plan(
        &context,
        &AdmitPlan {
            decisions: vec![AdmissionDecision::Accept, AdmissionDecision::Defer],
        },
    )
    .unwrap();
    assert!(matches!(
        validate_admit_plan(
            &context,
            &AdmitPlan {
                decisions: vec![AdmissionDecision::Accept, AdmissionDecision::Accept],
            }
        ),
        Err(ContractValidationError::ResourceCapacityExceeded { .. })
    ));
}

#[test]
fn joint_route_assignment_handles_the_non_greedy_counterexample() {
    let state = state(vec![
        request_state("a", RequestStatus::Admitted),
        request_state("b", RequestStatus::Admitted),
    ]);
    let target = |id: &str| RouteTarget {
        target_id: id.into(),
        max_assignments: 1,
        capacity: vec![ResourceLimit {
            name: "kv".into(),
            unit: "bytes".into(),
            maximum: 10,
        }],
        revision: 1,
        facts: json!({}),
    };
    let edge = |request_index, target_index, utility| RouteEdge {
        request_index,
        target_index,
        demand: vec![ResourceAmount {
            name: "kv".into(),
            unit: "bytes".into(),
            amount: 1,
        }],
        facts: json!({"utility": utility}),
    };
    let context = RouteContext {
        meta: meta(),
        cause: RouteCause::Admission,
        requests: vec![
            RouteRequest {
                request: request_ref("a"),
                facts: json!({}),
            },
            RouteRequest {
                request: request_ref("b"),
                facts: json!({}),
            },
        ],
        targets: vec![target("x"), target("y")],
        feasible_edges: vec![edge(0, 0, 10), edge(0, 1, 9), edge(1, 0, 8), edge(1, 1, 0)],
    };
    validate_route_context(&state, &context).unwrap();
    validate_route_plan(
        &context,
        &RoutePlan {
            decisions: vec![RouteDecision::Assign(1), RouteDecision::Assign(2)],
        },
    )
    .unwrap();
    assert!(matches!(
        validate_route_plan(
            &context,
            &RoutePlan {
                decisions: vec![RouteDecision::Assign(0), RouteDecision::Assign(2)],
            }
        ),
        Err(ContractValidationError::CountCapacityExceeded {
            field: "route.target.max_assignments",
            ..
        })
    ));
}

#[test]
fn multi_request_schedule_selection_is_all_or_none() {
    let state = state(vec![
        request_state("w1", RequestStatus::Active),
        request_state("w2", RequestStatus::Active),
        request_state("x", RequestStatus::Active),
    ]);
    let mut context = ScheduleContext {
        meta: meta(),
        cause: ScheduleCause::CapacityChanged,
        runnable: ["w1", "w2", "x"]
            .into_iter()
            .map(|id| ScheduleCandidate {
                request: request_ref(id),
                max_token_budget: 4,
                facts: json!({}),
            })
            .collect(),
        capacity: ScheduleCapacity {
            max_selections: 2,
            max_requests: 2,
            max_total_tokens: 8,
            facts: json!({}),
        },
    };
    validate_schedule_context(&state, &context).unwrap();
    let bundle = SchedulePlan {
        selections: vec![ScheduleSelection {
            requests: vec![0, 1],
            token_budgets: vec![4, 4],
        }],
    };
    validate_schedule_plan(&context, &bundle).unwrap();

    context.capacity.max_requests = 1;
    assert!(matches!(
        validate_schedule_plan(&context, &bundle),
        Err(ContractValidationError::CountCapacityExceeded {
            field: "schedule.max_requests",
            ..
        })
    ));
}

#[test]
fn cache_plan_supports_bypass_and_legal_reclaim() {
    let object = |id: &str, size_bytes| CacheObject {
        object_id: id.into(),
        size_bytes,
        beneficiaries: Vec::new(),
        beneficiary_count: 0,
        facts: json!({}),
    };
    let context = CacheContext {
        meta: meta(),
        cause: CacheCause::Insertion,
        resident: vec![ResidentCacheObject {
            object: object("resident", 8),
            reclaimable: true,
        }],
        prospective: vec![object("prospective", 4)],
        capacity: CacheCapacity {
            max_bytes: 10,
            fixed_bytes: 0,
            facts: json!({}),
        },
        episode: None,
    };
    validate_cache_context(&context).unwrap();
    validate_cache_plan(
        &context,
        &CachePlan {
            admissions: vec![CacheAdmission::Bypass],
            reclaim: Vec::new(),
        },
    )
    .unwrap();
    validate_cache_plan(
        &context,
        &CachePlan {
            admissions: vec![CacheAdmission::Cache],
            reclaim: vec![0],
        },
    )
    .unwrap();
    assert!(matches!(
        validate_cache_plan(
            &context,
            &CachePlan {
                admissions: vec![CacheAdmission::Cache],
                reclaim: Vec::new(),
            }
        ),
        Err(ContractValidationError::CountCapacityExceeded {
            field: "cache.max_bytes",
            ..
        })
    ));
}

#[test]
fn feedback_and_state_updates_are_structurally_bounded() {
    let state = state(vec![request_state("a", RequestStatus::Active)]);
    validate_feedback_context(&FeedbackContext {
        delivery_id: "delivery-1".into(),
        records: vec![FeedbackRecord {
            subject: FeedbackSubject::Request("a".into()),
            outcome: OutcomeKind::Progress,
            facts: json!({"tokens": 4}),
        }],
    })
    .unwrap();
    assert!(matches!(
        validate_feedback_context(&FeedbackContext {
            delivery_id: "delivery-2".into(),
            records: Vec::new(),
        }),
        Err(ContractValidationError::EmptyFeedback)
    ));
    assert!(matches!(
        validate_feedback_context(&FeedbackContext {
            delivery_id: "delivery-3".into(),
            records: vec![FeedbackRecord {
                subject: FeedbackSubject::Request("a".into()),
                outcome: OutcomeKind::ActionSucceeded,
                facts: json!({}),
            }],
        }),
        Err(ContractValidationError::FeedbackOutcomeMismatch)
    ));

    validate_state_update(
        &state,
        &StateUpdate {
            shared: None,
            groups: Vec::new(),
            requests: vec![RequestStateUpdate {
                request_id: "a".into(),
                fields: None,
                scratch: Some(json!({"service": 4})),
            }],
        },
    )
    .unwrap();

    validate_policy_error(&PolicyError {
        code: "capacity-exhausted".into(),
        message: "no legal assignment".into(),
        details: json!({"retryable": true}),
    })
    .unwrap();
}
