//! Engine-lifetime inferlet metadata KV store tests.

use std::sync::Mutex;

use pie::metadata_store::{
    self, MAX_KEY_BYTES, MAX_NAMESPACE_BYTES, MAX_TOTAL_BYTES, MAX_VALUE_BYTES, MetadataOwner,
};
use uuid::Uuid;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn ns(label: &str) -> String {
    format!("test.{label}.{}", Uuid::new_v4())
}

fn owner(label: &str) -> MetadataOwner {
    MetadataOwner::new(format!("user-{label}"), format!("program-{label}")).unwrap()
}

#[test]
fn metadata_values_are_namespaced_and_overwritten() {
    let _guard = TEST_LOCK.lock().unwrap();
    let owner = owner("overwrite");
    let left = ns("left");
    let right = ns("right");

    metadata_store::put(&owner, &left, "thread", b"alpha".to_vec()).unwrap();
    metadata_store::put(&owner, &right, "thread", b"beta".to_vec()).unwrap();

    assert_eq!(
        metadata_store::get(&owner, &left, "thread").unwrap(),
        Some(b"alpha".to_vec())
    );
    assert_eq!(
        metadata_store::get(&owner, &right, "thread").unwrap(),
        Some(b"beta".to_vec())
    );

    metadata_store::put(&owner, &left, "thread", b"gamma".to_vec()).unwrap();
    assert_eq!(
        metadata_store::get(&owner, &left, "thread").unwrap(),
        Some(b"gamma".to_vec())
    );

    metadata_store::delete(&owner, &left, "thread").unwrap();
    metadata_store::delete(&owner, &right, "thread").unwrap();
}

#[test]
fn deleting_missing_or_present_values_is_explicit() {
    let _guard = TEST_LOCK.lock().unwrap();
    let owner = owner("delete");
    let namespace = ns("delete");

    assert!(!metadata_store::delete(&owner, &namespace, "sidecar").unwrap());

    metadata_store::put(&owner, &namespace, "sidecar", b"state".to_vec()).unwrap();
    assert!(metadata_store::delete(&owner, &namespace, "sidecar").unwrap());
    assert_eq!(
        metadata_store::get(&owner, &namespace, "sidecar").unwrap(),
        None
    );
    assert!(!metadata_store::delete(&owner, &namespace, "sidecar").unwrap());
}

#[test]
fn empty_namespace_or_key_is_rejected() {
    let _guard = TEST_LOCK.lock().unwrap();
    let owner = owner("validation");

    assert!(metadata_store::put(&owner, "", "key", b"value".to_vec()).is_err());
    assert!(metadata_store::put(&owner, "namespace", "", b"value".to_vec()).is_err());
    assert!(metadata_store::get(&owner, "", "key").is_err());
    assert!(metadata_store::delete(&owner, "namespace", "").is_err());
}

#[test]
fn metadata_values_are_isolated_by_owner() {
    let _guard = TEST_LOCK.lock().unwrap();
    let alice = MetadataOwner::new("alice", "program-a").unwrap();
    let bob = MetadataOwner::new("bob", "program-a").unwrap();
    let other_program = MetadataOwner::new("alice", "program-b").unwrap();
    let namespace = ns("isolation");

    metadata_store::put(&alice, &namespace, "sidecar", b"alice".to_vec()).unwrap();
    metadata_store::put(&bob, &namespace, "sidecar", b"bob".to_vec()).unwrap();
    metadata_store::put(&other_program, &namespace, "sidecar", b"program-b".to_vec()).unwrap();

    assert_eq!(
        metadata_store::get(&alice, &namespace, "sidecar").unwrap(),
        Some(b"alice".to_vec())
    );
    assert_eq!(
        metadata_store::get(&bob, &namespace, "sidecar").unwrap(),
        Some(b"bob".to_vec())
    );
    assert_eq!(
        metadata_store::get(&other_program, &namespace, "sidecar").unwrap(),
        Some(b"program-b".to_vec())
    );

    assert!(metadata_store::delete(&bob, &namespace, "sidecar").unwrap());
    assert_eq!(
        metadata_store::get(&alice, &namespace, "sidecar").unwrap(),
        Some(b"alice".to_vec())
    );
    assert_eq!(
        metadata_store::get(&other_program, &namespace, "sidecar").unwrap(),
        Some(b"program-b".to_vec())
    );

    metadata_store::delete(&alice, &namespace, "sidecar").unwrap();
    metadata_store::delete(&other_program, &namespace, "sidecar").unwrap();
}

#[test]
fn oversized_namespace_key_or_value_is_rejected() {
    let _guard = TEST_LOCK.lock().unwrap();
    let owner = owner("limits");
    let namespace = ns("limits");

    assert!(
        metadata_store::put(&owner, &"n".repeat(MAX_NAMESPACE_BYTES + 1), "key", vec![])
            .unwrap_err()
            .to_string()
            .contains("namespace")
    );
    assert!(
        metadata_store::put(&owner, &namespace, &"k".repeat(MAX_KEY_BYTES + 1), vec![])
            .unwrap_err()
            .to_string()
            .contains("key")
    );
    assert!(
        metadata_store::put(&owner, &namespace, "key", vec![0; MAX_VALUE_BYTES + 1])
            .unwrap_err()
            .to_string()
            .contains("value")
    );
}

#[test]
fn total_store_cap_is_enforced_and_accounted_across_deletes() {
    let _guard = TEST_LOCK.lock().unwrap();
    let owner = owner("total-cap");
    let namespace = ns("total-cap");
    let chunk_size = MAX_VALUE_BYTES;
    let chunk_count = MAX_TOTAL_BYTES / chunk_size;
    let mut keys = Vec::new();

    for index in 0..chunk_count {
        let key = format!("chunk-{index}");
        metadata_store::put(&owner, &namespace, &key, vec![0; chunk_size]).unwrap();
        keys.push(key);
    }

    let err = metadata_store::put(&owner, &namespace, "overflow", vec![0; chunk_size])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("metadata store total byte cap"),
        "unexpected error: {err}"
    );

    assert!(metadata_store::delete(&owner, &namespace, &keys[0]).unwrap());
    metadata_store::put(&owner, &namespace, "replacement", vec![0; chunk_size]).unwrap();

    for key in keys {
        let _ = metadata_store::delete(&owner, &namespace, &key);
    }
    let _ = metadata_store::delete(&owner, &namespace, "replacement");
}
