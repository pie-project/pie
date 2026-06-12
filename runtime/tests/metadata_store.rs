use std::sync::Mutex;

use pie::metadata_store::{
    self, MetadataOwner, MAX_ENTRIES, MAX_KEY_BYTES, MAX_NAMESPACE_BYTES, MAX_TOTAL_BYTES,
    MAX_VALUE_BYTES,
};
use uuid::Uuid;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn ns(label: &str) -> String {
    format!("test.{label}.{}", Uuid::new_v4())
}

fn owner(label: &str) -> MetadataOwner {
    MetadataOwner::new(format!("user-{label}"), format!("program-{label}"))
}

#[test]
fn metadata_values_are_isolated_by_owner() {
    let _guard = TEST_LOCK.lock().unwrap();
    let alice = MetadataOwner::new("alice", "program-a");
    let bob = MetadataOwner::new("bob", "program-a");
    let other_program = MetadataOwner::new("alice", "program-b");
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
fn invalid_guest_inputs_are_rejected() {
    let _guard = TEST_LOCK.lock().unwrap();
    let owner = owner("limits");
    let namespace = ns("limits");

    assert!(metadata_store::put(&owner, "", "key", vec![]).is_err());
    assert!(metadata_store::put(&owner, &namespace, "", vec![]).is_err());
    assert!(metadata_store::get(&owner, "", "key").is_err());
    assert!(metadata_store::delete(&owner, &namespace, "").is_err());

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
fn total_store_cap_accounts_replacements_and_deletes() {
    let _guard = TEST_LOCK.lock().unwrap();
    let owner = owner("total-cap");
    let namespace = ns("total-cap");
    let chunk_size = MAX_VALUE_BYTES;
    let chunk_count = (MAX_TOTAL_BYTES / chunk_size) - 1;
    let mut keys = Vec::new();

    for index in 0..chunk_count {
        let key = format!("chunk-{index}");
        metadata_store::put(&owner, &namespace, &key, vec![0; chunk_size]).unwrap();
        keys.push(key);
    }

    let last = keys.last().unwrap().clone();
    metadata_store::put(&owner, &namespace, &last, vec![0; chunk_size / 2]).unwrap();
    metadata_store::put(
        &owner,
        &namespace,
        "replacement-delta",
        vec![0; chunk_size / 2],
    )
    .unwrap();

    let err = metadata_store::put(&owner, &namespace, "overflow", vec![0; chunk_size])
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("metadata store total byte cap"),
        "unexpected error: {err}"
    );

    assert!(metadata_store::delete(&owner, &namespace, &keys[0]).unwrap());
    metadata_store::put(&owner, &namespace, "after-delete", vec![0; chunk_size]).unwrap();

    for key in keys {
        let _ = metadata_store::delete(&owner, &namespace, &key);
    }
    let _ = metadata_store::delete(&owner, &namespace, "replacement-delta");
    let _ = metadata_store::delete(&owner, &namespace, "after-delete");
}

#[test]
fn empty_value_entries_are_capped() {
    let _guard = TEST_LOCK.lock().unwrap();
    let owner = owner("entry-cap");
    let namespace = ns("entry-cap");
    let mut keys = Vec::new();

    for index in 0..MAX_ENTRIES {
        let key = format!("empty-{index}");
        metadata_store::put(&owner, &namespace, &key, Vec::new()).unwrap();
        keys.push(key);
    }

    metadata_store::put(&owner, &namespace, &keys[0], b"overwrite".to_vec()).unwrap();

    let err = metadata_store::put(&owner, &namespace, "overflow", Vec::new())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("metadata store entry cap"),
        "unexpected error: {err}"
    );

    assert!(metadata_store::delete(&owner, &namespace, &keys[0]).unwrap());
    metadata_store::put(&owner, &namespace, "replacement", Vec::new()).unwrap();

    for key in keys {
        let _ = metadata_store::delete(&owner, &namespace, &key);
    }
    let _ = metadata_store::delete(&owner, &namespace, "replacement");
}
