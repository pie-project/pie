//! Engine-lifetime inferlet metadata KV store tests.

use uuid::Uuid;

fn ns(label: &str) -> String {
    format!("test.{label}.{}", Uuid::new_v4())
}

#[test]
fn metadata_values_are_namespaced_and_overwritten() {
    let left = ns("left");
    let right = ns("right");

    pie::metadata_store::put(&left, "thread", b"alpha".to_vec()).unwrap();
    pie::metadata_store::put(&right, "thread", b"beta".to_vec()).unwrap();

    assert_eq!(
        pie::metadata_store::get(&left, "thread").unwrap(),
        Some(b"alpha".to_vec())
    );
    assert_eq!(
        pie::metadata_store::get(&right, "thread").unwrap(),
        Some(b"beta".to_vec())
    );

    pie::metadata_store::put(&left, "thread", b"gamma".to_vec()).unwrap();
    assert_eq!(
        pie::metadata_store::get(&left, "thread").unwrap(),
        Some(b"gamma".to_vec())
    );
}

#[test]
fn deleting_missing_or_present_values_is_explicit() {
    let namespace = ns("delete");

    assert!(!pie::metadata_store::delete(&namespace, "sidecar").unwrap());

    pie::metadata_store::put(&namespace, "sidecar", b"state".to_vec()).unwrap();
    assert!(pie::metadata_store::delete(&namespace, "sidecar").unwrap());
    assert_eq!(
        pie::metadata_store::get(&namespace, "sidecar").unwrap(),
        None
    );
    assert!(!pie::metadata_store::delete(&namespace, "sidecar").unwrap());
}

#[test]
fn empty_namespace_or_key_is_rejected() {
    assert!(pie::metadata_store::put("", "key", b"value".to_vec()).is_err());
    assert!(pie::metadata_store::put("namespace", "", b"value".to_vec()).is_err());
    assert!(pie::metadata_store::get("", "key").is_err());
    assert!(pie::metadata_store::delete("namespace", "").is_err());
}
