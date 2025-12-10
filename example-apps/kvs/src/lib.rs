//! Demonstrates the key-value store API for persistent storage.
//!
//! This example tests the basic CRUD operations (Create, Read, Update, Delete)
//! on the inferlet key-value store, verifying that data persists correctly.

use inferlet::{Args, Result, anyhow};

const HELP: &str = "\
Usage: kvs [OPTIONS]

A program to test the key-value store API.

Options:
  -h, --help  Prints this help message";

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    println!("--- Starting Key-Value Store Test ---");

    // 1. Clean up any previous state by deleting known keys
    println!("\n[Step 1: Cleanup]");
    inferlet::store_delete("test_key_1");
    inferlet::store_delete("test_key_2");
    println!("Cleaned up old keys.");

    // 2. Set initial key-value pairs
    println!("\n[Step 2: Set Values]");
    inferlet::store_set("test_key_1", "Hello");
    inferlet::store_set("test_key_2", "World");
    println!("Set 'test_key_1' to 'Hello'");
    println!("Set 'test_key_2' to 'World'");

    // 3. List keys and verify
    println!("\n[Step 3: List Keys]");
    let mut keys = inferlet::store_list_keys();
    keys.sort();
    println!("Found keys: {:?}", keys);
    if keys != vec!["test_key_1", "test_key_2"] {
        return Err(anyhow!(
            "ListKeys check failed. Expected ['test_key_1', 'test_key_2'], got {:?}",
            keys
        ));
    }
    println!("ListKeys check: PASSED");

    // 4. Get values and verify
    println!("\n[Step 4: Get Values]");
    let val1 = inferlet::store_get("test_key_1").unwrap_or_default();
    println!("Get 'test_key_1': '{}'", val1);
    if val1 != "Hello" {
        return Err(anyhow!(
            "Get check failed for 'test_key_1'. Expected 'Hello', got '{}'",
            val1
        ));
    }
    println!("Get 'test_key_1' check: PASSED");

    let val2 = inferlet::store_get("test_key_2").unwrap_or_default();
    println!("Get 'test_key_2': '{}'", val2);
    if val2 != "World" {
        return Err(anyhow!(
            "Get check failed for 'test_key_2'. Expected 'World', got '{}'",
            val2
        ));
    }
    println!("Get 'test_key_2' check: PASSED");

    // 5. Check for existence
    println!("\n[Step 5: Check Existence]");
    let exists1 = inferlet::store_exists("test_key_1");
    println!("Does 'test_key_1' exist? {}", exists1);
    if !exists1 {
        return Err(anyhow!(
            "Exists check failed for 'test_key_1'. Expected true."
        ));
    }
    println!("Exists 'test_key_1' check: PASSED");

    let exists_nonexistent = inferlet::store_exists("non_existent_key");
    println!("Does 'non_existent_key' exist? {}", exists_nonexistent);
    if exists_nonexistent {
        return Err(anyhow!(
            "Exists check failed for 'non_existent_key'. Expected false."
        ));
    }
    println!("Exists 'non_existent_key' check: PASSED");

    // 6. Delete a key
    println!("\n[Step 6: Delete a Key]");
    inferlet::store_delete("test_key_1");
    println!("Deleted 'test_key_1'");

    // 7. Verify deletion
    println!("\n[Step 7: Verify Deletion]");
    let exists_after_delete = inferlet::store_exists("test_key_1");
    println!(
        "Does 'test_key_1' exist after delete? {}",
        exists_after_delete
    );
    if exists_after_delete {
        return Err(anyhow!(
            "Delete verification failed. 'test_key_1' should not exist."
        ));
    }
    println!("Existence check after delete: PASSED");

    let mut keys_after_delete = inferlet::store_list_keys();
    keys_after_delete.sort();
    println!("Keys after delete: {:?}", keys_after_delete);
    if keys_after_delete != vec!["test_key_2"] {
        return Err(anyhow!(
            "ListKeys after delete failed. Expected ['test_key_2'], got {:?}",
            keys_after_delete
        ));
    }
    println!("ListKeys check after delete: PASSED");

    println!("\n--- Key-Value Store Test Completed Successfully ---");
    Ok(())
}
