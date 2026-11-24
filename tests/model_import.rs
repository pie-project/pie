//! `pie model import` against a fake Hugging Face hub on loopback, named
//! through `HF_ENDPOINT`.
//!
//! Each case runs the real binary in its own `HF_HOME` and `PIE_HOME`. The
//! import fetches only the commit `--revision` pins, checks every file
//! against the hash the listing names before it links it, refuses a listing
//! that would write outside its folders, resumes a split model whose
//! download stopped after its index landed, reports progress as JSON lines
//! under `--json`, and stops when the program that started it closes stdin.
//! What the import does after the download (a conversion this build may not
//! have the engine for) is not what these cases are about.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const REPO: &str = "pie-test/tiny";
const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";
const OTHER: &str = "89abcdef0123456789abcdef0123456789abcdef";
const SHARD_1: &str = "model-00001-of-00002.safetensors";
const SHARD_2: &str = "model-00002-of-00002.safetensors";

/// One file of the fake repository.
#[derive(Clone)]
struct File {
    path: String,
    bytes: Vec<u8>,
    /// Listed as an LFS file, named by its SHA-256; otherwise by its git blob SHA-1.
    lfs: bool,
}

/// What the fake hub serves, and how it misbehaves.
#[derive(Clone, Default)]
struct Repo {
    files: Vec<File>,
    /// The commit every revision resolves to.
    sha: String,
    /// Raw entries added to the listing as they are.
    extra: Vec<serde_json::Value>,
    /// A file served with one byte changed, at its listed size.
    tampered: Option<String>,
    /// A file whose first download sends half its bytes and then hangs.
    stall: Option<String>,
    /// Every request hangs without an answer.
    hang: bool,
}

/// The fake hub: its address, what it serves, and every request it saw.
struct Hub {
    url: String,
    repo: Arc<Mutex<Repo>>,
    seen: Arc<Mutex<Vec<String>>>,
}

impl Hub {
    fn serve(repo: Repo) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind the fake hub");
        let url = format!("http://{}", listener.local_addr().expect("its address"));
        let repo = Arc::new(Mutex::new(repo));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (shared, log) = (repo.clone(), seen.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (repo, seen) = (shared.clone(), log.clone());
                std::thread::spawn(move || answer(stream, &repo, &seen));
            }
        });
        Self { url, repo, seen }
    }

    fn seen(&self) -> Vec<String> {
        self.seen.lock().expect("request log poisoned").clone()
    }
}

fn answer(mut stream: TcpStream, repo: &Mutex<Repo>, seen: &Mutex<Vec<String>>) {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).unwrap_or(0) == 0 {
            return;
        }
        head.push(byte[0]);
    }
    let head = String::from_utf8_lossy(&head).into_owned();
    let target = head.split_whitespace().nth(1).unwrap_or("").to_string();
    let from: usize = head
        .lines()
        .find_map(|line| {
            let line = line.to_ascii_lowercase();
            let range = line.strip_prefix("range: bytes=")?;
            range.trim_end_matches('-').parse().ok()
        })
        .unwrap_or(0);
    seen.lock().expect("request log poisoned").push(match from {
        0 => target.clone(),
        from => format!("{target} from {from}"),
    });
    let repo = repo.lock().expect("repo poisoned").clone();
    if repo.hang {
        std::thread::sleep(Duration::from_secs(120));
        return;
    }

    if target.starts_with(&format!("/api/models/{REPO}/revision/")) {
        let body = serde_json::json!({ "sha": repo.sha }).to_string();
        reply(&mut stream, "200 OK", &[], body.as_bytes());
    } else if target.starts_with(&format!("/api/models/{REPO}/tree/")) {
        let mut listing: Vec<serde_json::Value> = repo.files.iter().map(entry).collect();
        listing.extend(repo.extra.iter().cloned());
        let body = serde_json::Value::Array(listing).to_string();
        reply(&mut stream, "200 OK", &[], body.as_bytes());
    } else if let Some(path) = target.strip_prefix(&format!("/{REPO}/resolve/{}/", repo.sha)) {
        let Some(file) = repo.files.iter().find(|file| file.path == path) else {
            reply(&mut stream, "404 Not Found", &[], b"");
            return;
        };
        let mut bytes = file.bytes.clone();
        if repo.tampered.as_deref() == Some(path)
            && let Some(last) = bytes.last_mut()
        {
            *last ^= 1;
        }
        if repo.stall.as_deref() == Some(path) && from == 0 {
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(&bytes[..bytes.len() / 2]);
            let _ = stream.flush();
            std::thread::sleep(Duration::from_secs(120));
            return;
        }
        if from > 0 {
            let range = format!(
                "Content-Range: bytes {from}-{}/{}",
                bytes.len() - 1,
                bytes.len()
            );
            reply(&mut stream, "206 Partial Content", &[range], &bytes[from..]);
        } else {
            reply(&mut stream, "200 OK", &[], &bytes);
        }
    } else {
        reply(&mut stream, "404 Not Found", &[], b"");
    }
}

fn reply(stream: &mut TcpStream, status: &str, headers: &[String], body: &[u8]) {
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for header in headers {
        head.push_str(header);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body);
}

/// The listing entry the hub's tree API gives for `file`.
fn entry(file: &File) -> serde_json::Value {
    if file.lfs {
        serde_json::json!({
            "type": "file",
            "path": file.path,
            "oid": git_oid(b"version https://git-lfs.github.com/spec/v1\n"),
            "size": 134,
            "lfs": { "oid": sha256(&file.bytes), "size": file.bytes.len() },
        })
    } else {
        serde_json::json!({
            "type": "file",
            "path": file.path,
            "oid": git_oid(&file.bytes),
            "size": file.bytes.len(),
        })
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::Digest;
    hex(&sha2::Sha256::digest(bytes))
}

fn git_oid(bytes: &[u8]) -> String {
    use sha2::Digest;
    let mut hash = sha1::Sha1::new();
    hash.update(format!("blob {}\0", bytes.len()));
    hash.update(bytes);
    hex(&hash.finalize())
}

fn small(path: &str, text: &str) -> File {
    File {
        path: path.to_string(),
        bytes: text.as_bytes().to_vec(),
        lfs: false,
    }
}

fn weights(path: &str, len: usize, seed: u8) -> File {
    File {
        path: path.to_string(),
        bytes: (0..len).map(|i| (i % 251) as u8 ^ seed).collect(),
        lfs: true,
    }
}

/// A model split in two shards, with the index that names them.
fn split_model() -> Vec<File> {
    let index = serde_json::json!({
        "metadata": { "total_size": 2 * 256 * 1024 },
        "weight_map": { "a.weight": SHARD_1, "b.weight": SHARD_2 },
    });
    vec![
        small("config.json", r#"{"model_type": "tiny"}"#),
        small("model.safetensors.index.json", &index.to_string()),
        weights(SHARD_1, 256 * 1024, 0x11),
        weights(SHARD_2, 256 * 1024, 0x22),
    ]
}

fn repo(files: Vec<File>) -> Repo {
    Repo {
        files,
        sha: COMMIT.to_string(),
        ..Repo::default()
    }
}

/// `pie --json model import` of the fake repository at `COMMIT`, in `home`.
fn import(home: &Path, hub: &Hub, extra: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pie"));
    command
        .args(["--json", "model", "import", REPO, "--revision", COMMIT])
        .args(["--drafter", "none"])
        .args(extra)
        .env("HF_ENDPOINT", &hub.url)
        .env("HF_HOME", home.join("hf"))
        .env("PIE_HOME", home.join("pie"))
        .env_remove("HF_TOKEN")
        .env_remove("HUGGING_FACE_HUB_TOKEN")
        .env_remove("HUGGINGFACE_HUB_TOKEN")
        .env_remove("HF_HUB_CACHE")
        .env_remove("PIE_CONFIG")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    command
}

/// The import's exit status, stdout and stderr, once it finishes.
fn finished(command: &mut Command) -> (ExitStatus, String, String) {
    let output = command
        .stdin(Stdio::null())
        .output()
        .expect("run pie model import");
    (
        output.status,
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn until(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !done() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn exited_within(child: &mut Child, limit: Duration) -> Option<ExitStatus> {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().expect("poll the import") {
            return Some(status);
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    None
}

fn repo_dir(home: &Path) -> PathBuf {
    home.join("hf/hub/models--pie-test--tiny")
}

fn snapshot(home: &Path) -> PathBuf {
    repo_dir(home).join("snapshots").join(COMMIT)
}

/// Every file under `dir`, at any depth.
fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(files_under(&path));
        } else {
            found.push(path);
        }
    }
    found
}

/// The `{"progress": "download", "done", "total"}` lines, as `(done, total)`.
fn download_progress(stdout: &str) -> Vec<(u64, u64)> {
    stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|line| line["progress"] == "download")
        .map(|line| {
            (
                line["done"].as_u64().expect("done"),
                line["total"].as_u64().expect("total"),
            )
        })
        .collect()
}

#[test]
fn a_tampered_shard_is_deleted_and_the_import_fails_naming_it() {
    let home = tempfile::tempdir().expect("home");
    let mut served = repo(vec![
        small("config.json", "{}"),
        weights("model.safetensors", 64 * 1024, 0x33),
    ]);
    served.tampered = Some("model.safetensors".to_string());
    let oid = sha256(&served.files[1].bytes);
    let hub = Hub::serve(served);

    let (status, _, stderr) = finished(&mut import(home.path(), &hub, &[]));

    assert!(!status.success(), "a tampered file must fail the import");
    assert!(
        stderr.contains("model.safetensors") && stderr.contains(&oid),
        "the refusal names the file and the hash it should have had: {stderr}"
    );
    assert!(
        !snapshot(home.path()).join("model.safetensors").exists(),
        "a file that does not match is never linked into the snapshot"
    );
    let blobs: Vec<PathBuf> = files_under(&repo_dir(home.path()).join("blobs"));
    assert!(
        blobs
            .iter()
            .all(|blob| !blob.to_string_lossy().contains(&oid)),
        "neither the blob nor its partial stays: {blobs:?}"
    );
}

#[test]
fn a_revision_the_hub_resolves_to_another_commit_is_refused() {
    let home = tempfile::tempdir().expect("home");
    let mut served = repo(vec![small("config.json", "{}")]);
    served.sha = OTHER.to_string();
    let hub = Hub::serve(served);

    let (status, _, stderr) = finished(&mut import(home.path(), &hub, &[]));

    assert!(!status.success());
    assert!(
        stderr.contains(COMMIT) && stderr.contains(OTHER),
        "the refusal names both commits: {stderr}"
    );
    assert!(
        hub.seen()
            .iter()
            .all(|request| request.contains("/revision/")),
        "nothing past the revision is asked for: {:?}",
        hub.seen()
    );
    assert!(
        files_under(&home.path().join("hf")).is_empty(),
        "nothing is written"
    );
}

#[test]
fn a_listing_with_a_path_out_of_the_snapshot_is_refused_before_any_write() {
    let home = tempfile::tempdir().expect("home");
    let mut served = repo(vec![small("config.json", "{}")]);
    served.extra.push(serde_json::json!({
        "type": "file",
        "path": "../../../../../escape.json",
        "oid": git_oid(b"{}"),
        "size": 2,
    }));
    let hub = Hub::serve(served);

    let (status, _, stderr) = finished(&mut import(home.path(), &hub, &[]));

    assert!(!status.success());
    assert!(stderr.contains("escape.json"), "{stderr}");
    assert!(
        hub.seen()
            .iter()
            .all(|request| !request.contains("/resolve/")),
        "no file is fetched from a refused listing"
    );
    assert!(
        files_under(home.path())
            .iter()
            .all(|file| !file.ends_with("escape.json")),
        "nothing is written out of the cache"
    );
    assert!(
        files_under(&home.path().join("hf")).is_empty(),
        "nor in it: {:?}",
        files_under(&home.path().join("hf"))
    );
}

#[test]
fn a_listing_with_an_object_id_that_is_not_hex_is_refused_before_any_write() {
    let home = tempfile::tempdir().expect("home");
    let mut served = repo(vec![small("config.json", "{}")]);
    served.extra.push(serde_json::json!({
        "type": "file",
        "path": "tokenizer.json",
        "oid": "../../../escape",
        "size": 2,
    }));
    let hub = Hub::serve(served);

    let (status, _, stderr) = finished(&mut import(home.path(), &hub, &[]));

    assert!(!status.success());
    assert!(stderr.contains("../../../escape"), "{stderr}");
    assert!(
        hub.seen()
            .iter()
            .all(|request| !request.contains("/resolve/"))
    );
    assert!(
        files_under(&home.path().join("hf")).is_empty(),
        "nothing is written: {:?}",
        files_under(&home.path().join("hf"))
    );
}

#[test]
fn an_import_stops_when_its_parent_closes_stdin() {
    let home = tempfile::tempdir().expect("home");
    let mut served = repo(vec![small("config.json", "{}")]);
    served.hang = true;
    let hub = Hub::serve(served);

    let mut child = import(home.path(), &hub, &["--parent-stdin"])
        .stdin(Stdio::piped())
        .spawn()
        .expect("start the import");
    until("the import to ask the hub", || !hub.seen().is_empty());
    drop(child.stdin.take());

    let status = exited_within(&mut child, Duration::from_secs(2));
    if status.is_none() {
        let _ = child.kill();
    }
    assert!(
        status.is_some_and(|status| !status.success()),
        "the import outlived its parent's stdin: {status:?}"
    );
}

#[test]
fn a_split_import_stopped_after_its_index_landed_resumes_with_json_progress() {
    let home = tempfile::tempdir().expect("home");
    let files = split_model();
    let total: u64 = files.iter().map(|file| file.bytes.len() as u64).sum();
    let mut served = repo(files.clone());
    served.stall = Some(SHARD_2.to_string());
    let hub = Hub::serve(served);

    // The first import fetches everything but half of the second shard, and
    // its parent goes away.
    let mut child = import(home.path(), &hub, &["--parent-stdin"])
        .stdin(Stdio::piped())
        .spawn()
        .expect("start the import");
    let partial = repo_dir(home.path())
        .join("blobs")
        .join(format!("{}.incomplete", sha256(&files[3].bytes)));
    until("the index, the first shard and half of the second", || {
        snapshot(home.path())
            .join("model.safetensors.index.json")
            .is_file()
            && snapshot(home.path()).join(SHARD_1).is_file()
            && std::fs::metadata(&partial).is_ok_and(|m| m.len() >= 128 * 1024)
    });
    drop(child.stdin.take());
    assert!(
        exited_within(&mut child, Duration::from_secs(2)).is_some(),
        "the first import outlived its parent's stdin"
    );
    let _ = child.kill();
    assert!(!snapshot(home.path()).join(SHARD_2).exists());
    let half = std::fs::metadata(&partial).expect("the partial").len();
    let on_disk = half
        + files[..3]
            .iter()
            .map(|file| file.bytes.len() as u64)
            .sum::<u64>();

    // The second import sees the index without its second shard and
    // fetches the rest of it.
    hub.repo.lock().expect("repo poisoned").stall = None;
    let (_, stdout, stderr) = finished(&mut import(home.path(), &hub, &[]));

    let asked = format!("/{REPO}/resolve/{COMMIT}/{SHARD_2} from {half}");
    assert!(
        hub.seen().contains(&asked),
        "the second shard resumes from the bytes on disk; the hub saw {:?}\n{stderr}",
        hub.seen()
    );
    assert_eq!(
        std::fs::read(snapshot(home.path()).join(SHARD_2)).expect("the second shard"),
        files[3].bytes,
        "the resumed shard is whole, and it passed its hash"
    );
    assert!(
        files_under(&repo_dir(home.path()).join("blobs"))
            .iter()
            .all(|blob| blob.extension().is_none_or(|e| e != "incomplete")),
        "no partial stays"
    );

    let progress = download_progress(&stdout);
    let (first, last) = (progress.first(), progress.last());
    assert!(
        first.is_some_and(|&(done, all)| done >= on_disk && all == total),
        "the first progress line counts what was on disk: {progress:?}\n{stdout}"
    );
    assert_eq!(
        last,
        Some(&(total, total)),
        "the last one reaches the total"
    );
    assert!(
        stdout
            .lines()
            .filter(|line| line.starts_with('{') && line.contains("\"progress\""))
            .all(|line| line.len() < 200 && !line.contains('\n')),
        "each progress update is one short line"
    );
}
