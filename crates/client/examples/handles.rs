//! The Rust client against a real server, for `tests/handles/e2e.py`: argv is
//! the uri, the program, a PNG and a viewer, and a JSON list of prompt tokens.

use client::client::{Client, Launch, ProcessEvent};
use client::message::Handles;

async fn turn(
    client: &Client,
    prog: &str,
    ws: u64,
    tokens: &[u32],
    n: u32,
    drawn: u32,
) -> Vec<u64> {
    let input = serde_json::json!({
        "ws": {"$handle": "ws"}, "tokens": tokens, "max_tokens": n, "seed": 7, "drawn": drawn,
    });
    let moved = Handles::from([("ws".to_string(), ws)]);
    let mut process = client
        .launch(
            prog.to_string(),
            input.to_string(),
            Launch {
                capture_outputs: true,
                handles: moved.clone(),
                returns: moved,
                ..Default::default()
            },
        )
        .await
        .expect("launch");
    let returned = process.wait_for_return().await.expect("return");
    let value: serde_json::Value = serde_json::from_str(&returned).expect("json");
    value["tokens"]
        .as_array()
        .expect("tokens")
        .iter()
        .map(|t| t.as_u64().expect("a token"))
        .collect()
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let [_, uri, prog, see, png, prompt] = &args[..] else {
        panic!("usage: handles <uri> <prog> <see> <png> <prompt>");
    };
    let prompt: Vec<u32> = serde_json::from_str(prompt).expect("prompt");
    let client = Client::connect_with_identity(&format!("{uri}/v1/ws"), "default/bench")
        .await
        .expect("connect");

    let ws = client.new_working_set().await.expect("new");
    let first = turn(&client, prog, ws, &prompt, 12, 0).await;
    let fork = client.fork_handle(ws).await.expect("fork");
    let last = *first.last().expect("a token") as u32;
    let rest = turn(&client, prog, fork, &[last], 20, 12).await;
    let stat = client.stat_handle(ws).await.expect("stat");
    client.close_handle(ws).await.expect("close");
    client.close_handle(fork).await.expect("close");
    let closed = client.stat_handle(ws).await.expect_err("closed");

    let up = client
        .new_frames(std::fs::read(png).expect("png"))
        .await
        .expect("upload");
    let input = serde_json::json!({"image": {"$handle": "image", "kind": "frames"}});
    let moved = Handles::from([("image".to_string(), up)]);
    let mut viewer = client
        .launch(
            see.clone(),
            input.to_string(),
            Launch {
                capture_outputs: true,
                handles: moved.clone(),
                returns: moved,
                ..Default::default()
            },
        )
        .await
        .expect("launch");
    let seen = viewer.wait_for_return().await.expect("return");

    let mut reader = client
        .read_handle(up, "raw-rgb8".to_string())
        .await
        .expect("read");
    let mut raw = 0;
    loop {
        match reader.recv().await.expect("event") {
            ProcessEvent::File(file) => raw = file.data.len(),
            ProcessEvent::Return(_) => break,
            ProcessEvent::Error(why) => panic!("{why}"),
            _ => {}
        }
    }
    println!(
        "{}",
        serde_json::json!({
            "tokens": first.iter().chain(&rest).collect::<Vec<_>>(),
            "stat": serde_json::from_str::<serde_json::Value>(&stat).expect("stat"),
            "closed": closed.to_string(),
            "seen": serde_json::from_str::<serde_json::Value>(&seen).expect("seen"),
            "raw": raw,
        })
    );
    // A `Process` keeps the connection's writer alive, so these go first.
    drop((viewer, reader));
    client.close().await.expect("close");
}
