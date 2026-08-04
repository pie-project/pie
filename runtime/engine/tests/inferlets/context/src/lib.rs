//! Context test inferlet — exercises model, working-set, and tokenizer host
//! APIs on the raw keep-core surface (off the `Context` facade).

use inferlet::working_set::KvWorkingSet;
use inferlet::{Result, model};

#[inferlet::main]
async fn main(_input: String) -> Result<String> {
    // The engine serves exactly one model — encode directly.
    let encoded = model::encode("hello world");

    // The KV working set — the raw keep-core replacement for `Context`.
    let kv = KvWorkingSet::new();

    // The guest owns its own token buffer (the facade's `Context::append` /
    // `buffer()` staging is now a plain guest-held Vec).
    let buffered = encoded.clone();

    // Page size is a driver constant, so it comes off `model`; the extent is
    // a property of this working set, so it comes off `kv`.
    let page_size = model::kv_page_size();
    let page_len = kv.page_len();

    Ok(format!(
        "encoded:{} buffered:{} page_size:{} page_len:{}",
        encoded.len(),
        buffered.len(),
        page_size,
        page_len
    ))
}
