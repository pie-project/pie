use inferlet::{
    Context, Result, Speculator,
    adapter::Adapter,
    model::Model,
    runtime,
    sample::Sampler,
    wstd::http::{Client, Method, Request},
    wstd::io::{AsyncRead, empty},
};
use std::collections::HashMap;

#[inferlet::main]
async fn main(prompt: String) -> Result<String> {
    let model = Model::load(runtime::models().first().unwrap())?;

    let bytes = http_get("https://example.com/loras/math-tutor.safetensors").await?;
    std::fs::write("/scratch/lora.safetensors", &bytes).map_err(|e| e.to_string())?;
    let lora = Adapter::create(&model, "math-tutor")?;
    lora.load("/scratch/lora.safetensors")?;

    let mut ctx = Context::new(&model)?;
    ctx.system("Solve the problem step by step.").user(&prompt).cue();
    ctx.flush().await?;

    let speculator = NGram::new(ctx.seq_len());
    Ok(ctx.generate(Sampler::Argmax)
        .adapter(&lora)
        .speculator(speculator)
        .max_tokens(512)
        .collect_text().await?)
}

async fn http_get(url: &str) -> Result<Vec<u8>> {
    let req = Request::builder().uri(url).method(Method::GET).body(empty())
        .map_err(|e| e.to_string())?;
    let mut body = Client::new().send(req).await.map_err(|e| e.to_string())?.into_body();
    let mut buf = Vec::new();
    body.read_to_end(&mut buf).await.map_err(|e| e.to_string())?;
    Ok(buf)
}

struct NGram { cursor: u32, history: Vec<u32>, table: HashMap<u32, u32> }
impl NGram {
    fn new(start: u32) -> Self { Self { cursor: start, history: Vec::new(), table: HashMap::new() } }
}
impl Speculator for NGram {
    fn draft(&mut self) -> (Vec<u32>, Vec<u32>) {
        let Some(&last) = self.history.last() else { return (Vec::new(), Vec::new()) };
        let mut drafts = Vec::new();
        let mut probe = last;
        for _ in 0..4 {
            let Some(&t) = self.table.get(&probe) else { break };
            drafts.push(t);
            probe = t;
        }
        let positions = (self.cursor..self.cursor + drafts.len() as u32).collect();
        (drafts, positions)
    }
    fn accept(&mut self, accepted: &[u32]) {
        for &t in accepted {
            if let Some(&prev) = self.history.last() { self.table.insert(prev, t); }
            self.history.push(t);
        }
        self.cursor += accepted.len() as u32;
    }
}
