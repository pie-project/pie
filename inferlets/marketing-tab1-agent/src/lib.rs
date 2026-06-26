use inferlet::{Context, Result, sample::Sampler, tool, tools};

/// Search the web for current information.
#[tool]
async fn web_search(query: String) -> Result<String> {
    Ok(format!("(stub result for: {query})"))
}

#[inferlet::main]
async fn main(prompt: String) -> Result<String> {
    let mut ctx = Context::new()?;
    ctx.system("Use web_search if you need fresh facts, then answer.")
        .equip(&[&web_search])?
        .user(&prompt);

    loop {
        let mut tdec = tools::Decoder::new();
        let mut full = Vec::new();
        let call = {
            let mut g = ctx.generate(Sampler::Argmax).max_tokens(512);
            loop {
                let Some(step) = g.next()? else { break None };
                let out = step.execute().await?;
                full.extend_from_slice(&out.tokens);
                if let tools::Event::Call(name, args) = tdec.feed(&out.tokens)? {
                    break Some((name, args));
                }
            }
        };

        let Some((name, args)) = call else {
            return Ok(inferlet::model::decode(&full)?);
        };

        let result = match name.as_str() {
            "web_search" => web_search::call(&args).await?,
            _ => return Err(format!("unknown tool: {name}")),
        };
        ctx.append(&tools::answer_prefix(&name, &result));
    }
}
