//! Demonstrates prefix tree caching with concurrent generation from shared context.
//!
//! This example creates a 1 × 2 × 2 × 2 = 8 prompt tree structure:
//!
//! ```text
//!                          [System Prompt]
//!                        /                 \
//!         [Photosynthesis]                 [Cellular Respiration]
//!         /              \                     /                \
//!    [ELI5]            [High School]   [Location in Cell]     [Main Products]
//!    /    \           /            \         /         \          /    \
//!  [Chef] [Sunlight] [Equation] [Algae] [Mitochondria] [P&A]   [ATP] [CO2]
//! ```
//!
//! Each of the 8 leaf nodes generates text concurrently, sharing KV cache from
//! their common prefixes.

use futures::future;
use inferlet::{
    context::Context, model::Model, runtime,
    ContextExt, InstructExt, Result,
    inference::Sampler,
};

const HELP: &str = "\
Usage: prefix_tree [OPTIONS]

A program to test prefix tree caching by generating text from 8 related prompts concurrently.

Options:
  -n, --num-tokens <TOKENS>  Sets the number of tokens to generate for each prompt [default: 128]
  -h, --help                 Prints this help message
";

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = inferlet::parse_args(args);

    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let max_num_outputs_per_prompt: usize =
        args.value_from_str(["-n", "--num-tokens"]).unwrap_or(128);

    let start = std::time::Instant::now();

    let models = runtime::models();
    let model = Model::load(models.first().ok_or("No models available")?)?;

    let ctx_root = Context::create(&model)?;

    // 1. --- Root Context (Level 0) ---
    ctx_root.system(
        "You are a helpful, friendly, and knowledgeable science tutor for students of all ages. \
        Your goal is to explain complex biological concepts in a clear, accessible, and engaging \
        manner, tailoring your language to the specified audience.",
    );
    ctx_root.flush().await?;

    // 2. --- First Level Forks (Level 1) ---
    let ctx_photo = ctx_root.fork()?;
    ctx_photo.user(
        "I'm curious about the fundamental process of photosynthesis. \
        Could you provide a detailed overview of how plants create their own food using sunlight, \
        water, and carbon dioxide?",
    );

    let ctx_resp = ctx_root.fork()?;
    ctx_resp.user(
        "Now, could you explain the equally important process of cellular respiration? \
        I'd like to understand how organisms, including plants and animals, break down glucose to \
        release the energy needed for life.",
    );

    future::join_all([ctx_photo.flush(), ctx_resp.flush()]).await;

    // 3. --- Second Level Forks (Level 2) ---
    let ctx_photo_eli5 = ctx_photo.fork()?;
    ctx_photo_eli5.user(
        "That sounds complicated. Could you simplify it significantly for me? \
        Please explain the core idea in a way that a curious 5-year-old child could easily grasp \
        and remember. Use a simple analogy.",
    );

    let ctx_photo_hs = ctx_photo.fork()?;
    ctx_photo_hs.user(
        "Thank you. Now, could you provide a more technical explanation suitable for a high school \
        biology student? I'm familiar with basic cell biology and chemistry, so please include \
        relevant terminology like chloroplasts, chlorophyll, and light-dependent reactions.",
    );

    let ctx_resp_loc = ctx_resp.fork()?;
    ctx_resp_loc.user(
        "I'm interested in the specific location within the cell where this process occurs. \
        Can you describe the organelles involved and why their specific structures are uniquely \
        suited for this essential energy-releasing function?",
    );

    let ctx_resp_prod = ctx_resp.fork()?;
    ctx_resp_prod.user(
        "Focusing on the outputs of this metabolic reaction, what are the primary products \
        that result from this process? Please list and briefly describe the significance of \
        each of these molecules for the cell.",
    );

    future::join_all([
        ctx_photo_eli5.flush(),
        ctx_photo_hs.flush(),
        ctx_resp_loc.flush(),
        ctx_resp_prod.flush(),
    ])
    .await;

    // 4. --- Third Level Forks (Level 3) ---
    let mut ctxs = vec![];

    // Photosynthesis -> ELI5 -> ...
    let p1 = ctx_photo_eli5.fork()?;
    p1.user(
        "To make it really fun, please begin your explanation with the exact phrase \
        'Plants are like little chefs...' and continue that cooking analogy to describe \
        how they make their sugary food.",
    );
    p1.cue();
    ctxs.push(p1);

    let p2 = ctx_photo_eli5.fork()?;
    p2.user(
        "Let's zoom in on the energy source for this recipe. Can you specifically detail the \
        crucial role that sunlight plays in this process? Explain what the sun's energy does \
        and why it's so important for the plant's 'kitchen'.",
    );
    p2.cue();
    ctxs.push(p2);

    // Photosynthesis -> High School -> ...
    let p3 = ctx_photo_hs.fork()?;
    p3.user(
        "For a more precise, scientific understanding, please provide the balanced chemical \
        equation for the overall photosynthetic reaction. Also, briefly explain what each part \
        of the equation represents in the context of the plant's metabolism.",
    );
    p3.cue();
    ctxs.push(p3);

    let p4 = ctx_photo_hs.fork()?;
    p4.user(
        "How does this process in terrestrial plants compare to what happens in aquatic organisms \
        like algae or cyanobacteria? Are there any significant differences in the mechanism, \
        pigments used, or the cellular location?",
    );
    p4.cue();
    ctxs.push(p4);

    // Cellular Respiration -> Location -> ...
    let p5 = ctx_resp_loc.fork()?;
    p5.user(
        "Please elaborate specifically on the role of the mitochondria. Describe its inner and \
        outer membranes and the matrix, and explain how this structure makes it the perfect \
        'powerhouse' of the cell during this process.",
    );
    p5.cue();
    ctxs.push(p5);

    let p6 = ctx_resp_loc.fork()?;
    p6.user(
        "Is this metabolic pathway entirely identical in both plant and animal cells? Please \
        compare and contrast the process, highlighting any key similarities or differences in \
        where or how cellular respiration occurs in these two major kingdoms.",
    );
    p6.cue();
    ctxs.push(p6);

    // Cellular Respiration -> Products -> ...
    let p7 = ctx_resp_prod.fork()?;
    p7.user(
        "One of the key products is usable energy. Could you explain in detail the role of \
        adenosine triphosphate (ATP) as the main energy currency? How is it synthesized and \
        then used by the cell to power its activities?",
    );
    p7.cue();
    ctxs.push(p7);

    let p8 = ctx_resp_prod.fork()?;
    p8.user(
        "I understand that carbon dioxide is considered a waste product of this process. Can you \
        elaborate on what exactly happens to this CO2? How does the organism expel it, and what \
        is its ultimate fate in the larger ecosystem?",
    );
    p8.cue();
    ctxs.push(p8);

    // 5. --- Prepare and Execute Futures Concurrently ---
    println!(
        "--- Starting concurrent generation for 8 prompts (max {} tokens each) ---",
        max_num_outputs_per_prompt
    );

    let sampler = Sampler::TopP((0.0, 1.0));

    let generation_futures: Vec<_> = ctxs
        .iter()
        .map(|ctx| {
            let sampler = sampler.clone();
            async move {
                ctx.generate(sampler)
                    .with_max_tokens(max_num_outputs_per_prompt)
                    .collect_text()
                    .await
            }
        })
        .collect();

    let results = future::join_all(generation_futures).await;

    println!(
        "\n--- All 8 generations completed in {:?} ---\n",
        start.elapsed()
    );

    for (i, output_text) in results.iter().enumerate() {
        match output_text {
            Ok(text) => println!("Prompt #{}:\n{:?}\n", i + 1, text),
            Err(e) => println!("Prompt #{}: Error: {}\n", i + 1, e),
        }
    }

    Ok(String::new())
}
