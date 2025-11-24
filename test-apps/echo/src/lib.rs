use inferlet::{Args, Result, wasi, wstd};

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let text: Option<String> = args.opt_value_from_str(["-t", "--text"])?;
    let delay: Option<u64> = args.opt_value_from_str(["-d", "--delay"])?;
    let text_before_delay: Option<String> =
        args.opt_value_from_str(["-b", "--text-before-delay"])?;

    if let Some(text_before_delay) = text_before_delay {
        println!("{}", text_before_delay);
    }
    if let Some(delay) = delay {
        let nanos = delay * 1_000_000; // Convert milliseconds to nanoseconds
        let pollable = wasi::clocks::monotonic_clock::subscribe_duration(nanos);
        wstd::runtime::AsyncPollable::new(pollable).wait_for().await;
    }
    if let Some(text) = text {
        println!("{}", text);
    }

    Ok(())
}
