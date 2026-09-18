//! `tracing` to the browser console.

use std::io;

/// A `MakeWriter` that buffers one event's bytes and hands the line to the
/// console when the writer is dropped (tracing's fmt layer writes an event
/// through one writer instance).
pub struct ConsoleWriter(Vec<u8>);

impl io::Write for ConsoleWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for ConsoleWriter {
    fn drop(&mut self) {
        let line = String::from_utf8_lossy(&self.0);
        let line = line.trim_end();
        if line.is_empty() {
            return;
        }
        // Milliseconds on the page clock, so a step's phases can be timed
        // from the console alone.
        let at = web_rt::time::Instant::now().elapsed_since_origin_ms();
        console_log(&format!("{at:9.1} {line}"));
    }
}

#[cfg(target_arch = "wasm32")]
fn console_log(line: &str) {
    web_sys::console::log_1(&line.into());
}

#[cfg(not(target_arch = "wasm32"))]
fn console_log(line: &str) {
    eprintln!("{line}");
}

pub struct MakeConsoleWriter;

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for MakeConsoleWriter {
    type Writer = ConsoleWriter;
    fn make_writer(&'a self) -> ConsoleWriter {
        ConsoleWriter(Vec::new())
    }
}

/// Install the console subscriber. `filter` is an `EnvFilter` directive
/// string such as `info` or `runtime=debug`.
pub fn install(filter: &str) {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let filter = tracing_subscriber::EnvFilter::try_new(filter)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let layer = tracing_subscriber::fmt::layer()
        .with_writer(MakeConsoleWriter)
        .with_ansi(false)
        .without_time();
    let _ = tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .try_init();
}
