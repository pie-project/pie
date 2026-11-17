use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Result, anyhow, bail};

use crate::store::registry::{self, with_kv_lock};

const TEARDOWN_GRACE: Duration = Duration::from_secs(2);

static VERSION: AtomicU64 = AtomicU64::new(0);

static LAUNCHES: tokio::sync::RwLock<()> = tokio::sync::RwLock::const_new(());

static RELOADING: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub fn version() -> u64 {
    VERSION.load(Ordering::Acquire)
}

/// The next boot's model starts at version 0.
pub(crate) fn reset() {
    VERSION.store(0, Ordering::Release);
}

pub(crate) type Launching = tokio::sync::RwLockReadGuard<'static, ()>;

pub(crate) fn launching() -> Result<Launching> {
    LAUNCHES.try_read().map_err(|_| {
        anyhow!("the model's weights are being reloaded; launch again once the reload answers")
    })
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct Reloaded {
    pub version: u64,
    pub seconds: f64,
    pub planes: u32,
    pub bytes: u64,
    pub dropped_pages: usize,
}

pub async fn reload(checkpoint: &Path, version: Option<u64>) -> Result<Reloaded> {
    let started = crate::rt::Instant::now();
    if !checkpoint.is_absolute() {
        bail!(
            "{}: a reload names its checkpoint by an absolute path, which the server reads \
             as it is",
            checkpoint.display()
        );
    }
    if !checkpoint.exists() {
        bail!("{}: no such checkpoint on the server", checkpoint.display());
    }
    let _one = RELOADING.lock().await;
    let _launches = LAUNCHES.write().await;
    let stores = registry::all_for_model(0);
    let open = || {
        let held: usize = stores
            .iter()
            .map(|stores| with_kv_lock(&stores.kv, "reload-held", |kv| kv.live_working_sets()))
            .sum();
        (crate::inferlet::process::live_count(), held)
    };
    let mut tick = crate::rt::time::interval(Duration::from_millis(5));
    let grace = crate::rt::Instant::now();
    loop {
        match open() {
            (0, 0) => break,
            (processes, held) if grace.elapsed() >= TEARDOWN_GRACE => bail!(
                "{processes} process(es) running and {held} working set(s) open hold kv \
                 the current weights computed, which new ones would extend: a reload runs \
                 on an idle model, so let them finish (or terminate them, and close the \
                 session's handles) and reload again"
            ),
            _ => {
                tick.tick().await;
            }
        }
    }
    let engines = crate::scheduler::schedulers();
    if engines.is_empty() {
        bail!("no engine serves this model");
    }
    let mut written = ::engine::Reloaded::default();
    let mut failed = None;
    for engine in &engines {
        match crate::scheduler::reload_weights(engine, checkpoint).await {
            Ok(one) => {
                written.planes = written.planes.max(one.planes);
                written.bytes += one.bytes;
            }
            Err(why) => {
                failed = Some(why);
                break;
            }
        }
    }
    let dropped_pages = if written.planes > 0 {
        let domain: crate::store::kv::hash::Hash256 = rand::random();
        stores
            .iter()
            .map(|stores| with_kv_lock(&stores.kv, "reload-rekey", |kv| kv.rekey(domain)))
            .sum()
    } else {
        0
    };
    if let Some(why) = failed {
        return Err(why);
    }
    let version = version.unwrap_or_else(|| self::version().saturating_add(1));
    VERSION.store(version, Ordering::Release);
    let reloaded = Reloaded {
        version,
        seconds: started.elapsed().as_secs_f64(),
        planes: written.planes,
        bytes: written.bytes,
        dropped_pages,
    };
    tracing::info!(
        checkpoint = %checkpoint.display(),
        version,
        seconds = reloaded.seconds,
        planes = reloaded.planes,
        bytes = reloaded.bytes,
        dropped_pages,
        "reloaded the model's weights"
    );
    Ok(reloaded)
}
