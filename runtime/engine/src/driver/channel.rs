//! Channel lifecycle: the registered native binding ([`RegisteredChannel`]),
//! the owning-side handle applications hold ([`ChannelEndpoint`]) and its
//! wait/poison/close semantics, plus the seed-value wire payload
//! ([`ChannelValue`]) [`super::instance::InstanceBindingPlan`] carries.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use pie_driver_abi::PieChannelEndpointBinding;

/// One channel's initial (seed) value delivered at bind time — `channel` is
/// the global channel identity, `bytes` its native-encoded wire payload. No
/// IR semantics live here; this is purely the driver-facing seed-table
/// entry `InstanceBindingPlan::seed_values` carries, next to the
/// `LaunchPlan`/`InstanceBindingPlan` it feeds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChannelValue {
    pub channel: u64,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisteredChannel {
    pub driver_id: usize,
    pub binding: PieChannelEndpointBinding,
    pub reader_wait_id: u64,
    pub writer_wait_id: u64,
}

/// Notifies whichever layer owns the channel's native binding that this
/// endpoint has closed (physically closes/deregisters `channel_id` on the
/// driver that owns it). A leaf callback type — it names no scheduler
/// type — installed by whoever registers the channel and therefore already
/// holds a handle to the owning driver's scheduler
/// (`scheduler::dispatch::register_channel`); `None` in tests that only
/// exercise wait/poison semantics and never call [`ChannelEndpoint::new`]
/// with a closer installed.
pub type ChannelCloser = Arc<dyn Fn(u64) -> anyhow::Result<()> + Send + Sync>;

pub struct ChannelEndpoint {
    registered: RegisteredChannel,
    closed: AtomicBool,
    /// Whether the driver close notification has been handed to an external
    /// batcher (see [`Self::detach_close_notification`]); when set, `close`
    /// still sweeps/frees the waker slots but no longer invokes the closer.
    notify_detached: AtomicBool,
    closer: Option<ChannelCloser>,
}

impl std::fmt::Debug for ChannelEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelEndpoint")
            .field("registered", &self.registered)
            .field("closed", &self.closed)
            .field("closer", &self.closer.is_some())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelWaitError {
    Poisoned(u64),
    Closed,
}

impl std::fmt::Display for ChannelWaitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Poisoned(epoch) => write!(f, "channel is poisoned at epoch {epoch}"),
            Self::Closed => write!(f, "channel is closed"),
        }
    }
}

impl std::error::Error for ChannelWaitError {}

fn load_channel_word(word_base: u64, index: u32) -> u64 {
    unsafe { (&*((word_base as *const AtomicU64).add(index as usize))).load(Ordering::Acquire) }
}

impl ChannelEndpoint {
    pub fn new(registered: RegisteredChannel) -> Self {
        Self {
            registered,
            closed: AtomicBool::new(false),
            notify_detached: AtomicBool::new(false),
            closer: None,
        }
    }

    /// Installs the close-notification callback (see [`ChannelCloser`]);
    /// called by the scheduler dispatch facade, which already holds the
    /// owning driver's scheduler handle.
    pub fn with_closer(mut self, closer: ChannelCloser) -> Self {
        self.closer = Some(closer);
        self
    }

    pub fn registered(&self) -> &RegisteredChannel {
        &self.registered
    }

    pub async fn wait_for_reader_change(&self, observed_tail: u64) -> Result<(), ChannelWaitError> {
        self.wait_for_word_change(
            self.registered.reader_wait_id,
            self.registered.binding.tail_word_index,
            observed_tail,
        )
        .await
    }

    pub async fn wait_for_writer_change(&self, observed_head: u64) -> Result<(), ChannelWaitError> {
        self.wait_for_word_change(
            self.registered.writer_wait_id,
            self.registered.binding.head_word_index,
            observed_head,
        )
        .await
    }

    async fn wait_for_word_change(
        &self,
        wait_id: u64,
        word_index: u32,
        observed: u64,
    ) -> Result<(), ChannelWaitError> {
        let binding = self.registered.binding;
        pie_waker::WaitFuture::new(pie_waker::WakerTable::global(), wait_id, move || {
            let poison = load_channel_word(binding.word_base, binding.poison_word_index);
            if poison != 0 {
                return pie_waker::Readiness::Ready(Err(ChannelWaitError::Poisoned(poison)));
            }
            if load_channel_word(binding.word_base, binding.closed_word_index) != 0 {
                return pie_waker::Readiness::Ready(Err(ChannelWaitError::Closed));
            }
            let current = load_channel_word(binding.word_base, word_index);
            if current > observed {
                pie_waker::Readiness::Ready(Ok(()))
            } else {
                pie_waker::Readiness::Pending {
                    observed_epoch: current,
                }
            }
        })
        .await
    }

    /// Takes over this endpoint's driver close notification: returns the
    /// channel id the caller is now responsible for closing (via a batched
    /// scheduler post), or `None` if the endpoint already closed (the closer
    /// already notified) or the notification was already taken. Wait/poison
    /// bookkeeping is untouched — the endpoint's own drop still sweeps and
    /// frees its waker slots. Callers must outlive-order the endpoint's drop
    /// (e.g. hold it through a resource table they drop themselves) — this
    /// method does not synchronize against a concurrent drop.
    pub fn detach_close_notification(&self) -> Option<u64> {
        if self.closed.load(Ordering::Acquire) {
            return None;
        }
        if self.notify_detached.swap(true, Ordering::AcqRel) {
            return None;
        }
        Some(self.registered.binding.channel_id)
    }

    fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let table = pie_waker::WakerTable::global();
        let wait_ids = [
            self.registered.reader_wait_id,
            self.registered.writer_wait_id,
        ];
        if !self.notify_detached.load(Ordering::Acquire)
            && let Some(closer) = self.closer.as_ref()
            && let Err(error) = closer(self.registered.binding.channel_id)
        {
            tracing::warn!(
                channel_id = self.registered.binding.channel_id,
                ?error,
                "ordered channel close failed"
            );
        }
        table.sweep(&wait_ids);
        for wait_id in wait_ids {
            table.deregister(wait_id);
            table.free(wait_id);
        }
    }
}

impl Drop for ChannelEndpoint {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_endpoint() -> (ChannelEndpoint, Box<[u8]>, Box<[AtomicU64]>, u64, u64) {
        let mirror = vec![0; 8].into_boxed_slice();
        let words = (0..4)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let table = pie_waker::WakerTable::global();
        let reader_wait_id = table.alloc();
        let writer_wait_id = table.alloc();
        let endpoint = ChannelEndpoint::new(RegisteredChannel {
            driver_id: usize::MAX,
            binding: PieChannelEndpointBinding {
                channel_id: 1,
                mirror_base: mirror.as_ptr() as u64,
                word_base: words.as_ptr() as u64,
                mirror_bytes: mirror.len() as u64,
                word_bytes: (words.len() * std::mem::size_of::<AtomicU64>()) as u64,
                cell_bytes: 4,
                capacity: 1,
                head_word_index: 0,
                tail_word_index: 1,
                poison_word_index: 2,
                closed_word_index: 3,
            },
            reader_wait_id,
            writer_wait_id,
        });
        (endpoint, mirror, words, reader_wait_id, writer_wait_id)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn channel_waits_register_then_recheck_reader_and_writer_words() {
        let (endpoint, _mirror, words, reader_wait_id, writer_wait_id) = test_endpoint();
        let reader = endpoint.wait_for_reader_change(0);
        let publish_reader = async {
            tokio::task::yield_now().await;
            words[1].store(1, Ordering::Release);
            let _ = pie_waker::WakerTable::global().publish(reader_wait_id, 1);
        };
        let (result, ()) = tokio::join!(reader, publish_reader);
        result.unwrap();

        let writer = endpoint.wait_for_writer_change(0);
        let publish_writer = async {
            tokio::task::yield_now().await;
            words[0].store(1, Ordering::Release);
            let _ = pie_waker::WakerTable::global().publish(writer_wait_id, 1);
        };
        let (result, ()) = tokio::join!(writer, publish_writer);
        result.unwrap();
    }

    #[test]
    fn detach_close_notification_takes_over_the_driver_close_once() {
        let (endpoint, _mirror, _words, _reader, _writer) = test_endpoint();
        let closes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&closes);
        let endpoint = endpoint.with_closer(Arc::new(move |_id| {
            counter.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }));

        assert_eq!(
            endpoint.detach_close_notification(),
            Some(1),
            "first detach hands the caller the channel id"
        );
        assert_eq!(
            endpoint.detach_close_notification(),
            None,
            "a second detach finds the notification already taken"
        );
        drop(endpoint);
        assert_eq!(
            closes.load(Ordering::Acquire),
            0,
            "drop after detach must not double-notify through the closer"
        );
    }

    #[test]
    fn detach_close_notification_after_close_yields_nothing() {
        let (endpoint, _mirror, _words, _reader, _writer) = test_endpoint();
        let closes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&closes);
        let endpoint = endpoint.with_closer(Arc::new(move |_id| {
            counter.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }));
        endpoint.close();
        assert_eq!(closes.load(Ordering::Acquire), 1, "close notified once");
        assert_eq!(
            endpoint.detach_close_notification(),
            None,
            "the closer already notified; nothing left to hand off"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn channel_wait_surfaces_poison_after_wakeup() {
        let (endpoint, _mirror, words, reader_wait_id, _writer_wait_id) = test_endpoint();
        let reader = endpoint.wait_for_reader_change(0);
        let poison = async {
            tokio::task::yield_now().await;
            words[2].store(7, Ordering::Release);
            let _ = pie_waker::WakerTable::global().publish(reader_wait_id, 7);
        };
        let (result, ()) = tokio::join!(reader, poison);
        assert_eq!(result, Err(ChannelWaitError::Poisoned(7)));
    }
}
