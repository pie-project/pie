use anyhow::{Result, anyhow, bail, ensure};
use dashmap::DashMap;
use std::future::Future;
use std::hash::Hash;
use std::sync::Mutex;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::task;

pub(crate) trait ServiceHandler: Send + 'static {
    type Message: Send + 'static;

    fn started(&mut self) -> impl Future<Output = ()> + Send {
        async {}
    }

    fn handle(&mut self, msg: Self::Message) -> impl Future<Output = ()> + Send;

    fn stopped(&mut self) -> impl Future<Output = ()> + Send {
        async {}
    }
}

fn run_handler<H: ServiceHandler>(
    mut handler: H,
    mut rx: UnboundedReceiver<H::Message>,
) -> task::JoinHandle<()> {
    task::spawn(async move {
        handler.started().await;
        while let Some(msg) = rx.recv().await {
            handler.handle(msg).await;
        }
        handler.stopped().await;
    })
}

struct SingletonState<Msg: Send + 'static> {
    tx: UnboundedSender<Msg>,
    handle: task::JoinHandle<()>,
}

pub struct Service<Msg: Send + 'static> {
    state: Mutex<Option<SingletonState<Msg>>>,
}

impl<Msg: Send + 'static> Service<Msg> {
    pub const fn new() -> Self {
        Self {
            state: Mutex::new(None),
        }
    }

    pub fn spawn<H, F>(&self, factory: F) -> Result<()>
    where
        H: ServiceHandler<Message = Msg>,
        F: FnOnce() -> H,
    {
        let handler = factory();
        let (tx, rx) = unbounded_channel();
        let handle = run_handler(handler, rx);

        let mut state = self.state.lock().unwrap();
        ensure!(state.is_none(), "Service already spawned");
        *state = Some(SingletonState { tx, handle });
        Ok(())
    }

    pub fn send(&self, msg: Msg) -> Result<()> {
        let tx = self
            .state
            .lock()
            .unwrap()
            .as_ref()
            .map(|state| state.tx.clone())
            .ok_or_else(|| anyhow!("Service not spawned"))?;
        tx.send(msg).map_err(|_| anyhow!("Service channel closed"))
    }

    #[allow(dead_code)]
    pub async fn shutdown(&self) -> Result<()> {
        let Some(SingletonState { tx, handle }) = self.state.lock().unwrap().take() else {
            return Ok(());
        };
        drop(tx);
        handle
            .await
            .map_err(|e| anyhow!("Service task panicked: {}", e))
    }

    pub fn is_spawned(&self) -> bool {
        self.state.lock().unwrap().is_some()
    }
}

pub struct ServiceMap<K, Msg>
where
    K: Eq + Hash + Send + Sync + 'static,
    Msg: Send + 'static,
{
    map: DashMap<K, UnboundedSender<Msg>>,
    handles: DashMap<K, task::JoinHandle<()>>,
}

impl<K, Msg> ServiceMap<K, Msg>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    Msg: Send + 'static,
{
    pub fn new() -> Self {
        Self {
            map: DashMap::new(),
            handles: DashMap::new(),
        }
    }

    pub fn spawn<H, F>(&self, key: K, factory: F) -> Result<()>
    where
        H: ServiceHandler<Message = Msg>,
        F: FnOnce() -> H,
    {
        let handler = factory();
        let (tx, rx) = unbounded_channel();

        ensure!(
            self.map.insert(key.clone(), tx).is_none(),
            "Service with this key already exists"
        );

        let handle = run_handler(handler, rx);
        self.handles.insert(key, handle);
        Ok(())
    }

    pub fn send(&self, key: &K, msg: Msg) -> Result<()> {
        let tx = self
            .map
            .get(key)
            .ok_or_else(|| anyhow!("Service not found"))?;
        if tx.send(msg).is_err() {
            let closed_tx = tx.clone();
            drop(tx);
            self.map.remove_if(key, |_, v| v.same_channel(&closed_tx));
            self.handles.remove(key);
            bail!("Service channel closed");
        }
        Ok(())
    }

    pub fn remove(&self, key: &K) -> bool {
        self.handles.remove(key);
        self.map.remove(key).is_some()
    }

    #[allow(dead_code)]
    pub async fn join(&self, key: &K) -> Result<()> {
        self.map.remove(key);
        let (_, handle) = self
            .handles
            .remove(key)
            .ok_or_else(|| anyhow!("Service not found"))?;
        handle
            .await
            .map_err(|e| anyhow!("Service task panicked: {}", e))
    }

    #[allow(dead_code)]
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn keys(&self) -> Vec<K> {
        self.map.iter().map(|r| r.key().clone()).collect()
    }

    #[cfg(feature = "session-lifetime-diagnostic")]
    pub fn handle_keys(&self) -> Vec<K> {
        self.handles.iter().map(|r| r.key().clone()).collect()
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}
