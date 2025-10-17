use crate::api::inferlet;
use crate::instance::InstanceState;
use crate::kvs;
use tokio::sync::oneshot;

impl inferlet::core::kvs::Host for InstanceState {
    async fn store_get(&mut self, key: String) -> anyhow::Result<Option<String>> {
        let (tx, rx) = oneshot::channel();
        kvs::dispatch(kvs::Command::Get { key, response: tx })?;
        let res = rx.await?;
        Ok(res)
    }

    async fn store_set(&mut self, key: String, value: String) -> anyhow::Result<()> {
        kvs::dispatch(kvs::Command::Set { key, value })?;
        Ok(())
    }
    async fn store_delete(&mut self, key: String) -> anyhow::Result<()> {
        kvs::dispatch(kvs::Command::Delete { key })?;
        Ok(())
    }

    async fn store_exists(&mut self, key: String) -> anyhow::Result<bool> {
        let (tx, rx) = oneshot::channel();
        kvs::dispatch(kvs::Command::Exists { key, response: tx })?;
        let res = rx.await?;
        Ok(res)
    }

    async fn store_list_keys(&mut self) -> anyhow::Result<Vec<String>> {
        let (tx, rx) = oneshot::channel();
        kvs::dispatch(kvs::Command::ListKeys { response: tx })?;
        let res = rx.await?;
        Ok(res)
    }
}
