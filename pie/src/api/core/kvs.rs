use crate::api::inferlet;
use crate::instance::InstanceState;
use crate::kvs;
use crate::service::ServiceCommand;
use tokio::sync::oneshot;

impl inferlet::core::kvs::Host for InstanceState {
    async fn store_get(&mut self, key: String) -> anyhow::Result<Option<String>> {
        let (tx, rx) = oneshot::channel();
        kvs::Command::Get { key, response: tx }.dispatch();
        let res = rx.await?;
        Ok(res)
    }

    async fn store_set(&mut self, key: String, value: String) -> anyhow::Result<()> {
        kvs::Command::Set { key, value }.dispatch();
        Ok(())
    }
    async fn store_delete(&mut self, key: String) -> anyhow::Result<()> {
        kvs::Command::Delete { key }.dispatch();
        Ok(())
    }

    async fn store_exists(&mut self, key: String) -> anyhow::Result<bool> {
        let (tx, rx) = oneshot::channel();
        kvs::Command::Exists { key, response: tx }.dispatch();
        let res = rx.await?;
        Ok(res)
    }

    async fn store_list_keys(&mut self) -> anyhow::Result<Vec<String>> {
        let (tx, rx) = oneshot::channel();
        kvs::Command::ListKeys { response: tx }.dispatch();
        let res = rx.await?;
        Ok(res)
    }
}
