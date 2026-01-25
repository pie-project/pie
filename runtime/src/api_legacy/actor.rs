use crate::api_legacy::inferlet;
use crate::instance::InstanceState;
use anyhow::Result;
use wasmtime::component::Resource;
use wasmtime_wasi::WasiView;

#[derive(Debug, Clone)]
pub struct GlobalContext {
    pub uid: String,
}

#[derive(Debug, Clone)]
pub struct Adapter {
    pub uid: String,
}

#[derive(Debug, Clone)]
pub struct Optimizer {
    pub uid: String,
}

impl inferlet::actor::common::Host for InstanceState {}

impl inferlet::actor::common::HostGlobalContext for InstanceState {
    async fn new(&mut self, uid: String) -> Result<Resource<GlobalContext>> {
        let ctx = GlobalContext { uid };
        Ok(self.ctx().table.push(ctx)?)
    }

    async fn destroy(&mut self, this: Resource<GlobalContext>) -> Result<()> {
        let _ = self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn extend(&mut self, this: Resource<GlobalContext>, _page_ids: Vec<u32>, _last_page_len: u32) -> Result<()> {
        let _ctx = self.ctx().table.get(&this)?;
        // TODO: Implement actual logic
        Ok(())
    }

    async fn trim(&mut self, this: Resource<GlobalContext>, _len: u32) -> Result<()> {
        let _ctx = self.ctx().table.get(&this)?;
        // TODO: Implement actual logic
        Ok(())
    }

    async fn read(&mut self, this: Resource<GlobalContext>, _num_tokens: u32, _offset: u32) -> Result<Vec<u32>> {
        let _ctx = self.ctx().table.get(&this)?;
        // TODO: Implement actual logic
        Ok(vec![])
    }

     async fn drop(&mut self, this: Resource<GlobalContext>) -> Result<()> {
        let _ = self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl inferlet::actor::common::HostAdapter for InstanceState {
    async fn new(&mut self, uid: String) -> Result<Resource<Adapter>> {
        let adapter = Adapter { uid };
        Ok(self.ctx().table.push(adapter)?)
    }

    async fn destroy(&mut self, this: Resource<Adapter>) -> Result<()> {
         let _ = self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn blank(&mut self, this: Resource<Adapter>, _rank: u32, _alpha: f32) -> Result<()> {
         let _adapter = self.ctx().table.get(&this)?;
         // TODO: Implement actual logic
        Ok(())
    }

    async fn load(&mut self, this: Resource<Adapter>, _path: String) -> Result<()> {
         let _adapter = self.ctx().table.get(&this)?;
         // TODO: Implement actual logic
        Ok(())
    }

     async fn drop(&mut self, this: Resource<Adapter>) -> Result<()> {
        let _ = self.ctx().table.delete(this)?;
        Ok(())
    }
}


impl inferlet::actor::common::HostOptimizer for InstanceState {
    async fn new(&mut self, uid: String) -> Result<Resource<Optimizer>> {
        let optimizer = Optimizer { uid };
        Ok(self.ctx().table.push(optimizer)?)
    }

    async fn destroy(&mut self, this: Resource<Optimizer>) -> Result<()> {
         let _ = self.ctx().table.delete(this)?;
        Ok(())
    }

    async fn load(&mut self, this: Resource<Optimizer>, _path: String) -> Result<()> {
        let _opt = self.ctx().table.get(&this)?;
        // TODO: Implement actual logic
        Ok(())
    }

    async fn save(&mut self, this: Resource<Optimizer>, _path: String) -> Result<()> {
       let _opt = self.ctx().table.get(&this)?;
        // TODO: Implement actual logic
        Ok(())
    }

    async fn initialize(&mut self, this: Resource<Optimizer>, adapter: Resource<Adapter>, _params: Vec<u8>) -> Result<()> {
         let _opt = self.ctx().table.get(&this)?;
         let _adapter = self.ctx().table.get(&adapter)?;
         // TODO: Implement actual logic
        Ok(())
    }

    async fn update(&mut self, this: Resource<Optimizer>, _params: Vec<u8>) -> Result<()> {
        let _opt = self.ctx().table.get(&this)?;
        // TODO: Implement actual logic
        Ok(())
    }
    
     async fn drop(&mut self, this: Resource<Optimizer>) -> Result<()> {
        let _ = self.ctx().table.delete(this)?;
        Ok(())
    }
}
