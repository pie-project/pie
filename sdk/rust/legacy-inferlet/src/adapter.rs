use crate::api;
use crate::forward::ForwardPass;
use crate::{Blob, Queue, Resource};

pub trait Adapter {
    fn allocate_adapter(&self) -> u32;
    fn deallocate_adapter(&self, ptr: u32);
    fn export_adapter(&self, ptr: u32, name: &str);
    fn import_adapter(&self, name: &str) -> u32;
    fn get_all_exported_adapters(&self) -> Vec<String>;
    fn release_exported_adapter(&self, name: &str);

    fn upload_adapter(&self, adapter_ptr: u32, name: &str, adapter_data: Blob);
    fn download_adapter(&self, adapter_ptr: u32, name: &str);
}

pub trait SetAdapter {
    fn set_adapter(&self, adapter_ptr: u32);
}

impl Adapter for Queue {
    fn allocate_adapter(&self) -> u32 {
        self.allocate_resources(Resource::Adapter, 1)
            .into_iter()
            .next()
            .unwrap()
    }

    fn deallocate_adapter(&self, ptr: u32) {
        self.deallocate_resources(Resource::Adapter, &[ptr])
    }

    fn export_adapter(&self, ptr: u32, name: &str) {
        self.export_resource(Resource::Adapter, &[ptr], name)
    }

    fn import_adapter(&self, name: &str) -> u32 {
        self.import_resource(Resource::Adapter, name)
            .into_iter()
            .next()
            .unwrap()
    }

    fn get_all_exported_adapters(&self) -> Vec<String> {
        self.get_all_exported_resources(Resource::Adapter)
            .into_iter()
            .map(|(name, _)| name)
            .collect()
    }

    fn release_exported_adapter(&self, name: &str) {
        self.release_exported_resources(Resource::Adapter, name)
    }

    fn upload_adapter(&self, adapter_ptr: u32, name: &str, blob: Blob) {
        api::adapter::common::upload_adapter(&self.inner, adapter_ptr, name, blob.inner);
    }

    fn download_adapter(&self, adapter_ptr: u32, name: &str) {
        let _ = api::adapter::common::download_adapter(&self.inner, adapter_ptr, name);
    }
}

impl SetAdapter for ForwardPass {
    fn set_adapter(&self, adapter_ptr: u32) {
        api::adapter::common::set_adapter(&self.inner, adapter_ptr);
    }
}
