use checkpoint::contract::ModelContract;
use checkpoint_dsl::Error;
use poem_dsl::Platform;

use super::model::Model;

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.trunk.import_from_diffusion(src, platform)
    }
}
