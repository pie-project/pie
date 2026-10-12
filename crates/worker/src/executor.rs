use serde::{Deserialize, Serialize};

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelComponent {
    Full,
    Text,
    Encode,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub hash: [u8; 32],
    pub component: ModelComponent,
}
