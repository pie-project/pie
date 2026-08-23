#[derive(Clone, Debug)]
pub struct ModelMetadata {
    pub tokenizer: Option<Vec<(String, Vec<u8>)>>,

    pub config: Vec<u8>,
}
