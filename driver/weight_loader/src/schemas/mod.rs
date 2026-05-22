mod generic;

use crate::schema::ModelSchema;

use generic::GenericSchema;

pub fn builtin_schemas() -> &'static [&'static dyn ModelSchema] {
    BUILTIN_SCHEMAS
}

pub fn generic_schema() -> &'static dyn ModelSchema {
    &GENERIC_SCHEMA
}

static GENERIC_SCHEMA: GenericSchema = GenericSchema { names: &[] };
static LLAMA_SCHEMA: GenericSchema = GenericSchema {
    names: &["llama", "mistral", "ministral", "mixtral", "phi3", "phi-3"],
};
static QWEN_SCHEMA: GenericSchema = GenericSchema {
    names: &["qwen", "qwen2", "qwen3", "qwen3.6"],
};
static GEMMA_SCHEMA: GenericSchema = GenericSchema {
    names: &["gemma", "gemma4"],
};
static OLMO_SCHEMA: GenericSchema = GenericSchema {
    names: &["olmo", "olmo3", "olmo-3"],
};
static GPT_OSS_SCHEMA: GenericSchema = GenericSchema {
    names: &["gpt_oss", "gpt-oss", "gptoss"],
};
static BUILTIN_SCHEMAS: &[&dyn ModelSchema] = &[
    &LLAMA_SCHEMA,
    &QWEN_SCHEMA,
    &GEMMA_SCHEMA,
    &OLMO_SCHEMA,
    &GPT_OSS_SCHEMA,
];
