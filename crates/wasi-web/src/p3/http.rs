//! `wasi:http@0.3.0` resource types, for the bindings only: a tab hosts no
//! HTTP client, so the interfaces are not linked and a guest that imports
//! them is refused at instantiation, by name.

#[derive(Clone, Default)]
pub struct Fields;

#[derive(Clone, Default)]
pub struct RequestOptions;

pub struct Request;

pub struct Response;
