//! Dynamic linking support for Wasm components.
//!
//! This module implements dynamic linking for WebAssembly Component Model components,
//! enabling runtime composition where one component can import and use exports from
//! another component. This is achieved through a "proxy registration" mechanism.
//!
//! # Overview
//!
//! When a program depends on library components, we need to:
//! 1. Instantiate the library components first
//! 2. Make their exports available to dependents (other libraries and the main program)
//! 3. Handle resource handle translation between components
//! 4. Track and properly end cross-component borrows
//!
//! # Proxy Registration
//!
//! The core mechanism is "proxy registration": For each export from a library component,
//! we register a corresponding host-defined entity in the linker that acts as a proxy.
//!
//! ## Proxy Functions
//!
//! For each function exported by a library, we register a host function that:
//! - Receives calls from the caller component
//! - Transforms arguments (especially resource handles)
//! - Forwards the call to the actual library function
//! - Transforms return values back to the caller's view
//!
//! ## Proxy Resources
//!
//! For each resource type exported by a library, we register a host-defined resource type
//! (`ProxyResource`) that:
//! - Acts as a proxy for the actual guest resource
//! - Maintains a mapping between host `rep` values and guest `ResourceAny` handles
//! - Forwards destructor calls to the actual guest resource
//!
//! # Resource Handle Transformation
//!
//! Resource handle translation is necessary because the component that defines a resource
//! type operates on the real guest resource handles, while components that depend on it
//! receive host-defined proxy resource handles. When a dependent component passes a
//! resource to the defining component (e.g., calling a method on a resource), the proxy
//! handle must be translated back to the real guest handle.
//!
//! We maintain bidirectional mappings in `InstanceState`:
//! - `dynamic_resource_map`: host rep → guest `ResourceAny`
//! - `guest_resource_map`: guest `ResourceAny` → host rep (for identity preservation)
//!
//! ## Caller → Callee Transformation (Arguments)
//!
//! When a caller passes a resource to a callee:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────────────┐
//! │                    Argument Transformation (Caller → Callee)                     │
//! └──────────────────────────────────────────────────────────────────────────────────┘
//!
//!   CASE 1: Callee defines the resource
//!   ────────────────────────────────────
//!
//!   Caller sees:          Host does:                      Callee receives:
//!   ┌──────────────┐      ┌────────────────────────┐      ┌──────────────────┐
//!   │ Proxy Handle ├─────►│ Extract rep, look up   │─────►│ Original Guest   │
//!   │ (rep=42)     │      │ guest ResourceAny      │      │ ResourceAny      │
//!   └──────────────┘      └────────────────────────┘      └──────────────────┘
//!
//!   The caller holds a proxy handle. We extract the rep, look up the corresponding
//!   guest ResourceAny in dynamic_resource_map, and pass that to the callee.
//!
//!
//!   CASE 2: Callee doesn't define the resource
//!   ──────────────────────────────────────────
//!
//!   Caller sees:          Host does:                      Callee receives:
//!   ┌──────────────┐      ┌────────────────────────┐      ┌──────────────────┐
//!   │ Proxy Handle ├─────►│ Pass through unchanged │─────►│ Same Proxy       │
//!   │ (rep=99)     │      │                        │      │ Handle (rep=99)  │
//!   └──────────────┘      └────────────────────────┘      └──────────────────┘
//!
//!   If the resource is not defined by the callee, the proxy handle passes through
//!   unchanged. For borrow types, we additionally track them in borrows_to_end
//!   for cleanup after the call completes.
//! ```
//!
//! ## Callee → Caller Transformation (Return Values)
//!
//! When a callee returns a resource to a caller:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────────────┐
//! │                  Return Value Transformation (Callee → Caller)                   │
//! └──────────────────────────────────────────────────────────────────────────────────┘
//!
//!   CASE 1: Resource defined by callee
//!   ───────────────────────────────────
//!
//!   Callee returns:       Host does:                      Caller receives:
//!   ┌──────────────┐      ┌────────────────────────┐      ┌──────────────────┐
//!   │ Guest        │      │ 1. Check if already    │      │ Proxy Handle     │
//!   │ ResourceAny  ├─────►│    known → reuse rep   │─────►│ (rep=42)         │
//!   │              │      │ 2. Else alloc new rep  │      │                  │
//!   │              │      │ 3. Store mapping       │      │                  │
//!   └──────────────┘      └────────────────────────┘      └──────────────────┘
//!
//!   The callee returns a guest ResourceAny. If already in our map, we reuse
//!   the same rep to preserve identity. Otherwise, allocate a new rep and
//!   create the proxy handle.
//!
//!
//!   CASE 2: Resource defined elsewhere
//!   ───────────────────────────────────
//!
//!   Callee returns:       Host does:                      Caller receives:
//!   ┌──────────────┐      ┌────────────────────────┐      ┌──────────────────┐
//!   │ Proxy Handle ├─────►│ Pass through unchanged │─────►│ Same Proxy       │
//!   │ (rep=99)     │      │                        │      │ Handle (rep=99)  │
//!   └──────────────┘      └────────────────────────┘      └──────────────────┘
//!
//!   If the resource is not defined by the callee, the handle is already a
//!   proxy handle and passes through unchanged.
//! ```
//!
//! # Borrow Tracking
//!
//! The Component Model has "borrow" semantics where a resource can be temporarily
//! lent to another component. The borrow must be "ended" after the call completes.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────────────┐
//! │                          Cross-Component Borrow Tracking                         │
//! └──────────────────────────────────────────────────────────────────────────────────┘
//!
//!   Timeline of a call with a borrowed resource:
//!
//!   Caller                    Host (Proxy Function)                    Callee
//!      │                              │                                   │
//!      │  call foo(borrow<R>)         │                                   │
//!      ├─────────────────────────────►│                                   │
//!      │                              │  1. Transform args                │
//!      │                              │     - Detect cross-component      │
//!      │                              │       borrow (R not from callee)  │
//!      │                              │     - Track in borrows_to_end     │
//!      │                              │                                   │
//!      │                              │  2. Forward call                  │
//!      │                              ├──────────────────────────────────►│
//!      │                              │                                   │
//!      │                              │  3. Call returns                  │
//!      │                              │◄──────────────────────────────────┤
//!      │                              │                                   │
//!      │                              │  4. End borrows:                  │
//!      │                              │     resource_drop_async() on      │
//!      │                              │     each borrow in borrows_to_end │
//!      │                              │                                   │
//!      │  return                      │  5. Transform return values       │
//!      │◄─────────────────────────────┤                                   │
//!      │                              │                                   │
//!
//!   Key insight: When a borrow crosses from Component A → Host → Component B,
//!   the host must explicitly signal the end of the borrow by calling
//!   resource_drop_async() on the borrowed ResourceAny handle AFTER the callee
//!   returns but BEFORE returning to the caller.
//!
//!   Note: For borrows where the callee owns the resource (same component),
//!   the borrow is automatically ended when we extract the rep from the
//!   proxy handle via try_from_resource_any().
//! ```
//!
//! # Call Forwarding Flow
//!
//! The complete flow for a forwarded function call:
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────────────────┐
//! │                              Call Forwarding Flow                                │
//! └──────────────────────────────────────────────────────────────────────────────────┘
//!
//!   ┌─────────────────────────────────────────────────────────────────────────────┐
//!   │ 1. TRANSFORM ARGUMENTS TO CALLEE VIEW                                       │
//!   │    ┌─────────────────────────────────────────────────────────────────────┐  │
//!   │    │ For each argument:                                                  │  │
//!   │    │   • Primitive types: pass through                                   │  │
//!   │    │   • Own<R> where callee defines R: proxy handle → guest resource    │  │
//!   │    │   • Own<R> where callee doesn't define R: pass through              │  │
//!   │    │   • Borrow<R> where callee defines R: proxy → guest (auto-ended)    │  │
//!   │    │   • Borrow<R> cross-component: pass through + track for cleanup     │  │
//!   │    │   • Composite types (list, record, etc.): recurse into elements     │  │
//!   │    └─────────────────────────────────────────────────────────────────────┘  │
//!   └─────────────────────────────────────────────────────────────────────────────┘
//!                                       │
//!                                       ▼
//!   ┌─────────────────────────────────────────────────────────────────────────────┐
//!   │ 2. CALL CALLEE FUNCTION                                                     │
//!   │    call_async() + post_return_async()                                       │
//!   └─────────────────────────────────────────────────────────────────────────────┘
//!                                       │
//!                                       ▼
//!   ┌─────────────────────────────────────────────────────────────────────────────┐
//!   │ 3. END CROSS-COMPONENT BORROWS                                              │
//!   │    For each borrow in borrows_to_end:                                       │
//!   │      resource_drop_async() to signal borrow completion                      │
//!   └─────────────────────────────────────────────────────────────────────────────┘
//!                                       │
//!                                       ▼
//!   ┌─────────────────────────────────────────────────────────────────────────────┐
//!   │ 4. TRANSFORM RETURN VALUES TO CALLER VIEW                                   │
//!   │    ┌─────────────────────────────────────────────────────────────────────┐  │
//!   │    │ For each return value:                                              │  │
//!   │    │   • Primitive types: pass through                                   │  │
//!   │    │   • Own<R> where callee defines R: guest resource → proxy handle    │  │
//!   │    │     (reuse existing rep if known, else allocate new)                │  │
//!   │    │   • Own<R> where callee doesn't define R: pass through              │  │
//!   │    │   • Composite types: recurse into elements                          │  │
//!   │    └─────────────────────────────────────────────────────────────────────┘  │
//!   └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use wasmtime::component::types::{ComponentInstance as ComponentInstanceType, ComponentItem, Type};
use wasmtime::component::{
    Component, Func, Instance, Linker, LinkerInstance, Resource, ResourceAny, ResourceType, Val,
};
use wasmtime::{Engine, Store, StoreContextMut};

use super::{InstanceState, RuntimeError};

/// Proxy marker type for host-defined resources used in dynamic linking.
/// This is a phantom type used to create host resource handles.
struct ProxyResource;

/// Categories of functions in the component model
enum FuncCategory {
    Constructor { resource_name: String },
    Method { resource_name: String },
    StaticMethod { resource_name: String },
    FreeFunction,
}

impl FuncCategory {
    /// Categorize a function based on its name.
    fn from_name(func_name: &str) -> Self {
        // Fast path: free functions don't start with '['
        if !func_name.starts_with('[') {
            return Self::FreeFunction;
        }

        // Check in order of likelihood: method > static > constructor

        // Method: [method]resource-name.method-name
        if let Some(resource_name) = func_name
            .strip_prefix("[method]")
            .and_then(|rest| rest.find('.').map(|pos| &rest[..pos]))
        {
            return Self::Method {
                resource_name: resource_name.into(),
            };
        }

        // Static method: [static]resource-name.method-name
        if let Some(resource_name) = func_name
            .strip_prefix("[static]")
            .and_then(|rest| rest.find('.').map(|pos| &rest[..pos]))
        {
            return Self::StaticMethod {
                resource_name: resource_name.into(),
            };
        }

        // Constructor: [constructor]resource-name
        if let Some(resource_name) = func_name.strip_prefix("[constructor]") {
            return Self::Constructor {
                resource_name: resource_name.into(),
            };
        }

        // Fallback case
        Self::FreeFunction
    }
}

/// When function calls are forwarded from one component to another, the resource
/// handles in the arguments sometimes need to be transformed. For example, consider
/// the following scenario: Caller component A owns a resource defined in another
/// component B, and A calls a method on that resource. The control flow will transfer
/// from A to B. However, under our dynamic linking mechanism, the actual resource handle
/// the component A is holding is the host-defined proxy resource handle. Therefore,
/// during function call forwarding, we need to transform the resource handle from the
/// proxy resource handle to the actual resource handle defined in the called component B.
///
/// In addition, during the transformation, we need to track any cross-component borrows
/// that need to be ended after the call completes. This is because the cross-component
/// borrows are not automatically ended when the call completes. Therefore, we need to
/// track them and end them after the call completes.
struct TransformedArgs {
    /// The transformed argument values
    args: Vec<Val>,
    /// Borrowed `ResourceAny` handles that need to be dropped after the call completes
    /// to signal that the borrows have ended (cross-component borrows only)
    borrows_to_end: Vec<ResourceAny>,
}

/// Transform arguments from caller view to callee view.
/// Only resources defined in the callee component are transformed from the host-defined
/// proxy resource handle to the actual resource handle defined in the callee component.
/// Cross-component borrows are tracked in borrows_to_end for cleanup after the call.
fn transform_args_to_callee_view(
    store: &mut StoreContextMut<'_, InstanceState>,
    args: &[Val],
    arg_types: &[Type],
    callee_defined_resource_types: &[ResourceType],
) -> Result<TransformedArgs, wasmtime::Error> {
    if args.len() != arg_types.len() {
        return Err(wasmtime::Error::msg(format!(
            "argument count mismatch: got {}, expected {}",
            args.len(),
            arg_types.len()
        )));
    }

    let mut borrows_to_end = Vec::new();
    let mut transformed_args = Vec::with_capacity(args.len());

    for (val, ty) in args.iter().zip(arg_types.iter()) {
        let transformed = recursive_transform_args_to_callee_view(
            store,
            val.clone(),
            ty,
            callee_defined_resource_types,
            &mut borrows_to_end,
        )?;
        transformed_args.push(transformed);
    }

    Ok(TransformedArgs {
        args: transformed_args,
        borrows_to_end,
    })
}

/// Transform results from callee view to caller view.
/// Only returned resources defined in the callee component are transformed to the host-defined
/// proxy resource handle.
fn transform_returns_to_caller_view(
    store: &mut StoreContextMut<'_, InstanceState>,
    returns: Vec<Val>,
    return_type: &[Type],
    callee_defined_resource_types: &[ResourceType],
) -> Result<Vec<Val>, wasmtime::Error> {
    if returns.len() != return_type.len() {
        return Err(wasmtime::Error::msg(format!(
            "result count mismatch: got {}, expected {}",
            returns.len(),
            return_type.len()
        )));
    }

    let mut transformed_returns = Vec::with_capacity(returns.len());
    for (val, ty) in returns.into_iter().zip(return_type.iter()) {
        let transformed = recursive_transform_returns_to_caller_view(
            store,
            val,
            ty,
            callee_defined_resource_types,
        )?;
        transformed_returns.push(transformed);
    }
    Ok(transformed_returns)
}

/// Transform resource handles from caller view to callee view, collecting any
/// cross-component borrows that need to be ended after the call completes.
/// This function recursively processes composite types to find all nested resource handles.
fn recursive_transform_args_to_callee_view(
    store: &mut StoreContextMut<'_, InstanceState>,
    val: Val,
    ty: &Type,
    callee_defined_resource_types: &[ResourceType],
    borrows_to_end: &mut Vec<ResourceAny>,
) -> Result<Val, wasmtime::Error> {
    match ty {
        Type::Borrow(resource_type) => match val {
            Val::Resource(resource_any) => {
                // If callee defines this resource type, convert the host-defined proxy resource
                // handle to the actual resource handle defined in the callee component.
                if callee_defined_resource_types.contains(resource_type) {
                    // We need not explicitly track the borrow for cleanup here, because when the
                    // resource handle in the argument is converted to the host-defined proxy
                    // resource handle, the borrow is ended inside `try_from_resource_any`.
                    let host_resource: Resource<ProxyResource> =
                        Resource::try_from_resource_any(resource_any, &mut *store)?;
                    let rep = host_resource.rep();
                    let guest_resource =
                        store.data().get_dynamic_resource(rep).ok_or_else(|| {
                            wasmtime::Error::msg(format!("unknown resource rep={}", rep))
                        })?;
                    Ok(Val::Resource(guest_resource))
                // If the callee doesn't define this resource type, pass through the host-defined
                // proxy resource handle. In addition, since this is a cross-component borrow,
                // we need to track it for cleanup after the call completes.
                } else {
                    borrows_to_end.push(resource_any);
                    Ok(Val::Resource(resource_any))
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected resource for borrow {:?}, got {:?}",
                ty, other
            ))),
        },
        Type::Own(resource_type) => match val {
            Val::Resource(resource_any) => {
                // If callee defines this resource type, convert the host-defined proxy resource
                // handle to the actual resource handle defined in the callee component.
                if callee_defined_resource_types.contains(resource_type) {
                    let host_resource: Resource<ProxyResource> =
                        Resource::try_from_resource_any(resource_any, &mut *store)?;
                    let rep = host_resource.rep();
                    let guest_resource =
                        store.data().get_dynamic_resource(rep).ok_or_else(|| {
                            wasmtime::Error::msg(format!("unknown resource rep={}", rep))
                        })?;
                    Ok(Val::Resource(guest_resource))
                } else {
                    // If the callee doesn't define this resource type, pass through the host-defined
                    // proxy resource handle.
                    Ok(Val::Resource(resource_any))
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected resource for own {:?}, got {:?}",
                ty, other
            ))),
        },
        // For composite types, recursively transform and collect any nested resource handles.
        Type::List(list_type) => match val {
            Val::List(values) => {
                let element_type = list_type.ty();
                let transformed = values
                    .into_iter()
                    .map(|v| {
                        recursive_transform_args_to_callee_view(
                            store,
                            v,
                            &element_type,
                            callee_defined_resource_types,
                            borrows_to_end,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Val::List(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected list, got {:?}",
                other
            ))),
        },
        Type::Record(record_type) => match val {
            Val::Record(fields) => {
                let field_types: Vec<_> = record_type.fields().collect();
                if field_types.len() != fields.len() {
                    return Err(wasmtime::Error::msg(format!(
                        "record field count mismatch: got {}, expected {}",
                        fields.len(),
                        field_types.len()
                    )));
                }
                let mut transformed = Vec::with_capacity(fields.len());
                for ((name, value), field) in fields.into_iter().zip(field_types.into_iter()) {
                    if name != field.name {
                        return Err(wasmtime::Error::msg(format!(
                            "record field name mismatch: got {}, expected {}",
                            name, field.name
                        )));
                    }
                    let value = recursive_transform_args_to_callee_view(
                        store,
                        value,
                        &field.ty,
                        callee_defined_resource_types,
                        borrows_to_end,
                    )?;
                    transformed.push((name, value));
                }
                Ok(Val::Record(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected record, got {:?}",
                other
            ))),
        },
        Type::Tuple(tuple_type) => match val {
            Val::Tuple(values) => {
                let types: Vec<_> = tuple_type.types().collect();
                if types.len() != values.len() {
                    return Err(wasmtime::Error::msg(format!(
                        "tuple size mismatch: got {}, expected {}",
                        values.len(),
                        types.len()
                    )));
                }
                let transformed = values
                    .into_iter()
                    .zip(types.iter())
                    .map(|(v, t)| {
                        recursive_transform_args_to_callee_view(
                            store,
                            v,
                            t,
                            callee_defined_resource_types,
                            borrows_to_end,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Val::Tuple(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected tuple, got {:?}",
                other
            ))),
        },
        Type::Variant(variant_type) => match val {
            Val::Variant(case_name, payload) => {
                let mut case_type = None;
                for case in variant_type.cases() {
                    if case.name == case_name {
                        case_type = case.ty;
                        break;
                    }
                }
                match (case_type, payload) {
                    (None, None) => Ok(Val::Variant(case_name, None)),
                    (Some(ty), Some(value)) => {
                        let inner = recursive_transform_args_to_callee_view(
                            store,
                            *value,
                            &ty,
                            callee_defined_resource_types,
                            borrows_to_end,
                        )?;
                        Ok(Val::Variant(case_name, Some(Box::new(inner))))
                    }
                    (None, Some(_)) => Err(wasmtime::Error::msg(format!(
                        "variant {} has no payload but value provided",
                        case_name
                    ))),
                    (Some(_), None) => Err(wasmtime::Error::msg(format!(
                        "variant {} expects payload but none provided",
                        case_name
                    ))),
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected variant, got {:?}",
                other
            ))),
        },
        Type::Option(option_type) => match val {
            Val::Option(Some(value)) => {
                let inner = recursive_transform_args_to_callee_view(
                    store,
                    *value,
                    &option_type.ty(),
                    callee_defined_resource_types,
                    borrows_to_end,
                )?;
                Ok(Val::Option(Some(Box::new(inner))))
            }
            Val::Option(None) => Ok(Val::Option(None)),
            other => Err(wasmtime::Error::msg(format!(
                "expected option, got {:?}",
                other
            ))),
        },
        Type::Result(result_type) => match val {
            Val::Result(Ok(value)) => match (result_type.ok(), value) {
                (Some(ty), Some(inner)) => {
                    let inner = recursive_transform_args_to_callee_view(
                        store,
                        *inner,
                        &ty,
                        callee_defined_resource_types,
                        borrows_to_end,
                    )?;
                    Ok(Val::Result(Ok(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Ok(None))),
                (None, Some(_)) => Err(wasmtime::Error::msg(
                    "result ok has no payload but value provided",
                )),
                (Some(_), None) => Err(wasmtime::Error::msg(
                    "result ok expects payload but none provided",
                )),
            },
            Val::Result(Err(value)) => match (result_type.err(), value) {
                (Some(ty), Some(inner)) => {
                    let inner = recursive_transform_args_to_callee_view(
                        store,
                        *inner,
                        &ty,
                        callee_defined_resource_types,
                        borrows_to_end,
                    )?;
                    Ok(Val::Result(Err(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Err(None))),
                (None, Some(_)) => Err(wasmtime::Error::msg(
                    "result err has no payload but value provided",
                )),
                (Some(_), None) => Err(wasmtime::Error::msg(
                    "result err expects payload but none provided",
                )),
            },
            other => Err(wasmtime::Error::msg(format!(
                "expected result, got {:?}",
                other
            ))),
        },
        // For primitive types, no transformation or borrow tracking needed
        _ => Ok(val),
    }
}

/// Transform resource handles from callee view to caller view.
/// This function recursively processes composite types to find all nested resource handles.
fn recursive_transform_returns_to_caller_view(
    store: &mut StoreContextMut<'_, InstanceState>,
    val: Val,
    ty: &Type,
    callee_defined_resource_types: &[ResourceType],
) -> Result<Val, wasmtime::Error> {
    match ty {
        Type::Own(resource_type) => match val {
            Val::Resource(resource_any) => {
                // If the returned resource is defined in the callee component, convert the
                // resource handle to a host-defined proxy resource handle.
                if callee_defined_resource_types.contains(resource_type) {
                    // Reuse existing host rep if the callee returns an already-known resource.
                    // This preserves identity and avoids double-dropping.
                    let rep =
                        if let Some(existing) = store.data().rep_for_guest_resource(resource_any) {
                            existing
                        } else {
                            let rep = store.data_mut().alloc_dynamic_rep();
                            store
                                .data_mut()
                                .insert_dynamic_resource_mapping(rep, resource_any);
                            rep
                        };
                    let host_resource = Resource::<ProxyResource>::new_own(rep);
                    let host_resource_any =
                        ResourceAny::try_from_resource(host_resource, &mut *store)?;
                    Ok(Val::Resource(host_resource_any))
                // If the callee doesn't define this resource type, this resource handle is already
                // the host-defined proxy resource handle, so we can pass it through unchanged.
                } else {
                    Ok(Val::Resource(resource_any))
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected resource for {:?}, got {:?}",
                ty, other
            ))),
        },
        Type::List(list_type) => match val {
            Val::List(values) => {
                let element_type = list_type.ty();
                let transformed = values
                    .into_iter()
                    .map(|v| {
                        recursive_transform_returns_to_caller_view(
                            store,
                            v,
                            &element_type,
                            callee_defined_resource_types,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Val::List(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected list, got {:?}",
                other
            ))),
        },
        Type::Record(record_type) => match val {
            Val::Record(fields) => {
                let field_types: Vec<_> = record_type.fields().collect();
                if field_types.len() != fields.len() {
                    return Err(wasmtime::Error::msg(format!(
                        "record field count mismatch: got {}, expected {}",
                        fields.len(),
                        field_types.len()
                    )));
                }
                let mut transformed = Vec::with_capacity(fields.len());
                for ((name, value), field) in fields.into_iter().zip(field_types.into_iter()) {
                    if name != field.name {
                        return Err(wasmtime::Error::msg(format!(
                            "record field name mismatch: got {}, expected {}",
                            name, field.name
                        )));
                    }
                    let value = recursive_transform_returns_to_caller_view(
                        store,
                        value,
                        &field.ty,
                        callee_defined_resource_types,
                    )?;
                    transformed.push((name, value));
                }
                Ok(Val::Record(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected record, got {:?}",
                other
            ))),
        },
        Type::Tuple(tuple_type) => match val {
            Val::Tuple(values) => {
                let types: Vec<_> = tuple_type.types().collect();
                if types.len() != values.len() {
                    return Err(wasmtime::Error::msg(format!(
                        "tuple size mismatch: got {}, expected {}",
                        values.len(),
                        types.len()
                    )));
                }
                let transformed = values
                    .into_iter()
                    .zip(types.iter())
                    .map(|(v, t)| {
                        recursive_transform_returns_to_caller_view(
                            store,
                            v,
                            t,
                            callee_defined_resource_types,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Val::Tuple(transformed))
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected tuple, got {:?}",
                other
            ))),
        },
        Type::Variant(variant_type) => match val {
            Val::Variant(case_name, payload) => {
                let mut case_type = None;
                for case in variant_type.cases() {
                    if case.name == case_name {
                        case_type = case.ty;
                        break;
                    }
                }
                match (case_type, payload) {
                    (None, None) => Ok(Val::Variant(case_name, None)),
                    (Some(ty), Some(value)) => {
                        let inner = recursive_transform_returns_to_caller_view(
                            store,
                            *value,
                            &ty,
                            callee_defined_resource_types,
                        )?;
                        Ok(Val::Variant(case_name, Some(Box::new(inner))))
                    }
                    (None, Some(_)) => Err(wasmtime::Error::msg(format!(
                        "variant {} has no payload but value provided",
                        case_name
                    ))),
                    (Some(_), None) => Err(wasmtime::Error::msg(format!(
                        "variant {} expects payload but none provided",
                        case_name
                    ))),
                }
            }
            other => Err(wasmtime::Error::msg(format!(
                "expected variant, got {:?}",
                other
            ))),
        },
        Type::Option(option_type) => match val {
            Val::Option(Some(value)) => {
                let inner = recursive_transform_returns_to_caller_view(
                    store,
                    *value,
                    &option_type.ty(),
                    callee_defined_resource_types,
                )?;
                Ok(Val::Option(Some(Box::new(inner))))
            }
            Val::Option(None) => Ok(Val::Option(None)),
            other => Err(wasmtime::Error::msg(format!(
                "expected option, got {:?}",
                other
            ))),
        },
        Type::Result(result_type) => match val {
            Val::Result(Ok(value)) => match (result_type.ok(), value) {
                (Some(ty), Some(inner)) => {
                    let inner = recursive_transform_returns_to_caller_view(
                        store,
                        *inner,
                        &ty,
                        callee_defined_resource_types,
                    )?;
                    Ok(Val::Result(Ok(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Ok(None))),
                (None, Some(_)) => Err(wasmtime::Error::msg(
                    "result ok has no payload but value provided",
                )),
                (Some(_), None) => Err(wasmtime::Error::msg(
                    "result ok expects payload but none provided",
                )),
            },
            Val::Result(Err(value)) => match (result_type.err(), value) {
                (Some(ty), Some(inner)) => {
                    let inner = recursive_transform_returns_to_caller_view(
                        store,
                        *inner,
                        &ty,
                        callee_defined_resource_types,
                    )?;
                    Ok(Val::Result(Err(Some(Box::new(inner)))))
                }
                (None, None) => Ok(Val::Result(Err(None))),
                (None, Some(_)) => Err(wasmtime::Error::msg(
                    "result err has no payload but value provided",
                )),
                (Some(_), None) => Err(wasmtime::Error::msg(
                    "result err expects payload but none provided",
                )),
            },
            other => Err(wasmtime::Error::msg(format!(
                "expected result, got {:?}",
                other
            ))),
        },
        // Primitive types pass through unchanged
        _ => Ok(val),
    }
}

/// Centralized call forwarding: transform arguments, call callee, transform results, end borrows.
///
/// 1. Transform arguments to callee view and collect any cross-component borrows
/// 2. Call the callee function
/// 3. End any cross-component borrows by dropping their `ResourceAny` handles
/// 4. Transform results back to caller view
async fn forward_call(
    store: &mut StoreContextMut<'_, InstanceState>,
    callee_func: &Func,
    args: &[Val],
    returns: &mut [Val],
    arg_types: &[Type],
    return_types: &[Type],
    callee_defined_resource_types: &[ResourceType],
) -> Result<(), wasmtime::Error> {
    if returns.len() != return_types.len() {
        return Err(wasmtime::Error::msg(format!(
            "result slot mismatch: got {}, expected {}",
            returns.len(),
            return_types.len()
        )));
    }

    // Transform arguments and collect any cross-component borrows that need cleanup
    let TransformedArgs {
        args: args_in_callee_view,
        borrows_to_end,
    } = transform_args_to_callee_view(store, args, arg_types, callee_defined_resource_types)?;

    let mut callee_returns = vec![Val::Bool(false); return_types.len()];

    callee_func
        .call_async(&mut *store, &args_in_callee_view, &mut callee_returns)
        .await?;
    callee_func.post_return_async(&mut *store).await?;

    // End any borrows to the caller by dropping their `ResourceAny` handles.
    // This signals to the caller that the borrows have completed.
    // This must happen after the callee function returns but before we return to the caller.
    for borrow in borrows_to_end {
        borrow
            .resource_drop_async::<InstanceState>(&mut *store)
            .await?;
    }

    let returns_in_caller_view = transform_returns_to_caller_view(
        store,
        callee_returns,
        return_types,
        callee_defined_resource_types,
    )?;

    returns
        .iter_mut()
        .zip(returns_in_caller_view)
        .for_each(|(dest, value)| *dest = value);

    Ok(())
}

/// Add stub definitions for all exports of a component to the linker.
///
/// This scans the component's exports and creates stub functions with matching
/// signatures. The stubs are never actually called. Instead, they're only used for
/// dependency validation via `instantiate_pre`.
fn register_stub_component_exports(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    component: &Component,
) -> Result<(), wasmtime::Error> {
    let component_type = linker.substituted_component_type(component)?;

    for (interface_name, export_item) in component_type.exports(engine) {
        if let ComponentItem::ComponentInstance(instance_type) = export_item {
            register_stub_interface_exports(engine, linker, &interface_name, &instance_type)?;
        }
    }

    Ok(())
}

/// Add stub definitions for an interface to the linker. The stubs are never actually called.
/// Instead, they're only used for dependency validation via `instantiate_pre`.
fn register_stub_interface_exports(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    interface_name: &str,
    instance_type: &ComponentInstanceType,
) -> Result<(), wasmtime::Error> {
    let mut root = linker.root();
    let mut inst = root.instance(interface_name).map_err(|_| {
        wasmtime::Error::msg(format!(
            "Interface '{}' is already defined in linker during dependency validation",
            interface_name
        ))
    })?;

    // Register both resources and functions as stubs.
    for (export_name, export_item) in instance_type.exports(engine) {
        match export_item {
            ComponentItem::Resource(_) => {
                inst.resource_async(
                    export_name,
                    ResourceType::host::<ProxyResource>(),
                    |_store, _rep| {
                        Box::new(async move {
                            unreachable!("Stub resource was unexpectedly called during validation");
                        })
                    },
                )?;
            }
            ComponentItem::ComponentFunc(_) => {
                inst.func_new_async(export_name, move |_store, _args, _results| {
                    Box::new(async move {
                        unreachable!("Stub function was unexpectedly called during validation");
                    })
                })?;
            }
            _ => {}
        }
    }

    Ok(())
}

/// Register forwarding implementations for a library's exports.
/// This scans the library component's exports and registers functions that forward calls
/// to the library instance.
fn register_component_exports(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    store: &mut Store<InstanceState>,
    library_component: &Component,
    library_instance: Instance,
) -> Result<(), wasmtime::Error> {
    let component_type = linker.substituted_component_type(library_component)?;

    for (interface_name, export_item) in component_type.exports(engine) {
        if let ComponentItem::ComponentInstance(instance_type) = export_item {
            register_interface_exports(
                engine,
                linker,
                store,
                &interface_name,
                &instance_type,
                library_instance,
            )?;
        }
    }

    Ok(())
}

/// Register forwarding implementations for an interface.
fn register_interface_exports(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    store: &mut Store<InstanceState>,
    interface_name: &str,
    instance_type: &ComponentInstanceType,
    library_instance: Instance,
) -> Result<(), wasmtime::Error> {
    let (_, interface_idx) = library_instance
        .get_export(&mut *store, None, interface_name)
        .ok_or_else(|| {
            wasmtime::Error::msg(format!(
                "Interface '{}' not found in library exports",
                interface_name
            ))
        })?;

    let mut root = linker.root();
    let mut inst = root.instance(interface_name).map_err(|_| {
        wasmtime::Error::msg(format!(
            "Interface '{}' is already defined in linker",
            interface_name
        ))
    })?;

    let mut resource_type_by_name: HashMap<String, ResourceType> = HashMap::new();
    let mut functions = Vec::new();

    // Loop through the interface's exports. For each exported resource, create a host-defined
    // proxy resource and register it with the linker. For each exported function, collect
    // them to register them later.
    for (export_name, export_item) in instance_type.exports(engine) {
        match export_item {
            ComponentItem::Resource(resource_type) => {
                let resource_name = export_name.to_string();
                resource_type_by_name.insert(resource_name, resource_type);

                // Register a host-defined proxy resource with the same name as the exported
                // resource. The destructor of the proxy resource will forward the call to the
                // library instance.
                inst.resource_async(
                    export_name,
                    ResourceType::host::<ProxyResource>(),
                    move |mut store, rep| {
                        Box::new(async move {
                            // Look up and remove the guest resource from the resource map
                            let guest_resource = store
                                .data_mut()
                                .remove_dynamic_resource_mapping(rep)
                                .ok_or_else(|| {
                                    wasmtime::Error::msg(format!(
                                        "Guest resource not found for rep={}",
                                        rep
                                    ))
                                })?;

                            // Call the destructor of the guest resource
                            guest_resource
                                .resource_drop_async::<InstanceState>(&mut store)
                                .await
                        })
                    },
                )?;
            }
            ComponentItem::ComponentFunc(func_type) => {
                functions.push((export_name.to_string(), func_type));
            }
            _ => {}
        }
    }

    // Identify resources that this interface defines.
    // Imported resources will not appear in constructor/method/static patterns.
    let mut defined_resource_names: HashSet<String> = HashSet::new();
    for (func_name, _func_type) in functions.iter() {
        match FuncCategory::from_name(func_name) {
            FuncCategory::Constructor { resource_name }
            | FuncCategory::Method { resource_name }
            | FuncCategory::StaticMethod { resource_name } => {
                defined_resource_names.insert(resource_name);
            }
            FuncCategory::FreeFunction => {}
        }
    }

    // Convert defined resource names to `ResourceType` handles
    let mut defined_resource_types: Vec<ResourceType> = Vec::new();
    for resource_name in &defined_resource_names {
        if let Some(resource_type) = resource_type_by_name.get(resource_name) {
            defined_resource_types.push(*resource_type);
        }
    }
    let defined_resource_types = Arc::new(defined_resource_types);

    // Register forwarding implementations for each function.
    for (func_name, func_type) in functions {
        // Look up the function.
        let (_, func_idx) = library_instance
            .get_export(&mut *store, Some(&interface_idx), &func_name)
            .ok_or_else(|| {
                wasmtime::Error::msg(format!(
                    "Function '{}' not found in interface '{}'",
                    func_name, interface_name
                ))
            })?;
        let func = library_instance
            .get_func(&mut *store, &func_idx)
            .ok_or_else(|| {
                wasmtime::Error::msg(format!(
                    "Export '{}' in interface '{}' is not a function",
                    func_name, interface_name
                ))
            })?;

        // Collect argument and return types.
        let arg_types: Arc<Vec<Type>> = Arc::new(func_type.params().map(|(_, ty)| ty).collect());
        let return_types: Arc<Vec<Type>> = Arc::new(func_type.results().collect());

        register_call_forwarding(
            &mut inst,
            &func_name,
            func,
            arg_types,
            return_types,
            defined_resource_types.clone(),
        )?;
    }

    Ok(())
}

/// Register a function that forwards calls to the library instance.
fn register_call_forwarding(
    inst: &mut LinkerInstance<'_, InstanceState>,
    func_name: &str,
    func: Func,
    arg_types: Arc<Vec<Type>>,
    return_types: Arc<Vec<Type>>,
    defined_resource_types: Arc<Vec<ResourceType>>,
) -> Result<(), wasmtime::Error> {
    inst.func_new_async(func_name, move |mut store, args, returns| {
        let arg_types = Arc::clone(&arg_types);
        let return_types = Arc::clone(&return_types);
        let defined_resource_types = Arc::clone(&defined_resource_types);

        Box::new(async move {
            forward_call(
                &mut store,
                &func,
                args,
                returns,
                &arg_types,
                &return_types,
                &defined_resource_types,
            )
            .await
        })
    })
}

/// Instantiate library components and register their exports in the linker.
///
/// This iterates through library components in dependency order, instantiates each one,
/// and registers forwarding implementations for their exports so that subsequent
/// components can import them.
pub(super) async fn instantiate_libraries(
    engine: &Engine,
    linker: &mut Linker<InstanceState>,
    store: &mut Store<InstanceState>,
    library_components: Vec<Component>,
) -> Result<(), RuntimeError> {
    for lib_component in library_components {
        let lib_instance = linker
            .instantiate_async(&mut *store, &lib_component)
            .await
            .map_err(|e| RuntimeError::Other(format!("Failed to instantiate library: {e}")))?;

        // Register forwarding implementations for this library's exports
        register_component_exports(engine, linker, store, &lib_component, lib_instance).map_err(
            |e| RuntimeError::Other(format!("Failed to register exports for library: {e}")),
        )?;
    }

    Ok(())
}
