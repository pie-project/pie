//! Dynamic linking for Wasm components: registers host-side proxy functions
//! and resources so one component can import another's exports, translating
//! resource handles and tracking cross-component borrows across the call.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use smallvec::SmallVec;

use wasmtime::component::types::{ComponentInstance as ComponentInstanceType, ComponentItem, Type};
use wasmtime::component::{
    Component, Func, Instance, Linker, LinkerInstance, Resource, ResourceAny, ResourceType, Val,
};
use wasmtime::{Engine, Store, StoreContextMut};

use crate::inferlet::ProcessCtx;

/// Phantom marker type for host-defined resource handles used in dynamic
/// linking: when a component exports a resource type, `ProxyResource`
/// instances manage it from the host side for cross-component passing.
struct ProxyResource;

/// Precomputed metadata for a forwarded function, consolidating all per-call data
/// into a single Arc to minimize atomic reference count operations on the hot path.
struct FuncForwardingInfo {
    arg_types: Vec<Type>,
    return_types: Vec<Type>,
    defined_resource_types: Arc<Vec<ResourceType>>,
}

/// Check if a type (recursively) contains any resource types (Own or Borrow).
/// Used at registration time to precompute whether transformation is needed.
fn type_contains_resource(ty: &Type) -> bool {
    match ty {
        Type::Own(_) | Type::Borrow(_) => true,
        Type::List(lt) => type_contains_resource(&lt.ty()),
        Type::Record(rt) => rt.fields().any(|f| type_contains_resource(&f.ty)),
        Type::Tuple(tt) => tt.types().any(|t| type_contains_resource(&t)),
        Type::Variant(vt) => vt
            .cases()
            .any(|c| c.ty.as_ref().is_some_and(type_contains_resource)),
        Type::Option(ot) => type_contains_resource(&ot.ty()),
        Type::Result(rt) => {
            rt.ok().is_some_and(|t| type_contains_resource(&t))
                || rt.err().is_some_and(|t| type_contains_resource(&t))
        }
        _ => false,
    }
}

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

/// When a call forwards from component A to component B, A's resource
/// handles are host-defined proxy handles and must be transformed into the
/// actual handles B defines. Cross-component borrows made during that
/// transform are not auto-ended, so they are tracked here to be dropped
/// after the call completes.
struct TransformedArgs {
    /// The transformed argument values
    args: SmallVec<[Val; 8]>,
    /// Borrowed `ResourceAny` handles to drop after the call, ending the
    /// cross-component borrow.
    borrows_to_end: SmallVec<[ResourceAny; 8]>,
}

/// Transform arguments from caller view to callee view.
/// Only resources defined in the callee component are transformed from the host-defined
/// proxy resource handle to the actual resource handle defined in the callee component.
/// Cross-component borrows are tracked in borrows_to_end for cleanup after the call.
fn transform_args_to_callee_view(
    store: &mut StoreContextMut<'_, ProcessCtx>,
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

    let mut borrows_to_end = SmallVec::new();
    let mut transformed_args = SmallVec::with_capacity(args.len());

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
    store: &mut StoreContextMut<'_, ProcessCtx>,
    returns: SmallVec<[Val; 8]>,
    return_type: &[Type],
    callee_defined_resource_types: &[ResourceType],
) -> Result<SmallVec<[Val; 8]>, wasmtime::Error> {
    if returns.len() != return_type.len() {
        return Err(wasmtime::Error::msg(format!(
            "result count mismatch: got {}, expected {}",
            returns.len(),
            return_type.len()
        )));
    }

    let mut transformed_returns = SmallVec::with_capacity(returns.len());
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
    store: &mut StoreContextMut<'_, ProcessCtx>,
    val: Val,
    ty: &Type,
    callee_defined_resource_types: &[ResourceType],
    borrows_to_end: &mut SmallVec<[ResourceAny; 8]>,
) -> Result<Val, wasmtime::Error> {
    match ty {
        Type::Borrow(resource_type) => match val {
            Val::Resource(resource_any) => {
                // Convert the proxy handle to the callee's own handle if it
                // defines this resource type (cleanup happens inside
                // `try_from_resource_any`, no explicit tracking needed here).
                if callee_defined_resource_types.contains(resource_type) {
                    let host_resource: Resource<ProxyResource> =
                        Resource::try_from_resource_any(resource_any, &mut *store)?;
                    let rep = host_resource.rep();
                    let guest_resource =
                        store.data().get_dynamic_resource(rep).ok_or_else(|| {
                            wasmtime::Error::msg(format!("unknown resource rep={}", rep))
                        })?;
                    Ok(Val::Resource(guest_resource))
                // Otherwise pass the proxy handle through, tracked as a
                // cross-component borrow for cleanup after the call.
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
                // Convert to the callee's own handle if it defines this
                // resource type; otherwise pass the proxy handle through.
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
                for ((name, value), field) in fields.into_iter().zip(field_types) {
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
    store: &mut StoreContextMut<'_, ProcessCtx>,
    val: Val,
    ty: &Type,
    callee_defined_resource_types: &[ResourceType],
) -> Result<Val, wasmtime::Error> {
    match ty {
        Type::Own(resource_type) => match val {
            Val::Resource(resource_any) => {
                // Convert to a host-defined proxy handle if the callee
                // defines this resource type.
                if callee_defined_resource_types.contains(resource_type) {
                    // Reuse an existing host rep for an already-known resource,
                    // to preserve identity and avoid double-dropping.
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
                // Otherwise this is already the proxy handle; pass it through.
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
                for ((name, value), field) in fields.into_iter().zip(field_types) {
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

/// Transform arguments to callee view, call the callee, end any
/// cross-component borrows, then transform results back to caller view.
async fn forward_call(
    store: &mut StoreContextMut<'_, ProcessCtx>,
    callee_func: &Func,
    args: &[Val],
    returns: &mut [Val],
    info: &FuncForwardingInfo,
) -> Result<(), wasmtime::Error> {
    if returns.len() != info.return_types.len() {
        return Err(wasmtime::Error::msg(format!(
            "result slot mismatch: got {}, expected {}",
            returns.len(),
            info.return_types.len()
        )));
    }

    let TransformedArgs {
        args: args_in_callee_view,
        borrows_to_end,
    } = transform_args_to_callee_view(store, args, &info.arg_types, &info.defined_resource_types)?;

    let mut callee_returns: SmallVec<[Val; 8]> =
        smallvec::smallvec![Val::Bool(false); info.return_types.len()];

    callee_func
        .call_async(&mut *store, &args_in_callee_view, &mut callee_returns)
        .await?;

    for borrow in borrows_to_end {
        borrow.resource_drop_async(&mut *store).await?;
    }

    let returns_in_caller_view = transform_returns_to_caller_view(
        store,
        callee_returns,
        &info.return_types,
        &info.defined_resource_types,
    )?;

    returns
        .iter_mut()
        .zip(returns_in_caller_view)
        .for_each(|(dest, value)| *dest = value);

    Ok(())
}

/// Scans a library component's exports and registers functions that forward
/// calls to the library instance.
fn register_component_exports(
    engine: &Engine,
    linker: &mut Linker<ProcessCtx>,
    store: &mut Store<ProcessCtx>,
    library_component: &Component,
    library_instance: Instance,
) -> Result<(), wasmtime::Error> {
    let component_type = linker.substituted_component_type(library_component)?;

    // First pass: collect all defined resource types across all interfaces.
    // A resource is "defined" where it has constructors, methods, or static
    // methods; one that only appears via `use other-interface.{type}` is a
    // re-exported alias and must reuse that interface's proxy.
    let mut component_defined_resource_types: Vec<ResourceType> = Vec::new();

    for (_, export_item) in component_type.exports(engine) {
        if let ComponentItem::ComponentInstance(instance_type) = export_item.ty {
            let mut resource_types_by_name: HashMap<String, ResourceType> = HashMap::new();
            let mut defined_names: HashSet<String> = HashSet::new();

            for (name, item) in instance_type.exports(engine) {
                match item.ty {
                    ComponentItem::Resource(rt) => {
                        resource_types_by_name.insert(name.to_string(), rt);
                    }
                    ComponentItem::ComponentFunc(_) => match FuncCategory::from_name(name) {
                        FuncCategory::Constructor { resource_name }
                        | FuncCategory::Method { resource_name }
                        | FuncCategory::StaticMethod { resource_name } => {
                            defined_names.insert(resource_name);
                        }
                        FuncCategory::FreeFunction => {}
                    },
                    _ => {}
                }
            }

            for name in &defined_names {
                if let Some(rt) = resource_types_by_name.get(name)
                    && !component_defined_resource_types.contains(rt)
                {
                    component_defined_resource_types.push(*rt);
                }
            }
        }
    }

    let component_defined_resource_types = Arc::new(component_defined_resource_types);

    // Second pass: register exports per interface, passing the component-wide set.
    for (interface_name, export_item) in component_type.exports(engine) {
        if let ComponentItem::ComponentInstance(instance_type) = export_item.ty {
            register_interface_exports(
                engine,
                linker,
                store,
                interface_name,
                &instance_type,
                library_instance,
                &component_defined_resource_types,
            )?;
        }
    }

    Ok(())
}

/// Register forwarding implementations for an interface.
fn register_interface_exports(
    engine: &Engine,
    linker: &mut Linker<ProcessCtx>,
    store: &mut Store<ProcessCtx>,
    interface_name: &str,
    instance_type: &ComponentInstanceType,
    library_instance: Instance,
    component_defined_resource_types: &Arc<Vec<ResourceType>>,
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

    // Collect all resources and functions before registering anything.
    let mut resource_type_by_name: HashMap<String, ResourceType> = HashMap::new();
    let mut functions = Vec::new();

    for (export_name, export_item) in instance_type.exports(engine) {
        match export_item.ty {
            ComponentItem::Resource(resource_type) => {
                resource_type_by_name.insert(export_name.to_string(), resource_type);
            }
            ComponentItem::ComponentFunc(func_type) => {
                functions.push((export_name.to_string(), func_type));
            }
            _ => {}
        }
    }

    // Resources this interface defines (imported ones via `use` have no
    // constructor/method/static pattern here).
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

    // Only resources defined in this interface get a new proxy; a
    // re-exported alias reuses its defining interface's proxy (a second one
    // would be an incompatible resource type).
    for resource_name in resource_type_by_name.keys() {
        if !defined_resource_names.contains(resource_name) {
            continue;
        }

        inst.resource_async(
            resource_name,
            ResourceType::host::<ProxyResource>(),
            move |mut store, rep| {
                Box::new(async move {
                    let guest_resource = store
                        .data_mut()
                        .remove_dynamic_resource_mapping(rep)
                        .ok_or_else(|| {
                            wasmtime::Error::msg(format!(
                                "Guest resource not found for rep={}",
                                rep
                            ))
                        })?;

                    guest_resource.resource_drop_async(&mut store).await
                })
            },
        )?;
    }

    // Use the component-wide defined resource types so resources defined
    // in any interface of this component translate correctly.
    for (func_name, func_type) in functions {
        let (_, func_idx) = library_instance
            .get_export(&mut *store, Some(&interface_idx), &func_name)
            .ok_or_else(|| {
                wasmtime::Error::msg(format!(
                    "Function '{}' not found in interface '{}'",
                    func_name, interface_name
                ))
            })?;
        let func = library_instance
            .get_func(&mut *store, func_idx)
            .ok_or_else(|| {
                wasmtime::Error::msg(format!(
                    "Export '{}' in interface '{}' is not a function",
                    func_name, interface_name
                ))
            })?;

        let arg_types: Vec<Type> = func_type.params().map(|(_, ty)| ty).collect();
        let return_types: Vec<Type> = func_type.results().collect();

        register_call_forwarding(
            &mut inst,
            &func_name,
            func,
            arg_types,
            return_types,
            component_defined_resource_types.clone(),
        )?;
    }

    Ok(())
}

/// Register a function that forwards calls to the library instance. The
/// common resource-free case gets a minimal closure capturing only the
/// callee `Func`; a signature with resource types gets the full forwarding
/// closure with argument/return transformation.
fn register_call_forwarding(
    inst: &mut LinkerInstance<'_, ProcessCtx>,
    func_name: &str,
    func: Func,
    arg_types: Vec<Type>,
    return_types: Vec<Type>,
    defined_resource_types: Arc<Vec<ResourceType>>,
) -> Result<(), wasmtime::Error> {
    let has_resource_args = arg_types.iter().any(type_contains_resource);
    let has_resource_returns = return_types.iter().any(type_contains_resource);

    if !has_resource_args && !has_resource_returns {
        // Fast path: no resource types, forward directly with no transform.
        inst.func_new_async(func_name, move |mut store, _ty, args, returns| {
            Box::new(async move {
                func.call_async(&mut store, args, returns).await?;
                Ok(())
            })
        })
    } else {
        // Slow path: resource types need transforming both ways, via the
        // shared `FuncForwardingInfo`.
        let info = Arc::new(FuncForwardingInfo {
            arg_types,
            return_types,
            defined_resource_types,
        });

        inst.func_new_async(func_name, move |mut store, _ty, args, returns| {
            let info = Arc::clone(&info);

            Box::new(async move { forward_call(&mut store, &func, args, returns, &info).await })
        })
    }
}

/// Instantiate library components in dependency order and register their
/// exports so subsequent components can import them.
pub(crate) async fn instantiate_libraries(
    engine: &Engine,
    linker: &mut Linker<ProcessCtx>,
    store: &mut Store<ProcessCtx>,
    library_components: Vec<Component>,
) -> anyhow::Result<()> {
    for lib_component in library_components {
        let lib_instance = linker
            .instantiate_async(&mut *store, &lib_component)
            .await?;

        register_component_exports(engine, linker, store, &lib_component, lib_instance)?;
    }

    Ok(())
}
