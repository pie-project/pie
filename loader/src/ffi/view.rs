//! Reading a marshalled plan back as a safe view.
//!
//! The mirror of [`super::arena`]: that module writes a [`crate::plan::LoadPlan`]
//! into the POD form the driver holds, and this one reads that POD form back.
//!
//! Verification runs *here*, on the marshalled bytes, and nowhere else. There is
//! no path that verifies the Rust plan directly, and that is the point: a bug in
//! the marshalling is in scope for every caller, including the CLI and the
//! goldens. When the two directions were separate constructors of the same
//! `PlanView` they had already drifted — the two sides derived a plan's file
//! reads from different fields, and nothing compared the answers.

use super::arena;
use super::entry::as_str;
use super::types::*;
use crate::plan::LoadPlan;
use crate::verify::{Certificate, ContractView, Violation, verify};

/// Compile-free verification of a plan, as the driver will see it.
///
/// Marshals `plan` into the POD form, reads it back, and verifies that. The
/// arena is released before returning, so the certificate and the violations
/// own their strings.
pub fn verify_marshalled(
    plan: &LoadPlan,
    contract: Option<&ContractView<'_>>,
) -> Result<Certificate, Vec<Violation>> {
    let pod = arena::build(plan, arena::UNKEYED);
    if pod.is_null() {
        return Err(vec![Violation::plan("plan could not be marshalled")]);
    }
    let outcome = match unsafe { plan_view(&*pod) } {
        Ok(view) => verify(&view, contract).and_then(|certificate| {
            let groups = unsafe { slice_of((*pod).groups.ptr, (*pod).groups.len) };
            let mut found = Vec::new();
            for group in groups {
                unsafe { verify_group(group, &mut found) };
            }
            if found.is_empty() {
                Ok(certificate)
            } else {
                Err(found)
            }
        }),
        Err(err) => Err(vec![Violation::plan(err)]),
    };
    unsafe { arena::release(pod) };
    outcome
}

/// Verify a group: its plan, and then every instance's reads.
///
/// The instances are the point. A group's plan is compiled at index 0, so
/// verifying it alone would check one instance out of `arity` and leave the
/// other bindings -- which are what the driver will actually read -- unchecked.
/// Rewriting the template's reads with each instance's binding and running the
/// same file-bounds check over the result is the cheapest way to say "every
/// instance stays inside its file", and it reuses the check rather than
/// restating it.
///
/// # Safety
///
/// `group`'s plan pointer and slices are those of a plan produced by
/// [`arena::build`].
unsafe fn verify_group(group: &PieLoaderGroupView, found: &mut Vec<Violation>) {
    let name = unsafe { as_str(&group.name, "group.name") }.unwrap_or("<unnamed>");
    let Ok(mut template) = (unsafe { plan_view(&*group.plan) }) else {
        found.push(Violation::plan(format!(
            "group '{name}': plan is malformed"
        )));
        return;
    };
    if let Err(mut violations) = verify(&template, None) {
        for violation in &mut violations {
            violation.message = format!("group '{name}': {}", violation.message);
        }
        found.append(&mut violations);
        return;
    }

    let per = group.bindings_per_instance;
    let bindings = unsafe { slice_of(group.bindings.ptr, group.bindings.len) };
    if per != template.reads.len() || bindings.len() != per * group.arity as usize {
        found.push(Violation::plan(format!(
            "group '{name}': {} bindings for {} instances of a plan with {} \
             reads",
            bindings.len(),
            group.arity,
            template.reads.len()
        )));
        return;
    }

    // Mutated in place, one instance at a time: every read's file and offset is
    // overwritten on each pass, so there is nothing to restore between them.
    for index in 0..group.arity as usize {
        for (read, binding) in template
            .reads
            .iter_mut()
            .zip(&bindings[index * per..(index + 1) * per])
        {
            if read.instr != binding.instr_id {
                found.push(Violation::plan(format!(
                    "group '{name}' index {index}: binding names instruction {} \
                     where the plan reads at instruction {}",
                    binding.instr_id, read.instr
                )));
                return;
            }
            read.file_id = binding.file_id;
            read.file_offset = binding.file_offset;
        }
        if let Err(violations) = verify(&template, None) {
            for violation in violations {
                found.push(Violation::plan(format!(
                    "group '{name}' index {index}: {}",
                    violation.message
                )));
            }
            return;
        }
    }
}

/// Borrow a POD slice, treating a null pointer as empty.
///
/// # Safety
///
/// `ptr` is null, or valid for `len` elements for the lifetime `'a`.
pub(super) unsafe fn slice_of<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

/// Reconstruct the safe view [`crate::verify`] works on from the POD plan.
///
/// This is the only place that dereferences the plan's slice pointers, so it is
/// where a malformed handle is caught rather than propagated. Names are decoded
/// here too: a non-UTF-8 name is reported once, as itself, instead of silently
/// failing to match later.
///
/// # Safety
///
/// `plan`'s slices are null or valid for their stated lengths, which is true of
/// any plan produced by [`pie_loader_compile`].
pub(super) unsafe fn plan_view(
    plan: &PieLoaderPlan,
) -> Result<crate::verify::PlanView<'_>, String> {
    let instrs = unsafe { slice_of(plan.instrs.ptr, plan.instrs.len) };
    let mut files = Vec::with_capacity(plan.files.len);
    for file in unsafe { slice_of(plan.files.ptr, plan.files.len) } {
        files.push(crate::verify::FileView {
            id: file.id,
            path: unsafe { as_str(&file.path, "file.path") }
                .map_err(|err| format!("file {}: {err}", file.id))?,
            size_bytes: file.size_bytes,
        });
    }
    let mut sources = Vec::with_capacity(plan.sources.len);
    for source in unsafe { slice_of(plan.sources.ptr, plan.sources.len) } {
        sources.push(crate::verify::SourceView {
            name: unsafe { as_str(&source.name, "source.name") }
                .map_err(|err| format!("source tensor {}: {err}", source.id))?,
            file_id: source.file_id,
            offset_bytes: source.file_offset,
            span_bytes: source.span_bytes,
        });
    }
    let mut tensors = Vec::with_capacity(plan.tensors.len);
    for tensor in unsafe { slice_of(plan.tensors.ptr, plan.tensors.len) } {
        tensors.push(crate::verify::TensorView::new(
            unsafe { as_str(&tensor.name, "tensor.name") }
                .map_err(|err| format!("tensor {}: {err}", tensor.id))?,
            unsafe { slice_of(tensor.shape.ptr, tensor.shape.len) },
            &join_encoding(tensor),
            tensor.visibility.into(),
        ));
    }

    let mut finalized = Vec::new();
    let mut reads = Vec::new();
    // Which instructions read a file used to be a question about resting
    // values: `source` was present on all of them and meaningful on three. The
    // operation now says so itself.
    for instr in instrs {
        match &instr.op {
            PieLoaderStorageOp::Finalize { name, .. } => finalized.push(
                unsafe { as_str(name, "instr.name") }
                    .map_err(|err| format!("instruction {}: {err}", instr.id))?,
            ),
            PieLoaderStorageOp::ExtentWrite { source, .. }
            | PieLoaderStorageOp::BulkExtentWrite { source, .. } => {
                reads.push(crate::verify::ReadView {
                    instr: instr.id,
                    file_id: source.file_id,
                    file_offset: source.file_offset,
                    span_bytes: source.span_bytes,
                });
            }
            PieLoaderStorageOp::TileMap {
                source, has_source, ..
            } => {
                if *has_source {
                    reads.push(crate::verify::ReadView {
                        instr: instr.id,
                        file_id: source.file_id,
                        file_offset: source.file_offset,
                        span_bytes: source.span_bytes,
                    });
                }
            }
            // Named rather than left to a wildcard: `verify` decides a file is
            // unread by finding no `ReadView` for it, so an operation that
            // grows a source and is not added above would make the plan look
            // like it never touched the checkpoint. That is not hypothetical —
            // `TileMap` used to report file 0 for every device-side transform,
            // because a flat struct left `source` at its zero default.
            PieLoaderStorageOp::Allocate { .. }
            | PieLoaderStorageOp::CreateView { .. }
            | PieLoaderStorageOp::Fill { .. } => {}
        }
    }

    Ok(crate::verify::PlanView {
        compiler_version: plan.compiler_version,
        files,
        sources,
        tensors,
        instr_count: instrs.len(),
        schedule: unsafe { slice_of(plan.schedule.ptr, plan.schedule.len) }.to_vec(),
        finalized,
        reads,
    })
}

/// The inverse of `arena::split_encoding`: rebuild an [`Encoding`] from the flat
/// fields a POD view carries, so a marshalled tensor can be compared against
/// one the loader still holds in typed form.
fn join_encoding(tensor: &PieLoaderTensorDeclView) -> crate::types::Encoding {
    match tensor.encoding_kind {
        PieLoaderEncodingKind::Quant => crate::types::Encoding::Quant(crate::types::QuantSpec {
            scheme: tensor.quant_scheme.into(),
            logical_dtype: tensor.dtype.into(),
            bits_per_element: tensor.quant_bits_per_element,
            group_size: tensor.quant_group_size,
            channel_axis: None,
        }),
        _ => crate::types::Encoding::Raw(tensor.dtype.into()),
    }
}
