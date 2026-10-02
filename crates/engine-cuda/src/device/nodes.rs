use crate::device::graph::Graph;
use crate::error::{Fault, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub offset: usize,
    pub size: usize,
    pub bytes: Vec<u8>,
}

impl Param {
    #[must_use]
    pub fn word(&self) -> Option<u64> {
        if self.bytes.len() > 8 {
            return None;
        }
        let mut cell = [0u8; 8];
        cell[..self.bytes.len()].copy_from_slice(&self.bytes);
        Some(u64::from_le_bytes(cell))
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub at: usize,
    pub depth: usize,
    pub kind: u32,
    pub symbol: String,
    pub func: u64,
    pub node: *mut core::ffi::c_void,
    pub entry: *mut core::ffi::c_void,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
    pub params: Vec<Param>,
    pub opaque: Option<&'static str>,
}

impl Node {
    #[must_use]
    pub fn kernel(&self) -> bool {
        self.kind == 0
    }
}

#[derive(Clone, Debug, Default)]
pub struct Walked {
    pub nodes: Vec<Node>,
    pub ambiguous: usize,
    pub edges: usize,
    pub links: Vec<(usize, usize)>,
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::needless_pass_by_value, unused_variables)]
pub fn walk(graph: &Graph) -> Result<Walked> {
    Err(Fault::Runtimeless)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_lines)]
pub fn walk(graph: &Graph) -> Result<Walked> {
    use cudarc::driver::sys as dr;

    let raw: dr::CUgraph = graph.raw().cast();

    let mut count: usize = 0;
    said("cuGraphGetNodes", unsafe {
        dr::cuGraphGetNodes(raw, core::ptr::null_mut(), &raw mut count)
    })?;
    let mut handles: Vec<dr::CUgraphNode> = vec![core::ptr::null_mut(); count];
    said("cuGraphGetNodes", unsafe {
        dr::cuGraphGetNodes(raw, handles.as_mut_ptr(), &raw mut count)
    })?;
    handles.truncate(count);

    let mut edge_count: usize = 0;
    said("cuGraphGetEdges_v2", unsafe {
        dr::cuGraphGetEdges_v2(
            raw,
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            &raw mut edge_count,
        )
    })?;
    let mut from: Vec<dr::CUgraphNode> = vec![core::ptr::null_mut(); edge_count];
    let mut to: Vec<dr::CUgraphNode> = vec![core::ptr::null_mut(); edge_count];
    said("cuGraphGetEdges_v2", unsafe {
        dr::cuGraphGetEdges_v2(
            raw,
            from.as_mut_ptr(),
            to.as_mut_ptr(),
            core::ptr::null_mut(),
            &raw mut edge_count,
        )
    })?;

    let index_of = |node: dr::CUgraphNode| handles.iter().position(|held| *held == node);
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut indegree = vec![0usize; count];
    let mut links: Vec<(usize, usize)> = Vec::with_capacity(edge_count);
    for at in 0..edge_count {
        let (Some(a), Some(b)) = (index_of(from[at]), index_of(to[at])) else {
            continue;
        };
        succ[a].push(b);
        indegree[b] += 1;
        links.push((a, b));
    }

    let mut depth = vec![0usize; count];
    let mut ready: Vec<usize> = (0..count).filter(|at| indegree[*at] == 0).collect();
    let mut left = indegree.clone();
    let mut seen = 0usize;
    while let Some(at) = ready.pop() {
        seen += 1;
        for next in succ[at].clone() {
            depth[next] = depth[next].max(depth[at] + 1);
            left[next] -= 1;
            if left[next] == 0 {
                ready.push(next);
            }
        }
    }
    if seen != count {
        return Err(Fault::Device {
            call: "cuGraphGetEdges_v2 (the captured graph is not acyclic)",
            code: seen as i32,
        });
    }

    let mut nodes = Vec::with_capacity(count);
    for (at, node) in handles.iter().enumerate() {
        let mut kind = dr::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        said("cuGraphNodeGetType", unsafe {
            dr::cuGraphNodeGetType(*node, &raw mut kind)
        })?;
        let kind = kind as u32;
        if kind != 0 {
            nodes.push(Node {
                at,
                depth: depth[at],
                kind,
                symbol: String::new(),
                func: 0,
                node: (*node).cast(),
                entry: core::ptr::null_mut(),
                grid: [0; 3],
                block: [0; 3],
                smem: 0,
                params: Vec::new(),
                opaque: Some("not a kernel node"),
            });
            continue;
        }

        let mut params: dr::CUDA_KERNEL_NODE_PARAMS = unsafe { core::mem::zeroed() };
        said("cuGraphKernelNodeGetParams_v2", unsafe {
            dr::cuGraphKernelNodeGetParams_v2(*node, &raw mut params)
        })?;

        let symbol = name_of(params.func);
        let (read, opaque) = read_params(params.func, params.kernelParams, params.extra);

        nodes.push(Node {
            at,
            depth: depth[at],
            kind,
            symbol,
            func: params.func.addr() as u64,
            node: (*node).cast(),
            entry: params.func.cast(),
            grid: [params.gridDimX, params.gridDimY, params.gridDimZ],
            block: [params.blockDimX, params.blockDimY, params.blockDimZ],
            smem: params.sharedMemBytes,
            params: read,
            opaque,
        });
    }

    nodes.sort_by(|a, b| {
        a.depth
            .cmp(&b.depth)
            .then_with(|| a.symbol.cmp(&b.symbol))
            .then_with(|| a.at.cmp(&b.at))
    });
    let mut ambiguous = 0usize;
    for pair in nodes.windows(2) {
        if pair[0].depth == pair[1].depth && pair[0].symbol == pair[1].symbol {
            ambiguous += 1;
        }
    }
    let mut place = vec![0usize; count];
    for (canonical, node) in nodes.iter().enumerate() {
        place[node.at] = canonical;
    }
    for (at, node) in nodes.iter_mut().enumerate() {
        node.at = at;
    }
    for link in &mut links {
        *link = (place[link.0], place[link.1]);
    }

    Ok(Walked {
        nodes,
        ambiguous,
        edges: edge_count,
        links,
    })
}

#[cfg(feature = "cuda")]
fn name_of(func: cudarc::driver::sys::CUfunction) -> String {
    use cudarc::driver::sys as dr;
    let mut name: *const core::ffi::c_char = core::ptr::null();
    let code = unsafe { dr::cuFuncGetName(&raw mut name, func) };
    if code != dr::CUresult::CUDA_SUCCESS || name.is_null() {
        return String::new();
    }
    unsafe { core::ffi::CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(feature = "cuda")]
fn read_params(
    func: cudarc::driver::sys::CUfunction,
    kernel_params: *mut *mut core::ffi::c_void,
    extra: *mut *mut core::ffi::c_void,
) -> (Vec<Param>, Option<&'static str>) {
    use cudarc::driver::sys as dr;

    let mut shape: Vec<(usize, usize)> = Vec::new();
    loop {
        let mut offset: usize = 0;
        let mut size: usize = 0;
        let code =
            unsafe { dr::cuFuncGetParamInfo(func, shape.len(), &raw mut offset, &raw mut size) };
        if code != dr::CUresult::CUDA_SUCCESS {
            break;
        }
        shape.push((offset, size));
        if shape.len() > 512 {
            break;
        }
    }
    if shape.is_empty() {
        return (
            Vec::new(),
            Some("cuFuncGetParamInfo names no parameters for this function"),
        );
    }

    if !kernel_params.is_null() {
        let mut read = Vec::with_capacity(shape.len());
        for (at, (offset, size)) in shape.iter().copied().enumerate() {
            let cell = unsafe { *kernel_params.add(at) };
            if cell.is_null() {
                return (read, Some("a kernelParams cell is null"));
            }
            let mut bytes = vec![0u8; size];
            unsafe {
                core::ptr::copy_nonoverlapping(cell.cast::<u8>(), bytes.as_mut_ptr(), size);
            }
            read.push(Param {
                offset,
                size,
                bytes,
            });
        }
        return (read, None);
    }

    if !extra.is_null() {
        let mut buffer: *mut u8 = core::ptr::null_mut();
        let mut len: usize = 0;
        let mut at = 0usize;
        loop {
            let entry = unsafe { *extra.add(at) };
            if entry.is_null() {
                break;
            }
            match entry.addr() {
                1 => buffer = unsafe { *extra.add(at + 1) }.cast::<u8>(),
                2 => len = unsafe { *(*extra.add(at + 1)).cast::<usize>() },
                _ => {}
            }
            at += 2;
            if at > 8 {
                break;
            }
        }
        if buffer.is_null() {
            return (Vec::new(), Some("an `extra` pack with no buffer pointer"));
        }
        let mut read = Vec::with_capacity(shape.len());
        for (offset, size) in shape.iter().copied() {
            if len != 0 && offset + size > len {
                return (read, Some("an `extra` pack shorter than the ABI block"));
            }
            let mut bytes = vec![0u8; size];
            unsafe {
                core::ptr::copy_nonoverlapping(buffer.add(offset), bytes.as_mut_ptr(), size);
            }
            read.push(Param {
                offset,
                size,
                bytes,
            });
        }
        return (read, None);
    }

    (
        Vec::new(),
        Some("the node carries neither kernelParams nor extra"),
    )
}

#[cfg(feature = "cuda")]
fn said(call: &'static str, code: cudarc::driver::sys::CUresult) -> Result<()> {
    if code == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Fault::Device {
            call,
            code: code as i32,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Rebind {
    pub identity_nodes: usize,
    pub identity_us: f64,
    pub grid: core::result::Result<(), i32>,
    pub arg: core::result::Result<(), i32>,
    pub smem: core::result::Result<(), i32>,
    pub func: core::result::Result<(), i32>,
    pub null_func: core::result::Result<(), i32>,
    pub zero_grid: core::result::Result<(), i32>,
    pub one_block: core::result::Result<(), i32>,
    pub func_from: (String, usize),
    pub func_to: (String, usize),
    pub subset_us: f64,
    pub subset_nodes: usize,
}

#[cfg(not(feature = "cuda"))]
#[allow(unused_variables)]
pub fn rebind(exec: &crate::device::GraphExec, graph: &Graph, subset: &[usize]) -> Result<Rebind> {
    Err(Fault::Runtimeless)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_lines)]
pub fn rebind(exec: &crate::device::GraphExec, graph: &Graph, subset: &[usize]) -> Result<Rebind> {
    use cudarc::driver::sys as dr;

    let raw: dr::CUgraph = graph.raw().cast();
    let hexec: dr::CUgraphExec = exec.raw().cast();

    let mut count: usize = 0;
    said("cuGraphGetNodes", unsafe {
        dr::cuGraphGetNodes(raw, core::ptr::null_mut(), &raw mut count)
    })?;
    let mut handles: Vec<dr::CUgraphNode> = vec![core::ptr::null_mut(); count];
    said("cuGraphGetNodes", unsafe {
        dr::cuGraphGetNodes(raw, handles.as_mut_ptr(), &raw mut count)
    })?;
    handles.truncate(count);

    let mut out = Rebind {
        identity_nodes: 0,
        identity_us: 0.0,
        grid: Err(-1),
        arg: Err(-1),
        smem: Err(-1),
        func: Err(-1),
        null_func: Err(-1),
        zero_grid: Err(-1),
        one_block: Err(-1),
        func_from: (String::new(), 0),
        func_to: (String::new(), 0),
        subset_us: 0.0,
        subset_nodes: 0,
    };

    let mut held: Vec<(dr::CUgraphNode, dr::CUDA_KERNEL_NODE_PARAMS)> = Vec::new();
    for node in &handles {
        let mut kind = dr::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
        said("cuGraphNodeGetType", unsafe {
            dr::cuGraphNodeGetType(*node, &raw mut kind)
        })?;
        if kind as u32 != 0 {
            continue;
        }
        let mut params: dr::CUDA_KERNEL_NODE_PARAMS = unsafe { core::mem::zeroed() };
        said("cuGraphKernelNodeGetParams_v2", unsafe {
            dr::cuGraphKernelNodeGetParams_v2(*node, &raw mut params)
        })?;
        held.push((*node, params));
    }

    let began = std::time::Instant::now();
    for (node, params) in &held {
        let code = unsafe { dr::cuGraphExecKernelNodeSetParams_v2(hexec, *node, params) };
        if code != dr::CUresult::CUDA_SUCCESS {
            out.identity_nodes = usize::MAX;
            break;
        }
        out.identity_nodes += 1;
    }
    out.identity_us = began.elapsed().as_secs_f64() * 1e6;

    let try_one =
        |mutate: &dyn Fn(&mut dr::CUDA_KERNEL_NODE_PARAMS)| -> core::result::Result<(), i32> {
            let Some((node, params)) = held.first() else {
                return Err(-1);
            };
            let mut changed = *params;
            mutate(&mut changed);
            let code =
                unsafe { dr::cuGraphExecKernelNodeSetParams_v2(hexec, *node, &raw const changed) };
            let answer = if code == dr::CUresult::CUDA_SUCCESS {
                Ok(())
            } else {
                Err(code as i32)
            };
            let _ = unsafe { dr::cuGraphExecKernelNodeSetParams_v2(hexec, *node, params) };
            answer
        };

    out.grid = try_one(&|p| p.gridDimX = p.gridDimX.max(1) + 1);
    out.smem = try_one(&|p| p.sharedMemBytes += 16);
    let stash: std::cell::RefCell<Vec<Box<[u8; 8]>>> = std::cell::RefCell::new(Vec::new());
    let cells: std::cell::RefCell<Vec<Box<[*mut core::ffi::c_void]>>> =
        std::cell::RefCell::new(Vec::new());
    out.arg = try_one(&|p| {
        if p.kernelParams.is_null() {
            return;
        }
        let mut size: usize = 0;
        let mut offset: usize = 0;
        if unsafe { dr::cuFuncGetParamInfo(p.func, 0, &raw mut offset, &raw mut size) }
            != dr::CUresult::CUDA_SUCCESS
        {
            return;
        }
        let mut cell = Box::new([0u8; 8]);
        let width = size.min(8);
        unsafe {
            core::ptr::copy_nonoverlapping(
                (*p.kernelParams).cast::<u8>(),
                cell.as_mut_ptr(),
                width,
            );
        }
        let at: *mut core::ffi::c_void = cell.as_mut_ptr().cast();
        stash.borrow_mut().push(cell);
        let mut n = 0usize;
        while unsafe { dr::cuFuncGetParamInfo(p.func, n, &raw mut offset, &raw mut size) }
            == dr::CUresult::CUDA_SUCCESS
        {
            n += 1;
            if n > 64 {
                break;
            }
        }
        let mut array: Vec<*mut core::ffi::c_void> =
            (0..n).map(|i| unsafe { *p.kernelParams.add(i) }).collect();
        if array.is_empty() {
            return;
        }
        array[0] = at;
        let mut boxed = array.into_boxed_slice();
        p.kernelParams = boxed.as_mut_ptr();
        cells.borrow_mut().push(boxed);
    });

    let mine = held.first().map(|(_, p)| arity(p.func)).unwrap_or(0);
    let other = held
        .iter()
        .find(|(_, p)| held.first().is_some_and(|(_, q)| p.func != q.func) && arity(p.func) != mine)
        .or_else(|| {
            held.iter()
                .find(|(_, p)| held.first().is_some_and(|(_, q)| p.func != q.func))
        })
        .map(|(_, p)| p.func);
    out.func = match other {
        Some(func) => {
            out.func_from = held
                .first()
                .map(|(_, p)| (name_of(p.func), arity(p.func)))
                .unwrap_or_default();
            out.func_to = (name_of(func), arity(func));
            let farm: std::cell::RefCell<Vec<Box<[u8]>>> = std::cell::RefCell::new(Vec::new());
            let plots: std::cell::RefCell<Vec<Box<[*mut core::ffi::c_void]>>> =
                std::cell::RefCell::new(Vec::new());
            try_one(&|p| {
                let mut ptrs: Vec<*mut core::ffi::c_void> = Vec::new();
                let mut at = 0usize;
                loop {
                    let (mut offset, mut size) = (0usize, 0usize);
                    if unsafe { dr::cuFuncGetParamInfo(func, at, &raw mut offset, &raw mut size) }
                        != dr::CUresult::CUDA_SUCCESS
                        || at > 64
                    {
                        break;
                    }
                    let mut cell = vec![0u8; size.max(1)].into_boxed_slice();
                    if !p.kernelParams.is_null() && at < arity(p.func) {
                        let (mut was_offset, mut was) = (0usize, 0usize);
                        if unsafe {
                            dr::cuFuncGetParamInfo(p.func, at, &raw mut was_offset, &raw mut was)
                        } == dr::CUresult::CUDA_SUCCESS
                        {
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    (*p.kernelParams.add(at)).cast::<u8>(),
                                    cell.as_mut_ptr(),
                                    size.min(was),
                                );
                            }
                        }
                    }
                    ptrs.push(cell.as_mut_ptr().cast());
                    farm.borrow_mut().push(cell);
                    at += 1;
                }
                let mut block = ptrs.into_boxed_slice();
                p.func = func;
                if !block.is_empty() {
                    p.kernelParams = block.as_mut_ptr();
                }
                plots.borrow_mut().push(block);
            })
        }
        None => Err(-1),
    };
    out.null_func = try_one(&|p| p.func = core::ptr::null_mut());
    out.zero_grid = try_one(&|p| {
        p.gridDimX = 0;
        p.gridDimY = 0;
        p.gridDimZ = 0;
    });
    out.one_block = try_one(&|p| {
        p.gridDimX = 1;
        p.gridDimY = 1;
        p.gridDimZ = 1;
    });

    let picked: Vec<usize> = subset
        .iter()
        .copied()
        .filter(|at| *at < held.len())
        .collect();
    let began = std::time::Instant::now();
    for at in &picked {
        let (node, params) = &held[*at];
        let _ = unsafe { dr::cuGraphExecKernelNodeSetParams_v2(hexec, *node, params) };
    }
    out.subset_us = began.elapsed().as_secs_f64() * 1e6;
    out.subset_nodes = picked.len();

    Ok(out)
}

#[cfg(feature = "cuda")]
fn arity(func: cudarc::driver::sys::CUfunction) -> usize {
    use cudarc::driver::sys as dr;
    let mut n = 0usize;
    loop {
        let mut offset: usize = 0;
        let mut size: usize = 0;
        if unsafe { dr::cuFuncGetParamInfo(func, n, &raw mut offset, &raw mut size) }
            != dr::CUresult::CUDA_SUCCESS
        {
            return n;
        }
        n += 1;
        if n > 512 {
            return n;
        }
    }
}

#[cfg(feature = "cuda")]
pub fn exec_footprint(graph: &Graph, copies: usize) -> Result<(f64, f64)> {
    use cudarc::runtime::sys as rt;

    let mem = || -> (usize, usize) {
        let (mut free, mut total) = (0usize, 0usize);
        let _ = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
        (free, total)
    };
    let (before, _) = mem();
    let began = std::time::Instant::now();
    let mut held = Vec::with_capacity(copies);
    for _ in 0..copies {
        held.push(graph.instantiate(core::ptr::null_mut())?);
    }
    let millis = began.elapsed().as_secs_f64() * 1000.0 / copies as f64;
    let (after, _) = mem();
    let bytes = before.saturating_sub(after) as f64 / copies as f64;
    drop(held);
    Ok((bytes, millis))
}

#[cfg(not(feature = "cuda"))]
#[allow(unused_variables)]
pub fn exec_footprint(graph: &Graph, copies: usize) -> Result<(f64, f64)> {
    Err(Fault::Runtimeless)
}

#[cfg(feature = "cuda")]
#[must_use]
pub fn free_bytes() -> Option<usize> {
    use cudarc::runtime::sys as rt;

    let (mut free, mut total) = (0usize, 0usize);
    let said = unsafe { rt::cudaMemGetInfo(&raw mut free, &raw mut total) };
    (said == rt::cudaError::cudaSuccess).then_some(free)
}

#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn free_bytes() -> Option<usize> {
    None
}
