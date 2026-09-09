use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Bound, RangeBounds};
use std::rc::Rc;

use eta_dsl::builder::Builder;
use eta_dsl::channel::PutValue;
use eta_dsl::value::ConstData;
use eta_dsl::{Channel as DslChannel, IntoConst, IntoPut, IntoShape, Port, Shape, Stage, Tensor};

use crate::pie::inferlet::channel as wit_channel;
use crate::pie::inferlet::forward as wit_attention;
use crate::pie::inferlet::forward_diffusion as wit_diffusion;
use crate::pie::inferlet::forward_hybrid as wit_hybrid;
use crate::pie::inferlet::forward_recurrent as wit_recurrent;
pub use crate::pie::inferlet::model::LaneStream;
use crate::pie::inferlet::pipeline as wit_pipeline;
use crate::pie::inferlet::types::Dtype as WitDtype;
use crate::working_set::{KvWorkingSet, PageRange, PageSpan};

pub use eta_dsl::intrinsics;

pub use eta_dsl::Dtype;
pub use eta_dsl::{
    abs, add, and, broadcast, cast, causal_mask, cos, cummass_le, cumprod, cumsum, div, dtype,
    entropy, entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_max,
    indptr, iota, l2norm, le, log, log_softmax, lt, mask_apply, masked_argmax, matmul, max_elem,
    min_elem, mul, ne, neg, normal, not, nucleus_sample, or, pivot_threshold, prob_ge, rank_le,
    recip, reduce_argmax, reduce_max, reduce_min, reduce_sum, rem, reshape, rng, row_membership,
    rsqrt, scalar_gather, scatter_add, scatter_set, select, sign, sin, sink_window_mask,
    sliding_window_mask, softmax, sort_desc, sqrt, sub, top_k, transpose,
};

thread_local! {
    static WIT_CHANNELS: RefCell<HashMap<u64, Rc<wit_channel::Channel>>> = RefCell::new(HashMap::new());
    static DECLARED: RefCell<HashMap<u64, ChannelSpec>> = RefCell::new(HashMap::new());
}

#[derive(Clone)]
struct ChannelSpec {
    dims: Vec<u32>,
    dtype: WitDtype,
    capacity: u32,
}

fn declare_channel(gid: u64, spec: ChannelSpec) {
    DECLARED.with(|m| {
        m.borrow_mut().insert(gid, spec);
    });
}

fn channel_exists(gid: u64) -> bool {
    WIT_CHANNELS.with(|m| m.borrow().contains_key(&gid))
}

fn set_declared_capacity(gid: u64, capacity: u32) {
    DECLARED.with(|m| {
        if let Some(spec) = m.borrow_mut().get_mut(&gid) {
            spec.capacity = capacity;
        }
    });
}

fn lookup_channel(gid: u64) -> Option<Rc<wit_channel::Channel>> {
    if let Some(wit) = WIT_CHANNELS.with(|m| m.borrow().get(&gid).cloned()) {
        return Some(wit);
    }
    let spec = DECLARED.with(|m| m.borrow().get(&gid).cloned())?;
    let wit = Rc::new(wit_channel::Channel::new(
        &spec.dims,
        spec.dtype,
        spec.capacity,
    ));
    WIT_CHANNELS.with(|m| {
        m.borrow_mut().insert(gid, Rc::clone(&wit));
    });
    Some(wit)
}

fn to_wit_dtype(d: Dtype) -> WitDtype {
    match d {
        Dtype::F32 => WitDtype::F32,
        Dtype::I32 => WitDtype::I32,
        Dtype::U32 => WitDtype::U32,
        Dtype::Bool => WitDtype::Bool,
        other => panic!("{other:?} is not a dtype ETA computes in; it has no WIT tag"),
    }
}

fn dims_of(shape: Shape) -> Vec<u32> {
    shape.dims().to_vec()
}

fn claim_port(port: Port, ch: &Channel) -> DslChannel {
    let dsl = ch.dsl();
    dsl.note_desc_claim(port.consumes());
    dsl
}

#[derive(Clone, Copy)]
pub struct Channel {
    gid: u64,
    shape: Shape,
    dtype: Dtype,
}

pub const TOKEN_PAD: i32 = -1;

impl Channel {
    pub fn new(shape: impl IntoShape, dtype: Dtype) -> Channel {
        Channel::build(shape.into_shape(), dtype, false)
    }

    pub fn writer(shape: impl IntoShape, dtype: Dtype) -> Channel {
        let channel = Channel::build(shape.into_shape(), dtype, false);
        channel.dsl().note_host_put();
        channel
    }

    fn dsl(&self) -> DslChannel {
        DslChannel::by_gid(self.gid).expect("channel token resolves in the DSL registry")
    }

    fn wit(&self) -> Rc<wit_channel::Channel> {
        lookup_channel(self.gid).expect("channel token resolves in the WIT registry")
    }

    pub fn capacity(self, n: u32) -> Channel {
        assert!(
            !channel_exists(self.gid),
            "capacity must be set before the channel is used"
        );
        self.dsl().capacity(n);
        set_declared_capacity(self.gid, n);
        self
    }

    pub fn named(self, name: &str) -> Channel {
        let _ = self.dsl().named(name);
        self
    }

    pub fn from(v: impl IntoConst) -> Channel {
        let data: ConstData = v.into_const();
        let ch = Channel::build(data.shape, data.dtype, true);
        ch.wit()
            .put(&data.bytes)
            .expect("stage seed on a fresh channel");
        ch
    }

    pub fn seeded(shape: impl IntoShape, dtype: Dtype) -> Channel {
        Channel::build(shape.into_shape(), dtype, true)
    }

    pub fn from_shaped(shape: impl IntoShape, v: impl IntoConst) -> Channel {
        let data: ConstData = v.into_const();
        let shape = shape.into_shape();
        assert_eq!(
            shape.numel(),
            data.shape.numel(),
            "from_shaped: element count mismatch"
        );
        let ch = Channel::build(shape, data.dtype, true);
        ch.wit()
            .put(&data.bytes)
            .expect("stage seed on a fresh channel");
        ch
    }

    fn build(shape: Shape, dtype: Dtype, seeded: bool) -> Channel {
        let dsl = if seeded {
            DslChannel::seeded(shape, dtype)
        } else {
            DslChannel::new(shape, dtype)
        };
        let gid = dsl.gid();
        declare_channel(
            gid,
            ChannelSpec {
                dims: dims_of(shape),
                dtype: to_wit_dtype(dtype),
                capacity: 1,
            },
        );
        Channel { gid, shape, dtype }
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }

    pub fn take(&self) -> Tensor {
        self.dsl().take()
    }

    pub fn read(&self) -> Tensor {
        self.dsl().read()
    }

    pub async fn take_host<T: FromChannel>(&self) -> Result<T, String> {
        self.check_host::<T>("take")?;
        self.dsl().note_host_take();
        let raw = self.wit().take().await;
        self.decode_host::<T>(raw, "take")
    }

    pub fn take_frames(
        &self,
        width: u32,
        height: u32,
        count: u32,
        fps: f32,
    ) -> Result<crate::pie::inferlet::frames::Frames, String> {
        self.dsl().note_host_take();
        crate::pie::inferlet::frames::Frames::from_channel(&self.wit(), width, height, count, fps)
            .map_err(|why| format!("{}: {why}", self.host_label("take-frames")))
    }

    pub fn set_frames(&self, pixels: &crate::pie::inferlet::frames::Frames) -> Result<(), String> {
        pixels
            .to_channel(&self.wit())
            .map_err(|why| format!("{}: {why}", self.host_label("set-frames")))
    }

    pub async fn read_host<T: FromChannel>(&self) -> Result<T, String> {
        self.check_host::<T>("read")?;
        self.dsl().note_host_read();
        let raw = self.wit().read().await;
        self.decode_host::<T>(raw, "read")
    }

    fn host_label(&self, verb: &str) -> String {
        format!("{} {verb}", self.dsl().name())
    }

    fn check_host<T: FromChannel>(&self, verb: &str) -> Result<(), String> {
        if T::DTYPE != self.dtype {
            return Err(format!(
                "{}: channel holds {:?}, decoded as {:?}",
                self.host_label(verb),
                self.dtype,
                T::DTYPE
            ));
        }
        Ok(())
    }

    fn decode_host<T: FromChannel>(
        &self,
        raw: Result<Vec<u8>, String>,
        verb: &str,
    ) -> Result<T, String> {
        let label = self.host_label(verb);
        let raw = raw.map_err(|e| format!("{label}: {e}"))?;
        T::from_bytes(&raw).map_err(|e| format!("{label}: {e}"))
    }

    pub fn put(&self, v: impl IntoPut) {
        match v.into_put() {
            PutValue::Tensor(t) => {
                self.dsl().put(t);
            }
            PutValue::Data(data) => {
                self.dsl().note_host_put();
                let _ = self.wit().put(&data.bytes);
            }
        }
    }

    pub fn set(&self, v: impl IntoConst) -> Result<(), String> {
        let data: ConstData = v.into_const();
        self.wit().set(&data.bytes)
    }
}

macro_rules! channel_from_iter {
    ($t:ty) => {
        impl FromIterator<$t> for Channel {
            fn from_iter<I: IntoIterator<Item = $t>>(iter: I) -> Channel {
                Channel::from(iter.into_iter().collect::<Vec<$t>>())
            }
        }
    };
}
channel_from_iter!(u32);
channel_from_iter!(i32);
channel_from_iter!(f32);
channel_from_iter!(bool);

pub trait HostElem: Copy {
    const DTYPE: Dtype;
    fn decode(raw: &[u8]) -> Vec<Self>;
}
impl HostElem for i32 {
    const DTYPE: Dtype = Dtype::I32;
    fn decode(raw: &[u8]) -> Vec<i32> {
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for u32 {
    const DTYPE: Dtype = Dtype::U32;
    fn decode(raw: &[u8]) -> Vec<u32> {
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for f32 {
    const DTYPE: Dtype = Dtype::F32;
    fn decode(raw: &[u8]) -> Vec<f32> {
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
impl HostElem for bool {
    const DTYPE: Dtype = Dtype::Bool;
    fn decode(raw: &[u8]) -> Vec<bool> {
        raw.iter().map(|&byte| byte != 0).collect()
    }
}

pub trait FromChannel: Sized {
    const DTYPE: Dtype;
    fn from_bytes(raw: &[u8]) -> Result<Self, String>;
}

macro_rules! from_channel {
    ($t:ty) => {
        impl FromChannel for Vec<$t> {
            const DTYPE: Dtype = <$t as HostElem>::DTYPE;
            fn from_bytes(raw: &[u8]) -> Result<Self, String> {
                Ok(<$t as HostElem>::decode(raw))
            }
        }
        impl FromChannel for $t {
            const DTYPE: Dtype = <$t as HostElem>::DTYPE;
            fn from_bytes(raw: &[u8]) -> Result<Self, String> {
                <$t as HostElem>::decode(raw)
                    .into_iter()
                    .next()
                    .ok_or_else(|| format!("channel is empty, expected one {}", stringify!($t)))
            }
        }
    };
}
from_channel!(i32);
from_channel!(u32);
from_channel!(f32);
from_channel!(bool);

pub struct WorkingSet {
    kv: Rc<KvWorkingSet>,
}

impl WorkingSet {
    pub fn new() -> WorkingSet {
        WorkingSet {
            kv: Rc::new(KvWorkingSet::new()),
        }
    }

    pub fn page_len(&self) -> u32 {
        self.kv.page_len()
    }

    pub fn reserve(&self, pages: u32) -> Result<PageGrant, String> {
        let range = self.kv.reserve(pages)?;
        Ok(PageGrant {
            start: range.start,
            ids: (range.start..range.start + range.len).collect(),
        })
    }

    pub fn update_index(&self, key: &[u8]) -> Result<(), String> {
        self.kv.update_index(key)
    }

    pub fn from_index(key: &[u8]) -> Result<Option<WorkingSet>, String> {
        Ok(KvWorkingSet::from_index(key)?.map(|kv| WorkingSet { kv: Rc::new(kv) }))
    }

    pub fn remove_index(key: &[u8]) -> Result<bool, String> {
        KvWorkingSet::remove_index(key)
    }

    pub fn discard(&self, on: &Pipeline, ranges: &[PageRange]) -> Result<(), String> {
        self.kv.discard(&on.wit, ranges)
    }

    pub fn fork(&self, on: &Pipeline) -> Result<WorkingSet, String> {
        Ok(WorkingSet {
            kv: Rc::new(self.kv.fork(&on.wit)?),
        })
    }

    pub fn slice(&self, on: &Pipeline, start: u32, len: u32) -> Result<WorkingSet, String> {
        let child = self.kv.slice(&on.wit, PageRange { start, len })?;
        Ok(WorkingSet { kv: Rc::new(child) })
    }

    pub fn copy_into(
        &self,
        on: &Pipeline,
        dst_page_ids: &[u32],
        dst_tok_idx: &[u32],
        src_page_ids: &[u32],
        src_tok_idx: &[u32],
    ) -> Result<(), String> {
        self.kv.copy_into(
            &on.wit,
            dst_page_ids,
            dst_tok_idx,
            src_page_ids,
            src_tok_idx,
        )
    }
}

impl Default for WorkingSet {
    fn default() -> Self {
        WorkingSet::new()
    }
}

pub struct PageGrant {
    start: u32,
    ids: Vec<u32>,
}

impl PageGrant {
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn range(&self) -> PageRange {
        PageRange {
            start: self.start,
            len: self.ids.len() as u32,
        }
    }
}

impl IntoPut for PageGrant {
    fn into_put(self) -> PutValue {
        PutValue::Data(self.ids.into_const())
    }
}

pub struct RsWorkingSet {
    rs: Rc<crate::working_set::RsWorkingSet>,
}

impl RsWorkingSet {
    pub fn new() -> RsWorkingSet {
        RsWorkingSet {
            rs: Rc::new(crate::working_set::RsWorkingSet::new()),
        }
    }

    pub fn state_size(&self) -> u64 {
        thread_local! {
            static SIZE: std::cell::OnceCell<u64> = const { std::cell::OnceCell::new() };
        }
        SIZE.with(|c| *c.get_or_init(crate::model::rs_state_size))
    }

    pub fn buffer_size(&self) -> u32 {
        self.rs.buffer_size()
    }

    pub fn buffer_page_size(&self) -> u32 {
        thread_local! {
            static SIZE: std::cell::OnceCell<u32> = const { std::cell::OnceCell::new() };
        }
        SIZE.with(|c| *c.get_or_init(crate::model::rs_buffer_page_size))
    }

    pub fn alloc_buffer(&self, n: u32) -> Result<crate::working_set::PageRange, String> {
        self.rs.alloc_buffer(n)
    }

    pub fn free_buffer(&self, indices: &[u32]) -> Result<(), String> {
        self.rs.free_buffer(indices)
    }

    pub fn discard_buffered(&self, count: u32) -> Result<(), String> {
        self.rs.discard_buffered(count)
    }

    pub fn reorder_buffer(&self, perm: &[u32]) -> Result<(), String> {
        self.rs.reorder_buffer(perm)
    }

    pub fn fork(&self, on: &Pipeline) -> Result<RsWorkingSet, String> {
        Ok(RsWorkingSet {
            rs: Rc::new(self.rs.fork(&on.wit)?),
        })
    }
}

impl Default for RsWorkingSet {
    fn default() -> Self {
        RsWorkingSet::new()
    }
}

type StageClosure = Box<dyn Fn()>;

pub trait PassWit: Sized + 'static {
    fn new() -> Self;

    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String>;

    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String>;

    fn set_max_layers(&self, max_layers: u32) -> Result<(), String>;

    fn set_drafting_block(&self, on: bool) -> Result<(), String>;

    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String>;

    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String>;

    fn reading(&self, name: &str) -> Result<(), String> {
        Err(format!(
            "this pass interface carries no readings; `reading(\"{name}\")` is a `forward` / \
             `forward-diffusion` verb"
        ))
    }
    fn input(&self, port: &str, _ch: &wit_channel::Channel) -> Result<(), String> {
        Err(format!(
            "this pass interface carries no float ports; `input(\"{port}\")` is a `forward` / \
             `forward-diffusion` verb"
        ))
    }
    fn stream(&self, _s: LaneStream) -> Result<(), String> {
        Err(
            "this pass interface carries no streams; `stream` is a `forward` / \
             `forward-diffusion` verb"
                .to_string(),
        )
    }
    fn group(&self, _id: u32) -> Result<(), String> {
        Err(
            "this pass interface carries no groups; `group` is a `forward` / \
             `forward-diffusion` verb"
                .to_string(),
        )
    }
    fn peer(&self, _ordinal: u32) -> Result<(), String> {
        Err("this pass interface carries no peers; `peer` is a \
             `forward-diffusion` verb, and a peer is a velocity to guide with"
            .to_string())
    }
}

impl PassWit for wit_attention::ForwardPass {
    fn new() -> Self {
        wit_attention::ForwardPass::new()
    }
    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String> {
        wit_attention::ForwardPass::embed(self, tokens, indptr)
    }
    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String> {
        wit_attention::ForwardPass::readout(self, indices)
    }
    fn set_drafting_block(&self, on: bool) -> Result<(), String> {
        wit_attention::ForwardPass::set_drafting_block(self, on).map_err(|e| e.to_string())
    }

    fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        wit_attention::ForwardPass::set_max_layers(self, max_layers).map_err(|e| e.to_string())
    }
    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String> {
        wit_attention::ForwardPass::program(self, bytes, channels)
    }
    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String> {
        wit_attention::submit(on, slots)
    }
    fn reading(&self, name: &str) -> Result<(), String> {
        wit_attention::ForwardPass::reading(self, name)
    }
    fn input(&self, port: &str, ch: &wit_channel::Channel) -> Result<(), String> {
        wit_attention::ForwardPass::input(self, port, ch)
    }
    fn stream(&self, s: LaneStream) -> Result<(), String> {
        wit_attention::ForwardPass::stream(self, s)
    }
    fn group(&self, id: u32) -> Result<(), String> {
        wit_attention::ForwardPass::group(self, id)
    }
    fn peer(&self, ordinal: u32) -> Result<(), String> {
        wit_attention::ForwardPass::peer(self, ordinal)
    }
}

impl PassWit for wit_diffusion::ForwardPass {
    fn new() -> Self {
        wit_diffusion::ForwardPass::new()
    }
    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String> {
        wit_diffusion::ForwardPass::embed(self, tokens, indptr)
    }
    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String> {
        wit_diffusion::ForwardPass::readout(self, indices)
    }
    fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        wit_diffusion::ForwardPass::set_max_layers(self, max_layers).map_err(|e| e.to_string())
    }
    fn set_drafting_block(&self, on: bool) -> Result<(), String> {
        wit_diffusion::ForwardPass::set_drafting_block(self, on).map_err(|e| e.to_string())
    }
    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String> {
        wit_diffusion::ForwardPass::program(self, bytes, channels)
    }
    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String> {
        wit_diffusion::submit(on, slots)
    }
    fn reading(&self, name: &str) -> Result<(), String> {
        wit_diffusion::ForwardPass::reading(self, name)
    }
    fn input(&self, port: &str, ch: &wit_channel::Channel) -> Result<(), String> {
        wit_diffusion::ForwardPass::input(self, port, ch)
    }
    fn stream(&self, s: LaneStream) -> Result<(), String> {
        wit_diffusion::ForwardPass::stream(self, s)
    }
    fn group(&self, id: u32) -> Result<(), String> {
        wit_diffusion::ForwardPass::group(self, id)
    }
    fn peer(&self, ordinal: u32) -> Result<(), String> {
        wit_diffusion::ForwardPass::peer(self, ordinal)
    }
}

impl PassWit for wit_recurrent::ForwardPass {
    fn new() -> Self {
        wit_recurrent::ForwardPass::new()
    }
    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String> {
        wit_recurrent::ForwardPass::embed(self, tokens, indptr)
    }
    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String> {
        wit_recurrent::ForwardPass::readout(self, indices)
    }
    fn set_drafting_block(&self, on: bool) -> Result<(), String> {
        wit_recurrent::ForwardPass::set_drafting_block(self, on).map_err(|e| e.to_string())
    }

    fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        wit_recurrent::ForwardPass::set_max_layers(self, max_layers).map_err(|e| e.to_string())
    }
    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String> {
        wit_recurrent::ForwardPass::program(self, bytes, channels)
    }
    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String> {
        wit_recurrent::submit(on, slots)
    }
}

impl PassWit for wit_hybrid::ForwardPass {
    fn new() -> Self {
        wit_hybrid::ForwardPass::new()
    }
    fn embed(
        &self,
        tokens: &wit_channel::Channel,
        indptr: &wit_channel::Channel,
    ) -> Result<(), String> {
        wit_hybrid::ForwardPass::embed(self, tokens, indptr)
    }
    fn readout(&self, indices: &wit_channel::Channel) -> Result<(), String> {
        wit_hybrid::ForwardPass::readout(self, indices)
    }
    fn set_drafting_block(&self, on: bool) -> Result<(), String> {
        wit_hybrid::ForwardPass::set_drafting_block(self, on).map_err(|e| e.to_string())
    }

    fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        wit_hybrid::ForwardPass::set_max_layers(self, max_layers).map_err(|e| e.to_string())
    }
    fn program(&self, bytes: &[u8], channels: &[&wit_channel::Channel]) -> Result<(), String> {
        wit_hybrid::ForwardPass::program(self, bytes, channels)
    }
    fn submit(on: &wit_pipeline::Pipeline, slots: &[Option<&Self>]) -> Result<(), String> {
        wit_hybrid::submit(on, slots)
    }
}

pub mod adapter {
    use super::Channel;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Site {
        Q,
        K,
        V,
        O,
        GateUp,
        Down,
    }

    impl Site {
        pub fn bit(self) -> u32 {
            match self {
                Site::Q => 1 << 0,
                Site::K => 1 << 1,
                Site::V => 1 << 2,
                Site::O => 1 << 3,
                Site::GateUp => 1 << 4,
                Site::Down => 1 << 5,
            }
        }
    }

    pub struct Expr {
        pub(crate) kind: ExprKind,
    }

    pub(crate) enum ExprKind {
        X,
        Y,
        Mm(Channel, Box<Expr>),
        Add(Box<Expr>, Box<Expr>),
        Scale(Channel, Box<Expr>),
    }

    impl Expr {
        pub(crate) fn x() -> Self {
            Expr { kind: ExprKind::X }
        }
        pub(crate) fn y() -> Self {
            Expr { kind: ExprKind::Y }
        }
    }

    pub fn mm(w: &Channel, e: Expr) -> Expr {
        Expr {
            kind: ExprKind::Mm(*w, Box::new(e)),
        }
    }

    pub fn scale(e: Expr, l: &Channel) -> Expr {
        Expr {
            kind: ExprKind::Scale(*l, Box::new(e)),
        }
    }

    impl std::ops::Add for Expr {
        type Output = Expr;
        fn add(self, rhs: Expr) -> Expr {
            Expr {
                kind: ExprKind::Add(Box::new(self), Box::new(rhs)),
            }
        }
    }
}

pub struct Pass<W: PassWit> {
    wit: W,
    inner: RefCell<ForwardInner>,
}

struct ForwardInner {
    ports: Vec<(Port, DslChannel)>,
    stages: Vec<(Stage, StageClosure)>,
    vocab: u32,
    page_size: u32,
    attention_ws: Option<Rc<KvWorkingSet>>,
    rs_working_sets: Vec<Rc<crate::working_set::RsWorkingSet>>,
    program_attached: bool,
    adapter_lowrank_sites: u32,
    adapter_scale_sites: u32,
    port_inputs: Vec<(String, DslChannel)>,
    latent_rows: Option<u32>,
}

struct StagedKv {
    ws: Rc<KvWorkingSet>,
    readable: PageDeclaration,
    writable: PageDeclaration,
    kv_len: Rc<wit_channel::Channel>,
    pages: Rc<wit_channel::Channel>,
    page_indptr: Rc<wit_channel::Channel>,
    w_slot: Rc<wit_channel::Channel>,
    w_off: Rc<wit_channel::Channel>,
    positions: Rc<wit_channel::Channel>,
    mask: Option<Rc<wit_channel::Channel>>,
}

pub struct KvGeometry<'a, R, W> {
    pub readable_pages: R,
    pub writable_pages: W,
    pub kv_len: &'a Channel,
    pub pages: &'a Channel,
    pub page_indptr: &'a Channel,
    pub w_slot: &'a Channel,
    pub w_off: &'a Channel,
    pub positions: &'a Channel,
    pub mask: Option<&'a Channel>,
}

pub struct RsGeometry<'a, B> {
    pub fold_len: Option<&'a Channel>,
    pub buffer: B,
}

pub struct KvBinding<'a, R, W> {
    pub working_set: &'a WorkingSet,
    pub geometry: KvGeometry<'a, R, W>,
}

thread_local! {
    static FOLD_ALL: Channel = Channel::from(vec![u32::MAX]);
}

struct StagedRs {
    working_sets: Vec<Rc<crate::working_set::RsWorkingSet>>,
    fold_len: Rc<wit_channel::Channel>,
    buffer: PageDeclaration,
}

#[derive(Clone, Copy)]
struct PageDeclaration {
    start: u32,
    end: Option<u32>,
}

impl PageDeclaration {
    fn from_range(range: impl RangeBounds<u32>) -> Result<Self, String> {
        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(&start) => start,
            Bound::Excluded(&start) => start
                .checked_add(1)
                .ok_or_else(|| "attention page-span start overflows u32".to_string())?,
        };
        let end = match range.end_bound() {
            Bound::Unbounded => None,
            Bound::Excluded(&end) => Some(end),
            Bound::Included(&end) => Some(
                end.checked_add(1)
                    .ok_or_else(|| "attention page-span end overflows u32".to_string())?,
            ),
        };
        if end.is_some_and(|end| start > end) {
            return Err(format!(
                "attention page-span start {start} exceeds end {}",
                end.unwrap()
            ));
        }
        Ok(Self { start, end })
    }

    fn wit(self) -> PageSpan {
        PageSpan {
            start: self.start,
            end: self.end,
        }
    }
}

#[cfg(test)]
mod page_declaration_tests {
    use super::*;

    #[test]
    fn normalizes_inclusive_and_exclusive_bounds() {
        let closed = PageDeclaration::from_range(2..5).unwrap();
        assert_eq!((closed.start, closed.end), (2, Some(5)));

        let inclusive = PageDeclaration::from_range(2..=5).unwrap();
        assert_eq!((inclusive.start, inclusive.end), (2, Some(6)));
    }
}

impl<W: PassWit> Pass<W> {
    fn binds_device_mask(&self) -> bool {
        self.inner
            .borrow()
            .ports
            .iter()
            .any(|(port, _)| *port == Port::AttnMask)
    }

    pub fn new() -> Pass<W> {
        let vocab = crate::model::output_vocab_size();
        let page_size = kv_page_size();
        Pass {
            wit: W::new(),
            inner: RefCell::new(ForwardInner {
                ports: Vec::new(),
                stages: Vec::new(),
                vocab,
                page_size,
                attention_ws: None,
                rs_working_sets: Vec::new(),
                program_attached: false,
                adapter_lowrank_sites: 0,
                adapter_scale_sites: 0,
                port_inputs: Vec::new(),
                latent_rows: None,
            }),
        }
    }

    pub fn reading(&self, name: &str) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        self.wit.reading(name)
    }

    pub fn input(&self, port: &str, ch: &Channel) -> Result<(), String> {
        {
            let inner = self.inner.borrow();
            if inner.program_attached {
                return Err("forward pass program is already attached".to_string());
            }
            if inner.port_inputs.iter().any(|(bound, _)| bound == port) {
                return Err(format!("port `{port}` is already bound on this pass"));
            }
        }
        let wit = ch.wit();
        self.wit.input(port, wit.as_ref())?;
        let row_port = crate::model::readings().iter().any(|reading| {
            reading.ports.iter().any(|fact| {
                fact.name == port
                    && matches!(
                        fact.kind,
                        crate::model::PortKind::Latents | crate::model::PortKind::Context
                    )
            })
        });
        let mut inner = self.inner.borrow_mut();
        if row_port && let Some(&rows) = ch.shape().dims().first() {
            inner.latent_rows = Some(rows);
        }
        inner.port_inputs.push((port.to_string(), ch.dsl()));
        Ok(())
    }

    pub fn stream(&self, stream: LaneStream) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        self.wit.stream(stream)
    }

    pub fn group(&self, id: u32) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        self.wit.group(id)
    }

    pub fn peer(&self, ordinal: u32) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        self.wit.peer(ordinal)
    }

    fn ensure_ports_available(&self, ports: &[Port]) -> Result<(), String> {
        let inner = self.inner.borrow();
        if inner.program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        if let Some(port) = ports
            .iter()
            .find(|port| inner.ports.iter().any(|(bound, _)| bound == *port))
        {
            return Err(format!(
                "forward pass port {} is already bound",
                port.name()
            ));
        }
        Ok(())
    }

    pub fn embed(&self, tokens: &Channel, indptr: &Channel) -> Result<(), String> {
        self.ensure_ports_available(&[Port::EmbedTokens, Port::EmbedIndptr])?;
        let token_wit = tokens.wit();
        let indptr_wit = indptr.wit();
        self.wit.embed(token_wit.as_ref(), indptr_wit.as_ref())?;
        self.inner.borrow_mut().ports.extend([
            (Port::EmbedTokens, claim_port(Port::EmbedTokens, tokens)),
            (Port::EmbedIndptr, claim_port(Port::EmbedIndptr, indptr)),
        ]);
        Ok(())
    }

    fn stage_kv<R, Wr>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, Wr>,
    ) -> Result<StagedKv, String>
    where
        R: RangeBounds<u32>,
        Wr: RangeBounds<u32>,
    {
        let KvGeometry {
            readable_pages,
            writable_pages,
            kv_len,
            pages,
            page_indptr,
            w_slot,
            w_off,
            positions,
            mask,
        } = geom;
        let rebind = self.inner.borrow().program_attached;
        if !rebind {
            let mut ports = vec![
                Port::KvLen,
                Port::Pages,
                Port::PageIndptr,
                Port::WSlot,
                Port::WOff,
                Port::Positions,
            ];
            if mask.is_some() {
                ports.push(Port::AttnMask);
            }
            self.ensure_ports_available(&ports)?;
        }
        let staged = StagedKv {
            ws: ws.kv.clone(),
            readable: PageDeclaration::from_range(readable_pages)?,
            writable: PageDeclaration::from_range(writable_pages)?,
            kv_len: kv_len.wit(),
            pages: pages.wit(),
            page_indptr: page_indptr.wit(),
            w_slot: w_slot.wit(),
            w_off: w_off.wit(),
            positions: positions.wit(),
            mask: mask.map(Channel::wit),
        };
        let mut inner = self.inner.borrow_mut();
        inner.attention_ws = Some(ws.kv.clone());
        if !rebind {
            inner.ports.extend([
                (Port::KvLen, claim_port(Port::KvLen, kv_len)),
                (Port::Pages, claim_port(Port::Pages, pages)),
                (Port::PageIndptr, claim_port(Port::PageIndptr, page_indptr)),
                (Port::WSlot, claim_port(Port::WSlot, w_slot)),
                (Port::WOff, claim_port(Port::WOff, w_off)),
                (Port::Positions, claim_port(Port::Positions, positions)),
            ]);
            if let Some(mask) = mask {
                inner
                    .ports
                    .push((Port::AttnMask, claim_port(Port::AttnMask, mask)));
            }
        }
        Ok(staged)
    }

    fn stage_rs<B>(
        &self,
        working_sets: &[RsWorkingSet],
        geom: RsGeometry<'_, B>,
    ) -> Result<StagedRs, String>
    where
        B: RangeBounds<u32>,
    {
        let buffer = PageDeclaration::from_range(geom.buffer)?;
        let staged = match geom.fold_len {
            Some(fold_len) => {
                if !self.inner.borrow().program_attached {
                    self.ensure_ports_available(&[Port::RsFoldLen])?;
                    let mut inner = self.inner.borrow_mut();
                    inner
                        .ports
                        .push((Port::RsFoldLen, claim_port(Port::RsFoldLen, fold_len)));
                }
                fold_len.wit()
            }
            None => FOLD_ALL.with(Channel::wit),
        };
        let working_sets: Vec<Rc<crate::working_set::RsWorkingSet>> =
            working_sets.iter().map(|rs| rs.rs.clone()).collect();
        self.inner.borrow_mut().rs_working_sets = working_sets.clone();
        Ok(StagedRs {
            working_sets,
            fold_len: staged,
            buffer,
        })
    }

    pub fn readout(&self, indices: &Channel) -> Result<(), String> {
        self.ensure_ports_available(&[Port::Readout])?;
        let indices_wit = indices.wit();
        self.wit.readout(indices_wit.as_ref())?;
        self.inner
            .borrow_mut()
            .ports
            .push((Port::Readout, claim_port(Port::Readout, indices)));
        Ok(())
    }

    pub fn set_max_layers(&self, max_layers: u32) -> Result<(), String> {
        self.wit.set_max_layers(max_layers)
    }

    pub fn set_drafting_block(&self, on: bool) -> Result<(), String> {
        self.wit.set_drafting_block(on)
    }

    pub fn adapter(
        &self,
        site: adapter::Site,
        f: impl FnOnce(adapter::Expr, adapter::Expr) -> adapter::Expr,
    ) -> Result<(), String> {
        use adapter::ExprKind as K;
        let expr = f(adapter::Expr::x(), adapter::Expr::y());
        if let K::Scale(l, inner) = &expr.kind
            && let K::Add(lhs, rhs) = &inner.kind
        {
            let delta = match (&lhs.kind, &rhs.kind) {
                (K::Y, _) => &rhs.kind,
                (_, K::Y) => &lhs.kind,
                _ => &inner.kind,
            };
            if let K::Mm(b, mid) = delta
                && let K::Mm(a, x) = &mid.kind
                && matches!(x.kind, K::X)
            {
                let (a, b, l) = (*a, *b, *l);
                {
                    let mut st = self.inner.borrow_mut();
                    if (st.adapter_lowrank_sites | st.adapter_scale_sites) & site.bit() != 0 {
                        return Err(format!(
                            "adapter: site {site:?} already carries an \
                                         adapter on this pass"
                        ));
                    }
                    st.adapter_lowrank_sites |= site.bit();
                    st.adapter_scale_sites |= site.bit();
                }
                self.prologue(move || {
                    intrinsics::kernel::lora(a.read(), b.read(), Tensor::constant(site.bit()));
                    intrinsics::kernel::adapter_scale(l.read(), Tensor::constant(site.bit()));
                });
                return Ok(());
            }
        }
        if let K::Scale(l, inner) = &expr.kind
            && matches!(inner.kind, K::Y)
        {
            let l = *l;
            {
                let mut inner_state = self.inner.borrow_mut();
                if inner_state.adapter_scale_sites & site.bit() != 0 {
                    return Err(format!(
                        "adapter: site {site:?} already carries a scale on \
                             this pass"
                    ));
                }
                inner_state.adapter_scale_sites |= site.bit();
            }
            self.prologue(move || {
                intrinsics::kernel::adapter_scale(l.read(), Tensor::constant(site.bit()));
            });
            return Ok(());
        }
        let (lhs, rhs) = match expr.kind {
            K::Add(l, r) => (*l, *r),
            _ => {
                return Err("adapter: form not lowerable (v0 lowers the low-rank \
                     form `y + mm(b, mm(a, x))` only)"
                    .to_string());
            }
        };
        let delta = match (&lhs.kind, &rhs.kind) {
            (K::Y, _) => rhs.kind,
            (_, K::Y) => lhs.kind,
            _ => return Err("adapter: the base output `y` must be one addend".to_string()),
        };
        let (b, a) = match delta {
            K::Mm(b, inner) => match inner.kind {
                K::Mm(a, x) if matches!(x.kind, K::X) => (b, a),
                _ => return Err("adapter: the delta must be mm(b, mm(a, x))".to_string()),
            },
            _ => return Err("adapter: the delta must be mm(b, mm(a, x))".to_string()),
        };
        {
            let mut inner = self.inner.borrow_mut();
            if inner.adapter_lowrank_sites & site.bit() != 0 {
                return Err(format!(
                    "adapter: site {site:?} already carries an adapter on this pass"
                ));
            }
            inner.adapter_lowrank_sites |= site.bit();
        }
        self.prologue(move || {
            intrinsics::kernel::lora(a.read(), b.read(), Tensor::constant(site.bit()));
        });
        Ok(())
    }

    pub fn prologue(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::Prologue, body);
    }
    pub fn epilogue(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::Epilogue, body);
    }

    fn set_stage(&self, stage: Stage, body: impl Fn() + 'static) {
        let mut inner = self.inner.borrow_mut();
        assert!(
            !inner.program_attached,
            "stage attachment is construction-only"
        );
        if let Some(slot) = inner.stages.iter_mut().find(|(s, _)| *s == stage) {
            slot.1 = Box::new(body);
        } else {
            inner.stages.push((stage, Box::new(body)));
        }
    }

    pub fn submit(&self, on: &Pipeline) -> Result<(), String> {
        submit_frame(on, &[Some(self)])
    }

    fn attach_program(&self) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Ok(());
        }
        let inner = self.inner.borrow();
        let mut builder = Builder::new(inner.vocab, inner.page_size);
        if let Some(rows) = inner.latent_rows {
            builder.rows_hint(rows);
        }
        for (port, channel) in &inner.ports {
            builder.bind_port_recorded(*port, channel.clone());
        }
        let port_channels: Vec<DslChannel> =
            inner.port_inputs.iter().map(|(_, ch)| ch.clone()).collect();
        let author_prologue = inner
            .stages
            .iter()
            .find(|(stage, _)| *stage == Stage::Prologue)
            .map(|(_, body)| body);
        if !port_channels.is_empty() {
            builder.stage(Stage::Prologue, move || {
                for ch in &port_channels {
                    let _ = ch.read();
                }
                if let Some(body) = author_prologue {
                    body();
                }
            });
        }
        for (stage, body) in &inner.stages {
            if *stage == Stage::Prologue && !inner.port_inputs.is_empty() {
                continue;
            }
            builder.stage(*stage, body);
        }
        let traced = builder.build().map_err(|error| error.to_string())?;
        drop(builder);
        let handles: Vec<Rc<wit_channel::Channel>> = traced
            .channel_order()
            .iter()
            .map(|gid| lookup_channel(*gid).expect("channel registered before submit"))
            .collect();
        let borrows: Vec<&wit_channel::Channel> = handles.iter().map(Rc::as_ref).collect();
        let bytes = traced.encode();
        self.wit.program(&bytes, &borrows)?;
        drop(inner);
        self.inner.borrow_mut().program_attached = true;
        Ok(())
    }
}

impl<W: PassWit> Default for Pass<W> {
    fn default() -> Self {
        Pass::new()
    }
}

pub fn frame_size() -> usize {
    thread_local! {
        static FRAME_SIZE: std::cell::OnceCell<usize> = const { std::cell::OnceCell::new() };
    }
    FRAME_SIZE.with(|k| *k.get_or_init(|| crate::model::frame_size().max(1) as usize))
}

pub fn submit_deadline() -> std::time::Duration {
    thread_local! {
        static DEADLINE: std::cell::OnceCell<u64> = const { std::cell::OnceCell::new() };
    }
    std::time::Duration::from_micros(
        DEADLINE.with(|d| *d.get_or_init(crate::model::submit_deadline_us)),
    )
}

pub fn channel_capacity() -> usize {
    (crate::model::channel_capacity() as usize).max(2)
}

pub fn kv_page_size() -> u32 {
    thread_local! {
        static PAGE: std::cell::OnceCell<u32> = const { std::cell::OnceCell::new() };
    }
    PAGE.with(|c| *c.get_or_init(crate::model::kv_page_size))
}

pub fn max_embed_length() -> usize {
    thread_local! {
        static MAX_EMBED: std::cell::OnceCell<usize> = const { std::cell::OnceCell::new() };
    }
    MAX_EMBED.with(|c| *c.get_or_init(|| crate::model::max_embed_length().max(1) as usize))
}

pub fn prefill_chunk_hint() -> usize {
    (crate::model::prefill_chunk_hint().max(1) as usize).min(max_embed_length())
}

pub fn prefill_chunks(n: u32, cap: Option<u32>) -> Vec<(u32, u32)> {
    let cap = cap
        .unwrap_or_else(|| prefill_chunk_hint() as u32)
        .min(max_embed_length().max(1) as u32)
        .max(1);
    even_spans(n, cap)
}

fn even_spans(n: u32, cap: u32) -> Vec<(u32, u32)> {
    if n == 0 {
        return Vec::new();
    }
    let cap = cap.min(n).max(1);
    let k = n.div_ceil(cap).max(1);
    let (q, r) = (n / k, n % k);
    let mut out = Vec::with_capacity(k as usize);
    let mut base = 0u32;
    for i in 0..k {
        let end = base + q + u32::from(i < r);
        out.push((base, end));
        base = end;
    }
    debug_assert_eq!(base, n);
    out
}

pub fn submit_frame<W: PassWit>(on: &Pipeline, slots: &[Option<&Pass<W>>]) -> Result<(), String> {
    let k = frame_size();
    if slots.len() > k {
        return Err(format!(
            "frame holds {} slot(s); model.frame-size() is {k}",
            slots.len()
        ));
    }
    for pass in slots.iter().flatten() {
        pass.attach_program()?;
    }
    if slots.iter().flatten().next().is_none() {
        return Ok(());
    }
    let mut borrows: Vec<Option<&W>> = slots.iter().map(|slot| slot.map(|p| &p.wit)).collect();
    borrows.resize(k, None);
    W::submit(&on.wit, &borrows)
}

pub async fn run_ahead<W: PassWit>(
    on: &Pipeline,
    pass: &Pass<W>,
    budget: usize,
    mut on_token: impl AsyncFnMut() -> Result<std::ops::ControlFlow<()>, String>,
) -> Result<usize, String> {
    use std::ops::ControlFlow;

    if budget == 0 {
        return Ok(0);
    }
    let r = if pass.binds_device_mask()
        || crate::model::pass_kind() != crate::model::ForwardKind::Attention
    {
        1
    } else {
        frame_size()
    };
    let window_fires = crate::model::run_ahead_window() as usize;
    let window_frames = (window_fires / r.max(1)).max(1);

    let mut submitted = 0usize;
    let mut consumed = 0usize;

    let submit_one_frame = |submitted: &mut usize| -> Result<(), String> {
        let live = r.min(budget - *submitted);
        if live == 0 {
            return Ok(());
        }
        let slots: Vec<Option<&Pass<W>>> = vec![Some(pass); live];
        submit_frame(on, &slots)?;
        *submitted += live;
        Ok(())
    };

    for _ in 0..window_frames {
        if submitted >= budget {
            break;
        }
        submit_one_frame(&mut submitted)?;
    }

    let mut ended = false;

    if submitted >= budget && !ended {
        on.close();
        ended = true;
    }
    while consumed < submitted {
        if on_token().await? == ControlFlow::Break(()) {
            if !ended {
                on.close();
            }
            return Ok(consumed + 1);
        }
        consumed += 1;
        if submitted < budget && submitted - consumed <= (window_frames - 1) * r {
            submit_one_frame(&mut submitted)?;
        }
        if submitted >= budget && !ended {
            on.close();
            ended = true;
        }
    }
    if !ended {
        on.close();
    }
    Ok(consumed)
}

pub struct Pipeline {
    wit: wit_pipeline::Pipeline,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {
            wit: wit_pipeline::Pipeline::new(),
        }
    }

    pub fn close(&self) {
        self.wit.close();
    }

    pub fn park(&self) {
        wit_attention::park(&self.wit);
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Pipeline::new()
    }
}

pub mod shared_prelude {
    pub use super::{
        Channel, KvBinding, KvGeometry, LaneStream, PageGrant, Pipeline, RsGeometry, RsWorkingSet,
        TOKEN_PAD, WorkingSet, channel_capacity, frame_size, kv_page_size, max_embed_length,
        prefill_chunk_hint, prefill_chunks,
    };
    pub use crate::{Context, Result, model};
    pub use eta_dsl::Stage;
    pub use eta_dsl::dtype;
    pub use eta_dsl::intrinsics;
    pub use eta_dsl::value::{
        Tensor, abs, and, broadcast, cast, causal_mask, cos, cummass_le, cumprod, cumsum, entropy,
        entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_max, indptr,
        iota, l2norm, le, log, log_softmax, lt, mask_apply, masked_argmax, matmul, max_elem,
        min_elem, ne, normal, not, nucleus_sample, or, pivot_threshold, prob_ge, rank_le, recip,
        reduce_argmax, reduce_max, reduce_min, reduce_sum, reshape, rng, row_membership, rsqrt,
        scalar_gather, scatter_add, scatter_set, select, sign, sin, sink_window_mask,
        sliding_window_mask, softmax, sort_desc, sqrt, top_k, transpose,
    };
    pub use std::ops::ControlFlow;
}

impl Pass<wit_attention::ForwardPass> {
    pub fn on_attn_proj(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttnProj, body);
    }
    pub fn on_attn(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttn, body);
    }

    pub fn attention<R, W>(&self, ws: &WorkingSet, geom: KvGeometry<'_, R, W>) -> Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        let kv = self.stage_kv(ws, geom)?;
        wit_attention::ForwardPass::attention(
            &self.wit,
            kv.ws.as_ref(),
            &wit_attention::KvGeometry {
                readable_pages: kv.readable.wit(),
                writable_pages: kv.writable.wit(),
                kv_len: kv.kv_len.as_ref(),
                pages: kv.pages.as_ref(),
                page_indptr: kv.page_indptr.as_ref(),
                w_slot: kv.w_slot.as_ref(),
                w_off: kv.w_off.as_ref(),
                positions: kv.positions.as_ref(),
                mask: kv.mask.as_deref(),
            },
        )
    }

    pub fn media(&self, spans: &[wit_attention::MediaSpan<'_>]) -> Result<(), String> {
        wit_attention::ForwardPass::media(&self.wit, spans)
    }
}

impl Pass<wit_hybrid::ForwardPass> {
    pub fn media(&self, spans: &[wit_attention::MediaSpan<'_>]) -> Result<(), String> {
        wit_hybrid::ForwardPass::media(&self.wit, spans)
    }

    pub fn on_attn_proj(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttnProj, body);
    }
    pub fn on_attn(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttn, body);
    }

    pub fn attention<R, W, B>(
        &self,
        kv: Option<KvBinding<'_, R, W>>,
        rs: &[RsWorkingSet],
        rs_geom: RsGeometry<'_, B>,
    ) -> Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
        B: RangeBounds<u32>,
    {
        let kv = kv
            .map(|kv| self.stage_kv(kv.working_set, kv.geometry))
            .transpose()?;
        let rs = self.stage_rs(rs, rs_geom)?;
        let binding = kv.as_ref().map(|kv| wit_hybrid::KvBinding {
            working_set: kv.ws.as_ref(),
            geometry: wit_hybrid::KvGeometry {
                readable_pages: kv.readable.wit(),
                writable_pages: kv.writable.wit(),
                kv_len: kv.kv_len.as_ref(),
                pages: kv.pages.as_ref(),
                page_indptr: kv.page_indptr.as_ref(),
                w_slot: kv.w_slot.as_ref(),
                w_off: kv.w_off.as_ref(),
                positions: kv.positions.as_ref(),
                mask: kv.mask.as_deref(),
            },
        });
        let borrows: Vec<&crate::working_set::RsWorkingSet> =
            rs.working_sets.iter().map(Rc::as_ref).collect();
        wit_hybrid::ForwardPass::attention(
            &self.wit,
            binding.as_ref(),
            &borrows,
            &wit_hybrid::RsGeometry {
                fold_len: rs.fold_len.as_ref(),
                buffer: rs.buffer.wit(),
            },
        )
    }
}

impl Pass<wit_diffusion::ForwardPass> {
    pub fn on_attn_proj(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttnProj, body);
    }
    pub fn on_attn(&self, body: impl Fn() + 'static) {
        self.set_stage(Stage::OnAttn, body);
    }

    pub fn attention<R, W>(&self, ws: &WorkingSet, geom: KvGeometry<'_, R, W>) -> Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        let kv = self.stage_kv(ws, geom)?;
        wit_diffusion::ForwardPass::attention(
            &self.wit,
            kv.ws.as_ref(),
            &wit_diffusion::KvGeometry {
                readable_pages: kv.readable.wit(),
                writable_pages: kv.writable.wit(),
                kv_len: kv.kv_len.as_ref(),
                pages: kv.pages.as_ref(),
                page_indptr: kv.page_indptr.as_ref(),
                w_slot: kv.w_slot.as_ref(),
                w_off: kv.w_off.as_ref(),
                positions: kv.positions.as_ref(),
                mask: kv.mask.as_deref(),
            },
        )
    }

    pub fn canvas(&self, mode: wit_diffusion::Mode) -> Result<(), String> {
        wit_diffusion::ForwardPass::canvas(&self.wit, mode)
    }

    pub fn self_conditioning(&self, rows: &[u32], weights: &[f32]) -> Result<(), String> {
        wit_diffusion::ForwardPass::self_conditioning(&self.wit, rows, weights)
    }

    pub fn self_conditioning_from(&self, rows: &Channel, weights: &Channel) -> Result<(), String> {
        wit_diffusion::ForwardPass::self_conditioning_from(&self.wit, &rows.wit(), &weights.wit())
    }

    pub fn media(&self, spans: &[wit_attention::MediaSpan<'_>]) -> Result<(), String> {
        wit_diffusion::ForwardPass::media(&self.wit, spans)
    }
}

impl Pass<wit_recurrent::ForwardPass> {
    pub fn attention<B>(&self, rs: &[RsWorkingSet], geom: RsGeometry<'_, B>) -> Result<(), String>
    where
        B: RangeBounds<u32>,
    {
        let rs = self.stage_rs(rs, geom)?;
        let borrows: Vec<&crate::working_set::RsWorkingSet> =
            rs.working_sets.iter().map(Rc::as_ref).collect();
        wit_recurrent::ForwardPass::attention(
            &self.wit,
            &borrows,
            &wit_recurrent::RsGeometry {
                fold_len: rs.fold_len.as_ref(),
                buffer: rs.buffer.wit(),
            },
        )
    }
}

pub mod attention {
    pub type ForwardPass = super::Pass<super::wit_attention::ForwardPass>;
    pub use super::{run_ahead, submit_frame};

    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::eta::shared_prelude::*;
    }
}

pub mod diffusion {
    pub type ForwardPass = super::Pass<super::wit_diffusion::ForwardPass>;
    pub use super::wit_diffusion::Mode;
    pub use super::{run_ahead, submit_frame};

    pub mod prelude {
        pub use super::{
            ForwardPass, Mode, entropy_bound_accept, linear_temperature, run_ahead,
            stable_and_confident, submit_frame,
        };
        pub use crate::eta::shared_prelude::*;
    }

    use super::{
        Tensor, and, cast, cumsum, dtype, eq, iota, le, lt, reduce_sum, scatter_set, sort_desc,
    };

    pub fn linear_temperature(remaining: u32, max_steps: u32, t_max: f32, t_min: f32) -> f32 {
        t_min + (t_max - t_min) * (remaining as f32 / max_steps.max(1) as f32)
    }

    pub fn entropy_bound_accept(entropy: &Tensor, bound: f32) -> Tensor {
        let n = entropy.shape().dims()[0];
        let (neg_sorted, order) = sort_desc(-entropy);
        let sorted = -&neg_sorted;
        let below = le(&(&cumsum(&sorted) - &sorted), bound);
        let none = lt(iota(n), 0u32);
        scatter_set(&none, &order, &below)
    }

    pub fn stable_and_confident(
        argmax: &Tensor,
        previous: &Tensor,
        entropy: &Tensor,
        threshold: f32,
    ) -> Tensor {
        let n = argmax.shape().dims()[0];
        let unchanged = reduce_sum(cast(eq(argmax, previous), dtype::i32));
        let stable = eq(&unchanged, n as i32);
        let mean = &reduce_sum(entropy) / (n as f32);
        and(&stable, lt(&mean, threshold))
    }
}

pub mod recurrent {
    pub type ForwardPass = super::Pass<super::wit_recurrent::ForwardPass>;
    pub use super::{run_ahead, submit_frame};

    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::eta::shared_prelude::*;
    }
}

pub mod hybrid {
    pub type ForwardPass = super::Pass<super::wit_hybrid::ForwardPass>;
    pub use super::{run_ahead, submit_frame};

    pub mod prelude {
        pub use super::{ForwardPass, run_ahead, submit_frame};
        pub use crate::eta::shared_prelude::*;
    }
}
