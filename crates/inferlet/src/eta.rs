use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use eta_dsl::builder::Builder;
use eta_dsl::channel::PutValue;
use eta_dsl::eta::registry::{Axis, OutputDecl};
use eta_dsl::value::ConstData;
use eta_dsl::{Channel as DslChannel, IntoConst, IntoPut, IntoShape, Port, Shape, Stage, Tensor};

use crate::pie::inferlet::channel as wit_channel;
use crate::pie::inferlet::forward as wit_forward;
use crate::pie::inferlet::pipeline as wit_pipeline;
use crate::pie::inferlet::types::Dtype as WitDtype;
use crate::pie::inferlet::working_set as wit_ws;

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
    #[track_caller]
    pub fn new(shape: impl IntoShape, dtype: Dtype) -> Channel {
        Channel::build(shape.into_shape(), dtype, false)
    }

    #[track_caller]
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

    /// A queue between passes: another pass consumes it, so the pass that
    /// puts it does not read it as its output for the host; a pass that
    /// peeks it as a port (`kv_len`, `pages`, `page_indptr`) takes each value
    /// in its own epilogue, and another pass's put appends to the ring
    /// rather than replacing the value a peek reads. Set before use.
    pub fn queued(self) -> Channel {
        let _ = self.dsl().queued();
        self
    }

    #[track_caller]
    pub fn from(v: impl IntoConst) -> Channel {
        let data: ConstData = v.into_const();
        let ch = Channel::build(data.shape, data.dtype, true);
        ch.wit()
            .put(&data.bytes)
            .expect("stage seed on a fresh channel");
        ch
    }

    #[track_caller]
    pub fn seeded(shape: impl IntoShape, dtype: Dtype) -> Channel {
        Channel::build(shape.into_shape(), dtype, true)
    }

    #[track_caller]
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

    #[track_caller]
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
        format!("channel {} {verb}", self.dsl().site().at())
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

/// One sequence's model state. Slots below [`WorkingSet::folded`] are frozen;
/// the rest is a buffer a pass may write again.
pub struct WorkingSet {
    ws: Rc<wit_ws::WorkingSet>,
}

impl WorkingSet {
    pub fn new() -> WorkingSet {
        WorkingSet {
            ws: Rc::new(wit_ws::WorkingSet::new()),
        }
    }

    /// Lets passes write slots below `end`: needed only by a pass whose slots
    /// are computed on the device.
    pub fn reserve(&self, end: u32) -> Result<(), String> {
        self.ws.reserve(end)
    }

    /// The frozen length; while a device-counted fold is in flight, the length
    /// before it.
    pub fn folded(&self) -> u32 {
        self.ws.folded()
    }

    /// Slots mapped in from state the runtime already held: a prompt's cached
    /// tokens.
    pub fn cached_tokens(&self) -> u32 {
        self.ws.cached_tokens()
    }

    /// Later rows stop reading the whole pages inside frozen `[start, end)`;
    /// returns the slots dropped.
    pub fn forget(&self, start: u32, end: u32) -> Result<u32, String> {
        self.ws
            .forget(start, end)
            .map_err(|error| error.to_string())
    }

    /// Later rows also read the whole frozen pages of `other` inside
    /// `[start, end)`; returns the slots merged.
    pub fn merge(&self, other: &WorkingSet, start: u32, end: u32) -> Result<u32, String> {
        self.ws
            .merge(&other.ws, start, end)
            .map_err(|error| error.to_string())
    }

    /// A child sharing the frozen part, with a copy of the buffer.
    pub fn fork(&self, on: &Pipeline) -> Result<WorkingSet, String> {
        Ok(WorkingSet {
            ws: Rc::new(self.ws.fork(&on.wit)?),
        })
    }
}

impl Default for WorkingSet {
    fn default() -> Self {
        WorkingSet::new()
    }
}

/// How many more slots each lane freezes after a fire writes.
#[derive(Clone, Copy, Default)]
pub enum Fold<'a> {
    /// None: everything stays in the buffer.
    #[default]
    None,
    /// Everything the lane has written.
    All,
    /// Per lane (or one for all), possibly computed on the device; lanes
    /// sharing a working set fold by the first one's.
    Count(&'a Channel),
}

/// Where a pass's rows land and what they read. Left out, rows append, are
/// roped at their slot and read every slot up to their own.
#[derive(Clone, Copy, Default)]
pub struct StateGeometry<'a> {
    /// Per row, the RoPE position.
    pub positions: Option<&'a Channel>,
    /// Per row, the slot it writes; a lane's rows write consecutive slots.
    pub slots: Option<&'a Channel>,
    /// Per lane, how many slots the lane reads.
    pub kv_len: Option<&'a Channel>,
    pub fold: Fold<'a>,
}

impl StateGeometry<'_> {
    /// Append and freeze everything written.
    pub fn folding() -> StateGeometry<'static> {
        StateGeometry {
            fold: Fold::All,
            ..StateGeometry::default()
        }
    }
}

type StageClosure = Box<dyn Fn()>;

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

/// A forward pass: its lanes' bindings and switches, and the program its fires run.
pub struct ForwardPass {
    wit: wit_forward::ForwardPass,
    inner: RefCell<ForwardInner>,
}

struct ForwardInner {
    ports: Vec<(Port, DslChannel)>,
    stages: Vec<(Stage, StageClosure)>,
    vocab: u32,
    page_size: u32,
    lanes: Vec<Rc<wit_ws::WorkingSet>>,
    fold_all: Option<Channel>,
    program_attached: bool,
    adapter_lowrank_sites: u32,
    adapter_scale_sites: u32,
    inputs: Vec<(u32, String, DslChannel)>,
    switches: Vec<(u32, String, String)>,
}

impl ForwardPass {
    /// Whether a switch is off its model's default; such a pass posts one fire at a time.
    fn switched(&self) -> bool {
        let inner = self.inner.borrow();
        if inner.switches.is_empty() {
            return false;
        }
        let declared = crate::model::switches();
        inner.switches.iter().any(|(_, name, value)| {
            declared
                .iter()
                .find(|decl| decl.name == *name)
                .is_some_and(|decl| decl.values.first() != Some(value))
        })
    }

    pub fn new() -> ForwardPass {
        let vocab = crate::model::output_vocab_size();
        let page_size = kv_page_size();
        ForwardPass {
            wit: wit_forward::ForwardPass::new(),
            inner: RefCell::new(ForwardInner {
                ports: Vec::new(),
                stages: Vec::new(),
                vocab,
                page_size,
                lanes: Vec::new(),
                fold_all: None,
                program_attached: false,
                adapter_lowrank_sites: 0,
                adapter_scale_sites: 0,
                inputs: Vec::new(),
                switches: Vec::new(),
            }),
        }
    }

    /// Sets lane `lane`'s model switch `name` (`model::switches()`) to `value`.
    pub fn switch(&self, lane: u32, name: &str, value: &str) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        self.wit.switch(lane, name, value)?;
        self.inner
            .borrow_mut()
            .switches
            .push((lane, name.to_string(), value.to_string()));
        Ok(())
    }

    /// Binds `ch` to lane `lane`'s model input `name`, its committed cell read every submit.
    pub fn input(&self, lane: u32, name: &str, ch: &Channel) -> Result<(), String> {
        {
            let inner = self.inner.borrow();
            if inner.program_attached {
                return Err("forward pass program is already attached".to_string());
            }
            if inner
                .inputs
                .iter()
                .any(|(at, bound, _)| *at == lane && bound == name)
            {
                return Err(format!("input `{name}` is already bound on lane {lane}"));
            }
        }
        let wit = ch.wit();
        self.wit.input(lane, name, wit.as_ref())?;
        self.inner
            .borrow_mut()
            .inputs
            .push((lane, name.to_string(), ch.dsl()));
        Ok(())
    }

    /// Puts lane `lane` in attention group `id`.
    pub fn group(&self, lane: u32, id: u32) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        self.wit.group(lane, id)
    }

    /// Per row, the rows it attends: kv slots if the lanes read kv, else its group's rows.
    pub fn mask(&self, rows: &Channel) -> Result<(), String> {
        self.ensure_ports_available(&[Port::AttnMask])?;
        let wit = rows.wit();
        self.wit.mask(wit.as_ref())?;
        self.inner
            .borrow_mut()
            .ports
            .push((Port::AttnMask, claim_port(Port::AttnMask, rows)));
        Ok(())
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

    /// Binds one working set per lane and where rows land; after attach it may
    /// bind other working sets (a beam's forks).
    pub fn state(&self, lanes: &[&WorkingSet], geom: StateGeometry<'_>) -> Result<(), String> {
        let rebind = self.inner.borrow().program_attached;
        let fold = match geom.fold {
            Fold::None => None,
            Fold::Count(channel) => Some(*channel),
            Fold::All => Some(
                *self
                    .inner
                    .borrow_mut()
                    .fold_all
                    .get_or_insert_with(|| Channel::from(vec![u32::MAX])),
            ),
        };
        let stated = [
            (Port::Positions, geom.positions.copied()),
            (Port::Slots, geom.slots.copied()),
            (Port::KvLen, geom.kv_len.copied()),
            (Port::Fold, fold),
        ];
        if !rebind {
            let ports: Vec<Port> = stated
                .iter()
                .filter(|(_, channel)| channel.is_some())
                .map(|(port, _)| *port)
                .collect();
            self.ensure_ports_available(&ports)?;
        }
        let wits: Vec<Option<Rc<wit_channel::Channel>>> = stated
            .iter()
            .map(|(_, channel)| channel.as_ref().map(Channel::wit))
            .collect();
        let handles: Vec<Rc<wit_ws::WorkingSet>> = lanes.iter().map(|ws| ws.ws.clone()).collect();
        let borrows: Vec<&wit_ws::WorkingSet> = handles.iter().map(Rc::as_ref).collect();
        self.wit.state(
            &borrows,
            &wit_forward::StateGeometry {
                positions: wits[0].as_deref(),
                slots: wits[1].as_deref(),
                kv_len: wits[2].as_deref(),
                fold: wits[3].as_deref(),
            },
        )?;
        let mut inner = self.inner.borrow_mut();
        inner.lanes = handles;
        if !rebind {
            for (port, channel) in &stated {
                if let Some(channel) = channel {
                    inner.ports.push((*port, claim_port(*port, channel)));
                }
            }
        }
        Ok(())
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

    pub fn media(&self, lane: u32, spans: &[wit_forward::MediaSpan<'_>]) -> Result<(), String> {
        self.wit.media(lane, spans)
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

    pub fn bind(&self) -> Result<(), String> {
        self.attach_program()
    }

    fn attach_program(&self) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Ok(());
        }
        let inner = self.inner.borrow();
        let mut builder = Builder::new(inner.vocab, inner.page_size);
        builder.outputs(lane_outputs(&inner)?);
        for (port, channel) in &inner.ports {
            builder.bind_port_recorded(*port, channel.clone());
        }
        // A channel several lanes read is one input of the program.
        let mut listed: Vec<u64> = Vec::new();
        for (_, _, channel) in &inner.inputs {
            if !listed.contains(&channel.gid()) {
                listed.push(channel.gid());
                builder.bind_input(channel.clone());
            }
        }
        for (stage, body) in &inner.stages {
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

impl Default for ForwardPass {
    fn default() -> Self {
        ForwardPass::new()
    }
}

type Declared = (Vec<(String, String)>, Vec<OutputDecl>);

/// Each lane's outputs, in lane order, by its switches.
fn lane_outputs(inner: &ForwardInner) -> Result<Vec<Vec<OutputDecl>>, String> {
    thread_local! {
        static DECLARED: RefCell<Vec<Declared>> =
            const { RefCell::new(Vec::new()) };
    }
    let lanes = inner
        .switches
        .iter()
        .map(|(lane, _, _)| lane + 1)
        .chain(inner.inputs.iter().map(|(lane, _, _)| lane + 1))
        .chain([u32::try_from(inner.lanes.len()).unwrap_or(u32::MAX), 1])
        .max()
        .unwrap_or(1);
    (0..lanes)
        .map(|lane| {
            let switches: Vec<(String, String)> = inner
                .switches
                .iter()
                .filter(|(at, _, _)| *at == lane)
                .map(|(_, name, value)| (name.clone(), value.clone()))
                .collect();
            if let Some(hit) = DECLARED.with_borrow(|seen| {
                seen.iter()
                    .find(|(key, _)| *key == switches)
                    .map(|(_, outputs)| outputs.clone())
            }) {
                return Ok(hit);
            }
            let outputs: Vec<OutputDecl> = crate::model::lane(&switches)?
                .outputs
                .iter()
                .map(output_decl)
                .collect();
            DECLARED.with_borrow_mut(|seen| seen.push((switches, outputs.clone())));
            Ok(outputs)
        })
        .collect()
}

/// An output as the program reads it.
fn output_decl(decl: &crate::model::OutputDecl) -> OutputDecl {
    use crate::model::InputAxis;
    OutputDecl {
        name: decl.name.clone(),
        axis: match decl.axis {
            InputAxis::Lane => Axis::Lane,
            InputAxis::Row => Axis::Row,
            InputAxis::Voxel => Axis::Voxel,
        },
        width: decl.width,
        dtype: match decl.dtype {
            crate::types::Dtype::I32 => eta_dsl::Dtype::I32,
            _ => eta_dsl::Dtype::F32,
        },
        layers: decl.layers,
        readout: decl.readout,
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

/// Even spans of `n` rows, each at most `max` (an output's `max-rows`).
pub fn readout_spans(n: u32, max: u32) -> Vec<(u32, u32)> {
    even_spans(n, max.max(1))
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

pub fn submit_frame(on: &Pipeline, slots: &[Option<&ForwardPass>]) -> Result<(), String> {
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
    let mut borrows: Vec<Option<&wit_forward::ForwardPass>> =
        slots.iter().map(|slot| slot.map(|p| &p.wit)).collect();
    borrows.resize(k, None);
    wit_forward::submit(&on.wit, &borrows)
}

pub async fn run_ahead(
    on: &Pipeline,
    pass: &ForwardPass,
    budget: usize,
    mut on_token: impl AsyncFnMut() -> Result<std::ops::ControlFlow<()>, String>,
) -> Result<usize, String> {
    use std::ops::ControlFlow;

    if budget == 0 {
        return Ok(0);
    }
    // Every slot of a frame is its own fire: the runtime advances the pass's
    // host shadow and its working sets' cursors (recurrent state included)
    // fire by fire, so a mask or a recurrent state fills the frame like any
    // decode. A pass off its model's default path (a canvas redrawn whole, a
    // drafter block) posts one fire at a time: no model served here runs one
    // under run-ahead, so its frame shape has never been measured.
    let r = if pass.switched() { 1 } else { frame_size() };
    let window_fires = crate::model::run_ahead_window() as usize;
    let window_frames = (window_fires / r.max(1)).max(1);

    let mut submitted = 0usize;
    let mut consumed = 0usize;

    let submit_one_frame = |submitted: &mut usize| -> Result<(), String> {
        let live = r.min(budget - *submitted);
        if live == 0 {
            return Ok(());
        }
        let slots: Vec<Option<&ForwardPass>> = vec![Some(pass); live];
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
        wit_forward::park(&self.wit);
    }

    /// How many pipelines are open on the engine, this one included.
    #[must_use]
    pub fn live() -> u32 {
        wit_pipeline::live()
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Pipeline::new()
    }
}

pub mod shared_prelude {
    pub use super::{
        Channel, Fold, Pipeline, StateGeometry, TOKEN_PAD, WorkingSet, channel_capacity,
        frame_size, kv_page_size, readout_spans,
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

impl ForwardPass {
    /// Slots at which the runtime keeps a recurrent state, so a later pass over
    /// the same tokens resumes there.
    pub fn checkpoints(&self, slots: &[u32]) -> Result<(), String> {
        self.wit.checkpoints(slots)
    }
}

pub mod prelude {
    pub use super::{ForwardPass, run_ahead, submit_frame};
    pub use crate::eta::shared_prelude::*;
}

pub mod diffusion {
    pub mod prelude {
        pub use super::{entropy_bound_accept, linear_temperature, stable_and_confident};
        pub use crate::eta::prelude::*;
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
