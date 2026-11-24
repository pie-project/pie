use super::*;

thread_local! {
    static WIT_CHANNELS: RefCell<HashMap<u64, Rc<wit_channel::Channel>>> = RefCell::new(HashMap::new());
    static DECLARED: RefCell<HashMap<u64, ChannelSpec>> = RefCell::new(HashMap::new());
    /// Channels whose seed the host refused, with its error: a constructor
    /// has no `Result` to return it in, so the channel's next fallible use
    /// (a pass binding it, a submit tracing it, a host take) returns it.
    static REFUSED: RefCell<HashMap<u64, String>> = RefCell::new(HashMap::new());
}

pub(super) fn refuse(gid: u64, why: String) {
    REFUSED.with(|m| {
        m.borrow_mut().entry(gid).or_insert(why);
    });
}

/// The channel's WIT handle, or the error the host refused its seed with.
pub(super) fn channel_handle(gid: u64) -> Result<Rc<wit_channel::Channel>, String> {
    if let Some(why) = REFUSED.with(|m| m.borrow().get(&gid).cloned()) {
        return Err(why);
    }
    lookup_channel(gid).ok_or_else(|| format!("channel {gid} was never declared"))
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

/// Panics on a dtype no channel holds: a guest program's authoring error at
/// the channel's construction, unreachable from any host error.
fn to_wit_dtype(d: Dtype) -> WitDtype {
    match d {
        Dtype::F32 => WitDtype::F32,
        Dtype::I32 => WitDtype::I32,
        Dtype::U32 => WitDtype::U32,
        Dtype::Bool => WitDtype::Bool,
        Dtype::Bf16 => panic!(
            "a channel holds f32, i32, u32 or bool: bf16 is a guest program's intermediate \
             dtype and has no WIT channel tag; cast a bf16 value to f32 before a put"
        ),
        other => panic!("{other:?} is not a dtype ETA computes in; it has no WIT tag"),
    }
}

fn dims_of(shape: Shape) -> Vec<u32> {
    shape.dims().to_vec()
}

pub(super) fn claim_port(port: Port, ch: &Channel) -> DslChannel {
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

    /// The channel as the program traces it: what [`crate::sampler`]'s
    /// stage bodies take and put.
    #[must_use]
    pub fn traced(&self) -> DslChannel {
        self.dsl()
    }

    pub(super) fn dsl(&self) -> DslChannel {
        // An invariant, not a host error: every `Channel` registers its gid
        // as it is built (`Channel::build`) and nothing in this crate
        // releases one, so the lookup cannot miss whatever the host did.
        DslChannel::by_gid(self.gid).expect("channel token resolves in the DSL registry")
    }

    pub(super) fn wit(&self) -> Result<Rc<wit_channel::Channel>, String> {
        channel_handle(self.gid)
    }

    /// Stages `bytes` as the channel's seed; a refusal is returned by the
    /// channel's next fallible use.
    fn seed(&self, bytes: &[u8]) {
        if let Err(why) = self.wit().and_then(|wit| wit.put(bytes)) {
            refuse(self.gid, format!("{}: {why}", self.host_label("seed")));
        }
    }

    /// Panics if the channel is already in use: an authoring error, not a
    /// host one.
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
        ch.seed(&data.bytes);
        ch
    }

    #[track_caller]
    pub fn seeded(shape: impl IntoShape, dtype: Dtype) -> Channel {
        Channel::build(shape.into_shape(), dtype, true)
    }

    /// Panics if `v` holds other than `shape`'s element count: an authoring
    /// error, not a host one.
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
        ch.seed(&data.bytes);
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
        let raw = self.wit()?.take().await;
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
        let wit = self.wit()?;
        crate::pie::inferlet::frames::Frames::from_channel(&wit, width, height, count, fps)
            .map_err(|why| format!("{}: {why}", self.host_label("take-frames")))
    }

    pub fn set_frames(&self, pixels: &crate::pie::inferlet::frames::Frames) -> Result<(), String> {
        let wit = self.wit()?;
        pixels
            .to_channel(&wit)
            .map_err(|why| format!("{}: {why}", self.host_label("set-frames")))
    }

    pub async fn read_host<T: FromChannel>(&self) -> Result<T, String> {
        self.check_host::<T>("read")?;
        self.dsl().note_host_read();
        let raw = self.wit()?.read().await;
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
                let _ = self.wit().and_then(|wit| wit.put(&data.bytes));
            }
        }
    }

    pub fn set(&self, v: impl IntoConst) -> Result<(), String> {
        let data: ConstData = v.into_const();
        self.wit()?.set(&data.bytes)
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
