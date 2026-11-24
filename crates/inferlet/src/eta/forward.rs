use super::*;

type StageClosure = Box<dyn Fn()>;

/// A forward pass: its lanes' bindings and switches, and the program its fires run.
pub struct ForwardPass {
    pub(super) wit: wit_forward::ForwardPass,
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
    retractable: bool,
}

impl ForwardPass {
    /// Whether a switch is off its model's default; such a pass posts one fire at a time.
    pub(super) fn switched(&self) -> bool {
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
                retractable: false,
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
        let wit = ch.wit()?;
        self.wit.input(lane, name, wit.as_ref())?;
        self.inner
            .borrow_mut()
            .inputs
            .push((lane, name.to_string(), ch.dsl()));
        Ok(())
    }

    /// Puts lane `lane` in attention group `id`.
    /// A beam search's finish: once this pass's last fire has landed, its
    /// working set holds beam `beam`'s hypothesis at its slots
    /// `[folded, end)`, in its own order, and the other beams' pages are let
    /// go ([`crate::beam::BeamGroup::finish`]).
    pub fn adopt_beam(&self, beam: u32, end: u32) -> Result<(), String> {
        self.wit.adopt_beam(beam, end)
    }

    pub fn group(&self, lane: u32, id: u32) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        self.wit.group(lane, id)
    }

    /// Per row, what it attends past the frozen part: column `c` is slot
    /// `folded() + c` of its lane's working set, the buffer and then the
    /// pass's rows. Frozen slots are always read. Kv-less lanes: its group's
    /// rows.
    pub fn mask(&self, rows: &Channel) -> Result<(), String> {
        self.ensure_ports_available(&[Port::AttnMask])?;
        let wit = rows.wit()?;
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
        let token_wit = tokens.wit()?;
        let indptr_wit = indptr.wit()?;
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
            (Port::Parents, geom.parents.copied()),
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
            .map(|(_, channel)| channel.as_ref().map(Channel::wit).transpose())
            .collect::<Result<_, String>>()?;
        let handles: Vec<Rc<wit_ws::WorkingSet>> = lanes.iter().map(|ws| ws.ws.clone()).collect();
        let borrows: Vec<&wit_ws::WorkingSet> = handles.iter().map(Rc::as_ref).collect();
        self.wit.state(
            &borrows,
            &wit_forward::StateGeometry {
                positions: wits[0].as_deref(),
                slots: wits[1].as_deref(),
                kv_len: wits[2].as_deref(),
                fold: wits[3].as_deref(),
                parents: wits[4].as_deref(),
                parents_end: geom.parents_end,
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
        let indices_wit = indices.wit()?;
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

    /// Panics once the program is attached: an authoring error (a stage set
    /// after the first submit), not a host one.
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

    pub(super) fn attach_program(&self) -> Result<(), String> {
        if self.inner.borrow().program_attached {
            return Ok(());
        }
        state_final(self.inner.borrow().retractable, || self.wit.final_rows())?;
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
            .map(|gid| channel_handle(*gid))
            .collect::<Result<_, String>>()?;
        let borrows: Vec<&wit_channel::Channel> = handles.iter().map(Rc::as_ref).collect();
        let bytes = traced.encode();
        self.wit.program(&bytes, &borrows)?;
        drop(inner);
        self.inner.borrow_mut().program_attached = true;
        Ok(())
    }
}

/// States a pass's rows final through `final_rows` unless the program said
/// the pass is retractable: a pass is final by default, and
/// this is the program's own statement, made when its program is attached
/// because the runtime takes it before the program.
pub(super) fn state_final(
    retractable: bool,
    final_rows: impl FnOnce() -> Result<(), String>,
) -> Result<(), String> {
    if retractable {
        return Ok(());
    }
    final_rows().map_err(|error| {
        format!(
            "{error} (a pass that binds `parents`, or otherwise takes rows back, states \
             `ForwardPass::retractable`)"
        )
    })
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

impl ForwardPass {
    /// Slots at which the runtime keeps a recurrent state, so a later pass over
    /// the same tokens resumes there.
    pub fn checkpoints(&self, slots: &[u32]) -> Result<(), String> {
        self.wit.checkpoints(slots)
    }

    /// States that this pass may take back rows it sampled: the program
    /// retracts ([`Pipeline::retract`]), truncates, rolls back or rewinds
    /// its working sets below a row a fire of it sampled (a stop string
    /// cut in a set it continues, a draft or speculative verify, a beam, a
    /// rollback). Its run-ahead fires on a recurrent or windowed set then
    /// fold only what the program has read, so a stop can take the rest
    /// back, and [`run_ahead`] retracts the rows past a break.
    ///
    /// Without it a pass is final: the SDK states its
    /// sampled rows final when the program is attached, its lanes fold
    /// every row at fire and run the plain step, [`Pipeline::retract`] over
    /// one of its fires is refused by name, and [`run_ahead`] leaves its
    /// rows past a break in place. A pass binding `parents` (a beam) must
    /// state this. Set before the program is attached.
    pub fn retractable(&self) -> Result<(), String> {
        let mut inner = self.inner.borrow_mut();
        if inner.program_attached {
            return Err("forward pass program is already attached".to_string());
        }
        inner.retractable = true;
        Ok(())
    }

    /// Whether this pass's rows are final: [`Self::retractable`] was not
    /// stated.
    pub fn is_final(&self) -> bool {
        !self.inner.borrow().retractable
    }
}
