//! What a layout may state of a generative model: its readings, the ports
//! each reads, its positions, latent space and schedule; and the canvas a
//! text diffusion model denoises.

use std::collections::HashSet;
use std::fmt;
use std::sync::{LazyLock, Mutex};

use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::environment::GlobalsBuilder;
use starlark::starlark_simple_value;
use starlark::values::float::UnpackFloat;
use starlark::values::list::UnpackList;
use starlark::values::none::NoneOr;
use starlark::values::tuple::UnpackTuple;
use starlark::values::{
    NoSerialize, StarlarkPagableUnsupported, StarlarkValue, UnpackValue, Value,
};
use starlark_derive::{starlark_module, starlark_value};

use crate::Stream;
use crate::generative::{
    AxisRole, Diffusion, Generative, LatentSpace, PortFact, PortKind, PositionConvention,
    ReadingFact, ReadoutKind, ScheduleFact, ScheduleKind,
};

/// A fact of a generative model a layout states.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub(crate) struct Stated(#[allocative(skip)] pub(crate) Fact);

#[derive(Debug, Clone)]
pub(crate) enum Fact {
    Reading(ReadingFact),
    Port(PortFact),
    Positions(PositionConvention),
    Latent(LatentSpace),
    Schedule(ScheduleFact),
    Generative(Generative),
    Diffusion(Diffusion),
}

starlark_simple_value!(Stated);

impl fmt::Display for Stated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[starlark_value(type = "generative_fact")]
impl<'v> StarlarkValue<'v> for Stated {}

macro_rules! fact_of {
    ($vis:vis $fn:ident, $variant:ident, $t:ty, $what:literal) => {
        $vis fn $fn(v: Value<'_>) -> anyhow::Result<$t> {
            match v.downcast_ref::<Stated>() {
                Some(Stated(Fact::$variant(t))) => Ok(t.clone()),
                _ => Err(anyhow::anyhow!(
                    concat!($what, " was wanted, not {}"),
                    v.get_type()
                )),
            }
        }
    };
}

fact_of!(reading_of, Reading, ReadingFact, "a reading");
fact_of!(port_of, Port, PortFact, "a port");
fact_of!(
    positions_of,
    Positions,
    PositionConvention,
    "a position convention"
);
fact_of!(latent_of, Latent, LatentSpace, "a latent space");
fact_of!(schedule_of, Schedule, ScheduleFact, "a schedule");
fact_of!(pub(crate) generative_of, Generative, Generative, "a generative model's facts");
fact_of!(pub(crate) diffusion_of, Diffusion, Diffusion, "a canvas");

/// `name`, held for as long as the process runs: the facts name readings and
/// ports as the runtime's tables do, once each.
fn interned(name: &str) -> &'static str {
    static NAMES: LazyLock<Mutex<HashSet<&'static str>>> = LazyLock::new(Mutex::default);
    let mut names = NAMES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(held) = names.get(name) {
        return held;
    }
    let held: &'static str = Box::leak(name.to_string().into_boxed_str());
    names.insert(held);
    held
}

use crate::star::bind::word as spelled;

/// The stream `word` names.
pub(crate) fn stream(word: &str) -> anyhow::Result<Stream> {
    spelled(
        word,
        "a stream",
        &[
            ("text", Stream::Text),
            ("image", Stream::Image),
            ("video", Stream::Video),
            ("audio", Stream::Audio),
            ("context", Stream::Context),
            ("reference", Stream::Reference),
        ],
    )
}

fn streams(words: UnpackList<String>) -> anyhow::Result<Vec<Stream>> {
    words.items.iter().map(|w| stream(w)).collect()
}

#[starlark_module]
pub(crate) fn generative(builder: &mut GlobalsBuilder) {
    /// The reading `name`, run by passes of index `index`: the streams its
    /// rows are of, the ports it reads, how its positions are laid out and
    /// what it reads out, `readout_width` wide.
    fn reading(
        #[starlark(require = pos)] name: &str,
        #[starlark(require = named)] index: u32,
        #[starlark(require = named)] streams: UnpackList<String>,
        #[starlark(require = named)] readout: &str,
        #[starlark(require = named)] readout_width: u32,
        #[starlark(require = named, default = false)] has_kv: bool,
        #[starlark(require = named, default = false)] takes_tokens: bool,
        #[starlark(require = named, default = UnpackList::default())] ports: UnpackList<Value<'_>>,
        #[starlark(require = named, default = NoneOr::None)] positions: NoneOr<Value<'_>>,
    ) -> anyhow::Result<Stated> {
        Ok(Stated(Fact::Reading(ReadingFact {
            name: interned(name),
            index: u8::try_from(index)
                .map_err(|_| anyhow::anyhow!("reading {index} is past a u8"))?,
            has_kv,
            takes_tokens,
            streams: self::streams(streams)?,
            ports: ports
                .items
                .into_iter()
                .map(port_of)
                .collect::<anyhow::Result<_>>()?,
            positions: positions.into_option().map(positions_of).transpose()?,
            readout: spelled(
                readout,
                "a readout",
                &[
                    ("logits", ReadoutKind::Logits),
                    ("velocity", ReadoutKind::Velocity),
                    ("hidden", ReadoutKind::Hidden),
                    ("pixels", ReadoutKind::Pixels),
                ],
            )?,
            readout_width,
        })))
    }

    /// The port `name`, of `kind`, `width` wide, its rows of `streams`; `at`
    /// pins its index among the ports of its kind, `rows` its row count.
    fn port(
        #[starlark(require = pos)] name: &str,
        #[starlark(require = pos)] kind: &str,
        #[starlark(require = pos)] width: u32,
        #[starlark(require = pos)] streams: UnpackList<String>,
        #[starlark(require = named, default = NoneOr::None)] at: NoneOr<u32>,
        #[starlark(require = named, default = NoneOr::None)] rows: NoneOr<u32>,
    ) -> anyhow::Result<Stated> {
        Ok(Stated(Fact::Port(PortFact {
            name: interned(name),
            kind: spelled(
                kind,
                "a port's kind",
                &[
                    ("latents", PortKind::Latents),
                    ("lane_vector", PortKind::LaneVector),
                    ("context", PortKind::Context),
                    ("axis_positions", PortKind::AxisPositions),
                    ("voxels", PortKind::Voxels),
                ],
            )?,
            width,
            streams: self::streams(streams)?,
            at: at
                .into_option()
                .map(u8::try_from)
                .transpose()
                .map_err(|_| anyhow::anyhow!("a port's index is past a u8"))?,
            rows: rows.into_option(),
        })))
    }

    /// How a reading's positions are laid out over `axes`.
    fn positions(
        #[starlark(require = named)] axes: UnpackList<String>,
        #[starlark(require = named)] text_axis: u32,
        #[starlark(require = named)] text_origin: u32,
        #[starlark(require = named)] image_follows_text: bool,
        #[starlark(require = named, default = NoneOr::None)] reference_stride: NoneOr<u32>,
    ) -> anyhow::Result<Stated> {
        let axes = axes
            .items
            .iter()
            .map(|a| {
                spelled(
                    a,
                    "an axis",
                    &[
                        ("time", AxisRole::Time),
                        ("height", AxisRole::Height),
                        ("width", AxisRole::Width),
                        ("index", AxisRole::Index),
                    ],
                )
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Stated(Fact::Positions(PositionConvention {
            axes,
            text_axis,
            text_origin,
            image_follows_text,
            reference_stride: reference_stride.into_option(),
        })))
    }

    /// The latent space a model denoises in.
    fn latent_space(
        #[starlark(require = named)] channels: u32,
        #[starlark(require = named)] patch_t: u32,
        #[starlark(require = named)] patch_h: u32,
        #[starlark(require = named)] patch_w: u32,
        #[starlark(require = named)] spatial_compression: u32,
        #[starlark(require = named)] temporal_compression: u32,
    ) -> anyhow::Result<Stated> {
        Ok(Stated(Fact::Latent(LatentSpace {
            channels,
            patch_t,
            patch_h,
            patch_w,
            spatial_compression,
            temporal_compression,
        })))
    }

    /// The noise schedule a model was trained on.
    fn schedule(
        #[starlark(require = pos)] kind: &str,
        #[starlark(require = named)] shift: UnpackFloat,
        #[starlark(require = named)] train_steps: u32,
        #[starlark(require = named, default = UnpackList::default())] pinned_sigmas: UnpackList<
            UnpackFloat,
        >,
        #[starlark(require = named, default = UnpackList::default())] stream_shifts: UnpackList<
            UnpackTuple<Value<'_>>,
        >,
    ) -> anyhow::Result<Stated> {
        let stream_shifts = stream_shifts
            .items
            .into_iter()
            .map(|pair| match pair.items[..] {
                [s, x] => {
                    let s = s
                        .unpack_str()
                        .ok_or_else(|| anyhow::anyhow!("a stream is named"))?;
                    let x = UnpackFloat::unpack_value_err(x).map_err(|e| anyhow::anyhow!("{e}"))?;
                    Ok((stream(s)?, x.0 as f32))
                }
                _ => Err(anyhow::anyhow!(
                    "a stream's shift is a (stream, shift) pair"
                )),
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Stated(Fact::Schedule(ScheduleFact {
            kind: spelled(
                kind,
                "a schedule",
                &[
                    ("flow", ScheduleKind::Flow),
                    ("epsilon", ScheduleKind::Epsilon),
                    ("v", ScheduleKind::V),
                ],
            )?,
            shift: shift.0 as f32,
            train_steps,
            boundary: None,
            pinned_sigmas: pinned_sigmas
                .items
                .into_iter()
                .map(|s| s.0 as f32)
                .collect(),
            stream_shifts,
        })))
    }

    /// What a generative model states of itself: its readings, the latent
    /// space and schedule it denoises by, and the rows a pass may hold.
    fn generation(
        #[starlark(require = named)] readings: UnpackList<Value<'_>>,
        #[starlark(require = named)] max_rows: u32,
        #[starlark(require = named, default = NoneOr::None)] latent: NoneOr<Value<'_>>,
        #[starlark(require = named, default = NoneOr::None)] schedule: NoneOr<Value<'_>>,
    ) -> anyhow::Result<Stated> {
        Ok(Stated(Fact::Generative(Generative {
            readings: readings
                .items
                .into_iter()
                .map(reading_of)
                .collect::<anyhow::Result<_>>()?,
            latent: latent.into_option().map(latent_of).transpose()?,
            schedule: schedule.into_option().map(schedule_of).transpose()?,
            max_rows,
        })))
    }

    /// The canvas a text diffusion model denoises: `canvas` rows of a
    /// `hidden`-wide state, with `self_cond_taps` taps of self-conditioning.
    fn canvas(
        #[starlark(require = named)] canvas: u32,
        #[starlark(require = named)] hidden: u32,
        #[starlark(require = named)] self_cond_taps: u32,
    ) -> anyhow::Result<Stated> {
        Ok(Stated(Fact::Diffusion(Diffusion {
            canvas,
            hidden,
            self_cond_taps,
        })))
    }
}
