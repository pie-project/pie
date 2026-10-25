//! What a layout may state of a generative model: its latent space,
//! schedule.

use std::fmt;

use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::environment::GlobalsBuilder;
use starlark::starlark_simple_value;
use starlark::values::ValueLike;
use starlark::values::float::UnpackFloat;
use starlark::values::list::UnpackList;
use starlark::values::none::NoneOr;
use starlark::values::tuple::UnpackTuple;
use starlark::values::{NoSerialize, StarlarkValue, UnpackValue, Value};
use starlark_derive::{starlark_module, starlark_value};

use crate::generative::{Generative, LatentSpace, ScheduleFact, ScheduleKind};

/// A fact of a generative model a layout states.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, Allocative)]
pub(crate) struct Stated(#[allocative(skip)] pub(crate) Fact);

#[derive(Debug, Clone)]
pub(crate) enum Fact {
    Latent(LatentSpace),
    Schedule(ScheduleFact),
    Generative(Generative),
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

fact_of!(latent_of, Latent, LatentSpace, "a latent space");
fact_of!(schedule_of, Schedule, ScheduleFact, "a schedule");
fact_of!(pub(crate) generative_of, Generative, Generative, "a generative model's facts");

use crate::star::bind::word as spelled;

#[starlark_module]
pub(crate) fn generative(builder: &mut GlobalsBuilder) {
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
                    Ok((s.to_string(), x.0 as f32))
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

    /// What a generative model states of itself: the latent space and
    /// schedule it denoises by.
    fn generation(
        #[starlark(require = named, default = NoneOr::None)] latent: NoneOr<Value<'_>>,
        #[starlark(require = named, default = NoneOr::None)] schedule: NoneOr<Value<'_>>,
    ) -> anyhow::Result<Stated> {
        Ok(Stated(Fact::Generative(Generative {
            latent: latent.into_option().map(latent_of).transpose()?,
            schedule: schedule.into_option().map(schedule_of).transpose()?,
        })))
    }
}
