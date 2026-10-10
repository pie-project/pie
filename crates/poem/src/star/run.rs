//! A deployment of a package traced and read: its layout evaluated, then the
//! caches and forward of `forward.poem` recorded into a trace, or the formats
//! of `formats.poem` read into a contract.

use std::cell::RefCell;

use checkpoint::contract::ModelContract;

use crate::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Trace, Value};
use starlark::environment::Module;
use starlark::eval::Evaluator;
use starlark::values::ValueLike;
use starlark::values::structs::AllocStruct;
use starlark::values::{Heap, Value as Star};

use crate::star::forward::{SpecHandle, TRAIL, Trail, ValueHandle, held, hold_input};
use crate::star::package::Package;
use crate::star::values::DtypeValue;

/// How a deployment reads a still: cut into `patch` × `patch` patches in
/// `block` × `block` blocks, its tokens placed by mrope or in sequence,
/// spelled between `prefix` and `suffix` as `placeholder` tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageSpec {
    pub patch: u32,
    pub block: u32,
    pub mrope: bool,
    pub prefix: String,
    pub placeholder: String,
    pub suffix: String,
}

impl ImageSpec {
    fn of<'v>(image: Star<'v>, heap: Heap<'v>) -> anyhow::Result<ImageSpec> {
        let field = |name: &str| -> anyhow::Result<Star<'v>> {
            image
                .get_attr(name, heap)
                .map_err(error)?
                .ok_or_else(|| anyhow::anyhow!("the image front end states no `{name}`"))
        };
        let text = |name: &str| -> anyhow::Result<String> {
            field(name)?
                .unpack_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("`{name}` is a string"))
        };
        let int = |name: &str| -> anyhow::Result<u32> {
            field(name)?
                .unpack_i32()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| anyhow::anyhow!("`{name}` is a count"))
        };
        Ok(ImageSpec {
            patch: int("patch")?,
            block: int("block")?,
            mrope: field("mrope")?.to_bool(),
            prefix: text("prefix")?,
            placeholder: text("placeholder")?,
            suffix: text("suffix")?,
        })
    }
}

/// A deployment as a package's layout reads it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Deploy {
    pub weights: Vec<Dtype>,
    pub kv: Dtype,
    pub tp: u32,
    pub parts: Vec<String>,
    pub drafter: Option<String>,
}

impl Deploy {
    fn alloc<'v>(&self, heap: Heap<'v>) -> Star<'v> {
        let weights: Vec<DtypeValue> = self.weights.iter().copied().map(DtypeValue).collect();
        heap.alloc(AllocStruct([
            ("weights", heap.alloc(weights)),
            ("kv", heap.alloc(DtypeValue(self.kv))),
            ("tp", heap.alloc(self.tp)),
            ("parts", heap.alloc(self.parts.clone())),
            (
                "drafter",
                match &self.drafter {
                    Some(d) => heap.alloc(d.as_str()),
                    None => Star::new_none(),
                },
            ),
        ]))
    }
}

fn error(e: starlark::Error) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

impl Package {
    /// The layout `id`'s deployment `deploy` takes, evaluated in `eval`.
    fn layout<'v>(
        &self,
        id: &str,
        deploy: &Deploy,
        module: &Module<'v>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<Star<'v>> {
        let layout = module
            .heap()
            .access_owned_frozen_value(&self.module("model.poem")?.get("layout")?);
        let heap = module.heap();
        eval.eval_function(layout, &[heap.alloc(id), deploy.alloc(heap)], &[])
            .map_err(error)
    }

    /// What `id`'s deployment `deploy` states of itself as a generative
    /// model, if its `model.poem` states it (`generative(m)`).
    pub fn generative(
        &self,
        id: &str,
        deploy: &Deploy,
    ) -> anyhow::Result<Option<crate::generative::Generative>> {
        self.stated(
            id,
            deploy,
            "generative",
            crate::star::generative::generative_of,
        )
    }

    /// The canvas `id`'s deployment `deploy` denoises as a text diffusion
    /// model, if its `model.poem` states one (`diffusion(m)`).
    pub fn diffusion(
        &self,
        id: &str,
        deploy: &Deploy,
    ) -> anyhow::Result<Option<crate::generative::Diffusion>> {
        self.stated(
            id,
            deploy,
            "diffusion",
            crate::star::generative::diffusion_of,
        )
    }

    fn stated<T>(
        &self,
        id: &str,
        deploy: &Deploy,
        function: &str,
        of: fn(Star<'_>) -> anyhow::Result<T>,
    ) -> anyhow::Result<Option<T>> {
        let Ok(stating) = self.module("model.poem")?.get(function) else {
            return Ok(None);
        };
        Module::with_temp_heap(|module| {
            let mut eval = Evaluator::new(&module);
            let m = self.layout(id, deploy, &module, &mut eval)?;
            let stating = module.heap().access_owned_frozen_value(&stating);
            let stated = eval.eval_function(stating, &[m], &[]).map_err(error)?;
            if stated.is_none() {
                return Ok(None);
            }
            of(stated).map(Some)
        })
    }

    /// How `id`'s deployment `deploy` reads a still, if its `model.poem`
    /// states a `media(id, deploy)` with an `image`.
    pub fn image(&self, id: &str, deploy: &Deploy) -> anyhow::Result<Option<ImageSpec>> {
        self.with_image(id, deploy, |_, heap, image| ImageSpec::of(image, heap))
    }

    /// The size a `h` × `w` still is framed to before `id`'s deployment
    /// `deploy` reads it, under `budget` (`still` or `video`).
    pub fn frame(
        &self,
        id: &str,
        deploy: &Deploy,
        h: u32,
        w: u32,
        budget: &str,
    ) -> anyhow::Result<(u32, u32)> {
        self.with_image(id, deploy, |eval, heap, image| {
            let frame = image
                .get_attr("frame", heap)
                .map_err(error)?
                .ok_or_else(|| anyhow::anyhow!("the image front end states no `frame`"))?;
            let framed = eval
                .eval_function(
                    frame,
                    &[heap.alloc(h), heap.alloc(w), heap.alloc(budget)],
                    &[],
                )
                .map_err(error)?;
            let (fh, fw): (u32, u32) = starlark::values::UnpackValue::unpack_value_err(framed)
                .map_err(|e| anyhow::anyhow!("`frame` returns a `(height, width)` pair: {e}"))?;
            Ok((fh, fw))
        })?
        .ok_or_else(|| anyhow::anyhow!("`{id}` reads no stills"))
    }

    fn with_image<T>(
        &self,
        id: &str,
        deploy: &Deploy,
        f: impl for<'v> FnOnce(&mut Evaluator<'v, '_, '_>, Heap<'v>, Star<'v>) -> anyhow::Result<T>,
    ) -> anyhow::Result<Option<T>> {
        let Ok(media) = self.module("model.poem")?.get("media") else {
            return Ok(None);
        };
        Module::with_temp_heap(|module| {
            let mut eval = Evaluator::new(&module);
            let heap = module.heap();
            let media = heap.access_owned_frozen_value(&media);
            let stated = eval
                .eval_function(media, &[heap.alloc(id), deploy.alloc(heap)], &[])
                .map_err(error)?;
            if stated.is_none() {
                return Ok(None);
            }
            let Some(image) = stated.get_attr("image", heap).map_err(error)? else {
                return Ok(None);
            };
            if image.is_none() {
                return Ok(None);
            }
            f(&mut eval, heap, image).map(Some)
        })
    }

    /// The trace of `id`'s deployment `deploy`, named `name`, on `platform`.
    pub fn trace(
        &self,
        id: &str,
        deploy: &Deploy,
        name: &str,
        platform: Platform,
    ) -> anyhow::Result<Trace> {
        Module::with_temp_heap(|module| {
            let mut eval = Evaluator::new(&module);
            let m = self.layout(id, deploy, &module, &mut eval)?;
            let forward = self.module("forward.poem")?;
            let caches = module
                .heap()
                .access_owned_frozen_value(&forward.get("caches")?);
            let run = module
                .heap()
                .access_owned_frozen_value(&forward.get("forward")?);

            TRAIL.with(|t| *t.borrow_mut() = Trail::default());
            let declared = eval
                .eval_function(caches, &[m, module.heap().alloc(SpecHandle)], &[])
                .map_err(error);
            let spec = TRAIL
                .with(|t| t.borrow_mut().spec.take())
                .unwrap_or_default();
            declared?;

            let model = Model {
                spec,
                eval: RefCell::new(&mut eval),
                run,
                m,
                failed: RefCell::new(None),
            };
            let traced = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::trace_hybrid(name, &model, platform)
            }));
            TRAIL.with(|t| *t.borrow_mut() = Trail::default());
            if let Some(why) = model.failed.borrow_mut().take() {
                return Err(why);
            }
            traced.map_err(|panic| {
                anyhow::anyhow!(
                    "`{name}` does not trace: {}",
                    crate::star::forward::panic_message(&*panic)
                )
            })
        })
    }
}

/// A package's forward, as the DSL traces a model.
struct Model<'a, 'v, 'e> {
    spec: HybridSpec,
    eval: RefCell<&'a mut Evaluator<'v, 'e, 'e>>,
    run: Star<'v>,
    m: Star<'v>,
    failed: RefCell<Option<anyhow::Error>>,
}

impl ForwardHybrid for Model<'_, '_, '_> {
    fn caches(&self) -> HybridSpec {
        self.spec.clone()
    }

    fn forward(&self, inputs: Input) -> Value {
        let mut eval = self.eval.borrow_mut();
        let handle = hold_input(inputs);
        let heap = eval.heap();
        let out = eval.eval_function(self.run, &[self.m, heap.alloc(handle)], &[]);
        let value = match out {
            Ok(out) => match out.downcast_ref::<ValueHandle>() {
                Some(h) => Ok(held(*h)),
                None => Err(anyhow::anyhow!(
                    "`forward` returned {}, not the value its rows read out",
                    out.get_type()
                )),
            },
            Err(e) => Err(error(e)),
        };
        // The trail's values hold the trace, which finishes once this returns.
        TRAIL.with(|t| {
            let mut t = t.borrow_mut();
            t.values.clear();
            t.inputs.clear();
        });
        match value {
            Ok(value) => value,
            Err(why) => {
                let said = why.to_string();
                *self.failed.borrow_mut() = Some(why);
                panic!("{said}")
            }
        }
    }
}

impl Package {
    /// The contract that reads `src` into `id`'s deployment `deploy`, by the
    /// one format of `formats.poem` that recognizes it.
    pub fn import(
        &self,
        id: &str,
        deploy: &Deploy,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<checkpoint::contract::ModelContract, crate::import::Error> {
        let name = self.name();
        self.formats(id, deploy, src, platform, |built| {
            crate::import::format::read_one(name, src, built)
        })?
    }

    /// `then` of the formats of `formats.poem`, each reading `src` into
    /// `id`'s deployment `deploy`.
    fn formats<R>(
        &self,
        id: &str,
        deploy: &Deploy,
        src: &ztensor::Source,
        platform: Platform,
        then: impl FnOnce(Vec<crate::import::format::Format<'_, ModelContract>>) -> R,
    ) -> Result<R, crate::import::Error> {
        use crate::import::format::Format;

        let illegible = |detail: String| crate::import::Error::Illegible {
            name: id.to_string(),
            detail,
        };
        /// The checkpoint a format reads, held while this import runs.
        struct Held;
        impl Drop for Held {
            fn drop(&mut self) {
                crate::star::formats::SOURCE.with(|s| *s.borrow_mut() = None);
            }
        }
        crate::star::formats::SOURCE
            .with(|s| *s.borrow_mut() = Some(crate::star::formats::Snapshot::of(src)));
        let _held = Held;
        Module::with_temp_heap(|module| {
            let mut eval = Evaluator::new(&module);
            let heap = module.heap();
            let m = self
                .layout(id, deploy, &module, &mut eval)
                .map_err(|e| illegible(format!("{e:#}")))?;
            let formats = self
                .module("formats.poem")
                .and_then(|f| f.get("formats"))
                .map_err(|e| illegible(format!("{e:#}")))?;
            let formats = heap.access_owned_frozen_value(&formats);
            let listed = eval
                .eval_function(formats, &[m], &[])
                .map_err(|e| illegible(format!("{e}")))?;
            let listed: Vec<Star<'_>> = listed
                .iterate(heap)
                .map_err(|e| illegible(format!("`formats` returned no list: {e}")))?
                .collect();

            let eval = RefCell::new(eval);
            let mut built = Vec::new();
            for format in listed {
                let field = |name: &str| {
                    format
                        .get_attr(name, heap)
                        .ok()
                        .flatten()
                        .ok_or_else(|| illegible(format!("a format states no `{name}`")))
                };
                let name: String = field("name")?
                    .unpack_str()
                    .ok_or_else(|| illegible("a format's name is text".to_string()))?
                    .to_string();
                let read = field("read")?;
                let recognizes = field("recognizes")?;
                let states: Vec<crate::import::format::Stated> = field("states")?
                    .iterate(heap)
                    .map_err(|e| illegible(format!("{e}")))?
                    .map(|s| {
                        s.downcast_ref::<crate::star::formats::StatedValue>()
                            .map(|s| s.0.clone())
                            .ok_or_else(|| {
                                illegible(format!(
                                    "a format states {}, not a stated value",
                                    s.get_type()
                                ))
                            })
                    })
                    .collect::<Result<_, _>>()?;
                let eval = &eval;
                let said = name.clone();
                let reads =
                    move || -> Result<checkpoint::contract::ModelContract, crate::import::Error> {
                        crate::star::formats::READS.with(|r| r.borrow_mut().clear());
                        crate::star::formats::missed();
                        eval.borrow_mut()
                            .eval_function(
                                read,
                                &[heap.alloc(crate::star::formats::ReadsHandle)],
                                &[],
                            )
                            .map_err(|e| match crate::star::formats::missed() {
                                Some(name) => crate::import::Error::Missing(name),
                                None => crate::import::Error::Illegible {
                                    name: said.clone(),
                                    detail: format!("{e}"),
                                },
                            })?;
                        let reads = crate::star::formats::READS
                            .with(|r| std::mem::take(&mut *r.borrow_mut()));
                        let mut b = crate::import::Builder::new(src, 1, platform);
                        for read in reads {
                            read.onto(&mut b)?;
                        }
                        Ok(b.build())
                    };
                let format = if recognizes.is_none() {
                    Format::reading(name, reads)
                } else {
                    Format::new(
                        name,
                        move |_src| {
                            eval.borrow_mut()
                                .eval_function(
                                    recognizes,
                                    &[heap.alloc(crate::star::formats::SourceHandle)],
                                    &[],
                                )
                                .ok()
                                .and_then(|v| v.unpack_bool())
                                .unwrap_or(false)
                        },
                        reads,
                    )
                };
                built.push(format.stating(states));
            }
            Ok(then(built))
        })
    }
}
