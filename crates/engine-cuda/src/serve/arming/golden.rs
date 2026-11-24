//! The golden arm: a body's replay against its eager walk, and the evidence
//! it names where they part.

#[allow(clippy::wildcard_imports)]
use super::*;

impl Shell {
    pub(super) fn golden(&mut self, key: &record::BodyKey, owned: &[Synthetic]) -> Result<()> {
        use crate::serve::Golden;

        let refused = |why: String| Fault::Golden {
            key: key.to_string(),
            why: format!(
                "lanes=[{}] {why}",
                owned
                    .iter()
                    .map(|lane| format!("{:#x}/{}r{}", lane.word, lane.tokens.len(), lane.slot))
                    .collect::<Vec<String>>()
                    .join(" "),
            ),
        };
        let slots: Vec<u32> = owned.iter().map(|lane| lane.slot).collect();
        let fire = |shell: &mut Shell, arm: Golden| -> Result<Vec<Vec<f32>>> {
            for &slot in &slots {
                shell.open(slot)?;
            }
            let out = shell.fire_synthetic_as(owned, arm)?;
            shell.device.synchronize()?;
            Ok(out)
        };
        if crate::record::ptr_traced(key) {
            crate::record::PTR_TAG.store(2, std::sync::atomic::Ordering::Relaxed);
        }
        let walked = fire(self, Golden::Eager);
        crate::record::PTR_TAG.store(0, std::sync::atomic::Ordering::Relaxed);
        let walked =
            walked.map_err(|fault| refused(format!("the control arm would not fire: {fault}")))?;
        let replayed = fire(self, Golden::Body)
            .map_err(|fault| refused(format!("the body arm would not fire: {fault}")))?;
        match evidence(&walked, &replayed) {
            None => Ok(()),
            Some(why) => {
                if super::super::diag::on().golden_probe {
                    let walked_again = fire(self, Golden::Eager);
                    let replayed_again = fire(self, Golden::Body);
                    let same = |a: &Vec<Vec<f32>>, b: &Vec<Vec<f32>>| {
                        evidence(a, b).map_or("identical".to_string(), |why| {
                            why.split("  lanes=").next().unwrap_or(&why).to_string()
                        })
                    };
                    if let Ok(w2) = &walked_again {
                        eprintln!("[golden-probe] {key} walk vs walk: {}", same(&walked, w2));
                    }
                    if let Ok(b2) = &replayed_again {
                        eprintln!("[golden-probe] {key} body vs body: {}", same(&replayed, b2));
                    }
                    let script = self.cache.body_script(key);
                    let execs = script.iter().filter(|(island, ..)| !island).count();
                    let mut culprit = None;
                    for k in 1..=execs {
                        crate::record::REPLAY_UPTO.store(k, std::sync::atomic::Ordering::Relaxed);
                        let partial = fire(self, Golden::Body);
                        crate::record::REPLAY_UPTO
                            .store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
                        match partial {
                            Ok(out) => {
                                let agrees = evidence(&walked, &out).is_none();
                                eprintln!(
                                    "[golden-probe] {key} replay first {k} exec(s): {}",
                                    if agrees {
                                        "agrees with the walk"
                                    } else {
                                        "DIFFERS"
                                    }
                                );
                                if !agrees {
                                    culprit = Some(k);
                                    break;
                                }
                            }
                            Err(fault) => {
                                eprintln!(
                                    "[golden-probe] {key} replay first {k} exec(s): would not fire: {fault}"
                                );
                                break;
                            }
                        }
                    }
                    if let Some(k) = culprit {
                        let mut seen = 0usize;
                        for (at_step, (island, from, upto)) in script.iter().enumerate() {
                            if *island {
                                continue;
                            }
                            seen += 1;
                            if seen == k {
                                eprintln!(
                                    "[golden-probe] {key} first disagreeing exec is step {at_step}: regions {from}..{upto}"
                                );
                                for region in *from..*upto {
                                    if let Some(template) =
                                        self.compiled.template().get(region as usize)
                                    {
                                        eprintln!(
                                            "[golden-probe]   region {region}: nodes {:?} phase {:?} stream {} lowering {:?}",
                                            template.nodes,
                                            template.phase,
                                            template.stream,
                                            template.lowering
                                        );
                                        for node in template.nodes.clone() {
                                            if let Some(held) = self.trace.nodes.get(node as usize)
                                            {
                                                let op = format!("{:?}", held.op);
                                                let head: String = op.chars().take(90).collect();
                                                eprintln!(
                                                    "[golden-probe]     node {node} layer {:?}: {head}",
                                                    held.layer
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let mut nth = 0usize;
                    for (at_step, (island, from, upto)) in script.iter().enumerate() {
                        if *island {
                            continue;
                        }
                        let j = nth;
                        nth += 1;
                        crate::record::REPLAY_FROM.store(j, std::sync::atomic::Ordering::Relaxed);
                        crate::record::REPLAY_UPTO
                            .store(j + 1, std::sync::atomic::Ordering::Relaxed);
                        let alone = fire(self, Golden::Body);
                        crate::record::REPLAY_FROM.store(0, std::sync::atomic::Ordering::Relaxed);
                        crate::record::REPLAY_UPTO
                            .store(usize::MAX, std::sync::atomic::Ordering::Relaxed);
                        let verdict = match &alone {
                            Ok(out) => match evidence(&walked, out) {
                                None => "agrees".to_string(),
                                Some(why) => format!(
                                    "DIFFERS ({})",
                                    why.chars().take(120).collect::<String>()
                                ),
                            },
                            Err(fault) => format!("would not fire: {fault}"),
                        };
                        let nodes = self
                            .compiled
                            .template()
                            .get(*from as usize)
                            .map(|t| t.nodes.start)
                            .unwrap_or(0);
                        let nodes_end = self
                            .compiled
                            .template()
                            .get((*upto as usize).saturating_sub(1))
                            .map(|t| t.nodes.end)
                            .unwrap_or(0);
                        let layers: Vec<Option<u32>> = {
                            let mut seen = Vec::new();
                            for n in nodes..nodes_end {
                                if let Some(h) = self.trace.nodes.get(n as usize)
                                    && seen.last() != Some(&h.layer)
                                {
                                    seen.push(h.layer);
                                }
                            }
                            seen
                        };
                        eprintln!(
                            "[golden-probe] {key} exec {j} alone (step {at_step}, regions {from}..{upto}, nodes {nodes}..{nodes_end}, layers {layers:?}): {verdict}"
                        );
                    }
                    for (i, body_lane) in replayed.iter().enumerate() {
                        let best = walked
                            .iter()
                            .enumerate()
                            .map(|(j, walk_lane)| {
                                let agree = body_lane
                                    .iter()
                                    .zip(walk_lane)
                                    .filter(|(x, y)| x.to_bits() == y.to_bits())
                                    .count();
                                (agree, j)
                            })
                            .max()
                            .unwrap_or((0, 0));
                        eprintln!(
                            "[golden-probe] {key} body lane {i} best matches walk lane {} ({} of {} cells bit-exact); head body={:?} walk[{}]={:?} walk[{i}]={:?}",
                            best.1,
                            best.0,
                            body_lane.len(),
                            &body_lane[..body_lane.len().min(4)],
                            best.1,
                            &walked[best.1][..walked[best.1].len().min(4)],
                            &walked[i][..walked[i].len().min(4)],
                        );
                        if i >= 7 {
                            break;
                        }
                    }
                }
                Err(refused(why))
            }
        }
    }

    pub(super) fn golden_real(&mut self, key: &record::BodyKey, owned: &[Synthetic]) -> Result<()> {
        use crate::serve::Golden;

        let context = self.pools.paging().context();
        if owned.is_empty()
            || owned.len() > self.held.len()
            || owned
                .iter()
                .any(|lane| lane.tokens.is_empty() || 2 * lane.tokens.len() > context as usize)
        {
            return Ok(());
        }
        let row = u32::from(self.adapters().slots().seats() >= 2);
        let real: Vec<Synthetic> = owned
            .iter()
            .map(|lane| {
                let rows = lane.tokens.len() as u32;
                let have = rows.max(1);
                let extent = have + rows;
                Synthetic {
                    tokens: (0..rows).map(|at| 1 + at % 16).collect(),
                    mask: lane.mask.as_ref().map(|_| {
                        Masking::Rows(
                            (0..rows)
                                .map(|at| {
                                    let allowed = have + at + 1;
                                    let mut runs = vec![0, allowed];
                                    if extent > allowed {
                                        runs.push(extent - allowed);
                                    }
                                    Mask::new(runs, u64::from(extent))
                                })
                                .collect(),
                        )
                    }),
                    adapter: lane.adapter.map(|_| row),
                    held: Some(have),
                    ..lane.clone()
                }
            })
            .collect();
        let refused = |why: String| Fault::Golden {
            key: key.to_string(),
            why: format!(
                "realistic pair: lanes=[{}] {why}",
                real.iter()
                    .map(|lane| format!("{:#x}/{}r{}", lane.word, lane.tokens.len(), lane.slot))
                    .collect::<Vec<String>>()
                    .join(" "),
            ),
        };
        let slots: Vec<u32> = owned.iter().map(|lane| lane.slot).collect();
        let fire = |shell: &mut Shell, arm: Golden| -> Result<Vec<Vec<f32>>> {
            for &slot in &slots {
                shell.open(slot)?;
            }
            shell.fire_synthetic_as(owned, Golden::Eager)?;
            let out = shell.fire_synthetic_as(&real, arm)?;
            shell.device.synchronize()?;
            Ok(out)
        };
        let Ok(walked) = fire(self, Golden::Eager) else {
            return Ok(());
        };
        let replayed = fire(self, Golden::Body)
            .map_err(|fault| refused(format!("the body arm would not fire: {fault}")))?;
        match evidence(&walked, &replayed) {
            None => Ok(()),
            Some(why) => Err(refused(why)),
        }
    }
}

pub(super) fn evidence(walked: &[Vec<f32>], replayed: &[Vec<f32>]) -> Option<String> {
    fn ordered(bits: u32) -> u32 {
        if bits & 0x8000_0000 != 0 {
            !bits
        } else {
            bits | 0x8000_0000
        }
    }

    if walked.len() != replayed.len() {
        return Some(format!(
            "the two arms answered {} and {} lanes  class=structural",
            walked.len(),
            replayed.len(),
        ));
    }
    let mut first: Option<(usize, f32, f32)> = None;
    let mut differing = 0usize;
    let mut total = 0usize;
    let mut worst = 0u32;
    let mut worst_bf16 = 0u32;
    let mut worst_bf16_at: Option<(usize, usize, f32, f32)> = None;
    let mut beyond_one_bf16 = 0usize;
    let mut beyond_eight_bf16 = 0usize;
    let mut non_finite = (0usize, 0usize);
    let mut per_lane: Vec<usize> = Vec::with_capacity(walked.len());
    for (lane, (a, b)) in walked.iter().zip(replayed).enumerate() {
        per_lane.push(0);
        if a.len() != b.len() {
            return Some(format!(
                "lane {lane} answered {} and {} elements  class=structural",
                a.len(),
                b.len(),
            ));
        }
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            total += 1;
            if x.to_bits() == y.to_bits() {
                continue;
            }
            differing += 1;
            per_lane[lane] += 1;
            non_finite.0 += usize::from(!x.is_finite());
            non_finite.1 += usize::from(!y.is_finite());
            worst = worst.max(ordered(x.to_bits()).abs_diff(ordered(y.to_bits())));
            let steps = ordered(x.to_bits() & 0xffff_0000)
                .abs_diff(ordered(y.to_bits() & 0xffff_0000))
                >> 16;
            beyond_one_bf16 += usize::from(steps > 1);
            beyond_eight_bf16 += usize::from(steps > 8);
            if steps > worst_bf16 {
                worst_bf16 = steps;
                worst_bf16_at = Some((lane, at, *x, *y));
            }
            if first.is_none() {
                first = Some((at, *x, *y));
            }
        }
    }
    let (at, x, y) = first?;
    let class = if worst <= 2 { "numeric" } else { "structural" };
    let lane_runs = {
        let mut runs: Vec<String> = Vec::new();
        let mut start = 0usize;
        while start < per_lane.len() {
            let differs = per_lane[start] > 0;
            let mut end = start;
            while end < per_lane.len() && (per_lane[end] > 0) == differs {
                end += 1;
            }
            let width = walked[start].len().max(1);
            let mean = per_lane[start..end].iter().sum::<usize>() as f64
                / ((end - start) as f64 * width as f64);
            runs.push(if differs {
                format!("{start}..{end} differ ({:.0}% of cells)", mean * 100.0)
            } else {
                format!("{start}..{end} agree")
            });
            start = end;
        }
        runs.join(", ")
    };
    let worst_bf16_line = worst_bf16_at.map_or(String::new(), |(lane, at, x, y)| {
        format!("  worst_bf16=lane {lane} #{at} (walk={x}, body={y}, {worst_bf16} bf16 step(s))")
    });
    Some(format!(
        "differs at #{at} (walk={x} {:#010x}, body={y} {:#010x})  n_diff={differing}/{total}  \
         max_ulp={worst}  class={class}  beyond_1_bf16={beyond_one_bf16}  \
         beyond_8_bf16={beyond_eight_bf16}  non_finite(walk,body)={non_finite:?}  \
         lanes=[{lane_runs}]{worst_bf16_line}",
        x.to_bits(),
        y.to_bits(),
    ))
}
