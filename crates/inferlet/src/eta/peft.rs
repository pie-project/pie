use std::collections::BTreeMap;

use serde_json::Value;

use super::Channel;
use super::adapter::Site;

pub struct Peft {
    pub(crate) sites: Vec<(Site, Channel, Channel)>,
}

impl Peft {
    pub fn load(config: &[u8], weights: &[u8]) -> Result<Peft, String> {
        let sites = planes(config, weights)?
            .into_iter()
            .map(|(site, a, b)| (site, Channel::from(&a[..]), Channel::from(&b[..])))
            .collect();
        Ok(Peft { sites })
    }
}

type Planes = Vec<(Site, Vec<f32>, Vec<f32>)>;
type Found = BTreeMap<u32, (Site, BTreeMap<usize, [Option<Tensor>; 2]>)>;

pub(crate) fn planes(config: &[u8], weights: &[u8]) -> Result<Planes, String> {
    let config: Value =
        serde_json::from_slice(config).map_err(|e| format!("adapter config: {e}"))?;
    let rank = config["r"].as_f64().ok_or("adapter config: no `r`")?;
    let alpha = config["lora_alpha"].as_f64().unwrap_or(rank);
    let scale = if config["use_rslora"].as_bool().unwrap_or(false) {
        alpha / rank.sqrt()
    } else {
        alpha / rank
    } as f32;

    let mut found: Found = BTreeMap::new();
    let mut layers = 0;
    for (name, tensor) in safetensors(weights)? {
        let Some((layer, site, role)) = locate(&name) else {
            if name.contains("lora_") {
                return Err(format!(
                    "adapter tensor `{name}` corrects no supported projection"
                ));
            }
            continue;
        };
        if tensor.shape.len() != 2 {
            return Err(format!("adapter tensor `{name}` is not a matrix"));
        }
        layers = layers.max(layer + 1);
        found
            .entry(site.bit())
            .or_insert_with(|| (site, BTreeMap::new()))
            .1
            .entry(layer)
            .or_default()[role] = Some(tensor);
    }

    let mut out = Vec::new();
    for (site, per_layer) in found.into_values() {
        let mut shapes = None;
        for (layer, [a, b]) in &per_layer {
            let (Some(a), Some(b)) = (a, b) else {
                return Err(format!(
                    "layer {layer}'s `{site:?}` adapter lacks lora_A or lora_B"
                ));
            };
            let shape = (a.shape[0], a.shape[1], b.shape[0]);
            if b.shape[1] != shape.0 || shapes.is_some_and(|s| s != shape) {
                return Err(format!(
                    "layer {layer}'s `{site:?}` adapter is not one rank and width"
                ));
            }
            shapes = Some(shape);
        }
        let Some((r, fan_in, fan_out)) = shapes else {
            continue;
        };
        let mut a = vec![0.0f32; layers * r * fan_in];
        let mut b = vec![0.0f32; layers * fan_out * r];
        for (layer, [ta, tb]) in per_layer {
            let (ta, tb) = (ta.unwrap(), tb.unwrap());
            a[layer * r * fan_in..][..r * fan_in].copy_from_slice(&ta.data);
            for (to, from) in b[layer * fan_out * r..][..fan_out * r]
                .iter_mut()
                .zip(&tb.data)
            {
                *to = from * scale;
            }
        }
        out.push((site, a, b));
    }
    Ok(out)
}

fn locate(name: &str) -> Option<(usize, Site, usize)> {
    let rest = &name[name.find("layers.")? + "layers.".len()..];
    let (layer, rest) = rest.split_once('.')?;
    let role = if rest.contains(".lora_A.") {
        0
    } else if rest.contains(".lora_B.") {
        1
    } else {
        return None;
    };
    let site = match rest.split(".lora_").next()?.rsplit('.').next()? {
        "q_proj" => Site::Q,
        "k_proj" => Site::K,
        "v_proj" => Site::V,
        "o_proj" => Site::O,
        "gate_proj" => Site::Gate,
        "up_proj" => Site::Up,
        "down_proj" => Site::Down,
        _ => return None,
    };
    Some((layer.parse().ok()?, site, role))
}

struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn safetensors(bytes: &[u8]) -> Result<Vec<(String, Tensor)>, String> {
    let bad = || "adapter weights: not a safetensors file".to_string();
    let len = u64::from_le_bytes(bytes.get(..8).ok_or_else(bad)?.try_into().unwrap()) as usize;
    let header = bytes.get(8..8 + len).ok_or_else(bad)?;
    let data = &bytes[8 + len..];
    let header: serde_json::Map<String, Value> =
        serde_json::from_slice(header).map_err(|_| bad())?;
    let mut tensors = Vec::new();
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        let shape: Vec<usize> = serde_json::from_value(info["shape"].clone()).map_err(|_| bad())?;
        let span: [usize; 2] =
            serde_json::from_value(info["data_offsets"].clone()).map_err(|_| bad())?;
        let raw = data.get(span[0]..span[1]).ok_or_else(bad)?;
        let data: Vec<f32> = match info["dtype"].as_str() {
            Some("F32") => raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Some("BF16") => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
                .collect(),
            Some("F16") => raw
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16(u16::from_le_bytes(*c)))
                .collect(),
            other => {
                return Err(format!(
                    "adapter tensor `{name}` is {other:?}; F32, BF16 or F16 read"
                ));
            }
        };
        if data.len() != shape.iter().product::<usize>() {
            return Err(bad());
        }
        tensors.push((name, Tensor { shape, data }));
    }
    Ok(tensors)
}

fn f16(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 if frac == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let start = data.len();
            data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend(header);
        out.extend(data);
        out
    }

    #[test]
    fn a_peft_adapter_reads_as_per_site_planes_with_its_scale_in_b() {
        let prefix = "base_model.model.model.layers";
        let weights = file(&[
            (
                &format!("{prefix}.1.self_attn.q_proj.lora_A.weight"),
                &[1, 2],
                &[1.0, 2.0],
            ),
            (
                &format!("{prefix}.1.self_attn.q_proj.lora_B.weight"),
                &[3, 1],
                &[1.0, 2.0, 3.0],
            ),
            (
                &format!("{prefix}.0.mlp.down_proj.lora_A.weight"),
                &[1, 3],
                &[4.0, 5.0, 6.0],
            ),
            (
                &format!("{prefix}.0.mlp.down_proj.lora_B.weight"),
                &[2, 1],
                &[7.0, 8.0],
            ),
        ]);
        let planes = planes(br#"{"r": 1, "lora_alpha": 2}"#, &weights).unwrap();
        assert_eq!(planes.len(), 2);
        let (site, a, b) = &planes[0];
        assert_eq!(*site, Site::Q);
        assert_eq!(a, &[0.0, 0.0, 1.0, 2.0]);
        assert_eq!(b, &[0.0, 0.0, 0.0, 2.0, 4.0, 6.0]);
        let (site, a, b) = &planes[1];
        assert_eq!(*site, Site::Down);
        assert_eq!(a, &[4.0, 5.0, 6.0, 0.0, 0.0, 0.0]);
        assert_eq!(b, &[14.0, 16.0, 0.0, 0.0]);
    }

    #[test]
    fn an_unplaceable_lora_tensor_is_refused() {
        let weights = file(&[(
            "base_model.model.model.embed_tokens.lora_embedding_A",
            &[1, 1],
            &[1.0],
        )]);
        assert!(planes(br#"{"r": 1}"#, &weights).is_err());
    }

    #[test]
    fn half_floats_read_exactly() {
        assert_eq!(f16(0x3c00), 1.0);
        assert_eq!(f16(0xc000), -2.0);
        assert_eq!(f16(0x0001), 2f32.powi(-24));
    }
}
