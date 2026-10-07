#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Published {
    pub target: &'static str,
    pub head: &'static str,
    pub drafter: &'static str,
    pub deployment: &'static str,
}

pub const PUBLISHED: &[Published] = &[
    Published {
        target: "mlx-community/Qwen3.6-27B-4bit",
        head: "z-lab/Qwen3.6-27B-DFlash",
        drafter: "dflash",
        deployment: "qwen36-27b-dflash-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/Qwen3.8-27B-4bit",
        head: "z-lab/Qwen3.8-27B-DFlash2",
        drafter: "dflash2",
        deployment: "qwen38-27b-dflash2-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/Qwen3.8-27B-4bit",
        head: "DimInfer/Qwen3.8-27B-Dspark-v1",
        drafter: "dspark",
        deployment: "qwen38-27b-dspark-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/Qwen3.6-35B-A3B-4bit",
        head: "z-lab/Qwen3.6-35B-A3B-DFlash",
        drafter: "dflash",
        deployment: "qwen36-35b-a3b-dflash-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/Qwen3.5-9B-4bit",
        head: "z-lab/Qwen3.5-9B-DFlash",
        drafter: "dflash",
        deployment: "qwen35-d9b-dflash-u4g64-kv-bf16",
    },
    Published {
        target: "mlx-community/gpt-oss-20b-MXFP4-Q4",
        head: "z-lab/gpt-oss-20b-DFlash",
        drafter: "dflash",
        deployment: "gptoss-20b-dflash-u4g64-mxfp4-kv-bf16",
    },
    Published {
        target: "mlx-community/gemma-4-26b-a4b-it-4bit",
        head: "z-lab/gemma-4-26B-A4B-it-DFlash",
        drafter: "dflash",
        deployment: "gemma4-26b-a4b-dflash-u4g64-kv-bf16",
    },
];

#[must_use]
pub fn lookup(target: &str, drafter: &str) -> Option<&'static Published> {
    let wanted = target.to_ascii_lowercase().replace("--", "/");
    PUBLISHED.iter().find(|p| {
        p.drafter.eq_ignore_ascii_case(drafter) && p.target.to_ascii_lowercase() == wanted
    })
}

pub fn for_target(target: &str) -> impl Iterator<Item = &'static Published> {
    let wanted = target.to_ascii_lowercase().replace("--", "/");
    PUBLISHED
        .iter()
        .filter(move |p| p.target.to_ascii_lowercase() == wanted)
}
