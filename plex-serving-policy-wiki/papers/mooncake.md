# Mooncake

> Canonical title: **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://www.usenix.org/conference/fast25/presentation/qin](https://www.usenix.org/conference/fast25/presentation/qin) |
| Venue / status | FAST 2025 |
| Year | 2025 |
| Authors | Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, Xinran Xu |
| Institutions / group context | Not reliably resolved |
| Reputation evidence | strong publication signal from a selective systems/ML venue (FAST 2025); author affiliations were not reliably resolved; a public implementation or artifact is linked. |
| OpenAlex cited-by count | 13 (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| DOI | https://doi.org/10.48550/arxiv.2407.00079 |
| arXiv | Not resolved |
| Artifact | [Public artifact](https://github.com/kvcache-ai/Mooncake) |
| Corpus category | Routing, placement, and rebalancing |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies how requests or state should be placed across replicas, tiers, models, or heterogeneous resources. Its central policy contribution is: KV-centric routing and distributed cache hierarchy.

The primary source has an abstract indexed in the metadata source used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

KV-centric routing and distributed cache hierarchy.

## PLEX mapping

`R+E/P`; separate policy from cache transport.

## Datasets and workloads

- No named dataset was reliably identified from accessible metadata.

No named dataset was reliably identified from the accessible metadata; consult the evaluation section.

## Suggested citation

Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, et al.. “Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving.” FAST 2025, 2025. https://doi.org/10.48550/arxiv.2407.00079.

## Sources

- [Primary paper](https://www.usenix.org/conference/fast25/presentation/qin)
- [OpenAlex record](https://openalex.org/W4400377444)
- [Artifact](https://github.com/kvcache-ai/Mooncake)
