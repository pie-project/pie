# Parrot

> Canonical title: **Parrot: Efficient Serving of LLM-based Applications with Semantic Variable**

## Metadata

| Field | Value |
|---|---|
| Primary paper | [https://www.usenix.org/conference/osdi24/presentation/lin-chaofan](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan) |
| Venue / status | OSDI 2024 |
| Year | 2024 |
| Authors | Chaofan Lin, Zhenhua Han, C. R. Zhang, Yuqing Yang, Fan Yang, Chen Chen, Lili Qiu |
| Institutions / group context | Shanghai Jiao Tong University, Microsoft Research |
| Reputation evidence | peer-reviewed at OSDI 2024; author affiliations include Shanghai Jiao Tong University and Microsoft Research; a public implementation is available. |
| Citation count | 5 via OpenAlex (retrieved 2026-07-23; preprint and proceedings records may split citations) |
| Metadata provenance | OpenAlex |
| DOI | https://doi.org/10.48550/arxiv.2405.19888 |
| arXiv | 2405.19888 |
| Artifact | [Public implementation](https://github.com/microsoft/ParrotServe) |
| Corpus category | Agent, workflow, and multi-turn serving |

## Abstract synopsis

_Editorial paraphrase, not the paper's verbatim abstract._

This work studies serving when one user-visible task spans multiple LLM calls, workflow nodes, or tool boundaries rather than one isolated request. The proposed policy centers on semantic-variable dag, dependency-aware placement and execution. The reported evaluation context includes Agent/workflow trace-driven evaluation; no named public dataset was reliably recovered.

An abstract was available through the indexed or primary-page metadata used for this catalog. Follow the primary-paper link for the authoritative abstract and version history.

## Serving-policy summary

Semantic-variable DAG, dependency-aware placement and execution.

## PLEX mapping

`R+S`; good corpus, larger application contract.

## Datasets and evaluation workloads

- Agent/workflow trace-driven evaluation; no named public dataset was reliably recovered

No named public dataset was reliably confirmed; the workload description is categorical and the paper's evaluation section is authoritative.

<!-- plex-v0.6-replication:start -->
## PLEX v0.6 replication status

- Component: `plex_paper_parrot`
- Operations: `route`, `schedule`
- Evidence: `inspired-adaptation`
- Validation: `passing`
- Metadata: [`tests/policies/replications/parrot/metadata.json`](../../tests/policies/replications/parrot/metadata.json)
- Deferred mechanics: general semantic-variable DAG runtime
<!-- plex-v0.6-replication:end -->

## Suggested citation

Chaofan Lin, Zhenhua Han, C. R. Zhang, Yuqing Yang, Fan Yang, Chen Chen, et al. “Parrot: Efficient Serving of LLM-based Applications with Semantic Variable.” arXiv preprint, 2024. https://doi.org/10.48550/arxiv.2405.19888.

## Sources

- [Primary paper](https://www.usenix.org/conference/osdi24/presentation/lin-chaofan)
- [Public artifact](https://github.com/microsoft/ParrotServe)
- [OpenAlex record](https://openalex.org/W4399252473)
