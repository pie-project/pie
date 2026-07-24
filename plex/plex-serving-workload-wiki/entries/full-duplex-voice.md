# Full-Duplex Voice and Interruption

## Metadata

| Field | Value |
|---|---|
| Status | Proposed PLEX workload |
| Kind | PLEX synthetic/reconstructed workload |
| Recommended tier | extension |

## Seed sources

- tau3-bench voice
- Video-MME audio
- Andes QoE

## Generator factors

- audio chunk rate
- barge-in
- partial transcript
- simultaneous user/agent speech
- codec/model mix
- deadline

## Required lifecycle events

- audio-in
- partial-decode
- interrupt
- cancel
- resume
- audio-out

## PLEX operations exercised

- admit
- route
- schedule
- feedback

## Metrics

- response onset
- interruption latency
- wasted audio/tokens
- QoE
- goodput

## Why this is new

Turn-based text traces cannot validate preemption and responsiveness for realtime full-duplex agents.
