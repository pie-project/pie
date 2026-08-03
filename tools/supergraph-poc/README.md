# Supergraph PoC — conditional-node capture on this deployment

Proves the three physical premises of the unionized supergraph
(north-star-dsl.md, "The supergraph directive") on the actual target
(L40S, CUDA 13.0, driver 580.159.03):

1. conditional IF/ELSE nodes can be inserted DURING stream capture
   (`cudaStreamGetCaptureInfo` → `cudaGraphAddNode` →
   `cudaStreamUpdateCaptureDependencies`), with arm bodies filled by
   `cudaStreamBeginCaptureToGraph`;
2. the predicate is set INSIDE the graph by a kernel reading DEVICE
   memory (`cudaGraphSetConditional`) — no host round-trip at replay;
3. one instantiated exec serves both arms across replays.

Build & run:
    nvcc -arch=sm_89 -o poc conditional_capture_poc.cu && ./poc
Expected: SUPERGRAPH-POC-OK (1101 / 1212).
