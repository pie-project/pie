"""Method tag constants — synthetic IDs preserved from the pre-rkyv design.

The new pie_bridge schema dispatches via two fields — `Frame.payload_kind`
(REQUEST_FORWARD / REQUEST_COPY / REQUEST_ADAPTER / REQUEST_HEALTH) plus
a sub-field for copy direction or adapter op. The dev worker's polling
loop still expects a single flat method_tag, so this module keeps the
old numeric identity and `shmem_schema.parse_request` translates
(payload_kind, sub_op) → method_tag on the way in.

These IDs match the OLD `pie_bridge.Method` enum (Forward=0, CopyD2H=1,
CopyH2D=2, CopyD2D=3, CopyH2H=4, LoadAdapter=5, SaveAdapter=6,
ZoInitializeAdapter=7, ZoUpdateAdapter=8) so callers in `worker.py` need
no changes.
"""

# Forward
FORWARD = 0

# Copy ops (encoded as (REQUEST_COPY, CopyDir) in the new wire format)
COPY_D2H = 1
COPY_H2D = 2
COPY_D2D = 3
COPY_H2H = 4

# Adapter ops (encoded as (REQUEST_ADAPTER, AdapterOp) in the new wire format)
LOAD_ADAPTER = 5
SAVE_ADAPTER = 6
ZO_INITIALIZE_ADAPTER = 7
ZO_UPDATE_ADAPTER = 8

# Health request (no driver-side handler — replied with status=0 directly)
HEALTH = 9
