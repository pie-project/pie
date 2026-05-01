"""Wire format for the cuda_native control socket.

Mirrors the canonical C++ definition in
`driver/cuda/src/control_socket.hpp`. Tests in
`pie/tests/pie_driver_cuda_native/test_control_protocol.py` parse the
header and assert that values declared here still match — flagging
drift the moment one side moves without the other.

Layout (little-endian SOCK_SEQPACKET frame):

    u32 method      // see METHOD_COPY_*
    u32 layer
    u32 num_pairs
    u32 reserved    // pad to 16 bytes
    u32 srcs[num_pairs]
    u32 dsts[num_pairs]

Response: a single u32 status (0 = ok, non-zero = error).
"""

from __future__ import annotations

import struct

# Method tags (must match `CTRL_METHOD_COPY_*` in control_socket.hpp).
METHOD_COPY_D2H = 1
METHOD_COPY_H2D = 2
METHOD_COPY_D2D = 3
METHOD_COPY_H2H = 4

# 16-byte header before the (srcs, dsts) payload.
HEADER = struct.Struct("<IIII")
HEADER_SIZE = HEADER.size

# Response is a single u32 status. Sized for `recv(RESPONSE_SIZE)`.
RESPONSE_SIZE = 4
