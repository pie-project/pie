#ifndef PIE_METAL_ROW_GATHER_PARAMS_H
#define PIE_METAL_ROW_GATHER_PARAMS_H

// Shared host/shader ABI for row_gather.metal (buffer 3).
struct RowGatherParams {
  unsigned int width;  // elements per row
  unsigned int count;  // rows to gather
};

#endif
