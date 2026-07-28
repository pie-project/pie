(
    M1Status* status,
    const M1ValueDesc* descriptors,
    const m1_u8* a0,
    const m1_u8* a1,
    const m1_u8* a2,
    m1_u8* o0,
    m1_u8* o1,
    m1_u8* temporary,
    const M1OpParams* params) {
  if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
      threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0)
    return;
  ptir_m1_execute(