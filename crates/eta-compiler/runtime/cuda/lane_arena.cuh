// A lane's span of the engine's lane arenas, shared by every region whose
// lane runs on several blocks (a sampler, a split select): an arena a block,
// its first 64-byte line block 0's barrier words (zeroed before the launch),
// then the lane's gather buffers. The launcher lays the same geometry
// (`program::launch::lane_span`).

#define M1_LANE_HALF 24576u
#define M1_LANE_HEAD 64u
#define M1_LANE_MAX_SLICES 64u
// The gather homes a lane has (the launcher's LANE_SAMPLER_BLOCKS: the most
// blocks a lane runs on), and the gathered elements each holds (two buffers
// of them).
#define M1_LANE_ARENAS 16u
#define M1_FAST_GATHER 512u
// A block's arena, from one to the next in a lane's span: its line and its
// halves, to 256 bytes (the launcher's `lane_arena_stride`).
#define M1_LANE_ARENA_STRIDE ((M1_LANE_HEAD + 2u * M1_LANE_HALF + 255u) & ~255u)
// A lane's gather buffers, after its blocks' arenas (the launcher's
// `LANE_SAMPLER_GATHERS`).
#define M1_LANE_GATHERS (M1_LANE_ARENAS * 2u * M1_FAST_GATHER * 8u)
// A lane's span of the lane arenas on `blocks` blocks (the launcher's
// `lane_span`).
#define M1_LANE_SPAN(blocks) ((m1_u64)(blocks) * M1_LANE_ARENA_STRIDE + M1_LANE_GATHERS)
