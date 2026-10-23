#ifndef PIE_SPATIAL_GRID_METAL
#define PIE_SPATIAL_GRID_METAL

#include <metal_stdlib>
using namespace metal;


struct SpatialClip {
    int t;
    int h;
    int w;
    int off;
};

inline int clip_voxels(SpatialClip g) { return g.t * g.h * g.w; }
inline int clip_plane(SpatialClip g) { return g.h * g.w; }

inline SpatialClip clip_at(const device int* grid, int l) {
    const device int* row = grid + 4 * l;
    SpatialClip g;
    g.t = row[0];
    g.h = row[1];
    g.w = row[2];
    g.off = row[3];
    return g;
}

inline int clip_of(const device int* grid, int clips, int row, thread SpatialClip& out) {
    for (int l = 0; l < clips; ++l) {
        const SpatialClip g = clip_at(grid, l);
        const int local = row - g.off;
        if (local >= 0 && local < clip_voxels(g)) {
            out = g;
            return l;
        }
    }
    return -1;
}

struct SpatialVoxel {
    int t;
    int h;
    int w;
};

inline SpatialVoxel unravel(SpatialClip g, int local) {
    SpatialVoxel v;
    v.w = local % g.w;
    const int rest = local / g.w;
    v.t = rest / g.h;
    v.h = rest % g.h;
    return v;
}

inline int ravel(SpatialClip g, int t, int h, int w) {
    return g.off + (t * g.h + h) * g.w + w;
}

inline void segment_of(
    SpatialClip g, int local, int seg_frames, thread int& begin, thread int& end) {
  if (seg_frames <= 0) {
    begin = g.off;
    end = g.off + clip_voxels(g);
    return;
  }
  const int plane = clip_plane(g);
  const int frame = plane > 0 ? local / plane : 0;
  const int first = frame / seg_frames * seg_frames;
  const int last = min(first + seg_frames, g.t);
  begin = g.off + first * plane;
  end = g.off + last * plane;
}

#endif
