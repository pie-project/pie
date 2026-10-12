#ifndef PIE_VULKAN_SPATIAL_GRID_SLANG
#define PIE_VULKAN_SPATIAL_GRID_SLANG

// A clip table row is (t, h, w, off): the clip's extent and its first row.
struct SpatialClip {
    int t;
    int h;
    int w;
    int off;
};

int clip_voxels(SpatialClip g) { return g.t * g.h * g.w; }
int clip_plane(SpatialClip g) { return g.h * g.w; }

SpatialClip clip_at(StructuredBuffer<int> grid, int l) {
    SpatialClip g;
    g.t = grid[4 * l];
    g.h = grid[4 * l + 1];
    g.w = grid[4 * l + 2];
    g.off = grid[4 * l + 3];
    return g;
}

int clip_of(StructuredBuffer<int> grid, int clips, int row, out SpatialClip found) {
    found = clip_at(grid, 0);
    for (int l = 0; l < clips; ++l) {
        SpatialClip g = clip_at(grid, l);
        int local = row - g.off;
        if (local >= 0 && local < clip_voxels(g)) {
            found = g;
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

SpatialVoxel unravel(SpatialClip g, int local) {
    SpatialVoxel v;
    v.w = local % g.w;
    int rest = local / g.w;
    v.t = rest / g.h;
    v.h = rest % g.h;
    return v;
}

int ravel(SpatialClip g, int t, int h, int w) {
    return g.off + (t * g.h + h) * g.w + w;
}

void segment_of(SpatialClip g, int local, int seg_frames, out int begin, out int end) {
    if (seg_frames <= 0) {
        begin = g.off;
        end = g.off + clip_voxels(g);
        return;
    }
    int plane = clip_plane(g);
    int frame = plane > 0 ? local / plane : 0;
    int first = frame / seg_frames * seg_frames;
    int last = min(first + seg_frames, g.t);
    begin = g.off + first * plane;
    end = g.off + last * plane;
}

uint spatial_row(uint3 group, uint span) {
    return group.y + group.z * span;
}

#endif
