#pragma once

// Clip tables: a batch of video clips (or images, one frame each) laid out
// row after row, a row a voxel. A table row is the i32 quadruple (t, h, w,
// off): the clip's extent and its first row.

struct Clip {
    int t;
    int h;
    int w;
    int off;
};

struct Voxel {
    int t;
    int h;
    int w;
};

int clip_voxels(Clip g) {
    return g.t * g.h * g.w;
}

int clip_plane(Clip g) {
    return g.h * g.w;
}

Clip clip_at(StructuredBuffer<int> grid, int l) {
    Clip g;
    g.t = grid[4 * l];
    g.h = grid[4 * l + 1];
    g.w = grid[4 * l + 2];
    g.off = grid[4 * l + 3];
    return g;
}

// The clip that holds `row`, and its index; -1 (and clip 0) when none does.
int clip_of(StructuredBuffer<int> grid, int clips, int row, out Clip found) {
    found = clip_at(grid, 0);
    for (int l = 0; l < clips; ++l) {
        Clip g = clip_at(grid, l);
        int local = row - g.off;
        if (local >= 0 && local < clip_voxels(g)) {
            found = g;
            return l;
        }
    }
    return -1;
}

Voxel unravel(Clip g, int local) {
    Voxel v;
    v.w = local % g.w;
    int rest = local / g.w;
    v.t = rest / g.h;
    v.h = rest % g.h;
    return v;
}

int ravel(Clip g, int t, int h, int w) {
    return g.off + (t * g.h + h) * g.w + w;
}

// The rows `[begin, end)` of the segment that holds a clip's `local` row:
// the whole clip, or (`seg_frames > 0`) its run of `seg_frames` frames.
void segment_of(Clip g, int local, int seg_frames, out int begin, out int end) {
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

// A row grid past 65535 workgroups rides the z axis: row = `group.y +
// group.z * span`.
uint clip_row(uint3 group, uint span) {
    return group.y + group.z * span;
}
