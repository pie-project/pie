#pragma once

// `v` from the lane whose index is this lane's xor `mask`, within the
// subgroup. Slang lowers `WaveReadLaneAt` on WGSL to `subgroupShuffle` with
// an i32 lane, which Naga refuses; WGSL's own `subgroupShuffleXor` takes the
// mask as u32, and the module then asks for `enable subgroups;` (a browser
// needs it; engine-wgpu drops it for Naga).
float pie_shuffle_xor(float v, uint mask) {
    __target_switch
    {
    case wgsl:
        __requirePrelude("enable subgroups;");
        __intrinsic_asm "subgroupShuffleXor($0, $1)";
    default:
        return WaveReadLaneAt(v, WaveGetLaneIndex() ^ mask);
    }
}
