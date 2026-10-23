#pragma once

// A workgroup barrier that also orders device-memory writes, for a kernel
// whose lanes read back what other lanes wrote to a storage buffer. Slang's
// own `DeviceMemoryBarrierWithGroupSync` adds a texture barrier on WGSL,
// which needs a language feature not every browser has; no kernel here
// touches a texture.
void pie_storage_barrier() {
    __target_switch
    {
    case wgsl:
        __intrinsic_asm "storageBarrier(); workgroupBarrier()";
    default:
        DeviceMemoryBarrierWithGroupSync();
    }
}
