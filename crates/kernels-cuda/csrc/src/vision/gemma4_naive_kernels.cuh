// COPIED from driver-cuda/csrc/src/model/gemma4/gemma4_naive_kernels.cuh
// (2026-08-09, the VL tower bridge; includes localized). The OLD driver
// keeps its copy until phase E deletes it. Do not diverge.
#pragma once

// The kernels the two gemma-4 tower translation units shared.
//
// The cross-tower half (k_matmul, k_rms) is in
// `model/tower_naive_kernels.cuh`; these are the ones only this tower's
// translation units share. Same way station, same limit -- see that header.

#include "vision/tower_naive_kernels.cuh"

namespace pie_cuda_driver::model {
namespace {

__global__ void k_clamp(const bf* x,bf* o,const bf* lo,const bf* hi,long t){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i>=t)return;
    float v=F(x[i]),l=lo?F(*lo):-CUDART_INF_F,h=hi?F(*hi):CUDART_INF_F;o[i]=Bf(v<l?l:(v>h?h:v));}

}  // namespace
}  // namespace pie_cuda_driver::model
