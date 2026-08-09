#pragma once

// The kernels the three CSM translation units shared.
//
// The cross-tower half (k_matmul, k_rms) is in
// `model/tower_naive_kernels.cuh`; these are the ones only this tower's
// translation units share. Same way station, same limit -- see that header.

#include "model/tower_naive_kernels.cuh"

namespace pie_cuda_driver::model {
namespace {

__global__ void k_swiglu(const bf* gate,const bf* up,bf* o,long t){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i>=t)return;
    float g=F(gate[i]);o[i]=Bf((g/(1.f+__expf(-g)))*F(up[i]));
}

__global__ void k_argmax(const bf* logits,int V,int* out){
    int t=threadIdx.x;float bv=-1e30f;int bi=0;
    for(int v=t;v<V;v+=blockDim.x){float x=F(logits[v]);if(x>bv){bv=x;bi=v;}}
    __shared__ float sv[256];__shared__ int si[256];
    sv[t]=bv;si[t]=bi;__syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){if(t<s){if(sv[t+s]>sv[t]||(sv[t+s]==sv[t]&&si[t+s]<si[t])){sv[t]=sv[t+s];si[t]=si[t+s];}}__syncthreads();}
    if(t==0)*out=si[0];
}

}  // namespace
}  // namespace pie_cuda_driver::model
