#pragma once

// The naive kernels more than one TOWER defines identically.
//
// `k_matmul` and `k_rms` are byte-identical in `model/csm/` and
// `model/gemma4/` -- fingerprinted, not read. Every tower writes its kernels
// in an anonymous namespace, so no author can see the copy next door; that
// invisibility is the mechanism, and it produced the same scalar matmul three
// times and the same RMSNorm twice.
//
// This is a WAY STATION. The answer is to stop having them: `gemm::act_x_wt_bf16`
// and `norm::rmsnorm_bf16` already exist and are what these should call. That
// swap changes the arithmetic -- a naive scalar loop is not cuBLAS -- so it
// needs the tower parity harnesses (`gemma4_vision_full_parity_bf16`,
// `csm_backbone_parity`) run against reference dumps. Collapsing identical
// bodies does not, which is why this can land and that cannot.
//
// Reading the seven that looked different settled each one:
//
//   k_add          SAME. Three copies whose only difference was parameter
//                  NAMES (a/b vs h/x) and a line break. Here now.
//   k_f32_to_bf16  SAME. Not two definitions at all -- qwen3_vl forward-
//                  declares it and defines it later in the same file, and the
//                  fingerprint had swallowed the next kernel. Here now.
//   k_layernorm    SAME ALGORITHM, wider contract. qwen3_vl's takes gamma/beta
//                  as optional (`g ? F(g[d]) : 1.f`) where mimi's dereferences
//                  them; mimi always passes non-null, so the general one is
//                  bit-identical for both. The general one is here.
//   k_gelu         DIFFERENT FUNCTION. mimi computes the exact erf form
//                  (`transformers` ACT2FN["gelu"]); qwen3_vl the tanh
//                  approximation. Merging them would have been a silent
//                  numerics change. Both stay put, and a shared `mlp::gelu`
//                  will have to offer both.
//   k_addpos       DIFFERENT OP. gemma4 indexes a 2-D grid position table
//                  twice per token; qwen3_vl adds a precomputed vector.
//   k_rope         DIFFERENT OP. csm is 1-D with YaRN scaling, gemma4 is
//                  2-D axial. See .wiki/kernel-refactor.md §2.2.
//   k_attn         SAME SHAPE, different axes -- one takes a q-offset into a
//                  KV cache, the other a sliding window. One parameterised
//                  kernel could cover both, which makes it a numerics change
//                  and so a job for the parity harnesses.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pie_cuda_driver::model {
namespace {

using bf = __nv_bfloat16;
__device__ __forceinline__ float F(bf x){return __bfloat162float(x);}
__device__ __forceinline__ bf   Bf(float x){return __float2bfloat16(x);}

__global__ void k_matmul(const bf* x,const bf* W,bf* y,int N,int K,int O){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=N||o>=O)return;
    const bf* xr=x+(long)n*K;const bf* wr=W+(long)o*K;
    float a=0;for(int k=0;k<K;k++)a+=F(xr[k])*F(wr[k]);
    y[(long)n*O+o]=Bf(a);
}

__global__ void k_rms(const bf* x,const bf* w,bf* o,int R,int D,float eps){
    int r=blockIdx.x;if(r>=R)return;const bf* xr=x+(long)r*D;bf* orow=o+(long)r*D;
    float loc=0;for(int d=threadIdx.x;d<D;d+=blockDim.x){float v=F(xr[d]);loc+=v*v;}
    for(int s=warpSize/2;s>0;s>>=1)loc+=__shfl_down_sync(0xffffffff,loc,s);
    __shared__ float warp[32],ss;if((threadIdx.x&31)==0)warp[threadIdx.x>>5]=loc;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=warp[i];ss=rsqrtf(t/D+eps);}__syncthreads();
    float inv=ss;for(int d=threadIdx.x;d<D;d+=blockDim.x)orow[d]=Bf(F(xr[d])*inv*(w?F(w[d]):1.f));
}

__global__ void k_add(bf* a,const bf* b,long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<n)a[i]=Bf(F(a[i])+F(b[i]));}

__global__ void k_f32_to_bf16(const float* a, bf* o, long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) o[i]=Bf(a[i]);
}

// Exact GELU: 0.5*x*(1+erf(x/√2)) — transformers' ACT2FN["gelu"], and
// nn.GELU(approximate='none'). The name carries `_erf` because the OTHER form
// exists in this tree: qwen3_vl's patch tower uses the tanh approximation, and
// for a while both were called `k_gelu` in different files. Merging those two
// by name would have changed numerics silently; keeping the form in the name
// is what makes the duplicate below a real duplicate.
//
// mimi and qwen3_vl each had this, spelled differently (`if(i>=n)return` vs a
// guarded block) with the √2 reciprocal written to different lengths --
// 0.70710678118f and 0.70710678118654752f both round to the same float, so the
// two were bit-identical all along.
__global__ void k_gelu_erf(const bf* x,bf* o,long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i>=n)return;
    float v=F(x[i]);o[i]=Bf(0.5f*v*(1.f+erff(v*0.70710678118654752f)));}

__global__ void k_layernorm(const bf* x,const bf* g,const bf* bta,bf* o,int R,int D,float eps){
    int r=blockIdx.x;if(r>=R)return;const bf* xr=x+(long)r*D;bf* orow=o+(long)r*D;
    float sum=0;for(int d=threadIdx.x;d<D;d+=blockDim.x)sum+=F(xr[d]);
    for(int s=warpSize/2;s>0;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);
    __shared__ float warp[32],smean,svar;if((threadIdx.x&31)==0)warp[threadIdx.x>>5]=sum;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=warp[i];smean=t/D;}__syncthreads();
    float mean=smean,v=0;for(int d=threadIdx.x;d<D;d+=blockDim.x){float dx=F(xr[d])-mean;v+=dx*dx;}
    for(int s=warpSize/2;s>0;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
    if((threadIdx.x&31)==0)warp[threadIdx.x>>5]=v;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=warp[i];svar=rsqrtf(t/D+eps);}__syncthreads();
    float inv=svar;for(int d=threadIdx.x;d<D;d+=blockDim.x){
        float nrm=(F(xr[d])-mean)*inv;orow[d]=Bf(nrm*(g?F(g[d]):1.f)+(bta?F(bta[d]):0.f));}}

}  // namespace
}  // namespace pie_cuda_driver::model
