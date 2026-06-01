// Qwen3-VL vision encoder forward (bf16). See qwen3_vl_vision_forward.hpp.
//
// First-cut draft mirroring gemma4_vision_forward.cu as closely as possible:
// same naive kernel patterns + cudaMalloc scratch (correctness over speed),
// bf16 storage + fp32 compute (matching the driver). CUDA-only includes (no
// model/loader headers) so nvcc never sees the toml++ config headers.
//
// What differs from gemma (added here): LayerNorm-with-bias (gamma+beta, mean
// AND variance — gemma's k_rms is variance-only RMSNorm), bias-add after the
// matmuls (QKV/o/fc1/fc2/merger have bias; gemma's clipped-linears do not),
// plain non-gated GELU MLP (gemma is gated gate·up), the 2×2 spatial-merge
// gather (gemma uses 2D avg-pool by patch coords), and the ViT 2D-RoPE which
// uses transformers' rotate_half layout (head_dim split into a row-half and a
// col-half, then full rotate_half over each — NOT gemma's interleaved pairs).
// The abs pos-embed bilinear interpolation is done on the host (see the scatter
// + the header note); the kernel just adds the precomputed `[n_patch, hidden]`.
//
// NOT yet parity-verified — a faithful first draft to be checked against
// scripts/qwen3_vl_vision_parity_ref.py dumps (/tmp/qwen3_vl_vision_parity/).
// Per MULTIMODAL.md §11, parity is bf16-vs-bf16 rel_rms + cosine, not max_abs.

#include "model/qwen3_vl_vision_forward.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <map>
#include <math_constants.h>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <string>
#include <vector>

namespace pie_cuda_driver::model {

// cuBLAS bf16 GEMM: y[M,N] = x[M,K] @ W[N,K]^T (row-major; bf16 in/out, fp32
// accumulate — same math as the old naive k_matmul). Defined in the
// g++-compiled adapter (qwen3_vl_vision_adapter.cpp), which can include the
// heavy ops/gemm.hpp; declared here so this CUDA-only TU stays light. The
// caller guarantees the handle's stream == the `S` passed to the forward, so
// the bias kernel below orders correctly after the GEMM.
void qwen3vl_vis_gemm_bf16(cublasHandle_t blas, const void* x, const void* W,
                           void* y, int M, int N, int K, float beta = 0.f);

// Vision attention via flashinfer (defined in the adapter — owns a dedicated
// attention workspace + plan). Block-diagonal non-causal MHA over `num_seqs`
// images (multi-sequence); q/k already carry 2D-RoPE.
void qwen3vl_vis_attn(const void* q, void* k, void* v, void* o,
                      int num_seqs, const int* seqlens, int NH, int HEAD, cudaStream_t S);

// ── per-layer checkpoint hook (parity debugging) ─────────────────────────────
static Qwen3VLVisionCkptFn g_qvis_ckpt = nullptr;
static void* g_qvis_ckpt_user = nullptr;
void set_qwen3vl_vision_ckpt(Qwen3VLVisionCkptFn fn, void* user) {
    g_qvis_ckpt = fn; g_qvis_ckpt_user = user;
}

namespace {

typedef __nv_bfloat16 bf;
#define QCK(x) do{cudaError_t e=(x);if(e)throw std::runtime_error(std::string("qwen3vl_vision: ")+cudaGetErrorString(e));}while(0)
__device__ __forceinline__ float F(bf x){return __bfloat162float(x);}
__device__ __forceinline__ bf   Bf(float x){return __float2bfloat16(x);}

// Defined later (also used by scatter); forward-declared for the attention loop.
__global__ void k_f32_to_bf16(const float* a, bf* o, long n);

// Add bias[col] to y[m,col] (the GEMM epilogue the old k_matmul folded in).
__global__ void k_bias(bf* y,const bf* b,long M,int N){
    long i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=M*(long)N)return;
    y[i]=Bf(F(y[i])+F(b[i%N]));}

// cuBLAS GEMM + optional bias, replacing the naive k_matmul. M=rows, O=out, K=in.
inline void gemm_bias(cublasHandle_t blas,const bf* x,const QVisLinear& lin,
                      bf* y,int M,int O,int K,cudaStream_t S){
    qwen3vl_vis_gemm_bf16(blas,x,lin.w,y,M,O,K);
    if(lin.b) k_bias<<<((long)M*O+255)/256,256,0,S>>>(y,lin.b,(long)M,O);
}

// (Projections now route through cuBLAS via gemm_bias(); the naive per-output
// k_matmul that lived here has been removed.)

// LayerNorm over the last dim D (mean + variance), gamma+beta.  One block/row.
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

__global__ void k_add_inplace(bf* h,const bf* x,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t)h[i]=Bf(F(h[i])+F(x[i]));}

// Add the precomputed interpolated abs pos-embed `pe[n_patch,D]` into `h`.
__global__ void k_addpos(bf* h,const bf* pe,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t)h[i]=Bf(F(h[i])+F(pe[i]));}

// Plain (non-gated) gelu-tanh: o = 0.5*x*(1+tanh(√(2/π)(x+0.044715x³))).
// Used by the ViT block MLP (hidden_act = gelu_pytorch_tanh).
__global__ void k_gelu(const bf* x,bf* o,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;
    if(i<t){float v=F(x[i]);o[i]=Bf(0.5f*v*(1.f+tanhf(0.7978845608f*(v+0.044715f*v*v*v))));}}

// Exact (erf) GELU: o = 0.5*x*(1+erf(x/√2)).  Used by the patch mergers, which
// in transformers use nn.GELU() (approximate='none'), NOT the tanh approximation.
__global__ void k_gelu_erf(const bf* x,bf* o,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;
    if(i<t){float v=F(x[i]);o[i]=Bf(0.5f*v*(1.f+erff(v*0.70710678118654752f)));}}

// Split fused QKV `[N, 3*hidden]` (row layout q|k|v) into q,k,v `[N, hidden]`.
__global__ void k_split_qkv(const bf* qkv,bf* q,bf* k,bf* v,int N,int H){
    int n=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||d>=H)return;
    const bf* r=qkv+(long)n*3*H;q[(long)n*H+d]=r[d];k[(long)n*H+d]=r[H+d];v[(long)n*H+d]=r[2*H+d];}

// Split fused QKV AND add the (per-section) qkv bias in one pass — folds the
// post-GEMM bias add into the split so the qkv projection skips its bias kernel.
__global__ void k_split_qkv_bias(const bf* qkv,const bf* b,bf* q,bf* k,bf* v,int N,int H){
    int n=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||d>=H)return;
    const bf* r=qkv+(long)n*3*H;
    float bq=b?F(b[d]):0.f, bk=b?F(b[H+d]):0.f, bv=b?F(b[2*H+d]):0.f;
    q[(long)n*H+d]=Bf(F(r[d])+bq);k[(long)n*H+d]=Bf(F(r[H+d])+bk);v[(long)n*H+d]=Bf(F(r[2*H+d])+bv);}

// gelu_pytorch_tanh with a fused per-column bias add — folds fc1's bias kernel
// into the activation (block MLP only; mergers keep the separate erf path).
__global__ void k_gelu_bias(bf* x,const bf* b,int N,int D){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x; long t=(long)N*D; if(i>=t)return;
    float v=F(x[i])+(b?F(b[i%D]):0.f);
    x[i]=Bf(0.5f*v*(1.f+tanhf(0.7978845608f*(v+0.044715f*v*v*v))));}

// ViT 2D-RoPE (transformers rotate_half layout). Per HF: the rotary table is
// built from `position_ids` `[N,2]=(row,col)`: rope dim = head_dim/2 = 32, the
// inv_freq has head_dim/4 = 16 entries. `rotary_pos_emb = (pos[...,None]*inv_freq)
// .flatten(1)` → per token a length-32 vector = [row*invf(0..15), col*invf(0..15)],
// then `emb = cat(rope,rope)` → length 64; cos/sin length 64. Attention applies
// q_embed = q*cos + rotate_half(q)*sin over the full head_dim=64, where
// rotate_half splits at 32. So effectively: pair index j in [0,16) rotates
// (q[j], q[j+32]) by angle row*invf[j]; pair index j in [16,32) rotates
// (q[j], q[j+32]) by angle col*invf[j-16].  invf[c] = theta^(-2c/(head_dim/2)).
//   half = head_dim/2 (=32); quarter = head_dim/4 (=16).
// PARITY TODO: confirm the rotate_half pairing (j, j+half) and that the row/col
// split of the rope dim is [0:quarter)=row, [quarter:half)=col, against
// transformers apply_rotary_pos_emb_vision + Qwen3VLVisionRotaryEmbedding(head_dim//2).
__global__ void k_rope_vis(bf* q,const float* pos,int N,int NH,int HEAD,float theta){
    int n=blockIdx.z,head=blockIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;
    int half=HEAD/2, quarter=HEAD/4; if(n>=N||head>=NH||j>=half)return;
    bf* v=q+(((long)n*NH+head)*HEAD);
    float row=pos[2L*n],col=pos[2L*n+1];
    // angle for rope-dim index j: first `quarter` use row, next `quarter` use col.
    int c = (j<quarter) ? j : (j-quarter);
    float coord = (j<quarter) ? row : col;
    float invf = powf(theta, -2.f*(float)c/(float)half);
    float ang = coord*invf, cs=cosf(ang), sn=sinf(ang);
    // rotate_half over the full head: pair (j, j+half).
    float a=F(v[j]), b=F(v[j+half]);
    v[j]      = Bf(a*cs - b*sn);
    v[j+half] = Bf(b*cs + a*sn);}

// Apply the same 2D-RoPE to BOTH q and k in one launch (q/k share positions),
// halving the rope kernel count per layer. Math identical to k_rope_vis.
__global__ void k_rope_qk(bf* q,bf* k,const float* pos,int N,int NH,int HEAD,float theta){
    int n=blockIdx.z,head=blockIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;
    int half=HEAD/2, quarter=HEAD/4; if(n>=N||head>=NH||j>=half)return;
    float row=pos[2L*n],col=pos[2L*n+1];
    int c = (j<quarter) ? j : (j-quarter);
    float coord = (j<quarter) ? row : col;
    float invf = powf(theta, -2.f*(float)c/(float)half);
    float ang = coord*invf, cs=cosf(ang), sn=sinf(ang);
    long base=((long)n*NH+head)*HEAD;
    bf* vq=q+base; float aq=F(vq[j]), bq=F(vq[j+half]);
    vq[j]=Bf(aq*cs-bq*sn); vq[j+half]=Bf(bq*cs+aq*sn);
    bf* vk=k+base; float ak=F(vk[j]), bk=F(vk[j+half]);
    vk[j]=Bf(ak*cs-bk*sn); vk[j+half]=Bf(bk*cs+ak*sn);}

// Fused split-qkv(+bias)+RoPE: read fused qkv[N,3H] once, add the per-section
// bias, write q,k,v[N,H] with q/k already 2D-RoPE'd — one pass instead of a
// split kernel then a rope kernel (saves the rope's q/k read+write round-trip).
// One block per (n, head); thread j handles rope pair (j, j+half) of q and k
// plus the v copy for both lanes. Math identical to k_split_qkv_bias + k_rope_qk.
__global__ void k_split_rope_qkv(const bf* qkv,const bf* b,bf* q,bf* k,bf* v,
                                 const float* pos,int N,int NH,int HEAD,float theta){
    int n=blockIdx.y, head=blockIdx.x; if(n>=N||head>=NH) return;
    const int H=NH*HEAD, half=HEAD/2, quarter=HEAD/4;
    const bf* qr=qkv+(long)n*3*H + head*HEAD;   // q section for this head
    const bf* kr=qr + H;                         // k section
    const bf* vr=kr + H;                         // v section
    const bf* bq=b? b+head*HEAD          : nullptr;
    const bf* bk=b? b+H+head*HEAD        : nullptr;
    const bf* bv=b? b+2*H+head*HEAD      : nullptr;
    long o=((long)n*NH+head)*HEAD; bf* qo=q+o; bf* ko=k+o; bf* vo=v+o;
    float row=pos[2L*n], col=pos[2L*n+1];
    for(int j=threadIdx.x; j<half; j+=blockDim.x){
        int c=(j<quarter)?j:(j-quarter);
        float coord=(j<quarter)?row:col;
        float invf=powf(theta,-2.f*(float)c/(float)half);
        float ang=coord*invf, cs=cosf(ang), sn=sinf(ang);
        float q0=F(qr[j])+(bq?F(bq[j]):0.f), q1=F(qr[j+half])+(bq?F(bq[j+half]):0.f);
        qo[j]=Bf(q0*cs-q1*sn); qo[j+half]=Bf(q1*cs+q0*sn);
        float k0=F(kr[j])+(bk?F(bk[j]):0.f), k1=F(kr[j+half])+(bk?F(bk[j+half]):0.f);
        ko[j]=Bf(k0*cs-k1*sn); ko[j+half]=Bf(k1*cs+k0*sn);
        vo[j]=Bf(F(vr[j])+(bv?F(bv[j]):0.f));
        vo[j+half]=Bf(F(vr[j+half])+(bv?F(bv[j+half]):0.f));
    }}

// (Attention QK/softmax/AV now run inside flashinfer; the cuBLAS+softmax kernels
// that lived here were removed.)

// (AV now runs via cuBLAS in the block loop; the naive k_av was removed.)

// 2×2 spatial-merge gather: input `h[n_patch, C]` is already in spatial-merge
// order (every `merge²` consecutive patches form one output token — see the
// header / the host `get_vision_position_ids` reorder). Output `g[n_token, merge²*C]`
// concatenates the `merge²` patch rows of each group: g[tok, u*C + c] = h[tok*U+u, c].
// PARITY TODO: confirm the within-group order (u = 0..U-1) matches the HF reshape
// `(h//m, m, w//m, m, C) -> (h//m * w//m, m*m, C) -> (..., m*m*C)`; the host
// reorder is built so consecutive rows ARE that group, so a plain concat suffices.
__global__ void k_merge_gather(const bf* h,bf* g,int n_token,int U,int C){
    int tok=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;
    int W=U*C; if(tok>=n_token||d>=W)return;
    int u=d/C, c=d%C;
    g[(long)tok*W+d]=h[((long)tok*U+u)*C+c];}

dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}

// fc1 → GELU → fc2 with bias on both (merger MLP + block MLP share this).
// `erf_gelu`: false = gelu_pytorch_tanh (ViT block MLP), true = exact erf GELU
// (the patch mergers' nn.GELU(), approximate='none').
void mlp(cublasHandle_t blas,const bf* in,bf* mid,bf* out,const QVisLinear& fc1,const QVisLinear& fc2,
         int N,int Din,int Dmid,int Dout,cudaStream_t S,bool erf_gelu=false){
    gemm_bias(blas,in,fc1,mid,N,Dmid,Din,S);
    if(erf_gelu) k_gelu_erf<<<((long)N*Dmid+255)/256,256,0,S>>>(mid,mid,(long)N*Dmid);
    else         k_gelu    <<<((long)N*Dmid+255)/256,256,0,S>>>(mid,mid,(long)N*Dmid);
    gemm_bias(blas,mid,fc2,out,N,Dout,Dmid,S);
}

// Run one patch merger over `h[n_patch, hidden]` → `out[n_token, out_hidden]`.
//   main     : LayerNorm(hidden) on h → 2×2 group(→4*hidden) → fc1 → GELU → fc2.
//   deepstack: 2×2 group(→4*hidden) → LayerNorm(4*hidden) → fc1 → GELU → fc2.
void run_merger(cublasHandle_t blas,const QVisMerger& m,const bf* h,int n_patch,int n_token,
                int hidden,int merge_unit,int out_hidden,float eps,bf* out,cudaStream_t S){
    const int W=merge_unit*hidden;  // 4096
    auto MAL=[&](long n){bf* d;QCK(cudaMallocAsync(&d,n*sizeof(bf),S));return d;};
    bf* normed=MAL((long)n_patch*hidden);
    bf* grouped=MAL((long)n_token*W);
    bf* mid=MAL((long)n_token*W);
    if(!m.is_postshuffle){
        // main: norm over hidden first, then gather.
        k_layernorm<<<n_patch,256,0,S>>>(h,m.norm.g,m.norm.b,normed,n_patch,hidden,eps);
        k_merge_gather<<<G2(W,n_token),B2,0,S>>>(normed,grouped,n_token,merge_unit,hidden);
    }else{
        // deepstack: gather first (→4096), then norm over 4096.
        k_merge_gather<<<G2(W,n_token),B2,0,S>>>(h,grouped,n_token,merge_unit,hidden);
        k_layernorm<<<n_token,256,0,S>>>(grouped,m.norm.g,m.norm.b,grouped,n_token,W,eps);
    }
    mlp(blas,grouped,mid,out,m.fc1,m.fc2,n_token,W,W,out_hidden,S,/*erf_gelu=*/true);
    cudaFreeAsync(normed,S);cudaFreeAsync(grouped,S);cudaFreeAsync(mid,S);
}

}  // namespace

// Batched vision tower: `num_img` images concatenated row-wise. pixel/rope_pos/
// pos_embed_interp are [Ntot, ...] (Ntot = Σ n_patch); the per-row layer kernels
// run over all Ntot rows at once (bigger GEMMs), attention is block-diagonal
// per image (flashinfer multi-seq), and the mergers loop per image. out_main /
// out_deep[d] are [Σ NTOK, OUT] with per-image segments. num_img==1 is the
// single-image case. Requires each image's patch count to be a multiple of the
// attention page size (16) so per-image rows are page-aligned in the KV view —
// the caller guarantees this (else it falls back to per-image calls).
void run_qwen3vl_vision(cublasHandle_t blas,const QwenVisRawWeights& w,
                        const bf* pixel,const float* rope_pos,const bf* pos_embed_interp,
                        int num_img,const int* n_patch_h,
                        bf* out_main,bf* const* out_deep,int num_deep,cudaStream_t S){
    const int Hd=w.hidden, NH=w.heads, HEAD=w.head_dim, IM=w.intermediate;
    const int OUT=w.out_hidden, MERGE=w.spatial_merge_size, U=MERGE*MERGE;
    const int PATCH_DIM=w.in_channels*w.temporal_patch_size*w.patch_size*w.patch_size; // 1536
    const float EPS=w.ln_eps, THETA=w.rope_theta;  // attn scale is applied inside flashinfer
    std::vector<int> off(num_img+1,0), tok(num_img+1,0);  // per-image row / token offsets
    for(int i=0;i<num_img;i++){ off[i+1]=off[i]+n_patch_h[i]; tok[i+1]=tok[i]+n_patch_h[i]/U; }
    const int N=off[num_img];                      // total patch rows (all images)
    if(Hd!=NH*HEAD) throw std::runtime_error("qwen3vl_vision: hidden != heads*head_dim");
    if(N%U!=0) throw std::runtime_error("qwen3vl_vision: n_patch not divisible by merge^2");

    auto MAL=[&](long n){bf* d;QCK(cudaMallocAsync(&d,n*sizeof(bf),S));return d;};
    bf *h=MAL((long)N*Hd),*hn=MAL((long)N*Hd),*qkv=MAL((long)N*3*Hd),
       *q=MAL((long)N*Hd),*k=MAL((long)N*Hd),*v=MAL((long)N*Hd),
       *attn=MAL((long)N*Hd),*tmp=MAL((long)N*Hd),
       *mid=MAL((long)N*IM);
    // (Attention scores are no longer materialized — flashinfer keeps them in
    // shared/registers — so the old scr/scr_bf buffers are gone.)

    // Optional phase timing (PIE_VIS_TIMING=1): patch / per-layer attention /
    // layers-total / merger, computed after the final sync so it doesn't perturb.
    const bool VTIM = std::getenv("PIE_VIS_TIMING")!=nullptr;
    // Ablation knob (PIE_VIS_GEMM_ONLY=1): skip the per-layer elementwise launches
    // (norm/split/rope/gelu/bias) so VTIM's `rest` measures GEMM-only time —
    // isolates tensor-core MFU from the memory-bound elementwise stalls between
    // GEMMs. Output is garbage under this flag; timing only.
    static const bool GEMM_ONLY = std::getenv("PIE_VIS_GEMM_ONLY")!=nullptr;
    const int NL=(int)w.blocks.size();
    cudaEvent_t e_p0=0,e_p1=0,e_le=0,e_me=0;
    std::vector<cudaEvent_t> aS,aE;
    auto MKEV=[&]{cudaEvent_t e;cudaEventCreate(&e);return e;};
    if(VTIM){ e_p0=MKEV();e_p1=MKEV();e_le=MKEV();e_me=MKEV();
        aS.resize(NL);aE.resize(NL); for(int i=0;i<NL;i++){aS[i]=MKEV();aE[i]=MKEV();}
        cudaEventRecord(e_p0,S); }

    // Patch embed: Conv3d-as-matmul [hidden, PATCH_DIM] (+bias) over pixel[N,PATCH_DIM].
    gemm_bias(blas,pixel,w.patch,h,N,Hd,PATCH_DIM,S);

    // Emit a checkpoint of the (hidden) state for parity debugging.
    auto emit_ckpt=[&](const char* nm,const bf* d,long n){
        if(!g_qvis_ckpt)return; QCK(cudaStreamSynchronize(S));
        g_qvis_ckpt(nm,d,n,g_qvis_ckpt_user); };

    // patch_embed dump (scripts/qwen3_vl_vision_parity_ref.py) hooks the
    // patch_embed module output, i.e. BEFORE the pos-embed add — emit here.
    emit_ckpt("patch_embed",h,(long)N*Hd);

    // Add the host-interpolated abs pos-embed (already [N, hidden]).
    k_addpos<<<((long)N*Hd+255)/256,256,0,S>>>(h,pos_embed_interp,(long)N*Hd);
    if(VTIM) cudaEventRecord(e_p1,S);

    int deep_written=0;
    for(int li=0; li<(int)w.blocks.size(); ++li){
        const QVisBlock& L=w.blocks[li];
        // ── attention (pre-norm: norm1 → qkv → rope → attn → o → residual) ──
        // Fused epilogues: qkv bias folds into the split, q/k rope share one
        // launch, the o-projection writes the residual directly (cuBLAS beta=1)
        // so h += attn@Wo^T in-place — only the o-bias remains as a kernel.
        if(!GEMM_ONLY) k_layernorm<<<N,256,0,S>>>(h,L.norm1.g,L.norm1.b,hn,N,Hd,EPS);
        qwen3vl_vis_gemm_bf16(blas,hn,L.qkv.w,qkv,N,3*Hd,Hd);
        // Fused split(+bias)+RoPE: one pass over qkv → q,k,v (q/k roped).
        if(!GEMM_ONLY) k_split_rope_qkv<<<dim3(NH,N),HEAD/2,0,S>>>(qkv,L.qkv.b,q,k,v,rope_pos,N,NH,HEAD,THETA);
        // Full bidirectional attention over all N patches (single image).
        if(VTIM) cudaEventRecord(aS[li],S);
        // Flash attention (flashinfer): block-diagonal per image (multi-seq), all
        // heads at once, no [N,N] gmem materialization. q/k already 2D-RoPE'd.
        qwen3vl_vis_attn(q, k, v, attn, num_img, n_patch_h, NH, HEAD, S);
        if(VTIM) cudaEventRecord(aE[li],S);
        qwen3vl_vis_gemm_bf16(blas,attn,L.o.w,h,N,Hd,Hd,/*beta=*/1.0f);
        if(!GEMM_ONLY && L.o.b) k_bias<<<((long)N*Hd+255)/256,256,0,S>>>(h,L.o.b,(long)N,Hd);
        // ── mlp (pre-norm: norm2 → fc1 → gelu(+bias) → fc2 → residual) ──
        if(!GEMM_ONLY) k_layernorm<<<N,256,0,S>>>(h,L.norm2.g,L.norm2.b,hn,N,Hd,EPS);
        qwen3vl_vis_gemm_bf16(blas,hn,L.fc1.w,mid,N,IM,Hd);
        if(!GEMM_ONLY) k_gelu_bias<<<((long)N*IM+255)/256,256,0,S>>>(mid,L.fc1.b,N,IM);
        qwen3vl_vis_gemm_bf16(blas,mid,L.fc2.w,h,N,Hd,IM,/*beta=*/1.0f);
        if(!GEMM_ONLY && L.fc2.b) k_bias<<<((long)N*Hd+255)/256,256,0,S>>>(h,L.fc2.b,(long)N,Hd);
        // ── deepstack tap (post-block, before next layer) — per image ──
        for(int d=0; d<(int)w.deepstack_layer_idx.size(); ++d){
            if(w.deepstack_layer_idx[d]==li && deep_written<num_deep){
                for(int im=0;im<num_img;++im){
                    const int ni=n_patch_h[im];
                    run_merger(blas,w.deepstack[d],h+(long)off[im]*Hd,ni,ni/U,Hd,U,OUT,EPS,
                               out_deep[deep_written]+(long)tok[im]*OUT,S);
                }
                deep_written++;
            }
        }
        // post-block hidden checkpoints for parity (layers {0,5,11,17,23}).
        {
            char nm[16]; std::snprintf(nm,sizeof(nm),"layer%d",li);
            emit_ckpt(nm,h,(long)N*Hd);
        }
    }
    emit_ckpt("last_hidden",h,(long)N*Hd);
    if(VTIM) cudaEventRecord(e_le,S);
    // main merger → out_main [Σ NTOK, OUT], per image.
    for(int im=0;im<num_img;++im){
        const int ni=n_patch_h[im];
        run_merger(blas,w.merger,h+(long)off[im]*Hd,ni,ni/U,Hd,U,OUT,EPS,out_main+(long)tok[im]*OUT,S);
    }
    if(VTIM) cudaEventRecord(e_me,S);

    if(VTIM){
        QCK(cudaStreamSynchronize(S));  // events need the stream drained to read times
        float patch=0,layers=0,merger=0,att=0;
        cudaEventElapsedTime(&patch,e_p0,e_p1);
        cudaEventElapsedTime(&layers,e_p1,e_le);
        cudaEventElapsedTime(&merger,e_le,e_me);
        for(int i=0;i<NL;i++){float a;cudaEventElapsedTime(&a,aS[i],aE[i]);att+=a;}
        fprintf(stderr,"[vtim] N=%d patch+pos=%.1fms layers=%.1fms (attn=%.1f rest=%.1f) merger=%.1fms\n",
                N,patch,layers,att,layers-att,merger);
        for(cudaEvent_t e:{e_p0,e_p1,e_le,e_me})cudaEventDestroy(e);
        for(int i=0;i<NL;i++){cudaEventDestroy(aS[i]);cudaEventDestroy(aE[i]);}
    }
    for(bf* b:{h,hn,qkv,q,k,v,attn,tmp,mid})cudaFreeAsync(b,S);
}

namespace {
__global__ void k_f32_to_bf16(const float* a, bf* o, long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) o[i]=Bf(a[i]);
}

// ── Host helpers mirroring transformers vision_utils (pixel order + side inputs).

// Build the spatial-merge reorder permutation `perm[k]` = source patch index for
// output position k, for one (t,h,w) grid. Mirrors the `reorder` in
// get_vision_bilinear_indices_and_weights / get_vision_position_ids:
//   reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :])
//             .transpose(1,2).flatten().repeat(t)
// where h_idx = arange(h).view(h/m, m), w_idx = arange(w).view(w/m, m).
// i.e. iterate blocks (bh, bw), then within-block (ih, iw):
//   src = bh*m*w + ih*w + bw*m + iw  (+ frame*h*w for t>1).
std::vector<int> merge_reorder(int t,int h,int w,int m){
    std::vector<int> perm; perm.reserve((long)t*h*w);
    for(int f=0; f<t; ++f)
        for(int bh=0; bh<h/m; ++bh)
            for(int bw=0; bw<w/m; ++bw)
                for(int ih=0; ih<m; ++ih)
                    for(int iw=0; iw<m; ++iw)
                        perm.push_back(f*h*w + (bh*m+ih)*w + (bw*m+iw));
    return perm;
}

// (row,col) RoPE position ids per patch, in spatial-merge order (== merge_reorder).
// row = (bh*m+ih), col = (bw*m+iw).  Matches get_vision_position_ids.
std::vector<float> vision_rope_positions(int t,int h,int w,int m){
    std::vector<float> pos; pos.reserve((long)t*h*w*2);
    for(int f=0; f<t; ++f)
        for(int bh=0; bh<h/m; ++bh)
            for(int bw=0; bw<w/m; ++bw)
                for(int ih=0; ih<m; ++ih)
                    for(int iw=0; iw<m; ++iw){
                        pos.push_back((float)(bh*m+ih));  // row
                        pos.push_back((float)(bw*m+iw));  // col
                    }
    return pos;
}

// Bilinear-interpolate the [num_pos_embed, hidden] abs pos-embed table to the
// (h,w) grid, in spatial-merge order, → [t*h*w, hidden].  Mirrors
// get_vision_bilinear_indices_and_weights + (pos_embed(idx)*weight).sum(0).
// `table_host` is the [num_pos_embed, hidden] table copied to host as float.
// PARITY TODO: verify against scripts/qwen3_vl_vision_parity_ref.py
// `pos_embed_interp` — esp. the linspace(0, side-1, h/w) endpoints and the
// floor/ceil clamp at side-1.
std::vector<float> interp_pos_embed(const std::vector<float>& table,int side,int hidden,
                                    int t,int h,int w,int m){
    auto idx=[&](int gh,int gw){return gh*side+gw;};
    // per-(row,col) interpolated embedding into a row-major [h*w, hidden] in
    // ROW-MAJOR (un-reordered) order, then gather via merge_reorder.
    std::vector<float> lin((long)h*w*hidden);
    auto frac=[&](int n,int i){ // linspace(0, side-1, n) value at i
        return (n==1)?0.f:(float)i*(float)(side-1)/(float)(n-1);};
    for(int i=0;i<h;i++){
        float hg=frac(h,i); int hf=(int)hg; int hc=std::min(hf+1,side-1); float hfr=hg-hf;
        for(int j=0;j<w;j++){
            float wg=frac(w,j); int wf=(int)wg; int wc=std::min(wf+1,side-1); float wfr=wg-wf;
            float w00=(1-hfr)*(1-wfr),w01=(1-hfr)*wfr,w10=hfr*(1-wfr),w11=hfr*wfr;
            const float* p00=&table[(long)idx(hf,wf)*hidden];
            const float* p01=&table[(long)idx(hf,wc)*hidden];
            const float* p10=&table[(long)idx(hc,wf)*hidden];
            const float* p11=&table[(long)idx(hc,wc)*hidden];
            float* o=&lin[((long)i*w+j)*hidden];
            for(int c=0;c<hidden;c++) o[c]=w00*p00[c]+w01*p01[c]+w10*p10[c]+w11*p11[c];
        }
    }
    // gather into spatial-merge order, repeated over t frames.
    std::vector<int> perm=merge_reorder(t,h,w,m);
    std::vector<float> out((long)perm.size()*hidden);
    for(size_t k=0;k<perm.size();++k){
        int src=perm[k]%(h*w);  // frame-independent (table is per-spatial position)
        std::copy(&lin[(long)src*hidden],&lin[(long)src*hidden+hidden],&out[(long)k*hidden]);
    }
    return out;
}
}  // namespace

void scatter_qwen3vl_vision(const Qwen3VLVisionInputs& vin, bf* hidden,
                            int n_rows, int out_hidden,
                            bf* deepstack_scratch, int num_deep,
                            cublasHandle_t blas, cudaStream_t S){
    if(vin.weights==nullptr || vin.num_images<=0) return;
    const QwenVisRawWeights& w=*vin.weights;
    const int Hd=w.hidden, MERGE=w.spatial_merge_size, U=MERGE*MERGE, OUT=out_hidden;
    const int PATCH_DIM=w.in_channels*w.temporal_patch_size*w.patch_size*w.patch_size;
    const bool VTIM = std::getenv("PIE_VIS_TIMING")!=nullptr;
    using clk=std::chrono::steady_clock;
    auto MS=[](clk::time_point a,clk::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();};

    // Zero the full deepstack scratch so the decoder can add it into hidden as a
    // plain whole-tensor residual-add (non-image rows contribute 0 — mirrors
    // HF `_deepstack_process` which only updates the visual rows).
    if(deepstack_scratch != nullptr && num_deep > 0){
        QCK(cudaMemsetAsync(deepstack_scratch, 0,
            (long)num_deep*n_rows*OUT*sizeof(bf), S));
    }

    // The abs pos-embed table is a constant model weight. Cache the host-side
    // fp32 copy (keyed by the device pointer) so we don't re-do the ~5 MB D2H +
    // convert on every forward pass. Forward passes for one model are serialized
    // by the engine; the mutex guards the (rare) first-touch.
    auto t_tbl0=clk::now();
    static std::mutex tbl_mu; static const void* tbl_key=nullptr; static std::vector<float> tbl_cache;
    {
        std::lock_guard<std::mutex> lk(tbl_mu);
        if(tbl_key != w.pos_embed){
            std::vector<bf> tbf((long)w.num_pos_embed*Hd);
            QCK(cudaMemcpy(tbf.data(),w.pos_embed,(long)w.num_pos_embed*Hd*sizeof(bf),cudaMemcpyDeviceToHost));
            tbl_cache.resize(tbf.size());
            for(size_t i=0;i<tbf.size();++i) tbl_cache[i]=__bfloat162float(tbf[i]);
            tbl_key = w.pos_embed;
        }
    }
    const std::vector<float>& table = tbl_cache;
    if(VTIM) fprintf(stderr,"[vtim] num_images=%d  pos_embed table = %.1fms\n", vin.num_images, MS(t_tbl0,clk::now()));

    double cpu_host_ms=0; auto t_scat0=clk::now();  // pure-CPU host work accumulator
    constexpr int PS=16;  // attention page size; per-image rows must be a multiple

    // rope positions + interpolated pos-embed are a deterministic function of the
    // grid — identical for every same-size image. Cache the device buffers by grid
    // so the CPU interp + bf16 convert + H2D run ONCE, not per image.
    static std::mutex pe_mu;
    static std::map<std::tuple<int,int,int>, std::pair<float*,bf*>> pe_cache;
    auto grid_rope_pe = [&](int gt,int gh,int gw)->std::pair<float*,bf*>{
        std::lock_guard<std::mutex> lk(pe_mu);
        auto key=std::make_tuple(gt,gh,gw); auto it=pe_cache.find(key);
        if(it!=pe_cache.end()) return it->second;
        auto c0=clk::now();
        std::vector<float> rope=vision_rope_positions(gt,gh,gw,MERGE);
        std::vector<float> pe=interp_pos_embed(table,w.num_grid_per_side,Hd,gt,gh,gw,MERGE);
        std::vector<bf> pe_bf(pe.size()); for(size_t i=0;i<pe.size();++i) pe_bf[i]=__float2bfloat16(pe[i]);
        cpu_host_ms+=MS(c0,clk::now());
        float* rd; bf* pd;
        QCK(cudaMalloc(&rd,(long)rope.size()*4)); QCK(cudaMemcpy(rd,rope.data(),(long)rope.size()*4,cudaMemcpyHostToDevice));
        QCK(cudaMalloc(&pd,(long)pe_bf.size()*sizeof(bf))); QCK(cudaMemcpy(pd,pe_bf.data(),(long)pe_bf.size()*sizeof(bf),cudaMemcpyHostToDevice));
        pe_cache[key]={rd,pd}; return {rd,pd};
    };

    // Uniform batch test: all images share a grid and are page-aligned → encode
    // them in ONE batched tower pass (bigger GEMMs, per-image block-diag attention).
    // OFF by default: measured ~6% SLOWER here — the per-image vision GEMMs are
    // already compute-efficient at M=n_patch, so batching to M=Σn_patch gives no
    // GEMM win while the multi-seq attention + larger buffers add overhead. Kept
    // (env-gated) since it's correct and could help many-tiny-image batches.
    bool uniform = (vin.num_images > 1) && std::getenv("PIE_VIS_BATCH_IMAGES") != nullptr;
    int g0t=0,g0h=0,g0w=0,np0=0;
    if(uniform){
        g0t=(int)vin.grids_h[0]; g0h=(int)vin.grids_h[1]; g0w=(int)vin.grids_h[2];
        np0=g0t*g0h*g0w;
        if(np0%PS!=0) uniform=false;
        for(int im=0; im<vin.num_images && uniform; ++im)
            if((int)vin.grids_h[3*im]!=g0t||(int)vin.grids_h[3*im+1]!=g0h||(int)vin.grids_h[3*im+2]!=g0w)
                uniform=false;
    }

    if(uniform){
        const int num_img=vin.num_images, n_patch=np0, n_token=n_patch/U;
        const long blo=vin.pixel_byte_indptr_h[0], bhi=vin.pixel_byte_indptr_h[num_img];
        const int n_floats=(int)((bhi-blo)/4);             // == num_img*n_patch*PATCH_DIM
        const float* pix_h=vin.pixels_h + blo/4;
        float* pix_f32_d; QCK(cudaMallocAsync(&pix_f32_d,(long)n_floats*4,S));
        QCK(cudaMemcpyAsync(pix_f32_d,pix_h,(long)n_floats*4,cudaMemcpyHostToDevice,S));
        bf* pix_bf_d; QCK(cudaMallocAsync(&pix_bf_d,(long)n_floats*sizeof(bf),S));
        k_f32_to_bf16<<<(n_floats+255)/256,256,0,S>>>(pix_f32_d,pix_bf_d,n_floats);
        // tile the cached single-grid rope/pe across the num_img concatenated rows.
        auto rp=grid_rope_pe(g0t,g0h,g0w);
        float* rope_d; QCK(cudaMallocAsync(&rope_d,(long)num_img*n_patch*2*4,S));
        bf* pe_d; QCK(cudaMallocAsync(&pe_d,(long)num_img*n_patch*Hd*sizeof(bf),S));
        for(int im=0;im<num_img;++im){
            QCK(cudaMemcpyAsync(rope_d+(long)im*n_patch*2, rp.first, (long)n_patch*2*4, cudaMemcpyDeviceToDevice,S));
            QCK(cudaMemcpyAsync(pe_d+(long)im*n_patch*Hd, rp.second, (long)n_patch*Hd*sizeof(bf), cudaMemcpyDeviceToDevice,S));
        }
        bf* main_d; QCK(cudaMallocAsync(&main_d,(long)num_img*n_token*OUT*sizeof(bf),S));
        std::vector<bf*> deep_d(num_deep,nullptr);
        for(int d=0;d<num_deep;++d) QCK(cudaMallocAsync(&deep_d[d],(long)num_img*n_token*OUT*sizeof(bf),S));
        std::vector<int> npv(num_img,n_patch);
        run_qwen3vl_vision(blas,w,pix_bf_d,rope_d,pe_d,num_img,npv.data(),main_d,deep_d.data(),num_deep,S);
        for(int im=0;im<num_img;++im){
            const std::uint32_t anchor=vin.anchor_rows_h[im];
            QCK(cudaMemcpyAsync(hidden+(long)anchor*OUT, main_d+(long)im*n_token*OUT,
                                (long)n_token*OUT*sizeof(bf), cudaMemcpyDeviceToDevice,S));
            for(int d=0;d<num_deep;++d)
                QCK(cudaMemcpyAsync(deepstack_scratch+(long)d*n_rows*OUT+(long)anchor*OUT,
                                    deep_d[d]+(long)im*n_token*OUT,
                                    (long)n_token*OUT*sizeof(bf), cudaMemcpyDeviceToDevice,S));
        }
        cudaFreeAsync(pix_f32_d,S);cudaFreeAsync(pix_bf_d,S);cudaFreeAsync(rope_d,S);cudaFreeAsync(pe_d,S);cudaFreeAsync(main_d,S);
        for(int d=0;d<num_deep;++d) cudaFreeAsync(deep_d[d],S);
    } else {
        // Per-image fallback (mixed grids or non-page-aligned): one run per image.
        for(int im=0; im<vin.num_images; ++im){
            const long blo=vin.pixel_byte_indptr_h[im], bhi=vin.pixel_byte_indptr_h[im+1];
            const int n_floats=(int)((bhi-blo)/4);
            const int n_patch=n_floats/PATCH_DIM;
            if(n_patch<=0) continue;
            const int gt=(int)vin.grids_h[3*im], gh=(int)vin.grids_h[3*im+1], gw=(int)vin.grids_h[3*im+2];
            const int n_token=n_patch/U;
            const std::uint32_t anchor=vin.anchor_rows_h[im];
            const float* pix_h=vin.pixels_h + blo/4;
            float* pix_f32_d; QCK(cudaMallocAsync(&pix_f32_d,(long)n_floats*4,S));
            QCK(cudaMemcpyAsync(pix_f32_d,pix_h,(long)n_floats*4,cudaMemcpyHostToDevice,S));
            bf* pix_bf_d; QCK(cudaMallocAsync(&pix_bf_d,(long)n_floats*sizeof(bf),S));
            k_f32_to_bf16<<<(n_floats+255)/256,256,0,S>>>(pix_f32_d,pix_bf_d,n_floats);
            auto rp=grid_rope_pe(gt,gh,gw);
            bf* main_d; QCK(cudaMallocAsync(&main_d,(long)n_token*OUT*sizeof(bf),S));
            std::vector<bf*> deep_d(num_deep,nullptr);
            for(int d=0;d<num_deep;++d) QCK(cudaMallocAsync(&deep_d[d],(long)n_token*OUT*sizeof(bf),S));
            int np1=n_patch;
            run_qwen3vl_vision(blas,w,pix_bf_d,rp.first,rp.second,1,&np1,main_d,deep_d.data(),num_deep,S);
            QCK(cudaMemcpyAsync(hidden+(long)anchor*OUT,main_d,(long)n_token*OUT*sizeof(bf),cudaMemcpyDeviceToDevice,S));
            for(int d=0;d<num_deep;++d)
                QCK(cudaMemcpyAsync(deepstack_scratch+(long)d*n_rows*OUT+(long)anchor*OUT,
                                    deep_d[d],(long)n_token*OUT*sizeof(bf),cudaMemcpyDeviceToDevice,S));
            cudaFreeAsync(pix_f32_d,S);cudaFreeAsync(pix_bf_d,S);cudaFreeAsync(main_d,S);
            for(int d=0;d<num_deep;++d) cudaFreeAsync(deep_d[d],S);
        }
    }
    // No sync here: all work is stream-ordered on S; the decoder layers that
    // follow (same stream) see the scattered embeddings, and the fire's final
    // sync settles the async pixel H2D before the host pixel buffer is reused.
    // Draining mid-forward only stalled the CPU from queuing the LLM layers.
    if(VTIM) QCK(cudaStreamSynchronize(S));  // debug timing needs the stream drained
    if(VTIM) fprintf(stderr,"[vtim] scatter total=%.1fms  pure-CPU host=%.1fms  (%d imgs, %s)\n",
                     MS(t_scat0,clk::now()), cpu_host_ms, vin.num_images, uniform?"batched":"per-image");
}

}  // namespace pie_cuda_driver::model
