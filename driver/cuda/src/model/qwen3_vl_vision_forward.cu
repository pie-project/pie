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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <math_constants.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_cuda_driver::model {

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

// y[n,o] = sum_k x[n,k]*W[o,k] (+ bias[o] if b!=nullptr).  W is row-major [O,K].
__global__ void k_matmul(const bf* x,const bf* W,const bf* b,bf* y,int N,int K,int O){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||o>=O)return;
    const bf* xr=x+(long)n*K;const bf* wr=W+(long)o*K;float a=b?F(b[o]):0.f;
    for(int k=0;k<K;k++)a+=F(xr[k])*F(wr[k]);y[(long)n*O+o]=Bf(a);}

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

// QK^T logits, accumulated in fp32 (HF/SDPA keep the score matrix in fp32 before
// the fp32 softmax), then scaled by 1/sqrt(head_dim).
__global__ void k_qk(const bf* q,const bf* k,float* s,int N,int NH,int head,int HEAD,float scale){
    int i=blockIdx.y*blockDim.y+threadIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N||j>=N)return;
    const bf* qi=q+((long)i*NH+head)*HEAD;const bf* kj=k+((long)j*NH+head)*HEAD;
    float a=0;for(int d=0;d<HEAD;d++)a+=F(qi[d])*F(kj[d]);s[(long)i*N+j]=a*scale;}

__global__ void k_softmax(float* s,int N){int i=blockIdx.x;if(i>=N)return;float* r=s+(long)i*N;
    float mx=-1e30f;for(int j=threadIdx.x;j<N;j+=blockDim.x)mx=fmaxf(mx,r[j]);
    for(int o=warpSize/2;o>0;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    __shared__ float wm[32],wsv[32],smx,ssum;if((threadIdx.x&31)==0)wm[threadIdx.x>>5]=mx;__syncthreads();
    if(threadIdx.x==0){float m=-1e30f;int nw=(blockDim.x+31)/32;for(int i2=0;i2<nw;i2++)m=fmaxf(m,wm[i2]);smx=m;}__syncthreads();
    float sm=0;for(int j=threadIdx.x;j<N;j+=blockDim.x){float e=__expf(r[j]-smx);r[j]=e;sm+=e;}
    for(int o=warpSize/2;o>0;o>>=1)sm+=__shfl_down_sync(0xffffffff,sm,o);if((threadIdx.x&31)==0)wsv[threadIdx.x>>5]=sm;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i2=0;i2<nw;i2++)t+=wsv[i2];ssum=t;}__syncthreads();
    float inv=1.f/ssum;for(int j=threadIdx.x;j<N;j+=blockDim.x)r[j]*=inv;}

// AV matmul, fp32 accumulation over the fp32 softmax probabilities.
__global__ void k_av(const float* s,const bf* v,bf* o,int N,int NH,int head,int HEAD){
    int n=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||d>=HEAD)return;
    const float* sr=s+(long)n*N;float a=0;for(int j=0;j<N;j++)a+=sr[j]*F(v[((long)j*NH+head)*HEAD+d]);
    o[((long)n*NH+head)*HEAD+d]=Bf(a);}

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
void mlp(const bf* in,bf* mid,bf* out,const QVisLinear& fc1,const QVisLinear& fc2,
         int N,int Din,int Dmid,int Dout,cudaStream_t S,bool erf_gelu=false){
    k_matmul<<<G2(Dmid,N),B2,0,S>>>(in,fc1.w,fc1.b,mid,N,Din,Dmid);
    if(erf_gelu) k_gelu_erf<<<((long)N*Dmid+255)/256,256,0,S>>>(mid,mid,(long)N*Dmid);
    else         k_gelu    <<<((long)N*Dmid+255)/256,256,0,S>>>(mid,mid,(long)N*Dmid);
    k_matmul<<<G2(Dout,N),B2,0,S>>>(mid,fc2.w,fc2.b,out,N,Dmid,Dout);
}

// Run one patch merger over `h[n_patch, hidden]` → `out[n_token, out_hidden]`.
//   main     : LayerNorm(hidden) on h → 2×2 group(→4*hidden) → fc1 → GELU → fc2.
//   deepstack: 2×2 group(→4*hidden) → LayerNorm(4*hidden) → fc1 → GELU → fc2.
void run_merger(const QVisMerger& m,const bf* h,int n_patch,int n_token,
                int hidden,int merge_unit,int out_hidden,float eps,bf* out,cudaStream_t S){
    const int W=merge_unit*hidden;  // 4096
    auto MAL=[&](long n){bf* d;QCK(cudaMalloc(&d,n*sizeof(bf)));return d;};
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
    mlp(grouped,mid,out,m.fc1,m.fc2,n_token,W,W,out_hidden,S,/*erf_gelu=*/true);
    cudaFree(normed);cudaFree(grouped);cudaFree(mid);
}

}  // namespace

void run_qwen3vl_vision(const QwenVisRawWeights& w,
                        const bf* pixel,const float* rope_pos,const bf* pos_embed_interp,
                        int grid_t,int grid_h,int grid_w,
                        bf* out_main,bf* const* out_deep,int num_deep,cudaStream_t S){
    const int Hd=w.hidden, NH=w.heads, HEAD=w.head_dim, IM=w.intermediate;
    const int OUT=w.out_hidden, MERGE=w.spatial_merge_size, U=MERGE*MERGE;
    const int PATCH_DIM=w.in_channels*w.temporal_patch_size*w.patch_size*w.patch_size; // 1536
    const float EPS=w.ln_eps, THETA=w.rope_theta, SCALE=1.f/sqrtf((float)HEAD);
    const int N=grid_t*grid_h*grid_w;             // n_patch
    const int NTOK=N/U;                            // merged tokens
    if(Hd!=NH*HEAD) throw std::runtime_error("qwen3vl_vision: hidden != heads*head_dim");
    if(NTOK*U!=N) throw std::runtime_error("qwen3vl_vision: n_patch not divisible by merge^2");

    auto MAL=[&](long n){bf* d;QCK(cudaMalloc(&d,n*sizeof(bf)));return d;};
    bf *h=MAL((long)N*Hd),*hn=MAL((long)N*Hd),*qkv=MAL((long)N*3*Hd),
       *q=MAL((long)N*Hd),*k=MAL((long)N*Hd),*v=MAL((long)N*Hd),
       *attn=MAL((long)N*Hd),*tmp=MAL((long)N*Hd),
       *mid=MAL((long)N*IM);
    float* scr;QCK(cudaMalloc(&scr,(long)N*N*4));

    // Patch embed: Conv3d-as-matmul [hidden, PATCH_DIM] (+bias) over pixel[N,PATCH_DIM].
    k_matmul<<<G2(Hd,N),B2,0,S>>>(pixel,w.patch.w,w.patch.b,h,N,PATCH_DIM,Hd);

    // Emit a checkpoint of the (hidden) state for parity debugging.
    auto emit_ckpt=[&](const char* nm,const bf* d,long n){
        if(!g_qvis_ckpt)return; QCK(cudaStreamSynchronize(S));
        g_qvis_ckpt(nm,d,n,g_qvis_ckpt_user); };

    // patch_embed dump (scripts/qwen3_vl_vision_parity_ref.py) hooks the
    // patch_embed module output, i.e. BEFORE the pos-embed add — emit here.
    emit_ckpt("patch_embed",h,(long)N*Hd);

    // Add the host-interpolated abs pos-embed (already [N, hidden]).
    k_addpos<<<((long)N*Hd+255)/256,256,0,S>>>(h,pos_embed_interp,(long)N*Hd);

    int deep_written=0;
    for(int li=0; li<(int)w.blocks.size(); ++li){
        const QVisBlock& L=w.blocks[li];
        // ── attention (pre-norm: norm1 → qkv → rope → attn → o → residual) ──
        k_layernorm<<<N,256,0,S>>>(h,L.norm1.g,L.norm1.b,hn,N,Hd,EPS);
        k_matmul<<<G2(3*Hd,N),B2,0,S>>>(hn,L.qkv.w,L.qkv.b,qkv,N,Hd,3*Hd);
        k_split_qkv<<<G2(Hd,N),B2,0,S>>>(qkv,q,k,v,N,Hd);
        dim3 rg(1,NH,N);  // one block per (n, head); blockDim.x covers head_dim/2
        k_rope_vis<<<rg,HEAD/2,0,S>>>(q,rope_pos,N,NH,HEAD,THETA);
        k_rope_vis<<<rg,HEAD/2,0,S>>>(k,rope_pos,N,NH,HEAD,THETA);
        // Full bidirectional attention over all N patches (single image / cu_seqlen).
        // PARITY TODO: for multi-image batched cu_seqlens, restrict each query to
        // its own image's key range; here one image == one full block.
        for(int hh=0;hh<NH;hh++){
            k_qk<<<G2(N,N),B2,0,S>>>(q,k,scr,N,NH,hh,HEAD,SCALE);
            k_softmax<<<N,256,0,S>>>(scr,N);
            k_av<<<G2(HEAD,N),B2,0,S>>>(scr,v,attn,N,NH,hh,HEAD);
        }
        k_matmul<<<G2(Hd,N),B2,0,S>>>(attn,L.o.w,L.o.b,tmp,N,Hd,Hd);
        k_add_inplace<<<((long)N*Hd+255)/256,256,0,S>>>(h,tmp,(long)N*Hd);
        // ── mlp (pre-norm: norm2 → fc1 → gelu → fc2 → residual) ──
        k_layernorm<<<N,256,0,S>>>(h,L.norm2.g,L.norm2.b,hn,N,Hd,EPS);
        mlp(hn,mid,tmp,L.fc1,L.fc2,N,Hd,IM,Hd,S);
        k_add_inplace<<<((long)N*Hd+255)/256,256,0,S>>>(h,tmp,(long)N*Hd);
        // ── deepstack tap (post-block, before next layer) ──
        for(int d=0; d<(int)w.deepstack_layer_idx.size(); ++d){
            if(w.deepstack_layer_idx[d]==li && deep_written<num_deep){
                run_merger(w.deepstack[d],h,N,NTOK,Hd,U,OUT,EPS,out_deep[deep_written],S);
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
    // main merger → out_main [NTOK, OUT].
    run_merger(w.merger,h,N,NTOK,Hd,U,OUT,EPS,out_main,S);

    QCK(cudaStreamSynchronize(S));
    for(bf* b:{h,hn,qkv,q,k,v,attn,tmp,mid})cudaFree(b);
    cudaFree(scr);
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
                            bf* deepstack_scratch, int num_deep, cudaStream_t S){
    if(vin.weights==nullptr || vin.num_images<=0) return;
    const QwenVisRawWeights& w=*vin.weights;
    const int Hd=w.hidden, MERGE=w.spatial_merge_size, U=MERGE*MERGE, OUT=out_hidden;
    const int PATCH_DIM=w.in_channels*w.temporal_patch_size*w.patch_size*w.patch_size;

    // Zero the full deepstack scratch so the decoder can add it into hidden as a
    // plain whole-tensor residual-add (non-image rows contribute 0 — mirrors
    // HF `_deepstack_process` which only updates the visual rows).
    if(deepstack_scratch != nullptr && num_deep > 0){
        QCK(cudaMemsetAsync(deepstack_scratch, 0,
            (long)num_deep*n_rows*OUT*sizeof(bf), S));
    }

    // Copy the abs pos-embed table to host once (for bilinear interp).
    std::vector<float> table((long)w.num_pos_embed*Hd);
    {
        std::vector<bf> tbf((long)w.num_pos_embed*Hd);
        QCK(cudaMemcpy(tbf.data(),w.pos_embed,(long)w.num_pos_embed*Hd*sizeof(bf),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<tbf.size();++i) table[i]=__bfloat162float(tbf[i]);
    }

    for(int im=0; im<vin.num_images; ++im){
        const long blo=vin.pixel_byte_indptr_h[im], bhi=vin.pixel_byte_indptr_h[im+1];
        const int n_floats=(int)((bhi-blo)/4);
        const int n_patch=n_floats/PATCH_DIM;
        if(n_patch<=0) continue;
        const int gt=(int)vin.grids_h[3*im], gh=(int)vin.grids_h[3*im+1], gw=(int)vin.grids_h[3*im+2];
        const int n_token=n_patch/U;
        const std::uint32_t anchor=vin.anchor_rows_h[im];
        const float* pix_h=vin.pixels_h + blo/4;

        // pixels f32 (host) → device → bf16.  NOTE: pixels are assumed already in
        // spatial-merge patch order (the host processor emits them so), matching
        // merge_reorder / vision_rope_positions below.
        // PARITY TODO: if the host emits row-major patches instead, gather them
        // through merge_reorder here before the encoder.
        float* pix_f32_d; QCK(cudaMalloc(&pix_f32_d,(long)n_floats*4));
        QCK(cudaMemcpyAsync(pix_f32_d,pix_h,(long)n_floats*4,cudaMemcpyHostToDevice,S));
        bf* pix_bf_d; QCK(cudaMalloc(&pix_bf_d,(long)n_floats*sizeof(bf)));
        k_f32_to_bf16<<<(n_floats+255)/256,256,0,S>>>(pix_f32_d,pix_bf_d,n_floats);

        // rope (row,col) positions (host) → device f32.
        std::vector<float> rope=vision_rope_positions(gt,gh,gw,MERGE);
        float* rope_d; QCK(cudaMalloc(&rope_d,(long)rope.size()*4));
        QCK(cudaMemcpyAsync(rope_d,rope.data(),(long)rope.size()*4,cudaMemcpyHostToDevice,S));

        // interpolated abs pos-embed (host) → device bf16.
        std::vector<float> pe=interp_pos_embed(table,w.num_grid_per_side,Hd,gt,gh,gw,MERGE);
        std::vector<bf> pe_bf(pe.size()); for(size_t i=0;i<pe.size();++i) pe_bf[i]=__float2bfloat16(pe[i]);
        bf* pe_d; QCK(cudaMalloc(&pe_d,(long)pe_bf.size()*sizeof(bf)));
        QCK(cudaMemcpyAsync(pe_d,pe_bf.data(),(long)pe_bf.size()*sizeof(bf),cudaMemcpyHostToDevice,S));

        // encode → main [n_token, OUT] + deepstack [num_deep][n_token, OUT].
        bf* main_d; QCK(cudaMalloc(&main_d,(long)n_token*OUT*sizeof(bf)));
        std::vector<bf*> deep_d(num_deep,nullptr);
        for(int d=0;d<num_deep;++d) QCK(cudaMalloc(&deep_d[d],(long)n_token*OUT*sizeof(bf)));

        run_qwen3vl_vision(w,pix_bf_d,rope_d,pe_d,gt,gh,gw,main_d,deep_d.data(),num_deep,S);

        // main → overwrite hidden[anchor .. anchor+n_token).
        QCK(cudaMemcpyAsync(hidden+(long)anchor*OUT,main_d,(long)n_token*OUT*sizeof(bf),
                            cudaMemcpyDeviceToDevice,S));
        // deepstack → block d at deepstack_scratch[d*n_rows*OUT + anchor*OUT ..].
        for(int d=0;d<num_deep;++d){
            bf* dst=deepstack_scratch + (long)d*n_rows*OUT + (long)anchor*OUT;
            QCK(cudaMemcpyAsync(dst,deep_d[d],(long)n_token*OUT*sizeof(bf),cudaMemcpyDeviceToDevice,S));
        }
        QCK(cudaStreamSynchronize(S));
        cudaFree(pix_f32_d);cudaFree(pix_bf_d);cudaFree(rope_d);cudaFree(pe_d);cudaFree(main_d);
        for(int d=0;d<num_deep;++d) cudaFree(deep_d[d]);
    }
}

}  // namespace pie_cuda_driver::model
