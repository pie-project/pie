// Mimi neural-codec DECODER forward (bf16): RVQ codes → 24 kHz waveform.
// See mimi_decoder_forward.hpp + AUDIO_OUTPUT.md.
//
// First-draft scaffold mirroring gemma4_audio_forward.cu (naive kernels +
// cudaMalloc scratch; bf16 storage + fp32 compute). The per-stage math is
// transcribed from transformers 5.9 `modeling_mimi.py` (MimiModel.decode /
// _decode_frame: MimiSplitResidualVectorQuantizer.decode → upsample
// (MimiConvTranspose1d) → decoder_transformer (MimiTransformerModel) → decoder
// (MimiDecoder SEANet)) and checked shape-wise against
// scripts/mimi_decoder_parity_ref.py
// (codes [1,32,25] → dequant [1,512,25] → upsample [1,512,50] → xf [1,50,512]
//  → SEANet → waveform [1,1,48000]).
//
// CUDA-only includes (no model/loader headers) so nvcc never sees toml++.
//
// PARITY: validated bf16-vs-bf16 against /tmp/mimi_decoder_parity/*.npy via
// tests/mimi_decoder_full_parity.cu — every staged checkpoint and the final
// 24 kHz waveform match the HF reference at cosine ≥ 0.9999 (output_waveform
// cosine 0.99993, rel_rms 1.19%). Per-stage cosines:
//   dequantized_embeddings 0.99999 · upsampled_embeddings 0.99999 ·
//   decoder_transformer_out 0.99996 · seanet_conv0 0.99999 ·
//   seanet_convtr0 0.99998 · seanet_conv_last 0.99993.
// The one real bug found+fixed: the decoder_transformer layer_scale was being
// applied along the TOKEN axis (k_layerscale_add_ct on a [Tu,HID] token-major
// tensor) instead of the per-CHANNEL feature axis — replaced with
// k_layerscale_add_rd (scale indexes the inner feature dim). All other drafted
// choices were confirmed CORRECT:
//   * RVQ SUM ordering — sum each group's 256-d residual over its codebooks,
//     then the 1×1 output_proj PER GROUP, then add the two 512-d results
//     (MimiResidualVectorQuantizer.decode + MimiSplitResidualVectorQuantizer).
//   * CONV PADDING/STRIDE — MimiConv1d causal left-pad padding_total =
//     (k-1)*dilation+1-stride (all decode convs are stride 1 → out len = T,
//     zero/constant pad); MimiConvTranspose1d trims padding_total from the
//     RIGHT (trim_right_ratio=1, padding_left=0) → out len = Tin*stride.
//   * decoder_transformer attention — full causal (window 250 ≥ T'=50 so it is
//     a no-op here) + RoPE (theta 10000, head_dim 64, rotate-half) confirmed.

#include "model/csm/mimi_decoder_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <math_constants.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_cuda_driver::model {

// ── Optional parity-debug checkpoint hook ────────────────────────────────────
// When set (by a standalone parity harness), it is invoked after each named
// pipeline stage with the stage's DEVICE bf16 buffer + element count, so the
// harness can copy back and compare against the HF staged dumps. No-op in
// production (pointer stays null; zero overhead). See
// tests/mimi_decoder_full_parity.cu.
typedef void (*MimiDecoderCkptFn)(const char* name, const __nv_bfloat16* dev,
                                  long numel, void* user);
static MimiDecoderCkptFn g_mimi_ckpt = nullptr;
static void* g_mimi_ckpt_user = nullptr;
void set_mimi_decoder_ckpt(MimiDecoderCkptFn fn, void* user) {
    g_mimi_ckpt = fn; g_mimi_ckpt_user = user;
}

namespace {

typedef __nv_bfloat16 bf;
#define MCK(x) do{cudaError_t e=(x);if(e)throw std::runtime_error(std::string("mimi_decoder: ")+cudaGetErrorString(e));}while(0)
__device__ __forceinline__ float F(bf x){return __bfloat162float(x);}
__device__ __forceinline__ bf   Bf(float x){return __float2bfloat16(x);}

dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}

// ── Elementwise / activation ─────────────────────────────────────────────────
__global__ void k_elu(const bf* x,bf* o,long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i>=n)return;
    float v=F(x[i]);o[i]=Bf(v>0.f?v:(__expf(v)-1.f));}     // ELU(alpha=1)
__global__ void k_gelu(const bf* x,bf* o,long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i>=n)return;
    float v=F(x[i]);
    // exact (erf) GELU — transformers ACT2FN["gelu"] is the erf form.
    o[i]=Bf(0.5f*v*(1.f+erff(v*0.70710678118f)));}
__global__ void k_add(bf* a,const bf* b,long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<n)a[i]=Bf(F(a[i])+F(b[i]));}
__global__ void k_bf16_to_f32(const bf* a,float* o,long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<n)o[i]=F(a[i]);}
// out[c,t] += scale[c] * x[c,t]  (per-CHANNEL layer scale on a [C, T] tensor)
__global__ void k_layerscale_add_ct(bf* res,const bf* x,const bf* scale,int C,int T){
    int c=blockIdx.y*blockDim.y+threadIdx.y,t=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=C||t>=T)return;long i=(long)c*T+t;res[i]=Bf(F(res[i])+F(scale[c])*F(x[i]));}
// res[r,d] += scale[d] * x[r,d]  (per-CHANNEL layer scale on a [R, D] token-major
// tensor: each ROW is a feature vector, so the scale indexes the INNER/feature
// dim d — this is what the decoder_transformer needs, [Tu, HID]).
__global__ void k_layerscale_add_rd(bf* res,const bf* x,const bf* scale,int R,int D){
    int r=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=R||d>=D)return;long i=(long)r*D+d;res[i]=Bf(F(res[i])+F(scale[d])*F(x[i]));}

// Standard matmul y[n,o] = sum_k x[n,k]*W[o,k]   (W is [O, K], row-major).
__global__ void k_matmul(const bf* x,const bf* W,bf* y,int N,int K,int O){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=N||o>=O)return;const bf* xr=x+(long)n*K;const bf* wr=W+(long)o*K;
    float a=0;for(int k=0;k<K;k++)a+=F(xr[k])*F(wr[k]);y[(long)n*O+o]=Bf(a);}

// LayerNorm over the feature axis (with weight + bias), on a [R, D] row-major
// tensor (one feature vector per row). transformers nn.LayerNorm.
__global__ void k_layernorm(const bf* x,const bf* w,const bf* b,bf* o,int R,int D,float eps){
    int r=blockIdx.x;if(r>=R)return;const bf* xr=x+(long)r*D;bf* orow=o+(long)r*D;
    float m=0;for(int d=threadIdx.x;d<D;d+=blockDim.x)m+=F(xr[d]);
    for(int s=warpSize/2;s>0;s>>=1)m+=__shfl_down_sync(0xffffffff,m,s);
    __shared__ float wm[32],wv[32],mean,inv;if((threadIdx.x&31)==0)wm[threadIdx.x>>5]=m;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=wm[i];mean=t/D;}__syncthreads();
    float v=0;for(int d=threadIdx.x;d<D;d+=blockDim.x){float dd=F(xr[d])-mean;v+=dd*dd;}
    for(int s=warpSize/2;s>0;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
    if((threadIdx.x&31)==0)wv[threadIdx.x>>5]=v;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=wv[i];inv=rsqrtf(t/D+eps);}__syncthreads();
    for(int d=threadIdx.x;d<D;d+=blockDim.x)orow[d]=Bf((F(xr[d])-mean)*inv*F(w[d])+F(b[d]));}

// ── RVQ dequantize ───────────────────────────────────────────────────────────
// For one RVQ GROUP: sum the per-codebook embedding rows (codebook_dim=256)
// over its codebooks, into a [codebook_dim, T] residual (channels-first to feed
// the 1×1 output_proj conv). codes is host-side [num_codebooks, T]; this kernel
// reads the device-uploaded code ids for the group's codebooks.
// PARITY TODO: confirm the RVQ sum is plain addition across codebooks (no
// per-layer residual subtraction on DECODE — that only happens on ENCODE).
__global__ void k_rvq_sum_group(const bf* const* embeds,   // [n_cb][2048,256] ptrs
                                const int* codes_d,         // [n_cb, T] device
                                bf* out,                    // [256, T]
                                int n_cb,int T,int dim){
    int d=blockIdx.y*blockDim.y+threadIdx.y,t=blockIdx.x*blockDim.x+threadIdx.x;
    if(d>=dim||t>=T)return;
    float acc=0;
    for(int c=0;c<n_cb;c++){
        int idx=codes_d[(long)c*T+t];                       // code id in [0,2048)
        acc+=F(embeds[c][(long)idx*dim+d]);                 // embed row [idx, d]
    }
    out[(long)d*T+t]=Bf(acc);                               // channels-first [dim,T]
}

// 1×1 conv (pointwise) over a [Cin, T] channels-first tensor: out[co,t] =
// sum_ci W[co,ci]*x[ci,t]. Weight is [Cout, Cin, 1] → treat as [Cout, Cin].
__global__ void k_conv1x1(const bf* x,const bf* W,bf* out,int Cin,int Cout,int T){
    int co=blockIdx.y*blockDim.y+threadIdx.y,t=blockIdx.x*blockDim.x+threadIdx.x;
    if(co>=Cout||t>=T)return;
    float a=0;for(int ci=0;ci<Cin;ci++)a+=F(x[(long)ci*T+t])*F(W[(long)co*Cin+ci]);
    out[(long)co*T+t]=Bf(a);}

// ── SEANet causal Conv1d ─────────────────────────────────────────────────────
// Causal MimiConv1d on a [Cin, T] channels-first tensor → [Cout, To].
// Left pad = padding_total = (k-1)*dil + 1 - stride; right extra padding is
// added so the output length matches torch. Output length To = T (stride 1) or
// the strided count. PARITY TODO: verify _get_extra_padding_for_conv1d + the
// causal left-pad amount + constant(zero) pad mode vs torch.
__global__ void k_conv1d_causal(const bf* x,const bf* W,const bf* b,bf* out,
                                 int Cin,int Cout,int T,int To,int k,int stride,
                                 int dilation,int pad_left){
    int co=blockIdx.z;int to=blockIdx.y*blockDim.y+threadIdx.y;
    if(co>=Cout||to>=To)return;
    float acc=b?F(b[co]):0.f;
    int base=to*stride-pad_left;                            // start index in padded coords
    for(int ci=0;ci<Cin;ci++){
        const bf* wk=W+((long)co*Cin+ci)*k;
        for(int j=0;j<k;j++){
            int ti=base+j*dilation;
            if(ti<0||ti>=T)continue;                        // zero (constant) pad
            acc+=F(x[(long)ci*T+ti])*F(wk[j]);
        }
    }
    out[(long)co*To+to]=Bf(acc);
}

// ── SEANet transposed Conv1d ─────────────────────────────────────────────────
// MimiConvTranspose1d on [Cin, T] → [Cout, To]. groups=1 for the SEANet
// upsamplers; groups=Cout=Cin for the codec `upsample` (depthwise). Raw
// transposed-conv length is (T-1)*stride + k; we then trim padding_left from the
// left and padding_right(=padding_total) from the right (trim_right_ratio=1).
// PARITY TODO: verify the grouped weight layout [Cin, Cout/groups, k] and the
// left/right trim amounts vs torch ConvTranspose1d + Mimi's unpad.
__global__ void k_convtr1d(const bf* x,const bf* W,const bf* b,bf* out,
                           int Cin,int Cout,int Tin,int To,int k,int stride,
                           int groups,int trim_left){
    int co=blockIdx.z;int to=blockIdx.y*blockDim.y+threadIdx.y;
    if(co>=Cout||to>=To)return;
    int cpg_in=Cin/groups, cpg_out=Cout/groups;             // channels per group
    int grp=co/cpg_out, co_in_grp=co%cpg_out;
    float acc=b?F(b[co]):0.f;
    // raw position in the (T-1)*stride + k transposed output, before trimming
    int raw=to+trim_left;
    for(int ci_g=0;ci_g<cpg_in;ci_g++){
        int ci=grp*cpg_in+ci_g;
        // W layout [Cin, Cout/groups, k]
        const bf* wk=W+(((long)ci*cpg_out+co_in_grp)*k);
        for(int j=0;j<k;j++){
            int num=raw-j;                                  // = ti*stride
            if(num<0)continue; if(num%stride!=0)continue;
            int ti=num/stride; if(ti<0||ti>=Tin)continue;
            acc+=F(x[(long)ci*Tin+ti])*F(wk[j]);
        }
    }
    out[(long)co*To+to]=Bf(acc);
}

// Causal-conv output length (stride>=1): mirrors MimiConv1d after left+right pad.
// With causal left-pad = padding_total and right-extra padding, torch keeps
// To = ceil(T / stride) for stride>1 and To = T for stride 1. We compute the
// exact padded length here.  PARITY TODO: confirm extra-padding rounding.
inline int causal_conv_out_len(int T,int k,int stride,int /*dilation*/){
    // All forward MimiConv1d on the DECODE path are stride 1 (upsampling is via
    // MimiConvTranspose1d), so To = T — validated to parity. The stride>1 branch
    // (To = ceil(T/stride)) is unexercised by decode; only the encoder uses it.
    if(stride==1) return T;
    return (T + stride - 1)/stride;
}

}  // namespace

// ── decoder_transformer: one layer (naive full-causal attention + RoPE) ──────
// Reference: MimiTransformerLayer. hidden 512, heads 8, head_dim 64. Works on a
// [T, hidden] row-major tensor (sequence-major). PARITY TODO: sliding-window
// 250 mask + exact RoPE; layer_scale before residual.
namespace {
__global__ void k_rope_inplace(bf* q,bf* kk,int T,int H,int KVH,int hd,float theta){
    int t=blockIdx.y*blockDim.y+threadIdx.y;int e=blockIdx.x*blockDim.x+threadIdx.x;
    int half=hd/2; if(t>=T||e>=half)return;
    float inv=__powf(theta,-2.f*e/hd);float ang=t*inv;float c=__cosf(ang),s=__sinf(ang);
    // rotate-half layout (transformers): pairs (e, e+half).
    for(int h=0;h<H;h++){long b=((long)t*H+h)*hd;
        float a=F(q[b+e]),d=F(q[b+e+half]);q[b+e]=Bf(a*c-d*s);q[b+e+half]=Bf(d*c+a*s);}
    for(int h=0;h<KVH;h++){long b=((long)t*KVH+h)*hd;
        float a=F(kk[b+e]),d=F(kk[b+e+half]);kk[b+e]=Bf(a*c-d*s);kk[b+e+half]=Bf(d*c+a*s);}
}
// Naive causal attention with GQA + sliding window. out[t,h,:] over keys
// j in [max(0,t-window+1), t]. scaling = 1/sqrt(head_dim).
__global__ void k_attn(const bf* q,const bf* kk,const bf* v,bf* out,
                       int T,int H,int KVH,int hd,int window,float scale){
    int h=blockIdx.y,t=blockIdx.x*blockDim.x+threadIdx.x;if(h>=H||t>=T)return;
    int kvh=h/(H/KVH);
    float acc[64]; for(int d=0;d<hd;d++)acc[d]=0.f;
    int lo=t-window+1; if(lo<0)lo=0;
    float mx=-1e30f;
    for(int j=lo;j<=t;j++){float s=0;
        for(int d=0;d<hd;d++)s+=F(q[((long)t*H+h)*hd+d])*F(kk[((long)j*KVH+kvh)*hd+d]);
        s*=scale; mx=fmaxf(mx,s);}
    float denom=0;
    for(int j=lo;j<=t;j++){float s=0;
        for(int d=0;d<hd;d++)s+=F(q[((long)t*H+h)*hd+d])*F(kk[((long)j*KVH+kvh)*hd+d]);
        s*=scale;float w=__expf(s-mx);denom+=w;
        for(int d=0;d<hd;d++)acc[d]+=w*F(v[((long)j*KVH+kvh)*hd+d]);}
    float inv=denom>0.f?1.f/denom:0.f;
    for(int d=0;d<hd;d++)out[((long)t*H+h)*hd+d]=Bf(acc[d]*inv);
}
// Transpose [C, T] (channels-first) ↔ [T, C] (sequence-major).
__global__ void k_transpose(const bf* in,bf* out,int A,int Bn){
    int a=blockIdx.y*blockDim.y+threadIdx.y,b=blockIdx.x*blockDim.x+threadIdx.x;
    if(a>=A||b>=Bn)return;out[(long)b*A+a]=in[(long)a*Bn+b];}
}  // namespace

int run_mimi_decoder(const MimiDecoderRawWeights& w,
                     const std::int32_t* codes,int n_frames,
                     float* out_wave,cudaStream_t S){
    const int DIM=w.codebook_dim, HID=w.hidden, T=n_frames, NCB=w.num_codebooks;
    const float EPS=w.norm_eps;
    auto MAL=[&](long n){bf* d;MCK(cudaMalloc(&d,n*sizeof(bf)));return d;};
    // Parity-debug checkpoint: sync the stream then hand the device buffer to
    // the registered hook (no-op when none is set).
    auto CK=[&](const char* nm,const bf* d,long n){
        if(!g_mimi_ckpt)return; MCK(cudaStreamSynchronize(S));
        g_mimi_ckpt(nm,d,n,g_mimi_ckpt_user); };

    // ── upload codes [NCB, T] (host i32) → device ───────────────────────────
    int* codes_d; MCK(cudaMalloc(&codes_d,(long)NCB*T*sizeof(int)));
    MCK(cudaMemcpyAsync(codes_d,codes,(long)NCB*T*sizeof(int),cudaMemcpyHostToDevice,S));

    // codebook embed pointer table on device.
    const bf** embeds_d; MCK(cudaMalloc(&embeds_d,NCB*sizeof(bf*)));
    MCK(cudaMemcpyAsync(embeds_d,w.codebook_embed.data(),NCB*sizeof(bf*),
                        cudaMemcpyHostToDevice,S));

    // ── 1) RVQ dequantize: sum per group, output_proj 256→512, add groups ───
    // PARITY TODO: see file-top note on the SUM + per-group output_proj order.
    const int NS=w.num_semantic, NA=NCB-NS;
    bf* sem256=MAL((long)DIM*T); bf* aco256=MAL((long)DIM*T);
    { dim3 g((T+15)/16,(DIM+15)/16);
      k_rvq_sum_group<<<g,B2,0,S>>>(embeds_d, codes_d, sem256, NS, T, DIM); }
    { dim3 g((T+15)/16,(DIM+15)/16);
      k_rvq_sum_group<<<g,B2,0,S>>>(embeds_d+NS, codes_d+(long)NS*T, aco256, NA, T, DIM); }
    bf* emb=MAL((long)HID*T); bf* aco512=MAL((long)HID*T);
    { dim3 g((T+15)/16,(HID+15)/16);
      k_conv1x1<<<g,B2,0,S>>>(sem256,w.semantic_output_proj,emb,DIM,HID,T);
      k_conv1x1<<<g,B2,0,S>>>(aco256,w.acoustic_output_proj,aco512,DIM,HID,T); }
    k_add<<<((long)HID*T+255)/256,256,0,S>>>(emb,aco512,(long)HID*T);   // [512, T]
    CK("dequantized_embeddings",emb,(long)HID*T);

    // ── 2) upsample (ConvTranspose1d k4 s2 groups=512) → [512, 2T] ──────────
    const int Tu=T*w.upsample.stride;     // stride 2
    bf* up=MAL((long)HID*Tu);
    { dim3 g(1,(Tu+15)/16,HID);
      // trim_left = padding_total - padding_right; for k4 s2: total=2, right=2,
      // left=0 (trim_right_ratio=1). PARITY TODO: confirm trim arithmetic.
      int total=w.upsample.kernel-w.upsample.stride; int trim_left=total - total; // =0
      k_convtr1d<<<g,B2,0,S>>>(emb,w.upsample.w,w.upsample.b,up,
                               HID,HID,T,Tu,w.upsample.kernel,w.upsample.stride,
                               w.upsample.groups,trim_left); }
    CK("upsampled_embeddings",up,(long)HID*Tu);

    // ── 3) decoder_transformer (8 layers) on [Tu, hidden] sequence-major ────
    bf* hs=MAL((long)Tu*HID);
    { dim3 g((Tu+15)/16,(HID+15)/16); k_transpose<<<g,B2,0,S>>>(up,hs,HID,Tu); }
    const int H=w.xf_heads, KVH=w.xf_kv_heads, hd=w.xf_head_dim, IM=w.xf_intermediate;
    const float ascale=1.f/sqrtf((float)hd);
    bf *ln=MAL((long)Tu*HID),*q=MAL((long)Tu*H*hd),*kk=MAL((long)Tu*KVH*hd),
       *vv=MAL((long)Tu*KVH*hd),*ao=MAL((long)Tu*H*hd),*proj=MAL((long)Tu*HID),
       *mid=MAL((long)Tu*IM),*mlp=MAL((long)Tu*HID);
    for(const auto& L:w.xf_layers){
        // self-attn (pre-norm)
        k_layernorm<<<Tu,128,0,S>>>(hs,L.in_ln_w,L.in_ln_b,ln,Tu,HID,EPS);
        k_matmul<<<G2(H*hd,Tu),B2,0,S>>>(ln,L.q,q,Tu,HID,H*hd);
        k_matmul<<<G2(KVH*hd,Tu),B2,0,S>>>(ln,L.k,kk,Tu,HID,KVH*hd);
        k_matmul<<<G2(KVH*hd,Tu),B2,0,S>>>(ln,L.v,vv,Tu,HID,KVH*hd);
        { dim3 g((hd/2+15)/16,(Tu+15)/16);
          k_rope_inplace<<<g,B2,0,S>>>(q,kk,Tu,H,KVH,hd,w.xf_rope_theta); }
        { dim3 g((Tu+127)/128,H); k_attn<<<g,128,0,S>>>(q,kk,vv,ao,Tu,H,KVH,hd,
                                                        w.xf_sliding_window,ascale); }
        k_matmul<<<G2(HID,Tu),B2,0,S>>>(ao,L.o,proj,Tu,H*hd,HID);
        // layer_scale (per-channel) then residual. proj/hs are [Tu, HID]
        // token-major, so the scale indexes the inner (feature) dim.
        { dim3 g((HID+15)/16,(Tu+15)/16);
          // res[t,c] += scale[c]*proj[t,c]
          k_layerscale_add_rd<<<g,B2,0,S>>>(hs,proj,L.attn_scale,Tu,HID); }
        // mlp (pre-norm)
        k_layernorm<<<Tu,128,0,S>>>(hs,L.post_ln_w,L.post_ln_b,ln,Tu,HID,EPS);
        k_matmul<<<G2(IM,Tu),B2,0,S>>>(ln,L.fc1,mid,Tu,HID,IM);
        k_gelu<<<((long)Tu*IM+255)/256,256,0,S>>>(mid,mid,(long)Tu*IM);
        k_matmul<<<G2(HID,Tu),B2,0,S>>>(mid,L.fc2,mlp,Tu,IM,HID);
        { dim3 g((HID+15)/16,(Tu+15)/16);
          k_layerscale_add_rd<<<g,B2,0,S>>>(hs,mlp,L.mlp_scale,Tu,HID); }
    }
    if(w.xf_final_ln_w) k_layernorm<<<Tu,128,0,S>>>(hs,w.xf_final_ln_w,w.xf_final_ln_b,hs,Tu,HID,EPS);
    CK("decoder_transformer_out",hs,(long)Tu*HID);   // [Tu, HID] sequence-major
    // back to channels-first [HID, Tu] for the SEANet stack.
    bf* feat=MAL((long)HID*Tu);
    { dim3 g((HID+15)/16,(Tu+15)/16); k_transpose<<<g,B2,0,S>>>(hs,feat,Tu,HID); }

    // ── 4) SEANet decoder ────────────────────────────────────────────────────
    // helper: causal conv on channels-first [Cin, Tcur] → [Cout, To].
    auto conv=[&](const bf* x,const MimiConvRaw& c,int Tcur,int& To)->bf*{
        To=causal_conv_out_len(Tcur,c.kernel,c.stride,c.dilation);
        int eff=(c.kernel-1)*c.dilation+1; int pad_left=eff-c.stride; if(pad_left<0)pad_left=0;
        bf* o=MAL((long)c.out_ch*To);
        dim3 g(1,(To+15)/16,c.out_ch);
        k_conv1d_causal<<<g,B2,0,S>>>(x,c.w,c.b,o,c.in_ch,c.out_ch,Tcur,To,
                                      c.kernel,c.stride,c.dilation,pad_left);
        return o; };

    int Tc; bf* cur=conv(feat,w.seanet_in,Tu,Tc);      // Conv1d k7 512→1024
    int Cc=w.seanet_in.out_ch;
    CK("seanet_conv0",cur,(long)Cc*Tc);

    bool first_stage=true;
    for(const auto& st:w.seanet_stages){
        // ELU
        k_elu<<<((long)Cc*Tc+255)/256,256,0,S>>>(cur,cur,(long)Cc*Tc);
        // ConvTranspose1d (ratio*2 kernel, stride=ratio), groups=1
        const auto& ct=st.convtr;
        int Tt=(Tc)*ct.stride;                         // upsampled length (pre-trim)
        // raw transposed length = (Tc-1)*stride + k; trim total=k-stride from
        // right + left. Mimi trims padding_left from left, padding_right(=total)
        // from right (trim_right_ratio=1 → right=total, left=0). To = Tc*stride.
        int total=ct.kernel-ct.stride; int trim_left=total-total; // =0
        bf* to=MAL((long)ct.out_ch*Tt);
        { dim3 g(1,(Tt+15)/16,ct.out_ch);
          k_convtr1d<<<g,B2,0,S>>>(cur,ct.w,ct.b,to,ct.in_ch,ct.out_ch,Tc,Tt,
                                   ct.kernel,ct.stride,ct.groups,trim_left); }
        cudaFree(cur); cur=to; Cc=ct.out_ch; Tc=Tt;
        if(first_stage){ CK("seanet_convtr0",cur,(long)Cc*Tc); first_stage=false; }
        // MimiResnetBlock: out = x + conv2(ELU(conv1(ELU(x))))
        bf* res=MAL((long)Cc*Tc); MCK(cudaMemcpyAsync(res,cur,(long)Cc*Tc*sizeof(bf),cudaMemcpyDeviceToDevice,S));
        k_elu<<<((long)Cc*Tc+255)/256,256,0,S>>>(cur,cur,(long)Cc*Tc);
        int T1; bf* h1=conv(cur,st.resnet.conv1,Tc,T1);   // dim→dim/compress, k3
        k_elu<<<((long)st.resnet.conv1.out_ch*T1+255)/256,256,0,S>>>(h1,h1,(long)st.resnet.conv1.out_ch*T1);
        int T2; bf* h2=conv(h1,st.resnet.conv2,T1,T2);     // →dim, k1
        k_add<<<((long)Cc*Tc+255)/256,256,0,S>>>(res,h2,(long)Cc*Tc);
        cudaFree(cur); cudaFree(h1); cudaFree(h2); cur=res;
    }
    // final ELU + Conv1d k3 64→1
    k_elu<<<((long)Cc*Tc+255)/256,256,0,S>>>(cur,cur,(long)Cc*Tc);
    int Tw; bf* wave_bf=conv(cur,w.seanet_out,Tc,Tw);      // [1, n_samples]
    CK("seanet_conv_last",wave_bf,(long)Tw);
    cudaFree(cur);

    // ── output: bf16 [1, Tw] → f32 device waveform ──────────────────────────
    k_bf16_to_f32<<<(Tw+255)/256,256,0,S>>>(wave_bf,out_wave,Tw);
    MCK(cudaStreamSynchronize(S));

    for(bf* b:{sem256,aco256,emb,aco512,up,hs,ln,q,kk,vv,ao,proj,mid,mlp,feat,wave_bf})
        cudaFree(b);
    cudaFree(codes_d); cudaFree(embeds_d);
    return Tw;
}

}  // namespace pie_cuda_driver::model
