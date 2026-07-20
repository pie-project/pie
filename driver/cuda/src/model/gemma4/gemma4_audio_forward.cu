// Gemma-4 audio encoder forward (bf16). See gemma4_audio_forward.hpp.
//
// First-draft scaffold mirroring gemma4_vision_forward.cu (naive kernels +
// cudaMalloc scratch; bf16 storage + fp32 compute, matching the driver). The
// per-stage math is transcribed from transformers 5.9 `modeling_gemma4.py`
// (Gemma4AudioModel / Gemma4AudioLayer / Gemma4AudioAttention /
// Gemma4AudioLightConv1d / Gemma4AudioSubSampleConvProjection) and checked
// shape-wise against scripts/gemma4_audio_parity_ref.py
// (199 mel frames → SSCP → 50 frames → 50 tokens → [50, 2560]).
//
// CUDA-only includes (no model/loader headers) so nvcc never sees toml++.
//
// PARITY (bf16-vs-bf16 verified — standalone harness vs /tmp/gemma4_audio_parity/*.npy:
// sscp 1.00000 / layer0 0.99999 / layer5 0.99998 / layer11 0.99957 /
// encoder_out 0.99925 / projected 0.99924, matching HF's own bf16 stability):
//   * chunked-attention masking — the HF 5D blocked local mask
//     (chunk 12 / past 12 / future 0) + `_rel_shift` collapses, for this config,
//     to a plain causal sliding window: query t attends keys j with
//     0 <= t-j < max_past (= context_left-1 = 12). matrix_bd uses the sinusoidal
//     relative-position embedding for distance (t-j), which after relative_k_proj
//     lives at pe row (P-1)-(t-j) [P = max_past+1 = 13]. Implemented exactly in
//     k_local_attn (flat O(N^2); verified flat-vs-HF-blocked to <1e-6).
//   * conv-module — GLU split, depthwise CAUSAL conv (left-pad kernel-1), and
//     the post-conv clamp→RMSNorm(conv_norm)→silu ordering.
//   * subsampling stride math — Conv2d(k3,s2,p1) twice over (time,freq); the
//     LayerNorm is over the CHANNEL axis (permute to channels-last) then ReLU.

#include "model/gemma4/gemma4_audio_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <math_constants.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_cuda_driver::model {

// ── Per-stage checkpoint hook (parity debugging). See .hpp. ─────────────────
static Gemma4AudioCkptFn g_audio_ckpt = nullptr;
static void* g_audio_ckpt_user = nullptr;
void set_gemma4_audio_ckpt(Gemma4AudioCkptFn fn, void* user) {
    g_audio_ckpt = fn; g_audio_ckpt_user = user;
}

namespace {

typedef __nv_bfloat16 bf;
#define ACK(x) do{cudaError_t e=(x);if(e)throw std::runtime_error(std::string("gemma4_audio: ")+cudaGetErrorString(e));}while(0)
__device__ __forceinline__ float F(bf x){return __bfloat162float(x);}
__device__ __forceinline__ bf   Bf(float x){return __float2bfloat16(x);}

class DeviceScratch {
public:
    ~DeviceScratch() {
        for (void* pointer : allocations_) {
            if (pointer != nullptr) cudaFree(pointer);
        }
    }

    template <typename T>
    T* alloc(long count) {
        T* pointer = nullptr;
        ACK(cudaMalloc(&pointer, count * sizeof(T)));
        allocations_.push_back(pointer);
        return pointer;
    }

private:
    std::vector<void*> allocations_;
};

// ── Shared elementwise / GEMM / norm kernels (vision-style) ──────────────────
__global__ void k_matmul(const bf* x,const bf* W,bf* y,int N,int K,int O){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||o>=O)return;
    const bf* xr=x+(long)n*K;const bf* wr=W+(long)o*K;float a=0;for(int k=0;k<K;k++)a+=F(xr[k])*F(wr[k]);y[(long)n*O+o]=Bf(a);}
__global__ void k_matmul_bias(const bf* x,const bf* W,const bf* b,bf* y,int N,int K,int O){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||o>=O)return;
    const bf* xr=x+(long)n*K;const bf* wr=W+(long)o*K;float a=b?F(b[o]):0.f;for(int k=0;k<K;k++)a+=F(xr[k])*F(wr[k]);y[(long)n*O+o]=Bf(a);}
__global__ void k_clamp(const bf* x,bf* o,const bf* lo,const bf* hi,long t){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i>=t)return;
    float v=F(x[i]),l=lo?F(*lo):-CUDART_INF_F,h=hi?F(*hi):CUDART_INF_F;o[i]=Bf(v<l?l:(v>h?h:v));}
// Plain RMSNorm: `w` may be null (parameterless, gamma=1).
__global__ void k_rms(const bf* x,const bf* w,bf* o,int R,int D,float eps){
    int r=blockIdx.x;if(r>=R)return;const bf* xr=x+(long)r*D;bf* orow=o+(long)r*D;
    float loc=0;for(int d=threadIdx.x;d<D;d+=blockDim.x){float v=F(xr[d]);loc+=v*v;}
    for(int s=warpSize/2;s>0;s>>=1)loc+=__shfl_down_sync(0xffffffff,loc,s);
    __shared__ float warp[32],ss;if((threadIdx.x&31)==0)warp[threadIdx.x>>5]=loc;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=warp[i];ss=rsqrtf(t/D+eps);}__syncthreads();
    float inv=ss;for(int d=threadIdx.x;d<D;d+=blockDim.x)orow[d]=Bf(F(xr[d])*inv*(w?F(w[d]):1.f));}
__global__ void k_silu(const bf* x,bf* o,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t){float v=F(x[i]);o[i]=Bf(v/(1.f+__expf(-v)));}}
__global__ void k_f32_to_bf16(const float* a,bf* o,long n){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<n)o[i]=Bf(a[i]);}
// out = a + scale*b   (residual add with macaron weight).
__global__ void k_axpy(bf* a,const bf* b,float scale,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t)a[i]=Bf(F(a[i])+scale*F(b[i]));}
__global__ void k_add(bf* a,const bf* b,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t)a[i]=Bf(F(a[i])+F(b[i]));}
// GLU over last dim: o[n,d] = x[n,d] * sigmoid(x[n, d+D]) for d in [0,D).
__global__ void k_glu(const bf* x,bf* o,int N,int D){
    int n=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||d>=D)return;
    float a=F(x[(long)n*2*D+d]),g=F(x[(long)n*2*D+D+d]);o[(long)n*D+d]=Bf(a/(1.f+__expf(-g)));}
// LayerNorm over channel axis (no bias, learnable scale) + ReLU, applied to a
// [rows, C] tensor where each row is the channel vector at one (t,f) cell.
__global__ void k_layernorm_relu(const bf* x,const bf* w,bf* o,int R,int C,float eps){
    int r=blockIdx.x;if(r>=R)return;const bf* xr=x+(long)r*C;bf* orow=o+(long)r*C;
    float m=0;for(int c=threadIdx.x;c<C;c+=blockDim.x)m+=F(xr[c]);
    for(int s=warpSize/2;s>0;s>>=1)m+=__shfl_down_sync(0xffffffff,m,s);
    __shared__ float wm[32],wv[32],mean,inv;if((threadIdx.x&31)==0)wm[threadIdx.x>>5]=m;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=wm[i];mean=t/C;}__syncthreads();
    float v=0;for(int c=threadIdx.x;c<C;c+=blockDim.x){float d=F(xr[c])-mean;v+=d*d;}
    for(int s=warpSize/2;s>0;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
    if((threadIdx.x&31)==0)wv[threadIdx.x>>5]=v;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=wv[i];inv=rsqrtf(t/C+eps);}__syncthreads();
    for(int c=threadIdx.x;c<C;c+=blockDim.x){float y=(F(xr[c])-mean)*inv*(w?F(w[c]):1.f);orow[c]=Bf(y>0.f?y:0.f);}}

// ── SSCP: Conv2d(in_ch,out_ch,k3,s2,p1) over a [in_ch, T, Freq] feature map.
// Output [out_ch, To, Fo] with To=floor((T-1)/2)+1, Fo=floor((Freq-1)/2)+1.
// PARITY TODO: verify padding=1 + stride=2 indexing vs torch Conv2d.
__global__ void k_conv2d_s2(const bf* in,const bf* W,bf* out,
                            int IC,int Tin,int Fin,int OC,int To,int Fo){
    int oc=blockIdx.z;int to=blockIdx.y*blockDim.y+threadIdx.y,fo=blockIdx.x*blockDim.x+threadIdx.x;
    if(oc>=OC||to>=To||fo>=Fo)return;
    float acc=0;
    for(int ic=0;ic<IC;ic++){
        const bf* wk=W+(((long)oc*IC+ic)*3)*3;            // [3,3]
        for(int kt=0;kt<3;kt++)for(int kf=0;kf<3;kf++){
            int ti=to*2+kt-1, fi=fo*2+kf-1;                // stride 2, pad 1
            if(ti<0||ti>=Tin||fi<0||fi>=Fin)continue;
            acc+=F(in[((long)ic*Tin+ti)*Fin+fi])*F(wk[kt*3+kf]);
        }
    }
    out[((long)oc*To+to)*Fo+fo]=Bf(acc);
}
// Reshape conv output [OC, To, Fo] → [To, Fo*OC] (channels-last per (t,f)) so
// the LayerNorm runs over the channel axis. Here we produce [To*Fo, OC].
__global__ void k_chlast(const bf* in,bf* out,int OC,int To,int Fo){
    int oc=blockIdx.z;int to=blockIdx.y*blockDim.y+threadIdx.y,fo=blockIdx.x*blockDim.x+threadIdx.x;
    if(oc>=OC||to>=To||fo>=Fo)return;
    out[(((long)to*Fo+fo)*OC)+oc]=in[((long)oc*To+to)*Fo+fo];
}
// After LayerNorm+ReLU on [To*Fo, OC], reshape back to [OC, To, Fo].
__global__ void k_chfirst(const bf* in,bf* out,int OC,int To,int Fo){
    int oc=blockIdx.z;int to=blockIdx.y*blockDim.y+threadIdx.y,fo=blockIdx.x*blockDim.x+threadIdx.x;
    if(oc>=OC||to>=To||fo>=Fo)return;
    out[((long)oc*To+to)*Fo+fo]=in[(((long)to*Fo+fo)*OC)+oc];
}
// Flatten the final SSCP map [OC, To, Fo] → [To, Fo*OC] for input_proj_linear.
// HF: permute(0,2,3,1) then reshape(B,To,Fo*OC).
__global__ void k_sscp_flatten(const bf* in,bf* out,int OC,int To,int Fo){
    int to=blockIdx.y*blockDim.y+threadIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;
    int FoOC=Fo*OC;if(to>=To||j>=FoOC)return;int fo=j/OC,oc=j%OC;
    out[(long)to*FoOC+j]=in[((long)oc*To+to)*Fo+fo];
}

// Depthwise CAUSAL conv1d over [T, C] with kernel [C,1,K], left-pad K-1.
// out[t,c] = sum_{j=0}^{K-1} in[t-(K-1)+j, c] * w[c, 0, j].
__global__ void k_depthwise_causal(const bf* in,const bf* W,bf* out,int T,int C,int K){
    int t=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;if(t>=T||c>=C)return;
    float acc=0;for(int j=0;j<K;j++){int ti=t-(K-1)+j;if(ti<0)continue;acc+=F(in[(long)ti*C+c])*F(W[(long)c*K+j]);}
    out[(long)t*C+c]=Bf(acc);
}

dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}

}  // namespace

// ── Chunked-local self-attention (one layer) ─────────────────────────────────
// Reference: Gemma4AudioAttention. q_scale=(hd^-0.5)/ln2, k_scale=ln(1+e)/ln2,
// per_dim_scale via softplus, logit cap (tanh), exact relative-position bias.
//
// The HF blocked-5D path (chunk 12 / past 12 / future 0) plus `_rel_shift` is,
// for the actual mask, identical to a plain causal sliding window: query t
// attends keys j with 0 <= t-j < max_past (=12, no future). And the rel_shift
// gather collapses to: matrix_bd[t,j] uses relative-position embedding index
// p = max_past - (t-j), i.e. the sinusoidal position whose position_id == t-j.
// Verified flat-vs-blocked to <1e-6 abs (scripts/ref_full_attn.py).
namespace {
__global__ void k_qkv_scale(bf* q,bf* k,const bf* pds,int N,int H,int hd,
                            float q_scale,float k_scale){
    int n=blockIdx.y*blockDim.y+threadIdx.y,e=blockIdx.x*blockDim.x+threadIdx.x;
    int HD=H*hd;if(n>=N||e>=HD)return;int d=e%hd;
    float sp=logf(1.f+expf(F(pds[d])));                 // softplus(per_dim_scale)
    q[(long)n*HD+e]=Bf(F(q[(long)n*HD+e])*q_scale*sp);
    k[(long)n*HD+e]=Bf(F(k[(long)n*HD+e])*k_scale);
}
// Build the sinusoidal relative-position encoding `pe[P, hidden]`, P=max_past+1.
// position_ids = arange(max_past, -1, -1) = [max_past, .., 1, 0]; row r holds
// position_id = max_past - r. scaled_time[r, m] = (max_past-r) * inv[m];
// pe[r] = concat(sin(scaled_time[r]), cos(scaled_time[r])).
__global__ void k_rel_pos_enc(bf* pe,int P,int hidden){
    int r=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=P||d>=hidden)return;
    int num_ts=hidden/2;
    float log_inc=logf(10000.f/1.f)/fmaxf((float)(num_ts-1),1.f);
    int m=d<num_ts?d:(d-num_ts);
    float inv=expf((float)m*-log_inc);
    float pos=(float)((P-1)-r);                           // position_id = max_past - r
    float t=pos*inv;
    pe[(long)r*hidden+d]=Bf(d<num_ts?sinf(t):cosf(t));
}
// Exact O(N^2) causal-sliding-window attention with relative-position bias and
// logit soft-cap. q,k are already pre-scaled (q_scale·softplus(pds), k_scale).
// `relk` is relative_k_proj(pe) reshaped [P, H, hd]; pe row r encodes
// position_id = (P-1)-r, so the embedding for relative distance (t-j) lives at
// row (P-1)-(t-j). Query t attends keys j in [t-(P-1), t].
__global__ void k_local_attn(const bf* q,const bf* k,const bf* v,
                             const bf* relk,bf* out,
                             int N,int H,int hd,int P,float cap){
    int head=blockIdx.y,i=blockIdx.x*blockDim.x+threadIdx.x;if(head>=H||i>=N)return;
    float acc[256];                                     // hd ≤ 256 (gemma4: 128)
    for(int d=0;d<hd;d++)acc[d]=0.f;
    // mask: query t attends keys j with 0 <= (t-j) < max_past (= P-1, no future).
    // So distance ∈ [0, P-2]; lo = i-(P-2). (Distance P-1 is excluded.)
    int lo=i-(P-2); if(lo<0)lo=0;
    const bf* qr=q+((long)i*H+head)*hd;
    float mx=-1e30f;
    for(int j=lo;j<=i;j++){
        const bf* kr=k+((long)j*H+head)*hd;
        const bf* rr=relk+((long)((P-1)-(i-j))*H+head)*hd; // pe row (P-1)-(t-j) ↔ position_id=t-j
        float s=0;for(int d=0;d<hd;d++)s+=F(qr[d])*(F(kr[d])+F(rr[d]));
        s=cap*tanhf(s/cap);                              // logit soft-cap
        mx=fmaxf(mx,s);
    }
    float denom=0;
    for(int j=lo;j<=i;j++){
        const bf* kr=k+((long)j*H+head)*hd;
        const bf* rr=relk+((long)((P-1)-(i-j))*H+head)*hd;
        float s=0;for(int d=0;d<hd;d++)s+=F(qr[d])*(F(kr[d])+F(rr[d]));
        s=cap*tanhf(s/cap);float w=__expf(s-mx);denom+=w;
        const bf* vr=v+((long)j*H+head)*hd;
        for(int d=0;d<hd;d++)acc[d]+=w*F(vr[d]);
    }
    float inv=denom>0.f?1.f/denom:0.f;
    for(int d=0;d<hd;d++)out[((long)i*H+head)*hd+d]=Bf(acc[d]*inv);
}
}  // namespace

void run_gemma4_audio(const AudioRawWeights& w,
                      const float* features,int n_frames,int n_mel,int out_len,
                      bf* out_proj,cudaStream_t S){
    const int Hd=w.hidden, NH=w.heads, hd=Hd/NH, IM=4*Hd, TXT=w.text_hidden, OPD=w.out_proj_dims;
    const float EPS=w.eps, CAP=w.logit_cap, RW=w.residual_weight;
    if(Hd!=1024||NH!=8) throw std::runtime_error("gemma4_audio: unexpected dims (expected hidden=1024, heads=8)");
    const float q_scale=(powf((float)hd,-0.5f))/logf(2.f);
    const float k_scale=logf(1.f+(float)M_E)/logf(2.f);
    const int past=w.context_left-1; (void)w.context_right;  // future horizon == 0 (mask is plain causal sliding window)

    DeviceScratch scratch;
    auto MAL=[&](long n){return scratch.alloc<bf>(n);};
    auto clin=[&](const bf* x,bf* out,bf* xc,const AudioClipRaw& c,int N,int Kin,int Out){
        k_clamp<<<((long)N*Kin+255)/256,256,0,S>>>(x,xc,c.imin,c.imax,(long)N*Kin);
        k_matmul<<<G2(Out,N),B2,0,S>>>(xc,c.w,out,N,Kin,Out);
        k_clamp<<<((long)N*Out+255)/256,256,0,S>>>(out,out,c.omin,c.omax,(long)N*Out);};

    // ── 1) SSCP subsampling conv stack ──────────────────────────────────────
    // input_features [n_frames, n_mel] → unsqueeze channel → [1, n_frames, n_mel].
    // PARITY TODO: confirm the (time, freq) axis mapping vs torch's [B,1,T,F].
    const int T0=n_frames, F0=n_mel;
    auto cdim=[](int n){return (n-1)/2+1;};
    const int T1=cdim(T0),F1=cdim(F0), C0=w.sscp_ch0;
    const int T2=cdim(T1),F2=cdim(F1), C1=w.sscp_ch1;
    if(T2!=out_len) throw std::runtime_error("gemma4_audio: out_len != subsampled frames");

    bf* feat_bf=MAL((long)T0*F0);
    {   // upload f32 features (host) → device → bf16 [1, T0, F0]
        float* f32d=scratch.alloc<float>((long)T0*F0);
        ACK(cudaMemcpyAsync(f32d,features,(long)T0*F0*4,cudaMemcpyHostToDevice,S));
        k_f32_to_bf16<<<((long)T0*F0+255)/256,256,0,S>>>(f32d,feat_bf,(long)T0*F0);
        ACK(cudaStreamSynchronize(S));
    }
    // layer0: conv [1,T0,F0]→[C0,T1,F1], LN-over-ch + ReLU.
    bf* c0=MAL((long)C0*T1*F1);
    { dim3 g((F1+15)/16,(T1+15)/16,C0); k_conv2d_s2<<<g,B2,0,S>>>(feat_bf,w.sscp0_conv,c0,1,T0,F0,C0,T1,F1); }
    bf* c0cl=MAL((long)T1*F1*C0);
    { dim3 g((F1+15)/16,(T1+15)/16,C0); k_chlast<<<g,B2,0,S>>>(c0,c0cl,C0,T1,F1); }
    k_layernorm_relu<<<T1*F1,128,0,S>>>(c0cl,w.sscp0_norm,c0cl,T1*F1,C0,EPS);
    { dim3 g((F1+15)/16,(T1+15)/16,C0); k_chfirst<<<g,B2,0,S>>>(c0cl,c0,C0,T1,F1); }
    // layer1: conv [C0,T1,F1]→[C1,T2,F2], LN-over-ch + ReLU.
    bf* c1=MAL((long)C1*T2*F2);
    { dim3 g((F2+15)/16,(T2+15)/16,C1); k_conv2d_s2<<<g,B2,0,S>>>(c0,w.sscp1_conv,c1,C0,T1,F1,C1,T2,F2); }
    bf* c1cl=MAL((long)T2*F2*C1);
    { dim3 g((F2+15)/16,(T2+15)/16,C1); k_chlast<<<g,B2,0,S>>>(c1,c1cl,C1,T2,F2); }
    k_layernorm_relu<<<T2*F2,128,0,S>>>(c1cl,w.sscp1_norm,c1cl,T2*F2,C1,EPS);
    { dim3 g((F2+15)/16,(T2+15)/16,C1); k_chfirst<<<g,B2,0,S>>>(c1cl,c1,C1,T2,F2); }
    // flatten [C1,T2,F2] → [T2, F2*C1] and input_proj → [T2, hidden].
    const int N=T2, FLAT=F2*C1;
    bf* flat=MAL((long)N*FLAT);
    { dim3 g((FLAT+15)/16,(N+15)/16); k_sscp_flatten<<<g,B2,0,S>>>(c1,flat,C1,T2,F2); }
    bf* h=MAL((long)N*Hd);
    k_matmul<<<G2(Hd,N),B2,0,S>>>(flat,w.sscp_input_proj,h,N,FLAT,Hd);

    // ckpt: sscp_out (input_proj output, before any conformer layer).
    auto CKPT=[&](const char* nm,const bf* d,long n){
        if(!g_audio_ckpt)return; ACK(cudaStreamSynchronize(S));
        g_audio_ckpt(nm,d,n,g_audio_ckpt_user); };
    CKPT("sscp_out",h,(long)N*Hd);

    // ── 2) Conformer layers ─────────────────────────────────────────────────
    bf *hn=MAL((long)N*Hd),*xc=MAL((long)N*IM),*ffmid=MAL((long)N*IM),*ffout=MAL((long)N*Hd),
       *q=MAL((long)N*Hd),*k=MAL((long)N*Hd),*v=MAL((long)N*Hd),*attn=MAL((long)N*Hd),
       *glu=MAL((long)N*Hd),*conv=MAL((long)N*Hd),*tmp=MAL((long)N*Hd),*start=MAL((long)N*2*Hd);

    // Sinusoidal relative-position encoding pe[P, hidden], P = max_past+1.
    // Shared across layers; relative_k_proj differs per layer so relk is per-layer.
    const int P=past+1;                                   // 13 (= context_left)
    bf* pe=MAL((long)P*Hd);
    { dim3 g((Hd+15)/16,(P+15)/16); k_rel_pos_enc<<<g,B2,0,S>>>(pe,P,Hd); }
    bf* relk=MAL((long)P*Hd);                              // relative_k_proj(pe) → [P, H*hd]

    auto ffn=[&](const AudioFfnRaw& ff){
        // residual = x; x=clamp; pre_ln; fc1; silu; fc2; clamp; post_ln; ×RW; +res
        k_rms<<<N,256,0,S>>>(h,ff.pre_ln,hn,N,Hd,EPS);
        clin(hn,ffmid,xc,ff.fc1,N,Hd,IM);
        k_silu<<<((long)N*IM+255)/256,256,0,S>>>(ffmid,ffmid,(long)N*IM);
        clin(ffmid,ffout,xc,ff.fc2,N,IM,Hd);
        k_rms<<<N,256,0,S>>>(ffout,ff.post_ln,ffout,N,Hd,EPS);
        k_axpy<<<((long)N*Hd+255)/256,256,0,S>>>(h,ffout,RW,(long)N*Hd);
    };

    int li=0;
    for(const auto& L:w.layers){
        // feed_forward1 (macaron half-step)
        ffn(L.ff1);
        // self-attention
        k_rms<<<N,256,0,S>>>(h,L.norm_pre_attn,hn,N,Hd,EPS);
        clin(hn,q,xc,L.q,N,Hd,Hd); clin(hn,k,xc,L.k,N,Hd,Hd); clin(hn,v,xc,L.v,N,Hd,Hd);
        k_qkv_scale<<<G2(Hd,N),B2,0,S>>>(q,k,L.per_dim_scale,N,NH,hd,q_scale,k_scale);
        // relative_k_proj(pe) → relk [P, H*hd]; NOT a clipped linear (plain matmul).
        k_matmul<<<G2(Hd,P),B2,0,S>>>(pe,L.relative_k,relk,P,Hd,Hd);
        { dim3 g((N+127)/128,NH); k_local_attn<<<g,128,0,S>>>(q,k,v,relk,attn,N,NH,hd,P,CAP); }
        clin(attn,tmp,xc,L.post,N,Hd,Hd);
        k_rms<<<N,256,0,S>>>(tmp,L.norm_post_attn,tmp,N,Hd,EPS);
        k_add<<<((long)N*Hd+255)/256,256,0,S>>>(h,tmp,(long)N*Hd);
        // light depthwise-conv module
        k_rms<<<N,256,0,S>>>(h,L.lconv_pre_ln,hn,N,Hd,EPS);
        clin(hn,start,xc,L.lconv_start,N,Hd,2*Hd);            // [N, 2*hidden]
        k_glu<<<G2(Hd,N),B2,0,S>>>(start,glu,N,Hd);            // GLU → [N, hidden]
        k_depthwise_causal<<<G2(Hd,N),B2,0,S>>>(glu,L.depthwise_conv,conv,N,Hd,w.conv_kernel);
        // clamp(±finfo_max) is a no-op in bf16 range → skip; conv_norm + silu
        k_rms<<<N,256,0,S>>>(conv,L.lconv_conv_norm,conv,N,Hd,EPS);
        k_silu<<<((long)N*Hd+255)/256,256,0,S>>>(conv,conv,(long)N*Hd);
        clin(conv,tmp,xc,L.lconv_end,N,Hd,Hd);
        k_add<<<((long)N*Hd+255)/256,256,0,S>>>(h,tmp,(long)N*Hd);
        // feed_forward2 (macaron half-step)
        ffn(L.ff2);
        // norm_out
        k_rms<<<N,256,0,S>>>(h,L.norm_out,h,N,Hd,EPS);
        // ckpt: layer{li} output (matches HF Gemma4AudioLayer hidden_states dump).
        { char nm[16]; snprintf(nm,sizeof nm,"layer%d",li); CKPT(nm,h,(long)N*Hd); }
        ++li;
    }

    // ── 3) output_proj (1024→1536 +bias) ────────────────────────────────────
    bf* enc=MAL((long)N*OPD);
    k_matmul_bias<<<G2(OPD,N),B2,0,S>>>(h,w.output_proj_w,w.output_proj_b,enc,N,Hd,OPD);
    CKPT("encoder_out",enc,(long)N*OPD);

    // ── 4) embedder: parameterless RMSNorm(1536) → projection (1536→2560) ────
    bf* en=MAL((long)N*OPD);
    k_rms<<<N,256,0,S>>>(enc,nullptr,en,N,OPD,EPS);
    k_matmul<<<G2(TXT,N),B2,0,S>>>(en,w.embed_proj,out_proj,N,OPD,TXT);
    CKPT("projected",out_proj,(long)N*TXT);

    ACK(cudaStreamSynchronize(S));
}

void scatter_gemma4_audio(const Gemma4AudioInputs& ain, bf* hidden,
                          int /*n_rows*/, int text_hidden, cudaStream_t S){
    if(ain.weights==nullptr || ain.num_clips<=0) return;
    const AudioRawWeights& w=*ain.weights;
    const int n_mel=ain.n_mel;
    for(int ci=0; ci<ain.num_clips; ++ci){
        const long blo=ain.feature_byte_indptr_h[ci], bhi=ain.feature_byte_indptr_h[ci+1];
        const int n_floats=(int)((bhi-blo)/4);
        const int n_frames=n_floats/n_mel;
        if(n_frames<=0) continue;
        const int out_len=gemma4_audio_subsampled_len(n_frames);
        const float* feat_h=ain.features_h + blo/4;
        const std::uint32_t anchor=ain.anchor_rows_h[ci];

        // encode → projected [out_len, text_hidden] → overwrite the anchor rows.
        DeviceScratch scratch;
        bf* proj_d=scratch.alloc<bf>((long)out_len*text_hidden);
        run_gemma4_audio(w, feat_h, n_frames, n_mel, out_len, proj_d, S);
        ACK(cudaMemcpyAsync(hidden + (long)anchor*text_hidden, proj_d,
                            (long)out_len*text_hidden*sizeof(bf),
                            cudaMemcpyDeviceToDevice, S));
        ACK(cudaStreamSynchronize(S));
    }
}

void encode_gemma4_audio(const Gemma4AudioInputs& ain,
                         std::uint16_t* output_rows_h,
                         std::size_t output_bytes,
                         std::uint32_t* output_row_indptr_h,
                         cudaStream_t S) {
    if (ain.weights == nullptr || ain.num_clips <= 0 ||
        output_rows_h == nullptr || output_row_indptr_h == nullptr) {
        throw std::runtime_error("gemma4_audio: invalid standalone encode inputs");
    }
    const AudioRawWeights& w = *ain.weights;
    const int n_mel = ain.n_mel;
    const std::size_t row_bytes =
        static_cast<std::size_t>(w.text_hidden) * sizeof(bf);
    std::size_t output_rows = 0;
    output_row_indptr_h[0] = 0;
    for (int clip = 0; clip < ain.num_clips; ++clip) {
        const long begin = ain.feature_byte_indptr_h[clip];
        const long end = ain.feature_byte_indptr_h[clip + 1];
        const int floats = static_cast<int>((end - begin) / sizeof(float));
        const int frames = floats / n_mel;
        if (frames <= 0 || floats % n_mel != 0) {
            throw std::runtime_error("gemma4_audio: invalid feature shape");
        }
        const int rows = gemma4_audio_subsampled_len(frames);
        if ((output_rows + static_cast<std::size_t>(rows)) * row_bytes >
            output_bytes) {
            throw std::runtime_error(
                "gemma4_audio: encode output buffer too small");
        }
        DeviceScratch scratch;
        bf* projected = scratch.alloc<bf>(
            static_cast<long>(rows) * w.text_hidden);
        run_gemma4_audio(
            w, ain.features_h + begin / sizeof(float), frames, n_mel,
            rows, projected, S);
        ACK(cudaMemcpyAsync(
            output_rows_h + output_rows * w.text_hidden, projected,
            static_cast<long>(rows) * w.text_hidden * sizeof(bf),
            cudaMemcpyDeviceToHost, S));
        ACK(cudaStreamSynchronize(S));
        output_rows += static_cast<std::size_t>(rows);
        output_row_indptr_h[clip + 1] =
            static_cast<std::uint32_t>(output_rows);
    }
}

}  // namespace pie_cuda_driver::model
