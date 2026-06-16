// CSM depth-decoder forward + RVQ frame sampler (bf16 store / fp32 compute).
// See csm_depth_decoder_forward.hpp. Naive kernels (the depth seq is <= 33 and
// hidden 1024 / 4 layers — correctness + parity first, like the gemma4 encoders
// and the mimi decoder). Math transcribed from transformers 5.9
// `CsmDepthDecoderForCausalLM` + the per-frame inner loop of
// `CsmGenerationMixin._sample`; checked against scripts/csm_depth_decoder_parity_ref.py.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/csm_depth_decoder_forward.hpp"

namespace pie_cuda_driver::model {
namespace {

using bf = __nv_bfloat16;
#define CK(x) do{cudaError_t e=(x);if(e)throw std::runtime_error(std::string("csm_depth: ")+cudaGetErrorString(e));}while(0)
__device__ __forceinline__ float F(bf x){return __bfloat162float(x);}
__device__ __forceinline__ bf   Bf(float x){return __float2bfloat16(x);}

// y[n,o] = sum_k x[n,k]*W[o,k]   (W is [O,K] row-major, PyTorch Linear layout).
__global__ void k_matmul(const bf* x,const bf* W,bf* y,int N,int K,int O){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=N||o>=O)return;
    const bf* xr=x+(long)n*K;const bf* wr=W+(long)o*K;
    float a=0;for(int k=0;k<K;k++)a+=F(xr[k])*F(wr[k]);
    y[(long)n*O+o]=Bf(a);
}
// Per-row RMSNorm over D (fp32 accumulate, matches CsmRMSNorm).
__global__ void k_rms(const bf* x,const bf* w,bf* o,int R,int D,float eps){
    int r=blockIdx.x;if(r>=R)return;const bf* xr=x+(long)r*D;bf* orow=o+(long)r*D;
    float loc=0;for(int d=threadIdx.x;d<D;d+=blockDim.x){float v=F(xr[d]);loc+=v*v;}
    for(int s=warpSize/2;s>0;s>>=1)loc+=__shfl_down_sync(0xffffffff,loc,s);
    __shared__ float warp[32],ss;if((threadIdx.x&31)==0)warp[threadIdx.x>>5]=loc;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=warp[i];ss=rsqrtf(t/D+eps);}__syncthreads();
    float inv=ss;for(int d=threadIdx.x;d<D;d+=blockDim.x)orow[d]=Bf(F(xr[d])*inv*(w?F(w[d]):1.f));
}
// SwiGLU MLP fused: out[n,i] = silu(gate[n,i]) * up[n,i].
__global__ void k_swiglu(const bf* gate,const bf* up,bf* o,long t){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i>=t)return;
    float g=F(gate[i]);o[i]=Bf((g/(1.f+__expf(-g)))*F(up[i]));
}
__global__ void k_add(bf* a,const bf* b,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t)a[i]=Bf(F(a[i])+F(b[i]));}

// llama3 YaRN inv-freq (mirrors rope.cu::yarn_freq).
__device__ __forceinline__ float yarn_freq(float base_freq,float factor,
        float low_freq_factor,float high_freq_factor,float orig_max_pos){
    const float TWO_PI=6.283185307179586f;
    const float wavelen=TWO_PI/base_freq;
    const float low_wave=orig_max_pos/low_freq_factor;
    const float high_wave=orig_max_pos/high_freq_factor;
    if(wavelen<high_wave)return base_freq;
    if(wavelen>low_wave)return base_freq/factor;
    const float smooth=(orig_max_pos/wavelen-low_freq_factor)/(high_freq_factor-low_freq_factor);
    return (1.f-smooth)*(base_freq/factor)+smooth*base_freq;
}
// Apply RoPE in place to one row q/k [H,hd] at absolute position `pos`. The
// rotate-half convention (first half / second half) matches modeling_csm.py.
__global__ void k_rope_row(bf* x,int H,int hd,int pos,
        float theta,float factor,float lo,float hi,float orig){
    int h=blockIdx.y*blockDim.y+threadIdx.y, d=blockIdx.x*blockDim.x+threadIdx.x;
    int half=hd/2;if(h>=H||d>=half)return;
    float base=powf(theta,(2.f*d)/(float)hd);     // theta^(2d/hd)
    float invf=1.f/base;
    float freq=yarn_freq(invf,factor,lo,hi,orig);  // scaled inverse frequency
    float ang=pos*freq;float c=cosf(ang),s=sinf(ang);
    bf* row=x+(long)h*hd;
    float a=F(row[d]),b=F(row[d+half]);
    row[d]      =Bf(a*c-b*s);
    row[d+half] =Bf(b*c+a*s);
}
// Single-query attention against a KV cache of length L (the new query is at
// index L-1, full causal => attends all 0..L-1). GQA: query head h maps to kv
// head h / (H/KV). out [H,hd].
__global__ void k_attn_decode(const bf* q,const bf* kcache,const bf* vcache,
        bf* out,int H,int KV,int hd,int L,float scale){
    int h=blockIdx.x;if(h>=H)return;
    int kvh=h/(H/KV);
    extern __shared__ float sh[];           // [L] scores
    const bf* qh=q+(long)h*hd;
    float mx=-1e30f;
    for(int j=threadIdx.x;j<L;j+=blockDim.x){
        const bf* kj=kcache+((long)j*KV+kvh)*hd;
        float s=0;for(int d=0;d<hd;d++)s+=F(qh[d])*F(kj[d]);
        s*=scale;sh[j]=s;mx=fmaxf(mx,s);
    }
    // block-wide max
    __shared__ float red[32];
    for(int o=warpSize/2;o>0;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    if((threadIdx.x&31)==0)red[threadIdx.x>>5]=mx;__syncthreads();
    __shared__ float gmax,gden;
    if(threadIdx.x==0){float m=-1e30f;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)m=fmaxf(m,red[i]);gmax=m;}__syncthreads();
    float den=0;for(int j=threadIdx.x;j<L;j+=blockDim.x){float e=__expf(sh[j]-gmax);sh[j]=e;den+=e;}
    for(int o=warpSize/2;o>0;o>>=1)den+=__shfl_down_sync(0xffffffff,den,o);
    if((threadIdx.x&31)==0)red[threadIdx.x>>5]=den;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=red[i];gden=t;}__syncthreads();
    float inv=gden>0.f?1.f/gden:0.f;
    // weighted sum of V over the cache
    for(int d=threadIdx.x;d<hd;d+=blockDim.x){
        float acc=0;for(int j=0;j<L;j++){const bf* vj=vcache+((long)j*KV+kvh)*hd;acc+=sh[j]*F(vj[d]);}
        out[(long)h*hd+d]=Bf(acc*inv);
    }
}
// argmax over a [V] logits row -> index.
__global__ void k_argmax(const bf* logits,int V,int* out){
    int t=threadIdx.x;float bv=-1e30f;int bi=0;
    for(int v=t;v<V;v+=blockDim.x){float x=F(logits[v]);if(x>bv){bv=x;bi=v;}}
    __shared__ float sv[256];__shared__ int si[256];
    sv[t]=bv;si[t]=bi;__syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){if(t<s){if(sv[t+s]>sv[t]||(sv[t+s]==sv[t]&&si[t+s]<si[t])){sv[t]=sv[t+s];si[t]=si[t+s];}}__syncthreads();}
    if(t==0)*out=si[0];
}
__global__ void k_logits_to_f32(const bf* in,float* out,int V){
    int v=blockIdx.x*blockDim.x+threadIdx.x;if(v<V)out[v]=F(in[v]);
}
// codebooks head: logits[v] = sum_h hidden[h] * W[h*V + v]  (W slab is [H,V],
// the [hidden,vocab] codebooks_head.weight[i-1] — modeling_csm.py uses
// linear(hidden, weight[i].T), i.e. weight stored [hidden,vocab]).
__global__ void k_head(const bf* hidden,const bf* W,bf* logits,int Hh,int V){
    int v=blockIdx.x*blockDim.x+threadIdx.x;if(v>=V)return;
    float a=0;for(int h=0;h<Hh;h++)a+=F(hidden[h])*F(W[(long)h*V+v]);
    logits[v]=Bf(a);
}

dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}

}  // namespace

// ── One depth-decoder step (token at depth position `pos`, KV length grows) ───
// hidden_in is the already-projected [1, hidden] input embed for this position.
// Returns the post-norm hidden [1, hidden] in `hidden_out` (caller applies head).
namespace {
struct DepthScratch {
    bf *normed, *q, *k, *v, *attn, *attn_o, *gate, *up, *mlp, *resid;
    // KV caches per layer: [maxL, KV, hd]
    std::vector<bf*> kcache, vcache;
    int maxL=0;
};

void depth_layer(const CsmDepthRawWeights& w,const CsmDepthLayerRaw& L,
                 DepthScratch& s,int li,int pos,int len,cudaStream_t S){
    const int H=w.hidden, NH=w.num_heads, KV=w.num_kv_heads, hd=w.head_dim;
    const int QD=NH*hd, KD=KV*hd;
    const float scale=1.f/sqrtf((float)hd);
    // input_layernorm
    k_rms<<<1,256,0,S>>>(s.resid,L.in_ln_w,s.normed,1,H,w.norm_eps);
    // qkv projections (single row)
    k_matmul<<<G2(QD,1),B2,0,S>>>(s.normed,L.q,s.q,1,H,QD);
    k_matmul<<<G2(KD,1),B2,0,S>>>(s.normed,L.k,s.k,1,H,KD);
    k_matmul<<<G2(KD,1),B2,0,S>>>(s.normed,L.v,s.v,1,H,KD);
    // RoPE on q [NH,hd] and k [KV,hd] at absolute position `pos`
    { dim3 g((hd/2+15)/16,(NH+15)/16); k_rope_row<<<g,B2,0,S>>>(s.q,NH,hd,pos,
        w.rope_theta,w.rope_factor,w.rope_low_freq_factor,w.rope_high_freq_factor,(float)w.rope_original_max_position); }
    { dim3 g((hd/2+15)/16,(KV+15)/16); k_rope_row<<<g,B2,0,S>>>(s.k,KV,hd,pos,
        w.rope_theta,w.rope_factor,w.rope_low_freq_factor,w.rope_high_freq_factor,(float)w.rope_original_max_position); }
    // append k,v into the cache at slot (len-1)
    CK(cudaMemcpyAsync(s.kcache[li]+(long)(len-1)*KD,s.k,(long)KD*sizeof(bf),cudaMemcpyDeviceToDevice,S));
    CK(cudaMemcpyAsync(s.vcache[li]+(long)(len-1)*KD,s.v,(long)KD*sizeof(bf),cudaMemcpyDeviceToDevice,S));
    // attention (single query against len keys)
    k_attn_decode<<<NH,128,(size_t)len*sizeof(float),S>>>(s.q,s.kcache[li],s.vcache[li],s.attn,NH,KV,hd,len,scale);
    // o_proj  [H <- QD]
    k_matmul<<<G2(H,1),B2,0,S>>>(s.attn,L.o,s.attn_o,1,QD,H);
    // residual add
    k_add<<<(H+255)/256,256,0,S>>>(s.resid,s.attn_o,H);
    // post_attention_layernorm
    k_rms<<<1,256,0,S>>>(s.resid,L.post_ln_w,s.normed,1,H,w.norm_eps);
    // MLP
    k_matmul<<<G2(w.intermediate,1),B2,0,S>>>(s.normed,L.gate,s.gate,1,H,w.intermediate);
    k_matmul<<<G2(w.intermediate,1),B2,0,S>>>(s.normed,L.up,s.up,1,H,w.intermediate);
    k_swiglu<<<(w.intermediate+255)/256,256,0,S>>>(s.gate,s.up,s.gate,w.intermediate);
    k_matmul<<<G2(H,1),B2,0,S>>>(s.gate,L.down,s.mlp,1,w.intermediate,H);
    k_add<<<(H+255)/256,256,0,S>>>(s.resid,s.mlp,H);
}
}  // namespace

void run_csm_depth_decoder_frame_dbg(const CsmDepthRawWeights& w,
                                     const bf* bb_hidden,std::int32_t cb0,
                                     std::int32_t* out_codes,float* out_logits,
                                     cudaStream_t S){
    const int H=w.hidden, BH=w.backbone_hidden, NCB=w.num_codebooks, V=w.vocab_size;
    const int QD=w.num_heads*w.head_dim, KD=w.num_kv_heads*w.head_dim;
    const int maxL=NCB;                              // depth positions 0..31 (<=33)
    auto MAL=[&](long n){bf* d;CK(cudaMalloc(&d,n*sizeof(bf)));return d;};

    DepthScratch s; s.maxL=maxL;
    bf* embed_raw=MAL(BH);          // pre-projection embed [backbone_hidden]
    s.normed=MAL(H); s.q=MAL(QD); s.k=MAL(KD); s.v=MAL(KD);
    s.attn=MAL(QD); s.attn_o=MAL(H); s.gate=MAL(w.intermediate); s.up=MAL(w.intermediate);
    s.mlp=MAL(H); s.resid=MAL(H);
    s.kcache.resize(w.num_layers); s.vcache.resize(w.num_layers);
    for(int l=0;l<w.num_layers;l++){ s.kcache[l]=MAL((long)maxL*KD); s.vcache[l]=MAL((long)maxL*KD); }
    bf* logits=MAL(V);
    int* d_arg;CK(cudaMalloc(&d_arg,sizeof(int)));

    // Helper: run the stack for the input row already in `embed_raw`
    // (pre-projection [BH]) -> project to [H] in resid, run layers, post-norm.
    auto step=[&](int pos,int len){
        // project embed_raw [BH] -> resid [H] (inputs_embeds_projector, no bias)
        k_matmul<<<G2(H,1),B2,0,S>>>(embed_raw,w.inputs_embeds_projector,s.resid,1,BH,H);
        for(int l=0;l<w.num_layers;l++) depth_layer(w,w.layers[l],s,l,pos,len,S);
        // final norm -> s.normed holds the post-norm hidden
        k_rms<<<1,256,0,S>>>(s.resid,w.norm_w,s.normed,1,H,w.norm_eps);
    };

    // Depth position 0: the placeholder slot, seeded by backbone_last_hidden_state.
    // (input_ids[0]=0 but its embed is overwritten by bb_hidden BEFORE the
    // projector — modeling_csm.py CsmDepthDecoderModel.forward.)
    CK(cudaMemcpyAsync(embed_raw,bb_hidden,(long)BH*sizeof(bf),cudaMemcpyDeviceToDevice,S));
    step(0,1);                       // KV length 1, position 0. (no head emitted)

    // Depth position 1: token = cb0, codebook_idx = clamp(1-1,0)=0 -> offset 0.
    // embed_tokens[cb0]. The head AFTER this step produces codebook 1.
    int prev_id=cb0; int prev_cb_off=0;   // offset for the token fed at next pos
    // Loop producing codebooks 1..31. At depth position p (1..31) we feed the
    // previously sampled code (offset = (p-1)*vocab) and the head weight[p-1].
    for(int p=1;p<NCB;p++){
        // embed the token fed at this position: embed_tokens[prev_id + prev_cb_off]
        long row=(long)(prev_id+prev_cb_off);
        CK(cudaMemcpyAsync(embed_raw,w.embed_tokens+row*BH,(long)BH*sizeof(bf),cudaMemcpyDeviceToDevice,S));
        int len=p+1;                 // KV positions 0..p
        step(p,len);
        // head: codebooks_head.weight[p-1] is [H, V] row-major; logits[v]=sum_h hidden[h]*W[h*V+v]
        const bf* head=w.codebooks_head+(long)(p-1)*H*V;
        k_head<<<(V+255)/256,256,0,S>>>(s.normed,head,logits,H,V);
        k_argmax<<<1,256,0,S>>>(logits,V,d_arg);
        if(out_logits){
            float* dlog;CK(cudaMalloc(&dlog,(long)V*4));
            k_logits_to_f32<<<(V+255)/256,256,0,S>>>(logits,dlog,V);
            CK(cudaMemcpyAsync(out_logits+(long)(p-1)*V,dlog,(long)V*4,cudaMemcpyDeviceToHost,S));
            CK(cudaStreamSynchronize(S));
            cudaFree(dlog);
        }
        int hid;CK(cudaMemcpyAsync(&hid,d_arg,sizeof(int),cudaMemcpyDeviceToHost,S));CK(cudaStreamSynchronize(S));
        out_codes[p-1]=hid;
        prev_id=hid; prev_cb_off=p*V;    // next position p+1 uses codebook_idx=p
    }

    cudaFree(embed_raw);cudaFree(s.normed);cudaFree(s.q);cudaFree(s.k);cudaFree(s.v);
    cudaFree(s.attn);cudaFree(s.attn_o);cudaFree(s.gate);cudaFree(s.up);cudaFree(s.mlp);cudaFree(s.resid);
    for(int l=0;l<w.num_layers;l++){cudaFree(s.kcache[l]);cudaFree(s.vcache[l]);}
    cudaFree(logits);cudaFree(d_arg);
}

void run_csm_depth_decoder_frame(const CsmDepthRawWeights& w,const bf* bb_hidden,
                                 std::int32_t cb0,std::int32_t* out_codes,cudaStream_t S){
    run_csm_depth_decoder_frame_dbg(w,bb_hidden,cb0,out_codes,nullptr,S);
}

}  // namespace pie_cuda_driver::model
