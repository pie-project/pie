// CSM backbone forward + frame-stepped generation primitive (bf16 store / fp32
// compute). See csm_backbone_forward.hpp. Naive kernels in the same style as
// csm_depth_decoder_forward.cu — the sequences are short (text prompt + audio
// frames) so correctness/parity comes first. Math transcribed from transformers
// 5.9 `CsmForConditionalGeneration` (backbone = stock Llama-3.2-1B) + the outer
// loop of `CsmGenerationMixin._sample`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/csm/csm_backbone_forward.hpp"

#include "gemm/gemm.hpp"
#include "mlp/swiglu.hpp"
#include "norm/rmsnorm.hpp"
#include "sample/argmax.hpp"
#include "model/csm/csm_naive_kernels.cuh"

namespace pie_cuda_driver::model {
namespace {

#define CK(x) do{cudaError_t e=(x);if(e)throw std::runtime_error(std::string("csm_bb: ")+cudaGetErrorString(e));}while(0)

// y[n,o] = sum_k x[n,k]*W[o,k]   (W is [O,K] row-major, PyTorch Linear layout).
// Per-row RMSNorm over D (fp32 accumulate, matches LlamaRMSNorm).
// llama3 YaRN inv-freq (mirrors rope.cu::yarn_freq / the depth decoder).
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
// Apply RoPE in place to a [R, H, hd] block where row r has absolute position
// pos0+r (rotate-half convention, matches modeling_csm / llama).
__global__ void k_rope_yarn_1d(bf* x,int R,int H,int hd,int pos0,
        float theta,float factor,float lo,float hi,float orig){
    int r=blockIdx.z; int h=blockIdx.y*blockDim.y+threadIdx.y, d=blockIdx.x*blockDim.x+threadIdx.x;
    int half=hd/2;if(r>=R||h>=H||d>=half)return;
    int pos=pos0+r;
    float base=powf(theta,(2.f*d)/(float)hd);
    float invf=1.f/base;
    float freq=yarn_freq(invf,factor,lo,hi,orig);
    float ang=pos*freq;float c=cosf(ang),s=sinf(ang);
    bf* row=x+((long)r*H+h)*hd;
    float a=F(row[d]),b=F(row[d+half]);
    row[d]      =Bf(a*c-b*s);
    row[d+half] =Bf(b*c+a*s);
}
// Multi-query causal attention. Query rows are at absolute positions
// [q0 .. q0+R-1]; the KV cache holds `L` keys at absolute positions [0..L-1]
// (so q0 = L - R). Row r attends keys 0..(q0+r). GQA: head h -> kv head h/(H/KV).
// out [R, H*hd].
__global__ void k_attn_causal_cached(const bf* q,const bf* kcache,const bf* vcache,
        bf* out,int R,int H,int KV,int hd,int L,int q0,float scale){
    int r=blockIdx.y; int h=blockIdx.x; if(r>=R||h>=H)return;
    int kvh=h/(H/KV);
    int lim=q0+r;                  // inclusive last key index (causal)
    extern __shared__ float sh[];  // [L]
    const bf* qh=q+((long)r*H+h)*hd;
    float mx=-1e30f;
    for(int j=threadIdx.x;j<=lim;j+=blockDim.x){
        const bf* kj=kcache+((long)j*KV+kvh)*hd;
        float s=0;for(int d=0;d<hd;d++)s+=F(qh[d])*F(kj[d]);
        s*=scale;sh[j]=s;mx=fmaxf(mx,s);
    }
    __shared__ float red[32];__shared__ float gmax,gden;
    for(int o=warpSize/2;o>0;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    if((threadIdx.x&31)==0)red[threadIdx.x>>5]=mx;__syncthreads();
    if(threadIdx.x==0){float m=-1e30f;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)m=fmaxf(m,red[i]);gmax=m;}__syncthreads();
    float den=0;for(int j=threadIdx.x;j<=lim;j+=blockDim.x){float e=__expf(sh[j]-gmax);sh[j]=e;den+=e;}
    for(int o=warpSize/2;o>0;o>>=1)den+=__shfl_down_sync(0xffffffff,den,o);
    if((threadIdx.x&31)==0)red[threadIdx.x>>5]=den;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=red[i];gden=t;}__syncthreads();
    float inv=gden>0.f?1.f/gden:0.f;
    for(int d=threadIdx.x;d<hd;d+=blockDim.x){
        float acc=0;for(int j=0;j<=lim;j++){const bf* vj=vcache+((long)j*KV+kvh)*hd;acc+=sh[j]*F(vj[d]);}
        out[((long)r*H+h)*hd+d]=Bf(acc*inv);
    }
}
// Sum the 32 codebook embeds of one frame into out [hidden].
//   embed_audio [num_codebooks*audio_vocab, hidden]; codes[c] (host-resident on
//   device) gives the id of codebook c; row = codes[c] + c*audio_vocab.
__global__ void k_embed_audio_sum(const bf* embed_audio,const int* codes,
        bf* out,int hidden,int num_cb,int audio_vocab){
    int d=blockIdx.x*blockDim.x+threadIdx.x;if(d>=hidden)return;
    float a=0;
    for(int c=0;c<num_cb;c++){
        long row=(long)codes[c]+(long)c*audio_vocab;
        a+=F(embed_audio[row*hidden+d]);
    }
    out[d]=Bf(a);
}

__global__ void k_cast_f32_bf16(const float* in,bf* out,long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) out[i]=Bf(in[i]);
}
__global__ void k_resolve_codebook(const bf* embed_sum,const bf* usage,bf* out,
        int rows,int dim,float eps){
    int r=blockIdx.y*blockDim.y+threadIdx.y, d=blockIdx.x*blockDim.x+threadIdx.x;
    if(r>=rows||d>=dim)return;
    float u=F(usage[r]); if(u<eps)u=eps;
    out[(long)r*dim+d]=Bf(F(embed_sum[(long)r*dim+d])/u);
}

dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}

struct BBScratch {
    int hidden, NH, KV, hd, QD, KD, inter, maxL;
    bf *resid, *normed, *q, *k, *v, *attn, *attn_o, *gate, *up, *mlp;
    std::vector<bf*> kcache, vcache;   // [maxL, KV, hd] per layer
    cudaStream_t S;
    cublasHandle_t cublas;   // owned by csm_generate_audio, bound to S
};

// Run one backbone block over R rows. `resid` holds [R, hidden] in/out. KV cache
// length grows to `L` (the new R rows occupy slots L-R .. L-1, already copied in
// by the caller via the q0 position math). q0 = L - R.
void bb_layer(const CsmBackboneRawWeights& w,const CsmBackboneLayerRaw& L,
              BBScratch& s,int li,int R,int Lkv){
    const int H=s.hidden,NH=s.NH,KV=s.KV,hd=s.hd,QD=s.QD,KD=s.KD;
    const int q0=Lkv-R;
    const float scale=1.f/sqrtf((float)hd);
    cudaStream_t S=s.S;
    kernels::norm::rmsnorm_bf16(s.resid,L.in_ln_w,s.normed,R,H,w.norm_eps,S);
    kernels::gemm::act_x_wt_bf16(s.cublas,s.normed,L.q,s.q,R,QD,H);
    kernels::gemm::act_x_wt_bf16(s.cublas,s.normed,L.k,s.k,R,KD,H);
    kernels::gemm::act_x_wt_bf16(s.cublas,s.normed,L.v,s.v,R,KD,H);
    { dim3 g((hd/2+15)/16,(NH+15)/16,R); k_rope_yarn_1d<<<g,B2,0,S>>>(s.q,R,NH,hd,q0,
        w.rope_theta,w.rope_factor,w.rope_low_freq_factor,w.rope_high_freq_factor,(float)w.rope_original_max_position); }
    { dim3 g((hd/2+15)/16,(KV+15)/16,R); k_rope_yarn_1d<<<g,B2,0,S>>>(s.k,R,KV,hd,q0,
        w.rope_theta,w.rope_factor,w.rope_low_freq_factor,w.rope_high_freq_factor,(float)w.rope_original_max_position); }
    // append the R new k,v rows into the cache at slots q0..q0+R-1
    CK(cudaMemcpyAsync(s.kcache[li]+(long)q0*KD,s.k,(long)R*KD*sizeof(bf),cudaMemcpyDeviceToDevice,S));
    CK(cudaMemcpyAsync(s.vcache[li]+(long)q0*KD,s.v,(long)R*KD*sizeof(bf),cudaMemcpyDeviceToDevice,S));
    { dim3 g(NH,R); k_attn_causal_cached<<<g,128,(size_t)Lkv*sizeof(float),S>>>(s.q,s.kcache[li],s.vcache[li],s.attn,R,NH,KV,hd,Lkv,q0,scale); }
    kernels::gemm::act_x_wt_bf16(s.cublas,s.attn,L.o,s.attn_o,R,H,QD);
    k_add<<<(long)(R*H+255)/256,256,0,S>>>(s.resid,s.attn_o,(long)R*H);
    kernels::norm::rmsnorm_bf16(s.resid,L.post_ln_w,s.normed,R,H,w.norm_eps,S);
    kernels::gemm::act_x_wt_bf16(s.cublas,s.normed,L.gate,s.gate,R,s.inter,H);
    kernels::gemm::act_x_wt_bf16(s.cublas,s.normed,L.up,s.up,R,s.inter,H);
    kernels::mlp::swiglu_bf16(s.gate,s.up,s.gate,R*s.inter,S);
    kernels::gemm::act_x_wt_bf16(s.cublas,s.gate,L.down,s.mlp,R,H,s.inter);
    k_add<<<(long)(R*H+255)/256,256,0,S>>>(s.resid,s.mlp,(long)R*H);
}

}  // namespace

void csm_cast_f32_to_bf16(const float* in,bf* out,long n,cudaStream_t S){
    k_cast_f32_bf16<<<(n+255)/256,256,0,S>>>(in,out,n);
    CK(cudaStreamSynchronize(S));
}

void mimi_resolve_codebook_embed(const bf* embed_sum,const bf* usage,bf* out,
        int rows,int dim,float eps,cudaStream_t S){
    dim3 g((dim+15)/16,(rows+15)/16);
    k_resolve_codebook<<<g,B2,0,S>>>(embed_sum,usage,out,rows,dim,eps);
    CK(cudaStreamSynchronize(S));
}

int csm_generate_audio(const CsmBackboneRawWeights& w,
                       const CsmDepthRawWeights& depth,
                       const MimiDecoderRawWeights& mimi,
                       const std::int32_t* prompt_ids,
                       int n_prompt,
                       int max_frames,
                       std::vector<float>& out_pcm,
                       std::vector<std::int32_t>* out_codes,
                       cudaStream_t S){
    const int H=w.hidden, NH=w.num_heads, KV=w.num_kv_heads, hd=w.head_dim;
    const int QD=NH*hd, KD=KV*hd, NCB=w.num_codebooks, AV=w.audio_vocab;
    if(n_prompt<=0) throw std::runtime_error("csm_generate_audio: empty prompt");
    if(max_frames<=0) max_frames=256;
    const int maxL=n_prompt+max_frames+1;

    auto MAL=[&](long n){bf* d;CK(cudaMalloc(&d,n*sizeof(bf)));return d;};
    // One handle for the whole generation, bound to this call's stream.
    // RAII: cublasCreate is expensive enough to not want per-layer, and
    // generate_audio is already the coarse entry point (it loops frames
    // internally). `d1` copies this scratch, so it carries the handle too.
    kernels::gemm::CublasHandle cublas(S);

    BBScratch s; s.cublas=cublas.handle(); s.hidden=H;s.NH=NH;s.KV=KV;s.hd=hd;s.QD=QD;s.KD=KD;s.inter=w.intermediate;s.maxL=maxL;s.S=S;
    const int R0=n_prompt;            // prefill rows
    s.resid=MAL((long)R0*H); s.normed=MAL((long)R0*H);
    s.q=MAL((long)R0*QD); s.k=MAL((long)R0*KD); s.v=MAL((long)R0*KD);
    s.attn=MAL((long)R0*QD); s.attn_o=MAL((long)R0*H);
    s.gate=MAL((long)R0*s.inter); s.up=MAL((long)R0*s.inter); s.mlp=MAL((long)R0*H);
    s.kcache.resize(w.num_layers); s.vcache.resize(w.num_layers);
    for(int l=0;l<w.num_layers;l++){ s.kcache[l]=MAL((long)maxL*KD); s.vcache[l]=MAL((long)maxL*KD); }
    bf* lm_logits=MAL(AV);
    bf* last_hidden=MAL(H);           // backbone last-row post-norm hidden (depth seed)
    int* d_codes; CK(cudaMalloc(&d_codes,sizeof(int)*NCB));   // device frame codes
    int* d_arg;   CK(cudaMalloc(&d_arg,sizeof(int)));

    // ── Prefill the text prompt (rows 0..n_prompt-1) ──────────────────────
    // embed_text_tokens[id] for each prompt id.
    {
        std::vector<bf> h_resid; (void)h_resid;
        // Copy each embed row into resid (device->device from embed_text).
        for(int r=0;r<R0;r++){
            long row=(long)prompt_ids[r];
            CK(cudaMemcpyAsync(s.resid+(long)r*H, w.embed_text+row*H,(long)H*sizeof(bf),cudaMemcpyDeviceToDevice,S));
        }
    }
    int Lkv=R0;
    for(int l=0;l<w.num_layers;l++) bb_layer(w,w.layers[l],s,l,R0,Lkv);
    // final norm on the LAST row only (the one that predicts frame 0's cb0).
    kernels::norm::rmsnorm_bf16(s.resid+(long)(R0-1)*H,w.norm_w,last_hidden,1,H,w.norm_eps,S);

    // Single-row decode scratch (reuse layer scratch sized for 1 row).
    bf* d_resid=MAL(H); bf* d_normed=MAL(H);
    bf* d_q=MAL(QD); bf* d_k=MAL(KD); bf* d_v=MAL(KD);
    bf* d_attn=MAL(QD); bf* d_attn_o=MAL(H);
    bf* d_gate=MAL(s.inter); bf* d_up=MAL(s.inter); bf* d_mlp=MAL(H);
    BBScratch d1=s; d1.resid=d_resid; d1.normed=d_normed; d1.q=d_q; d1.k=d_k; d1.v=d_v;
    d1.attn=d_attn; d1.attn_o=d_attn_o; d1.gate=d_gate; d1.up=d_up; d1.mlp=d_mlp;

    std::vector<std::int32_t> all_codes;       // [NCB * n_frames], codebook-major appended per frame
    std::vector<std::int32_t> frame(NCB);
    int n_frames=0;

    for(int f=0; f<max_frames; ++f){
        // cb0 = argmax(lm_head(last_hidden))
        kernels::gemm::act_x_wt_bf16(s.cublas,last_hidden,w.lm_head,lm_logits,1,AV,H);
        kernels::sample::argmax_bf16(lm_logits,d_arg,1,AV,S);
        int cb0; CK(cudaMemcpyAsync(&cb0,d_arg,sizeof(int),cudaMemcpyDeviceToHost,S));CK(cudaStreamSynchronize(S));

        // depth decoder -> cb1..cb31 (seeded by last_hidden + cb0)
        std::int32_t cb_rest[64];
        run_csm_depth_decoder_frame(depth,last_hidden,cb0,cb_rest,S);

        frame[0]=cb0; for(int c=1;c<NCB;c++) frame[c]=cb_rest[c-1];

        // Stop if codebooks 0..NCB-2 are all == codebook_eos (HF checks
        // input_ids[:, -1, :-1] == codebook_eos_token_id, i.e. the first 31
        // codebooks excluding the last — generation_csm.py::_sample).
        bool all_eos=true; for(int c=0;c<NCB-1;c++) if(frame[c]!=w.codebook_eos_token_id) { all_eos=false; break; }
        if(all_eos) break;

        // accumulate codes
        for(int c=0;c<NCB;c++) all_codes.push_back(frame[c]);
        ++n_frames;
        if(out_codes){ for(int c=0;c<NCB;c++) out_codes->push_back(frame[c]); }

        if(f+1>=max_frames) break;   // don't decode a row we won't use

        // ── Backbone decode step: re-embed the 32-code frame as the next row ──
        CK(cudaMemcpyAsync(d_codes,frame.data(),sizeof(int)*NCB,cudaMemcpyHostToDevice,S));
        k_embed_audio_sum<<<(H+255)/256,256,0,S>>>(w.embed_audio,d_codes,d1.resid,H,NCB,AV);
        int newL=Lkv+1;
        for(int l=0;l<w.num_layers;l++) bb_layer(w,w.layers[l],d1,l,1,newL);
        Lkv=newL;
        kernels::norm::rmsnorm_bf16(d1.resid,w.norm_w,last_hidden,1,H,w.norm_eps,S);
    }
    CK(cudaStreamSynchronize(S));

    // ── Mimi decode: codes [NCB, n_frames] (codebook-major) -> PCM ──────────
    if(n_frames>0){
        // all_codes is frame-major [n_frames][NCB]; transpose to codebook-major
        // [NCB][n_frames] for run_mimi_decoder.
        std::vector<std::int32_t> codes_cb(  (long)NCB*n_frames );
        for(int f=0;f<n_frames;f++) for(int c=0;c<NCB;c++) codes_cb[(long)c*n_frames+f]=all_codes[(long)f*NCB+c];
        int nsamp=mimi_decoder_num_samples(mimi,n_frames);
        float* d_wave; CK(cudaMalloc(&d_wave,(long)nsamp*sizeof(float)));
        int got=run_mimi_decoder(mimi,codes_cb.data(),n_frames,d_wave,S);
        CK(cudaStreamSynchronize(S));
        out_pcm.resize(got);
        CK(cudaMemcpy(out_pcm.data(),d_wave,(long)got*sizeof(float),cudaMemcpyDeviceToHost));
        cudaFree(d_wave);
    } else {
        out_pcm.clear();
    }

    // free
    cudaFree(s.resid);cudaFree(s.normed);cudaFree(s.q);cudaFree(s.k);cudaFree(s.v);
    cudaFree(s.attn);cudaFree(s.attn_o);cudaFree(s.gate);cudaFree(s.up);cudaFree(s.mlp);
    for(int l=0;l<w.num_layers;l++){cudaFree(s.kcache[l]);cudaFree(s.vcache[l]);}
    cudaFree(lm_logits);cudaFree(last_hidden);cudaFree(d_codes);cudaFree(d_arg);
    cudaFree(d_resid);cudaFree(d_normed);cudaFree(d_q);cudaFree(d_k);cudaFree(d_v);
    cudaFree(d_attn);cudaFree(d_attn_o);cudaFree(d_gate);cudaFree(d_up);cudaFree(d_mlp);
    return n_frames;
}

}  // namespace pie_cuda_driver::model
