// Parity harness for the Gemma-4 vision encoder, against the fp32 reference
// dumps from scripts/gemma4_vision_parity_ref.py.
//
// This used to carry its own copy of `run_gemma4_vision` and its own thirteen
// kernels -- the port source, from before the driver had them. The copy and
// the driver stayed identical (13 kernels, the same 25 launches in the same
// order), so the parity number was true of the driver by coincidence rather
// than by construction, and a drift-checking script existed to keep the
// coincidence honest. It calls the driver now. The copy and the script are
// gone, and a PASS here is a statement about the code that ships.
//
//   nvcc -O2 -arch=sm_89 -std=c++17 gemma4_vision_full_parity_bf16.cu -o /tmp/vbf
//   /tmp/vbf /tmp/gemma4_vision_parity
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "model/gemma4/gemma4_vision_forward.hpp"

typedef __nv_bfloat16 bf;
using pie_cuda_driver::model::VisRawWeights;
using pie_cuda_driver::model::VisLayerRaw;
using pie_cuda_driver::model::VisClipRaw;
using pie_cuda_driver::model::run_gemma4_vision;

namespace {
#define CK(x) do{cudaError_t e=(x);if(e){std::fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);}}while(0)
__device__ __forceinline__ float F(bf x){return __bfloat162float(x);}
__device__ __forceinline__ bf   Bf(float x){return __float2bfloat16(x);}

// ── harness: load f32 npy, convert to bf16, run, compare ────────────────────
struct Npy{std::vector<int64_t> shape;char kind=0;int isz=0;std::vector<uint8_t> data;int64_t numel()const{int64_t n=1;for(auto d:shape)n*=d;return n;}};
Npy load_npy(const std::string& p){std::ifstream f(p,std::ios::binary);if(!f){std::fprintf(stderr,"open %s\n",p.c_str());std::exit(2);}
    char m[6];f.read(m,6);uint8_t maj=f.get(),mn=f.get();(void)mn;uint32_t hl;if(maj==1){uint16_t h;f.read((char*)&h,2);hl=h;}else f.read((char*)&hl,4);
    std::string hdr(hl,0);f.read(hdr.data(),hl);Npy o;auto dp=hdr.find("'descr'");auto q=hdr.find('\'',hdr.find(':',dp)+1);
    std::string d=hdr.substr(q+1,hdr.find('\'',q+1)-q-1);o.kind=d[1];o.isz=std::atoi(d.substr(2).c_str());
    auto sp=hdr.find("'shape'");auto lp=hdr.find('(',sp),rp=hdr.find(')',lp);std::string sh=hdr.substr(lp+1,rp-lp-1);size_t i=0;
    while(i<sh.size()){while(i<sh.size()&&!isdigit(sh[i]))++i;if(i>=sh.size())break;int64_t v=0;while(i<sh.size()&&isdigit(sh[i]))v=v*10+(sh[i++]-'0');o.shape.push_back(v);}
    o.data.resize((size_t)o.numel()*o.isz);f.read((char*)o.data.data(),(std::streamsize)o.data.size());return o;}
std::string DIR,g_dir;std::map<std::string,bf*> cache;
// A staged checkpoint against the HF reference.
//
// `max_abs` alone is not readable on these tensors: they have a wide dynamic
// range (layer_last has rms 41 and a peak of 1664), so a max deviation quoted
// next to the rms looks alarming when it is ordinary bf16 rounding on the
// largest element. bf16 carries an 8-bit significand, so the quantum at
// magnitude m is about m/256 -- printed here as `quantum`, with the deviation
// as a fraction of the reference's own peak. Numbers you can judge without
// going away and computing the scale yourself.
void ckpt(const char* tag,const bf* d,long n){std::vector<bf> y(n);CK(cudaMemcpy(y.data(),d,n*sizeof(bf),cudaMemcpyDeviceToHost));
    Npy r=load_npy(g_dir+"/"+tag+"_f32.npy");const float* rp=(const float*)r.data.data();
    double ma=0,sq=0,rmax=0,esq=0,rsq=0;
    for(long i=0;i<n;i++){float v=__bfloat162float(y[i]);double e=(double)v-rp[i];
        ma=std::max(ma,std::abs(e));sq+=(double)v*v;esq+=e*e;rsq+=(double)rp[i]*rp[i];
        rmax=std::max(rmax,std::abs((double)rp[i]));}
    std::printf("  ckpt %-20s rms=%8.2f  max|ref|=%9.1f  quantum=%7.2f  "
                "max_abs=%.3e (%.2f%% of peak)  rel_rms=%.3f%%\n",
                tag,std::sqrt(sq/n),rmax,rmax/256.0,ma,100*ma/(rmax?rmax:1),
                100*std::sqrt(esq/(rsq?rsq:1)));}
bf* Wbf(const std::string& name){auto it=cache.find(name);if(it!=cache.end())return it->second;
    Npy n=load_npy(DIR+"/weights/"+name+".npy");std::vector<bf> hb(n.numel());const float* fp=(const float*)n.data.data();
    for(int64_t i=0;i<n.numel();i++)hb[i]=__float2bfloat16(fp[i]);bf* d;CK(cudaMalloc(&d,hb.size()*sizeof(bf)));
    CK(cudaMemcpy(d,hb.data(),hb.size()*sizeof(bf),cudaMemcpyHostToDevice));cache[name]=d;return d;}
// The driver holds clip bounds as DEVICE bf16 (they arrive as DeviceTensors
// with the model). The dump has them as fp32 scalars, so upload each as a
// one-element buffer rather than passing a host float.
const bf* scal_dev(const std::string& n){auto it=cache.find("@"+n);if(it!=cache.end())return it->second;
    Npy x=load_npy(DIR+"/weights/"+n+".npy");bf hv=__float2bfloat16(((float*)x.data.data())[0]);
    bf* d;CK(cudaMalloc(&d,sizeof(bf)));CK(cudaMemcpy(d,&hv,sizeof(bf),cudaMemcpyHostToDevice));
    cache["@"+n]=d;return d;}
VisClipRaw clip(const std::string& b){return {Wbf(b+".linear.weight"),scal_dev(b+".input_min"),
    scal_dev(b+".input_max"),scal_dev(b+".output_min"),scal_dev(b+".output_max")};}
}
int main(int argc,char**argv){
    DIR=argc>1?argv[1]:"/tmp/gemma4_vision_parity";
    const bool real = argc>2 && std::string(argv[2])=="real";  // real processor output (variable, padded)
    const int Hd=768,TXT=2560;
    const char* pixf = real? "/realimg_pixel_values_f32.npy" : "/input_pixel_values_f32.npy";
    const char* posf = real? "/realimg_position_ids.npy"     : "/input_position_ids.npy";
    auto pixn=load_npy(DIR+pixf);auto posn=load_npy(DIR+posf);
    // In real mode strip padding (positions (-1,-1)) → N = valid patch count.
    int N = (int)pixn.shape[pixn.shape.size()-2];
    if(real){const float* p=(const float*)posn.data.data();int nv=0;for(int i=0;i<N;i++)if(p[2*i]>=0)nv++;N=nv;}
    const int OUTL = real ? N/9 : 280;
    std::printf("mode=%s  N=%d  OUTL=%d\n", real?"real":"synthetic", N, OUTL);
    std::vector<bf> pixb(N*Hd);for(int i=0;i<N*Hd;i++)pixb[i]=__float2bfloat16(((float*)pixn.data.data())[i]);
    bf* d_pix;CK(cudaMalloc(&d_pix,(long)N*Hd*sizeof(bf)));CK(cudaMemcpy(d_pix,pixb.data(),(long)N*Hd*sizeof(bf),cudaMemcpyHostToDevice));
    float* d_pos;CK(cudaMalloc(&d_pos,(long)N*2*4));CK(cudaMemcpy(d_pos,posn.data.data(),(long)N*2*4,cudaMemcpyHostToDevice));
    const float* hp=(const float*)posn.data.data();int maxx=0;for(int n=0;n<N;n++)maxx=std::max(maxx,(int)llrintf(hp[2*n]));int gx=(maxx+1)/3;
    std::vector<int> grp(N);for(int n=0;n<N;n++)grp[n]=((int)llrintf(hp[2*n])/3)+gx*((int)llrintf(hp[2*n+1])/3);
    int* d_grp;CK(cudaMalloc(&d_grp,N*4));CK(cudaMemcpy(d_grp,grp.data(),N*4,cudaMemcpyHostToDevice));

    VisRawWeights W;W.patch_w=Wbf("vision.patch_embedder.input_proj.weight");
    W.pos_table=Wbf("vision.patch_embedder.position_embedding_table");W.embed_proj=Wbf("embed.embedding_projection.weight");
    for(int l=0;l<16;l++){std::string p="vision.encoder.layers."+std::to_string(l)+".";VisLayerRaw L;
        L.in_ln=Wbf(p+"input_layernorm.weight");L.post_attn_ln=Wbf(p+"post_attention_layernorm.weight");
        L.pre_ff_ln=Wbf(p+"pre_feedforward_layernorm.weight");L.post_ff_ln=Wbf(p+"post_feedforward_layernorm.weight");
        L.q_norm=Wbf(p+"self_attn.q_norm.weight");L.k_norm=Wbf(p+"self_attn.k_norm.weight");
        L.q=clip(p+"self_attn.q_proj");L.k=clip(p+"self_attn.k_proj");L.v=clip(p+"self_attn.v_proj");L.o=clip(p+"self_attn.o_proj");
        L.gate=clip(p+"mlp.gate_proj");L.up=clip(p+"mlp.up_proj");L.down=clip(p+"mlp.down_proj");W.layers.push_back(L);}

    bf* d_out;CK(cudaMalloc(&d_out,(long)OUTL*TXT*sizeof(bf)));
    g_dir=DIR;
    // Intermediate checkpoints exist only for the synthetic input; `real` mode
    // passes no tap and the driver skips them.
    run_gemma4_vision(W,d_pix,d_pos,d_grp,N,OUTL,d_out,/*stream=*/0,
                      real?nullptr:&ckpt);
    std::vector<bf> outb(OUTL*TXT);CK(cudaMemcpy(outb.data(),d_out,(long)OUTL*TXT*sizeof(bf),cudaMemcpyDeviceToHost));
    long n=(long)OUTL*TXT;std::vector<float> y(n);for(long i=0;i<n;i++)y[i]=__bfloat162float(outb[i]);
    auto report=[&](const char* tag,const char* file){
        Npy r=load_npy(DIR+"/"+file);const float* rp=(const float*)r.data.data();
        double dn=0,rn=0,dot=0,en=0;for(long i=0;i<n;i++){double e=y[i]-rp[i];en+=e*e;dn+=(double)y[i]*y[i];rn+=(double)rp[i]*rp[i];dot+=(double)y[i]*rp[i];}
        std::printf("  vs %-16s rel_rms_err=%.3f%%  cosine=%.5f\n",tag,100*std::sqrt(en/rn),dot/std::sqrt(dn*rn));
        return std::sqrt(en/rn);};
    std::printf("projected(bf16): rms=%.3f\n",std::sqrt([&]{double s=0;for(long i=0;i<n;i++)s+=(double)y[i]*y[i];return s;}()/n));
    double e;
    if(real){
        // Real processor output (variable patch count, padding stripped) →
        // encoder → projected, vs HF fp32 (bf16-vs-fp32 ≈ 2%).
        e=report("HF-fp32(real)","realimg_projected_f32.npy");
    } else {
        e=report("HF-bf16","projected.npy");      // both bf16 — the real comparison
        report("HF-fp32","projected_f32.npy");
    }
    bool pass = e < (real?0.06:0.10);
    std::printf("%s\n",pass?"BF16 PARITY PASS":"BF16 PARITY FAIL");return pass?0:1;}
