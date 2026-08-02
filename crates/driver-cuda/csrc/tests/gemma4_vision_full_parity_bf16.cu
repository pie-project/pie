// Full Gemma-4 vision encoder forward in **bf16** (driver precision), written
// as a driver-portable `run_gemma4_vision(...)` over raw device pointers, plus a
// standalone harness that checks it against the fp32 reference dumps.
//
// bf16 storage + fp32 compute (matching the driver). The kernels + the
// `run_gemma4_vision` entry point port directly into the driver module — the
// driver passes `DeviceTensor::data()` pointers and its own scratch.
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

typedef __nv_bfloat16 bf;
namespace {
#define CK(x) do{cudaError_t e=(x);if(e){std::fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);}}while(0)
__device__ __forceinline__ float F(bf x){return __bfloat162float(x);}
__device__ __forceinline__ bf   Bf(float x){return __float2bfloat16(x);}

// ── kernels (bf16 storage, fp32 compute) ────────────────────────────────────
__global__ void k_scale(const bf* p,bf* o,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t)o[i]=Bf(2.f*(F(p[i])-0.5f));}
__global__ void k_matmul(const bf* x,const bf* W,bf* y,int N,int K,int O){ // y[n,o]=sum_k x[n,k]*W[o,k]
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||o>=O)return;
    const bf* xr=x+(long)n*K;const bf* wr=W+(long)o*K;float a=0;for(int k=0;k<K;k++)a+=F(xr[k])*F(wr[k]);y[(long)n*O+o]=Bf(a);}
__global__ void k_addpos(bf* y,const bf* tb,const float* pos,int N,int O,int P){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||o>=O)return;
    long x=(long)llrintf(pos[2L*n]),yy=(long)llrintf(pos[2L*n+1]);if(x<0)x=0;if(yy<0)yy=0;
    y[(long)n*O+o]=Bf(F(y[(long)n*O+o])+F(tb[(0L*P+x)*O+o])+F(tb[(1L*P+yy)*O+o]));}
__global__ void k_clamp(const bf* x,bf* o,float lo,float hi,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t){float v=F(x[i]);o[i]=Bf(v<lo?lo:(v>hi?hi:v));}}
__global__ void k_rms(const bf* x,const bf* w,bf* o,int R,int D,float eps){
    int r=blockIdx.x;if(r>=R)return;const bf* xr=x+(long)r*D;bf* orow=o+(long)r*D;
    float loc=0;for(int d=threadIdx.x;d<D;d+=blockDim.x){float v=F(xr[d]);loc+=v*v;}
    for(int s=warpSize/2;s>0;s>>=1)loc+=__shfl_down_sync(0xffffffff,loc,s);
    __shared__ float warp[32],ss;if((threadIdx.x&31)==0)warp[threadIdx.x>>5]=loc;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=warp[i];ss=rsqrtf(t/D+eps);}__syncthreads();
    float inv=ss;for(int d=threadIdx.x;d<D;d+=blockDim.x)orow[d]=Bf(F(xr[d])*inv*(w?F(w[d]):1.f));}
__global__ void k_rope(bf* q,const float* pos,int N,int H,float theta){
    int n=blockIdx.z,head=blockIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||head>=H||c>=16)return;
    bf* v=q+(((long)n*H+head)*64);float px=pos[2L*n],py=pos[2L*n+1];float invf=powf(theta,-(float)c/16.f);
    float cx=cosf(px*invf),sx=sinf(px*invf),cy=cosf(py*invf),sy=sinf(py*invf);
    float a=F(v[c]),b=F(v[c+16]);v[c]=Bf(a*cx-b*sx);v[c+16]=Bf(b*cx+a*sx);
    float e=F(v[32+c]),f=F(v[48+c]);v[32+c]=Bf(e*cy-f*sy);v[48+c]=Bf(f*cy+e*sy);}
__global__ void k_qk(const bf* q,const bf* k,float* s,int N,int H,int head,float scale){
    int i=blockIdx.y*blockDim.y+threadIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N||j>=N)return;
    const bf* qi=q+((long)i*H+head)*64;const bf* kj=k+((long)j*H+head)*64;
    float a=0;for(int d=0;d<64;d++)a+=F(qi[d])*F(kj[d]);s[(long)i*N+j]=a*scale;}
__global__ void k_softmax(float* s,int N){int i=blockIdx.x;if(i>=N)return;float* r=s+(long)i*N;
    float mx=-1e30f;for(int j=threadIdx.x;j<N;j+=blockDim.x)mx=fmaxf(mx,r[j]);
    for(int o=warpSize/2;o>0;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    __shared__ float wm[32],wsv[32],smx,ssum;if((threadIdx.x&31)==0)wm[threadIdx.x>>5]=mx;__syncthreads();
    if(threadIdx.x==0){float m=-1e30f;int nw=(blockDim.x+31)/32;for(int i2=0;i2<nw;i2++)m=fmaxf(m,wm[i2]);smx=m;}__syncthreads();
    float sm=0;for(int j=threadIdx.x;j<N;j+=blockDim.x){float e=__expf(r[j]-smx);r[j]=e;sm+=e;}
    for(int o=warpSize/2;o>0;o>>=1)sm+=__shfl_down_sync(0xffffffff,sm,o);if((threadIdx.x&31)==0)wsv[threadIdx.x>>5]=sm;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i2=0;i2<nw;i2++)t+=wsv[i2];ssum=t;}__syncthreads();
    float inv=1.f/ssum;for(int j=threadIdx.x;j<N;j+=blockDim.x)r[j]*=inv;}
__global__ void k_av(const float* s,const bf* v,bf* o,int N,int H,int head){
    int n=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||d>=64)return;
    const float* sr=s+(long)n*N;float a=0;for(int j=0;j<N;j++)a+=sr[j]*F(v[((long)j*H+head)*64+d]);
    o[((long)n*H+head)*64+d]=Bf(a);}
__global__ void k_gelu_mul(const bf* g,const bf* u,bf* o,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t){float x=F(g[i]);float gl=0.5f*x*(1.f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)));o[i]=Bf(gl*F(u[i]));}}
__global__ void k_add(bf* h,const bf* x,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t)h[i]=Bf(F(h[i])+F(x[i]));}
__global__ void k_pool(const bf* h,const int* grp,float* o,int N,int D,float k2){
    int n=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||d>=D)return;
    atomicAdd(&o[(long)grp[n]*D+d],F(h[(long)n*D+d])/k2);}
__global__ void k_pool_finish(const float* in,bf* o,float s,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t)o[i]=Bf(in[i]*s);}

dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}

// debug: compare a bf16 device buffer to a fp32 reference dump (localize bf16 drift)
std::string g_dir; bool g_dbg=false;
struct NpyF;  // fwd
void ckpt(const char* tag,const bf* d,long n);

// ── driver-portable weight handles (raw bf16 pointers + fp32 clip scalars) ──
struct Clip{const bf* w;float imin,imax,omin,omax;};
struct VisLayer{const bf *in_ln,*post_attn_ln,*pre_ff_ln,*post_ff_ln,*q_norm,*k_norm;Clip q,k,v,o,gate,up,down;};
struct VisWeights{const bf* patch_w;const bf* pos_table;const bf* embed_proj;std::vector<VisLayer> layers;};

// Full forward. `pixel`(bf16 [N,768]), `pos`(f32 [N,2]) on device; writes
// `out_proj`(bf16 [OUTL,2560]). `grp`(int [N]) = precomputed pool groups.
// Allocates its own scratch (driver passes a scratch arena instead).
void run_gemma4_vision(const VisWeights& W,const bf* pixel,const float* pos,const int* grp,
                       int N,int OUTL,bf* out_proj){
    const int Hd=768,NH=12,IM=3072,P=10240,TXT=2560;const float EPS=1e-6f,THETA=100.f;
    auto MAL=[&](long n){bf* d;CK(cudaMalloc(&d,n*sizeof(bf)));return d;};
    bf *h=MAL((long)N*Hd),*hn=MAL((long)N*Hd),*xc=MAL((long)N*IM),*q=MAL((long)N*Hd),*k=MAL((long)N*Hd),*v=MAL((long)N*Hd),
       *attn=MAL((long)N*Hd),*gate=MAL((long)N*IM),*up=MAL((long)N*IM),*act=MAL((long)N*IM),*tmp=MAL((long)N*Hd);
    float* scr;CK(cudaMalloc(&scr,(long)N*N*4));
    auto clin=[&](const bf* x,bf* out,const Clip& c,int Kin,int Out){
        k_clamp<<<((long)N*Kin+255)/256,256>>>(x,xc,c.imin,c.imax,(long)N*Kin);
        k_matmul<<<G2(Out,N),B2>>>(xc,c.w,out,N,Kin,Out);
        k_clamp<<<((long)N*Out+255)/256,256>>>(out,out,c.omin,c.omax,(long)N*Out);};
    // patch embed
    k_scale<<<((long)N*Hd+255)/256,256>>>(pixel,hn,(long)N*Hd);
    k_matmul<<<G2(Hd,N),B2>>>(hn,W.patch_w,h,N,Hd,Hd);
    k_addpos<<<G2(Hd,N),B2>>>(h,W.pos_table,pos,N,Hd,P);
    int li=0;
    for(const auto& L:W.layers){
        k_rms<<<N,256>>>(h,L.in_ln,hn,N,Hd,EPS);
        clin(hn,q,L.q,Hd,Hd);clin(hn,k,L.k,Hd,Hd);clin(hn,v,L.v,Hd,Hd);
        k_rms<<<N*NH,64>>>(q,L.q_norm,q,N*NH,64,EPS);k_rms<<<N*NH,64>>>(k,L.k_norm,k,N*NH,64,EPS);k_rms<<<N*NH,64>>>(v,nullptr,v,N*NH,64,EPS);
        dim3 rg(1,NH,N);k_rope<<<rg,32>>>(q,pos,N,NH,THETA);k_rope<<<rg,32>>>(k,pos,N,NH,THETA);
        for(int hh=0;hh<NH;hh++){k_qk<<<G2(N,N),B2>>>(q,k,scr,N,NH,hh,1.0f);k_softmax<<<N,256>>>(scr,N);k_av<<<G2(64,N),B2>>>(scr,v,attn,N,NH,hh);}
        clin(attn,tmp,L.o,Hd,Hd);
        k_rms<<<N,256>>>(tmp,L.post_attn_ln,tmp,N,Hd,EPS);
        k_add<<<((long)N*Hd+255)/256,256>>>(h,tmp,(long)N*Hd);
        k_rms<<<N,256>>>(h,L.pre_ff_ln,hn,N,Hd,EPS);
        clin(hn,gate,L.gate,Hd,IM);clin(hn,up,L.up,Hd,IM);
        k_gelu_mul<<<((long)N*IM+255)/256,256>>>(gate,up,act,(long)N*IM);
        clin(act,tmp,L.down,IM,Hd);
        k_rms<<<N,256>>>(tmp,L.post_ff_ln,tmp,N,Hd,EPS);
        k_add<<<((long)N*Hd+255)/256,256>>>(h,tmp,(long)N*Hd);
        if(g_dbg&&li==0){CK(cudaDeviceSynchronize());ckpt("layer0",h,(long)N*Hd);}
        li++;
    }
    if(g_dbg){CK(cudaDeviceSynchronize());ckpt("layer_last",h,(long)N*Hd);}
    // pool → ×sqrt(Hd)
    float* pf;CK(cudaMalloc(&pf,(long)OUTL*Hd*4));CK(cudaMemset(pf,0,(long)OUTL*Hd*4));
    k_pool<<<G2(Hd,N),B2>>>(h,grp,pf,N,Hd,9.f);
    bf* pooled=MAL((long)OUTL*Hd);k_pool_finish<<<((long)OUTL*Hd+255)/256,256>>>(pf,pooled,sqrtf((float)Hd),(long)OUTL*Hd);
    if(g_dbg){CK(cudaDeviceSynchronize());ckpt("pooled_last_hidden",pooled,(long)OUTL*Hd);}
    // embed_vision: parameterless RMSNorm → projection
    bf* pn=MAL((long)OUTL*Hd);k_rms<<<OUTL,256>>>(pooled,nullptr,pn,OUTL,Hd,EPS);
    k_matmul<<<G2(TXT,OUTL),B2>>>(pn,W.embed_proj,out_proj,OUTL,Hd,TXT);
    CK(cudaDeviceSynchronize());
    cudaFree(h);cudaFree(hn);cudaFree(xc);cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(attn);
    cudaFree(gate);cudaFree(up);cudaFree(act);cudaFree(tmp);cudaFree(scr);cudaFree(pf);cudaFree(pooled);cudaFree(pn);
}

// ── harness: load f32 npy, convert to bf16, run, compare ────────────────────
struct Npy{std::vector<int64_t> shape;char kind=0;int isz=0;std::vector<uint8_t> data;int64_t numel()const{int64_t n=1;for(auto d:shape)n*=d;return n;}};
Npy load_npy(const std::string& p){std::ifstream f(p,std::ios::binary);if(!f){std::fprintf(stderr,"open %s\n",p.c_str());std::exit(2);}
    char m[6];f.read(m,6);uint8_t maj=f.get(),mn=f.get();(void)mn;uint32_t hl;if(maj==1){uint16_t h;f.read((char*)&h,2);hl=h;}else f.read((char*)&hl,4);
    std::string hdr(hl,0);f.read(hdr.data(),hl);Npy o;auto dp=hdr.find("'descr'");auto q=hdr.find('\'',hdr.find(':',dp)+1);
    std::string d=hdr.substr(q+1,hdr.find('\'',q+1)-q-1);o.kind=d[1];o.isz=std::atoi(d.substr(2).c_str());
    auto sp=hdr.find("'shape'");auto lp=hdr.find('(',sp),rp=hdr.find(')',lp);std::string sh=hdr.substr(lp+1,rp-lp-1);size_t i=0;
    while(i<sh.size()){while(i<sh.size()&&!isdigit(sh[i]))++i;if(i>=sh.size())break;int64_t v=0;while(i<sh.size()&&isdigit(sh[i]))v=v*10+(sh[i++]-'0');o.shape.push_back(v);}
    o.data.resize((size_t)o.numel()*o.isz);f.read((char*)o.data.data(),(std::streamsize)o.data.size());return o;}
std::string DIR;std::map<std::string,bf*> cache;
void ckpt(const char* tag,const bf* d,long n){std::vector<bf> y(n);CK(cudaMemcpy(y.data(),d,n*sizeof(bf),cudaMemcpyDeviceToHost));
    Npy r=load_npy(g_dir+"/"+tag+"_f32.npy");const float* rp=(const float*)r.data.data();double ma=0,sq=0;
    for(long i=0;i<n;i++){float v=__bfloat162float(y[i]);ma=std::max(ma,std::abs((double)v-rp[i]));sq+=(double)v*v;}
    std::printf("  ckpt %-20s max_abs=%.3e rms=%.3f\n",tag,ma,std::sqrt(sq/n));}
bf* Wbf(const std::string& name){auto it=cache.find(name);if(it!=cache.end())return it->second;
    Npy n=load_npy(DIR+"/weights/"+name+".npy");std::vector<bf> hb(n.numel());const float* fp=(const float*)n.data.data();
    for(int64_t i=0;i<n.numel();i++)hb[i]=__float2bfloat16(fp[i]);bf* d;CK(cudaMalloc(&d,hb.size()*sizeof(bf)));
    CK(cudaMemcpy(d,hb.data(),hb.size()*sizeof(bf),cudaMemcpyHostToDevice));cache[name]=d;return d;}
float scal(const std::string& n){Npy x=load_npy(DIR+"/weights/"+n+".npy");return ((float*)x.data.data())[0];}
Clip clip(const std::string& b){return {Wbf(b+".linear.weight"),scal(b+".input_min"),scal(b+".input_max"),scal(b+".output_min"),scal(b+".output_max")};}
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

    VisWeights W;W.patch_w=Wbf("vision.patch_embedder.input_proj.weight");
    W.pos_table=Wbf("vision.patch_embedder.position_embedding_table");W.embed_proj=Wbf("embed.embedding_projection.weight");
    for(int l=0;l<16;l++){std::string p="vision.encoder.layers."+std::to_string(l)+".";VisLayer L;
        L.in_ln=Wbf(p+"input_layernorm.weight");L.post_attn_ln=Wbf(p+"post_attention_layernorm.weight");
        L.pre_ff_ln=Wbf(p+"pre_feedforward_layernorm.weight");L.post_ff_ln=Wbf(p+"post_feedforward_layernorm.weight");
        L.q_norm=Wbf(p+"self_attn.q_norm.weight");L.k_norm=Wbf(p+"self_attn.k_norm.weight");
        L.q=clip(p+"self_attn.q_proj");L.k=clip(p+"self_attn.k_proj");L.v=clip(p+"self_attn.v_proj");L.o=clip(p+"self_attn.o_proj");
        L.gate=clip(p+"mlp.gate_proj");L.up=clip(p+"mlp.up_proj");L.down=clip(p+"mlp.down_proj");W.layers.push_back(L);}

    bf* d_out;CK(cudaMalloc(&d_out,(long)OUTL*TXT*sizeof(bf)));
    g_dir=DIR; g_dbg=!real;  // intermediate checkpoints only exist for the synthetic input
    run_gemma4_vision(W,d_pix,d_pos,d_grp,N,OUTL,d_out);
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
