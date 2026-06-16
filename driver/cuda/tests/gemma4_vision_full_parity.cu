// Full Gemma-4 vision encoder forward (fp32, standalone) vs the reference dump.
//
// Reproduces Gemma4VisionModel + Gemma4MultimodalEmbedder end-to-end and checks
// against scripts/gemma4_vision_parity_ref.py's fp32 dumps. Intermediate
// checkpoints (layer0, layer_last, pooled) localize bugs. Correctness over speed
// (naive kernels). Compile:
//   nvcc -O2 -arch=sm_89 -std=c++17 gemma4_vision_full_parity.cu -o /tmp/vfull
//   /tmp/vfull /tmp/gemma4_vision_parity
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

namespace {
#define CK(x) do { cudaError_t e=(x); if(e){std::fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);} } while(0)

struct Npy { std::vector<int64_t> shape; char kind=0; int isz=0; std::vector<uint8_t> data;
    int64_t numel() const { int64_t n=1; for(auto d:shape)n*=d; return n; } };
Npy load_npy(const std::string& p){
    std::ifstream f(p,std::ios::binary); if(!f){std::fprintf(stderr,"open %s\n",p.c_str());std::exit(2);}
    char m[6]; f.read(m,6); uint8_t maj=f.get(),min=f.get(); (void)min; uint32_t hl;
    if(maj==1){uint16_t h;f.read((char*)&h,2);hl=h;} else {f.read((char*)&hl,4);}
    std::string hdr(hl,0); f.read(hdr.data(),hl); Npy o;
    auto dp=hdr.find("'descr'"); auto q=hdr.find('\'',hdr.find(':',dp)+1);
    std::string d=hdr.substr(q+1,hdr.find('\'',q+1)-q-1); o.kind=d[1]; o.isz=std::atoi(d.substr(2).c_str());
    auto sp=hdr.find("'shape'"); auto lp=hdr.find('(',sp),rp=hdr.find(')',lp); std::string sh=hdr.substr(lp+1,rp-lp-1);
    size_t i=0; while(i<sh.size()){ while(i<sh.size()&&!isdigit(sh[i]))++i; if(i>=sh.size())break;
        int64_t v=0; while(i<sh.size()&&isdigit(sh[i]))v=v*10+(sh[i++]-'0'); o.shape.push_back(v);}
    o.data.resize((size_t)o.numel()*o.isz); f.read((char*)o.data.data(),(std::streamsize)o.data.size()); return o; }

std::string DIR;
std::map<std::string,float*> cache;
// Load weights/<name>.npy to device (cached). float32 expected.
float* W(const std::string& name, int64_t* numel=nullptr){
    auto it=cache.find(name); if(it!=cache.end()){ if(numel){auto n=load_npy(DIR+"/weights/"+name+".npy"); *numel=n.numel();} return it->second; }
    Npy n=load_npy(DIR+"/weights/"+name+".npy"); float* d; CK(cudaMalloc(&d,n.data.size()));
    CK(cudaMemcpy(d,n.data.data(),n.data.size(),cudaMemcpyHostToDevice)); cache[name]=d; if(numel)*numel=n.numel(); return d; }
float scalarW(const std::string& name){ Npy n=load_npy(DIR+"/weights/"+name+".npy"); return ((float*)n.data.data())[0]; }

// ── kernels ────────────────────────────────────────────────────────────────
__global__ void k_scale(const float* p,float* o,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<t)o[i]=2.f*(p[i]-0.5f);}
__global__ void k_matmul(const float* x,const float* Wt,float* y,int N,int K,int O){ // y[n,o]=sum_k x[n,k]*Wt[o,k]
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N||o>=O)return;
    const float* xr=x+(long)n*K; const float* wr=Wt+(long)o*K; float a=0; for(int k=0;k<K;k++)a+=xr[k]*wr[k]; y[(long)n*O+o]=a; }
__global__ void k_addpos(float* y,const float* tb,const float* pos,int N,int O,int P){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N||o>=O)return;
    long x=(long)llrintf(pos[2L*n]),yy=(long)llrintf(pos[2L*n+1]); if(x<0)x=0; if(yy<0)yy=0;
    y[(long)n*O+o]+=tb[(0L*P+x)*O+o]+tb[(1L*P+yy)*O+o]; }
__global__ void k_clamp(const float* x,float* o,float lo,float hi,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<t){float v=x[i]; o[i]=v<lo?lo:(v>hi?hi:v);}}
// RMSNorm over last D dims, rows = R. weight optional. one block per row.
__global__ void k_rms(const float* x,const float* w,float* o,int R,int D,float eps){
    int r=blockIdx.x; if(r>=R)return; const float* xr=x+(long)r*D; float* orow=o+(long)r*D;
    __shared__ float ss; float loc=0; for(int d=threadIdx.x;d<D;d+=blockDim.x)loc+=xr[d]*xr[d];
    for(int s=warpSize/2;s>0;s>>=1)loc+=__shfl_down_sync(0xffffffff,loc,s);
    __shared__ float warp[32]; if((threadIdx.x&31)==0)warp[threadIdx.x>>5]=loc; __syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=warp[i]; ss=rsqrtf(t/D+eps);} __syncthreads();
    float inv=ss; for(int d=threadIdx.x;d<D;d+=blockDim.x)orow[d]=xr[d]*inv*(w?w[d]:1.f); }
// 2D RoPE on [N, H, 64]; thread per (n,head,c) c in 0..16. invf[c]=theta^(-c/16).
__global__ void k_rope(float* q,const float* pos,int N,int H,float theta){
    int n=blockIdx.z, head=blockIdx.y, c=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N||head>=H||c>=16)return;
    float* v=q+(((long)n*H+head)*64); float px=pos[2L*n],py=pos[2L*n+1];
    float invf=powf(theta,-(float)c/16.f); float ax=px*invf, ay=py*invf;
    float cx=cosf(ax),sx=sinf(ax),cy=cosf(ay),sy=sinf(ay);
    float a=v[c],b=v[c+16]; v[c]=a*cx-b*sx; v[c+16]=b*cx+a*sx;
    float e=v[32+c],f=v[48+c]; v[32+c]=e*cy-f*sy; v[48+c]=f*cy+e*sy; }
// per-head scores[N,N] = qh@kh^T * scale, qh = q[n, head*64 + :]
__global__ void k_qk(const float* q,const float* k,float* s,int N,int H,int head,float scale){
    int i=blockIdx.y*blockDim.y+threadIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N||j>=N)return;
    const float* qi=q+((long)i*H+head)*64; const float* kj=k+((long)j*H+head)*64;
    float a=0; for(int d=0;d<64;d++)a+=qi[d]*kj[d]; s[(long)i*N+j]=a*scale; }
__global__ void k_softmax(float* s,int N){ int i=blockIdx.x; if(i>=N)return; float* r=s+(long)i*N;
    float mx=-1e30f; for(int j=threadIdx.x;j<N;j+=blockDim.x)mx=fmaxf(mx,r[j]);
    for(int o=warpSize/2;o>0;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    __shared__ float smx,ssum; __shared__ float wm[32],wsv[32]; if((threadIdx.x&31)==0)wm[threadIdx.x>>5]=mx; __syncthreads();
    if(threadIdx.x==0){float m=-1e30f;int nw=(blockDim.x+31)/32;for(int i2=0;i2<nw;i2++)m=fmaxf(m,wm[i2]);smx=m;} __syncthreads();
    float sm=0; for(int j=threadIdx.x;j<N;j+=blockDim.x){float e=__expf(r[j]-smx);r[j]=e;sm+=e;}
    for(int o=warpSize/2;o>0;o>>=1)sm+=__shfl_down_sync(0xffffffff,sm,o); if((threadIdx.x&31)==0)wsv[threadIdx.x>>5]=sm; __syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i2=0;i2<nw;i2++)t+=wsv[i2];ssum=t;} __syncthreads();
    float inv=1.f/ssum; for(int j=threadIdx.x;j<N;j+=blockDim.x)r[j]*=inv; }
// out[n, head*64+d] = sum_j s[n,j]*v[j,head*64+d]
__global__ void k_av(const float* s,const float* v,float* o,int N,int H,int head){
    int n=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N||d>=64)return;
    const float* sr=s+(long)n*N; float a=0; for(int j=0;j<N;j++)a+=sr[j]*v[((long)j*H+head)*64+d];
    o[((long)n*H+head)*64+d]=a; }
__global__ void k_gelu_mul(const float* g,const float* u,float* o,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<t){float x=g[i];float gl=0.5f*x*(1.f+tanhf(0.7978845608f*(x+0.044715f*x*x*x)));o[i]=gl*u[i];}}
__global__ void k_add(float* h,const float* x,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<t)h[i]+=x[i];}
// avg-pool: out[group[n], d] += h[n,d]/k2 ; group precomputed on host
__global__ void k_pool(const float* h,const int* grp,float* o,int N,int D,float k2){
    int n=blockIdx.y*blockDim.y+threadIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x; if(n>=N||d>=D)return;
    atomicAdd(&o[(long)grp[n]*D+d], h[(long)n*D+d]/k2); }
__global__ void k_scale_const(float* x,float s,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<t)x[i]*=s;}

double cmp(const char* tag,const float* dy,const std::string& reff,long n){
    std::vector<float> y(n); CK(cudaMemcpy(y.data(),dy,n*4,cudaMemcpyDeviceToHost));
    Npy r=load_npy(DIR+"/"+reff); const float* rp=(const float*)r.data.data();
    double ma=0,sq=0; for(long i=0;i<n;i++){double a=std::abs((double)y[i]-rp[i]);ma=std::max(ma,a);sq+=(double)y[i]*y[i];}
    std::printf("  %-16s max_abs=%.3e  rms=%.3f\n",tag,ma,std::sqrt(sq/n)); return ma; }

dim3 B2(16,16);
inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}
}

int main(int argc,char**argv){
    DIR=argc>1?argv[1]:"/tmp/gemma4_vision_parity";
    const int N=2520,Hd=768,NH=12,HD=64,IM=3072,L=16,OUTL=280,TXT=2560,P=10240; const float EPS=1e-6f,THETA=100.f;
    auto pixn=load_npy(DIR+"/input_pixel_values_f32.npy"); auto posn=load_npy(DIR+"/input_position_ids.npy");
    float *d_pix,*d_pos; CK(cudaMalloc(&d_pix,(long)N*Hd*4)); CK(cudaMalloc(&d_pos,(long)N*2*4));
    CK(cudaMemcpy(d_pix,pixn.data.data(),(long)N*Hd*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos,posn.data.data(),(long)N*2*4,cudaMemcpyHostToDevice));
    // host positions → pool group ids
    const float* hp=(const float*)posn.data.data(); int maxx=0; for(int n=0;n<N;n++)maxx=std::max(maxx,(int)llrintf(hp[2*n]));
    int gx=(maxx+1)/3; std::vector<int> grp(N); for(int n=0;n<N;n++){int x=(int)llrintf(hp[2*n]),y=(int)llrintf(hp[2*n+1]); grp[n]=(x/3)+gx*(y/3);}
    int* d_grp; CK(cudaMalloc(&d_grp,N*4)); CK(cudaMemcpy(d_grp,grp.data(),N*4,cudaMemcpyHostToDevice));

    auto MAL=[&](long n){float* d;CK(cudaMalloc(&d,n*4));return d;};
    float *h=MAL((long)N*Hd),*hn=MAL((long)N*Hd),*xc=MAL((long)N*IM),*q=MAL((long)N*Hd),*k=MAL((long)N*Hd),*v=MAL((long)N*Hd),
          *attn=MAL((long)N*Hd),*scr=MAL((long)N*N),*gate=MAL((long)N*IM),*up=MAL((long)N*IM),*act=MAL((long)N*IM),*tmp=MAL((long)N*Hd);

    // ── patch embed ──
    k_scale<<<((long)N*Hd+255)/256,256>>>(d_pix,hn,(long)N*Hd);
    k_matmul<<<G2(Hd,N),B2>>>(hn,W("vision.patch_embedder.input_proj.weight"),h,N,Hd,Hd);
    k_addpos<<<G2(Hd,N),B2>>>(h,W("vision.patch_embedder.position_embedding_table"),d_pos,N,Hd,P);
    CK(cudaDeviceSynchronize()); cmp("patch_embed",h,"patch_embed_f32.npy",(long)N*Hd);

    auto clin=[&](float* x,float* out,const std::string& base,int Kin,int Out){
        float imn=scalarW(base+".input_min"),imx=scalarW(base+".input_max"),omn=scalarW(base+".output_min"),omx=scalarW(base+".output_max");
        k_clamp<<<((long)N*Kin+255)/256,256>>>(x,xc,imn,imx,(long)N*Kin);
        k_matmul<<<G2(Out,N),B2>>>(xc,W(base+".linear.weight"),out,N,Kin,Out);
        k_clamp<<<((long)N*Out+255)/256,256>>>(out,out,omn,omx,(long)N*Out); };

    for(int l=0;l<L;l++){ std::string p="vision.encoder.layers."+std::to_string(l)+".";
        // attention block
        k_rms<<<N,256>>>(h,W(p+"input_layernorm.weight"),hn,N,Hd,EPS);
        clin(hn,q,p+"self_attn.q_proj",Hd,Hd); clin(hn,k,p+"self_attn.k_proj",Hd,Hd); clin(hn,v,p+"self_attn.v_proj",Hd,Hd);
        k_rms<<<N*NH,64>>>(q,W(p+"self_attn.q_norm.weight"),q,N*NH,HD,EPS);
        k_rms<<<N*NH,64>>>(k,W(p+"self_attn.k_norm.weight"),k,N*NH,HD,EPS);
        k_rms<<<N*NH,64>>>(v,nullptr,v,N*NH,HD,EPS);
        dim3 rg(1,NH,N); k_rope<<<rg,32>>>(q,d_pos,N,NH,THETA); k_rope<<<rg,32>>>(k,d_pos,N,NH,THETA);
        for(int hh=0;hh<NH;hh++){ k_qk<<<G2(N,N),B2>>>(q,k,scr,N,NH,hh,1.0f); k_softmax<<<N,256>>>(scr,N); k_av<<<G2(64,N),B2>>>(scr,v,attn,N,NH,hh);}
        clin(attn,tmp,p+"self_attn.o_proj",Hd,Hd);
        k_rms<<<N,256>>>(tmp,W(p+"post_attention_layernorm.weight"),tmp,N,Hd,EPS);
        k_add<<<((long)N*Hd+255)/256,256>>>(h,tmp,(long)N*Hd);
        // mlp block
        k_rms<<<N,256>>>(h,W(p+"pre_feedforward_layernorm.weight"),hn,N,Hd,EPS);
        clin(hn,gate,p+"mlp.gate_proj",Hd,IM); clin(hn,up,p+"mlp.up_proj",Hd,IM);
        k_gelu_mul<<<((long)N*IM+255)/256,256>>>(gate,up,act,(long)N*IM);
        clin(act,tmp,p+"mlp.down_proj",IM,Hd);
        k_rms<<<N,256>>>(tmp,W(p+"post_feedforward_layernorm.weight"),tmp,N,Hd,EPS);
        k_add<<<((long)N*Hd+255)/256,256>>>(h,tmp,(long)N*Hd);
        if(l==0){CK(cudaDeviceSynchronize()); cmp("layer0",h,"layer0_f32.npy",(long)N*Hd);} }
    CK(cudaDeviceSynchronize()); cmp("layer_last",h,"layer_last_f32.npy",(long)N*Hd);

    // ── pooler: avg-pool to OUTL, × sqrt(hidden) ──
    float* pooled=MAL((long)OUTL*Hd); CK(cudaMemset(pooled,0,(long)OUTL*Hd*4));
    k_pool<<<G2(Hd,N),B2>>>(h,d_grp,pooled,N,Hd,9.f);
    k_scale_const<<<((long)OUTL*Hd+255)/256,256>>>(pooled,sqrtf((float)Hd),(long)OUTL*Hd);
    CK(cudaDeviceSynchronize()); cmp("pooled",pooled,"pooled_last_hidden_f32.npy",(long)OUTL*Hd);

    // ── embed_vision: parameterless RMSNorm → projection ──
    float* pn=MAL((long)OUTL*Hd); k_rms<<<OUTL,256>>>(pooled,nullptr,pn,OUTL,Hd,EPS);
    float* proj=MAL((long)OUTL*TXT); k_matmul<<<G2(TXT,OUTL),B2>>>(pn,W("embed.embedding_projection.weight"),proj,OUTL,Hd,TXT);
    CK(cudaDeviceSynchronize()); double e=cmp("projected",proj,"projected_f32.npy",(long)OUTL*TXT);
    std::printf("%s\n", e<2e-2 ? "PARITY PASS (projected)" : "PARITY FAIL (projected)");
    return e<2e-2?0:1;
}
