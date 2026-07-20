// Gemma-4 vision encoder forward (bf16). See gemma4_vision_forward.hpp.
// Kernels ported verbatim from the parity-verified standalone
// (driver/cuda/tests/gemma4_vision_full_parity_bf16.cu): rel_rms 1.07%,
// cosine 0.99994 vs HF-bf16. bf16 storage + fp32 compute, matching the driver.
//
// First-cut: naive kernels + cudaMalloc scratch (correctness over speed); a
// cuBLAS/workspace pass is a follow-up. CUDA-only includes (no model/loader
// headers) so nvcc never sees the toml++ config headers.

#include "model/gemma4/gemma4_vision_forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <math_constants.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_cuda_driver::model {
namespace {

typedef __nv_bfloat16 bf;
#define VCK(x) do{cudaError_t e=(x);if(e)throw std::runtime_error(std::string("gemma4_vision: ")+cudaGetErrorString(e));}while(0)
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
        VCK(cudaMalloc(&pointer, count * sizeof(T)));
        allocations_.push_back(pointer);
        return pointer;
    }

private:
    std::vector<void*> allocations_;
};

__global__ void k_scale(const bf* p,bf* o,long t){long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i<t)o[i]=Bf(2.f*(F(p[i])-0.5f));}
__global__ void k_matmul(const bf* x,const bf* W,bf* y,int N,int K,int O){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||o>=O)return;
    const bf* xr=x+(long)n*K;const bf* wr=W+(long)o*K;float a=0;for(int k=0;k<K;k++)a+=F(xr[k])*F(wr[k]);y[(long)n*O+o]=Bf(a);}
__global__ void k_addpos(bf* y,const bf* tb,const float* pos,int N,int O,int P){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N||o>=O)return;
    long x=(long)llrintf(pos[2L*n]),yy=(long)llrintf(pos[2L*n+1]);if(x<0)x=0;if(yy<0)yy=0;
    y[(long)n*O+o]=Bf(F(y[(long)n*O+o])+F(tb[(0L*P+x)*O+o])+F(tb[(1L*P+yy)*O+o]));}
__global__ void k_clamp(const bf* x,bf* o,const bf* lo,const bf* hi,long t){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i>=t)return;
    float v=F(x[i]),l=lo?F(*lo):-CUDART_INF_F,h=hi?F(*hi):CUDART_INF_F;o[i]=Bf(v<l?l:(v>h?h:v));}
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

}  // namespace

void run_gemma4_vision(const VisRawWeights& w,
                       const bf* pixel,const float* pos,const int* grp,
                       int N,int OUTL,bf* out_proj,cudaStream_t S){
    const int Hd=w.hidden, NH=w.heads, IM=w.intermediate, TXT=w.text_hidden, PT=w.pos_table_size;
    const float EPS=w.eps, THETA=w.theta;
    if(Hd!=768||NH!=12) throw std::runtime_error("gemma4_vision: unexpected dims (expected hidden=768, heads=12)");

    DeviceScratch scratch;
    auto MAL=[&](long n){return scratch.alloc<bf>(n);};
    bf *h=MAL((long)N*Hd),*hn=MAL((long)N*Hd),*xc=MAL((long)N*IM),*q=MAL((long)N*Hd),*k=MAL((long)N*Hd),*v=MAL((long)N*Hd),
       *attn=MAL((long)N*Hd),*gate=MAL((long)N*IM),*up=MAL((long)N*IM),*act=MAL((long)N*IM),*tmp=MAL((long)N*Hd);
    float* scr=scratch.alloc<float>((long)N*N);
    auto clin=[&](const bf* x,bf* out,const VisClipRaw& c,int Kin,int Out){
        k_clamp<<<((long)N*Kin+255)/256,256,0,S>>>(x,xc,c.imin,c.imax,(long)N*Kin);
        k_matmul<<<G2(Out,N),B2,0,S>>>(xc,c.w,out,N,Kin,Out);
        k_clamp<<<((long)N*Out+255)/256,256,0,S>>>(out,out,c.omin,c.omax,(long)N*Out);};

    k_scale<<<((long)N*Hd+255)/256,256,0,S>>>(pixel,hn,(long)N*Hd);
    k_matmul<<<G2(Hd,N),B2,0,S>>>(hn,w.patch_w,h,N,Hd,Hd);
    k_addpos<<<G2(Hd,N),B2,0,S>>>(h,w.pos_table,pos,N,Hd,PT);
    for(const auto& L:w.layers){
        k_rms<<<N,256,0,S>>>(h,L.in_ln,hn,N,Hd,EPS);
        clin(hn,q,L.q,Hd,Hd);clin(hn,k,L.k,Hd,Hd);clin(hn,v,L.v,Hd,Hd);
        k_rms<<<N*NH,64,0,S>>>(q,L.q_norm,q,N*NH,64,EPS);k_rms<<<N*NH,64,0,S>>>(k,L.k_norm,k,N*NH,64,EPS);k_rms<<<N*NH,64,0,S>>>(v,nullptr,v,N*NH,64,EPS);
        dim3 rg(1,NH,N);k_rope<<<rg,32,0,S>>>(q,pos,N,NH,THETA);k_rope<<<rg,32,0,S>>>(k,pos,N,NH,THETA);
        for(int hh=0;hh<NH;hh++){k_qk<<<G2(N,N),B2,0,S>>>(q,k,scr,N,NH,hh,1.0f);k_softmax<<<N,256,0,S>>>(scr,N);k_av<<<G2(64,N),B2,0,S>>>(scr,v,attn,N,NH,hh);}
        clin(attn,tmp,L.o,Hd,Hd);
        k_rms<<<N,256,0,S>>>(tmp,L.post_attn_ln,tmp,N,Hd,EPS);
        k_add<<<((long)N*Hd+255)/256,256,0,S>>>(h,tmp,(long)N*Hd);
        k_rms<<<N,256,0,S>>>(h,L.pre_ff_ln,hn,N,Hd,EPS);
        clin(hn,gate,L.gate,Hd,IM);clin(hn,up,L.up,Hd,IM);
        k_gelu_mul<<<((long)N*IM+255)/256,256,0,S>>>(gate,up,act,(long)N*IM);
        clin(act,tmp,L.down,IM,Hd);
        k_rms<<<N,256,0,S>>>(tmp,L.post_ff_ln,tmp,N,Hd,EPS);
        k_add<<<((long)N*Hd+255)/256,256,0,S>>>(h,tmp,(long)N*Hd);
    }
    float* pf=scratch.alloc<float>((long)OUTL*Hd);VCK(cudaMemsetAsync(pf,0,(long)OUTL*Hd*4,S));
    k_pool<<<G2(Hd,N),B2,0,S>>>(h,grp,pf,N,Hd,9.f);
    bf* pooled=MAL((long)OUTL*Hd);k_pool_finish<<<((long)OUTL*Hd+255)/256,256,0,S>>>(pf,pooled,sqrtf((float)Hd),(long)OUTL*Hd);
    bf* pn=MAL((long)OUTL*Hd);k_rms<<<OUTL,256,0,S>>>(pooled,nullptr,pn,OUTL,Hd,EPS);
    k_matmul<<<G2(TXT,OUTL),B2,0,S>>>(pn,w.embed_proj,out_proj,OUTL,Hd,TXT);
    VCK(cudaStreamSynchronize(S));
}

namespace {
__global__ void k_f32_to_bf16(const float* a, bf* o, long n){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x; if(i<n) o[i]=Bf(a[i]);
}
}  // namespace

void scatter_gemma4_vision(const Gemma4VisionInputs& vin, bf* hidden,
                           int /*n_rows*/, int text_hidden, cudaStream_t S){
    if(vin.weights==nullptr || vin.num_images<=0) return;
    const VisRawWeights& w=*vin.weights;
    const int patch_dim = 3*16*16;            // 768 (Gemma patch 16, RGB)
    const int pk2 = w.pool_kernel*w.pool_kernel;
    long patch_off = 0;
    for(int im=0; im<vin.num_images; ++im){
        DeviceScratch scratch;
        const long blo=vin.pixel_byte_indptr_h[im], bhi=vin.pixel_byte_indptr_h[im+1];
        const int n_floats=(int)((bhi-blo)/4);
        const int n_patch=n_floats/patch_dim;
        if(n_patch<=0) continue;
        const int out_len=n_patch/pk2;
        const float* pix_h=vin.pixels_h + blo/4;
        const std::uint32_t* pos_h=vin.patch_positions_h + patch_off*2;
        const std::uint32_t anchor=vin.anchor_rows_h[im];

        // pixels f32 (host) → device → bf16
        float* pix_f32_d=scratch.alloc<float>(n_floats);
        VCK(cudaMemcpyAsync(pix_f32_d,pix_h,(long)n_floats*4,cudaMemcpyHostToDevice,S));
        bf* pix_bf_d=scratch.alloc<bf>(n_floats);
        k_f32_to_bf16<<<(n_floats+255)/256,256,0,S>>>(pix_f32_d,pix_bf_d,n_floats);

        // positions u32 (host) → f32 device; pool groups (host) → int device
        std::vector<float> posf(n_patch*2);
        std::vector<int> grp(n_patch);
        int maxx=0; for(int p=0;p<n_patch;++p) maxx=std::max(maxx,(int)pos_h[2*p]);
        const int gx=(maxx+1)/w.pool_kernel;
        for(int p=0;p<n_patch;++p){
            posf[2*p]=(float)pos_h[2*p]; posf[2*p+1]=(float)pos_h[2*p+1];
            grp[p]=((int)pos_h[2*p]/w.pool_kernel) + gx*((int)pos_h[2*p+1]/w.pool_kernel);
        }
        float* pos_d=scratch.alloc<float>((long)n_patch*2);
        VCK(cudaMemcpyAsync(pos_d,posf.data(),(long)n_patch*2*4,cudaMemcpyHostToDevice,S));
        int* grp_d=scratch.alloc<int>(n_patch);
        VCK(cudaMemcpyAsync(grp_d,grp.data(),(long)n_patch*4,cudaMemcpyHostToDevice,S));

        // encode → projected [out_len, text_hidden] → overwrite the anchor rows.
        bf* proj_d=scratch.alloc<bf>((long)out_len*text_hidden);
        run_gemma4_vision(w, pix_bf_d, pos_d, grp_d, n_patch, out_len, proj_d, S);
        VCK(cudaMemcpyAsync(hidden + (long)anchor*text_hidden, proj_d,
                            (long)out_len*text_hidden*sizeof(bf),
                            cudaMemcpyDeviceToDevice, S));
        VCK(cudaStreamSynchronize(S));
        patch_off += n_patch;
    }
}

void encode_gemma4_vision(const Gemma4VisionInputs& vin,
                          std::uint16_t* output_rows_h,
                          std::size_t output_bytes,
                          std::uint32_t* output_row_indptr_h,
                          cudaStream_t S) {
    if (vin.weights == nullptr || vin.num_images <= 0 ||
        output_rows_h == nullptr || output_row_indptr_h == nullptr) {
        throw std::runtime_error("gemma4_vision: invalid standalone encode inputs");
    }
    const VisRawWeights& w = *vin.weights;
    const int patch_dim = 3 * 16 * 16;
    const int pk2 = w.pool_kernel * w.pool_kernel;
    const std::size_t row_bytes =
        static_cast<std::size_t>(w.text_hidden) * sizeof(bf);
    std::size_t output_rows = 0;
    long patch_off = 0;
    output_row_indptr_h[0] = 0;
    for (int im = 0; im < vin.num_images; ++im) {
        DeviceScratch scratch;
        const long blo = vin.pixel_byte_indptr_h[im];
        const long bhi = vin.pixel_byte_indptr_h[im + 1];
        const int n_floats = static_cast<int>((bhi - blo) / 4);
        const int n_patch = n_floats / patch_dim;
        if (n_patch <= 0 || n_patch % pk2 != 0) {
            throw std::runtime_error("gemma4_vision: invalid patch count");
        }
        const int out_len = n_patch / pk2;
        if ((output_rows + static_cast<std::size_t>(out_len)) * row_bytes >
            output_bytes) {
            throw std::runtime_error("gemma4_vision: encode output buffer too small");
        }
        const float* pix_h = vin.pixels_h + blo / 4;
        const std::uint32_t* pos_h = vin.patch_positions_h + patch_off * 2;

        float* pix_f32_d = scratch.alloc<float>(n_floats);
        VCK(cudaMemcpyAsync(pix_f32_d, pix_h,
                            static_cast<long>(n_floats) * 4,
                            cudaMemcpyHostToDevice, S));
        bf* pix_bf_d = scratch.alloc<bf>(n_floats);
        k_f32_to_bf16<<<(n_floats + 255) / 256, 256, 0, S>>>(
            pix_f32_d, pix_bf_d, n_floats);

        std::vector<float> posf(n_patch * 2);
        std::vector<int> grp(n_patch);
        int maxx = 0;
        for (int p = 0; p < n_patch; ++p) {
            maxx = std::max(maxx, static_cast<int>(pos_h[2 * p]));
        }
        const int gx = (maxx + 1) / w.pool_kernel;
        for (int p = 0; p < n_patch; ++p) {
            posf[2 * p] = static_cast<float>(pos_h[2 * p]);
            posf[2 * p + 1] = static_cast<float>(pos_h[2 * p + 1]);
            grp[p] = static_cast<int>(pos_h[2 * p]) / w.pool_kernel +
                     gx * (static_cast<int>(pos_h[2 * p + 1]) /
                           w.pool_kernel);
        }
        float* pos_d = scratch.alloc<float>(
            static_cast<long>(n_patch) * 2);
        VCK(cudaMemcpyAsync(pos_d, posf.data(),
                            static_cast<long>(n_patch) * 2 * 4,
                            cudaMemcpyHostToDevice, S));
        int* grp_d = scratch.alloc<int>(n_patch);
        VCK(cudaMemcpyAsync(grp_d, grp.data(),
                            static_cast<long>(n_patch) * 4,
                            cudaMemcpyHostToDevice, S));
        bf* proj_d = scratch.alloc<bf>(
            static_cast<long>(out_len) * w.text_hidden);
        run_gemma4_vision(
            w, pix_bf_d, pos_d, grp_d, n_patch, out_len, proj_d, S);
        VCK(cudaMemcpyAsync(
            output_rows_h + output_rows * w.text_hidden, proj_d,
            static_cast<long>(out_len) * w.text_hidden * sizeof(bf),
            cudaMemcpyDeviceToHost, S));
        VCK(cudaStreamSynchronize(S));
        output_rows += static_cast<std::size_t>(out_len);
        output_row_indptr_h[im + 1] =
            static_cast<std::uint32_t>(output_rows);
        patch_off += n_patch;
    }
}

}  // namespace pie_cuda_driver::model
