// Standalone prototype + unit test for the WS4 matrix kernel templates and the
// greedy speculative-verify DAG (Lane L2 / charlie, Phase 2). Compiles
// NVRTC-safe sources (built on the W1 primitive prelude) for the device arch,
// launches via the driver API, and validates against a CPU reference.
//
// Two things proved here, ahead of wiring matrix into the codegen emitter:
//   1. Batched per-row reduce / softmax / argmax over a Matrix{rows, len} — the
//      W1 block primitives are already row-local, so the matrix wrapper is just
//      "grid = rows, block r handles buf + r*len". This validates that.
//   2. The greedy spec-verify kernel DAG: argmax (per draft row) → eq vs draft
//      tokens → cumprod (accept run) → reduce-sum (accept_len) → select (emit
//      the verified token prefix). Token-identical to a CPU greedy verifier.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "sampling_ir/primitives_src.hpp"

using namespace pie_cuda_driver::sampling_ir;

namespace {
int g_fail = 0, g_checks = 0;
int g_cc_major = 0, g_cc_minor = 0;

void expect(bool c, const char* what) {
    ++g_checks;
    if (!c) { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", what); }
}

#define CU_CHECK(e) do{ CUresult _e=(e); if(_e!=CUDA_SUCCESS){ const char* s=nullptr; \
  cuGetErrorString(_e,&s); std::fprintf(stderr,"CU err %s @%d: %s\n",s?s:"?",__LINE__,#e); std::exit(2);} }while(0)
#define RT_CHECK(e) do{ cudaError_t _e=(e); if(_e!=cudaSuccess){ \
  std::fprintf(stderr,"RT err %s @%d\n",cudaGetErrorString(_e),__LINE__); std::exit(2);} }while(0)
#define NV_CHECK(e) do{ nvrtcResult _e=(e); if(_e!=NVRTC_SUCCESS){ \
  std::fprintf(stderr,"NVRTC err %s @%d\n",nvrtcGetErrorString(_e),__LINE__); std::exit(2);} }while(0)

CUmodule compile(const std::string& body) {
    std::string src = std::string(primitive_prelude()) + "\n" + body;
    nvrtcProgram p;
    NV_CHECK(nvrtcCreateProgram(&p, src.c_str(), "m.cu", 0, nullptr, nullptr));
    char arch[64];
    std::snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", g_cc_major, g_cc_minor);
    const char* opts[] = {arch, "--std=c++17"};
    if (nvrtcCompileProgram(p, 2, opts) != NVRTC_SUCCESS) {
        size_t n=0; nvrtcGetProgramLogSize(p,&n); std::string log(n,'\0'); nvrtcGetProgramLog(p,log.data());
        std::fprintf(stderr, "compile failed:\n%s\n", log.c_str()); std::exit(2);
    }
    size_t sz=0; NV_CHECK(nvrtcGetPTXSize(p,&sz)); std::string ptx(sz,'\0'); NV_CHECK(nvrtcGetPTX(p,ptx.data()));
    NV_CHECK(nvrtcDestroyProgram(&p));
    CUmodule m; CU_CHECK(cuModuleLoadData(&m, ptx.c_str())); return m;
}
CUfunction fn_of(CUmodule m, const char* n){ CUfunction f; CU_CHECK(cuModuleGetFunction(&f,m,n)); return f; }
void launch(CUfunction f,int grid,int block,void** a){ CU_CHECK(cuLaunchKernel(f,grid,1,1,block,1,1,0,nullptr,a,nullptr)); CU_CHECK(cuCtxSynchronize()); }
template<class T> CUdeviceptr dup(const std::vector<T>& h){ CUdeviceptr d; CU_CHECK(cuMemAlloc(&d,h.size()*sizeof(T))); CU_CHECK(cuMemcpyHtoD(d,h.data(),h.size()*sizeof(T))); return d; }
template<class T> CUdeviceptr dalloc(size_t n){ CUdeviceptr d; CU_CHECK(cuMemAlloc(&d,n*sizeof(T))); CU_CHECK(cuMemsetD8(d,0,n*sizeof(T))); return d; }
template<class T> std::vector<T> down(CUdeviceptr d,size_t n){ std::vector<T> h(n); CU_CHECK(cuMemcpyDtoH(h.data(),d,n*sizeof(T))); return h; }

bool approx(float a,float b){ return std::fabs(a-b) <= 1e-4f + 1e-4f*std::fabs(b); }

// ---- 1. batched per-row reduce + softmax over Matrix{rows,len} ------------
const char* kMatrixKernels = R"TK(
// One block per row; block r reduces row (buf + r*len).
extern "C" __global__ void m_reduce(const float* in, int rows, int len,
                                    float* sum, float* mx, int* amax) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = in + (long long)r * len;
    float s  = pie_ir_block_sum(row, len);
    float mv = pie_ir_block_max(row, len);
    int   ai = pie_ir_block_argmax(row, len);
    if (threadIdx.x == 0) { sum[r]=s; mx[r]=mv; amax[r]=ai; }
}
// Per-row softmax: out[r,j] = exp(x-max)/sum.
extern "C" __global__ void m_softmax(const float* in, int rows, int len, float* out) {
    int r = blockIdx.x;
    if (r >= rows) return;
    const float* row = in + (long long)r * len;
    float* orow = out + (long long)r * len;
    float mv = pie_ir_block_max(row, len);
    __shared__ float ssum;
    float local = 0.f;
    for (int j=threadIdx.x;j<len;j+=256) local += expf(row[j]-mv);
    // block reduce the local partials via the shared helper pattern
    __shared__ float buf[256];
    buf[threadIdx.x]=local; __syncthreads();
    for(int o=128;o>0;o>>=1){ if(threadIdx.x<o) buf[threadIdx.x]+=buf[threadIdx.x+o]; __syncthreads(); }
    if(threadIdx.x==0) ssum=buf[0]; __syncthreads();
    for(int j=threadIdx.x;j<len;j+=256) orow[j]=expf(row[j]-mv)/ssum;
}
)TK";

void test_matrix() {
    std::printf("[matrix batched reduce/softmax]\n");
    const int rows = 5, len = 300;
    std::vector<float> in(rows*len);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> d(-4.f,4.f);
    for (auto& v: in) v = d(rng);
    for (int r=0;r<rows;++r) in[r*len + (r*37)%len] = 9.f;  // distinct per-row max

    CUmodule m = compile(kMatrixKernels);
    CUdeviceptr din=dup(in), dsum=dalloc<float>(rows), dmx=dalloc<float>(rows), damax=dalloc<int>(rows), dsm=dalloc<float>(rows*len);
    void* a1[]={&din,(void*)&rows,(void*)&len,&dsum,&dmx,&damax};
    launch(fn_of(m,"m_reduce"), rows, 256, a1);
    void* a2[]={&din,(void*)&rows,(void*)&len,&dsm};
    launch(fn_of(m,"m_softmax"), rows, 256, a2);

    auto hsum=down<float>(dsum,rows), hmx=down<float>(dmx,rows), hsm=down<float>(dsm,rows*len);
    auto hamax=down<int>(damax,rows);
    bool sok=true,mok=true,aok=true,smok=true;
    for(int r=0;r<rows;++r){
        double s=0; float mv=-INFINITY; int ai=0;
        for(int j=0;j<len;++j){ float v=in[r*len+j]; s+=v; if(v>mv){mv=v;ai=j;} }
        if(!approx(hsum[r],(float)s)) sok=false;
        if(hmx[r]!=mv) mok=false;
        if(hamax[r]!=ai) aok=false;
        double Z=0; for(int j=0;j<len;++j) Z+=std::exp(in[r*len+j]-mv);
        double smass=0; for(int j=0;j<len;++j) smass+=hsm[r*len+j];
        if(!approx((float)smass,1.0f)) smok=false;
    }
    expect(sok,"per-row sum");
    expect(mok,"per-row max");
    expect(aok,"per-row argmax");
    expect(smok,"per-row softmax sums to 1");
    cuMemFree(din);cuMemFree(dsum);cuMemFree(dmx);cuMemFree(damax);cuMemFree(dsm);
    cuModuleUnload(m);
}

// ---- 2. greedy speculative-verify DAG -------------------------------------
// Inputs: target_logits Matrix{k+1, V} (verify-block: k draft positions + 1
// bonus), draft_tokens Vector{k} (the proposed tokens). The verifier:
//   target_tok[i] = argmax(target_logits[i])            (per-row, i in [0,k])
//   accept[i]     = (target_tok[i] == draft_tok[i])     (i in [0,k))
//   run[i]        = cumprod(accept)[i]                  (1 while all-accepted)
//   accept_len    = sum(run)                            (= first reject index)
//   out_tok[i]    = (i <= accept_len) ? target_tok[i] : -1   (verified prefix +
//                   1 correction/bonus at accept_len; -1 sentinel after)
// out length is always k+1; the runtime emits the non-(-1) prefix.
const char* kSpecVerify = R"TK(
extern "C" __global__ void spec_argmax(const float* tlog, int kp1, int V, int* ttok) {
    int r = blockIdx.x;
    if (r >= kp1) return;
    int a = pie_ir_block_argmax(tlog + (long long)r*V, V);
    if (threadIdx.x==0) ttok[r]=a;
}
// Single block: compute accept run, accept_len, and emit the verified prefix.
extern "C" __global__ void spec_emit(const int* ttok, const int* dtok, int k,
                                     int* out_tok, int* out_len) {
    __shared__ int run[256];   // k <= 255 for MVP draft blocks
    int t = threadIdx.x;
    // accept[i] = ttok[i]==dtok[i] for i<k ; cumulative-AND via prefix.
    int acc = (t < k) ? (ttok[t]==dtok[t] ? 1 : 0) : 0;
    run[t] = acc; __syncthreads();
    // inclusive AND-scan (cumprod of 0/1) over [0,k).
    for (int o=1;o<k;o<<=1){ int v = (t>=o && t<k) ? run[t-o] : 1; __syncthreads();
        if(t<k) run[t] = run[t] * v; __syncthreads(); }
    // accept_len = sum(run[0..k)) = first reject index.
    __shared__ int alen;
    if (t==0){ int s=0; for(int i=0;i<k;++i) s+=run[i]; alen=s; }
    __syncthreads();
    // emit ttok[0..alen] (verified + 1 correction); -1 after. out has k+1 slots.
    if (t <= alen && t <= k) out_tok[t] = ttok[t];
    else if (t <= k)        out_tok[t] = -1;
    if (t==0) *out_len = alen + 1;
}
)TK";

void run_spec(int k, int V, const std::vector<float>& tlog, const std::vector<int>& dtok,
              std::vector<int>& out_tok, int& out_len) {
    int kp1 = k+1;
    CUmodule m = compile(kSpecVerify);
    CUdeviceptr dtl=dup(tlog), dtt=dalloc<int>(kp1), ddt=dup(dtok), dot=dalloc<int>(kp1), dol=dalloc<int>(1);
    void* a1[]={&dtl,(void*)&kp1,(void*)&V,&dtt};
    launch(fn_of(m,"spec_argmax"), kp1, 256, a1);
    void* a2[]={&dtt,&ddt,(void*)&k,&dot,&dol};
    launch(fn_of(m,"spec_emit"), 1, 256, a2);
    out_tok = down<int>(dot,kp1);
    out_len = down<int>(dol,1)[0];
    cuMemFree(dtl);cuMemFree(dtt);cuMemFree(ddt);cuMemFree(dot);cuMemFree(dol);
    cuModuleUnload(m);
}

// CPU greedy verifier reference.
void cpu_spec(int k, int V, const std::vector<float>& tlog, const std::vector<int>& dtok,
              std::vector<int>& out_tok, int& out_len) {
    std::vector<int> ttok(k+1);
    for (int r=0;r<=k;++r){ int ai=0; float mv=-INFINITY; for(int j=0;j<V;++j){ float v=tlog[r*V+j]; if(v>mv){mv=v;ai=j;} } ttok[r]=ai; }
    int alen=0; while(alen<k && ttok[alen]==dtok[alen]) ++alen;
    out_tok.assign(k+1,-1);
    for(int i=0;i<=alen;++i) out_tok[i]=ttok[i];
    out_len = alen+1;
}

void test_spec_verify() {
    std::printf("[greedy speculative-verify DAG]\n");
    const int k = 4, V = 256;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-5.f,5.f);

    auto run_case = [&](const char* name, std::function<void(std::vector<float>&,std::vector<int>&)> setup){
        std::vector<float> tlog((k+1)*V);
        for (auto& v: tlog) v = d(rng);
        std::vector<int> dtok(k);
        setup(tlog, dtok);
        std::vector<int> got, ref; int gl=0, rl=0;
        run_spec(k,V,tlog,dtok,got,gl);
        cpu_spec(k,V,tlog,dtok,ref,rl);
        bool ok = (gl==rl) && (got==ref);
        expect(ok, name);
    };

    // all accepted: draft tokens = target argmax for every position.
    run_case("all-accepted (full block + bonus)", [&](std::vector<float>& tl, std::vector<int>& dt){
        for(int r=0;r<k;++r){ int ai=0; float mv=-INFINITY; for(int j=0;j<V;++j){ if(tl[r*V+j]>mv){mv=tl[r*V+j];ai=j;} } dt[r]=ai; }
    });
    // reject at position 2: tokens 0,1 match argmax; 2 is wrong.
    run_case("reject mid-block", [&](std::vector<float>& tl, std::vector<int>& dt){
        for(int r=0;r<k;++r){ int ai=0; float mv=-INFINITY; for(int j=0;j<V;++j){ if(tl[r*V+j]>mv){mv=tl[r*V+j];ai=j;} } dt[r]=ai; }
        dt[2] = (dt[2]+1)%V;  // force mismatch at 2
    });
    // reject at position 0: nothing accepted.
    run_case("reject at first position", [&](std::vector<float>& tl, std::vector<int>& dt){
        for(int r=0;r<k;++r){ int ai=0; float mv=-INFINITY; for(int j=0;j<V;++j){ if(tl[r*V+j]>mv){mv=tl[r*V+j];ai=j;} } dt[r]=(ai+7)%V; }
    });
}

}  // namespace

int main() {
    CU_CHECK(cuInit(0));
    CUdevice dev; CU_CHECK(cuDeviceGet(&dev,0));
    CU_CHECK(cuDeviceGetAttribute(&g_cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU_CHECK(cuDeviceGetAttribute(&g_cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    CUcontext ctx; CU_CHECK(cuDevicePrimaryCtxRetain(&ctx,dev)); CU_CHECK(cuCtxSetCurrent(ctx));
    char nm[256]; CU_CHECK(cuDeviceGetName(nm,sizeof(nm),dev));
    std::printf("Device: %s (sm_%d%d)\n", nm, g_cc_major, g_cc_minor);

    test_matrix();
    test_spec_verify();

    CU_CHECK(cuDevicePrimaryCtxRelease(dev));
    std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
    if (g_fail==0) std::printf("ALL PASS\n");
    return g_fail==0 ? 0 : 1;
}
