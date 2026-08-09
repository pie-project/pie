// COPIED from driver-cuda/csrc/src/model/gemma4/gemma4_vision_forward.cu
// (2026-08-09, the VL tower bridge; includes localized). The OLD driver
// keeps its copy until phase E deletes it. Do not diverge.
// Gemma-4 vision encoder forward (bf16). See gemma4_vision_forward.hpp.
// Kernels ported verbatim from the parity-verified standalone
// (crates/driver-cuda/csrc/tests/gemma4_vision_full_parity_bf16.cu): rel_rms 1.07%,
// cosine 0.99994 vs HF-bf16. bf16 storage + fp32 compute, matching the driver.
//
// First-cut: naive kernels + cudaMalloc scratch (correctness over speed); a
// cuBLAS/workspace pass is a follow-up. CUDA-only includes (no model/loader
// headers) so nvcc never sees the toml++ config headers.
//
// THE KERNELS ARE NOT HERE. All nine moved to `vision/gemma4_vision.cuh` in
// the JIT crate's header tree, where they are named templates over the
// storage format; this file keeps the host half -- the three entry points,
// the scratch arena, the cuBLAS handle, the per-image loop -- and includes
// them. The move is what made them reachable from NVRTC at all: an anonymous
// namespace has no name to give `nvrtcAddNameExpression`, so the runtime
// could not resolve a `CUfunction` for one of them. It is a MOVE and not a
// copy, which `tests/sources.rs::no_global_is_defined_twice` enforces,
// because `norm/altup_aux` shipped a release with two copies that had drifted
// and every test passing.
//
// Every launch below therefore spells an INSTANTIATION -- `vd::k_scale<bfd>`
// -- and every bf16 pointer crosses through `D()`. See the alias block for
// why the cast is there and why it is not a conversion.
//
// FIVE OF THOSE LAUNCHES NAME ANOTHER FAMILY'S KERNEL. This tower used to
// reach `kernels::norm::residual_add_bf16`, `norm::rmsnorm_no_scale_bf16` and
// `mlp::geglu_tanh_bf16` -- three ahead-of-time host launchers -- from inside
// its own sequence of fires. That is C++ calling C++, and it is the thing
// that keeps a launcher alive after its kernel has been migrated: the rule is
// that a launcher goes when its WHOLE consumer set has gone, and the JIT shim
// is only one consumer. `norm::residual_add_bf16` has been the standing
// example since the split, named in `kernels-cuda-new/src/device.rs` as the
// row deliberately kept out of `JIT_DISPATCHED` because this file and
// `gemm/gemm.cpp` still called it.
//
// The device text those launchers fire is a template in the JIT header tree,
// reachable from here over the same `-iquote` path that already carries
// `vision/gemma4_vision.cuh`. So the calls became launches: each one copies
// its launcher's `<<<grid, block, smem, stream>>>` expression VERBATIM --
// including the zero-length guard where there was one -- and instantiates the
// same template at the same type. Not one instruction that runs on the device
// changed; what changed is who wrote the triple-chevron.
//
// The five `norm::rmsnorm_bf16` calls and the three `gemm::act_x_wt_bf16`
// calls stay, and for different reasons. `rmsnorm_bf16` is not a grid: it
// forwards to `rmsnorm_strided_bf16`, which reads six host values -- three
// pointer alignments and three strides -- to choose between a vec8 kernel and
// a scalar one. Copying that here would be a third copy of a decision that
// already exists twice, in C++ and in the Rust `Select`. `act_x_wt_bf16` is a
// cuBLAS call and has no `<<<>>>` to copy at all.
#include "vision/gemma4_vision.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemm/gemm.hpp"
#include "mlp/swiglu.cuh"
#include "norm/elementwise.cuh"
#include "norm/rmsnorm.cuh"
#include "norm/rmsnorm.hpp"
#include "vision/gemma4_naive_kernels.cuh"
#include "vision/gemma4_vision.cuh"

namespace pie_cuda_driver::model {
namespace {

typedef __nv_bfloat16 bf;

// The bridge between the tower's `bf` and the kernels' `T`.
//
// `bf` is NVIDIA's `__nv_bfloat16`, which is what `vision/gemma4_vision.hpp`
// declares and what every caller passes; the templates in the `.cuh` are
// instantiated at `device::bf16`, the prelude's own two-byte struct. They are
// the same sixteen bits and NOT the same type -- the prelude cannot name
// NVIDIA's, because NVRTC bundles no device headers and `<cuda_bf16.h>` is one
// of the 31 it answered none of.
//
// So the pointer is REINTERPRETED, once, at each launch. `D` is spelled as a
// pair of overloads rather than a cast at every argument because a
// `reinterpret_cast` written out thirty times is thirty places to get the
// constness wrong, and the compiler cannot tell a wrong one from a right one.
namespace vd = ::pie_cuda_driver::kernels::vision::device;
namespace md = ::pie_cuda_driver::kernels::mlp::device;
namespace nd = ::pie_cuda_driver::kernels::norm::device;
using bfd = ::pie_cuda_driver::kernels::device::bf16;
inline bfd* D(bf* p) { return reinterpret_cast<bfd*>(p); }
inline const bfd* D(const bf* p) { return reinterpret_cast<const bfd*>(p); }
#define VCK(x) do{cudaError_t e=(x);if(e)throw std::runtime_error(std::string("gemma4_vision: ")+cudaGetErrorString(e));}while(0)

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

dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}

}  // namespace

void run_gemma4_vision(const VisRawWeights& w,
                       const bf* pixel,const float* pos,const int* grp,
                       int N,int OUTL,bf* out_proj,cudaStream_t S,VisDebugTap dbg){
    auto tap=[&](const char* tag,const bf* d,long n){
        if(!dbg) return;
        VCK(cudaStreamSynchronize(S)); dbg(tag,d,n); };
    const int Hd=w.hidden, NH=w.heads, IM=w.intermediate, TXT=w.text_hidden, PT=w.pos_table_size;
    const float EPS=w.eps, THETA=w.theta;
    if(Hd!=768||NH!=12) throw std::runtime_error("gemma4_vision: unexpected dims (expected hidden=768, heads=12)");

    kernels::gemm::CublasHandle cublas(S);
    DeviceScratch scratch;
    auto MAL=[&](long n){return scratch.alloc<bf>(n);};
    bf *h=MAL((long)N*Hd),*hn=MAL((long)N*Hd),*xc=MAL((long)N*IM),*q=MAL((long)N*Hd),*k=MAL((long)N*Hd),*v=MAL((long)N*Hd),
       *attn=MAL((long)N*Hd),*gate=MAL((long)N*IM),*up=MAL((long)N*IM),*act=MAL((long)N*IM),*tmp=MAL((long)N*Hd);
    float* scr=scratch.alloc<float>((long)N*N);
    auto clin=[&](const bf* x,bf* out,const VisClipRaw& c,int Kin,int Out){
        vd::k_clamp<bfd><<<((long)N*Kin+255)/256,256,0,S>>>(D(x),D(xc),D(c.imin),D(c.imax),(long)N*Kin);
        kernels::gemm::act_x_wt_bf16(cublas.handle(),xc,c.w,out,N,Out,Kin);
        vd::k_clamp<bfd><<<((long)N*Out+255)/256,256,0,S>>>(D(out),D(out),D(c.omin),D(c.omax),(long)N*Out);};

    vd::k_scale<bfd><<<((long)N*Hd+255)/256,256,0,S>>>(D(pixel),D(hn),(long)N*Hd);
    kernels::gemm::act_x_wt_bf16(cublas.handle(),hn,w.patch_w,h,N,Hd,Hd);
    vd::k_addpos_grid2d<bfd><<<G2(Hd,N),B2,0,S>>>(D(h),D(w.pos_table),pos,N,Hd,PT);
    int li=0;
    for(const auto& L:w.layers){
        kernels::norm::rmsnorm_bf16(h,L.in_ln,hn,N,Hd,EPS,S);
        clin(hn,q,L.q,Hd,Hd);clin(hn,k,L.k,Hd,Hd);clin(hn,v,L.v,Hd,Hd);
        kernels::norm::rmsnorm_bf16(q,L.q_norm,q,N*NH,64,EPS,S);kernels::norm::rmsnorm_bf16(k,L.k_norm,k,N*NH,64,EPS,S);nd::rmsnorm_no_scale<bfd,256><<<dim3(N*NH),dim3(256),0,S>>>(D(v),D(v),64,EPS);
        dim3 rg(1,NH,N);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(q),pos,N,NH,THETA);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(k),pos,N,NH,THETA);
        for(int hh=0;hh<NH;hh++){vd::k_qk<bfd><<<G2(N,N),B2,0,S>>>(D(q),D(k),scr,N,NH,hh,1.0f);vd::k_softmax<bfd><<<N,256,0,S>>>(scr,N);vd::k_av<bfd><<<G2(64,N),B2,0,S>>>(scr,D(v),D(attn),N,NH,hh);}
        clin(attn,tmp,L.o,Hd,Hd);
        kernels::norm::rmsnorm_bf16(tmp,L.post_attn_ln,tmp,N,Hd,EPS,S);
        { const long n=(long)N*Hd; if(n) nd::residual_add<bfd><<<(unsigned)((n+255)/256),256,0,S>>>(D(h),D(tmp),n); }
        kernels::norm::rmsnorm_bf16(h,L.pre_ff_ln,hn,N,Hd,EPS,S);
        clin(hn,gate,L.gate,Hd,IM);clin(hn,up,L.up,Hd,IM);
        { const int ge=(int)((long)N*IM); md::geglu_tanh<bfd><<<(ge+255)/256,256,0,S>>>(D(gate),D(up),D(act),ge); }
        clin(act,tmp,L.down,IM,Hd);
        kernels::norm::rmsnorm_bf16(tmp,L.post_ff_ln,tmp,N,Hd,EPS,S);
        { const long n=(long)N*Hd; if(n) nd::residual_add<bfd><<<(unsigned)((n+255)/256),256,0,S>>>(D(h),D(tmp),n); }
        if(li++==0) tap("layer0",h,(long)N*Hd);
    }
    tap("layer_last",h,(long)N*Hd);
    float* pf=scratch.alloc<float>((long)OUTL*Hd);VCK(cudaMemsetAsync(pf,0,(long)OUTL*Hd*4,S));
    vd::k_pool<bfd><<<G2(Hd,N),B2,0,S>>>(D(h),grp,pf,N,Hd,9.f);
    bf* pooled=MAL((long)OUTL*Hd);vd::k_pool_finish<bfd><<<((long)OUTL*Hd+255)/256,256,0,S>>>(pf,D(pooled),sqrtf((float)Hd),(long)OUTL*Hd);
    tap("pooled_last_hidden",pooled,(long)OUTL*Hd);
    bf* pn=MAL((long)OUTL*Hd);nd::rmsnorm_no_scale<bfd,256><<<dim3(OUTL),dim3(256),0,S>>>(D(pooled),D(pn),Hd,EPS);
    kernels::gemm::act_x_wt_bf16(cublas.handle(),pn,w.embed_proj,out_proj,OUTL,TXT,Hd);
    VCK(cudaStreamSynchronize(S));
}

namespace {
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
        vd::k_f32_to_bf16<bfd><<<(n_floats+255)/256,256,0,S>>>(pix_f32_d,D(pix_bf_d),n_floats);

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
        vd::k_f32_to_bf16<bfd><<<(n_floats + 255) / 256, 256, 0, S>>>(
            pix_f32_d, D(pix_bf_d), n_floats);

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
