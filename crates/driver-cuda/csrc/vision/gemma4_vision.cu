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
// THE SIX `norm::rmsnorm_bf16` CALLS ARE GONE TOO, and the reason the
// previous round gave for keeping them is recorded below with what was wrong
// with it. It said: `rmsnorm_bf16` is not a grid -- it forwards to
// `rmsnorm_strided_bf16`, which reads six host values (three pointer
// alignments and three strides) to choose between a vec8 kernel and a scalar
// one -- and copying that decision here would be a THIRD copy of something
// that already exists twice, in C++ and in the Rust `Select`.
//
// Every clause of that is true and it is an argument about tidiness, which
// §10.10 outranks: **a launcher goes when its whole consumer set has gone**,
// and this file was one of exactly two members of `norm::rmsnorm_bf16`'s.
// While it stayed, the row could not be routed however ready it was, and
// `device.rs` had already recorded the cost of that -- the sentence naming
// this file's consumption was allowed to go stale for months because a
// consumer set nobody re-measures decays quietly. A third copy of six lines
// is cheaper than a row that can never move.
//
// So `rms` below IS that copy, and it is a copy in the strict sense: the
// predicate is `rmsnorm_vec8_ok`'s six clauses verbatim, the two launches are
// `rmsnorm_strided_bf16`'s two `<<<>>>` expressions verbatim at the same
// template arguments and the same block widths, and the strides are the
// `hidden, hidden, hidden` that `rmsnorm_bf16` itself substituted. Byte
// identity is not argued for here, it is MEASURED: the tower's whole output
// is compared between the two builds over four shapes and two weight sets,
// one shape degenerate, and the numbers are in `new-horizon.md` §42.
//
// The three `gemm::act_x_wt_bf16` calls stay. That one is a cuBLAS call with
// no `<<<>>>` to copy at all -- there is no launcher behind it to free, and
// `execution.rs` already calls it `Service::Cublas`.
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

// `norm::rmsnorm_strided_bf16`'s body, at the strides `norm::rmsnorm_bf16`
// substituted (`hidden` for both) -- see the header block for why the copy is
// here rather than a call.
//
// The predicate reads an ADDRESS, which is why no `LaunchRule` can state it
// and why the JIT row for this kernel fires the scalar arm unconditionally:
// `rmsnorm.cu`'s own header says so. Reproducing it is therefore the only way
// to keep what this tower launches today unchanged, and keeping it unchanged
// is the whole claim -- the two arms compute the same function and differ in
// the last bit, which is exactly the difference a tolerance cannot see.
inline bool rms_vec8_ok(const void* x,const void* y,const void* w,int hidden){
    auto aligned=[](const void* p){
        return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    return hidden%8==0 && aligned(x) && aligned(y) && aligned(w);
}

inline void rms(const bf* x,const bf* w,bf* y,int rows,int hidden,float eps,
                cudaStream_t S){
    dim3 grid(rows);
    if(rms_vec8_ok(x,y,w,hidden)){
        constexpr int VBLOCK=512;
        nd::rmsnorm_vec8<VBLOCK,/*WEIGHT_PLUS_ONE=*/false>
            <<<grid,VBLOCK,0,S>>>(D(x),D(w),D(y),nullptr,hidden,hidden,hidden,eps);
        return;
    }
    constexpr int BLOCK=256;
    nd::rmsnorm<bfd,BLOCK><<<grid,dim3(BLOCK),0,S>>>(
        D(x),D(w),D(y),hidden,hidden,hidden,eps);
}

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
        rms(h,L.in_ln,hn,N,Hd,EPS,S);
        clin(hn,q,L.q,Hd,Hd);clin(hn,k,L.k,Hd,Hd);clin(hn,v,L.v,Hd,Hd);
        rms(q,L.q_norm,q,N*NH,64,EPS,S);rms(k,L.k_norm,k,N*NH,64,EPS,S);nd::rmsnorm_no_scale<bfd,256><<<dim3(N*NH),dim3(256),0,S>>>(D(v),D(v),64,EPS);
        dim3 rg(1,NH,N);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(q),pos,N,NH,THETA);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(k),pos,N,NH,THETA);
        for(int hh=0;hh<NH;hh++){vd::k_qk<bfd><<<G2(N,N),B2,0,S>>>(D(q),D(k),scr,N,NH,hh,1.0f);vd::k_softmax<bfd><<<N,256,0,S>>>(scr,N);vd::k_av<bfd><<<G2(64,N),B2,0,S>>>(scr,D(v),D(attn),N,NH,hh);}
        clin(attn,tmp,L.o,Hd,Hd);
        rms(tmp,L.post_attn_ln,tmp,N,Hd,EPS,S);
        { const long n=(long)N*Hd; if(n) nd::residual_add<bfd><<<(unsigned)((n+255)/256),256,0,S>>>(D(h),D(tmp),n); }
        rms(h,L.pre_ff_ln,hn,N,Hd,EPS,S);
        clin(hn,gate,L.gate,Hd,IM);clin(hn,up,L.up,Hd,IM);
        { const int ge=(int)((long)N*IM); md::geglu_tanh<bfd><<<(ge+255)/256,256,0,S>>>(D(gate),D(up),D(act),ge); }
        clin(act,tmp,L.down,IM,Hd);
        rms(tmp,L.post_ff_ln,tmp,N,Hd,EPS,S);
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

// `scatter_gemma4_vision` WAS HERE, and it is deleted.
//
// It was the tower's other entry point -- encode each image and overwrite the
// soft-token rows of a device `hidden` buffer in place, the in-fire shape
// `qwen3vl_scatter` still has. `encode_gemma4_vision` below superseded it when
// gemma-4's towers moved to the encode ABI (host rows out, anchor-segmented
// CSR), and nothing was left calling it: not `gemma4_towers_c.cpp`, not the
// shim, not a test, not another `.cu`. The transitive audit
// (`scripts/csrc-reachability-audit.py`) reported it UNREACHABLE and a
// repository-wide search for the name found its own declaration, its own
// definition, and one mention in a `.cuh` comment.
//
// It carried one launch (`vd::k_f32_to_bf16`) and reached `run_gemma4_vision`,
// which survives on `encode_gemma4_vision`'s consumption. §10.10, from the
// other side: the launcher went because its whole consumer set had already
// gone, and nobody had noticed.

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
