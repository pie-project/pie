// COPIED from driver-cuda/csrc/src/model/gemma4/gemma4_audio_forward.cu
// (2026-08-09, the VL tower bridge; includes localized). The OLD driver
// keeps its copy until phase E deletes it. Do not diverge.
// Gemma-4 audio encoder forward (bf16). See gemma4_audio_forward.hpp.
//
// First-draft scaffold mirroring gemma4_vision_forward.cu (naive kernels +
// cudaMalloc scratch; bf16 storage + fp32 compute, matching the driver). The
// per-stage math is transcribed from transformers 5.9 `modeling_gemma4.py`
// (Gemma4AudioModel / Gemma4AudioLayer / Gemma4AudioAttention /
// Gemma4AudioLightConv1d / Gemma4AudioSubSampleConvProjection) and checked
// shape-wise against scripts/gemma4_audio_parity_ref.py.
//
// CUDA-only includes (no model/loader headers) so nvcc never sees toml++.
//
// PARITY (bf16-vs-bf16, `gemma4_audio_full_parity` vs
// /tmp/gemma4_audio_parity/*.npy at 188 mel frames → 47 audio tokens):
// sscp 1.00000 / layer0 1.00000 / layer5 0.99999 / layer11 0.99997 /
// encoder_out 0.99997 / projected 0.99997, rel_rms 0.744% — HF's own bf16
// stability. An earlier revision of this block quoted 199 frames → 50 tokens
// and cosines around 0.9992 from a script that was never committed;
// `gemma4_audio_parity_ref.py` exists now and these are its numbers.
// The stages verified:
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

//
// THE KERNELS ARE NOT HERE. All twelve moved to `vision/gemma4_audio.cuh` in
// the JIT crate's header tree, where they are named templates over the
// storage format; this file keeps the host half -- the three entry points,
// the checkpoint hook, the scratch arena, the conformer loop -- and includes
// them. The move is what made them reachable from NVRTC at all: an anonymous
// namespace has no name to give `nvrtcAddNameExpression`, so the runtime
// could not resolve a `CUfunction` for one of them. It is a MOVE and not a
// copy, which `tests/sources.rs::no_global_is_defined_twice` enforces,
// because `norm/altup_aux` shipped a release with two copies that had drifted
// and every test passing.
//
// Every launch below therefore spells an INSTANTIATION -- `vd::k_silu<bfd>`
// -- and every bf16 pointer crosses through `D()`. See the alias block for
// why the cast is there and why it is not a conversion.

#include "vision/gemma4_audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ssm/causal_conv1d.cuh"

#include "vision/gemma4_audio.cuh"
#include "vision/gemma4_naive_kernels.cuh"

namespace pie_cuda_driver::model {

// ── Per-stage checkpoint hook (parity debugging). See .hpp. ─────────────────
static Gemma4AudioCkptFn g_audio_ckpt = nullptr;
static void* g_audio_ckpt_user = nullptr;
void set_gemma4_audio_ckpt(Gemma4AudioCkptFn fn, void* user) {
    g_audio_ckpt = fn; g_audio_ckpt_user = user;
}

namespace {

typedef __nv_bfloat16 bf;

// The bridge between the tower's `bf` and the kernels' `T`.
//
// `bf` is NVIDIA's `__nv_bfloat16`, which is what `vision/gemma4_audio.hpp`
// declares and what every caller passes; the templates in the `.cuh` are
// instantiated at `device::bf16`, the prelude's own two-byte struct. They are
// the same sixteen bits and NOT the same type -- the prelude cannot name
// NVIDIA's, because NVRTC bundles no device headers and `<cuda_bf16.h>` is one
// of the 31 it answered none of.
//
// So the pointer is REINTERPRETED, once, at each launch. `D` is spelled as a
// pair of overloads rather than a cast at every argument because a
// `reinterpret_cast` written out fifty times is fifty places to get the
// constness wrong, and the compiler cannot tell a wrong one from a right one.
namespace vd = ::pie_cuda_driver::kernels::vision::device;
namespace sd = ::pie_cuda_driver::kernels::ssm::device;
using bfd = ::pie_cuda_driver::kernels::device::bf16;
inline bfd* D(bf* p) { return reinterpret_cast<bfd*>(p); }
inline const bfd* D(const bf* p) { return reinterpret_cast<const bfd*>(p); }
#define ACK(x) do{cudaError_t e=(x);if(e)throw std::runtime_error(std::string("gemma4_audio: ")+cudaGetErrorString(e));}while(0)

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

// The twelve kernels were here, between this line and `B2`. They are in
// `vision/gemma4_audio.cuh` now, templated over the storage format -- see
// that header for which two of them a `LaunchRule` states and why the other
// ten are refused.
//
// A thirteenth was here and is not anywhere: `k_depthwise_causal` is
// `ssm::device::causal_conv1d_prefill<T, false>` -- bit for bit the same
// accumulation in the same order -- and the conformer loop fires that
// template directly. It went through the `ssm::causal_conv1d_prefill_noact_bf16`
// host launcher until this session; that launcher is now unreferenced by any
// C++ in the tree and named by no table row, which is the whole of its
// consumer set.

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
// `k_qkv_scale`, `k_rel_pos_enc` and `k_local_attn` were here. They are in
// `vision/gemma4_audio.cuh` with the other nine; the commentary above is the
// contract they implement and stays with the call sites that rely on it.
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
        vd::k_clamp<bfd><<<((long)N*Kin+255)/256,256,0,S>>>(D(x),D(xc),D(c.imin),D(c.imax),(long)N*Kin);
        vd::k_matmul<bfd><<<G2(Out,N),B2,0,S>>>(D(xc),D(c.w),D(out),N,Kin,Out);
        vd::k_clamp<bfd><<<((long)N*Out+255)/256,256,0,S>>>(D(out),D(out),D(c.omin),D(c.omax),(long)N*Out);};

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
        vd::k_f32_to_bf16<bfd><<<((long)T0*F0+255)/256,256,0,S>>>(f32d,D(feat_bf),(long)T0*F0);
        ACK(cudaStreamSynchronize(S));
    }
    // layer0: conv [1,T0,F0]→[C0,T1,F1], LN-over-ch + ReLU.
    bf* c0=MAL((long)C0*T1*F1);
    { dim3 g((F1+15)/16,(T1+15)/16,C0); vd::k_conv2d_s2<bfd><<<g,B2,0,S>>>(D(feat_bf),D(w.sscp0_conv),D(c0),1,T0,F0,C0,T1,F1); }
    bf* c0cl=MAL((long)T1*F1*C0);
    { dim3 g((F1+15)/16,(T1+15)/16,C0); vd::k_chlast<bfd><<<g,B2,0,S>>>(D(c0),D(c0cl),C0,T1,F1); }
    vd::k_layernorm_relu<bfd><<<T1*F1,128,0,S>>>(D(c0cl),D(w.sscp0_norm),D(c0cl),T1*F1,C0,EPS);
    { dim3 g((F1+15)/16,(T1+15)/16,C0); vd::k_chfirst<bfd><<<g,B2,0,S>>>(D(c0cl),D(c0),C0,T1,F1); }
    // layer1: conv [C0,T1,F1]→[C1,T2,F2], LN-over-ch + ReLU.
    bf* c1=MAL((long)C1*T2*F2);
    { dim3 g((F2+15)/16,(T2+15)/16,C1); vd::k_conv2d_s2<bfd><<<g,B2,0,S>>>(D(c0),D(w.sscp1_conv),D(c1),C0,T1,F1,C1,T2,F2); }
    bf* c1cl=MAL((long)T2*F2*C1);
    { dim3 g((F2+15)/16,(T2+15)/16,C1); vd::k_chlast<bfd><<<g,B2,0,S>>>(D(c1),D(c1cl),C1,T2,F2); }
    vd::k_layernorm_relu<bfd><<<T2*F2,128,0,S>>>(D(c1cl),D(w.sscp1_norm),D(c1cl),T2*F2,C1,EPS);
    { dim3 g((F2+15)/16,(T2+15)/16,C1); vd::k_chfirst<bfd><<<g,B2,0,S>>>(D(c1cl),D(c1),C1,T2,F2); }
    // flatten [C1,T2,F2] → [T2, F2*C1] and input_proj → [T2, hidden].
    const int N=T2, FLAT=F2*C1;
    bf* flat=MAL((long)N*FLAT);
    { dim3 g((FLAT+15)/16,(N+15)/16); vd::k_sscp_flatten<bfd><<<g,B2,0,S>>>(D(c1),D(flat),C1,T2,F2); }
    bf* h=MAL((long)N*Hd);
    vd::k_matmul<bfd><<<G2(Hd,N),B2,0,S>>>(D(flat),D(w.sscp_input_proj),D(h),N,FLAT,Hd);

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
    { dim3 g((Hd+15)/16,(P+15)/16); vd::k_rel_pos_enc<bfd><<<g,B2,0,S>>>(D(pe),P,Hd); }
    bf* relk=MAL((long)P*Hd);                              // relative_k_proj(pe) → [P, H*hd]

    auto ffn=[&](const AudioFfnRaw& ff){
        // residual = x; x=clamp; pre_ln; fc1; silu; fc2; clamp; post_ln; ×RW; +res
        vd::k_rms<bfd><<<N,256,0,S>>>(D(h),D(ff.pre_ln),D(hn),N,Hd,EPS);
        clin(hn,ffmid,xc,ff.fc1,N,Hd,IM);
        vd::k_silu<bfd><<<((long)N*IM+255)/256,256,0,S>>>(D(ffmid),D(ffmid),(long)N*IM);
        clin(ffmid,ffout,xc,ff.fc2,N,IM,Hd);
        vd::k_rms<bfd><<<N,256,0,S>>>(D(ffout),D(ff.post_ln),D(ffout),N,Hd,EPS);
        vd::k_axpy<bfd><<<((long)N*Hd+255)/256,256,0,S>>>(D(h),D(ffout),RW,(long)N*Hd);
    };

    int li=0;
    for(const auto& L:w.layers){
        // feed_forward1 (macaron half-step)
        ffn(L.ff1);
        // self-attention
        vd::k_rms<bfd><<<N,256,0,S>>>(D(h),D(L.norm_pre_attn),D(hn),N,Hd,EPS);
        clin(hn,q,xc,L.q,N,Hd,Hd); clin(hn,k,xc,L.k,N,Hd,Hd); clin(hn,v,xc,L.v,N,Hd,Hd);
        vd::k_qkv_scale<bfd><<<G2(Hd,N),B2,0,S>>>(D(q),D(k),D(L.per_dim_scale),N,NH,hd,q_scale,k_scale);
        // relative_k_proj(pe) → relk [P, H*hd]; NOT a clipped linear (plain matmul).
        vd::k_matmul<bfd><<<G2(Hd,P),B2,0,S>>>(D(pe),D(L.relative_k),D(relk),P,Hd,Hd);
        { dim3 g((N+127)/128,NH); vd::k_local_attn<bfd><<<g,128,0,S>>>(D(q),D(k),D(v),D(relk),D(attn),N,NH,hd,P,CAP); }
        clin(attn,tmp,xc,L.post,N,Hd,Hd);
        vd::k_rms<bfd><<<N,256,0,S>>>(D(tmp),D(L.norm_post_attn),D(tmp),N,Hd,EPS);
        vd::k_add<bfd><<<((long)N*Hd+255)/256,256,0,S>>>(D(h),D(tmp),(long)N*Hd);
        // light depthwise-conv module
        vd::k_rms<bfd><<<N,256,0,S>>>(D(h),D(L.lconv_pre_ln),D(hn),N,Hd,EPS);
        clin(hn,start,xc,L.lconv_start,N,Hd,2*Hd);            // [N, 2*hidden]
        vd::k_glu<bfd><<<G2(Hd,N),B2,0,S>>>(D(start),D(glu),N,Hd);            // GLU → [N, hidden]
        // Was `k_depthwise_causal`, a local copy of this kernel: same [N, C]
        // layout, same [C, K] per-channel weight, same `t-(K-1)+j` indexing,
        // same zero pad. The library's only difference was a fused silu,
        // which is a template parameter there now. bias and state are nullptr
        // -- this caller has neither.
        //
        // This was a call to `kernels::ssm::causal_conv1d_prefill_noact_bf16`
        // until the launcher's grid was copied here whole: `BLOCK = 64`,
        // `dim3 grid(C)`, and the degenerate guard that the launcher opens
        // with. The device text is the same template instantiated at the same
        // two arguments, so nothing that runs on the GPU moved -- but that
        // launcher had exactly one caller in the tree and no table row, and
        // this was it.
        { constexpr int BLOCK=64; const int C=Hd, K=w.conv_kernel;
          if(N>0&&C>0&&K>0) sd::causal_conv1d_prefill<bfd,false><<<dim3(C),dim3(BLOCK),0,S>>>(
              D(glu),D(L.depthwise_conv),nullptr,D(conv),nullptr,N,C,K); }
        // clamp(±finfo_max) is a no-op in bf16 range → skip; conv_norm + silu
        vd::k_rms<bfd><<<N,256,0,S>>>(D(conv),D(L.lconv_conv_norm),D(conv),N,Hd,EPS);
        vd::k_silu<bfd><<<((long)N*Hd+255)/256,256,0,S>>>(D(conv),D(conv),(long)N*Hd);
        clin(conv,tmp,xc,L.lconv_end,N,Hd,Hd);
        vd::k_add<bfd><<<((long)N*Hd+255)/256,256,0,S>>>(D(h),D(tmp),(long)N*Hd);
        // feed_forward2 (macaron half-step)
        ffn(L.ff2);
        // norm_out
        vd::k_rms<bfd><<<N,256,0,S>>>(D(h),D(L.norm_out),D(h),N,Hd,EPS);
        // ckpt: layer{li} output (matches HF Gemma4AudioLayer hidden_states dump).
        { char nm[16]; snprintf(nm,sizeof nm,"layer%d",li); CKPT(nm,h,(long)N*Hd); }
        ++li;
    }

    // ── 3) output_proj (1024→1536 +bias) ────────────────────────────────────
    bf* enc=MAL((long)N*OPD);
    vd::k_matmul_bias<bfd><<<G2(OPD,N),B2,0,S>>>(D(h),D(w.output_proj_w),D(w.output_proj_b),D(enc),N,Hd,OPD);
    CKPT("encoder_out",enc,(long)N*OPD);

    // ── 4) embedder: parameterless RMSNorm(1536) → projection (1536→2560) ────
    bf* en=MAL((long)N*OPD);
    vd::k_rms<bfd><<<N,256,0,S>>>(D(enc),nullptr,D(en),N,OPD,EPS);
    vd::k_matmul<bfd><<<G2(TXT,N),B2,0,S>>>(D(en),D(w.embed_proj),D(out_proj),N,OPD,TXT);
    CKPT("projected",out_proj,(long)N*TXT);

    ACK(cudaStreamSynchronize(S));
}

// `scatter_gemma4_audio` WAS HERE, and it is deleted.
//
// The audio twin of `scatter_gemma4_vision`: encode each clip and overwrite
// the soft-token rows of a device `hidden` buffer in place. `encode_gemma4_
// audio` below superseded it when gemma-4's towers moved to the encode ABI,
// and its consumer set was empty by the same evidence -- the transitive audit
// reports it UNREACHABLE, and a repository-wide search for the name finds its
// declaration, its definition and nothing else.
//
// It launched nothing itself; what it reached, `run_gemma4_audio`, survives on
// `encode_gemma4_audio`'s consumption and keeps all thirty-five.

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
