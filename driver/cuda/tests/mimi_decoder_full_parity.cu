// Standalone parity harness for the Mimi neural-codec DECODER (codes → 24 kHz
// waveform) — the OUTPUT-modality module. Loads the HF-dumped RVQ codes,
// populates `MimiDecoderRawWeights` from weights/, runs `run_mimi_decoder`, and
// compares the output waveform (and every staged intermediate) against the
// reference dumps with rel_rms + cosine. bf16-vs-bf16 cosine is the real metric
// (MULTIMODAL.md §11); fp32 refs are shown too for context.
//
//   nvcc -O2 -arch=sm_89 -std=c++17 -I driver/cuda/src \
//        driver/cuda/tests/mimi_decoder_full_parity.cu -o /tmp/mdp
//   /tmp/mdp /tmp/mimi_decoder_parity
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "model/mimi_decoder_forward.cu"  // decoder under test + kernels + .hpp

using namespace pie_cuda_driver::model;
using BF = __nv_bfloat16;

#define HCK(x) do{cudaError_t e=(x);if(e){std::fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);}}while(0)

// ── npy loader (handles f32 / f16 / bf16-as-u16 / i64) ───────────────────────
struct Npy {
    std::vector<int64_t> shape; char kind = 0; int isz = 0; bool fortran = false;
    std::vector<uint8_t> data;
    int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
};
static Npy load_npy(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s\n", p.c_str()); std::exit(2); }
    char m[6]; f.read(m, 6);
    uint8_t maj = f.get(), mn = f.get(); (void)mn;
    uint32_t hl;
    if (maj == 1) { uint16_t h; f.read((char*)&h, 2); hl = h; } else { f.read((char*)&hl, 4); }
    std::string hdr(hl, 0); f.read(hdr.data(), hl);
    Npy o;
    auto dp = hdr.find("'descr'"); auto q = hdr.find('\'', hdr.find(':', dp) + 1);
    std::string d = hdr.substr(q + 1, hdr.find('\'', q + 1) - q - 1);
    o.kind = d[1]; o.isz = std::atoi(d.substr(2).c_str());
    o.fortran = hdr.find("'fortran_order': True") != std::string::npos;
    auto sp = hdr.find("'shape'"); auto lp = hdr.find('(', sp), rp = hdr.find(')', lp);
    std::string sh = hdr.substr(lp + 1, rp - lp - 1); size_t i = 0;
    while (i < sh.size()) {
        while (i < sh.size() && !isdigit(sh[i])) ++i; if (i >= sh.size()) break;
        int64_t v = 0; while (i < sh.size() && isdigit(sh[i])) v = v * 10 + (sh[i++] - '0');
        o.shape.push_back(v);
    }
    o.data.resize((size_t)o.numel() * o.isz);
    f.read((char*)o.data.data(), (std::streamsize)o.data.size());
    return o;
}
static std::vector<float> as_f32(const Npy& n) {
    std::vector<float> out(n.numel());
    if (n.kind == 'f' && n.isz == 4) std::memcpy(out.data(), n.data.data(), n.numel() * 4);
    else if (n.kind == 'f' && n.isz == 2) {  // float16
        const uint16_t* p = (const uint16_t*)n.data.data();
        for (int64_t i = 0; i < n.numel(); i++) {
            uint16_t h = p[i]; uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m2 = h & 0x3ff, bits;
            if (e == 0) { if (m2 == 0) bits = s << 31; else { e = 127 - 15 + 1; while (!(m2 & 0x400)) { m2 <<= 1; e--; } m2 &= 0x3ff; bits = (s << 31) | (e << 23) | (m2 << 13); } }
            else if (e == 0x1f) bits = (s << 31) | (0xff << 23) | (m2 << 13);
            else bits = (s << 31) | ((e - 15 + 127) << 23) | (m2 << 13);
            float v; std::memcpy(&v, &bits, 4); out[i] = v;
        }
    } else if (n.kind == 'i' && n.isz == 8) {
        const int64_t* p = (const int64_t*)n.data.data();
        for (int64_t i = 0; i < n.numel(); i++) out[i] = (float)p[i];
    } else if (n.kind == 'i' && n.isz == 4) {
        const int32_t* p = (const int32_t*)n.data.data();
        for (int64_t i = 0; i < n.numel(); i++) out[i] = (float)p[i];
    } else { std::fprintf(stderr, "unhandled dtype %c%d\n", n.kind, n.isz); std::exit(2); }
    if (n.fortran && n.shape.size() > 1) {
        // Reorder Fortran (column-major) → C (row-major) so the flat layout
        // matches the CUDA channels-/sequence-major buffers being compared.
        int nd = (int)n.shape.size();
        std::vector<int64_t> cst(nd), fst(nd);  // C-order / Fortran-order strides
        cst[nd - 1] = 1; for (int i = nd - 2; i >= 0; i--) cst[i] = cst[i + 1] * n.shape[i + 1];
        fst[0] = 1; for (int i = 1; i < nd; i++) fst[i] = fst[i - 1] * n.shape[i - 1];
        std::vector<float> reord(out.size());
        std::vector<int64_t> idx(nd, 0);
        for (int64_t lin = 0; lin < n.numel(); lin++) {
            int64_t fsrc = 0; for (int i = 0; i < nd; i++) fsrc += idx[i] * fst[i];
            reord[lin] = out[fsrc];
            for (int i = nd - 1; i >= 0; i--) { if (++idx[i] < n.shape[i]) break; idx[i] = 0; }
        }
        out.swap(reord);
    }
    return out;
}

static std::string DIR;
static std::map<std::string, BF*> g_cache;
static BF* upload_bf(const std::vector<float>& h) {
    std::vector<BF> hb(h.size());
    for (size_t i = 0; i < h.size(); i++) hb[i] = __float2bfloat16(h[i]);
    BF* d; HCK(cudaMalloc(&d, hb.size() * sizeof(BF)));
    HCK(cudaMemcpy(d, hb.data(), hb.size() * sizeof(BF), cudaMemcpyHostToDevice));
    return d;
}
static BF* Wbf(const std::string& name) {
    auto it = g_cache.find(name); if (it != g_cache.end()) return it->second;
    BF* d = upload_bf(as_f32(load_npy(DIR + "/weights/" + name + ".npy"))); g_cache[name] = d; return d;
}

// ── cosine / rel_rms report against a named dump ─────────────────────────────
static double cosine_vs(const std::vector<float>& y, const std::string& file) {
    std::vector<float> rp = as_f32(load_npy(DIR + "/" + file));
    long n = (long)std::min(y.size(), rp.size());
    if ((long)y.size() != (long)rp.size())
        std::printf("    [size mismatch: got %zu vs ref %zu]\n", y.size(), rp.size());
    double dn = 0, rn = 0, dot = 0, en = 0;
    for (long i = 0; i < n; i++) { double e = (double)y[i] - rp[i]; en += e * e; dn += (double)y[i] * y[i]; rn += (double)rp[i] * rp[i]; dot += (double)y[i] * rp[i]; }
    double cos = dot / std::sqrt(dn * rn);
    std::printf("    vs %-24s rel_rms=%6.2f%%  cosine=%.5f\n", file.c_str(), 100 * std::sqrt(en / rn), cos);
    return cos;
}

// ── checkpoint hook: copy device bf16 → host f32, compare vs the staged dumps ─
struct CkptState { std::map<std::string, double> cos; };
static void ckpt_cb(const char* name, const BF* dev, long numel, void* user) {
    auto* st = (CkptState*)user;
    std::vector<BF> hb(numel); std::vector<float> y(numel);
    HCK(cudaMemcpy(hb.data(), dev, numel * sizeof(BF), cudaMemcpyDeviceToHost));
    for (long i = 0; i < numel; i++) y[i] = __bfloat162float(hb[i]);
    std::printf("  [%s]  numel=%ld\n", name, numel);
    double c = cosine_vs(y, std::string(name) + ".npy");
    std::string f32 = std::string(name) + "_f32.npy";
    std::ifstream test(DIR + "/" + f32); if (test) cosine_vs(y, f32);
    st->cos[name] = c;
}

int main(int argc, char** argv) {
    DIR = argc > 1 ? argv[1] : "/tmp/mimi_decoder_parity";

    // ── input codes [1, 32, T] (int64) → host i32 [32, T] ───────────────────
    Npy codes_npy = load_npy(DIR + "/input_codes.npy");
    int NCB = (int)codes_npy.shape[codes_npy.shape.size() - 2];
    int T = (int)codes_npy.shape[codes_npy.shape.size() - 1];
    std::vector<float> cf = as_f32(codes_npy);
    std::vector<int32_t> codes(cf.size());
    for (size_t i = 0; i < cf.size(); i++) codes[i] = (int32_t)cf[i];
    std::printf("codes [%d, %d]  ->  expect %d samples\n", NCB, T, T * 2 * 8 * 6 * 5 * 4);

    // ── populate MimiDecoderRawWeights ──────────────────────────────────────
    MimiDecoderRawWeights W;
    W.codebook_dim = 256; W.hidden = 512; W.codebook_size = 2048;
    W.num_codebooks = NCB; W.num_semantic = 1; W.num_filters = 64;
    W.upsampling_ratios = {8, 6, 5, 4};
    W.xf_heads = 8; W.xf_kv_heads = 8; W.xf_head_dim = 64; W.xf_intermediate = 2048;
    W.xf_sliding_window = 250; W.xf_rope_theta = 10000.0f; W.norm_eps = 1e-5f;
    W.sampling_rate = 24000; W.causal = true;

    // RVQ codebook embeds (semantic 0, acoustic 0..30) in codebook order.
    W.codebook_embed.resize(NCB);
    W.codebook_embed[0] = Wbf("quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed");
    for (int i = 1; i < NCB; i++)
        W.codebook_embed[i] = Wbf("quantizer.acoustic_residual_vector_quantizer.layers." +
                                  std::to_string(i - 1) + ".codebook.embed");
    W.semantic_output_proj = Wbf("quantizer.semantic_residual_vector_quantizer.output_proj.weight");
    W.acoustic_output_proj = Wbf("quantizer.acoustic_residual_vector_quantizer.output_proj.weight");

    // upsample: ConvTranspose1d k4 s2 groups=512, no bias.
    W.upsample.w = Wbf("upsample.conv.weight");
    W.upsample.b = nullptr;
    W.upsample.in_ch = 512; W.upsample.out_ch = 512; W.upsample.kernel = 4;
    W.upsample.stride = 2; W.upsample.groups = 512;

    // decoder_transformer: 8 layers.
    for (int l = 0; l < 8; l++) {
        std::string p = "decoder_transformer.layers." + std::to_string(l) + ".";
        MimiXfLayerRaw L;
        L.in_ln_w = Wbf(p + "input_layernorm.weight");
        L.in_ln_b = Wbf(p + "input_layernorm.bias");
        L.q = Wbf(p + "self_attn.q_proj.weight");
        L.k = Wbf(p + "self_attn.k_proj.weight");
        L.v = Wbf(p + "self_attn.v_proj.weight");
        L.o = Wbf(p + "self_attn.o_proj.weight");
        L.attn_scale = Wbf(p + "self_attn_layer_scale.scale");
        L.post_ln_w = Wbf(p + "post_attention_layernorm.weight");
        L.post_ln_b = Wbf(p + "post_attention_layernorm.bias");
        L.fc1 = Wbf(p + "mlp.fc1.weight");
        L.fc2 = Wbf(p + "mlp.fc2.weight");
        L.mlp_scale = Wbf(p + "mlp_layer_scale.scale");
        W.xf_layers.push_back(L);
    }
    W.xf_final_ln_w = nullptr;  // MimiTransformerModel has no final norm
    W.xf_final_ln_b = nullptr;

    // ── SEANet decoder. layer indices (MimiDecoder.layers):
    //   0: Conv1d k7 512→1024
    //   1 ELU, 2 ConvTr 1024→512 k16 s8, 3 Resnet(512)
    //   4 ELU, 5 ConvTr 512→256  k12 s6, 6 Resnet(256)
    //   7 ELU, 8 ConvTr 256→128  k10 s5, 9 Resnet(128)
    //   10 ELU, 11 ConvTr 128→64 k8 s4, 12 Resnet(64)
    //   13 ELU, 14 Conv1d k3 64→1
    auto conv_w = [&](const std::string& base, int in_ch, int out_ch, int k,
                      int stride, int dil) {
        MimiConvRaw c; c.w = Wbf(base + ".weight"); c.b = Wbf(base + ".bias");
        c.in_ch = in_ch; c.out_ch = out_ch; c.kernel = k; c.stride = stride; c.dilation = dil;
        return c;
    };
    auto convt_w = [&](const std::string& base, int in_ch, int out_ch, int k, int stride) {
        MimiConvTRaw c; c.w = Wbf(base + ".weight"); c.b = Wbf(base + ".bias");
        c.in_ch = in_ch; c.out_ch = out_ch; c.kernel = k; c.stride = stride; c.groups = 1;
        return c;
    };

    W.seanet_in = conv_w("decoder.layers.0.conv", 512, 1024, 7, 1, 1);

    struct StageDef { int idx; int in_ch, out_ch, k, stride; };
    StageDef stages[4] = {{2, 1024, 512, 16, 8}, {5, 512, 256, 12, 6},
                          {8, 256, 128, 10, 5}, {11, 128, 64, 8, 4}};
    for (auto& s : stages) {
        MimiDecoderStageRaw st;
        st.convtr = convt_w("decoder.layers." + std::to_string(s.idx) + ".conv",
                            s.in_ch, s.out_ch, s.k, s.stride);
        int rb = s.idx + 1;  // resnet block layer index
        int dim = s.out_ch, hid = dim / 2;  // compress=2
        st.resnet.conv1 = conv_w("decoder.layers." + std::to_string(rb) + ".block.1.conv",
                                 dim, hid, 3, 1, 1);  // k=residual_kernel_size=3, dil=1
        st.resnet.conv2 = conv_w("decoder.layers." + std::to_string(rb) + ".block.3.conv",
                                 hid, dim, 1, 1, 1);  // k=1
        W.seanet_stages.push_back(st);
    }
    W.seanet_out = conv_w("decoder.layers.14.conv", 64, 1, 3, 1, 1);

    // ── run with the staged-checkpoint hook ─────────────────────────────────
    CkptState st;
    set_mimi_decoder_ckpt(ckpt_cb, &st);

    int n_samples = mimi_decoder_num_samples(W, T);
    float* d_out; HCK(cudaMalloc(&d_out, (long)n_samples * sizeof(float)));

    std::printf("\n=== staged checkpoints (bf16-vs-bf16) ===\n");
    int got = run_mimi_decoder(W, codes.data(), T, d_out);
    HCK(cudaDeviceSynchronize());

    std::vector<float> y(got);
    HCK(cudaMemcpy(y.data(), d_out, (long)got * sizeof(float), cudaMemcpyDeviceToHost));

    std::printf("\n=== Mimi decoder OUTPUT WAVEFORM parity (%d samples) ===\n", got);
    double cb = cosine_vs(y, "output_waveform.npy");
    cosine_vs(y, "output_waveform_f32.npy");

    bool pass = cb > 0.99;
    std::printf("\n%s (waveform cosine=%.5f)\n",
                pass ? "MIMI DECODER PARITY PASS" : "MIMI DECODER PARITY (needs work)", cb);
    return pass ? 0 : 1;
}
