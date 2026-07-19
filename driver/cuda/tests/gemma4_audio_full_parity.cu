// Standalone parity harness for the Gemma-4 audio (USM/Conformer) encoder.
//
// Feeds the HF-dumped log-mel features into `run_gemma4_audio(...)` and checks
// the projected [out_len, 2560] audio soft tokens against the reference dump
// (rel_rms + cosine, bf16-vs-bf16 being the real metric — MULTIMODAL.md §11).
//
//   nvcc -O2 -arch=sm_89 -std=c++17 -I ../src gemma4_audio_full_parity.cu -o /tmp/qap
//   /tmp/qap /tmp/gemma4_audio_parity
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "model/gemma4/gemma4_audio_forward.cu"  // encoder under test + kernels + .hpp

using pie_cuda_driver::model::AudioClipRaw;
using pie_cuda_driver::model::AudioFfnRaw;
using pie_cuda_driver::model::AudioLayerRaw;
using pie_cuda_driver::model::AudioRawWeights;
using pie_cuda_driver::model::run_gemma4_audio;
using pie_cuda_driver::model::set_gemma4_audio_ckpt;
using pie_cuda_driver::model::gemma4_audio_subsampled_len;
using BF = __nv_bfloat16;

#define HCK(x) do{cudaError_t e=(x);if(e){std::fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);}}while(0)

struct Npy {
    std::vector<int64_t> shape; char kind = 0; int isz = 0; std::vector<uint8_t> data;
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
    else if (n.kind == 'f' && n.isz == 2) {
        const uint16_t* p = (const uint16_t*)n.data.data();
        for (int64_t i = 0; i < n.numel(); i++) { uint32_t b = (uint32_t)p[i] << 16; float v; std::memcpy(&v, &b, 4); out[i] = v; }
    } else if (n.kind == 'i' && n.isz == 8) {
        const int64_t* p = (const int64_t*)n.data.data();
        for (int64_t i = 0; i < n.numel(); i++) out[i] = (float)p[i];
    } else { std::fprintf(stderr, "unhandled dtype %c%d\n", n.kind, n.isz); std::exit(2); }
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
static BF* scal_bf(const std::string& name) { return Wbf(name); }  // 1-elem npy → 1 bf16
static AudioClipRaw clip(const std::string& b) {
    AudioClipRaw c;
    c.w = Wbf(b + ".linear.weight");
    c.imin = scal_bf(b + ".input_min"); c.imax = scal_bf(b + ".input_max");
    c.omin = scal_bf(b + ".output_min"); c.omax = scal_bf(b + ".output_max");
    return c;
}
static AudioFfnRaw ffn(const std::string& b) {
    AudioFfnRaw f;
    f.pre_ln = Wbf(b + ".pre_layer_norm.weight"); f.post_ln = Wbf(b + ".post_layer_norm.weight");
    f.fc1 = clip(b + ".ffw_layer_1"); f.fc2 = clip(b + ".ffw_layer_2");
    return f;
}

static double report(const char* tag, const std::vector<float>& y, const std::string& file) {
    std::vector<float> rp = as_f32(load_npy(DIR + "/" + file));
    long n = (long)y.size(); double dn = 0, rn = 0, dot = 0, en = 0;
    for (long i = 0; i < n; i++) { double e = (double)y[i] - rp[i]; en += e * e; dn += (double)y[i] * y[i]; rn += (double)rp[i] * rp[i]; dot += (double)y[i] * rp[i]; }
    std::printf("  vs %-16s rel_rms_err=%.3f%%  cosine=%.5f\n", tag, 100 * std::sqrt(en / rn), dot / std::sqrt(dn * rn));
    return dot / std::sqrt(dn * rn);
}

// ── per-stage checkpoint hook: device bf16 → host f32, cosine vs the dumps ────
// Compares against the bf16 `.npy` (the real metric) and, if present, `_f32.npy`.
static void ckpt_cb(const char* name, const BF* dev, long numel, void* /*user*/) {
    std::vector<BF> hb(numel); std::vector<float> y(numel);
    HCK(cudaMemcpy(hb.data(), dev, numel * sizeof(BF), cudaMemcpyDeviceToHost));
    for (long i = 0; i < numel; i++) y[i] = __bfloat162float(hb[i]);
    std::ifstream test(DIR + "/" + name + ".npy");
    if (test) report(name, y, std::string(name) + ".npy");
}

int main(int argc, char** argv) {
    DIR = argc > 1 ? argv[1] : "/tmp/gemma4_audio_parity";
    const int N_MEL = 128, TXT = 2560;

    Npy feat = load_npy(DIR + "/input_features_f32.npy");
    int n_frames = (int)feat.shape[feat.shape.size() - 2];
    int n_mel = (int)feat.shape[feat.shape.size() - 1];
    int out_len = gemma4_audio_subsampled_len(n_frames);
    std::printf("log-mel [%d, %d]  ->  %d audio tokens\n", n_frames, n_mel, out_len);

    std::vector<float> fh = as_f32(feat);
    float* d_feat; HCK(cudaMalloc(&d_feat, fh.size() * 4));
    HCK(cudaMemcpy(d_feat, fh.data(), fh.size() * 4, cudaMemcpyHostToDevice));

    AudioRawWeights W;
    W.sscp0_conv = Wbf("audio.subsample_conv_projection.layer0.conv.weight");
    W.sscp0_norm = Wbf("audio.subsample_conv_projection.layer0.norm.weight");
    W.sscp1_conv = Wbf("audio.subsample_conv_projection.layer1.conv.weight");
    W.sscp1_norm = Wbf("audio.subsample_conv_projection.layer1.norm.weight");
    W.sscp_input_proj = Wbf("audio.subsample_conv_projection.input_proj_linear.weight");
    for (int l = 0; l < 12; l++) {
        std::string p = "audio.layers." + std::to_string(l) + ".";
        AudioLayerRaw L;
        L.ff1 = ffn(p + "feed_forward1"); L.ff2 = ffn(p + "feed_forward2");
        L.norm_pre_attn = Wbf(p + "norm_pre_attn.weight"); L.norm_post_attn = Wbf(p + "norm_post_attn.weight");
        L.q = clip(p + "self_attn.q_proj"); L.k = clip(p + "self_attn.k_proj");
        L.v = clip(p + "self_attn.v_proj"); L.post = clip(p + "self_attn.post");
        L.relative_k = Wbf(p + "self_attn.relative_k_proj.weight");
        L.per_dim_scale = Wbf(p + "self_attn.per_dim_scale");
        L.lconv_pre_ln = Wbf(p + "lconv1d.pre_layer_norm.weight");
        L.lconv_conv_norm = Wbf(p + "lconv1d.conv_norm.weight");
        L.lconv_start = clip(p + "lconv1d.linear_start"); L.lconv_end = clip(p + "lconv1d.linear_end");
        L.depthwise_conv = Wbf(p + "lconv1d.depthwise_conv1d.weight");
        L.norm_out = Wbf(p + "norm_out.weight");
        W.layers.push_back(L);
    }
    W.output_proj_w = Wbf("audio.output_proj.weight");
    W.output_proj_b = Wbf("audio.output_proj.bias");
    W.embed_proj = Wbf("embed.embedding_projection.weight");

    BF* d_out; HCK(cudaMalloc(&d_out, (long)out_len * TXT * sizeof(BF)));
    std::printf("=== staged checkpoints (bf16-vs-bf16) ===\n");
    set_gemma4_audio_ckpt(ckpt_cb, nullptr);
    run_gemma4_audio(W, d_feat, n_frames, n_mel, out_len, d_out);
    set_gemma4_audio_ckpt(nullptr, nullptr);
    HCK(cudaDeviceSynchronize());

    long n = (long)out_len * TXT; std::vector<BF> hb(n); std::vector<float> y(n);
    HCK(cudaMemcpy(hb.data(), d_out, n * sizeof(BF), cudaMemcpyDeviceToHost));
    for (long i = 0; i < n; i++) y[i] = __bfloat162float(hb[i]);

    std::printf("=== Gemma-4 audio encoder parity ===\n");
    double cb = report("HF-bf16", y, "projected.npy");
    report("HF-fp32", y, "projected_f32.npy");
    bool pass = cb > 0.99;
    std::printf("%s (cosine=%.5f)\n", pass ? "AUDIO PARITY PASS" : "AUDIO PARITY (needs work)", cb);
    return pass ? 0 : 1;
}
