// Standalone parity harness for the CSM backbone forward (the OUTER loop of the
// audio-output generation). Loads the dumped backbone weights (bf16 .bin from
// scripts/csm_backbone_dump.py) + the prompt token ids, runs the backbone
// prefill, and prints the codebook-0 argmax for frame 0 (HF reference = 420).
// This isolates the backbone forward from the (separately verified) depth
// decoder + Mimi decoder.
//
//   export PATH=/usr/local/cuda/bin:$PATH
//   nvcc -O2 -arch=sm_89 -std=c++17 -I driver/cuda/src \
//        driver/cuda/tests/csm_backbone_parity.cu -o /tmp/cbp
//   /tmp/cbp /tmp/csm_bb_dump
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "model/csm/csm_backbone_forward.cu"  // backbone forward + kernels under test

using namespace pie_cuda_driver::model;
using BF = __nv_bfloat16;
#define HCK(x) do{cudaError_t e=(x);if(e){std::fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);}}while(0)

static std::string DIR;
// Load a raw little-endian bf16 .bin (uint16 bits) onto the device.
static BF* load_bin(const std::string& name, long expect = -1) {
    std::ifstream f(DIR + "/" + name + ".bin", std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s\n", name.c_str()); std::exit(2); }
    f.seekg(0, std::ios::end); long n = f.tellg() / 2; f.seekg(0);
    if (expect > 0 && n != expect) std::fprintf(stderr, "WARN %s numel %ld != %ld\n", name.c_str(), n, expect);
    std::vector<uint16_t> h(n); f.read((char*)h.data(), n * 2);
    BF* d; HCK(cudaMalloc(&d, n * sizeof(BF)));
    HCK(cudaMemcpy(d, h.data(), n * sizeof(BF), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    DIR = argc > 1 ? argv[1] : "/tmp/csm_bb_dump";

    CsmBackboneRawWeights w;  // defaults = eustlb/csm-1b backbone
    w.embed_text  = load_bin("embed_text_tokens.weight");
    w.embed_audio = load_bin("backbone_model.embed_tokens.embed_audio_tokens.weight");
    w.norm_w      = load_bin("backbone_model.norm.weight");
    w.lm_head     = load_bin("lm_head.weight");
    w.layers.resize(w.num_layers);
    for (int i = 0; i < w.num_layers; ++i) {
        std::string p = "backbone_model.layers." + std::to_string(i) + ".";
        auto& L = w.layers[i];
        L.in_ln_w   = load_bin(p + "input_layernorm.weight");
        L.post_ln_w = load_bin(p + "post_attention_layernorm.weight");
        L.q = load_bin(p + "self_attn.q_proj.weight");
        L.k = load_bin(p + "self_attn.k_proj.weight");
        L.v = load_bin(p + "self_attn.v_proj.weight");
        L.o = load_bin(p + "self_attn.o_proj.weight");
        L.gate = load_bin(p + "mlp.gate_proj.weight");
        L.up   = load_bin(p + "mlp.up_proj.weight");
        L.down = load_bin(p + "mlp.down_proj.weight");
    }

    // Prompt ids (hardcoded from scripts/csm_backbone_dump.py output).
    std::vector<int32_t> prompt = {128000, 58, 15, 60, 9906, 11, 420, 374, 264, 1296, 13, 128001};
    if (argc > 2) {  // allow override: pass ids as extra args
        prompt.clear();
        for (int i = 2; i < argc; ++i) prompt.push_back(std::atoi(argv[i]));
    }
    int n_prompt = (int)prompt.size();
    std::printf("prompt %d tokens\n", n_prompt);

    // ── Run the backbone prefill manually (mirror csm_generate_audio prefill) ─
    const int H = w.hidden, NH = w.num_heads, KV = w.num_kv_heads, hd = w.head_dim;
    const int QD = NH * hd, KD = KV * hd, AV = w.audio_vocab;
    const int maxL = n_prompt + 4;
    auto MAL = [&](long n){ BF* d; HCK(cudaMalloc(&d, n * sizeof(BF))); return d; };
    BBScratch s; s.hidden=H; s.NH=NH; s.KV=KV; s.hd=hd; s.QD=QD; s.KD=KD; s.inter=w.intermediate; s.maxL=maxL; s.S=0;
    int R0 = n_prompt;
    s.resid=MAL((long)R0*H); s.normed=MAL((long)R0*H);
    s.q=MAL((long)R0*QD); s.k=MAL((long)R0*KD); s.v=MAL((long)R0*KD);
    s.attn=MAL((long)R0*QD); s.attn_o=MAL((long)R0*H);
    s.gate=MAL((long)R0*s.inter); s.up=MAL((long)R0*s.inter); s.mlp=MAL((long)R0*H);
    s.kcache.resize(w.num_layers); s.vcache.resize(w.num_layers);
    for (int l=0;l<w.num_layers;l++){ s.kcache[l]=MAL((long)maxL*KD); s.vcache[l]=MAL((long)maxL*KD); }
    for (int r=0;r<R0;r++){
        long row=(long)prompt[r];
        HCK(cudaMemcpy(s.resid+(long)r*H, w.embed_text+row*H, (long)H*sizeof(BF), cudaMemcpyDeviceToDevice));
    }
    int Lkv=R0;
    for (int l=0;l<w.num_layers;l++) bb_layer(w, w.layers[l], s, l, R0, Lkv);
    BF* last_hidden=MAL(H);
    k_rms<<<1,256>>>(s.resid+(long)(R0-1)*H, w.norm_w, last_hidden, 1, H, w.norm_eps);
    BF* lm_logits=MAL(AV);
    k_matmul<<<G2(AV,1),B2>>>(last_hidden, w.lm_head, lm_logits, 1, H, AV);
    int* d_arg; HCK(cudaMalloc(&d_arg, sizeof(int)));
    k_argmax<<<1,256>>>(lm_logits, AV, d_arg);
    int cb0; HCK(cudaMemcpy(&cb0, d_arg, sizeof(int), cudaMemcpyDeviceToHost));
    HCK(cudaDeviceSynchronize());

    // print top-5 logits for diagnostics
    std::vector<BF> hl(AV); HCK(cudaMemcpy(hl.data(), lm_logits, AV*sizeof(BF), cudaMemcpyDeviceToHost));
    float mx=-1e30f; int mi=0; for(int v=0;v<AV;v++){float x=__bfloat162float(hl[v]); if(x>mx){mx=x;mi=v;}}
    std::printf("frame0 cb0 argmax = %d (logit %.4f)   [HF reference = 420]\n", cb0, mx);
    std::printf("%s\n", cb0==420 ? "PASS (cb0 matches HF)" : "MISMATCH");
    return cb0==420 ? 0 : 1;
}
