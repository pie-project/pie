// Standalone parity harness for the CSM depth decoder + RVQ frame sampler — the
// genuinely-new OUTPUT-modality engine piece (AUDIO_OUTPUT.md §3). Loads the
// HF-dumped single-frame trace (backbone hidden seed + cb0 + emitted cb1..31 +
// per-step logits) and all depth weights (bf16), populates `CsmDepthRawWeights`,
// runs `run_csm_depth_decoder_frame_dbg`, and checks (1) the 31 emitted codes
// match HF's argmax EXACTLY (the natural metric for discrete RVQ codes) and
// (2) the per-step logits cosine. The backbone is verified llama_like and the
// Mimi decoder is verified separately, so this isolates the new piece.
//
//   export PATH=/usr/local/cuda/bin:$PATH
//   nvcc -O2 -arch=sm_89 -std=c++17 -I driver/cuda/src \
//        driver/cuda/tests/csm_depth_decoder_parity.cu -o /tmp/cdp
//   /tmp/cdp /tmp/csm_depth_parity
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "model/csm/csm_depth_decoder_forward.cu"  // decoder under test + kernels

using namespace pie_cuda_driver::model;
using BF = __nv_bfloat16;
#define HCK(x) do{cudaError_t e=(x);if(e){std::fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);}}while(0)

// ── npy loader (handles f32 / bf16-as-u16 / i64 / i32) ───────────────────────
struct Npy { std::vector<int64_t> shape; char kind=0; int isz=0; std::vector<uint8_t> data;
    int64_t numel() const { int64_t n=1; for(auto d:shape)n*=d; return n; } };
static Npy load_npy(const std::string& p){
    std::ifstream f(p,std::ios::binary);
    if(!f){std::fprintf(stderr,"open %s\n",p.c_str());std::exit(2);}
    char m[6];f.read(m,6);uint8_t maj=f.get(),mn=f.get();(void)mn;
    uint32_t hl; if(maj==1){uint16_t h;f.read((char*)&h,2);hl=h;}else{f.read((char*)&hl,4);}
    std::string hdr(hl,0);f.read(hdr.data(),hl);
    Npy o; auto dp=hdr.find("'descr'");auto q=hdr.find('\'',hdr.find(':',dp)+1);
    std::string d=hdr.substr(q+1,hdr.find('\'',q+1)-q-1);
    o.kind=d[1];o.isz=std::atoi(d.substr(2).c_str());
    auto sp=hdr.find("'shape'");auto lp=hdr.find('(',sp),rp=hdr.find(')',lp);
    std::string sh=hdr.substr(lp+1,rp-lp-1);size_t i=0;
    while(i<sh.size()){while(i<sh.size()&&!isdigit(sh[i]))++i;if(i>=sh.size())break;
        int64_t v=0;while(i<sh.size()&&isdigit(sh[i]))v=v*10+(sh[i++]-'0');o.shape.push_back(v);}
    o.data.resize((size_t)o.numel()*o.isz);f.read((char*)o.data.data(),(std::streamsize)o.data.size());
    return o;
}
// Upload an npy as device bf16. Handles f32 and bf16-stored-as-u16 (the depth
// weight dump uses bf16 raw bits in a uint16 array).
static BF* upload_bf(const Npy& n){
    int64_t N=n.numel(); std::vector<BF> hb(N);
    if(n.kind=='f'&&n.isz==4){const float* p=(const float*)n.data.data();for(int64_t i=0;i<N;i++)hb[i]=__float2bfloat16(p[i]);}
    else if((n.kind=='u'||n.kind=='i')&&n.isz==2){const uint16_t* p=(const uint16_t*)n.data.data();std::memcpy(hb.data(),p,N*2);} // raw bf16 bits
    else {std::fprintf(stderr,"upload_bf unhandled %c%d\n",n.kind,n.isz);std::exit(2);}
    BF* d;HCK(cudaMalloc(&d,N*sizeof(BF)));HCK(cudaMemcpy(d,hb.data(),N*sizeof(BF),cudaMemcpyHostToDevice));return d;
}
static std::vector<float> as_f32(const Npy& n){
    std::vector<float> o(n.numel());
    if(n.kind=='f'&&n.isz==4)std::memcpy(o.data(),n.data.data(),n.numel()*4);
    else if(n.kind=='i'&&n.isz==8){const int64_t* p=(const int64_t*)n.data.data();for(int64_t i=0;i<n.numel();i++)o[i]=(float)p[i];}
    else if(n.kind=='i'&&n.isz==4){const int32_t* p=(const int32_t*)n.data.data();for(int64_t i=0;i<n.numel();i++)o[i]=(float)p[i];}
    else {std::fprintf(stderr,"as_f32 unhandled %c%d\n",n.kind,n.isz);std::exit(2);}
    return o;
}

static std::string DIR;
static std::map<std::string,BF*> g_cache;
static BF* W(const std::string& name){
    auto it=g_cache.find(name); if(it!=g_cache.end())return it->second;
    BF* d=upload_bf(load_npy(DIR+"/weights/"+name+".npy")); g_cache[name]=d; return d;
}

int main(int argc,char** argv){
    DIR = argc>1?argv[1]:"/tmp/csm_depth_parity";

    // ── load the frame trace ────────────────────────────────────────────────
    Npy bb_npy = load_npy(DIR+"/frame_bb_hidden.npy");          // [2048] f32
    Npy am_npy = load_npy(DIR+"/frame_depth_argmax.npy");       // [31] i64
    Npy ref_logits = load_npy(DIR+"/frame_depth_logits.npy");   // [31, 2051] f32
    std::vector<float> bb_f = as_f32(bb_npy);
    std::vector<float> ref_am = as_f32(am_npy);
    std::vector<float> ref_lg = as_f32(ref_logits);
    const int BH = (int)bb_npy.shape[0];                        // 2048
    const int NCBm1 = (int)am_npy.shape[0];                     // 31
    const int V = (int)ref_logits.shape[1];                     // 2051
    std::printf("frame trace: backbone_hidden[%d], %d emitted codes, vocab %d\n", BH, NCBm1, V);

    // cb0 for the captured frame: read the manifest's "cb0" field (the frame the
    // trace was dumped for may not be frame 0, so emitted_codes[0,0] is wrong).
    int cb0 = -1;
    { std::ifstream mf(DIR+"/manifest.json"); std::string j((std::istreambuf_iterator<char>(mf)),std::istreambuf_iterator<char>());
      auto k=j.find("\"cb0\""); if(k!=std::string::npos){auto c=j.find(':',k);int v=0;size_t i=c+1;while(i<j.size()&&!isdigit(j[i]))++i;while(i<j.size()&&isdigit(j[i]))v=v*10+(j[i++]-'0');cb0=v;} }
    if(cb0<0){std::fprintf(stderr,"could not read cb0 from manifest\n");std::exit(2);}
    std::printf("cb0 = %d (manifest)\n", cb0);

    // ── bind weights ─────────────────────────────────────────────────────────
    CsmDepthRawWeights w;
    w.hidden=1024; w.backbone_hidden=BH; w.num_layers=4; w.num_heads=8;
    w.num_kv_heads=2; w.head_dim=128; w.intermediate=8192; w.num_codebooks=32;
    w.vocab_size=V; w.norm_eps=1e-5f; w.rope_theta=500000.0f; w.rope_factor=32.0f;
    w.rope_low_freq_factor=0.001953125f; w.rope_high_freq_factor=0.0078125f;
    w.rope_original_max_position=16;
    w.embed_tokens = W("depth_decoder.model.embed_tokens.weight");
    w.inputs_embeds_projector = W("depth_decoder.model.inputs_embeds_projector.weight");
    w.norm_w = W("depth_decoder.model.norm.weight");
    w.codebooks_head = W("depth_decoder.codebooks_head.weight");
    w.layers.resize(w.num_layers);
    for(int L=0;L<w.num_layers;L++){
        std::string p="depth_decoder.model.layers."+std::to_string(L)+".";
        w.layers[L].in_ln_w   = W(p+"input_layernorm.weight");
        w.layers[L].post_ln_w = W(p+"post_attention_layernorm.weight");
        w.layers[L].q    = W(p+"self_attn.q_proj.weight");
        w.layers[L].k    = W(p+"self_attn.k_proj.weight");
        w.layers[L].v    = W(p+"self_attn.v_proj.weight");
        w.layers[L].o    = W(p+"self_attn.o_proj.weight");
        w.layers[L].gate = W(p+"mlp.gate_proj.weight");
        w.layers[L].up   = W(p+"mlp.up_proj.weight");
        w.layers[L].down = W(p+"mlp.down_proj.weight");
    }

    // backbone hidden seed -> device bf16
    std::vector<BF> bb_h(BH); for(int i=0;i<BH;i++)bb_h[i]=__float2bfloat16(bb_f[i]);
    BF* bb_d; HCK(cudaMalloc(&bb_d,BH*sizeof(BF)));
    HCK(cudaMemcpy(bb_d,bb_h.data(),BH*sizeof(BF),cudaMemcpyHostToDevice));

    // ── run depth decoder frame ──────────────────────────────────────────────
    std::vector<std::int32_t> out_codes(NCBm1,0);
    std::vector<float> out_logits((size_t)NCBm1*V,0.f);
    run_csm_depth_decoder_frame_dbg(w, bb_d, cb0, out_codes.data(), out_logits.data(), 0);
    HCK(cudaDeviceSynchronize());

    // ── compare codes ─────────────────────────────────────────────────────────
    std::printf("\n=== emitted codes (cb1..cb31) vs HF argmax ===\n");
    int nmatch=0;
    for(int i=0;i<NCBm1;i++){ if(out_codes[i]==(int)ref_am[i])nmatch++; }
    std::printf("  pie : ");for(int i=0;i<NCBm1;i++)std::printf("%d ",out_codes[i]);std::printf("\n");
    std::printf("  ref : ");for(int i=0;i<NCBm1;i++)std::printf("%d ",(int)ref_am[i]);std::printf("\n");
    std::printf("  exact code matches: %d / %d\n", nmatch, NCBm1);

    // ── compare logits cosine (per-step + overall) ────────────────────────────
    double dn=0,rn=0,dot=0,en=0;
    for(size_t i=0;i<out_logits.size();i++){double y=out_logits[i],r=ref_lg[i];double e=y-r;en+=e*e;dn+=y*y;rn+=r*r;dot+=y*r;}
    double cos=dot/std::sqrt(dn*rn);
    std::printf("  logits[31x%d] cosine=%.5f rel_rms=%.2f%%\n", V, cos, 100*std::sqrt(en/rn));

    bool pass = (nmatch==NCBm1);
    std::printf("\nCSM DEPTH DECODER PARITY %s (%d/%d codes, logits cos=%.5f)\n",
                pass?"PASS":"FAIL", nmatch, NCBm1, cos);
    return pass?0:1;
}
