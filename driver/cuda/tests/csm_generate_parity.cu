// Full CSM generation parity harness: runs csm_generate_audio end-to-end on the
// real (dumped) weights and compares emitted RVQ codes against HF's greedy
// generation (/tmp/csm_depth_parity/emitted_codes.npy). Reproduces the live
// `pie run tts` path standalone so the frame loop can be debugged without the
// server. Loads: backbone .bin (scripts/csm_backbone_dump.py -> /tmp/csm_bb_dump),
// depth npy (/tmp/csm_depth_parity/weights), mimi npy (/tmp/mimi_decoder_parity/
// weights).
//
//   nvcc -O2 -arch=sm_89 -std=c++17 -diag-suppress 550 -I driver/cuda/src \
//        -I driver/common/include driver/cuda/tests/csm_generate_parity.cu \
//        driver/cuda/src/model/csm_depth_decoder_forward.cu \
//        driver/cuda/src/model/mimi_decoder_forward.cu -o /tmp/cgp
//   /tmp/cgp
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "model/csm_backbone_forward.cu"

using namespace pie_cuda_driver::model;
using BF = __nv_bfloat16;
#define HCK(x) do{cudaError_t e=(x);if(e){std::fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);}}while(0)

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
static BF* upload_bf(const Npy& n){
    int64_t N=n.numel(); std::vector<BF> hb(N);
    if(n.kind=='f'&&n.isz==4){const float* p=(const float*)n.data.data();for(int64_t i=0;i<N;i++)hb[i]=__float2bfloat16(p[i]);}
    else if((n.kind=='u'||n.kind=='i')&&n.isz==2){std::memcpy(hb.data(),n.data.data(),N*2);}
    else {std::fprintf(stderr,"upload_bf %c%d\n",n.kind,n.isz);std::exit(2);}
    BF* d;HCK(cudaMalloc(&d,N*sizeof(BF)));HCK(cudaMemcpy(d,hb.data(),N*sizeof(BF),cudaMemcpyHostToDevice));return d;
}
static std::vector<float> as_f32(const Npy& n){
    std::vector<float> o(n.numel());
    if(n.kind=='f'&&n.isz==4)std::memcpy(o.data(),n.data.data(),n.numel()*4);
    else if(n.kind=='i'&&n.isz==8){const int64_t* p=(const int64_t*)n.data.data();for(int64_t i=0;i<n.numel();i++)o[i]=(float)p[i];}
    else {std::fprintf(stderr,"as_f32 %c%d\n",n.kind,n.isz);std::exit(2);}
    return o;
}

static std::string BBDIR="/tmp/csm_bb_dump", DDIR="/tmp/csm_depth_parity", MDIR="/tmp/mimi_decoder_parity";
static BF* load_bin(const std::string& name){
    std::ifstream f(BBDIR+"/"+name+".bin",std::ios::binary);
    if(!f){std::fprintf(stderr,"open %s\n",name.c_str());std::exit(2);}
    f.seekg(0,std::ios::end);long n=f.tellg()/2;f.seekg(0);
    std::vector<uint16_t> h(n);f.read((char*)h.data(),n*2);
    BF* d;HCK(cudaMalloc(&d,n*sizeof(BF)));HCK(cudaMemcpy(d,h.data(),n*sizeof(BF),cudaMemcpyHostToDevice));return d;
}
static BF* Wd(const std::string& nm){ return upload_bf(load_npy(DDIR+"/weights/"+nm+".npy")); }
static BF* Wm(const std::string& nm){ return upload_bf(load_npy(MDIR+"/weights/"+nm+".npy")); }

int main(int argc,char** argv){
    // ── backbone ──────────────────────────────────────────────────────────
    CsmBackboneRawWeights bb;
    bb.embed_text  = load_bin("embed_text_tokens.weight");
    bb.embed_audio = load_bin("backbone_model.embed_tokens.embed_audio_tokens.weight");
    bb.norm_w      = load_bin("backbone_model.norm.weight");
    bb.lm_head     = load_bin("lm_head.weight");
    bb.layers.resize(bb.num_layers);
    for(int i=0;i<bb.num_layers;i++){
        std::string p="backbone_model.layers."+std::to_string(i)+".";
        auto& L=bb.layers[i];
        L.in_ln_w=load_bin(p+"input_layernorm.weight"); L.post_ln_w=load_bin(p+"post_attention_layernorm.weight");
        L.q=load_bin(p+"self_attn.q_proj.weight"); L.k=load_bin(p+"self_attn.k_proj.weight");
        L.v=load_bin(p+"self_attn.v_proj.weight"); L.o=load_bin(p+"self_attn.o_proj.weight");
        L.gate=load_bin(p+"mlp.gate_proj.weight"); L.up=load_bin(p+"mlp.up_proj.weight"); L.down=load_bin(p+"mlp.down_proj.weight");
    }
    // ── depth ─────────────────────────────────────────────────────────────
    CsmDepthRawWeights dp;
    dp.embed_tokens=Wd("depth_decoder.model.embed_tokens.weight");
    dp.inputs_embeds_projector=Wd("depth_decoder.model.inputs_embeds_projector.weight");
    dp.norm_w=Wd("depth_decoder.model.norm.weight");
    dp.codebooks_head=Wd("depth_decoder.codebooks_head.weight");
    dp.layers.resize(dp.num_layers);
    for(int L=0;L<dp.num_layers;L++){
        std::string p="depth_decoder.model.layers."+std::to_string(L)+".";
        dp.layers[L].in_ln_w=Wd(p+"input_layernorm.weight"); dp.layers[L].post_ln_w=Wd(p+"post_attention_layernorm.weight");
        dp.layers[L].q=Wd(p+"self_attn.q_proj.weight"); dp.layers[L].k=Wd(p+"self_attn.k_proj.weight");
        dp.layers[L].v=Wd(p+"self_attn.v_proj.weight"); dp.layers[L].o=Wd(p+"self_attn.o_proj.weight");
        dp.layers[L].gate=Wd(p+"mlp.gate_proj.weight"); dp.layers[L].up=Wd(p+"mlp.up_proj.weight"); dp.layers[L].down=Wd(p+"mlp.down_proj.weight");
    }
    // ── mimi ──────────────────────────────────────────────────────────────
    MimiDecoderRawWeights mm;
    mm.codebook_dim=256; mm.hidden=512; mm.codebook_size=2048; mm.num_codebooks=32; mm.num_semantic=1;
    mm.num_filters=64; mm.upsampling_ratios={8,6,5,4}; mm.xf_heads=8; mm.xf_kv_heads=8; mm.xf_head_dim=64;
    mm.xf_intermediate=2048; mm.xf_sliding_window=250; mm.xf_rope_theta=10000.0f; mm.norm_eps=1e-5f;
    mm.sampling_rate=24000; mm.causal=true;
    mm.codebook_embed.resize(32);
    mm.codebook_embed[0]=Wm("quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed");
    for(int i=1;i<32;i++) mm.codebook_embed[i]=Wm("quantizer.acoustic_residual_vector_quantizer.layers."+std::to_string(i-1)+".codebook.embed");
    mm.semantic_output_proj=Wm("quantizer.semantic_residual_vector_quantizer.output_proj.weight");
    mm.acoustic_output_proj=Wm("quantizer.acoustic_residual_vector_quantizer.output_proj.weight");
    mm.upsample.w=Wm("upsample.conv.weight"); mm.upsample.b=nullptr; mm.upsample.in_ch=512; mm.upsample.out_ch=512;
    mm.upsample.kernel=4; mm.upsample.stride=2; mm.upsample.groups=512;
    for(int l=0;l<8;l++){ std::string p="decoder_transformer.layers."+std::to_string(l)+".";
        MimiXfLayerRaw L; L.in_ln_w=Wm(p+"input_layernorm.weight"); L.in_ln_b=Wm(p+"input_layernorm.bias");
        L.q=Wm(p+"self_attn.q_proj.weight"); L.k=Wm(p+"self_attn.k_proj.weight"); L.v=Wm(p+"self_attn.v_proj.weight"); L.o=Wm(p+"self_attn.o_proj.weight");
        L.attn_scale=Wm(p+"self_attn_layer_scale.scale"); L.post_ln_w=Wm(p+"post_attention_layernorm.weight"); L.post_ln_b=Wm(p+"post_attention_layernorm.bias");
        L.fc1=Wm(p+"mlp.fc1.weight"); L.fc2=Wm(p+"mlp.fc2.weight"); L.mlp_scale=Wm(p+"mlp_layer_scale.scale"); mm.xf_layers.push_back(L); }
    mm.xf_final_ln_w=nullptr; mm.xf_final_ln_b=nullptr;
    auto conv=[&](const std::string& b,int in,int oc,int k,int st,int dl){MimiConvRaw c;c.w=Wm(b+".weight");c.b=Wm(b+".bias");c.in_ch=in;c.out_ch=oc;c.kernel=k;c.stride=st;c.dilation=dl;return c;};
    auto convt=[&](const std::string& b,int in,int oc,int k,int st){MimiConvTRaw c;c.w=Wm(b+".weight");c.b=Wm(b+".bias");c.in_ch=in;c.out_ch=oc;c.kernel=k;c.stride=st;c.groups=1;return c;};
    mm.seanet_in=conv("decoder.layers.0.conv",512,1024,7,1,1);
    struct SD{int idx,in,oc,k,st;}; SD st[4]={{2,1024,512,16,8},{5,512,256,12,6},{8,256,128,10,5},{11,128,64,8,4}};
    for(auto& s:st){MimiDecoderStageRaw o;o.convtr=convt("decoder.layers."+std::to_string(s.idx)+".conv",s.in,s.oc,s.k,s.st);
        int rb=s.idx+1,dim=s.oc,hid=dim/2; o.resnet.conv1=conv("decoder.layers."+std::to_string(rb)+".block.1.conv",dim,hid,3,1,1);
        o.resnet.conv2=conv("decoder.layers."+std::to_string(rb)+".block.3.conv",hid,dim,1,1,1); mm.seanet_stages.push_back(o);}
    mm.seanet_out=conv("decoder.layers.14.conv",64,1,3,1,1);

    // ── prompt ──────────────────────────────────────────────────────────────
    std::vector<int32_t> prompt={128000,58,15,60,9906,11,420,374,264,1296,13,128001};
    int max_frames = argc>1?std::atoi(argv[1]):64;

    std::vector<float> pcm; std::vector<int32_t> codes;
    int nf = csm_generate_audio(bb,dp,mm,prompt.data(),(int)prompt.size(),max_frames,pcm,&codes,0);
    std::printf("generated %d frames, %zu pcm samples (%.2fs)\n", nf, pcm.size(), pcm.size()/24000.0f);

    // compare against HF emitted_codes (frame-major codes vs HF codebook-major)
    Npy ref=load_npy(DDIR+"/emitted_codes.npy"); // [32, Tref]
    int Tref=(int)ref.shape[1]; auto rf=as_f32(ref);
    std::printf("HF reference: %d frames\n", Tref);
    int cmp=nf<Tref?nf:Tref; int match=0,tot=0;
    for(int f=0; f<cmp; f++) for(int c=0;c<32;c++){ int mine=codes[(long)f*32+c]; int hf=(int)rf[(long)c*Tref+f]; tot++; if(mine==hf)match++; }
    std::printf("code match (first %d frames): %d/%d (%.1f%%)\n", cmp, match, tot, 100.0*match/tot);
    for(int f=0;f<3&&f<nf;f++){ std::printf("my frame%d:",f); for(int c=0;c<12;c++)std::printf(" %d",codes[(long)f*32+c]); std::printf("\n"); }
    std::printf("my cb0 per frame:"); for(int f=0;f<10&&f<nf;f++)std::printf(" %d",codes[(long)f*32]); std::printf("\n");
    std::printf("HF cb0 per frame:  420 1848 1340 1914 1834 1706 1706 1693 1786 1277\n");
    for(int f=0;f<3&&f<nf;f++){ int mt=0; for(int c=0;c<32;c++) if(codes[(long)f*32+c]==(int)rf[(long)c*Tref+f]) mt++;
        std::printf("frame %d: %d/32 codes match HF\n", f, mt); }
    return 0;
}
