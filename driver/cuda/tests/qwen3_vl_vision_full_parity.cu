// Standalone parity harness for the Qwen3-VL vision encoder.
//
// Feeds the HF-dumped pixel_values + precomputed side-inputs (pos_embed_interp,
// vision_rope_position_ids) into `run_qwen3vl_vision(...)` and checks the main
// merged output + the 3 DeepStack outputs against the reference dumps
// (rel_rms + cosine, the bf16-appropriate metric — see MULTIMODAL.md §11).
//
// Isolates the encoder *kernels* from the host-side interp/rope helpers (those
// inputs come straight from the dump), so a mismatch localizes to the block
// math / merger, not the geometry.
//
//   nvcc -O2 -arch=sm_89 -std=c++17 -I ../src qwen3_vl_vision_full_parity.cu -o /tmp/qvp
//   /tmp/qvp /tmp/qwen3_vl_vision_parity
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// Pull in the encoder under test (kernels + run_qwen3vl_vision + the .hpp).
#include "model/qwen3_vl/qwen3_vl_vision_forward.cu"

using pie_cuda_driver::model::QVisLinear;
using pie_cuda_driver::model::QVisLayerNorm;
using pie_cuda_driver::model::QVisBlock;
using pie_cuda_driver::model::QVisMerger;
using pie_cuda_driver::model::QwenVisRawWeights;
using pie_cuda_driver::model::run_qwen3vl_vision;
using pie_cuda_driver::model::set_qwen3vl_vision_ckpt;
using BF = __nv_bfloat16;

#define HCK(x) do{cudaError_t e=(x);if(e){std::fprintf(stderr,"cuda %s @%d\n",cudaGetErrorString(e),__LINE__);std::exit(2);}}while(0)

// ── minimal .npy reader (header parse + raw bytes; mirrors gemma harness) ──
struct Npy {
    std::vector<int64_t> shape;
    char kind = 0;   // 'f','i','u'
    int isz = 0;     // itemsize bytes
    std::vector<uint8_t> data;
    int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
};
static Npy load_npy(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s\n", p.c_str()); std::exit(2); }
    char m[6]; f.read(m, 6);
    uint8_t maj = f.get(), mn = f.get(); (void)mn;
    uint32_t hl;
    if (maj == 1) { uint16_t h; f.read((char*)&h, 2); hl = h; }
    else { f.read((char*)&hl, 4); }
    std::string hdr(hl, 0); f.read(hdr.data(), hl);
    Npy o;
    auto dp = hdr.find("'descr'");
    auto q = hdr.find('\'', hdr.find(':', dp) + 1);
    std::string d = hdr.substr(q + 1, hdr.find('\'', q + 1) - q - 1);
    o.kind = d[1]; o.isz = std::atoi(d.substr(2).c_str());
    auto sp = hdr.find("'shape'");
    auto lp = hdr.find('(', sp), rp = hdr.find(')', lp);
    std::string sh = hdr.substr(lp + 1, rp - lp - 1);
    size_t i = 0;
    while (i < sh.size()) {
        while (i < sh.size() && !isdigit(sh[i])) ++i;
        if (i >= sh.size()) break;
        int64_t v = 0;
        while (i < sh.size() && isdigit(sh[i])) v = v * 10 + (sh[i++] - '0');
        o.shape.push_back(v);
    }
    o.data.resize((size_t)o.numel() * o.isz);
    f.read((char*)o.data.data(), (std::streamsize)o.data.size());
    return o;
}

// Read an npy as f32 (accepts f4; converts i8 → f32).
static std::vector<float> as_f32(const Npy& n) {
    std::vector<float> out(n.numel());
    if (n.kind == 'f' && n.isz == 4) {
        std::memcpy(out.data(), n.data.data(), n.numel() * 4);
    } else if (n.kind == 'i' && n.isz == 8) {
        const int64_t* p = (const int64_t*)n.data.data();
        for (int64_t i = 0; i < n.numel(); i++) out[i] = (float)p[i];
    } else if (n.kind == 'f' && n.isz == 2) {
        const uint16_t* p = (const uint16_t*)n.data.data();
        for (int64_t i = 0; i < n.numel(); i++) {
            // bf16 → f32 (top 16 bits)
            uint32_t bits = (uint32_t)p[i] << 16; float v; std::memcpy(&v, &bits, 4); out[i] = v;
        }
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
static float* upload_f32(const std::vector<float>& h) {
    float* d; HCK(cudaMalloc(&d, h.size() * 4));
    HCK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}
// Weight loader: weights/<name>.npy (f32) → bf16 device, cached.
static BF* Wbf(const std::string& name) {
    auto it = g_cache.find(name); if (it != g_cache.end()) return it->second;
    Npy n = load_npy(DIR + "/weights/" + name + ".npy");
    BF* d = upload_bf(as_f32(n)); g_cache[name] = d; return d;
}
static QVisLinear lin(const std::string& base) { // base.weight / base.bias
    QVisLinear l; l.w = Wbf(base + ".weight"); l.b = Wbf(base + ".bias"); return l;
}
static QVisLayerNorm ln(const std::string& base) {
    QVisLayerNorm n; n.g = Wbf(base + ".weight"); n.b = Wbf(base + ".bias"); return n;
}

// Compare a host f32 buffer vs a reference dump (rel_rms + cosine).
static double report(const char* tag, const std::vector<float>& y, const std::string& file) {
    Npy r = load_npy(DIR + "/" + file);
    std::vector<float> rp = as_f32(r);
    long n = (long)y.size();
    double dn = 0, rn = 0, dot = 0, en = 0;
    for (long i = 0; i < n; i++) {
        double e = (double)y[i] - rp[i]; en += e * e;
        dn += (double)y[i] * y[i]; rn += (double)rp[i] * rp[i]; dot += (double)y[i] * rp[i];
    }
    double rel = std::sqrt(en / rn), cos = dot / std::sqrt(dn * rn);
    std::printf("  vs %-22s rel_rms_err=%.3f%%  cosine=%.5f\n", tag, 100 * rel, cos);
    return cos;
}

// Per-row cosine histogram: localizes whether error is uniform or concentrated.
static void rowwise(const std::vector<float>& y, const std::string& file, int rows) {
    Npy r = load_npy(DIR + "/" + file); std::vector<float> rp = as_f32(r);
    int cols = (int)(y.size() / rows);
    int worst = -1; double worstc = 2; int n_bad = 0;
    for (int i = 0; i < rows; i++) {
        double dn = 0, rn = 0, dot = 0;
        for (int j = 0; j < cols; j++) {
            double a = y[(long)i*cols+j], b = rp[(long)i*cols+j];
            dn += a*a; rn += b*b; dot += a*b;
        }
        double c = dot / std::sqrt(dn*rn);
        if (c < worstc) { worstc = c; worst = i; }
        if (c < 0.99) n_bad++;
    }
    std::printf("    rowwise: %d/%d rows cos<0.99, worst row %d cos=%.5f\n",
                n_bad, rows, worst, worstc);
    if (worst >= 0 && getenv("DUMPROW")) {
        std::printf("    worst row %d  mine vs ref (first 12 + argmax-abs):\n", worst);
        int amax=0; double mav=0;
        for (int j=0;j<cols;j++){double a=std::abs((double)rp[(long)worst*cols+j]); if(a>mav){mav=a;amax=j;}}
        for (int j=0;j<12;j++)
            std::printf("      [%4d] mine=%+.4f ref=%+.4f\n", j, y[(long)worst*cols+j], rp[(long)worst*cols+j]);
        std::printf("      argmax-abs ref [%4d] mine=%+.4f ref=%+.4f\n",
                    amax, y[(long)worst*cols+amax], rp[(long)worst*cols+amax]);
    }
}

// ── per-layer checkpoint hook: compare hidden vs the staged dumps (bf16) ──
static int g_rows = 1024;
static void qvis_ckpt_cb(const char* name, const BF* dev, long numel, void* /*user*/) {
    std::vector<BF> hb(numel); std::vector<float> y(numel);
    HCK(cudaMemcpy(hb.data(), dev, numel * sizeof(BF), cudaMemcpyDeviceToHost));
    for (long i = 0; i < numel; i++) y[i] = __bfloat162float(hb[i]);
    int cols = (int)(numel / g_rows);
    if (getenv("TRACECELL")) {
        // row 978 dim 201, row 18 dim 62 (known massive-activation cells).
        std::printf("[trace %s] r978d201=%+.3f r18d62=%+.3f\n", name,
                    y[(long)978*cols+201], y[(long)18*cols+62]);
        return;
    }
    std::printf("[ckpt %s]  numel=%ld\n", name, numel);
    std::ifstream chk(DIR + "/" + std::string(name) + ".npy"); if (!chk) return;
    char tag[48]; std::snprintf(tag, sizeof(tag), "%s HF-bf16", name);
    report(tag, y, std::string(name) + ".npy");
    rowwise(y, std::string(name) + ".npy", g_rows);
    std::string f32 = std::string(name) + "_f32.npy";
    std::ifstream test(DIR + "/" + f32);
    if (test) { std::snprintf(tag, sizeof(tag), "%s HF-fp32", name); report(tag, y, f32); }
}

int main(int argc, char** argv) {
    DIR = argc > 1 ? argv[1] : "/tmp/qwen3_vl_vision_parity";
    const int HID = 1024, OUT = 2048, DEPTH = 24, NDEEP = 3;

    // grid (t,h,w) in patch units → n_patch, n_token.
    Npy gthw = load_npy(DIR + "/input_grid_thw.npy");
    auto gf = as_f32(gthw);
    int gt = (int)gf[0], gh = (int)gf[1], gw = (int)gf[2];
    int n_patch = gt * gh * gw;
    int n_token = n_patch / 4; // spatial_merge 2×2
    std::printf("grid (t,h,w)=(%d,%d,%d)  n_patch=%d  n_token=%d\n", gt, gh, gw, n_patch, n_token);

    // Inputs (all in the HF patch order; pos_embed_interp + rope come from the dump).
    BF* d_pix = upload_bf(as_f32(load_npy(DIR + "/input_pixel_values_f32.npy")));
    BF* d_pe  = upload_bf(as_f32(load_npy(DIR + "/pos_embed_interp.npy")));
    float* d_rope = upload_f32(as_f32(load_npy(DIR + "/vision_rope_position_ids.npy")));

    // Weights.
    QwenVisRawWeights W;
    W.patch = lin("vision.patch_embed.proj");
    W.pos_embed = Wbf("vision.pos_embed.weight");
    for (int l = 0; l < DEPTH; l++) {
        std::string p = "vision.blocks." + std::to_string(l) + ".";
        QVisBlock b;
        b.norm1 = ln(p + "norm1"); b.norm2 = ln(p + "norm2");
        b.qkv = lin(p + "attn.qkv"); b.o = lin(p + "attn.proj");
        b.fc1 = lin(p + "mlp.linear_fc1"); b.fc2 = lin(p + "mlp.linear_fc2");
        W.blocks.push_back(b);
    }
    W.merger.norm = ln("vision.merger.norm");
    W.merger.fc1 = lin("vision.merger.linear_fc1");
    W.merger.fc2 = lin("vision.merger.linear_fc2");
    W.merger.is_postshuffle = false;
    int didx[NDEEP] = {5, 11, 17};
    for (int d = 0; d < NDEEP; d++) {
        std::string p = "vision.deepstack_merger_list." + std::to_string(d) + ".";
        QVisMerger m;
        m.norm = ln(p + "norm"); m.fc1 = lin(p + "linear_fc1"); m.fc2 = lin(p + "linear_fc2");
        m.is_postshuffle = true;
        W.deepstack.push_back(m);
        W.deepstack_layer_idx.push_back(didx[d]);
    }

    // Outputs.
    BF* d_main; HCK(cudaMalloc(&d_main, (long)n_token * OUT * sizeof(BF)));
    std::vector<BF*> d_deep(NDEEP);
    for (int d = 0; d < NDEEP; d++) HCK(cudaMalloc(&d_deep[d], (long)n_token * OUT * sizeof(BF)));
    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 2;
    }

    auto fetch = [&](BF* d) {
        long n = (long)n_token * OUT; std::vector<BF> hb(n); std::vector<float> y(n);
        HCK(cudaMemcpy(hb.data(), d, n * sizeof(BF), cudaMemcpyDeviceToHost));
        for (long i = 0; i < n; i++) y[i] = __bfloat162float(hb[i]);
        return y;
    };

    // Diagnostic: run the main merger on HF's *own* last_hidden dump to isolate
    // the merger from the upstream bf16 trajectory drift. If this is ~1.0, the
    // merger is faithful and the merged-vs-HF gap is purely upstream accumulation.
    if (getenv("MERGER_ONLY")) {
        const char* src = getenv("MERGER_ONLY");  // "bf16" or "f32"
        std::string lh = std::string("last_hidden") + (std::strcmp(src,"f32")==0?"_f32":"") + ".npy";
        std::printf("=== merger-only check (input = HF %s last_hidden) ===\n", src);
        BF* d_lh = upload_bf(as_f32(load_npy(DIR + "/" + lh)));
        pie_cuda_driver::model::run_merger(
            blas, W.merger, d_lh, n_patch, n_token,
            HID, 4, OUT, 1e-6f, d_main, 0);
        HCK(cudaDeviceSynchronize());
        auto my = fetch(d_main);
        report("merged HF-bf16", my, "merged.npy");
        report("merged HF-fp32", my, "merged_f32.npy");
        HCK(cudaFree(d_lh));
    }

    std::printf("=== per-layer hidden checkpoints (bf16-vs-bf16) ===\n");
    const int patch_counts[] = {n_patch};
    set_qwen3vl_vision_ckpt(qvis_ckpt_cb, nullptr);
    run_qwen3vl_vision(
        blas, W, d_pix, d_rope, d_pe, 1, patch_counts,
        d_main, d_deep.data(), NDEEP);
    set_qwen3vl_vision_ckpt(nullptr, nullptr);
    HCK(cudaDeviceSynchronize());
    cublasDestroy(blas);

    std::printf("=== Qwen3-VL vision encoder parity (bf16-vs-bf16 is the real metric) ===\n");
    auto mainy = fetch(d_main);
    std::printf("[main merger]\n");
    double cmain = report("HF-bf16", mainy, "merged.npy");          // ← the real comparison
    report("HF-fp32", mainy, "merged_f32.npy");                     // (bf16 vs fp32 ≈ HF's own gap)
    const char* dbf[NDEEP] = {"deepstack0_layer5.npy", "deepstack1_layer11.npy", "deepstack2_layer17.npy"};
    double cmin = cmain;
    for (int d = 0; d < NDEEP; d++) {
        char tag[40]; std::snprintf(tag, sizeof(tag), "deepstack%d HF-bf16", d);
        cmin = std::min(cmin, report(tag, fetch(d_deep[d]), dbf[d]));
    }
    bool pass = cmin > 0.99;
    std::printf("%s (min cosine=%.5f)\n", pass ? "QWEN VISION PARITY PASS" : "QWEN VISION PARITY (needs work)", cmin);
    return pass ? 0 : 1;
}
