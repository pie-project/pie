// Stage (a) parity test for the Gemma-4 vision patch embedder.
//
// Reproduces `Gemma4VisionPatchEmbedder.forward` in CUDA (fp32) and checks it
// against the fp32 reference dumped by `scripts/gemma4_vision_parity_ref.py`:
//
//   scaled = 2*(pixel_values - 0.5)
//   y      = scaled @ input_proj.weight^T              (plain linear, no clip)
//   y[n]  += pos_table[0][x_n] + pos_table[1][y_n]     (2D position embedding)
//
// Standalone — compile directly with nvcc, not via the driver CMake build:
//   nvcc -O2 -arch=sm_89 gemma4_vision_patch_parity.cu -o /tmp/patch_parity
//   /tmp/patch_parity /tmp/gemma4_vision_parity
//
// Exit 0 + "PARITY PASS" iff max abs error is below tolerance.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e = (x);                                                   \
        if (e != cudaSuccess) {                                                \
            std::fprintf(stderr, "cuda error %s at %s:%d\n",                   \
                         cudaGetErrorString(e), __FILE__, __LINE__);           \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

// Minimal .npy reader (v1.0/2.0, little-endian, C-order). Returns shape and
// raw bytes; caller reinterprets per the recorded dtype char ('f'=float32,
// 'i'=int64 here).
struct Npy {
    std::vector<int64_t> shape;
    char kind = 0;   // 'f' or 'i'
    int itemsize = 0;
    std::vector<uint8_t> data;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

Npy load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "bad npy magic in %s\n", path.c_str()); std::exit(2);
    }
    uint8_t major = f.get(), minor = f.get();
    (void)minor;
    uint32_t hlen;
    if (major == 1) {
        uint16_t h; f.read(reinterpret_cast<char*>(&h), 2); hlen = h;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string hdr(hlen, '\0');
    f.read(hdr.data(), hlen);

    Npy out;
    // descr: e.g. '<f4' or '<i8'
    auto dpos = hdr.find("'descr'");
    auto q = hdr.find('\'', hdr.find(':', dpos) + 1);
    std::string descr = hdr.substr(q + 1, hdr.find('\'', q + 1) - q - 1);
    out.kind = descr[1];
    out.itemsize = std::atoi(descr.substr(2).c_str());
    // shape: (a, b, ...)
    auto sp = hdr.find("'shape'");
    auto lp = hdr.find('(', sp), rp = hdr.find(')', lp);
    std::string sh = hdr.substr(lp + 1, rp - lp - 1);
    size_t i = 0;
    while (i < sh.size()) {
        while (i < sh.size() && !std::isdigit(sh[i])) ++i;
        if (i >= sh.size()) break;
        int64_t v = 0;
        while (i < sh.size() && std::isdigit(sh[i])) v = v * 10 + (sh[i++] - '0');
        out.shape.push_back(v);
    }
    out.data.resize(static_cast<size_t>(out.numel()) * out.itemsize);
    f.read(reinterpret_cast<char*>(out.data.data()),
           static_cast<std::streamsize>(out.data.size()));
    return out;
}

// y[n,o] = sum_k scaled[n,k] * W[o,k]   (W row-major [O,K])
__global__ void matmul_wt(const float* __restrict__ scaled,
                          const float* __restrict__ W, float* __restrict__ y,
                          int N, int K, int O) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= O) return;
    float acc = 0.f;
    const float* xr = scaled + static_cast<long>(n) * K;
    const float* wr = W + static_cast<long>(o) * K;
    for (int k = 0; k < K; ++k) acc += xr[k] * wr[k];
    y[static_cast<long>(n) * O + o] = acc;
}

// scaled[n,k] = 2*(pix[n,k] - 0.5)
__global__ void scale_pix(const float* pix, float* scaled, long total) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < total) scaled[i] = 2.f * (pix[i] - 0.5f);
}

// y[n,o] += table[0, x_n, o] + table[1, y_n, o]
// Positions arrive as float (the reference dump casts to float32); they hold
// integer coordinates, so round-to-nearest then clamp negatives (padding=-1).
__global__ void add_pos(float* y, const float* table, const float* pos,
                        int N, int O, int P) {
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= O) return;
    long x = (long)llrintf(pos[2 * (long)n + 0]);
    long yy = (long)llrintf(pos[2 * (long)n + 1]);
    if (x < 0) x = 0;
    if (yy < 0) yy = 0;
    const float* t0 = table + (0L * P + x) * O;
    const float* t1 = table + (1L * P + yy) * O;
    y[(long)n * O + o] += t0[o] + t1[o];
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "/tmp/gemma4_vision_parity";
    auto pix = load_npy(dir + "/input_pixel_values_f32.npy");  // [1,N,K]
    auto pos = load_npy(dir + "/input_position_ids.npy");      // [1,N,2] i8
    auto W = load_npy(dir + "/input_proj_w.npy");              // [O,K]
    auto table = load_npy(dir + "/pos_table.npy");             // [2,P,O]
    auto ref = load_npy(dir + "/patch_embed_f32.npy");         // [1,N,O]

    const int N = static_cast<int>(pix.shape[pix.shape.size() - 2]);
    const int K = static_cast<int>(pix.shape.back());
    const int O = static_cast<int>(W.shape[0]);
    const int P = static_cast<int>(table.shape[1]);
    std::printf("N=%d K=%d O=%d P=%d  (pos kind=%c isize=%d)\n", N, K, O, P,
                pos.kind, pos.itemsize);
    if (pos.kind != 'f' || pos.itemsize != 4) {
        std::fprintf(stderr, "expected float32 position ids (as dumped)\n"); return 2;
    }

    float *d_pix, *d_scaled, *d_W, *d_y, *d_table, *d_pos;
    CK(cudaMalloc(&d_pix, (long)N * K * 4));
    CK(cudaMalloc(&d_scaled, (long)N * K * 4));
    CK(cudaMalloc(&d_W, (long)O * K * 4));
    CK(cudaMalloc(&d_y, (long)N * O * 4));
    CK(cudaMalloc(&d_table, table.data.size()));
    CK(cudaMalloc(&d_pos, (long)N * 2 * 4));
    CK(cudaMemcpy(d_pix, pix.data.data(), (long)N * K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_W, W.data.data(), (long)O * K * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_table, table.data.data(), table.data.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pos, pos.data.data(), (long)N * 2 * 4, cudaMemcpyHostToDevice));

    long total = (long)N * K;
    scale_pix<<<(total + 255) / 256, 256>>>(d_pix, d_scaled, total);
    dim3 b(16, 16), g((O + 15) / 16, (N + 15) / 16);
    matmul_wt<<<g, b>>>(d_scaled, d_W, d_y, N, K, O);
    add_pos<<<g, b>>>(d_y, d_table, d_pos, N, O, P);
    CK(cudaDeviceSynchronize());

    std::vector<float> y((long)N * O);
    CK(cudaMemcpy(y.data(), d_y, (long)N * O * 4, cudaMemcpyDeviceToHost));

    const float* r = reinterpret_cast<const float*>(ref.data.data());
    double max_abs = 0, max_rel = 0, sum_sq = 0;
    for (long i = 0; i < (long)N * O; ++i) {
        double a = std::abs((double)y[i] - r[i]);
        max_abs = std::max(max_abs, a);
        max_rel = std::max(max_rel, a / (std::abs((double)r[i]) + 1e-6));
        sum_sq += (double)y[i] * y[i];
    }
    double rms = std::sqrt(sum_sq / ((long)N * O));
    std::printf("max_abs=%.3e  max_rel=%.3e  rms(out)=%.3f\n", max_abs, max_rel, rms);
    // fp32 naive vs torch fp32: summation-order differences only.
    bool pass = max_abs < 1e-2;
    std::printf("%s\n", pass ? "PARITY PASS" : "PARITY FAIL");
    return pass ? 0 : 1;
}
