#include <metal_stdlib>
using namespace metal;

// Patch rows of raw RGB bytes (patch x patch pixels, HWC) to the tower's
// rows: (v / 255 - mean[c]) / std[c], columns channel-major (order 0) or
// pixel-major (order 1), the frame repeated width / (3 patch^2) times.
template <typename T>
[[kernel]] void pixels(
    const device uchar* x       [[buffer(0)]],
    device T* y                 [[buffer(1)]],
    const constant int& patch   [[buffer(2)]],
    const constant int& width   [[buffer(3)]],
    const constant int& order   [[buffer(4)]],
    const constant float& mean0 [[buffer(5)]],
    const constant float& mean1 [[buffer(6)]],
    const constant float& mean2 [[buffer(7)]],
    const constant float& std0  [[buffer(8)]],
    const constant float& std1  [[buffer(9)]],
    const constant float& std2  [[buffer(10)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int c_out = int(tid.x);
  if (c_out >= width) {
    return;
  }
  const size_t n = size_t(tid.y);
  const int pixels = patch * patch;
  const int in_width = 3 * pixels;
  const int temporal = width / in_width;
  int ch, at;
  if (order == 0) {
    // (channel, frame, row, col): the frame repeats within each channel.
    ch = c_out / (temporal * pixels);
    at = (c_out % pixels) * 3 + ch;
  } else {
    // (frame, row, col, channel): the input's own order, repeated.
    const int k = c_out % in_width;
    ch = k % 3;
    at = k;
  }
  const float mean = ch == 0 ? mean0 : (ch == 1 ? mean1 : mean2);
  const float sd = ch == 0 ? std0 : (ch == 1 ? std1 : std2);
  const float v = float(x[n * size_t(in_width) + size_t(at)]) / 255.0f;
  y[n * size_t(width) + size_t(c_out)] = T((v - mean) / sd);
}

#define instantiate_pixels(name, itype)                                     \
  template [[host_name("pixels_" #name)]]                                   \
  [[kernel]] void pixels<itype>(                                            \
      const device uchar*, device itype*, const constant int&,              \
      const constant int&, const constant int&, const constant float&,      \
      const constant float&, const constant float&, const constant float&,  \
      const constant float&, const constant float&, uint2);

instantiate_pixels(bfloat16, bfloat)
