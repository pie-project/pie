#include "quant/dequant_wna16.cuh"
namespace d = pie_cuda_driver::kernels::quant::device;
using bf = pie_cuda_driver::kernels::device::bf16;
void fire(const __half*a,const int*t,const int*const*p,const void*const*s,bf*o,int k,int h,int i,int g,cudaStream_t st){
  d::wna16_down_decode<<<dim3(1,1),256,0,st>>>(a,t,p,s,o,k,h,i,g);
}
