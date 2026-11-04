#pragma once
// Requires E, W, H from bench_resident_multilayer_fp8.cu.
// GPU form of pack_down (bench_resident_multilayer_fp8.cu): out[(e*(W/4)+g)*H+row] packs the four
// bytes in[(e*W+g*4+j)*H+row], j=0..3, little-endian. One thread per output word.
__global__ void pack_down_kernel(const uint8_t*__restrict__ in,uint32_t*__restrict__ out){
 const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(E)*(W/4)*H)return;
 const size_t row=i%H,t=i/H,g=t%(W/4),e=t/(W/4);uint32_t v=0;
 for(int j=0;j<4;j++)v|=uint32_t(in[(e*W+g*4+j)*H+row])<<(8*j);out[i]=v;}
