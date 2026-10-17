#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#ifndef ROWS
#define ROWS 9504
#endif
union Pack8 { uint4 vector; unsigned short h[8]; };
__device__ float read_bf(unsigned short x){return __uint_as_float(unsigned(x)<<16);}
__device__ unsigned short write_bf(float x){unsigned u=__float_as_uint(x);return (u&0x7fffffff)>0x7f800000?0x7fc0:(u+0x7fff+((u>>16)&1))>>16;}
__device__ float tanh_fast(float x){float e=__expf(-2*fabsf(x));return copysignf((1-e)/(1+e),x);}
template<bool ROUND> __device__ void body(const unsigned short* gate,const unsigned short* up,unsigned short* out){
 int i=(blockIdx.x*blockDim.x+threadIdx.x)*8;if(i>=ROWS*3072)return;
 int r=i/3072,j=i%3072;
 Pack8 g,u,y;g.vector=*reinterpret_cast<const uint4*>(gate+r*6144+j);u.vector=*reinterpret_cast<const uint4*>(up+r*6144+j);
 #pragma unroll
 for(int k=0;k<8;k++){float a=read_bf(g.h[k]);float z=.5f*a*(1+tanh_fast(.7978845608028654f*(a+.044715f*a*a*a)));if(ROUND)z=read_bf(write_bf(z));y.h[k]=write_bf(z*read_bf(u.h[k]));}
 *reinterpret_cast<uint4*>(out+i)=y.vector;
}
extern "C" __global__ void hip_gelu_vec8(const unsigned short*g,const unsigned short*u,unsigned short*y){body<false>(g,u,y);}
extern "C" __global__ void hip_gelu_vec8_round(const unsigned short*g,const unsigned short*u,unsigned short*y){body<true>(g,u,y);}
