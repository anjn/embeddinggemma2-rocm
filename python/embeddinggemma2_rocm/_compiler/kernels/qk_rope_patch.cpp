#include <hip/hip_runtime.h>
#ifndef ROWS
#define ROWS 9504
#endif
__device__ float rf(unsigned short x){return __uint_as_float(unsigned(x)<<16);}
__device__ unsigned short bf(float x){unsigned u=__float_as_uint(x);return (u&0x7fffffff)>0x7f800000?0x7fc0:(u+0x7fff+((u>>16)&1))>>16;}
template<bool ROUND> __device__ void fused(const unsigned short*x,const float*w,const unsigned short*co,const unsigned short*si,unsigned short*y){
 int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(row>=ROWS*12)return;
 int src=(row/12)*2304+(row%12)*64, c=lane*2;
 float a=rf(x[src+c]),b=rf(x[src+c+1]),ss=a*a+b*b;
 for(int d=1;d<32;d*=2)ss+=__shfl_xor(ss,d);
 float den=sqrtf(ss/64.f+1e-6f);
 a=rf(bf((a/den)*w[c]));b=rf(bf((b/den)*w[c+1]));
 float ar=__shfl_xor(a,8),br=__shfl_xor(b,8),sg=lane%16<8?-1.f:1.f;
 int p=(row/12)*64+c;
 float u=a*rf(co[p]),v=sg*ar*rf(si[p]),u1=b*rf(co[p+1]),v1=sg*br*rf(si[p+1]);
 if(ROUND){u=rf(bf(u));v=rf(bf(v));u1=rf(bf(u1));v1=rf(bf(v1));}
 *reinterpret_cast<unsigned*>(y+row*64+c)=unsigned(bf(u+v))|(unsigned(bf(u1+v1))<<16);
}
extern "C" __global__ void hip_norm_rope(const unsigned short*x,const float*w,const unsigned short*c,const unsigned short*s,unsigned short*y){fused<false>(x,w,c,s,y);}
extern "C" __global__ void hip_norm_rope_round(const unsigned short*x,const float*w,const unsigned short*c,const unsigned short*s,unsigned short*y){fused<true>(x,w,c,s,y);}
