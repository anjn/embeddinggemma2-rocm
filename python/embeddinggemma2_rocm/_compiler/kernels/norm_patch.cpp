#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
using B=hip_bfloat16;
union P4{uint2 v;B x[4];__device__ P4(){};};
template<int L,bool R,bool F=false> __device__ void body(const B*x,const float*w,const B*r,B*y,B*z=nullptr){
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/L,lane=threadIdx.x%L;
 if(row>=16*594)return;
 constexpr int N=768/(L*4);P4 vals[N];float sum=0;
 #pragma unroll
 for(int i=0;i<N;i++){
  vals[i].v=reinterpret_cast<const uint2*>(x+row*768)[i*L+lane];
  #pragma unroll
  for(int j=0;j<4;j++){float z=float(vals[i].x[j]);sum+=z*z;}
 }
 for(int d=L/2;d;d/=2)sum+=__shfl_xor(sum,d,L);
 float inv=rsqrtf(sum*(1.f/768.f)+1e-6f);sum=0;
 #pragma unroll
 for(int i=0;i<N;i++){
  int col=(i*L+lane)*4;P4 out,res;float4 weight;
  if constexpr(R){res.v=*reinterpret_cast<const uint2*>(r+row*768+col);weight=*reinterpret_cast<const float4*>(w+col);}
  #pragma unroll
  for(int j=0;j<4;j++){
   float z=float(vals[i].x[j])*inv;
   if constexpr(R)z*=reinterpret_cast<float*>(&weight)[j];
   B v=B(z);out.x[j]=R?B(float(v)+float(res.x[j])):v;
  }
  *reinterpret_cast<uint2*>(y+row*768+col)=out.v;
  if constexpr(F){vals[i].v=out.v;for(int j=0;j<4;j++){float a=float(out.x[j]);sum+=a*a;}}
 }
 if constexpr(F){
  for(int d=L/2;d;d/=2)sum+=__shfl_xor(sum,d,L);
  float inv2=rsqrtf(sum*(1.f/768.f)+1e-6f);
  #pragma unroll
  for(int i=0;i<N;i++){
   P4 out;for(int j=0;j<4;j++)out.x[j]=B(float(vals[i].x[j])*inv2);
   reinterpret_cast<uint2*>(z+row*768)[i*L+lane]=out.v;
  }
 }
}
#define MAKE(L) extern "C" __global__ void hip_norm_##L(const B*x,B*y){body<L,false>(x,nullptr,nullptr,y);} \
extern "C" __global__ void hip_norm_res_##L(const B*x,const float*w,const B*r,B*y){body<L,true>(x,w,r,y);}
MAKE(8) MAKE(16) MAKE(32)
extern "C" int run(void*x,void*w,void*r,void*y,int lanes,int residual,void*s){
#define RUN(L) if(residual){hipLaunchKernelGGL(hip_norm_res_##L,dim3((16*594*L+255)/256),dim3(256),0,(hipStream_t)s,(B*)x,(float*)w,(B*)r,(B*)y);}else{hipLaunchKernelGGL(hip_norm_##L,dim3((16*594*L+255)/256),dim3(256),0,(hipStream_t)s,(B*)x,(B*)y);}
 if(lanes==8){RUN(8)}else if(lanes==16){RUN(16)}else{RUN(32)}
 return hipGetLastError();
}

extern "C" __global__ void hip_norm_pair(const B*x,const float*w,const B*r,B*y,B*z){body<32,true,true>(x,w,r,y,z);}
extern "C" int run_pair(void*x,void*w,void*r,void*y,void*z,void*s){
 hipLaunchKernelGGL(hip_norm_pair,dim3((16*594*32+255)/256),dim3(256),0,(hipStream_t)s,(B*)x,(float*)w,(B*)r,(B*)y,(B*)z);return hipGetLastError();
}

extern "C" int run_pair_block(void*x,void*w,void*r,void*y,void*z,int block,void*s){
 if(block!=64 && block!=128 && block!=256 && block!=512)return int(hipErrorInvalidValue);
 hipLaunchKernelGGL(hip_norm_pair,dim3((16*594*32+block-1)/block),dim3(block),0,(hipStream_t)s,(B*)x,(float*)w,(B*)r,(B*)y,(B*)z);return hipGetLastError();
}
extern "C" __global__ void hip_norm_pair_l8(const B*x,const float*w,const B*r,B*y,B*z){body<8,true,true>(x,w,r,y,z);}
extern "C" __global__ void hip_norm_pair_l16(const B*x,const float*w,const B*r,B*y,B*z){body<16,true,true>(x,w,r,y,z);}
extern "C" int run_pair_layout(void*x,void*w,void*r,void*y,void*z,int lanes,int block,void*s){
 if(block!=64 && block!=128 && block!=256 && block!=512)return int(hipErrorInvalidValue);
 #define PAIR(K) hipLaunchKernelGGL(K,dim3((16*594*lanes+block-1)/block),dim3(block),0,(hipStream_t)s,(B*)x,(float*)w,(B*)r,(B*)y,(B*)z)
 if(lanes==8){PAIR(hip_norm_pair_l8);}else if(lanes==16){PAIR(hip_norm_pair_l16);}else if(lanes==32){PAIR(hip_norm_pair);}else{return int(hipErrorInvalidValue);}
 return hipGetLastError();
}
