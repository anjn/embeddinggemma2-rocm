#include "gelu_patch.cpp"
union Pack2 {unsigned int vector;unsigned short h[2];};
union Pack4 {uint2 vector;unsigned short h[4];};
union Pack16 {uint4 vector[2];unsigned short h[16];};
template<int N,bool ROUND> __device__ void layout(const unsigned short*g,const unsigned short*u,unsigned short*y){
 int i=(blockIdx.x*blockDim.x+threadIdx.x)*N;if(i>=ROWS*3072)return;
 int r=i/3072,j=i%3072;
 if constexpr(N==1 || N==2){
  Pack2 a,b,c;
  if constexpr(N==2){a.vector=*reinterpret_cast<const unsigned int*>(g+r*6144+j);b.vector=*reinterpret_cast<const unsigned int*>(u+r*6144+j);}
  else{a.h[0]=g[r*6144+j];b.h[0]=u[r*6144+j];}
  #pragma unroll
  for(int k=0;k<N;k++){float x=read_bf(a.h[k]);float z=.5f*x*(1+tanh_fast(.7978845608028654f*(x+.044715f*x*x*x)));if(ROUND)z=read_bf(write_bf(z));c.h[k]=write_bf(z*read_bf(b.h[k]));}
  if constexpr(N==2)*reinterpret_cast<unsigned int*>(y+i)=c.vector;else y[i]=c.h[0];
 }else if constexpr(N==4){
  Pack4 a,b,c;a.vector=*reinterpret_cast<const uint2*>(g+r*6144+j);b.vector=*reinterpret_cast<const uint2*>(u+r*6144+j);
  #pragma unroll
  for(int k=0;k<N;k++){float x=read_bf(a.h[k]);float z=.5f*x*(1+tanh_fast(.7978845608028654f*(x+.044715f*x*x*x)));if(ROUND)z=read_bf(write_bf(z));c.h[k]=write_bf(z*read_bf(b.h[k]));}
  *reinterpret_cast<uint2*>(y+i)=c.vector;
 }else{
  union Wide {uint4 vector[N/8];unsigned short h[N];};Wide a,b,c;
  #pragma unroll
  for(int k=0;k<N/8;k++){a.vector[k]=reinterpret_cast<const uint4*>(g+r*6144+j)[k];b.vector[k]=reinterpret_cast<const uint4*>(u+r*6144+j)[k];}
  #pragma unroll
  for(int k=0;k<N;k++){float x=read_bf(a.h[k]);float z=.5f*x*(1+tanh_fast(.7978845608028654f*(x+.044715f*x*x*x)));if(ROUND)z=read_bf(write_bf(z));c.h[k]=write_bf(z*read_bf(b.h[k]));}
  #pragma unroll
  for(int k=0;k<N/8;k++)reinterpret_cast<uint4*>(y+i)[k]=c.vector[k];
 }
}
#define MAKE(N,R) extern "C" __global__ void hip_gelu_v##N##_r##R(const unsigned short*g,const unsigned short*u,unsigned short*y){layout<N,R>(g,u,y);}
MAKE(1,0)
MAKE(1,1)
MAKE(2,0)
MAKE(2,1)
MAKE(32,0)
MAKE(32,1)
MAKE(4,0)
MAKE(4,1)
MAKE(16,0)
MAKE(16,1)
extern "C" int test_gelu_layout(void*g,void*u,void*y,int n,int rounded,int block,void*s){
 dim3 grid((ROWS*3072/n+block-1)/block);
 #define RUN(K) hipLaunchKernelGGL(K,grid,dim3(block),0,(hipStream_t)s,(unsigned short*)g,(unsigned short*)u,(unsigned short*)y)
 if(n==1){if(rounded){RUN(hip_gelu_v1_r1);}else{RUN(hip_gelu_v1_r0);}}
 else if(n==2){if(rounded){RUN(hip_gelu_v2_r1);}else{RUN(hip_gelu_v2_r0);}}
 else if(n==32){if(rounded){RUN(hip_gelu_v32_r1);}else{RUN(hip_gelu_v32_r0);}}
 else if(n==4){if(rounded){RUN(hip_gelu_v4_r1);}else{RUN(hip_gelu_v4_r0);}}
 else if(n==16){if(rounded){RUN(hip_gelu_v16_r1);}else{RUN(hip_gelu_v16_r0);}}
 else if(n==8){if(rounded){RUN(hip_gelu_vec8_round);}else{RUN(hip_gelu_vec8);}}
 else return 1;
 return hipGetLastError();
}
