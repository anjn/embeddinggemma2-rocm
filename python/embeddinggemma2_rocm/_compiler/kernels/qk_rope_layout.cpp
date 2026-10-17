#include "qk_rope_patch.cpp"
template<int LANES> __device__ void compact(const unsigned short*x,const float*w,const unsigned short*co,const unsigned short*si,unsigned short*y){
 constexpr int N=64/LANES;
 int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid/LANES,lane=tid%LANES;
 if(row>=ROWS*12)return;
 int src=(row/12)*2304+(row%12)*64,c=lane*N,p=(row/12)*64+c;
 float v[N],ss=0;
 #pragma unroll
 for(int j=0;j<N;j++){v[j]=rf(x[src+c+j]);ss+=v[j]*v[j];}
 #pragma unroll
 for(int d=1;d<LANES;d*=2)ss+=__shfl_xor(ss,d);
 float den=sqrtf(ss/64.f+1e-6f),sg=(c%32)<16?-1.f:1.f;
 #pragma unroll
 for(int j=0;j<N;j++)v[j]=rf(bf((v[j]/den)*w[c+j]));
 #pragma unroll
 for(int j=0;j<N;j++){
  float rotated=__shfl_xor(v[j],16/N);
  float a=rf(bf(v[j]*rf(co[p+j]))),b=rf(bf(sg*rotated*rf(si[p+j])));
  y[row*64+c+j]=bf(a+b);
 }
}
#define MAKE(L) extern "C" __global__ void hip_norm_rope_l##L(const unsigned short*x,const float*w,const unsigned short*c,const unsigned short*s,unsigned short*y){compact<L>(x,w,c,s,y);}
MAKE(4)
MAKE(8)
MAKE(16)
extern "C" int test_layout(void*x,void*w,void*c,void*s,void*y,int lanes,void*stream){
 #define RUN(L) hipLaunchKernelGGL(hip_norm_rope_l##L,dim3(ROWS*12*L/128),dim3(128),0,(hipStream_t)stream,(unsigned short*)x,(float*)w,(unsigned short*)c,(unsigned short*)s,(unsigned short*)y)
 if(lanes==4){RUN(4);}else if(lanes==8){RUN(8);}else if(lanes==16){RUN(16);}else if(lanes==32){hipLaunchKernelGGL(hip_norm_rope_round,dim3(ROWS*12/4),dim3(128),0,(hipStream_t)stream,(unsigned short*)x,(float*)w,(unsigned short*)c,(unsigned short*)s,(unsigned short*)y);}else return 1;
 return hipGetLastError();
}
