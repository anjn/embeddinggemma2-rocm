#include "qk_rope_layout.cpp"
union QPack4{uint2 bits;unsigned short v[4];};
extern "C" __global__ void hip_norm_rope_packed4(const unsigned short*x,const float*w,const unsigned short*co,const unsigned short*si,unsigned short*y){
 int tid=blockIdx.x*blockDim.x+threadIdx.x,row=tid/16,lane=tid%16;
 if(row>=ROWS*12)return;
 int c=lane*4,src=(row/12)*2304+(row%12)*64+c,p=(row/12)*64+c;
 QPack4 in,cosine,sine,out;
 in.bits=*reinterpret_cast<const uint2*>(x+src);
 cosine.bits=*reinterpret_cast<const uint2*>(co+p);
 sine.bits=*reinterpret_cast<const uint2*>(si+p);
 float4 weight=*reinterpret_cast<const float4*>(w+c);float ww[4]={weight.x,weight.y,weight.z,weight.w};
 float v[4],ss=0;
 #pragma unroll
 for(int j=0;j<4;j++){v[j]=rf(in.v[j]);ss+=v[j]*v[j];}
 #pragma unroll
 for(int d=1;d<16;d*=2)ss+=__shfl_xor(ss,d);
 float den=sqrtf(ss/64.f+1e-6f),sg=c%32<16?-1.f:1.f;
 #pragma unroll
 for(int j=0;j<4;j++)v[j]=rf(bf((v[j]/den)*ww[j]));
 #pragma unroll
 for(int j=0;j<4;j++){
  float rotated=__shfl_xor(v[j],4);
  float a=rf(bf(v[j]*rf(cosine.v[j]))),b=rf(bf(sg*rotated*rf(sine.v[j])));
  out.v[j]=bf(a+b);
 }
 *reinterpret_cast<uint2*>(y+row*64+c)=out.bits;
}
extern "C" int test_packed(void*x,void*w,void*c,void*s,void*y,int block,void*stream){
 hipLaunchKernelGGL(hip_norm_rope_packed4,dim3((ROWS*12*16+block-1)/block),dim3(block),0,(hipStream_t)stream,(unsigned short*)x,(float*)w,(unsigned short*)c,(unsigned short*)s,(unsigned short*)y);
 return hipGetLastError();
}
