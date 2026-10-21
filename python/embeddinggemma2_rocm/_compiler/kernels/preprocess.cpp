#include <hip/hip_runtime.h>
// Half-pixel bilinear resize into non-padded 16x16 patches, RGB [0,1].
__device__ void patchify_one(const unsigned char *rgb, float *out, int count, int h,
                         int w, int gh, int gw, int i) {
  if (i >= count)
    return;
  int seq = gh * gw, b = i / (seq * 768), p = (i / 768) % seq, c = i % 3,
      px = (i % 768) / 3 % 16, py = (i % 768) / 48;
  int ox = (p % gw) * 16 + px, oy = (p / gw) * 16 + py;
  float fx = fmaxf(0.f, (ox + .5f) * w / (gw * 16.f) - .5f),
        fy = fmaxf(0.f, (oy + .5f) * h / (gh * 16.f) - .5f);
  int x0 = int(fx), y0 = int(fy), x1 = min(x0 + 1, w - 1),
      y1 = min(y0 + 1, h - 1);
  float dx = fx - x0, dy = fy - y0;
  const unsigned char *src = rgb + size_t(b) * h * w * 3;
  float a = src[(size_t(y0) * w + x0) * 3 + c] * (1 - dx) +
            src[(size_t(y0) * w + x1) * 3 + c] * dx;
  float z = src[(size_t(y1) * w + x0) * 3 + c] * (1 - dx) +
            src[(size_t(y1) * w + x1) * 3 + c] * dx;
  out[i] = (a * (1 - dy) + z * dy) / 255.f;
}
template<int V> __global__ void patchify_vector(const unsigned char *rgb,float*out,int count,int h,int w,int gh,int gw) {
  int start=(blockIdx.x*blockDim.x+threadIdx.x)*V;
  #pragma unroll
  for(int j=0;j<V;j++) patchify_one(rgb,out,count,h,w,gh,gw,start+j);
}
extern "C" int eg2_patchify_config(void*x,void*y,int b,int h,int w,int gh,int gw,void*s,int block,int vector) {
  if(block!=64 && block!=128 && block!=256 && block!=512) return int(hipErrorInvalidValue);
  int n=b*gh*gw*768;
#define PATCH(V) hipLaunchKernelGGL(patchify_vector<V>,dim3((n+block*V-1)/(block*V)),dim3(block),0,(hipStream_t)s,(unsigned char*)x,(float*)y,n,h,w,gh,gw)
  switch(vector) {case 1:PATCH(1);break;case 2:PATCH(2);break;case 4:PATCH(4);break;case 8:PATCH(8);break;case 16:PATCH(16);break;default:return int(hipErrorInvalidValue);}
  return hipGetLastError();
}
extern "C" int eg2_patchify(void*x,void*y,int b,int h,int w,int gh,int gw,void*s) {
  return eg2_patchify_config(x,y,b,h,w,gh,gw,s,256,1);
}
