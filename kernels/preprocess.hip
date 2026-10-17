#include <hip/hip_runtime.h>
// Half-pixel bilinear resize into non-padded 16x16 patches, RGB [0,1].
__global__ void patchify(const unsigned char *rgb, float *out, int count, int h,
                         int w, int gh, int gw) {
  int i = blockIdx.x * 256 + threadIdx.x;
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
extern "C" int eg2_patchify(void *x, void *y, int b, int h, int w, int gh,
                            int gw, void *s) {
  int n = b * gh * gw * 768;
  hipLaunchKernelGGL(patchify, dim3((n + 255) / 256), dim3(256), 0,
                     (hipStream_t)s, (unsigned char *)x, (float *)y, n, h, w,
                     gh, gw);
  return hipGetLastError();
}
