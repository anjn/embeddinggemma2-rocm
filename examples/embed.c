#include <eg2/eg2.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: eg2-c-example BUNDLE\n");
    return 2;
  }
  eg2_context *ctx = NULL;
  char error[4096];
  if (eg2_create(argv[1], 0, &ctx, error, sizeof(error))) {
    fprintf(stderr, "%s\n", error);
    return 1;
  }
  unsigned char *rgb = calloc(512 * 288 * 3, 1);
  float *out = calloc(768, sizeof(float));
  int status = eg2_encode_images(ctx, rgb, 1, 288, 512, 70, out, 768, error,
                                 sizeof(error));
  if (status)
    fprintf(stderr, "%s\n", error);
  else
    printf("embedding[0]=%f\n", out[0]);
  free(rgb);
  free(out);
  eg2_destroy(ctx);
  return status ? 1 : 0;
}
