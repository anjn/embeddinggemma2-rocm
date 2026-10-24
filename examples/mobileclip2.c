#include <mobileclip2/mobileclip2.h>
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: mc2-example-c BUNDLE gpu|npu\n");
    return 2;
  }
  mc2_context *ctx = NULL;
  char error[4096];
  float output[512];
  const char *text = "a person wearing a hat";
  mc2_status s = mc2_create(argv[1], argv[2], &ctx, error, sizeof error);
  if (!s)
    s = mc2_encode_texts(ctx, &text, 1, output, 512, error, sizeof error);
  if (!s) {
    unsigned char rgb[16 * 16 * 3];
    memset(rgb, 127, sizeof rgb);
    mc2_image image = {rgb, 16, 16, 48};
    s = mc2_encode_images(ctx, &image, 1, output, 512, error, sizeof error);
  }
  if (s)
    fprintf(stderr, "%s\n", error);
  else
    printf("dimension=512 first=%g\n", output[0]);
  mc2_destroy(ctx);
  return s ? 1 : 0;
}
