#include <eg2/eg2.h>
#include <iostream>
#include <vector>
int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: eg2-example BUNDLE [BATCH]\n";
    return 2;
  }
  eg2_context *ctx = nullptr;
  char error[4096];
  if (eg2_create(argv[1], 0, &ctx, error, sizeof(error))) {
    std::cerr << error << '\n';
    return 1;
  }
  size_t batch = argc == 3 ? std::stoul(argv[2]) : 1;
  std::vector<uint8_t> rgb(batch * 512 * 288 * 3, 127);
  std::vector<float> out(batch * 768);
  auto status = eg2_encode_images(ctx, rgb.data(), batch, 288, 512, 70,
                                  out.data(), out.size(), error, sizeof(error));
  if (status)
    std::cerr << error << '\n';
  else
    std::cout << "embedding[0]=" << out[0] << " dimension=" << out.size()
              << '\n';
  eg2_destroy(ctx);
  return status ? 1 : 0;
}
