#include <iostream>
#include <mobileclip2/mobileclip2.hpp>
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: mc2-example BUNDLE gpu|npu\n";
    return 2;
  }
  try {
    mobileclip2::Embedder model(argv[1], argv[2]);
    model.prepare();
    std::vector<uint8_t> rgb(192 * 108 * 3, 127);
    mc2_image im{rgb.data(), 108, 192, 108 * 3};
    auto images = model.images({im});
    auto texts = model.texts({"a person wearing a hat"});
    double similarity = 0;
    for (size_t i = 0; i < 512; i++)
      similarity += images[i] * texts[i];
    std::cout << "dimension=512 cosine=" << similarity << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
