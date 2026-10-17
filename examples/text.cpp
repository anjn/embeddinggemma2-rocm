#include <cmath>
#include <eg2/eg2.hpp>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: eg2-text-example BUNDLE\n";
    return 2;
  }
  try {
    eg2::Context model(argv[1]);
    const auto embeddings = model.texts({"native embeddings", "日本語の検索"});
    for (int row = 0; row < 2; ++row) {
      double squared_norm = 0;
      for (int column = 0; column < 768; ++column) {
        const auto value = embeddings[row * 768 + column];
        squared_norm += value * value;
      }
      const auto norm = std::sqrt(squared_norm);
      if (!std::isfinite(norm) || std::abs(norm - 1.) > .001)
        throw std::runtime_error("Invalid text embedding norm");
      std::cout << "text " << row << ": 768 dimensions, norm=" << norm << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
