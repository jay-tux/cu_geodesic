#include "io.hpp"
#include "kernel.hpp"
#include <fstream>
#include <iostream>

int main(int argc, const char **argv) {
  if (argc < 4) {
    std::cerr << "[ERROR] expected usage: " << argv[0]
              << " <data file> <core count> <granularity>" << std::endl;
    std::exit(1);
  }
  const auto &[poly, start, box] =
      cu_geodesic::loader::load(std::string(argv[1]));
  size_t core_count = std::strtol(argv[2], nullptr, 10);
  double gran = std::strtod(argv[3], nullptr);
  auto kernel =
      cu_geodesic::kernel{.data = poly, .start_box = box, .start = start};
  auto _res = kernel(gran, core_count);
  std::cout << "[MAIN]: starting output..." << std::endl;
  if (!_res.has_value()) {
    std::cout << "[MAIN]: kernel failed. No output generated." << std::endl;
    return -1;
  }
  auto res = _res.value();
  std::cout << "[MAIN]: using `" << argv[1] << ".csv' as output file."
            << std::endl;
  std::ofstream dump(std::string(argv[1]) + ".csv");
  dump << "point x,point y,distance" << std::endl;
  for (size_t i = 0; i < res.distances.size(); i++) {
    const auto &val = res.distances[i];
    dump << val.pt.x << "," << val.pt.y << "," << val.distance << "\n";
    if (i % 1000 == 0) {
      std::cout << "\r[MAIN]:   Outputted " << i << "/" << res.distances.size()
                << " (" << i * 100 / res.distances.size() << "%) values."
                << std::flush;
    }
  }

  std::ofstream meta(std::string(argv[1]) + ".meta");
  meta << "farthest point: (" << res.farthest.x << ", " << res.farthest.y << ")"
       << std::endl;
  meta << "bounding box: ((x min, x max), (y min, y max)) = ("
       << "(" << res.contained.min.x << ", " << res.contained.max.x << "), "
       << "(" << res.contained.min.y << ", " << res.contained.max.y << "))"
       << std::endl;

  return 0;
}
