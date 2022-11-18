#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include "io.hpp"

using namespace cu_geodesic;

std::vector<segment> read_segm_list(std::istream &strm) {
  std::string line;
  std::getline(strm, line);
  if(line.empty()) return {};

  std::stringstream polygon_strm(line);
  std::vector<point> points;
  while(!polygon_strm.eof()) {
    double x;
    double y;
    polygon_strm >> x >> y;
    points.push_back({x, y});

    if(!polygon_strm.eof()) {
      char c;
      polygon_strm >> c;
      if(c != ';') {
        std::cerr << "[WARNING]: expected ';', got '" << c << "' (ignoring)" << std::endl;
      }
    }
  }

  std::vector<segment> segments;
  for(size_t end = 0; end < points.size(); end++) {
    segment s {
      .begin = points[end == 0 ? (points.size() - 1) : (end - 1)],
      .end = points[end],
      .a = 0.0,
      .b = 0.0
    };
    // (y2-y1)/(x2-x1)
    s.a = (s.end.y - s.begin.y) / (s.end.x - s.begin.x);
    // y1 - a*x1
    s.b = s.begin.y - s.begin.x * s.a;
    segments.push_back(s);
  }

  return segments;
}

loader::res_t loader::load(const std::string &filename) {
  std::ifstream strm(filename);
  if(!strm.is_open()) {
    std::cerr << "[ERROR]: can't open " << filename << ". Exiting." << std::endl;
    std::exit(-1);
  }

  std::string line;
  std::getline(strm, line);
  point start{0,0};
  std::stringstream(line) >> start.x >> start.y;
  std::cout << "[MESSAGE]: starting point is (x, y) = (" << start.x << ", " << start.y << ")" << std::endl;

  if(strm.eof()) {
    std::cerr << "[ERROR]: no polygon data. Exiting." << std::endl;
    std::exit(-2);
  }
  auto segm_poly = read_segm_list(strm);
  std::cout << "[MESSAGE]: polygon boundary has " << segm_poly.size() << " segments." << std::endl;
  for(const auto &segm : segm_poly) {
    std::cout << "[MESSAGE]:    -> segment(from=("
              << segm.begin.x << "," << segm.begin.y << "), to=("
              << segm.end.x << "," << segm.end.y << "), ax+b="
              << segm.a << "x+" << segm.b << ")" << std::endl;
  }
  double xmin = std::numeric_limits<double>::max();
  double ymin = std::numeric_limits<double>::max();
  double xmax = std::numeric_limits<double>::min();
  double ymax = std::numeric_limits<double>::min();
  for(const auto &segm : segm_poly) {
    if(segm.begin.x < xmin) xmin = segm.begin.x;
    if(segm.begin.x > xmax) xmax = segm.begin.x;
    if(segm.end.x < xmin) xmin = segm.end.x;
    if(segm.end.x > xmax) xmax = segm.end.x;

    if(segm.begin.y < ymin) ymin = segm.begin.y;
    if(segm.begin.y > ymax) ymax = segm.begin.y;
    if(segm.end.y < ymin) ymin = segm.end.y;
    if(segm.end.y > ymax) ymax = segm.end.y;
  }
  bounding_box box{
      .min = { .x = xmin, .y = ymin },
      .max = { .x = xmax, .y = ymax }
  };
  std::cout << "[MESSAGE]: bounding box ranges [xmin, xmax] = [" << xmin << ", " << xmax
            << "] and [ymin, ymax] = [" << ymin << ", " << ymax << "]" << std::endl;

  decltype(polygon::holes) holes;
  while(!strm.eof()) {
    auto segm_hole = read_segm_list(strm);
    if(!segm_hole.empty())
      holes.push_back(segm_hole);
  }
  std::cout << "[MESSAGE]: polygon has " << holes.size() << " holes." << std::endl;
  for(size_t hole_id = 0; hole_id < holes.size(); hole_id++) {
    std::cout << "[MESSAGE]:     -> hole #" << hole_id << ": (" << holes[hole_id].size() << " segments)" << std::endl;
    for(const auto &segm : holes[hole_id]) {
      std::cout << "[MESSAGE]:       -> segment(from=("
                << segm.begin.x << "," << segm.begin.y << "), to=("
                << segm.end.x << "," << segm.end.y << "), ax+b="
                << segm.a << "x+" << segm.b << ")" << std::endl;
    }
  }

  return { polygon{ .boundary = segm_poly, .holes = holes }, start, box };
}

/*  ==== FILE FORMAT ====  */
/* -> 1st line: starting point in the form of `x y`
 * -> 2nd line: hull of the polygon in the form of `x1 y1;x2 y2`
 * -> 3rd+ lines: holes in the polygon; one per line; same form as above
 * */