#ifndef CUDIJKSTRA_MATRIX_HPP
#define CUDIJKSTRA_MATRIX_HPP

#include <vector>

namespace cudijkstra {
using data_t = double;

template <typename T, bool is_const = true> struct pass_const { using type = const T; };
template <typename T> struct pass_const<T, false> { using type = T; };
template <typename T, bool is_const = true> using pass_const_t = typename pass_const<T, is_const>::type;

template <typename T>
class matrix {
public:
  template <bool is_const = false>
  struct row {
    pass_const_t<matrix, is_const> &m;
    size_t rowid;
    typename std::enable_if_t<!is_const, T &> operator[](size_t col) { return m.at(rowid, col); }
    const T &operator[](size_t col) const { return m.at(rowid, col); }
  };
  matrix(size_t width, size_t height) : w{width}, h{height}, buf{std::vector<T>(width * height)} {}

  row<false> operator[](size_t rowid) { return row{ .m = *this, .rowid = rowid}; }
  row<true> operator[](size_t rowid) const { return row { .m = *this, .rowid = rowid }; }

  T &at(size_t rowid, size_t col) { return buf[rowid * w + col]; }
  const T &at(size_t rowid, size_t col) const { return buf[rowid * w + col]; }

  T *raw() { return buf.data(); }
  const T *raw() const { return buf.data(); }

  [[nodiscard]] size_t width() const { return w; }
  [[nodiscard]] size_t height() const { return h; }

private:
  size_t w;
  size_t h;
  std::vector<T> buf;
};
}

#endif // CUDIJKSTRA_MATRIX_HPP
