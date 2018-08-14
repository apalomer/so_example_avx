#include <sys/time.h>
#include <iomanip>
#include <iostream>

#include <immintrin.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <fstream>

inline double timestamp()
{
  struct timeval tp;
  gettimeofday(&tp, nullptr);
  return double(tp.tv_sec) + tp.tv_usec / 1000000.;
}

void getData(u_char* data, size_t cols, std::vector<double>& info)
{
  info.resize(cols);
  for (size_t i = 0; i < cols; i++)
  {
    info[i] = (double(data[cols + i] << 8) + double(data[2 * cols + i])) / 64.0;
    ;
  }
}

void getDataAVX2(u_char* data, size_t cols, std::vector<double>& info)
{
  __m256d dividend = _mm256_set_pd(1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0);
  info.resize(cols);
  __m256d result;
  for (size_t i = 0; i < cols / 4; i++)
  {
    __m256d divisor = _mm256_set_pd((double(data[4 * i + 3 + cols] << 8) + double(data[4 * i + 2 * cols + 3])),
                                    (double(data[4 * i + 2 + cols] << 8) + double(data[4 * i + 2 * cols + 2])),
                                    (double(data[4 * i + 1 + cols] << 8) + double(data[4 * i + 2 * cols + 1])),
                                    (double(data[4 * i + cols] << 8) + double(data[4 * i + 2 * cols])));
    result = _mm256_mul_pd(divisor, dividend);
    info[size_t(4 * i)] = result[0];
    info[size_t(4 * i + 1)] = result[1];
    info[size_t(4 * i + 2)] = result[2];
    info[size_t(4 * i + 3)] = result[3];
  }
}

int main(int argc, char** argv)
{
  // Check inputs
  if (argc == 1)
  {
    std::cout << "Usage: " << argv[0] << " image" << std::endl;
    return 0;
  }

  // Load image
  cv::Mat img = cv::imread(argv[1], cv::ImreadModes::IMREAD_UNCHANGED);
  if (img.empty())
  {
    std::cout << "Error loading imgae " << argv[1] << std::endl;
    return 0;
  }
  u_char* data = img.data;
  size_t cols = size_t(img.cols);

  // Normal
  std::cout << "Computing with normal way" << std::endl;
  std::vector<double> info;
  double tstart_normal = timestamp();
  getData(data, cols, info);
  double time_normal = timestamp() - tstart_normal;

  // AVX2
  std::cout << "Computing with avx" << std::endl;
  std::vector<double> info_avx2;
  double tstart_avx2 = timestamp();
  getDataAVX2(data, cols, info_avx2);
  double time_avx2 = timestamp() - tstart_avx2;

  // Display difference
  std::cout << "Time normal: " << time_normal << " s" << std::endl;
  std::cout << "Time AVX2:   " << time_avx2 << " s" << std::endl;
  std::cout << "Time improvement AVX2: " << time_normal / time_avx2 << std::endl;

  // Write to file
  std::ofstream file;
  file.open("out.csv");
  for (size_t i = 0; i < cols; i++)
  {
    file << info[size_t(i)] << "," << info_avx2[size_t(i)];
    file << std::endl;
  }
  file.close();

  // Exit
  return 0;
}
