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
    info[i] = sqrt((double(data[cols + i] << 8) + double(data[2 * cols + i])) / 64.0);
    ;
  }
}
void getDataAVX512(u_char* data, size_t cols, std::vector<double>& info)
{
  __m512d dividend = _mm512_set_pd(1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0);
  info.resize(cols);
  __m512d result;
  for (size_t i = 0; i < cols / 8; i++)
  {
    __m512d divisor = _mm512_set_pd((double(data[4 * i + 7 + cols] << 8) + double(data[4 * i + 2 * cols + 7])),
                                    (double(data[4 * i + 6 + cols] << 8) + double(data[4 * i + 2 * cols + 6])),
                                    (double(data[4 * i + 5 + cols] << 8) + double(data[4 * i + 2 * cols + 5])),
                                    (double(data[4 * i + 4 + cols] << 8) + double(data[4 * i + 2 * cols + 4])),
                                    (double(data[4 * i + 3 + cols] << 8) + double(data[4 * i + 2 * cols + 3])),
                                    (double(data[4 * i + 2 + cols] << 8) + double(data[4 * i + 2 * cols + 2])),
                                    (double(data[4 * i + 1 + cols] << 8) + double(data[4 * i + 2 * cols + 1])),
                                    (double(data[4 * i + cols] << 8) + double(data[4 * i + 2 * cols])));
    result = _mm512_sqrt_pd(_mm512_mul_pd(divisor, dividend));
    info[size_t(4 * i)] = result[0];
    info[size_t(4 * i + 1)] = result[1];
    info[size_t(4 * i + 2)] = result[2];
    info[size_t(4 * i + 3)] = result[3];
    info[size_t(4 * i + 4)] = result[4];
    info[size_t(4 * i + 5)] = result[5];
    info[size_t(4 * i + 6)] = result[6];
    info[size_t(4 * i + 7)] = result[7];
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

  // AVX512
  std::vector<double> info_avx512;
  std::cout << "Computing with avx512" << std::endl;
  double tstart_avx512 = timestamp();
  getDataAVX512(data, cols, info_avx512);
  double time_avx512 = timestamp() - tstart_avx512;

  // Display difference
  std::cout << "Time normal: " << time_normal << " s" << std::endl;
  std::cout << "Time AVX512: " << time_avx512 << " s" << std::endl;
  std::cout << "Time improvement AVX512: " << time_normal / time_avx512 << std::endl;

  // Write to file
  std::ofstream file;
  file.open("out.csv");
  for (size_t i = 0; i < cols; i++)
  {
    file << info[size_t(i)] << "," << info_avx512[size_t(i)];
    file << std::endl;
  }
  file.close();

  // Exit
  return 0;
}
