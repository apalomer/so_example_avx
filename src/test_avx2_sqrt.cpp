#include <chrono>
#include <iomanip>
#include <iostream>

#include <immintrin.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <fstream>

#include <omp.h>

void getData(u_char* data, size_t cols, std::vector<double>& info)
{
  //#pragma omp parallel for  // this slows it even more!
  for (size_t i = 0; i < cols; i++)
  {
    info[i] = sqrt(double((data[cols + i] << 8) + data[2 * cols + i]) / 64.0);
  }
}

void getDataAVX2(u_char* data, size_t cols, std::vector<double>& info)
{
  __m256d dividend = _mm256_set_pd(1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0);
  for (size_t i = 0; i < cols / 4; i++)
  {
    __m256d divisor = _mm256_set_pd(double((data[4 * i + 3 + cols] << 8) + data[4 * i + 2 * cols + 3]),
                                    double((data[4 * i + 2 + cols] << 8) + data[4 * i + 2 * cols + 2]),
                                    double((data[4 * i + 1 + cols] << 8) + data[4 * i + 2 * cols + 1]),
                                    double((data[4 * i + cols] << 8) + data[4 * i + 2 * cols]));
    _mm256_storeu_pd(&info[0] + 4 * i, _mm256_sqrt_pd(_mm256_mul_pd(divisor, dividend)));
  }
}

inline __m256d cvt_scale_sqrt(__m128i vi)
{
  __m256d vd = _mm256_cvtepi32_pd(vi);
  vd = _mm256_mul_pd(vd, _mm256_set1_pd(1. / 64.));
  return _mm256_sqrt_pd(vd);
}

void getDataAVX2_vector_unpack(const u_char* __restrict data, size_t cols, std::vector<double>& info_vec)
{
  double* info = &info_vec[0];  // our stores don't alias the vector control-block
                                // but gcc doesn't figure that out, so read the pointer into a local

  for (size_t i = 0; i < cols / 4; i += 4)
  {
    // 128-bit vectors because packed int->double expands to 256-bit
    __m128i a = _mm_loadu_si128((const __m128i*)&data[4 * i + cols]);  // 16 elements
    __m128i b = _mm_loadu_si128((const __m128i*)&data[4 * i + 2 * cols]);
    __m128i lo16 = _mm_unpacklo_epi8(b, a);  // a<<8 | b  packed 16-bit integers
    __m128i hi16 = _mm_unpackhi_epi8(b, a);

    __m128i lo_lo = _mm_unpacklo_epi16(lo16, _mm_setzero_si128());
    __m128i lo_hi = _mm_unpackhi_epi16(lo16, _mm_setzero_si128());

    __m128i hi_lo = _mm_unpacklo_epi16(hi16, _mm_setzero_si128());
    __m128i hi_hi = _mm_unpackhi_epi16(hi16, _mm_setzero_si128());

    _mm256_storeu_pd(&info[4 * (i + 0)], cvt_scale_sqrt(lo_lo));
    _mm256_storeu_pd(&info[4 * (i + 1)], cvt_scale_sqrt(lo_hi));
    _mm256_storeu_pd(&info[4 * (i + 2)], cvt_scale_sqrt(hi_lo));
    _mm256_storeu_pd(&info[4 * (i + 3)], cvt_scale_sqrt(hi_hi));
  }
}

void getDataAVX2f(const u_char* __restrict data, size_t cols, std::vector<float>& info_vec)
{
  __m256 dividend = _mm256_set_ps(1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0, 1 / 64.0);
  float* info = &info_vec[0];
  for (size_t i = 0; i < cols / 8; i++)
  {
    __m256 divisor = _mm256_set_ps(float((data[8 * i + 7 + cols] << 8) + data[8 * i + 2 * cols + 7]),
                                   float((data[8 * i + 6 + cols] << 8) + data[8 * i + 2 * cols + 6]),
                                   float((data[8 * i + 5 + cols] << 8) + data[8 * i + 2 * cols + 5]),
                                   float((data[8 * i + 4 + cols] << 8) + data[8 * i + 2 * cols + 4]),
                                   float((data[8 * i + 3 + cols] << 8) + data[8 * i + 2 * cols + 3]),
                                   float((data[8 * i + 2 + cols] << 8) + data[8 * i + 2 * cols + 2]),
                                   float((data[8 * i + 1 + cols] << 8) + data[8 * i + 2 * cols + 1]),
                                   float((data[8 * i + 0 + cols] << 8) + data[8 * i + 2 * cols + 0]));
    _mm256_storeu_ps(info + 8 * i, _mm256_sqrt_ps(_mm256_mul_ps(divisor, dividend)));
  }
}

inline __m256 cvt_scale_sqrt_f(__m256i vi)
{
  __m256 vd = _mm256_cvtepi32_ps(vi);
  vd = _mm256_mul_ps(vd, _mm256_set1_ps(1. / 64.));
  return _mm256_sqrt_ps(vd);
}

void getDataAVX2_vector_unpack_f(const u_char* __restrict data, size_t cols, std::vector<float>& info_vec)
{
  float* info = &info_vec[0];  // our stores don't alias the vector control-block
                               // but gcc doesn't figure that out, so read the pointer into a local

  for (size_t i = 0; i < cols / 8; i += 2)
  {
    // 128-bit vectors because packed int->double expands to 256-bit
    __m128i a = _mm_loadu_si128((const __m128i*)&data[8 * i + cols]);
    __m128i b = _mm_loadu_si128((const __m128i*)&data[8 * i + 2 * cols]);

    __m128i lo16 = _mm_unpacklo_epi8(b, a);  // a<<8 | b  packed 16-bit integers
    __m128i hi16 = _mm_unpackhi_epi8(b, a);

    __m128i lo_lo = _mm_unpacklo_epi16(lo16, _mm_setzero_si128());
    __m128i lo_hi = _mm_unpackhi_epi16(lo16, _mm_setzero_si128());
    __m256 lo = _mm256_castps128_ps256(lo_lo);
    lo = _mm256_insertf128_ps(lo, lo_hi, 1);

    __m128i hi_lo = _mm_unpacklo_epi16(hi16, _mm_setzero_si128());
    __m128i hi_hi = _mm_unpackhi_epi16(hi16, _mm_setzero_si128());
    __m256 hi = _mm256_castps128_ps256(hi_lo);
    hi = _mm256_insertf128_ps(hi, hi_hi, 1);

    _mm256_storeu_ps(info + 8 * (i + 0), cvt_scale_sqrt_f(lo));
    _mm256_storeu_ps(info + 8 * (i + 1), cvt_scale_sqrt_f(hi));
  }
}

int main(int argc, char** argv)
{
  /*
u_char data[] = {
  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0x11, 0xf,  0xf,  0xf,  0xf,  0xf,  0x10, 0xf,  0xf,
  0xf,  0xf,  0xe,  0x10, 0x10, 0xf,  0x10, 0xf,  0xf,  0x10, 0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0x10, 0x10, 0xf,
  0x10, 0xf,  0xe,  0xf,  0xf,  0x10, 0xf,  0xf,  0x10, 0xf,  0xf,  0xf,  0xf,  0x10, 0xf,  0xf,  0xf,  0xf,  0xf,
  0xf,  0xf,  0xf,  0x10, 0xf,  0xf,  0xf,  0x10, 0xf,  0xf,  0xf,  0xf,  0xe,  0xf,  0xf,  0xf,  0xf,  0xf,  0x10,
  0x10, 0xf,  0xf,  0xf,  0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2,
  0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2,
  0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2,
  0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2,
  0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xf2, 0xd3, 0xd1, 0xca, 0xc6, 0xd2, 0xd2, 0xcc, 0xc8, 0xc2, 0xd0, 0xd0,
  0xca, 0xc9, 0xcb, 0xc7, 0xc3, 0xc7, 0xca, 0xce, 0xca, 0xc9, 0xc2, 0xc8, 0xc2, 0xbe, 0xc2, 0xc0, 0xb8, 0xc4, 0xbd,
  0xc5, 0xc9, 0xbc, 0xbf, 0xbc, 0xb5, 0xb6, 0xc1, 0xbe, 0xb7, 0xb9, 0xc8, 0xb9, 0xb2, 0xb2, 0xba, 0xb4, 0xb4, 0xb7,
  0xad, 0xb2, 0xb6, 0xab, 0xb7, 0xaf, 0xa7, 0xa8, 0xa5, 0xaa, 0xb0, 0xa3, 0xae, 0xa9, 0xa0, 0xa6, 0xa5, 0xa8, 0x9f,
  0xa0, 0x9e, 0x94, 0x9f, 0xa3, 0x9d, 0x9f, 0x9c, 0x9e, 0x99, 0x9a, 0x97, 0x4,  0x5,  0x4,  0x5,  0x4,  0x4,  0x5,
  0x5,  0x5,  0x4,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x4,  0x4,  0x4,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,
  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,
  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x4,  0x4,  0x4,  0x5,  0x5,  0x5,  0x4,  0x4,
  0x5,  0x5,  0x5,  0x5,  0x4,  0x5,  0x5,  0x4,  0x4,  0x6,  0x4,  0x4,  0x6,  0x5,  0x4,  0x5,  0xf0, 0xf0, 0xf0,
  0xf0, 0xf0, 0xf0, 0xe0, 0xf0, 0xe0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
  0xf0, 0xf0, 0xe0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
  0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
  0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0, 0xf0,
  0xf0
};
size_t cols = 80;

for (size_t i = 0; i < cols; i++)
{
  double peakd = double((data[cols + i] << 8) + data[2 * cols + i]) / 64.0;
  double peakf = double(float((data[cols + i] << 8) + data[2 * cols + i]) / 64.0f);
  if (fabs(peakd - peakf) > 1e-5)
  {
    std::cout << i << std::endl;
  }
}
*/

  // Check inputs
  if (argc != 6)
  {
    std::cout << "Usage: " << argv[0] << " path_to_images n_lines angle_ini angle_step angle_end" << std::endl;
    return 0;
  }

  // Read inputs
  std::string path(argv[1]);
  int n_lines(atoi(argv[2]));
  int angle_ini(atoi(argv[3]));
  int angle_step(atoi(argv[4]));
  int angle_end(atoi(argv[5]));

  // Compute
  std::vector<double> info;
  std::vector<double> info_avx2;
  std::vector<double> info_avx22;
  std::vector<float> info_avx2f;
  std::vector<float> info_avx22f;
  double time_normal(0);
  double time_avx2(0);
  double time_avx22(0);
  double time_avx2f(0);
  double time_avx22f(0);
  for (int n = 0; n < n_lines; n++)
  {
    for (int ang = angle_ini; ang <= angle_end; ang += angle_step)
    {
      // Image name
      std::ostringstream oss;
      oss << path;
      oss << "/line_" << n << "_" << ang << ".png";
      std::cout << "Image line_" << n << "_" << ang << ".png\r";

      // Load image
      cv::Mat img = cv::imread(oss.str(), cv::ImreadModes::IMREAD_UNCHANGED);
      if (img.empty())
      {
        std::cout << "Error loading imgae " << oss.str() << std::endl;
        return 0;
      }
      u_char* data = img.data;
      size_t cols = size_t(img.cols);

      // Allocate
      info.clear();
      info.reserve(cols);
      info_avx2.clear();
      info_avx2.reserve(cols);
      info_avx22.clear();
      info_avx22.reserve(cols);
      info_avx2f.clear();
      info_avx2f.reserve(cols);
      info_avx22f.clear();
      info_avx22f.reserve(cols);

      // Compute with normal maths
      auto tstart_normal = std::chrono::high_resolution_clock::now();
      getData(data, cols, info);
      time_normal += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart_normal).count();

      // Compute with avx
      auto tstart_avx2 = std::chrono::high_resolution_clock::now();
      getDataAVX2(data, cols, info_avx2);
      time_avx2 += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart_avx2).count();

      // Compute with avx
      auto tstart_avx22 = std::chrono::high_resolution_clock::now();
      getDataAVX2_vector_unpack(data, cols, info_avx22);
      time_avx22 += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart_avx22).count();

      // Compute with avx
      auto tstart_avx2f = std::chrono::high_resolution_clock::now();
      getDataAVX2f(data, cols, info_avx2f);
      time_avx2f += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart_avx2f).count();

      // Compute with avx
      auto tstart_avx22f = std::chrono::high_resolution_clock::now();
      getDataAVX2_vector_unpack_f(data, cols, info_avx22f);
      time_avx22f += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tstart_avx22f).count();

      // Check results
      double max_err = 1e-3;
      for (size_t i = 0; i < cols; i++)
      {
        if (fabs(info[i] - info_avx2[i]) > max_err)
        {
          std::cout << "AVX2: " << n << "," << ang << "," << i << " (" << info[i] << "," << info_avx2[i] << ")"
                    << std::endl;
        }
        if (fabs(info[i] - info_avx22[i]) > max_err)
        {
          std::cout << "AVX22: " << n << "," << ang << "," << i << " (" << info[i] << "," << info_avx22[i] << ")"
                    << std::endl;
        }
        if (fabs(info[i] - double(info_avx2f[i])) > max_err)
        {
          std::cout << "AVX2F: " << n << "," << ang << "," << i << " (" << info[i] << "," << info_avx2f[i] << ")"
                    << std::endl;
        }
        if (fabs(info[i] - double(info_avx22f[i])) > max_err)
        {
          std::cout << "AVX22F: " << n << "," << ang << "," << i << " (" << info[i] << "," << info_avx22f[i] << ")"
                    << std::endl;
        }
      }
    }
  }
  std::cout << std::endl;

  // Display difference
  std::cout << "Time normal: " << time_normal * 1000 << " ms" << std::endl;
  std::cout << "Time AVX2:   " << time_avx2 * 1000 << " ms" << std::endl;
  std::cout << "Time AVX22:  " << time_avx22 * 1000 << " ms" << std::endl;
  std::cout << "Time AVX2f:  " << time_avx2f * 1000 << " ms" << std::endl;
  std::cout << "Time AVX22f: " << time_avx22f * 1000 << " ms" << std::endl;
  std::cout << "Time improvement Noramal/AVX2:   " << time_normal / time_avx2 << std::endl;
  std::cout << "Time improvement Noramal/AVX22:  " << time_normal / time_avx22 << std::endl;
  std::cout << "Time improvement Noramal/AVX2f:  " << time_normal / time_avx2f << std::endl;
  std::cout << "Time improvement Noramal/AVX22f: " << time_normal / time_avx22f << std::endl;

  // Exit
  return 0;
}
