#include "ddot.h"
#include <omp.h>
#include <immintrin.h>

/**
 * @brief Compute the dot product of two vectors
 * 
 * @param n Number of vector elements
 * @param x Input vector
 * @param y Input vector
 * @param result Pointer to scalar result value
 * @return int 0 if no error
 */
int ddot (const int n, const double * const x, const double * const y, double * const result) {  
  double local_result = 0.0;
  int parallelN = (n/4)*4;
  __m256d sumVec = _mm256_setzero_pd();
  if (y==x){
    #pragma omp parallel num_threads(4) reduction(+:local_result) firstprivate(sumVec)
    {
      #pragma omp for
      for (int i=0; i<parallelN; i+=4) {
        const __m256d xVec = _mm256_load_pd(x+i);
        sumVec = _mm256_add_pd(sumVec, _mm256_mul_pd(xVec, xVec));
      }
      local_result += sumVec[0] + sumVec[1] + sumVec[2] + sumVec[3];
    }
    for (int i=parallelN; i<n; i++) {
      local_result += x[i]*x[i];
    }

  } else {
    #pragma omp parallel num_threads(4) reduction(+:local_result) firstprivate(sumVec)
    {
      #pragma omp for
      for (int i=0; i<parallelN; i+=4) {
        sumVec = _mm256_add_pd(sumVec, _mm256_mul_pd(_mm256_load_pd(x+i), _mm256_load_pd(y+i)));
      }
      local_result += sumVec[0] + sumVec[1] + sumVec[2] + sumVec[3];
    }
    for (int i=parallelN; i<n; i++) {
      local_result += x[i]*y[i];
    }
  }
  
  *result = local_result;

  return 0;
}