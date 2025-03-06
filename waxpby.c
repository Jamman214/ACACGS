#include "waxpby.h"
#include <omp.h>
#include <immintrin.h>
#include <stdio.h>

/**
 * @brief Compute the update of a vector with the sum of two scaled vectors
 * 
 * @param n Number of vector elements
 * @param alpha Scalars applied to x
 * @param x Input vector
 * @param beta Scalars applied to y
 * @param y Input vector
 * @param w Output vector
 * @return int 0 if no error
 */
int waxpby (const int n, const double * const x, const double beta, const double * const y, double * const w) {  
  int parallelN = (n/4)*4;
  #pragma omp parallel for num_threads(4)
  for (int i=0; i<parallelN; i+=4) {
    _mm256_store_pd(w+i, _mm256_add_pd(_mm256_load_pd(x+i), _mm256_mul_pd(_mm256_set1_pd(beta), _mm256_load_pd(y+i))));
  }
  for (int i=parallelN; i<n; i++) {
    w[i] = x[i] + beta * y[i];
  }

  return 0;
}
