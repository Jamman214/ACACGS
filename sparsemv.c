#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

#include "sparsemv.h"

/**
 * @brief Compute matrix vector product (y = A*x)
 * 
 * @param A Known matrix
 * @param x Known vector
 * @param y Return vector
 * @return int 0 if no error
 */
int sparsemv(struct mesh *A, const double * const x, double * const y)
{
  const int nrow = (const int) A->local_nrow;
  #pragma omp parallel for num_threads(4)
  for (int i=0; i< nrow; i++) {
    const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
    const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
    const int cur_nnz = (const int) A->nnz_in_row[i];
    int j = 0;
    __m256d sumVec = _mm256_setzero_pd();
    if(j< cur_nnz-15) {
      // sums 16 iterations of j at once
      sumVec = _mm256_add_pd(
        sumVec,
        _mm256_add_pd(
          _mm256_add_pd(
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]])),
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j + 4), _mm256_set_pd(x[cur_inds[j+7]], x[cur_inds[j+6]], x[cur_inds[j+5]], x[cur_inds[j+4]]))
          ),
          _mm256_add_pd(
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j + 8), _mm256_set_pd(x[cur_inds[j+11]], x[cur_inds[j+10]], x[cur_inds[j+9]], x[cur_inds[j+8]])),
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j + 12), _mm256_set_pd(x[cur_inds[j+15]], x[cur_inds[j+14]], x[cur_inds[j+13]], x[cur_inds[j+12]]))
          )
        )
      );
      j+=16;
    }

    if (j< cur_nnz-7) {
      // sums 16 iterations of 8 at once
      sumVec = _mm256_add_pd(
        sumVec,
        _mm256_add_pd(
          _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]])),
          _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j + 4), _mm256_set_pd(x[cur_inds[j+7]], x[cur_inds[j+6]], x[cur_inds[j+5]], x[cur_inds[j+4]]))
        )
      );
      j+=8;
    }

    if (j< cur_nnz-3) {
      // sums 4 iterations of j at once
      sumVec = _mm256_add_pd(
        sumVec,
        _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]))
      );
      j+=4;
    }

    double sum = 0.0;
    for (; j< cur_nnz; j++) {
      // sums remaining iterations of j
      sum += cur_vals[j] * x[cur_inds[j]];
    }
    y[i] = sumVec[0] + sumVec[1] + sumVec[2] + sumVec[3] + sum;
  }
  return 0;
}