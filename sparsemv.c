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
  for (int i=0; i< nrow-3; i+=4) {
    __m256d sumVecA = _mm256_setzero_pd();
    {
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
      const int cur_nnz = (const int) A->nnz_in_row[i];
      int j = 0;
      if(j< cur_nnz-15) {
        sumVecA = _mm256_add_pd(
          sumVecA,
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
        sumVecA = _mm256_add_pd(
          sumVecA,
          _mm256_add_pd(
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]])),
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j + 4), _mm256_set_pd(x[cur_inds[j+7]], x[cur_inds[j+6]], x[cur_inds[j+5]], x[cur_inds[j+4]]))
          )
        );
        j+=8;
      }

      if (j< cur_nnz-3) {
        sumVecA = _mm256_add_pd(
          sumVecA,
          _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]))
        );
        j+=4;
      }

      
      if (j< cur_nnz){
        double sum = cur_vals[j] * x[cur_inds[j]];
        j++;
        for (; j< cur_nnz; j++) {
          sum += cur_vals[j] * x[cur_inds[j]];
        }
        sumVecA[0] += sum;
      }
      

    }
    i++;
    __m256d sumVecB = _mm256_setzero_pd();
    {
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
      const int cur_nnz = (const int) A->nnz_in_row[i];
      int j = 0;
      if(j< cur_nnz-15) {
        sumVecB = _mm256_add_pd(
          sumVecB,
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
        sumVecB = _mm256_add_pd(
          sumVecB,
          _mm256_add_pd(
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]])),
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j + 4), _mm256_set_pd(x[cur_inds[j+7]], x[cur_inds[j+6]], x[cur_inds[j+5]], x[cur_inds[j+4]]))
          )
        );
        j+=8;
      }

      if (j< cur_nnz-3) {
        sumVecB = _mm256_add_pd(
          sumVecB,
          _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]))
        );
        j+=4;
      }

      if (j< cur_nnz){
        double sum = cur_vals[j] * x[cur_inds[j]];
        j++;
        for (; j< cur_nnz; j++) {
          sum += cur_vals[j] * x[cur_inds[j]];
        }
        sumVecB[0] += sum;
      }
    }
    i++;
    __m256d sumVecC = _mm256_setzero_pd();
    {
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
      const int cur_nnz = (const int) A->nnz_in_row[i];
      int j = 0;
      if(j< cur_nnz-15) {
        sumVecC = _mm256_add_pd(
          sumVecC,
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
        sumVecC = _mm256_add_pd(
          sumVecC,
          _mm256_add_pd(
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]])),
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j + 4), _mm256_set_pd(x[cur_inds[j+7]], x[cur_inds[j+6]], x[cur_inds[j+5]], x[cur_inds[j+4]]))
          )
        );
        j+=8;
      }

      if (j< cur_nnz-3) {
        sumVecC = _mm256_add_pd(
          sumVecC,
          _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]))
        );
        j+=4;
      }

      if (j< cur_nnz){
        double sum = cur_vals[j] * x[cur_inds[j]];
        j++;
        for (; j< cur_nnz; j++) {
          sum += cur_vals[j] * x[cur_inds[j]];
        }
        sumVecC[0] += sum;
      }
    }
    i++;
    __m256d sumVecD = _mm256_setzero_pd();
    {
      const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
      const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
      const int cur_nnz = (const int) A->nnz_in_row[i];
      int j = 0;
      if(j< cur_nnz-15) {
        sumVecD = _mm256_add_pd(
          sumVecD,
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
        sumVecD = _mm256_add_pd(
          sumVecD,
          _mm256_add_pd(
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]])),
            _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j + 4), _mm256_set_pd(x[cur_inds[j+7]], x[cur_inds[j+6]], x[cur_inds[j+5]], x[cur_inds[j+4]]))
          )
        );
        j+=8;
      }

      if (j< cur_nnz-3) {
        sumVecD = _mm256_add_pd(
          sumVecD,
          _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]))
        );
        j+=4;
      }

      if (j< cur_nnz){
        double sum = cur_vals[j] * x[cur_inds[j]];
        j++;
        for (; j< cur_nnz; j++) {
          sum += cur_vals[j] * x[cur_inds[j]];
        }
        sumVecD[0] = sum;
      }
    }
    i-=3;

    _mm256_store_pd(
      y+i,
      _mm256_add_pd(
        _mm256_add_pd(
          _mm256_set_pd(sumVecD[0], sumVecC[0], sumVecB[0], sumVecA[0]),
          _mm256_set_pd(sumVecD[1], sumVecC[1], sumVecB[1], sumVecA[1])
        ),
        _mm256_add_pd(
          _mm256_set_pd(sumVecD[2], sumVecC[2], sumVecB[2], sumVecA[2]),
          _mm256_set_pd(sumVecD[3], sumVecC[3], sumVecB[3], sumVecA[3])
        )
      )
    );
  }
  for (int i=(nrow/4)*4; i< nrow; i++) {
    const double * const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];
    const int * const cur_inds = (const int * const) A->ptr_to_inds_in_row[i];
    const int cur_nnz = (const int) A->nnz_in_row[i];

    __m256d sumVec = _mm256_setzero_pd();
    int j = 0;
    if(j< cur_nnz-15) {
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
      sumVec = _mm256_add_pd(
        sumVec,
        _mm256_mul_pd(_mm256_loadu_pd(cur_vals + j), _mm256_set_pd(x[cur_inds[j+3]], x[cur_inds[j+2]], x[cur_inds[j+1]], x[cur_inds[j]]))
      );
      j+=4;
    }

    double sum = 0.0;
    if (j<cur_nnz) {
      double sum = cur_vals[j] * x[cur_inds[j]];
      j++;
      for (; j< cur_nnz; j++) {
        sum += cur_vals[j] * x[cur_inds[j]];
      }
    }


    y[i] = sum + sumVec[0] + sumVec[1] + sumVec[2] + sumVec[3];
  }
  return 0;
}