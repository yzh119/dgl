/*!
 *  Copyright (c) 2020 by Contributors.
 * \file dgl/omp.h
 * \brief DGL's openmp parallel API wrapper. 
 */

#ifndef DGL_OMP_H_
#define DGL_OMP_H_

#if defined(_OPENMP)
#include "omp.h"
#endif
#include "dmlc/logging.h"

namespace at {

/*!
 * Following code was modified from PyTorch:
 * - https://github.com/pytorch/pytorch/blob/93973ee6993c97c8b9d60c4f720423bc625073ea/aten/src/ATen/ParallelOpenMP.h
 */
template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F& f) {
  if (grain_size < 0)
    LOG(FATAL) << "grain size: " + grain_size + " must be greater or equal to 0.";
  //at::internal::lazy_init_num_threads();
  if (begin >= end) {
    return;
  }
#ifdef _OPENMP
  std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
  std::exception_ptr eptr;
  // Work around memory leak when using 1 thread in nested "omp parallel"
  // caused by some buggy OpenMP versions and the fact that omp_in_parallel()
  // returns false when omp_get_max_threads() == 1 inside nested "omp parallel"
  // See issue gh-32284

#pragma omp parallel if (omp_get_max_threads() > 1 && !omp_in_parallel() && ((end - begin) > grain_size))
  {
    // choose number of tasks based on grain size and number of threads
    // can't use num_threads clause due to bugs in GOMP's thread pool (See #32008)
    int64_t num_threads = omp_get_num_threads();
    if (grain_size > 0) {
      num_threads = std::min(num_threads, divup((end - begin), grain_size));
    }

    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = divup((end - begin), num_threads);
    int64_t begin_tid = begin + tid * chunk_size;
    if (begin_tid < end) {
      try {
        f(begin_tid, std::min(end, chunk_size + begin_tid));
      } catch (...) {
        if (!err_flag.test_and_set()) {
          eptr = std::current_exception();
        }
      }
    }
  }
  if (eptr) {
    std::rethrow_exception(eptr);
  }
#else
  f(begin, end);
#endif
}

#endif
