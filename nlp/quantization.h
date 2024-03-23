#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nexly
{
    namespace kernels
    {

        template <typename T>
        void invokeQuantization(
            int8_t* dst, T const* src, const int64_t size, float const* scalePtr, cudaStream_t stream = 0, int maxGirdSize = 0);

        template <typename T>
        void invokePerTokenQuantization(
            int8_t* dst, T const* src, const int64_t numRows, const int64_t numCols, float* scalePtr, cudaStream_t stream = 0);

    }
}
