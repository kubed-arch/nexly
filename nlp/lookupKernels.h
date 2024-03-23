#pragma once

#include "../common/cudaUtils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nexly
{
    namespace kernels
    {
        /// <summary>
        /// Invokes a lookup operation.
        /// </summary>
        /// <typeparam name="T">The type of the output and weight.</typeparam>
        /// <typeparam name="Idx">The type of the input and offset.</typeparam>
        /// <param name="out">Pointer to the output data.</param>
        /// <param name="input">Pointer to the input data.</param>
        /// <param name="weight">Pointer to the weight data.</param>
        /// <param name="batch_size">The batch size.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="size">The size.</param>
        /// <param name="n_embed">The number of embeddings.</param>
        /// <param name="stream">The CUDA stream for asynchronous execution.</param>
        template <typename T, typename Idx>
        void invokeLookUp(T* out, Idx const* input, T const* weight, const Idx batch_size, const Idx offset, const Idx size,
            int const n_embed, cudaStream_t stream = 0);
    }
}
