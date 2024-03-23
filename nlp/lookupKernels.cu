#include "../common/cudaTypeUtils.cuh"
#include "lookupKernels.h"

using namespace nexly::common;

namespace nexly
{
    namespace kernels
    {
        /// <summary>
        /// CUDA kernel for performing a lookup operation.
        /// </summary>
        /// <typeparam name="T">The type of the output and weight.</typeparam>
        /// <typeparam name="Idx">The type of the input and offset.</typeparam>
        /// <param name="output">Pointer to the output data.</param>
        /// <param name="input">Pointer to the input data.</param>
        /// <param name="weight">Pointer to the weight data.</param>
        /// <param name="batch_size">The batch size.</param>
        /// <param name="offset">The offset.</param>
        /// <param name="size">The size.</param>
        /// <param name="n_embed">The number of embeddings.</param>
        template <typename T, typename Idx>
        __global__ void lookup_kernel(T* output, Idx const* input, T const* weight, const Idx batch_size, const Idx offset,
            const Idx size, int const n_embed)
        {
            for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * n_embed;
                index += blockDim.x * gridDim.x)
            {
                int const word_index = input[index / n_embed] - offset;
                int const col_index = index % n_embed;
                T embedding;
                if (word_index < 0 || word_index >= size)
                {
                    embedding = T(0.f);
                }
                else
                {
                    embedding = weight[word_index * n_embed + col_index];
                }
                output[index] = embedding;
            }
        }

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
            int const n_embed, cudaStream_t stream)
        {
            dim3 grid(min(batch_size, 65536));
            dim3 block(min(n_embed, 512));
            lookup_kernel<T, Idx> << <grid, block, 0, stream >> > (out, input, weight, batch_size, offset, size, n_embed);
        }

#define INSTANTIATE_LOOK_UP(T, Idx)                                                                                    \
    template void invokeLookUp<T, Idx>(T * out, const Idx* input, const T* weight, const Idx batch_size,               \
        const Idx offset, const Idx size, const int n_embed, cudaStream_t stream)

        INSTANTIATE_LOOK_UP(float, int);

        /// <summary>
        /// Template instantiation for half input and integer index.
        /// </summary>
        INSTANTIATE_LOOK_UP(half, int);

#ifdef ENABLE_BF16
        /// <summary>
        /// Template instantiation for __nv_bfloat16 input and integer index.
        /// </summary>
        INSTANTIATE_LOOK_UP(__nv_bfloat16, int);
#endif

    }
}
