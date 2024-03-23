#include "../common/cudaTypeUtils.cuh"
#include "lookupKernels.h"

using namespace nexly::common;

namespace nexly
{
    namespace kernels
    {
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
        INSTANTIATE_LOOK_UP(half, int);

#ifdef ENABLE_BF16
        INSTANTIATE_LOOK_UP(__nv_bfloat16, int);
#endif

    }
}
