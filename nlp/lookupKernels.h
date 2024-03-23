#pragma once

#include "../common/cudaUtils.h"
#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nexly
{
    namespace kernels
    {
        template <typename T, typename Idx>
        void invokeLookUp(T* out, Idx const* input, T const* weight, const Idx batch_size, const Idx offset, const Idx size,
            int const n_embed, cudaStream_t stream = 0);

    }
}
