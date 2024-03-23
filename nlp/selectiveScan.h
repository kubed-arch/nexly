#pragma once

#include "../common/assert.h"
#include "../common/cudaUtils.h"

namespace nexly
{
    namespace kernels
    {

        struct SSMParamsBase
        {
            using index_t = uint32_t;

            int batch, dim, seqlen, dstate;
            bool is_variable_B;
            bool is_variable_C;

            bool delta_softplus;

            void* __restrict__ A_ptr;
            void* __restrict__ B_ptr;
            void* __restrict__ C_ptr;
            void* __restrict__ D_ptr;
            void* __restrict__ u_ptr;
            void* __restrict__ delta_ptr;
            void* __restrict__ delta_bias_ptr;
            void* __restrict__ out_ptr;
            void* __restrict__ x_ptr;
            void* __restrict__ z_ptr;
        };


        template <typename input_t, typename weight_t>
        void invokeSelectiveScan(SSMParamsBase& params, cudaStream_t stream);

        template <typename input_t, typename weight_t>
        void invokeSelectiveScanUpdate(SSMParamsBase& params, cudaStream_t stream);
    }
}
