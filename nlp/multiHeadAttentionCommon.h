#pragma once

#include <limits.h>
#include <stdint.h>

namespace nexly
{
    namespace kernels
    {
        /// <summary>
        /// Enumeration representing different data types.
        /// </summary>
        enum Data_type
        {
            DATA_TYPE_BOOL,     ///< Boolean data type.
            DATA_TYPE_FP16,     ///< 16-bit floating point data type.
            DATA_TYPE_FP32,     ///< 32-bit floating point data type.
            DATA_TYPE_INT4,     ///< 4-bit integer data type.
            DATA_TYPE_INT8,     ///< 8-bit integer data type.
            DATA_TYPE_INT32,    ///< 32-bit integer data type.
            DATA_TYPE_BF16,     ///< Brain Floating Point (BF16) data type.
            DATA_TYPE_E4M3,     ///< Extended 4-bit mixed precision data type.
            DATA_TYPE_E5M2      ///< Extended 5-bit mixed precision data type.
        };

        /// <summary>
        /// Constant representing the architecture code for SM 7.0.
        /// </summary>
        constexpr int32_t kSM_70 = 70;

        /// <summary>
        /// Constant representing the architecture code for SM 7.2.
        /// </summary>
        constexpr int32_t kSM_72 = 72;

        /// <summary>
        /// Constant representing the architecture code for SM 7.5.
        /// </summary>
        constexpr int32_t kSM_75 = 75;

        /// <summary>
        /// Constant representing the architecture code for SM 8.0.
        /// </summary>
        constexpr int32_t kSM_80 = 80;

        /// <summary>
        /// Constant representing the architecture code for SM 8.6.
        /// </summary>
        constexpr int32_t kSM_86 = 86;

        /// <summary>
        /// Constant representing the architecture code for SM 8.9.
        /// </summary>
        constexpr int32_t kSM_89 = 89;

        /// <summary>
        /// Constant representing the architecture code for SM 9.0.
        /// </summary>
        constexpr int32_t kSM_90 = 90;

    }
}
