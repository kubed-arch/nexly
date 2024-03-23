#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nexly
{
    namespace kernels
    {

        /// <summary>
        /// Types of attention masks.
        /// </summary>
        enum class AttentionMaskType
        {
            PADDING = 0,
            CAUSAL = 1,
            BIDIRECTIONAL = 2,
            BIDIRECTIONALGLM = 3
        };

        /// <summary>
        /// Types of position embeddings.
        /// </summary>
        enum class PositionEmbeddingType : int8_t
        {
            kLEARNED_ABSOLUTE = 0,
            kROPE_GPTJ = 1,
            kROPE_GPT_NEOX = 2,
            kALIBI = 3,
            kALIBI_WITH_SCALE = 4,
            kRELATIVE = 5
        };

        /// <summary>
        /// Types of rotary scaling.
        /// </summary>
        enum class RotaryScalingType : int8_t
        {
            kNONE = 0,
            kLINEAR = 1,
            kDYNAMIC = 2,
        };

        /// <summary>
        /// Parameters for building decoder information.
        /// </summary>
        /// <typeparam name="AttentionMaskDataType">The data type for attention mask.</typeparam>
        template <typename AttentionMaskDataType>
        struct BuildDecoderInfoParams
        {
            int* seqQOffsets;                           ///< Pointer to sequence query offsets.
            int* seqKVOffsets;                          ///< Pointer to sequence key/value offsets.
            int* paddingOffsets;                        ///< Pointer to padding offsets.
            AttentionMaskDataType* attentionMask;       ///< Pointer to attention mask data.
            int const* seqQLengths;                     ///< Pointer to sequence query lengths.
            int const* seqKVLengths;                    ///< Pointer to sequence key/value lengths.
            int batchSize;                              ///< Size of the batch.
            int maxSeqLength;                           ///< Maximum sequence length.
            int attentionWindowSize;                    ///< Attention window size.
            int sinkTokenLength;                        ///< Length of the sink token.
            int numTokens;                              ///< Number of tokens.
            AttentionMaskType attentionMaskType;        ///< Type of attention mask.
        };

        /// <summary>
        /// Invokes the kernel to build decoder information.
        /// </summary>
        /// <typeparam name="T">The data type for attention mask.</typeparam>
        /// <param name="params">Parameters for building decoder information.</param>
        /// <param name="stream">CUDA stream for asynchronous execution.</param>
        template <typename T>
        void invokeBuildDecoderInfo(BuildDecoderInfoParams<T> const& params, cudaStream_t stream);

    }
}