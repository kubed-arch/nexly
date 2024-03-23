#pragma once

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include "../common/assert.h"

namespace nexly
{
    namespace kernels
    {

        enum class KVIdxType : int32_t
        {
            K_IDX = 0,
            V_IDX = 1
        };

        struct KVBlockArray
        {

            int32_t mMaxBlocksPerSeq;
            int32_t mMaxSeqs;
            int32_t mTokensPerBlock;
            int32_t mTokensPerBlockLog2;
            int32_t mMaxAttentionWindow;
            int32_t mSinkTokens;
            int32_t mCyclicCacheLen;
            int32_t mBubbleLen;
            bool mEnableOneMoreBlock;
            int64_t* data;

            KVBlockArray() {}

            KVBlockArray(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t sizePerToken,
                int32_t maxAttentionWindow, int32_t sinkTokenLen, bool onlyKorV)
                : mMaxSeqs(batchSize)
                , mMaxBlocksPerSeq(maxBlocksPerSeq)
                , mTokensPerBlock(tokensPerBlock)
                , mMaxAttentionWindow(maxAttentionWindow)
                , mSinkTokens(sinkTokenLen)
                , data(nullptr)
            {
                float const tokensPerBlockSeqLog2 = log2(mTokensPerBlock);
                CHECK_WITH_INFO(
                    ceil(tokensPerBlockSeqLog2) == floor(tokensPerBlockSeqLog2), "tokensPerBlock must be power of 2");
                CHECK_WITH_INFO(static_cast<int64_t>(mMaxSeqs - 1) * mMaxBlocksPerSeq * 2 + maxBlocksPerSeq
                    <= std::numeric_limits<int32_t>::max(),
                    "kv cache is too large for gpt_attention_plugin");
                mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
                auto sinkTokensInLastBlock = mSinkTokens % mTokensPerBlock;
                mBubbleLen = sinkTokensInLastBlock == 0 ? 0 : mTokensPerBlock - sinkTokensInLastBlock;
                mEnableOneMoreBlock = (maxBlocksPerSeq - 1) * tokensPerBlock >= mMaxAttentionWindow + mBubbleLen;
                mCyclicCacheLen = (mEnableOneMoreBlock) ? mMaxAttentionWindow + mTokensPerBlock - mSinkTokens
                    : mMaxAttentionWindow - mSinkTokens;
            }

            __host__ __device__ inline bool isSinkToken(int32_t tokenIdx)
            {
                return tokenIdx < mSinkTokens;
            }

            __host__ __device__ inline void** getRowPtr(KVIdxType kvIdx, int32_t seqIdx)
            {
                return reinterpret_cast<void**>(
                    data + seqIdx * mMaxBlocksPerSeq * 2 + static_cast<int32_t>(kvIdx) * mMaxBlocksPerSeq);
            }

            __host__ __device__ inline int32_t getKVTokenIdx(int32_t tokenIdx)
            {
                if (!isSinkToken(tokenIdx))
                {
                    return mSinkTokens + mBubbleLen + (tokenIdx - mSinkTokens) % mCyclicCacheLen;
                }
                return tokenIdx;
            }

            __host__ __device__ inline void* getBlockPtr(void** pointer, int32_t tokenIdx)
            {
                return pointer[tokenIdx >> mTokensPerBlockLog2];
            }

            __host__ __device__ inline void* getBlockPtr(int32_t seqIdx, int32_t tokenIdx, KVIdxType kvIdx)
            {
                return getBlockPtr(getRowPtr(kvIdx, seqIdx), tokenIdx);
            }

            __host__ __device__ inline void* getKBlockPtr(int32_t seqIdx, int32_t tokenIdx)
            {
                return getBlockPtr(seqIdx, tokenIdx, KVIdxType::K_IDX);
            }

            __host__ __device__ inline void* getVBlockPtr(int32_t seqIdx, int32_t tokenIdx)
            {
                return getBlockPtr(seqIdx, tokenIdx, KVIdxType::V_IDX);
            }

            __host__ __device__ inline int32_t getLocalIdx(int32_t globalIdx)
            {
                return globalIdx & ((1 << mTokensPerBlockLog2) - 1);
            }

            __host__ __device__ inline int32_t getKVLocalIdx(
                int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx)
            {
                return headIdx * mTokensPerBlock * dimsPerHead + getLocalIdx(globalTokenIdx) * dimsPerHead + channelIdx;
            }
        };

        struct KVBlockArrayForContextFMHA
        {

            int32_t mMaxBlocksPerSeq;
            int32_t mMaxSeqs;
            int32_t mTokensPerBlock;
            int32_t mTokensPerBlockLog2;
            int64_t* data;

            KVBlockArrayForContextFMHA() {}

            KVBlockArrayForContextFMHA(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t sizePerToken)
                : mMaxSeqs(batchSize)
                , mMaxBlocksPerSeq(maxBlocksPerSeq)
                , mTokensPerBlock(tokensPerBlock)
                , data(nullptr)
            {
                float const tokensPerBlockSeqLog2 = log2(mTokensPerBlock);
                CHECK_WITH_INFO(
                    ceil(tokensPerBlockSeqLog2) == floor(tokensPerBlockSeqLog2), "tokensPerBlock must be power of 2");
                CHECK_WITH_INFO(static_cast<int64_t>(mMaxSeqs - 1) * mMaxBlocksPerSeq * 2 + maxBlocksPerSeq
                    <= std::numeric_limits<int32_t>::max(),
                    "kv cache is too large for gpt_attention_plugin");
                mTokensPerBlockLog2 = static_cast<int>(tokensPerBlockSeqLog2);
            }

            __host__ __device__ inline void** getRowPtr(KVIdxType kvIdx, int32_t seqIdx)
            {
                return reinterpret_cast<void**>(
                    data + seqIdx * mMaxBlocksPerSeq * 2 + static_cast<int32_t>(kvIdx) * mMaxBlocksPerSeq);
            }

            __host__ __device__ inline void* getBlockPtr(void** pointer, int32_t tokenIdx)
            {
                return pointer[tokenIdx >> mTokensPerBlockLog2];
            }

            __host__ __device__ inline void* getBlockPtr(int32_t seqIdx, int32_t tokenIdx, KVIdxType kvIdx)
            {
                return getBlockPtr(getRowPtr(kvIdx, seqIdx), tokenIdx);
            }

            __host__ __device__ inline void* getKBlockPtr(int32_t seqIdx, int32_t tokenIdx)
            {
                return getBlockPtr(seqIdx, tokenIdx, KVIdxType::K_IDX);
            }

            __host__ __device__ inline void* getVBlockPtr(int32_t seqIdx, int32_t tokenIdx)
            {
                return getBlockPtr(seqIdx, tokenIdx, KVIdxType::V_IDX);
            }

            __host__ __device__ inline int32_t getLocalIdx(int32_t globalIdx)
            {
                return globalIdx & ((1 << mTokensPerBlockLog2) - 1);
            }

            __host__ __device__ inline int32_t getKVLocalIdx(
                int32_t globalTokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx)
            {
                return headIdx * mTokensPerBlock * dimsPerHead + getLocalIdx(globalTokenIdx) * dimsPerHead + channelIdx;
            }
        };

        struct KVLinearBuffer
        {

            int32_t mMaxSeqs;
            int32_t mMaxSeqLen;
            int32_t mBytesPerSeq;
            int32_t mMaxAttentionWindow;
            int32_t mSinkTokens;
            int32_t mCyclicCacheLen;
            int32_t mBubbleLen;
            int32_t mValidRowsPerSeq;
            bool mEnableOneMoreBlock;
            int8_t* data;

            KVLinearBuffer() {}

            KVLinearBuffer(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock, int32_t sizePerToken,
                int32_t maxAttentionWindow, int32_t sinkTokenLen, bool onlyKorV)
                : mMaxSeqs(batchSize)
                , mMaxSeqLen(tokensPerBlock)
                , mBytesPerSeq(tokensPerBlock* sizePerToken)
                , mMaxAttentionWindow(maxAttentionWindow)
                , mSinkTokens(sinkTokenLen)
                , data(nullptr)
            {
                CHECK_WITH_INFO(
                    static_cast<int64_t>(mMaxSeqs - 1) * mBytesPerSeq * 2 + mBytesPerSeq <= std::numeric_limits<int32_t>::max(),
                    "kv cache is too large for gpt_attention_plugin");
                mCyclicCacheLen = mMaxAttentionWindow - mSinkTokens;
                mBubbleLen = 0;
                mValidRowsPerSeq = (onlyKorV) ? 1 : 2;
                mEnableOneMoreBlock = false;
            }

            __host__ __device__ inline bool isSinkToken(int32_t tokenIdx)
            {
                return tokenIdx < mSinkTokens;
            }

            __host__ __device__ inline void** getRowPtr(KVIdxType kvIdx, int32_t seqIdx)
            {
                return reinterpret_cast<void**>(data + seqIdx * mBytesPerSeq * mValidRowsPerSeq
                    + static_cast<int32_t>(kvIdx) * mBytesPerSeq * (mValidRowsPerSeq - 1));
            }

            __host__ __device__ inline int32_t getKVTokenIdx(int32_t tokenIdx)
            {
                if (!isSinkToken(tokenIdx))
                {
                    return mSinkTokens + (tokenIdx - mSinkTokens) % mCyclicCacheLen;
                }
                return tokenIdx;
            }

            __host__ __device__ inline void* getBlockPtr(void** pointer, int32_t tokenIdx)
            {
                return reinterpret_cast<void*>(pointer);
            }

            __host__ __device__ inline void* getKBlockPtr(int32_t seqIdx, int32_t)
            {
                return reinterpret_cast<void*>(getRowPtr(KVIdxType::K_IDX, seqIdx));
            }

            __host__ __device__ inline void* getVBlockPtr(int32_t seqIdx, int32_t)
            {
                return reinterpret_cast<void*>(getRowPtr(KVIdxType::V_IDX, seqIdx));
            }

            __host__ __device__ inline int32_t getKVLocalIdx(
                int32_t tokenIdx, int32_t headIdx, int32_t dimsPerHead, int32_t channelIdx)
            {
                return headIdx * mMaxSeqLen * dimsPerHead + tokenIdx * dimsPerHead + channelIdx;
            }
        };

    }
}