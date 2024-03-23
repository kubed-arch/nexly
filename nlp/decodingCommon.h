
#pragma once

#include <cstdint>
#include <curand_kernel.h>

namespace nexly
{
namespace kernels
{

class FinishedState
{
public:
    static auto constexpr empty()
    {
        return FinishedState{0};
    }

    static auto constexpr finished()
    {
        return FinishedState{kFinished};
    }

    static auto constexpr skipDecoding()
    {
        return FinishedState{kSkipDecoding};
    }

    static auto constexpr finishedEOS()
    {
        return FinishedState{kFinishedEos};
    }

    static auto constexpr finishedMaxLength()
    {
        return FinishedState{kFinishedMaxLength};
    }

    static auto constexpr finishedStopWords()
    {
        return FinishedState{kFinishedStopWords};
    }

    __host__ __device__ void constexpr setFinishedEOS()
    {
        mState |= kFinishedEos;
    }

    __host__ __device__ bool constexpr isFinishedEOS()
    {
        return anyBitSet(kFinishedEos);
    }

    __host__ __device__ void constexpr setFinishedStopWords()
    {
        mState |= kFinishedStopWords;
    }

    __host__ __device__ bool constexpr isFinishedStopWords()
    {
        return anyBitSet(kFinishedStopWords);
    }

    __host__ __device__ void constexpr setFinishedMaxLength()
    {
        mState |= kFinishedMaxLength;
    }

    __host__ __device__ bool constexpr isFinishedMaxLength()
    {
        return anyBitSet(kFinishedMaxLength);
    }

    __host__ __device__ void constexpr setFinished()
    {
        mState |= kFinished;
    }

    __host__ __device__ bool constexpr isFinished() const
    {
        return anyBitSet(kFinished);
    }

    __host__ __device__ void constexpr setSkipDecoding()
    {
        mState = kSkipDecoding;
    }

    __host__ __device__ bool constexpr isSkipDecoding() const
    {
        return anyBitSet(kSkipDecoding);
    }

    using UnderlyingType = uint8_t;

private:
    __host__ __device__ constexpr FinishedState(UnderlyingType state)
        : mState(state)
    {
    }

    static UnderlyingType constexpr kFinishedEos{1u << 0};
    static UnderlyingType constexpr kFinishedStopWords{1u << 1};
    static UnderlyingType constexpr kFinishedMaxLength{1u << 2};
    static UnderlyingType constexpr kFinished{kFinishedEos | kFinishedStopWords | kFinishedMaxLength};
    static UnderlyingType constexpr kSkipDecoding{1u << 3};

    __host__ __device__ bool constexpr anyBitSet(UnderlyingType bits) const
    {
        return (mState & bits) != 0;
    }

    UnderlyingType mState{};
};

static_assert(!FinishedState::empty().isFinished());
static_assert(!FinishedState::empty().isSkipDecoding());
static_assert(FinishedState::finished().isFinished());
static_assert(FinishedState::skipDecoding().isSkipDecoding());
static_assert(FinishedState::finishedEOS().isFinishedEOS());
static_assert(FinishedState::finishedStopWords().isFinishedStopWords());
static_assert(FinishedState::finishedMaxLength().isFinishedMaxLength());

void invokeCurandInitialize(
    curandState_t* state, int const* batchSlots, const size_t batchSize, uint64_t randomSeed, cudaStream_t stream);

void invokeCurandBatchInitialize(curandState_t* states, int const* batchSlots, const size_t batchSize,
    uint64_t const* randomSeeds, cudaStream_t stream);

template <typename T>
void invokeAddBiasSoftMax(T* logits, T** logitsPtrs, T* probs, T const* bias, int32_t const* endIds,
    FinishedState const* finished, int32_t const* batchSlots, int32_t batchSize, int32_t maxBatchSize,
    int32_t beamWidth, int32_t vocabSize, int32_t vocabSizePadded, bool skipSoftMax, bool batchSlotsLogits,
    cudaStream_t stream);

template <typename T>
void invokeScatterDecodingParams(T const* src, T* dst, int const* batchSlots, int batchSize, cudaStream_t stream);
}
}
