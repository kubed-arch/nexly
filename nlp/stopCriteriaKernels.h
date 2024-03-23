#pragma once

#include "decodingCommon.h"
#include <cuda_runtime.h>

namespace nexly
{
namespace kernels
{
void invokeStopWordsCriterion(int32_t const** outputIds, int32_t const** parentIds, int32_t const** stopWords,
    FinishedState* finished, int32_t const* sequenceLengths, int32_t const* batchSlots, int32_t const* stopWordsLen,
    int32_t maxStopWordsLen, int32_t batchSize, int32_t beamWidth, int32_t maxSeqLen, cudaStream_t stream);

void invokeLengthCriterion(FinishedState* finished, int32_t* finishedSum, uint32_t const* sequenceLimitLength,
    int32_t const* sequenceLengths, int32_t const* batchSlots, int32_t batchSize, int32_t beamWidth,
    cudaStream_t stream);
}
}
