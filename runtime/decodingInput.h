#pragma once

#include "common.h"
#include "iTensor.h"

#include <memory>
#include <optional>

namespace nexly::runtime
{
    class DecodingInput
    {
    public:
        using TensorPtr = std::shared_ptr<ITensor const>;

        DecodingInput(SizeType maxLength, SizeType maxAttentionWindow, SizeType sinkTokenLength, SizeType maxBatchSize,
            TensorPtr logits, TensorPtr endIds)
            : step{ maxLength }
            , maxLength{ maxLength }
            , maxAttentionWindow{ maxAttentionWindow }
            , sinkTokenLength{ sinkTokenLength }
            , maxBatchSize{ maxBatchSize }
            , maxStopWordsLen{ 0 }
            , maxBadWordsLen{ 0 }
            , logits{ std::move(logits) }
            , endIds{ std::move(endIds) }
        {
            CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
            CHECK_WITH_INFO(static_cast<bool>(this->endIds), "Invalid endIds tensor");
        }

        SizeType step;
        SizeType maxLength;
        SizeType maxAttentionWindow;
        SizeType sinkTokenLength;
        SizeType maxBatchSize;
        SizeType maxStopWordsLen;
        SizeType maxBadWordsLen;
        TensorPtr logits;
        std::optional<std::vector<TensorPtr>>
            logitsVec;
        TensorPtr endIds;

        TensorPtr finished;
        TensorPtr sequenceLimitLength;
        TensorPtr embeddingBias;
        TensorPtr lengths;
        TensorPtr badWordsList;
        TensorPtr badWordsPtrs;
        TensorPtr badWordsLens;
        TensorPtr stopWordsList;
        TensorPtr stopWordsPtrs;
        TensorPtr stopWordsLens;
        TensorPtr noRepeatNgramSize;
        TensorPtr
            batchSlots;

        TensorPtr cacheIndirection;
    };

}