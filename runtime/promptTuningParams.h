#pragma once

#include "bufferManager.h"
#include "common.h"
#include "iTensor.h"

#include <utility>

namespace nexly::runtime
{

    template <typename TTensor>
    class GenericPromptTuningParams
    {
    public:
        using TensorPtr = TTensor;
        using SizeType = nexly::runtime::SizeType;

        explicit GenericPromptTuningParams(
            TensorPtr embeddingTable = TensorPtr(), TensorPtr tasks = TensorPtr(), TensorPtr vocabSize = TensorPtr())
            : embeddingTable{ std::move(embeddingTable) }
            , tasks{ std::move(tasks) }
        , vocabSize{ std::move(vocabSize) } {};

        TensorPtr embeddingTable;
        TensorPtr tasks;
        TensorPtr vocabSize;

        std::vector<bool>
            promptTuningEnabled;
    };

    class PromptTuningParams : public GenericPromptTuningParams<ITensor::SharedPtr>
    {
    public:
        using TensorPtr = ITensor::SharedPtr;
        using SizeType = GenericPromptTuningParams::SizeType;

        explicit PromptTuningParams(
            TensorPtr embeddingTable = nullptr, TensorPtr tasks = nullptr, TensorPtr vocabSize = nullptr)
            : GenericPromptTuningParams(std::move(embeddingTable), std::move(tasks), std::move(vocabSize))
        {
        }

        void fillTasksTensor(TensorPtr tasksHost, const SizeType batchSize, const SizeType numContextRequests,
            std::vector<SizeType> const& reqBeamWidths, std::vector<SizeType> const& reqPromptLengths,
            BufferManager const& manager, bool packedInput);
    };

}
