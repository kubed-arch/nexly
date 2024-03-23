
#pragma once

#include "../common/logger.h"
#include "../executor/executor.h"
#include "../runtime/bufferManager.h"
#include "../runtime/iTensor.h"
#include "../runtime/samplingConfig.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace nexly::batch_manager
{

enum LlmRequestState_t
{
    REQUEST_STATE_UNKNOWN = 0,
    REQUEST_STATE_CONTEXT_INIT = 1,
    REQUEST_STATE_GENERATION_IN_PROGRESS = 2,
    REQUEST_STATE_GENERATION_COMPLETE = 3
};

template <typename TTensor, typename TStream = runtime::BufferManager::CudaStreamPtr>
class GenericLlmRequest
{
public:
    using SizeType = runtime::SizeType;
    using TokenIdType = runtime::TokenIdType;
    using RequestIdType = std::uint64_t;
    using VecTokens = std::vector<TokenIdType>;
    using VecLogProbs = std::vector<float>;
    using BeamTokens = std::vector<VecTokens>;
    using TensorPtr = TTensor;
    using LogitsPostProcessor = std::function<TensorPtr(RequestIdType, TensorPtr&, BeamTokens const&, TStream)>;

    GenericLlmRequest(RequestIdType requestId, SizeType maxNewTokens, std::shared_ptr<VecTokens> inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType> endId = std::nullopt,
        std::optional<SizeType> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType> promptVocabSize = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt, bool returnLogProbs = false,
        bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<std::shared_ptr<VecTokens>> draftTokens = std::nullopt,
        std::optional<TensorPtr> draftLogits = std::nullopt, bool excludeInputFromOutput = false,
        std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt)
        : mRequestId(requestId)
        , mPromptLen(inputTokens->size())
        , mMaxNewTokens(maxNewTokens)
        , mSamplingConfig(samplingConfig)
        , mState(REQUEST_STATE_CONTEXT_INIT)
        , mIsStreaming(isStreaming)
        , mEndId(endId)
        , mPadId(padId)
        , mSeqSlot(-1)
        , mLogitsPostProcessor(logitsPostProcessor)
        , mOrigPromptLen(mPromptLen)
        , mMaxSentTokenPos(mPromptLen - 1)
        , mEmbeddingBias(std::move(embeddingBias))
        , mBadWordsList(std::move(badWordsList))
        , mStopWordsList(std::move(stopWordsList))
        , mPromptEmbeddingTable(std::move(promptEmbeddingTable))
        , mPromptVocabSize(promptVocabSize)
        , mLoraWeights(std::move(loraWeights))
        , mLoraConfig(std::move(loraConfig))
        , mReturnLogProbs(returnLogProbs)
        , mContextChunkSize(std::nullopt)
        , mContextCurrentPosition(0)
        , mLogProbs(samplingConfig.beamWidth)
        , mCumLogProbs(samplingConfig.beamWidth)
        , mDraftTokens(draftTokens.value_or(std::make_shared<VecTokens>()))
        , mDraftLogits(draftLogits)
        , mReturnContextLogits(returnContextLogits)
        , mReturnGenerationLogits(returnGenerationLogits)
        , mExcludeInputFromOutput(excludeInputFromOutput)
    {
        initialize(*inputTokens);
    }

    GenericLlmRequest(RequestIdType requestId, executor::Request const& req)
        : mRequestId(requestId)
        , mPromptLen(req.getInputTokenIds().size())
        , mMaxNewTokens(req.getMaxNewTokens())
        , mSamplingConfig(req.getSamplingConfig(), req.getSpeculativeDecodingConfig())
        , mState(REQUEST_STATE_CONTEXT_INIT)
        , mIsStreaming(req.getStreaming())
        , mEndId(req.getEndId())
        , mPadId(req.getPadId())
        , mSeqSlot(-1)
        , mOrigPromptLen(mPromptLen)
        , mMaxSentTokenPos(mPromptLen - 1)
        , mReturnLogProbs(req.getOutputConfig().returnLogProbs)
        , mContextChunkSize(std::nullopt)
        , mContextCurrentPosition(0)
        , mLogProbs(mSamplingConfig.beamWidth)
        , mCumLogProbs(mSamplingConfig.beamWidth)
        , mDraftTokens(std::make_shared<VecTokens>())
        , mReturnContextLogits(req.getOutputConfig().returnContextLogits)
        , mReturnGenerationLogits(req.getOutputConfig().returnGenerationLogits)
        , mExcludeInputFromOutput(req.getOutputConfig().excludeInputFromOutput)
    {
        if (req.getEmbeddingBias())
        {
            mEmbeddingBias = executor::detail::toITensor(req.getEmbeddingBias().value());
            mEmbeddingBias.value()->unsqueeze(0);
        }
        if (req.getBadWords())
        {
            mBadWordsList = createListTensor(req.getBadWords().value());
        }
        if (req.getStopWords())
        {
            mStopWordsList = createListTensor(req.getStopWords().value());
        }

        auto pTuningConfig = req.getPromptTuningConfig();
        if (pTuningConfig)
        {
            mPromptEmbeddingTable = executor::detail::toITensor(pTuningConfig.value().getEmbeddingTable());
            TLLM_CHECK(mPromptEmbeddingTable.value()->getShape().nbDims == 2);
            mPromptVocabSize = mPromptEmbeddingTable.value()->getShape().d[0];
            mPromptEmbeddingTable.value()->unsqueeze(0);
        }

        auto loraConfig = req.getLoraConfig();
        if (loraConfig)
        {
            mLoraWeights = executor::detail::toITensor(loraConfig.value().getWeights());
            mLoraWeights.value()->unsqueeze(0);

            mLoraConfig = executor::detail::toITensor(loraConfig.value().getConfig());
            mLoraConfig.value()->unsqueeze(0);
        }

        auto speculativeDecodingConfig = req.getSpeculativeDecodingConfig();
        if (speculativeDecodingConfig)
        {
            mDraftTokens = std::make_shared<VecTokens>(speculativeDecodingConfig.value().getTokens());

            if (speculativeDecodingConfig.value().getLogits())
            {
                mDraftLogits = executor::detail::toITensor(speculativeDecodingConfig.value().getLogits().value());
            }

        }

        initialize(req.getInputTokenIds());
    }

    void validate(SizeType maxInputLen, SizeType maxSequenceLen)
    {
        if (mPromptLen > maxInputLen)
        {
            TLLM_THROW("Prompt length (%d) exceeds maximum input length (%d).", mPromptLen, maxInputLen);
        }

        if (mPromptLen + mMaxNewTokens > maxSequenceLen)
        {
            auto const maxNewTokens = maxSequenceLen - mPromptLen;
            TLLM_LOG_WARNING(
                "Number of requested output tokens (%d) exceeds maximum sequence length (%d). "
                "Number of requested output tokens is changed to (%d).",
                mMaxNewTokens, maxSequenceLen, maxNewTokens);
            mMaxNewTokens = maxNewTokens;
        }

        if (mSamplingConfig.beamWidth <= 0)
        {
            TLLM_THROW(
                "Requested value: %d for beamWidth is invalid. To de-activate beam searching "
                "set beamWidth to 1 instead.",
                mSamplingConfig.beamWidth);
        }
    }

    void setExcludeInputFromOutput(bool exclude)
    {
        mExcludeInputFromOutput = exclude;
    }

    [[nodiscard]] SizeType getNumTokens(SizeType beam) const
    {
        return mTokens.at(beam).size();
    }

    [[nodiscard]] SizeType getMaxBeamNumTokens() const
    {
        SizeType maxTokens = 0;
        for (SizeType beam = 0; beam < mSamplingConfig.beamWidth; ++beam)
        {
            maxTokens = std::max(maxTokens, static_cast<SizeType>(mTokens.at(beam).size()));
        }
        return maxTokens;
    }

    [[nodiscard]] TokenIdType getToken(SizeType beam, SizeType pos) const
    {
        return mTokens.at(beam).at(pos);
    }

    [[nodiscard]] VecTokens const& getTokens(SizeType beam) const
    {
        return mTokens.at(beam);
    }

    [[nodiscard]] BeamTokens const& getTokens() const
    {
        return mTokens;
    }

    [[nodiscard]] std::shared_ptr<VecTokens> const& getDraftTokens() const
    {
        return mDraftTokens;
    }

    [[nodiscard]] std::optional<TensorPtr> getDraftLogits() const
    {
        return mDraftLogits;
    }

    [[nodiscard]] bool hasDraftTokens() const
    {
        return mDraftTokens && !mDraftTokens->empty();
    }

    [[nodiscard]] SizeType getMaxNumGeneratedTokens() const
    {
        return getMaxBeamNumTokens() - mPromptLen;
    }

    void addNewToken(TokenIdType token, SizeType beam)
    {
        mTokens.at(beam).push_back(token);
    }

    void addNewTokens(VecTokens const& beamTokens)
    {
        assert(static_cast<size_t>(mSamplingConfig.beamWidth) == beamTokens.size());
        for (std::size_t beam = 0; beam < beamTokens.size(); ++beam)
        {
            auto const outputId = beamTokens[beam];
            mTokens.at(beam).push_back(outputId);
        }
    }

    void setGeneratedTokens(BeamTokens const& generatedBeamTokens)
    {
        assert(generatedBeamTokens.size() == static_cast<size_t>(mSamplingConfig.beamWidth));
        for (std::size_t beam = 0; beam < generatedBeamTokens.size(); ++beam)
        {
            auto& beamTokens = mTokens[beam];
            beamTokens.resize(mPromptLen);
            beamTokens.insert(beamTokens.end(), generatedBeamTokens[beam].begin(), generatedBeamTokens[beam].end());
        }
    }

    void pause(SizeType maxInputLen)
    {
        if (mSamplingConfig.beamWidth > 1)
        {
            for (std::size_t beam = 0; beam < mTokens.size(); ++beam)
            {
                auto& beamTokens = mTokens.at(beam);
                beamTokens.resize(mPromptLen);
                if (mReturnLogProbs)
                {
                    mLogProbs.at(beam).clear();
                }
            }
        }
        else
        {
            SizeType newPromptLen = std::min(maxInputLen, mPromptLen + getMaxNumGeneratedTokens());
            for (std::size_t beam = 0; beam < mTokens.size(); ++beam)
            {
                auto& beamTokens = mTokens.at(beam);
                beamTokens.resize(newPromptLen);

                if (mReturnLogProbs)
                {
                    auto& logProb = mLogProbs.at(beam);
                    logProb.resize(newPromptLen - mPromptLen);
                }
            }
            mMaxNewTokens -= (newPromptLen - mPromptLen);
            mPromptLen = newPromptLen;
        }
        mState = REQUEST_STATE_CONTEXT_INIT;
        mContextCurrentPosition = 0;
        mContextChunkSize = std::nullopt;
        mSeqSlot = -1;
    }

    [[nodiscard]] SizeType getMaxSentTokenPos() const
    {
        return mMaxSentTokenPos;
    }

    void setMaxSentTokenPos(SizeType pos)
    {
        mMaxSentTokenPos = pos;
    }

    [[nodiscard]] std::optional<TensorPtr> getPromptEmbeddingTable() const
    {
        return mPromptEmbeddingTable;
    }

    [[nodiscard]] std::optional<SizeType> getPromptVocabSize() const
    {
        return mPromptVocabSize;
    }

    [[nodiscard]] std::optional<TensorPtr> getLoraWeights() const
    {
        return mLoraWeights;
    }

    void setLoraWeights(TensorPtr weights)
    {
        mLoraWeights = weights;
    }

    void clearLoraWeights()
    {
        mLoraWeights = std::nullopt;
    }

    [[nodiscard]] std::optional<TensorPtr> getLoraConfig() const
    {
        return mLoraConfig;
    }

    void setLoraConfig(TensorPtr config)
    {
        mLoraConfig = config;
    }

    void clearLoraConfig()
    {
        mLoraConfig = std::nullopt;
    }

    [[nodiscard]] std::optional<TensorPtr> getEmbeddingBias() const
    {
        return mEmbeddingBias;
    }

    [[nodiscard]] std::optional<TensorPtr> getBadWordsList() const
    {
        return mBadWordsList;
    }

    [[nodiscard]] std::optional<TensorPtr> getStopWordsList() const
    {
        return mStopWordsList;
    }

    [[nodiscard]] bool returnLogProbs() const
    {
        return mReturnLogProbs;
    }

    void setReturnLogProbs(bool returnLogProbs)
    {
        mReturnLogProbs = returnLogProbs;
    }

    [[nodiscard]] std::vector<VecLogProbs> const& getLogProbs() const
    {
        return mLogProbs;
    }

    [[nodiscard]] VecLogProbs const& getLogProbs(SizeType beam) const
    {
        return mLogProbs.at(beam);
    }

    void setLogProbs(VecLogProbs const& logProbs, SizeType beam)
    {
        mLogProbs.at(beam).resize(mPromptLen - mOrigPromptLen);
        mLogProbs.at(beam).insert(mLogProbs.at(beam).end(), logProbs.begin(), logProbs.end());
    }

    [[nodiscard]] VecLogProbs const& getCumLogProbs() const
    {
        return mCumLogProbs;
    }

    void setCumLogProb(float cumLogProb, SizeType beam)
    {
        mCumLogProbs.at(beam) = cumLogProb;
    }

    [[nodiscard]] SizeType getOrigPromptLen() const
    {
        return mOrigPromptLen;
    }

    void setDraftTokens(std::shared_ptr<VecTokens> const& draftTokens)
    {
        mDraftTokens = draftTokens;
    }

    void setDraftLogits(std::optional<TensorPtr> const& draftLogits)
    {
        mDraftLogits = draftLogits;
    }

    SizeType getNumDraftTokens() const
    {
        return mDraftTokens->size();
    }

    void setReturnContextLogits(bool const returnContextLogits)
    {
        mReturnContextLogits = returnContextLogits;
    }

    [[nodiscard]] bool getReturnContextLogits() const
    {
        return mReturnContextLogits;
    }

    void setReturnGenerationLogits(bool const returnGenerationLogits)
    {
        mReturnGenerationLogits = returnGenerationLogits;
    }

    [[nodiscard]] bool getReturnGenerationLogits() const
    {
        return mReturnGenerationLogits;
    }

    [[nodiscard]] TensorPtr const& getContextLogitsHost() const
    {
        return mContextLogitsHost;
    }

    void setContextLogitsHost(TensorPtr contextLogitsHost)
    {
        mContextLogitsHost = std::move(contextLogitsHost);
    }

    void allocContextLogitsHost(SizeType vocabSizePadded, nvinfer1::DataType logitsDataType)
    {
        mContextLogitsHost = runtime::BufferManager::pinned(
            runtime::ITensor::makeShape({mPromptLen, vocabSizePadded}), logitsDataType);
    }

    [[nodiscard]] TensorPtr const& getGenerationLogitsHost() const
    {
        return mGenerationLogitsHost;
    }

    void setGenerationLogitsHost(TensorPtr generationLogitsHost)
    {
        mGenerationLogitsHost = std::move(generationLogitsHost);
    }

    void allocGenerationLogitsHost(SizeType vocabSizePadded, nvinfer1::DataType logitsDataType)
    {
        mGenerationLogitsHost = runtime::BufferManager::pinned(
            runtime::ITensor::makeShape({mSamplingConfig.beamWidth, mMaxNewTokens, vocabSizePadded}), logitsDataType);
    }

    [[nodiscard]] std::vector<TensorPtr> const& getGenerationLogitsFragments() const
    {
        return mGenerationLogitsFragments;
    }

    void addGenerationFragments(TensorPtr& genLogits)
    {
        mGenerationLogitsFragments.push_back(genLogits);
    }

    SizeType getGenerationLogitsFragmentsSize()
    {
        return mGenerationLogitsFragments.size();
    }

    void clearGenerationLogitsFragments()
    {
        mGenerationLogitsFragments.clear();
    }

    [[nodiscard]] bool isContextInitState() const noexcept
    {
        return mState == REQUEST_STATE_CONTEXT_INIT;
    }

    [[nodiscard]] bool isGenerationInProgressState() const noexcept
    {
        return mState == REQUEST_STATE_GENERATION_IN_PROGRESS;
    }

    [[nodiscard]] bool isGenerationCompleteState() const noexcept
    {
        return mState == REQUEST_STATE_GENERATION_COMPLETE;
    }

    [[nodiscard]] bool isFullContextRequest() const noexcept
    {
        return isContextInitState() && !mContextChunkSize;
    }

    [[nodiscard]] SizeType getContextCurrentPosition() const noexcept
    {
        return mContextCurrentPosition;
    }

    [[nodiscard]] SizeType getContextRemainingLength() const noexcept
    {
        return mPromptLen - getContextCurrentPosition();
    }

    [[nodiscard]] SizeType getContextChunkSize() const
    {
        TLLM_CHECK_WITH_INFO(
            isContextInitState() && mContextChunkSize, "The current request is not in context chunking state.");
        return mContextChunkSize.value();
    }

    void setContextChunkSize(SizeType size)
    {
        TLLM_CHECK_WITH_INFO(isContextInitState(), "Chunking is only possible during the context phase.");
        TLLM_CHECK_WITH_INFO(size >= 0, "The chunk size of context (%d) can't be negative.", size);
        mContextChunkSize = std::min(size, getContextRemainingLength());
    }

    [[nodiscard]] bool isLastContextChunk() const noexcept
    {
        return isFullContextRequest()
            || (isContextInitState() && getContextCurrentPosition() + getContextChunkSize() == mPromptLen);
    }

    [[nodiscard]] bool isFirstContextChunk() const noexcept
    {
        return isFullContextRequest() || getContextCurrentPosition() == 0;
    }

    void moveToNextContextChunk()
    {
        TLLM_CHECK_WITH_INFO(isContextInitState(), "Chunking is only possible during the context phase.");
        if (mContextChunkSize)
        {
            mContextCurrentPosition += getContextChunkSize();
            setContextChunkSize(0);
        }
        else
        {
            TLLM_CHECK_WITH_INFO(mContextCurrentPosition == 0, "Full context out of bounds.");
            mContextCurrentPosition = mPromptLen;
        }
    }

    std::optional<executor::Response> createResponse()
    {
        if (mState == batch_manager::REQUEST_STATE_GENERATION_COMPLETE
            || (mIsStreaming && mState == batch_manager::REQUEST_STATE_GENERATION_IN_PROGRESS))
        {
            executor::Result result;
            result.isFinal = mState == batch_manager::REQUEST_STATE_GENERATION_COMPLETE ? true : false;

            auto nbBeams = mSamplingConfig.beamWidth;
            auto maxNbTokens = getMaxBeamNumTokens();
            int nbTokensOut = mIsStreaming ? 1 : maxNbTokens;
            if (mExcludeInputFromOutput && !mIsStreaming)
            {
                nbTokensOut -= getOrigPromptLen();
            }

            result.outputTokenIds.resize(nbBeams);
            SizeType tokenPos = maxNbTokens - nbTokensOut;

            bool shouldSendResponse = (mState == batch_manager::REQUEST_STATE_GENERATION_COMPLETE)
                || (mIsStreaming && tokenPos > getMaxSentTokenPos());

            if (!shouldSendResponse)
            {
                return std::nullopt;
            }
            else
            {
                for (SizeType beam = 0; beam < nbBeams; ++beam)
                {
                    auto tokens = getTokens(beam);
                    auto nbTokens = mIsStreaming ? (tokenPos - getMaxSentTokenPos()) : tokens.size();
                    if (mExcludeInputFromOutput && !mIsStreaming)
                    {
                        nbTokens -= getOrigPromptLen();
                    }
                    if (nbTokens > 0)
                    {
                        result.outputTokenIds.at(beam).assign(
                            tokens.data() + tokenPos, tokens.data() + tokenPos + nbTokens);
                    }
                }

                if (returnLogProbs())
                {
                    result.cumLogProbs = getCumLogProbs();
                    result.logProbs = getLogProbs();
                }

                if (getReturnContextLogits())
                {
                    result.contextLogits = executor::detail::ofITensor(getContextLogitsHost());
                }

                if (getReturnGenerationLogits())
                {
                    result.generationLogits = executor::detail::ofITensor(getGenerationLogitsHost());
                }

                mMaxSentTokenPos = tokenPos;

                auto response = executor::Response(mRequestId, std::move(result));
                return response;
            }
        }
        else
        {
            return std::nullopt;
        }
    }

    RequestIdType mRequestId;
    SizeType mPromptLen;
    SizeType mMaxNewTokens;
    runtime::SamplingConfig mSamplingConfig;
    LlmRequestState_t mState;
    bool mIsStreaming;
    std::optional<SizeType> mEndId;
    std::optional<SizeType> mPadId;
    SizeType mSeqSlot;
    std::optional<LogitsPostProcessor> mLogitsPostProcessor;

protected:
    SizeType mOrigPromptLen;
    BeamTokens mTokens;
    SizeType mMaxSentTokenPos;

    std::optional<TensorPtr> mEmbeddingBias;
    std::optional<TensorPtr> mBadWordsList;
    std::optional<TensorPtr> mStopWordsList;

    std::optional<TensorPtr> mPromptEmbeddingTable;
    std::optional<SizeType> mPromptVocabSize;

    std::optional<TensorPtr> mLoraWeights;
    std::optional<TensorPtr> mLoraConfig;

    bool mReturnLogProbs;

    std::optional<SizeType> mContextChunkSize;
    SizeType mContextCurrentPosition;

    std::vector<VecLogProbs> mLogProbs;
    VecLogProbs mCumLogProbs;
    std::shared_ptr<VecTokens> mDraftTokens;
    std::optional<TensorPtr> mDraftLogits;

    bool mReturnContextLogits;
    bool mReturnGenerationLogits;
    TensorPtr mContextLogits;
    TensorPtr mContextLogitsHost;
    TensorPtr mGenerationLogits;
    TensorPtr mGenerationLogitsHost;
    std::vector<TensorPtr> mGenerationLogitsFragments;

    bool mExcludeInputFromOutput;

private:
    void initialize(VecTokens const& inputTokens)
    {
        mTokens = BeamTokens(mSamplingConfig.beamWidth, inputTokens);

        if ((mPromptEmbeddingTable.has_value() && !mPromptVocabSize.has_value())
            || (!mPromptEmbeddingTable.has_value() && mPromptVocabSize.has_value()))
        {
            std::string errStr
                = "Prompt embedding table and prompt vocab size tensors must both be provided for requests with "
                  "prompt "
                  "tuning enabled.";
            TLLM_THROW(errStr);
        }

        if (mDraftLogits.has_value() && mDraftTokens->empty())
        {
            TLLM_THROW("Draft tokens must be specified when draft logits are given.");
        }
    }

    TensorPtr createListTensor(std::list<VecTokens> const& wordsList)
    {
        std::vector<SizeType> offsets;
        VecTokens words;
        SizeType offsetCnt = 0;
        for (auto const& tokens : wordsList)
        {
            offsetCnt += tokens.size();
            offsets.push_back(offsetCnt);
            words.insert(words.end(), tokens.begin(), tokens.end());
        }
        offsets.resize(words.size(), -1);

        SizeType numWords = static_cast<SizeType>(words.size());
        auto shape = runtime::ITensor::makeShape({2, numWords});
        auto tensor = runtime::BufferManager::pinnedPool(shape, nvinfer1::DataType::kINT32);
        auto data = runtime::bufferCast<int32_t>(*tensor);
        std::memcpy(data, words.data(), numWords * sizeof(int32_t));
        std::memcpy(data + numWords, offsets.data(), numWords * sizeof(int32_t));
        tensor->unsqueeze(0);

        return tensor;
    }
};

class LlmRequest : public GenericLlmRequest<runtime::ITensor::SharedPtr>
{
public:
    using Base = GenericLlmRequest<runtime::ITensor::SharedPtr>;
    using TensorPtr = Base::TensorPtr;
    using SizeType = Base::SizeType;
    using TokenIdType = Base::TokenIdType;
    using RequestIdType = Base::RequestIdType;
    using VecLogProbs = Base::VecLogProbs;
    using BeamTokens = Base::BeamTokens;
    using VecTokens = Base::VecTokens;

    LlmRequest(RequestIdType requestId, SizeType maxNewTokens, std::shared_ptr<VecTokens> inputTokens,
        runtime::SamplingConfig const& samplingConfig, bool isStreaming, std::optional<SizeType> endId = std::nullopt,
        std::optional<SizeType> padId = std::nullopt, std::optional<TensorPtr> embeddingBias = std::nullopt,
        std::optional<TensorPtr> badWordsList = std::nullopt, std::optional<TensorPtr> stopWordsList = std::nullopt,
        std::optional<TensorPtr> promptEmbeddingTable = std::nullopt,
        std::optional<SizeType> promptVocabSize = std::nullopt, std::optional<TensorPtr> loraWeights = std::nullopt,
        std::optional<TensorPtr> loraConfig = std::nullopt, bool returnLogProbs = false,
        bool returnContextLogits = false, bool returnGenerationLogits = false,
        std::optional<std::shared_ptr<VecTokens>> draftTokens = std::nullopt,
        std::optional<TensorPtr> draftLogits = std::nullopt, bool excludeInputFromOutput = false,
        std::optional<LogitsPostProcessor> logitsPostProcessor = std::nullopt)
        : Base(requestId, maxNewTokens, std::move(inputTokens), samplingConfig, isStreaming, endId, padId,
            std::move(embeddingBias), std::move(badWordsList), std::move(stopWordsList),
            std::move(promptEmbeddingTable), promptVocabSize, std::move(loraWeights), std::move(loraConfig),
            returnLogProbs, returnContextLogits, returnGenerationLogits, std::move(draftTokens), std::move(draftLogits),
            excludeInputFromOutput, std::move(logitsPostProcessor))
    {
    }

    LlmRequest(RequestIdType requestId, executor::Request const& Request)
        : Base(requestId, Request)
    {
    }

    void movePromptEmbeddingTableToGpu(runtime::BufferManager const& manager)
    {
        if (!mPromptEmbeddingTable.has_value()
            || mPromptEmbeddingTable.value()->getMemoryType() == runtime::MemoryType::kGPU)
        {
            return;
        }
        else
        {
            TensorPtr gpuPromptEmbeddingTable
                = manager.copyFrom(*mPromptEmbeddingTable.value(), runtime::MemoryType::kGPU);
            mPromptEmbeddingTable = gpuPromptEmbeddingTable;
        }
    }

    void moveLoraWeightsToGpu(runtime::BufferManager const& manager)
    {
        if (!mLoraWeights.has_value() || mLoraWeights.value()->getMemoryType() == runtime::MemoryType::kGPU)
        {
            return;
        }
        TensorPtr gpuLoraWeights = manager.copyFrom(*mLoraWeights.value(), runtime::MemoryType::kGPU);
        mLoraWeights = gpuLoraWeights;
    }
};

}
