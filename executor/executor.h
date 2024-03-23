#pragma once

#include "tensor.h"
#include "types.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace nexly::executor
{

    class Model;
    class Serialization;

    class SamplingConfig
    {
    public:
        SamplingConfig(SizeType beamWidth = 1, std::optional<SizeType> topK = std::nullopt,
            std::optional<FloatType> topP = std::nullopt, std::optional<FloatType> topPMin = std::nullopt,
            std::optional<SizeType> topPResetIds = std::nullopt, std::optional<FloatType> topPDecay = std::nullopt,
            std::optional<RandomSeedType> randomSeed = std::nullopt, std::optional<FloatType> temperature = std::nullopt,
            std::optional<SizeType> minLength = std::nullopt,
            std::optional<FloatType> beamSearchDiversityRate = std::nullopt,
            std::optional<FloatType> repetitionPenalty = std::nullopt,
            std::optional<FloatType> presencePenalty = std::nullopt,
            std::optional<FloatType> frequencyPenalty = std::nullopt, std::optional<FloatType> lengthPenalty = std::nullopt,
            std::optional<SizeType> earlyStopping = std::nullopt);

        ~SamplingConfig();

        bool operator==(SamplingConfig const& other) const;

        [[nodiscard]] SizeType getBeamWidth() const;
        [[nodiscard]] std::optional<SizeType> getTopK() const;
        [[nodiscard]] std::optional<FloatType> getTopP() const;
        [[nodiscard]] std::optional<FloatType> getTopPMin() const;
        [[nodiscard]] std::optional<SizeType> getTopPResetIds() const;
        [[nodiscard]] std::optional<FloatType> getTopPDecay() const;
        [[nodiscard]] std::optional<RandomSeedType> getRandomSeed() const;
        [[nodiscard]] std::optional<FloatType> getTemperature() const;
        [[nodiscard]] std::optional<SizeType> getMinLength() const;
        [[nodiscard]] std::optional<FloatType> getBeamSearchDiversityRate() const;
        [[nodiscard]] std::optional<FloatType> getRepetitionPenalty() const;
        [[nodiscard]] std::optional<FloatType> getPresencePenalty() const;
        [[nodiscard]] std::optional<FloatType> getFrequencyPenalty() const;
        [[nodiscard]] std::optional<FloatType> getLengthPenalty() const;
        [[nodiscard]] std::optional<SizeType> getEarlyStopping() const;

    private:
        friend class Serialization;
        SizeType mBeamWidth;
        std::optional<SizeType> mTopK;
        std::optional<FloatType> mTopP;
        std::optional<FloatType> mTopPMin;
        std::optional<SizeType> mTopPResetIds;
        std::optional<FloatType> mTopPDecay;
        std::optional<RandomSeedType> mRandomSeed;
        std::optional<FloatType> mTemperature;
        std::optional<SizeType> mMinLength;
        std::optional<FloatType> mBeamSearchDiversityRate;
        std::optional<FloatType> mRepetitionPenalty;
        std::optional<FloatType> mPresencePenalty;
        std::optional<FloatType> mFrequencyPenalty;
        std::optional<FloatType> mLengthPenalty;
        std::optional<SizeType> mEarlyStopping;
    };

    class OutputConfig
    {
    public:
        OutputConfig(bool returnLogProbs = false, bool returnContextLogits = false, bool returnGenerationLogits = false,
            bool excludeInputFromOutput = false);

        bool returnLogProbs;
        bool returnContextLogits;
        bool returnGenerationLogits;
        bool excludeInputFromOutput;
    };

    class SpeculativeDecodingConfig
    {
    public:
        explicit SpeculativeDecodingConfig(VecTokens tokens, std::optional<Tensor> logits = std::nullopt,
            std::optional<FloatType> acceptanceThreshold = std::nullopt);

        ~SpeculativeDecodingConfig();

        [[nodiscard]] VecTokens getTokens() const;
        [[nodiscard]] std::optional<Tensor> getLogits() const;
        [[nodiscard]] std::optional<FloatType> getAcceptanceThreshold() const;

    private:
        friend class Serialization;
        VecTokens mTokens;
        std::optional<Tensor> mLogits;
        std::optional<FloatType> mAcceptanceThreshold;
    };

    class PromptTuningConfig
    {
    public:
        PromptTuningConfig(Tensor embeddingTable);
        ~PromptTuningConfig();

        [[nodiscard]] Tensor getEmbeddingTable() const;

    private:
        friend class Serialization;
        Tensor mEmbeddingTable;
    };

    class LoraConfig
    {
    public:
        LoraConfig(Tensor weights, Tensor config);
        ~LoraConfig();

        [[nodiscard]] Tensor getWeights() const;
        [[nodiscard]] Tensor getConfig() const;

    private:
        friend class Serialization;

        Tensor mWeights;
        Tensor mConfig;
    };

    class Request
    {
    public:
        Request(VecTokens inputTokenIds, SizeType maxNewTokens, bool streaming = false,
            SamplingConfig samplingConfig = SamplingConfig(), OutputConfig outputConfig = OutputConfig(),
            std::optional<SizeType> endId = std::nullopt, std::optional<SizeType> padId = std::nullopt,
            std::optional<std::list<VecTokens>> badWords = std::nullopt,
            std::optional<std::list<VecTokens>> stopWords = std::nullopt,
            std::optional<Tensor> embeddingBias = std::nullopt,
            std::optional<SpeculativeDecodingConfig> speculativeDecodingConfig = std::nullopt,
            std::optional<PromptTuningConfig> pTuningConfig = std::nullopt,
            std::optional<LoraConfig> loraConfig = std::nullopt);

        Request(Request const& other);
        Request(Request&& other) noexcept;
        Request& operator=(Request const& other);
        Request& operator=(Request&& other) noexcept;
        ~Request();

        [[nodiscard]] VecTokens getInputTokenIds() const;
        [[nodiscard]] SizeType getMaxNewTokens() const;
        [[nodiscard]] bool getStreaming() const;
        [[nodiscard]] SamplingConfig getSamplingConfig() const;
        [[nodiscard]] OutputConfig getOutputConfig() const;
        [[nodiscard]] std::optional<SizeType> getEndId() const;
        [[nodiscard]] std::optional<SizeType> getPadId() const;
        [[nodiscard]] std::optional<std::list<VecTokens>> getBadWords() const;
        [[nodiscard]] std::optional<std::list<VecTokens>> getStopWords() const;
        [[nodiscard]] std::optional<Tensor> getEmbeddingBias() const;
        [[nodiscard]] std::optional<SpeculativeDecodingConfig> getSpeculativeDecodingConfig() const;
        [[nodiscard]] std::optional<PromptTuningConfig> getPromptTuningConfig() const;
        [[nodiscard]] std::optional<LoraConfig> getLoraConfig() const;

        void setStreaming(bool streaming);
        void setSamplingConfig(SamplingConfig config);
        void setOutputConfig(OutputConfig outputConfig);
        void setEndId(SizeType endId);
        void setPadId(SizeType padId);
        void setBadWords(std::list<VecTokens> badWords);
        void setStopWords(std::list<VecTokens> stopWords);
        void setEmbeddingBias(Tensor);
        void setSpeculativeDecodingConfig(SpeculativeDecodingConfig specDecodingConfig);
        void setPromptTuningConfig(PromptTuningConfig pTuningConfig);
        void setLoraConfig(LoraConfig loraConfig);

    private:
        friend class Serialization;
        class Impl;
        std::unique_ptr<Impl> mImpl;
    };

    struct Result
    {
        bool isFinal;

        BeamTokens outputTokenIds;

        std::optional<VecLogProbs> cumLogProbs;
        std::optional<std::vector<VecLogProbs>> logProbs;
        std::optional<Tensor> contextLogits;
        std::optional<Tensor> generationLogits;
    };

    class Response
    {
    public:
        Response(IdType requestId, std::string errorMsg);
        Response(IdType requestId, Result Result);

        ~Response();
        Response(Response const& other);
        Response(Response&& other) noexcept;
        Response& operator=(Response const& other);
        Response& operator=(Response&& other) noexcept;

        IdType getRequestId() const;

        bool hasError() const;

        std::string getErrorMsg() const;

        Result getResult() const;

    private:
        class Impl;
        std::unique_ptr<Impl> mImpl;
    };

    class SchedulerConfig
    {
    public:
        explicit SchedulerConfig(SchedulerPolicy policy = SchedulerPolicy::kGUARANTEED_NO_EVICT);
        ~SchedulerConfig();

        [[nodiscard]] SchedulerPolicy getPolicy() const;

    private:
        SchedulerPolicy mPolicy;
    };

    class KvCacheConfig
    {
    public:
        KvCacheConfig(bool enableBlockReuse = false, std::optional<SizeType> maxTokens = std::nullopt,
            std::optional<SizeType> maxAttentionWindow = std::nullopt,
            std::optional<SizeType> sinkTokenLength = std::nullopt,
            std::optional<FloatType> freeGpuMemoryFraction = std::nullopt, bool useUvm = false);

        [[nodiscard]] bool getEnableBlockReuse() const;
        [[nodiscard]] std::optional<SizeType> getMaxTokens() const;
        [[nodiscard]] std::optional<SizeType> getMaxAttentionWindow() const;
        [[nodiscard]] std::optional<SizeType> getSinkTokenLength() const;
        [[nodiscard]] std::optional<FloatType> getFreeGpuMemoryFraction() const;
        [[nodiscard]] bool getUseUvm() const;

    private:
        bool mEnableBlockReuse;
        std::optional<SizeType> mMaxTokens;
        std::optional<SizeType> mMaxAttentionWindow;
        std::optional<SizeType> mSinkTokenLength;
        std::optional<FloatType> mFreeGpuMemoryFraction;
        bool mUseUvm;
    };

    SizeType const kDefaultIterStatsMaxIterations = 1000;

    class ParallelConfig
    {
    public:
        ParallelConfig(CommunicationType commType = CommunicationType::kMPI,
            CommunicationMode commMode = CommunicationMode::kLEADER,
            std::optional<std::vector<SizeType>> deviceIds = std::nullopt,
            std::optional<std::vector<SizeType>> participantIds = std::nullopt);
        ~ParallelConfig();

        [[nodiscard]] CommunicationType getCommunicationType() const;
        [[nodiscard]] CommunicationMode getCommunicationMode() const;
        [[nodiscard]] std::optional<std::vector<SizeType>> getDeviceIds() const;
        [[nodiscard]] std::optional<std::vector<SizeType>> getParticipantIds() const;

        void setCommunicationType(CommunicationType type);
        void setCommunicationMode(CommunicationMode mode);
        void setDeviceIds(std::vector<SizeType> deviceIds);
        void setParticipantIds(std::vector<SizeType> participantIds);

    private:
        CommunicationType mCommType;
        CommunicationMode mCommMode;
        std::optional<std::vector<SizeType>> mDeviceIds;
        std::optional<std::vector<SizeType>> mParticipantIds;
    };

    class ExecutorConfig
    {
    public:
        ExecutorConfig(SizeType maxBeamWidth = 1, SchedulerConfig schedulerConfig = SchedulerConfig(),
            KvCacheConfig kvCacheConfig = KvCacheConfig(), bool enableChunkedContext = false, bool normalizeLogProbs = true,
            bool enableTrtOverlap = false, SizeType iterStatsMaxIterations = kDefaultIterStatsMaxIterations,
            BatchingType batchingType = BatchingType::kINFLIGHT,
            std::optional<ParallelConfig> parallelConfig = std::nullopt);

        [[nodiscard]] SizeType getMaxBeamWidth() const;
        [[nodiscard]] SchedulerConfig getSchedulerConfig() const;
        [[nodiscard]] KvCacheConfig getKvCacheConfig() const;
        [[nodiscard]] bool getEnableChunkedContext() const;
        [[nodiscard]] bool getNormalizeLogProbs() const;
        [[nodiscard]] bool getEnableTrtOverlap() const;
        [[nodiscard]] SizeType getIterStatsMaxIterations() const;
        [[nodiscard]] BatchingType getBatchingType() const;
        [[nodiscard]] std::optional<ParallelConfig> getParallelConfig() const;

        void setMaxBeamWidth(SizeType maxBeamWidth);
        void setSchedulerConfig(SchedulerConfig schedulerConfig);
        void setKvCacheConfig(KvCacheConfig kvCacheConfig);
        void setEnableChunkedContext(bool enableChunkedContext);
        void setNormalizeLogProbs(bool normalizeLogProbs);
        void setEnableTrtOverlap(bool enableTrtOverlap);
        void setIterStatsMaxIterations(SizeType iterStatsMaxIterations);
        void setBatchingType(BatchingType batchingType);
        void setParallelConfig(ParallelConfig parallelConfig);

    private:
        SizeType mMaxBeamWidth;
        SchedulerConfig mSchedulerConfig;
        KvCacheConfig mKvCacheConfig;
        bool mEnableChunkedContext;
        bool mNormalizeLogProbs;
        bool mEnableTrtOverlap;
        SizeType mIterStatsMaxIterations;
        BatchingType mBatchingType;
        std::optional<ParallelConfig> mParallelConfig;
    };

    class Executor
    {
        using RequestPtr = std::shared_ptr<Request>;

    public:
        Executor(std::filesystem::path const& modelPath, ModelType modelType, ExecutorConfig executorConfig);

        Executor(std::vector<uint8_t> const& engineBuffer, std::string const& jsonConfigStr, ModelType modelType,
            ExecutorConfig executorConfig);

        Executor(std::shared_ptr<Model> model, ExecutorConfig executorConfig);

        ~Executor();

        IdType enqueueRequest(Request request);

        std::vector<IdType> enqueueRequests(std::vector<Request> requests);

        std::vector<Response> awaitResponses(
            std::optional<IdType> id = std::nullopt, std::optional<std::chrono::milliseconds> timeout = std::nullopt);

        SizeType getNumResponsesReady(std::optional<IdType> id = std::nullopt);

        void cancelRequest(IdType id);

        void shutdown();

        std::deque<std::string> getLatestIterationStats();

    private:
        class Impl;
        std::unique_ptr<Impl> mImpl;
    };

}
