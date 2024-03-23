
#pragma once

#include "kvCacheConfig.h"
#include "../executor/executor.h"
#include "../runtime/common.h"
#include "../runtime/decodingMode.h"

#include <optional>
#include <vector>

namespace nexly::batch_manager
{

class TrtGptModelOptionalParams
{
    using KvCacheConfig = kv_cache_manager::KvCacheConfig;

public:
    using SizeType = nexly::runtime::SizeType;

    explicit TrtGptModelOptionalParams(KvCacheConfig const& kvCacheConfig = KvCacheConfig{},
        bool enableTrtOverlap = false, std::optional<std::vector<SizeType>> const& deviceIds = std::nullopt,
        bool normalizeLogProbs = true, bool enableChunkedContext = false,
        std::optional<runtime::DecodingMode> const& decodingMode = std::nullopt)
        : kvCacheConfig{kvCacheConfig}
        , enableTrtOverlap{enableTrtOverlap}
        , deviceIds(deviceIds)
        , normalizeLogProbs{normalizeLogProbs}
        , enableChunkedContext{enableChunkedContext}
        , decodingMode{decodingMode}
    {
    }

    explicit TrtGptModelOptionalParams(executor::ExecutorConfig const& executorConfig)
        : TrtGptModelOptionalParams(KvCacheConfig(executorConfig.getKvCacheConfig()),
            executorConfig.getEnableTrtOverlap(),
            executorConfig.getParallelConfig().value_or(executor::ParallelConfig()).getDeviceIds(),
            executorConfig.getNormalizeLogProbs(), executorConfig.getEnableChunkedContext())
    {
    }

    bool operator==(TrtGptModelOptionalParams const& other) const
    {
        return kvCacheConfig == other.kvCacheConfig && enableTrtOverlap == other.enableTrtOverlap
            && deviceIds == other.deviceIds && normalizeLogProbs == other.normalizeLogProbs
            && enableChunkedContext == other.enableChunkedContext && decodingMode == other.decodingMode;
    }

    KvCacheConfig kvCacheConfig;

    bool enableTrtOverlap;
    std::optional<std::vector<SizeType>> deviceIds;
    bool normalizeLogProbs;
    bool enableChunkedContext;
    std::optional<runtime::DecodingMode> decodingMode;
};

}
