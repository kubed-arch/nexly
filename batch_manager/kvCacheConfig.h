
#pragma once

#include "../executor/executor.h"
#include "../runtime/common.h"

#include <optional>

namespace nexly::batch_manager::kv_cache_manager
{

class KvCacheConfig
{
public:
    using SizeType = nexly::runtime::SizeType;

    explicit KvCacheConfig(std::optional<SizeType> maxTokens = std::nullopt,
        std::optional<SizeType> maxAttentionWindow = std::nullopt,
        std::optional<SizeType> sinkTokenLength = std::nullopt,
        std::optional<float> freeGpuMemoryFraction = std::nullopt, bool enableBlockReuse = false, bool useUvm = false)
        : maxTokens{maxTokens}
        , maxAttentionWindow{maxAttentionWindow}
        , sinkTokenLength{sinkTokenLength}
        , freeGpuMemoryFraction{freeGpuMemoryFraction}
        , enableBlockReuse(enableBlockReuse)
        , useUvm(useUvm)
    {
    }

    explicit KvCacheConfig(executor::KvCacheConfig const& kvCacheConfig)
        : KvCacheConfig(kvCacheConfig.getMaxTokens(), kvCacheConfig.getMaxAttentionWindow(),
            kvCacheConfig.getSinkTokenLength(), kvCacheConfig.getFreeGpuMemoryFraction(),
            kvCacheConfig.getEnableBlockReuse(), kvCacheConfig.getUseUvm())
    {
    }

    bool operator==(KvCacheConfig const& other) const
    {
        return maxTokens == other.maxTokens && maxAttentionWindow == other.maxAttentionWindow
            && sinkTokenLength == other.sinkTokenLength && freeGpuMemoryFraction == other.freeGpuMemoryFraction
            && enableBlockReuse == other.enableBlockReuse && useUvm == other.useUvm;
    }

    std::optional<SizeType> maxTokens;
    std::optional<SizeType> maxAttentionWindow;
    std::optional<SizeType> sinkTokenLength;
    std::optional<float> freeGpuMemoryFraction;
    bool enableBlockReuse;
    static constexpr auto kDefaultGpuMemFraction = 0.9f;
    bool useUvm;
};
}
