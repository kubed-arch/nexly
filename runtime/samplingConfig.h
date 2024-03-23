#pragma once

#include "../executor/executor.h"
#include "common.h"

#include <functional>
#include <optional>
#include <vector>

namespace nexly::runtime
{

    class SamplingConfig
    {
    private:
        using FloatType = float;

        template <typename T>
        using OptVec = std::optional<std::vector<T>>;

    private:
        template <typename T>
        static OptVec<T> fuseValues(
            std::vector<SamplingConfig> const& configs, std::function<OptVec<T>(SizeType ci)> accessor)
        {
            std::vector<T> values;
            auto const hasValues = accessor(0).has_value();
            for (size_t ci = 0; ci < configs.size(); ++ci)
            {
                auto const& configValue = accessor(ci);
                TLLM_CHECK(hasValues == configValue.has_value());
                if (hasValues)
                {
                    TLLM_CHECK(configValue.value().size() == 1);
                    values.push_back(configValue.value().front());
                }
            }

            if (!hasValues)
            {
                return std::nullopt;
            }
            return std::make_optional<std::vector<T>>(values);
        }

        template <typename T>
        using Vec = std::vector<T>;

    public:
        explicit SamplingConfig(SizeType beamWidth = 1)
            : beamWidth{ beamWidth }
        {
        }

        explicit SamplingConfig(std::vector<SamplingConfig> const& configs)
        {
            CHECK(configs.size() > 0);
            beamWidth = configs.front().beamWidth;
            normalizeLogProbs = configs.front().normalizeLogProbs;
            temperature = fuseValues<FloatType>(configs, [&configs](SizeType ci) { return configs[ci].temperature; });
            minLength = fuseValues<SizeType>(configs, [&configs](SizeType ci) { return configs[ci].minLength; });
            repetitionPenalty
                = fuseValues<FloatType>(configs, [&configs](SizeType ci) { return configs[ci].repetitionPenalty; });
            presencePenalty
                = fuseValues<FloatType>(configs, [&configs](SizeType ci) { return configs[ci].presencePenalty; });
            topK = fuseValues<SizeType>(configs, [&configs](SizeType ci) { return configs[ci].topK; });
            topP = fuseValues<FloatType>(configs, [&configs](SizeType ci) { return configs[ci].topP; });
            randomSeed = fuseValues<uint64_t>(configs, [&configs](SizeType ci) { return configs[ci].randomSeed; });
            topPDecay = fuseValues<FloatType>(configs, [&configs](SizeType ci) { return configs[ci].topPDecay; });
            topPMin = fuseValues<FloatType>(configs, [&configs](SizeType ci) { return configs[ci].topPMin; });
            topPResetIds = fuseValues<SizeType>(configs, [&configs](SizeType ci) { return configs[ci].topPResetIds; });
            beamSearchDiversityRate
                = fuseValues<FloatType>(configs, [&configs](SizeType ci) { return configs[ci].beamSearchDiversityRate; });
            lengthPenalty = fuseValues<FloatType>(configs, [&configs](SizeType ci) { return configs[ci].lengthPenalty; });
            earlyStopping = fuseValues<SizeType>(configs, [&configs](SizeType ci) { return configs[ci].earlyStopping; });
            draftAcceptanceThreshold
                = fuseValues<FloatType>(configs, [&configs](SizeType ci) { return configs[ci].draftAcceptanceThreshold; });
        }

        explicit SamplingConfig(executor::SamplingConfig const& samplingConfig,
            std::optional<executor::SpeculativeDecodingConfig> const& specDecodingConfig)
            : beamWidth{ samplingConfig.getBeamWidth() }
        {

            if (specDecodingConfig && specDecodingConfig.value().getAcceptanceThreshold())
            {
                draftAcceptanceThreshold = Vec<FloatType>{ specDecodingConfig.value().getAcceptanceThreshold().value() };
            }

#define SET_FROM_OPTIONAL(varName, VarName, VarType)                                                                   \
                                                                                                                       \
    if (samplingConfig.get##VarName())                                                                                 \
    {                                                                                                                  \
        varName = Vec<VarType>{samplingConfig.get##VarName().value()};                                                 \
    }

            SET_FROM_OPTIONAL(topK, TopK, SizeType)
                SET_FROM_OPTIONAL(topP, TopP, FloatType)
                SET_FROM_OPTIONAL(topPMin, TopPMin, FloatType)
                SET_FROM_OPTIONAL(topPResetIds, TopPResetIds, SizeType)
                SET_FROM_OPTIONAL(topPDecay, TopPDecay, FloatType)
                SET_FROM_OPTIONAL(randomSeed, RandomSeed, uint64_t)
                SET_FROM_OPTIONAL(temperature, Temperature, FloatType)
                SET_FROM_OPTIONAL(minLength, MinLength, SizeType)
                SET_FROM_OPTIONAL(beamSearchDiversityRate, BeamSearchDiversityRate, FloatType)
                SET_FROM_OPTIONAL(repetitionPenalty, RepetitionPenalty, FloatType)
                SET_FROM_OPTIONAL(presencePenalty, PresencePenalty, FloatType)
                SET_FROM_OPTIONAL(frequencyPenalty, FrequencyPenalty, FloatType)
                SET_FROM_OPTIONAL(lengthPenalty, LengthPenalty, FloatType)
                SET_FROM_OPTIONAL(earlyStopping, EarlyStopping, SizeType)
#undef SET_FROM_OPTIONAL
        }

    public:
        SizeType beamWidth;

        OptVec<FloatType> temperature;
        OptVec<SizeType> minLength;
        OptVec<FloatType> repetitionPenalty;
        OptVec<FloatType> presencePenalty;
        OptVec<FloatType> frequencyPenalty;

        OptVec<SizeType> topK;
        OptVec<FloatType> topP;
        OptVec<uint64_t> randomSeed;
        OptVec<FloatType> topPDecay;
        OptVec<FloatType> topPMin;
        OptVec<SizeType> topPResetIds;

        OptVec<FloatType> beamSearchDiversityRate;
        OptVec<FloatType> lengthPenalty;
        OptVec<SizeType> earlyStopping;

        OptVec<FloatType> draftAcceptanceThreshold;

        std::optional<bool> normalizeLogProbs;
    };

}
