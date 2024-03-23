#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace nexly::executor
{

    /// <summary>
    /// Represents a request for execution.
    /// </summary>
    class Request;

    /// <summary>
    /// Represents a tensor.
    /// </summary>
    class Tensor;

    /// <summary>
    /// A shared pointer to a Tensor object.
    /// </summary>
    using TensorPtr = std::shared_ptr<Tensor>;

    /// <summary>
    /// Represents the size type, typically a 32-bit integer.
    /// </summary>
    using SizeType = std::int32_t;

    /// <summary>
    /// Represents the floating-point type, typically a 32-bit float.
    /// </summary>
    using FloatType = float;

    /// <summary>
    /// Represents the type for token IDs, typically a 32-bit integer.
    /// </summary>
    using TokenIdType = std::int32_t;

    /// <summary>
    /// Represents a vector of token IDs.
    /// </summary>
    using VecTokens = std::vector<TokenIdType>;

    /// <summary>
    /// Represents a vector of vectors of token IDs.
    /// </summary>
    using BeamTokens = std::vector<VecTokens>;

    /// <summary>
    /// Represents the ID type, typically a 64-bit unsigned integer.
    /// </summary>
    using IdType = std::uint64_t;

    /// <summary>
    /// Represents the type for random seed values, typically a 64-bit unsigned integer.
    /// </summary>
    using RandomSeedType = std::uint64_t;

    /// <summary>
    /// Represents a vector of floating-point log probabilities.
    /// </summary>
    using VecLogProbs = std::vector<FloatType>;

    /// <summary>
    /// Enumerates the different data types.
    /// </summary>
    enum class DataType
    {
        kBOOL,                  ///< Boolean data type.
        kUINT8,                 ///< Unsigned 8-bit integer data type.
        kINT8,                  ///< Signed 8-bit integer data type.
        kINT32,                 ///< Signed 32-bit integer data type.
        kINT64,                 ///< Signed 64-bit integer data type.
        kBF16,                  ///< Brain Floating Point (BF16) data type.
        kFP8,                   ///< 8-bit floating point data type.
        kFP16,                  ///< 16-bit floating point data type.
        kFP32,                  ///< 32-bit floating point data type.
        kUNKNOWN                ///< Unknown data type.
    };

    /// <summary>
    /// Template specialization for type traits of various data types.
    /// </summary>
    template <typename T, bool = false>
    struct TypeTraits
    {
    };

    /// <summary>
    /// Type traits specialization for floating-point type.
    /// </summary>
    template <>
    struct TypeTraits<float>
    {
        static constexpr auto value = DataType::kFP32;
    };

    /// <summary>
    /// Type traits specialization for half-precision floating-point type.
    /// </summary>
    template <>
    struct TypeTraits<half>
    {
        static constexpr auto value = DataType::kFP16;
    };

    /// <summary>
    /// Type traits specialization for signed 8-bit integer type.
    /// </summary>
    template <>
    struct TypeTraits<std::int8_t>
    {
        static constexpr auto value = DataType::kINT8;
    };

    /// <summary>
    /// Type traits specialization for signed 32-bit integer type.
    /// </summary>
    template <>
    struct TypeTraits<std::int32_t>
    {
        static constexpr auto value = DataType::kINT32;
    };

    /// <summary>
    /// Type traits specialization for signed 64-bit integer type.
    /// </summary>
    template <>
    struct TypeTraits<std::int64_t>
    {
        static constexpr auto value = DataType::kINT64;
    };

    /// <summary>
    /// Type traits specialization for boolean type.
    /// </summary>
    template <>
    struct TypeTraits<bool>
    {
        static constexpr auto value = DataType::kBOOL;
    };

    /// <summary>
    /// Type traits specialization for unsigned 8-bit integer type.
    /// </summary>
    template <>
    struct TypeTraits<std::uint8_t>
    {
        static constexpr auto value = DataType::kUINT8;
    };

#ifdef ENABLE_BF16
    /// <summary>
    /// Type traits specialization for Brain Floating Point (BF16) type.
    /// </summary>
    template <>
    struct TypeTraits<__nv_bfloat16>
    {
        static constexpr auto value = DataType::kBF16;
    };
#endif

#ifdef ENABLE_FP8
    /// <summary>
    /// Type traits specialization for 8-bit floating point type.
    /// </summary>
    template <>
    struct TypeTraits<__nv_fp8_e4m3>
    {
        static constexpr auto value = DataType::kFP8;
    };
#endif

    /// <summary>
    /// Type traits specialization for pointer type.
    /// </summary>
    template <typename T>
    struct TypeTraits<T*>
    {
        static constexpr auto value = DataType::kINT64;
    };

    /// <summary>
    /// Enumerates the different memory types.
    /// </summary>
    enum class MemoryType
    {
        kCPU,                   ///< CPU memory type.
        kCPU_PINNED,            ///< Pinned CPU memory type.
        kGPU,                   ///< GPU memory type.
        kUVM,                   ///< Unified Virtual Memory (UVM) memory type.
        kUNKNOWN                ///< Unknown memory type.
    };

    /// <summary>
    /// Enumerates the different model types.
    /// </summary>
    enum class ModelType
    {
        kDECODER_ONLY = 0,          ///< Decoder-only model type.
    };

    /// <summary>
    /// Enumerates the different batching types.
    /// </summary>
    enum class BatchingType
    {
        kSTATIC = 0,                ///< Static batching type.
        kINFLIGHT = 1,              ///< Inflight batching type.
        kINFLIGHT_UNFUSED = 2,      ///< Unfused inflight batching type.
    };

    /// <summary>
    /// Enumerates the different scheduler policies.
    /// </summary>
    enum class SchedulerPolicy
    {
        kMAX_UTILIZATION = 0,           ///< Maximum utilization scheduler policy.
        kGUARANTEED_NO_EVICT = 1,       ///< Guaranteed no eviction scheduler policy.
    };

    /// <summary>
    /// Enumerates the different communication types.
    /// </summary>
    enum class CommunicationType
    {
        kMPI = 0                        ///< Message Passing Interface (MPI) communication type.
    };

    /// <summary>
    /// Enumerates the different communication modes.
    /// </summary>
    enum class CommunicationMode
    {
        kLEADER,                        ///< Leader communication mode.
    };
}
