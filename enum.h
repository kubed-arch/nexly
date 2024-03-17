#pragma once

#include <stdexcept>
#include <type_traits>

namespace DataSetEnums
{
    /// <summary>
    /// Enumeration representing various attributes of a dataset.
    /// </summary>
    enum Attributes
    {
        Sparse = 1,
        Boolean = 2,
        Compressed = 4,
        Recurrent = 8,
        Mutable = 16,
        SparseIgnoreZero = 32,
        Indexed = 64,
        Weighted = 128
    };

    /// <summary>
    /// Enumeration representing the kind of dataset.
    /// </summary>
    enum Kind
    {
        Numeric = 0,
        Image = 1,
        Audio = 2,
        Text = 3
    };

    /// <summary>
    /// Enumeration representing sharding options for the dataset.
    /// </summary>
    enum Sharding
    {
        None = 0,
        Model = 1,
        Data = 2,
    };

    /// <summary>
    /// Enumeration representing the data types supported by the dataset.
    /// </summary>
    enum DataType
    {
        UInt = 0,
        Int = 1,
        LLInt = 2,
        ULLInt = 3,
        Float = 4,
        Double = 5,
        RGB8 = 6,
        RGB16 = 7,
        UChar = 8,
        Char = 9
    };

    /// <summary>
    /// Enumeration representing regularization types.
    /// </summary>
    enum class RegularizationType {
        L1,
        L2
    };

    /// <summary>
    /// Enumeration representing dataset types.
    /// </summary>
    enum class DatasetType {
        Indicator,
        Analog
    };

    /// <summary>
    /// Enumeration representing different types of attention masks.
    /// </summary>
    enum class AttentionMaskType {
        PADDING = 0,
        CAUSAL = 1,
        BIDIRECTIONAL = 2,
        BIDIRECTIONALGLM = 3
    };

    /// <summary>
    /// Enumeration representing different types of position embeddings.
    /// </summary>
    enum class PositionEmbeddingType : int8_t {
        kLEARNED_ABSOLUTE = 0,
        kROPE_GPTJ = 1,
        kROPE_GPT_NEOX = 2,
        kALIBI = 3,
        kALIBI_WITH_SCALE = 4,
        kRELATIVE = 5
    };

    /// <summary>
    /// Enumeration representing different types of rotary scaling.
    /// </summary>
    enum class RotaryScalingType : int8_t {
        kNONE = 0,
        kLINEAR = 1,
        kDYNAMIC = 2,
    };

    /// <summary>
    /// Enumeration representing different types of key-value cache data.
    /// </summary>
    enum class KvCacheDataType {
        BASE = 0,
        INT8,
        FP8
    };

    /// <summary>
    /// Template concept representing valid data types.
    /// </summary>
    template <typename T>
    concept ValidDataType = std::is_same_v<T, uint32_t> ||
        std::is_same_v<T, int32_t> ||
        std::is_same_v<T, int64_t> ||
        std::is_same_v<T, uint64_t> ||
        std::is_same_v<T, float> ||
        std::is_same_v<T, double> ||
        std::is_same_v<T, char> ||
        std::is_same_v<T, unsigned char>;

    /// <summary>
    /// Function template to get the DataType based on template parameter T.
    /// </summary>
    /// <typeparam name="T">The type of data.</typeparam>
    /// <returns>The DataType corresponding to the type T.</returns>
    template <ValidDataType T>
    inline DataType getDataType()
    {
        if constexpr (std::is_same_v<T, uint32_t>)
        {
            return DataType::UInt;
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            return DataType::Int;
        }
        else if constexpr (std::is_same_v<T, int64_t>)
        {
            return DataType::LLInt;
        }
        else if constexpr (std::is_same_v<T, uint64_t>)
        {
            return DataType::ULLInt;
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return DataType::Float;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return DataType::Double;
        }
        else if constexpr (std::is_same_v<T, char>)
        {
            return DataType::Char;
        }
        else if constexpr (std::is_same_v<T, unsigned char>)
        {
            return DataType::UChar;
        }
        else
        {
            throw std::runtime_error("Default data type not defined");
        }
    }
}
