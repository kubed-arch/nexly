#pragma once

namespace nexly
{
    namespace runtime
    {

        class DecodingMode
        {
        public:
            static auto constexpr None()
            {
                return DecodingMode{ kNone };
            }

            static auto constexpr TopK()
            {
                return DecodingMode{ kTopK };
            }

            static auto constexpr TopP()
            {
                return DecodingMode{ kTopP };
            }

            static auto constexpr TopKTopP()
            {
                return DecodingMode{ kTopKTopP };
            }

            static auto constexpr BeamSearch()
            {
                return DecodingMode{ kBeamSearch };
            }

            bool constexpr isNone()
            {
                return mState == 0;
            }

            bool constexpr isTopK()
            {
                return anyBitSet(kTopK);
            }

            bool constexpr isTopP()
            {
                return anyBitSet(kTopP);
            }

            bool constexpr isTopKorTopP()
            {
                return anyBitSet(kTopKTopP);
            }

            bool constexpr isTopKandTopP()
            {
                return allBitSet(kTopKTopP);
            }

            bool constexpr isBeamSearch()
            {
                return anyBitSet(kBeamSearch);
            }

            using UnderlyingType = uint8_t;

            bool operator==(DecodingMode const& other) const
            {
                return mState == other.mState;
            }

        private:
            constexpr DecodingMode(UnderlyingType state)
                : mState(state)
            {
            }

            static UnderlyingType constexpr kNone{ 0 };
            static UnderlyingType constexpr kTopK{ 1u << 0 };
            static UnderlyingType constexpr kTopP{ 1u << 1 };
            static UnderlyingType constexpr kBeamSearch{ 1u << 2 };
            static UnderlyingType constexpr kTopKTopP{ kTopK | kTopP };

            bool constexpr anyBitSet(UnderlyingType bits) const
            {
                return (mState & bits) != 0;
            }

            bool constexpr allBitSet(UnderlyingType bits) const
            {
                return (mState & bits) == bits;
            }

            UnderlyingType mState{};
        };

        static_assert(DecodingMode::None().isNone());
        static_assert(!DecodingMode::None().isTopK());
        static_assert(!DecodingMode::None().isTopP());
        static_assert(!DecodingMode::None().isBeamSearch());

        static_assert(DecodingMode::TopK().isTopK());
        static_assert(DecodingMode::TopK().isTopKorTopP());
        static_assert(!DecodingMode::TopK().isTopKandTopP());
        static_assert(!DecodingMode::TopK().isTopP());
        static_assert(!DecodingMode::TopK().isBeamSearch());

        static_assert(DecodingMode::TopP().isTopP());
        static_assert(DecodingMode::TopP().isTopKorTopP());
        static_assert(!DecodingMode::TopP().isTopKandTopP());
        static_assert(!DecodingMode::TopP().isTopK());
        static_assert(!DecodingMode::TopP().isBeamSearch());

        static_assert(DecodingMode::TopKTopP().isTopK());
        static_assert(DecodingMode::TopKTopP().isTopP());
        static_assert(DecodingMode::TopKTopP().isTopKorTopP());
        static_assert(DecodingMode::TopKTopP().isTopKandTopP());
        static_assert(!DecodingMode::TopKTopP().isBeamSearch());

        static_assert(DecodingMode::BeamSearch().isBeamSearch());
        static_assert(!DecodingMode::BeamSearch().isTopKorTopP());

    }
}