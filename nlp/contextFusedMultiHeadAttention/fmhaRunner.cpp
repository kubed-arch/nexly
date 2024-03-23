
#include "fmhaRunner.h"
#include "fused_multihead_attention_v2.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <math.h>
#include <tuple>
#include <vector>

namespace nexly
{
namespace kernels
{

union __half2_uint32_t_union
{
    half2 fp162;
    uint32_t u32;
};

union __float_uint32_t_union
{
    float fp32;
    uint32_t u32;
};

static inline void set_alpha(uint32_t& alpha, float norm, Data_type dtype)
{
    if (dtype == DATA_TYPE_FP16)
    {
        __half2_uint32_t_union temp;
        temp.fp162 = __float2half2_rn(norm);
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_FP32)
    {
        __float_uint32_t_union temp;
        temp.fp32 = norm;
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = reinterpret_cast<uint32_t const&>(inorm);
    }
    else if (dtype == DATA_TYPE_BF16)
    {
        alpha = reinterpret_cast<uint32_t const&>(norm);
    }
    else
    {
        assert(false);
    }
}


class FusedMHARunnerV2::mhaImpl
{
public:
    mhaImpl(const Data_type data_type, int const numHeads, int const headSize, float const qScaling, int sm_)
        : mDataType(data_type)
        , mNumHeads(numHeads)
        , mHeadSize(headSize)
        , mQScaling(qScaling)
        , sm(sm_)
    {
        CHECK_WITH_INFO(
            (sm == kSM_70 || sm == kSM_80 || sm == kSM_86 || sm == kSM_89 || sm == kSM_90), "Unsupported architecture");
        CHECK_WITH_INFO((mDataType == DATA_TYPE_FP16 || mDataType == DATA_TYPE_BF16), "Unsupported data type");

        pagedKVXmmaKernel = getPagedKVXMMAKernelsV2(mDataType, sm);
        xmmaKernel = getXMMAKernelsV2(mDataType, sm);

        mParams.clear();
        mPagedKVParams.clear();

        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&mLaunchParams.multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);
        cudaDeviceGetAttribute(&mLaunchParams.device_l2_cache_size, cudaDevAttrL2CacheSize, device_id);
    }

    ~mhaImpl() {}

    template <typename Params>
    void setup_params(Params& params, int const b, int const s_q, int const s_kv, int const sliding_window_size,
        int const total_seqlen, bool const has_alibi, bool const scale_alibi, int const tp_size, int const tp_rank)
    {

        float const inv_sqrt_scale = (1.f / (sqrtf(mHeadSize) * mQScaling));
        float const scale_after_alibi = scale_alibi ? inv_sqrt_scale : 1.0f;
        float const scale_bmm1 = scale_alibi ? 1.0f : inv_sqrt_scale;
        float const scale_softmax = 1.f;
        float const scale_bmm2 = 1.f;

        Data_type scale_type = mLaunchParams.force_fp32_acc ? DATA_TYPE_FP32 : mDataType;
        if (mLaunchParams.useBase2ExpTrick)
        {
            constexpr float kLog2e = 1.4426950408889634074;
            set_alpha(params.scale_bmm1, scale_bmm1 * float(kLog2e), DATA_TYPE_FP32);
        }
        else
        {
            set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        }
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b = b;
        params.h = mNumHeads;
        params.s = s_q;
        params.d = mHeadSize;
        params.sliding_window_size = sliding_window_size;

        params.o_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);

        mTotalSeqLen = total_seqlen;

        if (has_alibi && sm > kSM_70)
        {
            params.has_alibi = true;
            params.alibi_params = AlibiParams(mNumHeads, s_kv, tp_size, tp_rank, scale_after_alibi);
        }
    }

    void setup(int const b, int const s, int const sliding_window_size, int const total_seqlen, bool const has_alibi,
        bool const scale_alibi, int const tp_size, int const tp_rank)
    {

        mLaunchParams.set_default_kernel_selection_params();

        CHECK_WITH_INFO(mHeadSize > 0, "Head size should be greater than 0.");
        mLaunchParams.padded_d = (mHeadSize & (mHeadSize - 1)) == 0 ? mHeadSize : pow(2, int(log2(mHeadSize)) + 1);

        bool const isSm70 = (sm == kSM_70);
        bool const isSm90 = (sm == kSM_90);
        bool const isSm8x = (sm == kSM_86 || sm == kSM_89);
        bool const isSm80 = (sm == kSM_80);
        if (isSm70)
        {
            mLaunchParams.flash_attention = true;
            mLaunchParams.force_unroll = true;
            mLaunchParams.useKernelWithoutAlibi = true;
        }
        else if (isSm90 && (mHeadSize == 32 || mHeadSize == 64) && s <= 256)
        {
            mLaunchParams.flash_attention = false;
            mLaunchParams.kernel_s = getSFromMaxSeqLen(s);
        }
        else
        {
            mLaunchParams.flash_attention = true;
            mLaunchParams.kernel_s = 0;
            mLaunchParams.force_unroll = true;
            if (mLaunchParams.flash_attention && s <= 64)
            {
                mLaunchParams.granular_tiling = false;
            }
            else if (isSm8x && mHeadSize < 256)
            {
                mLaunchParams.granular_tiling = false;
            }
            else if (isSm80 || isSm8x)
            {
                mLaunchParams.granular_tiling = true;
            }
        }

        if (isSm90 && mLaunchParams.flash_attention)
        {
            mLaunchParams.warp_specialization = true;
            mLaunchParams.use_tma = true;
        }

        if (mLaunchParams.warp_specialization && !has_alibi)
        {
            mLaunchParams.useKernelWithoutAlibi = true;
            mLaunchParams.useBase2ExpTrick = true;
        }

        if (s > sliding_window_size && mLaunchParams.attention_mask_type == ContextAttentionMaskType::CAUSAL)
        {
            CHECK_WITH_INFO(!isSm70, "Sliding window attention is not supported for FMHA on Volta");
            mLaunchParams.attention_mask_type = ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL;
        }

        setup_params(mParams, b, s, s, sliding_window_size, total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
        mParams.qkv_stride_in_bytes = (mNumHeads + 2 * mParams.h_kv) * mHeadSize * sizeof(half);
    }

    void setup_paged_kv(int const b, int const s_q, int const s_kv, int const blocks_per_context_sequence,
        int const tokens_per_kv_block, int const sliding_window_size, int const total_seqlen, bool const has_alibi,
        bool const scale_alibi, int const tp_size, int const tp_rank)
    {

        mLaunchParams.set_default_kernel_selection_params();

        mLaunchParams.blocks_per_context_sequence = blocks_per_context_sequence;
        CHECK_WITH_INFO(mHeadSize > 0, "Head size should be greater than 0.");
        mLaunchParams.padded_d = (mHeadSize & (mHeadSize - 1)) == 0 ? mHeadSize : pow(2, int(log2(mHeadSize)) + 1);

        bool const isSm90 = (sm == kSM_90);
        bool const isSm8x = (sm == kSM_86 || sm == kSM_89);
        bool const isSm80 = (sm == kSM_80);

        mLaunchParams.flash_attention = true;
        mLaunchParams.kernel_s = 0;
        mLaunchParams.kernel_kv_s = s_kv;
        mLaunchParams.force_unroll = true;

        if (isSm90 && s_kv > 512)
        {
            mLaunchParams.warp_specialization = true;
            mLaunchParams.use_tma = true;
        }
        else
        {
            if (mLaunchParams.flash_attention && s_kv <= 64)
            {
                mLaunchParams.granular_tiling = false;
            }
            else if (isSm8x && mParams.d < 256)
            {
                mLaunchParams.granular_tiling = false;
            }
            else if (isSm90 || isSm80 || isSm8x)
            {
                mLaunchParams.granular_tiling = true;
            }
        }

        if (mLaunchParams.warp_specialization && !has_alibi)
        {
            mLaunchParams.useKernelWithoutAlibi = true;
            mLaunchParams.useBase2ExpTrick = true;
        }

        if (s_kv > sliding_window_size && mLaunchParams.attention_mask_type == ContextAttentionMaskType::CAUSAL)
        {
            mLaunchParams.attention_mask_type = ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL;
        }

        setup_params(
            mPagedKVParams, b, s_q, s_kv, sliding_window_size, total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
        mPagedKVParams.q_stride_in_bytes = mNumHeads * mHeadSize * sizeof(half);
        mPagedKVParams.kv_stride_in_bytes = tokens_per_kv_block * mHeadSize * sizeof(half);
    }

    void set_tma_descriptors()
    {
        const uint32_t d_in_bytes = mLaunchParams.padded_d * sizeof(uint16_t);
        const uint32_t d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

        Multiple_tma_descriptor<4> qkv_tma_descriptor;

        uint32_t tensor_size_qkv[4];
        if (mParams.h_kv < mParams.h)
        {
            tensor_size_qkv[2] = 1;
            tensor_size_qkv[1] = (mParams.h + 2 * mParams.h_kv);
            tensor_size_qkv[0] = mParams.d;
        }
        else
        {
            tensor_size_qkv[2] = 3;
            tensor_size_qkv[1] = mParams.h;
            tensor_size_qkv[0] = mParams.d;
        }

        uint32_t box_size[4];
        box_size[2] = 1;
        box_size[1] = 1;
        box_size[0] = mLaunchParams.padded_d / d_groups;

        uint64_t tensor_stride_qkv[3];
        tensor_stride_qkv[0] = tensor_size_qkv[0] * sizeof(uint16_t);
        tensor_stride_qkv[1] = tensor_size_qkv[1] * tensor_stride_qkv[0];
        tensor_stride_qkv[2] = tensor_size_qkv[2] * tensor_stride_qkv[1];

        uint32_t traversal_stride_qkv[4] = {1, 1, 1, 1};

        uint32_t oob_fill = 0;

        uint32_t fp32_to_tf32 = 0;

        const uint32_t d_bytes_per_group = (mLaunchParams.padded_d * sizeof(uint16_t)) / d_groups;
        const cudaTmaDescSwizzle swizzle_mode = (d_bytes_per_group > 64
                ? cudaTmaDescSwizzle::SWIZZLE_128B
                : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

        uint32_t q_step = 0, kv_step = 0;
        for (unsigned int i = 0u; i < sizeof(sTmaMetaInfo) / sizeof(sTmaMetaInfo[0]); ++i)
        {
            if (sTmaMetaInfo[i].mD == mLaunchParams.padded_d)
            {
                q_step = sTmaMetaInfo[i].mQStep;
                kv_step = sTmaMetaInfo[i].mKVStep;
                break;
            }
        }

        char const* qkv_ptr = reinterpret_cast<char const*>(mParams.qkv_ptr);
        tensor_size_qkv[3] = mTotalSeqLen;

        box_size[3] = q_step;
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_qkv, tensor_stride_qkv, traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32,
            &mParams.tma_desc_q);

        box_size[3] = kv_step;
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_qkv, tensor_stride_qkv, traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32,
            &mParams.tma_desc_k);
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_qkv, tensor_stride_qkv, traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32,
            &mParams.tma_desc_v);
    }

    void set_paged_kv_tma_descriptors(cudaStream_t stream)
    {
        const uint32_t d_in_bytes = mLaunchParams.padded_d * sizeof(uint16_t);
        const uint32_t d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

        uint32_t q_step = 0, kv_step = 0;
        for (unsigned int i = 0u; i < sizeof(sTmaMetaInfo) / sizeof(sTmaMetaInfo[0]); ++i)
        {
            if (sTmaMetaInfo[i].mD == mLaunchParams.padded_d)
            {
                q_step = sTmaMetaInfo[i].mQStep;
                kv_step = sTmaMetaInfo[i].mKVStep;
                break;
            }
        }

        Multiple_tma_descriptor<4> q_tma_descriptor;
        Multiple_tma_descriptor<4> paged_kv_tma_descriptor(
            mPagedKVParams.b * 2 * mLaunchParams.blocks_per_context_sequence);
        uint32_t tensor_size_q[4];
        tensor_size_q[3] = mTotalSeqLen;
        tensor_size_q[2] = 1;
        tensor_size_q[1] = mPagedKVParams.h;
        tensor_size_q[0] = mPagedKVParams.d;

        uint32_t box_size_q[4];
        box_size_q[3] = q_step;
        box_size_q[2] = 1;
        box_size_q[1] = 1;
        box_size_q[0] = mLaunchParams.padded_d / d_groups;

        uint64_t tensor_stride_q[3];
        tensor_stride_q[0] = tensor_size_q[0] * sizeof(uint16_t);
        tensor_stride_q[1] = tensor_size_q[1] * tensor_stride_q[0];
        tensor_stride_q[2] = tensor_size_q[2] * tensor_stride_q[1];

        uint32_t traversal_stride[4] = {1, 1, 1, 1};

        uint32_t oob_fill = 0;

        uint32_t fp32_to_tf32 = 0;

        const uint32_t d_bytes_per_group = (mLaunchParams.padded_d * sizeof(uint16_t)) / d_groups;
        const cudaTmaDescSwizzle swizzle_mode = (d_bytes_per_group > 64
                ? cudaTmaDescSwizzle::SWIZZLE_128B
                : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

        char const* q_ptr = reinterpret_cast<char const*>(mPagedKVParams.q_ptr);

        q_tma_descriptor.set_tma_desctriptor(q_ptr, cudaTmaDescFormat::F16_RN,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_q, tensor_stride_q, traversal_stride, box_size_q, oob_fill, fp32_to_tf32,
            &mPagedKVParams.tma_desc_q);

        uint32_t tokens_per_block = uint32_t(mPagedKVParams.paged_kv_cache.mTokensPerBlock);
        uint32_t tensor_size_kv[4];
        tensor_size_kv[3] = 1;
        tensor_size_kv[2] = mPagedKVParams.h_kv;
        tensor_size_kv[1] = tokens_per_block;
        tensor_size_kv[0] = mPagedKVParams.d;

        uint32_t box_size_kv[4];
        box_size_kv[3] = 1;
        box_size_kv[2] = 1;
        box_size_kv[1] = std::min(tokens_per_block, kv_step);
        box_size_kv[0] = mLaunchParams.padded_d / d_groups;

        CHECK_WITH_INFO(
            tokens_per_block % 2 == 0, "FMHA with paged kv cache needs tokens_per_block to be power of 2 !");
        mPagedKVParams.blocks_per_tma_load = std::max(1, int32_t(kv_step / tokens_per_block));
        mPagedKVParams.blocks_per_tma_load_log2 = log2(mPagedKVParams.blocks_per_tma_load);

        uint64_t tensor_stride_kv[3];
        tensor_stride_kv[0] = tensor_size_kv[0] * sizeof(uint16_t);
        tensor_stride_kv[1] = tensor_size_kv[1] * tensor_stride_kv[0];
        tensor_stride_kv[2] = tensor_size_kv[2] * tensor_stride_kv[1];

        CHECK_WITH_INFO(
            mPagedKVParams.paged_kv_cache.mMaxBlocksPerSeq == mLaunchParams.blocks_per_context_sequence,
            "Mismatching blocks_per_sequence for the paged kv FMHA.");
        for (int block_idx = 0; block_idx < mPagedKVParams.b * 2 * mLaunchParams.blocks_per_context_sequence;
             block_idx++)
        {
            paged_kv_tma_descriptor.set_tma_desctriptor(
                reinterpret_cast<char*>(mLaunchParams.paged_kv_block_ptrs[block_idx]), cudaTmaDescFormat::F16_RN,
                cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
                tensor_size_kv, tensor_stride_kv, traversal_stride, box_size_kv, oob_fill, fp32_to_tf32, block_idx);
        }

        paged_kv_tma_descriptor.copy_to_device(mPagedKVParams.tma_desc_paged_kv, stream);
    }

    void setup_flags(bool const force_fp32_acc, bool const is_s_padded, bool const causal_mask, int const num_kv_heads)
    {
        mLaunchParams.force_fp32_acc = mDataType == DATA_TYPE_BF16 || force_fp32_acc;
        mLaunchParams.attention_mask_type
            = causal_mask ? ContextAttentionMaskType::CAUSAL : ContextAttentionMaskType::PADDING;

        mPagedKVParams.h_kv = num_kv_heads;
        CHECK_WITH_INFO(mNumHeads % num_kv_heads == 0, "number of Query heads should be multiple of KV heads !");
        mPagedKVParams.h_q_per_kv = mNumHeads / num_kv_heads;
        mPagedKVParams.is_s_padded = is_s_padded;

        mParams.h_kv = num_kv_heads;
        mParams.is_s_padded = is_s_padded;
    }

    bool fmha_supported()
    {
        return MHARunner::fmha_supported(mHeadSize, sm);
    }

    void run(void const* qkvPtr, void const* cuSeqlenPtr, void* outputPtr, cudaStream_t stream)
    {
        mParams.qkv_ptr = qkvPtr;
        mParams.o_ptr = outputPtr;
        mParams.cu_seqlens = reinterpret_cast<int const*>(cuSeqlenPtr);

        if (sm == kSM_90 && mLaunchParams.use_tma)
        {
            set_tma_descriptors();
        }

        xmmaKernel->run(mParams, mLaunchParams, stream);
    }

    void run_paged_kv(void const* qPtr, void* pagedKVTmaDesc, void const* pagedKVBlockPtrsOnHost,
        const KVBlockArray pagedKVCache, void const* cuQSeqlenPtr, void const* cuKVSeqlenPtr, void* outputPtr,
        cudaStream_t stream)
    {
        KVBlockArrayForContextFMHA pagedKVCacheForContextMHA;
        pagedKVCacheForContextMHA = KVBlockArrayForContextFMHA(pagedKVCache.mMaxSeqs, pagedKVCache.mMaxBlocksPerSeq,
            pagedKVCache.mTokensPerBlock, mPagedKVParams.h_kv * mPagedKVParams.d * sizeof(half));
        pagedKVCacheForContextMHA.data = pagedKVCache.data;
        mPagedKVParams.q_ptr = qPtr;
        mPagedKVParams.tma_desc_paged_kv = reinterpret_cast<cudaTmaDesc*>(pagedKVTmaDesc);
        mPagedKVParams.paged_kv_cache = pagedKVCacheForContextMHA;
        mPagedKVParams.o_ptr = outputPtr;
        mPagedKVParams.cu_q_seqlens = reinterpret_cast<int const*>(cuQSeqlenPtr);
        mPagedKVParams.cu_seqlens = reinterpret_cast<int const*>(cuKVSeqlenPtr);
        mLaunchParams.paged_kv_block_ptrs = reinterpret_cast<int64_t const*>(pagedKVBlockPtrsOnHost);

        if (sm == kSM_90 && mLaunchParams.use_tma)
        {
            set_paged_kv_tma_descriptors(stream);
        }

        pagedKVXmmaKernel->run(mPagedKVParams, mLaunchParams, stream);
    }

    bool isValid(int s) const
    {
        return pagedKVXmmaKernel->isValid(s) && xmmaKernel->isValid(s);
    }

    int getSFromMaxSeqLen(int const max_seq_len)
    {
        int S = 1024;

        if (max_seq_len <= 64)
        {
            S = 64;
        }
        else if (max_seq_len <= 128)
        {
            S = 128;
        }
        else if (max_seq_len <= 256)
        {
            S = 256;
        }
        else if (max_seq_len <= 384)
        {
            S = 384;
        }
        else if (max_seq_len <= 512)
        {
            S = 512;
        }
        else if (max_seq_len > 512)
        {
            S = max_seq_len;
        }

        return S;
    }

private:
    Fused_multihead_attention_params_v2 mParams;
    Fused_multihead_attention_paged_kv_params_v2 mPagedKVParams;
    Launch_params mLaunchParams;
    int sm;
    FusedMultiHeadAttentionXMMAKernelV2 const* xmmaKernel;
    FusedMultiHeadAttentionPagedKVXMMAKernelV2 const* pagedKVXmmaKernel;
    bool use_flash_attention = false;
    const Data_type mDataType;
    int const mNumHeads;
    int const mHeadSize;
    float const mQScaling;
    int mTotalSeqLen;
};


FusedMHARunnerV2::FusedMHARunnerV2(
    const Data_type data_type, int const numHeads, int const headSize, float const qScaling)
    : pimpl(new mhaImpl(data_type, numHeads, headSize, qScaling, nexly::common::getSMVersion()))
{
}

FusedMHARunnerV2::~FusedMHARunnerV2() = default;

void FusedMHARunnerV2::setup(int const b, int const s, int const sliding_window_size, int const total_seqlen,
    bool const has_alibi, bool const scale_alibi, int const tp_size, int const tp_rank)
{
    pimpl->setup(b, s, sliding_window_size, total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
}

void FusedMHARunnerV2::setup_paged_kv(int const b, int const s_q, int const s_kv, int const blocks_per_context_sequence,
    int const tokens_per_kv_block, int const sliding_window_size, int const total_seqlen, bool const has_alibi,
    bool const scale_alibi, int const tp_size, int const tp_rank)
{
    pimpl->setup_paged_kv(b, s_q, s_kv, blocks_per_context_sequence, tokens_per_kv_block, sliding_window_size,
        total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
}

bool FusedMHARunnerV2::fmha_supported()
{
    return pimpl->fmha_supported();
}

void FusedMHARunnerV2::setup_flags(
    bool const force_fp32_acc, bool const is_s_padded, bool const causal_mask, int const num_kv_heads)
{
    pimpl->setup_flags(force_fp32_acc, is_s_padded, causal_mask, num_kv_heads);
}

void FusedMHARunnerV2::run(void const* qkvPtr, void const* cuSeqlenPtr, void* outputPtr, cudaStream_t stream)
{
    pimpl->run(qkvPtr, cuSeqlenPtr, outputPtr, stream);
}

void FusedMHARunnerV2::run_paged_kv(void const* qPtr, void* pagedKVTmaDesc, void const* pagedKVBlockPtrsOnHost,
    const KVBlockArray pagedKVCache, void const* cuQSeqlenPtr, void const* cuKVSeqlenPtr, void* outputPtr,
    cudaStream_t stream)
{
    pimpl->run_paged_kv(
        qPtr, pagedKVTmaDesc, pagedKVBlockPtrsOnHost, pagedKVCache, cuQSeqlenPtr, cuKVSeqlenPtr, outputPtr, stream);
}

bool FusedMHARunnerV2::isValid(int s) const
{
    return pimpl->isValid(s);
}

bool MHARunner::fmha_supported(int const headSize, int const sm)
{
    if (sm == kSM_70)
    {
        return (headSize == 32 || headSize == 40 || headSize == 64 || headSize == 80 || headSize == 128
            || headSize == 160 || headSize == 256);
    }
    else if (sm == kSM_80 || sm == kSM_86 || sm == kSM_89)
    {
        return (headSize == 16 || headSize == 32 || headSize == 40 || headSize == 64 || headSize == 80 || headSize == 96
            || headSize == 104 || headSize == 128 || headSize == 160 || headSize == 256);
    }
    else if (sm == kSM_90)
    {
        return (headSize == 32 || headSize == 40 || headSize == 64 || headSize == 80 || headSize == 96
            || headSize == 104 || headSize == 128 || headSize == 160 || headSize == 256);
    }

    return false;
}

}
}
