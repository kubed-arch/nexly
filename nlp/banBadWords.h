#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace nexly
{
    namespace nlp
    {
        /// <summary>
        /// Invokes the function to ban bad words from the logits.
        /// </summary>
        /// <typeparam name="T">Type of the logits.</typeparam>
        /// <param name="logits">The logits array.</param>
        /// <param name="output_ids_ptr">Pointer to the output ids.</param>
        /// <param name="parent_ids_ptr">Pointer to the parent ids.</param>
        /// <param name="batch_slot">Batch slot.</param>
        /// <param name="batch_size">Batch size.</param>
        /// <param name="beam_width">Beam width.</param>
        /// <param name="bad_words">Bad words array.</param>
        /// <param name="bad_words_len">Array containing lengths of bad words.</param>
        /// <param name="max_bad_words_len">Maximum length of bad words.</param>
        /// <param name="vocab_size_padded">Padded vocabulary size.</param>
        /// <param name="sequence_lengths">Array containing lengths of sequences.</param>
        /// <param name="max_seq_len">Maximum sequence length.</param>
        /// <param name="stream">CUDA stream.</param>
        template <typename T>
        void invokeBanBadWords(T* logits, int32_t const** output_ids_ptr, int32_t const** parent_ids_ptr,
            int32_t const* batch_slot, int32_t batch_size, int32_t beam_width, int32_t const** bad_words,
            int32_t const* bad_words_len, int32_t max_bad_words_len, int32_t vocab_size_padded, int32_t const* sequence_lengths,
            int32_t max_seq_len, cudaStream_t stream);
    }
}
