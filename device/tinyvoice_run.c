/* Standalone on-device speech recogniser for the Allwinner NPU.
 *
 *   ./tinyvoice_run model.nb filterbank.bin vocab.txt audio.wav
 *
 * The NPU runs the encoder (mel -> CTC logits) out of the .nb.  Everything
 * around it happens here on the CPU, mirroring conformer/frontend_np.py:
 * STFT, mel projection, log, per-utterance normalisation, then greedy CTC
 * decoding of the logits.  That split is deliberate -- the frontend is
 * numerically sensitive and cheap, the encoder is neither.
 *
 * The .nb takes pre-quantised int16 at its inputs (dynamic fixed point, so
 * value = q / 2^fl); awnn_get_output_buffers() hands the logits back already
 * converted to float.  The exponents live in the generated tinyvoice_model.h.
 *
 * Build with device/Makefile inside your Tina/Longan SDK -- it needs that
 * SDK's aarch64 cross toolchain and the viplite-tina libraries.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <awnn_lib.h>

#include "tinyvoice_frontend.h"
#include "tinyvoice_model.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.nb filterbank.bin vocab.txt audio.wav\n", argv[0]);
        return 1;
    }

    float *filterbank = tv_load_filterbank(argv[2]);
    char **vocab = tv_load_vocab(argv[3]);
    if (!filterbank || !vocab) return 1;

    size_t samples = 0;
    float *audio = tv_load_wav(argv[4], &samples);
    if (!audio || samples == 0) { fprintf(stderr, "no audio decoded\n"); return 1; }

    awnn_init();
    Awnn_Context_t *context = awnn_create(argv[1]);
    if (!context) { fprintf(stderr, "awnn_create failed on %s\n", argv[1]); awnn_uninit(); return 1; }

    float *mel = (float *)malloc((size_t)TV_MEL_FRAMES * TV_N_MELS * sizeof(float));
    int16_t *mel_q = (int16_t *)malloc((size_t)TV_MEL_FRAMES * TV_N_MELS * sizeof(int16_t));
    int16_t *mask_q = (int16_t *)malloc((size_t)TV_SEQ_LEN * sizeof(int16_t));
    if (!mel || !mel_q || !mask_q) { fprintf(stderr, "out of memory\n"); return 1; }

    const float mel_scale = (float)(1 << TV_MEL_FL);
    const float mask_scale = (float)(1 << TV_MASK_FL);

    /* The graph has a fixed sequence length, so longer audio is cut into
     * window-sized pieces.  Attention cannot see across a boundary. */
    size_t windows = (samples + TV_WINDOW_SAMPLES - 1) / TV_WINDOW_SAMPLES;
    int total_tokens = 0;
    int *collected = (int *)malloc(windows * TV_SEQ_LEN * sizeof(int));
    if (!collected) { fprintf(stderr, "out of memory\n"); return 1; }

    for (size_t w = 0; w < windows; w++) {
        size_t offset = w * (size_t)TV_WINDOW_SAMPLES;
        size_t length = samples - offset;
        if (length > (size_t)TV_WINDOW_SAMPLES) length = TV_WINDOW_SAMPLES;

        int valid_mel = 0;
        tv_log_mel(audio + offset, length, filterbank, mel, &valid_mel);
        int valid_out = tv_subsampled_length(valid_mel);
        if (valid_out < 1) valid_out = 1;
        if (valid_out > TV_SEQ_LEN) valid_out = TV_SEQ_LEN;

        for (size_t i = 0; i < (size_t)TV_MEL_FRAMES * TV_N_MELS; i++) {
            float v = roundf(mel[i] * mel_scale);
            if (v > 32767.0f) v = 32767.0f;
            if (v < -32768.0f) v = -32768.0f;
            mel_q[i] = (int16_t)v;
        }
        for (int i = 0; i < TV_SEQ_LEN; i++) {
            /* mask is 0 or 1 and fl is 15, so 1.0 saturates to 32767
             * (0.99997) -- harmless, and clamping is required regardless. */
            float v = roundf((i < valid_out ? 1.0f : 0.0f) * mask_scale);
            if (v > 32767.0f) v = 32767.0f;
            mask_q[i] = (int16_t)v;
        }

        void *inputs[] = { mel_q, mask_q };
        awnn_set_input_buffers(context, inputs);
        awnn_run(context);
        float **outputs = awnn_get_output_buffers(context);

        for (int t = 0; t < valid_out; t++) {
            const float *row = outputs[0] + (size_t)t * TV_VOCAB_SIZE;
            int best = 0;
            for (int v = 1; v < TV_VOCAB_SIZE; v++) if (row[v] > row[best]) best = v;
            collected[total_tokens++] = best;
        }
    }

    /* Greedy CTC: collapse repeats, drop blanks. */
    printf("transcription: ");
    int previous = TV_BLANK_ID;
    for (int i = 0; i < total_tokens; i++) {
        int id = collected[i];
        if (id != previous && id != TV_BLANK_ID && id != TV_PAD_ID) fputs(vocab[id], stdout);
        previous = id;
    }
    printf("\n");

    free(collected); free(mask_q); free(mel_q); free(mel);
    free(audio); free(filterbank);
    for (int i = 0; i < TV_VOCAB_SIZE; i++) free(vocab[i]);
    free(vocab);

    awnn_destroy(context);
    awnn_uninit();
    return 0;
}
