/* CPU-side pieces of the on-device recogniser: everything that does not touch
 * the NPU.  Split out from tinyvoice_run.c so the DSP can be compiled and
 * checked against conformer/frontend_np.py on the host, where there is no
 * VIPLite -- see device/host_test.c.
 */

#ifndef TINYVOICE_FRONTEND_H
#define TINYVOICE_FRONTEND_H

#include <stddef.h>

/* 16-bit PCM WAVE at the model's sample rate. Caller frees. */
float *tv_load_wav(const char *path, size_t *count);

/* float32 mel filterbank, TV_N_MELS x TV_N_FREQ row major. Caller frees. */
float *tv_load_filterbank(const char *path);

/* One UTF-8 token per line, in id order. Caller frees entries and array. */
char **tv_load_vocab(const char *path);

/* Normalised log-mel for one window, written as (TV_MEL_FRAMES, TV_N_MELS)
 * row major.  *valid_frames receives the count backed by real audio; the rest
 * is zero padding, excluded from the normalisation statistics. */
void tv_log_mel(const float *audio, size_t samples, const float *filterbank,
                float *mel, int *valid_frames);

/* Encoder frames left after the two stride-2 subsampling convolutions. */
int tv_subsampled_length(int frames);

#endif
