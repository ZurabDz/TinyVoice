/* CPU frontend for the on-device recogniser -- see tinyvoice_frontend.h.
 * Mirrors conformer/frontend_np.py so the board reproduces exactly the
 * preprocessing the model was trained and exported with.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyvoice_frontend.h"
#include "tinyvoice_model.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ audio */

/* Minimal RIFF/WAVE reader: 16-bit PCM only, which is what arecord produces.
 * Stereo is downmixed; a sample rate other than the model's is refused rather
 * than silently resampled, because resampling badly is worse than not doing it. */
float *tv_load_wav(const char *path, size_t *count)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return NULL; }

    unsigned char header[12];
    if (fread(header, 1, 12, fp) != 12 || memcmp(header, "RIFF", 4) || memcmp(header + 8, "WAVE", 4)) {
        fprintf(stderr, "%s is not a RIFF/WAVE file\n", path);
        fclose(fp);
        return NULL;
    }

    unsigned short channels = 0, bits = 0;
    unsigned int rate = 0;
    float *samples = NULL;

    for (;;) {
        unsigned char chunk[8];
        if (fread(chunk, 1, 8, fp) != 8) break;
        unsigned int size = chunk[4] | (chunk[5] << 8) | (chunk[6] << 16) | ((unsigned)chunk[7] << 24);

        if (!memcmp(chunk, "fmt ", 4)) {
            unsigned char fmt[16];
            if (fread(fmt, 1, 16, fp) != 16) break;
            channels = fmt[2] | (fmt[3] << 8);
            rate     = fmt[4] | (fmt[5] << 8) | (fmt[6] << 16) | ((unsigned)fmt[7] << 24);
            bits     = fmt[14] | (fmt[15] << 8);
            if (size > 16) fseek(fp, size - 16, SEEK_CUR);
        } else if (!memcmp(chunk, "data", 4)) {
            if (bits != 16 || channels < 1) {
                fprintf(stderr, "%s: need 16-bit PCM (got %u-bit, %u channels)\n", path, bits, channels);
                break;
            }
            if (rate != TV_SAMPLE_RATE) {
                fprintf(stderr, "%s: sample rate is %u, model needs %d -- resample first\n",
                        path, rate, TV_SAMPLE_RATE);
                break;
            }
            size_t frames = size / (size_t)(2 * channels);
            int16_t *raw = (int16_t *)malloc(size);
            samples = (float *)malloc(frames * sizeof(float));
            if (!raw || !samples) { free(raw); free(samples); samples = NULL; break; }
            if (fread(raw, 1, size, fp) != size) { free(raw); free(samples); samples = NULL; break; }

            for (size_t i = 0; i < frames; i++) {
                int accumulator = 0;
                for (int c = 0; c < channels; c++) accumulator += raw[i * channels + c];
                samples[i] = (float)accumulator / (channels * 32768.0f);
            }
            free(raw);
            *count = frames;
            break;
        } else {
            fseek(fp, size + (size & 1), SEEK_CUR);
        }
    }

    fclose(fp);
    return samples;
}

/* -------------------------------------------------------------------- fft */

/* In-place iterative radix-2 FFT.  TV_N_FFT is 512, a power of two, so the
 * simple form is enough -- this is a few percent of total runtime. */
static void fft(float *re, float *im, int n)
{
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double theta = -2.0 * M_PI / len;
        float wr = (float)cos(theta), wi = (float)sin(theta);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                float xr = re[b] * cr - im[b] * ci;
                float xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
                float nr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = nr;
            }
        }
    }
}

/* --------------------------------------------------------------- frontend */

/* Normalised log-mel for one window, written as (frames, n_mels) row major to
 * match the graph's NCHW input.  `valid_frames` bounds the real audio; the
 * statistics use only those, then the padding is zeroed -- same as training. */
void tv_log_mel(const float *audio, size_t samples, const float *filterbank,
                    float *mel, int *valid_frames)
{
    const int pad = TV_WIN_LENGTH / 2;
    const size_t padded_len = samples + 2 * (size_t)pad;

    float *padded = (float *)calloc(padded_len + TV_WIN_LENGTH, sizeof(float));
    float *window = (float *)malloc(TV_WIN_LENGTH * sizeof(float));
    float *re = (float *)malloc(TV_N_FFT * sizeof(float));
    float *im = (float *)malloc(TV_N_FFT * sizeof(float));
    float *power = (float *)malloc(TV_N_FREQ * sizeof(float));
    if (!padded || !window || !re || !im || !power) { fprintf(stderr, "out of memory\n"); exit(1); }

    memcpy(padded + pad, audio, samples * sizeof(float));

    /* Periodic Hann, matching scipy's get_window(..., fftbins=True). */
    double window_sum = 0.0;
    for (int i = 0; i < TV_WIN_LENGTH; i++) {
        window[i] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * i / TV_WIN_LENGTH));
        window_sum += window[i];
    }

    int frames = (int)(samples / TV_HOP_LENGTH) + 1;
    if (frames > TV_MEL_FRAMES) frames = TV_MEL_FRAMES;
    *valid_frames = frames;

    for (int t = 0; t < TV_MEL_FRAMES; t++) {
        if (t >= frames) {                       /* padding: zeroed below */
            memset(mel + (size_t)t * TV_N_MELS, 0, TV_N_MELS * sizeof(float));
            continue;
        }
        size_t start = (size_t)t * TV_HOP_LENGTH;
        memset(re, 0, TV_N_FFT * sizeof(float));
        memset(im, 0, TV_N_FFT * sizeof(float));
        for (int i = 0; i < TV_WIN_LENGTH; i++) re[i] = padded[start + i] * window[i];
        fft(re, im, TV_N_FFT);

        /* scaling="spectrum": amplitude divided by the window sum. */
        for (int f = 0; f < TV_N_FREQ; f++) {
            float r = re[f] / (float)window_sum, m = im[f] / (float)window_sum;
            power[f] = r * r + m * m;
        }
        for (int b = 0; b < TV_N_MELS; b++) {
            const float *row = filterbank + (size_t)b * TV_N_FREQ;
            float acc = 0.0f;
            for (int f = 0; f < TV_N_FREQ; f++) acc += row[f] * power[f];
            mel[(size_t)t * TV_N_MELS + b] = logf(acc + 5.9604645e-8f);  /* 2^-24 */
        }
    }

    /* Per-mel-channel mean/variance over the valid frames only. */
    for (int b = 0; b < TV_N_MELS; b++) {
        double mean = 0.0, var = 0.0;
        for (int t = 0; t < frames; t++) mean += mel[(size_t)t * TV_N_MELS + b];
        mean /= frames;
        for (int t = 0; t < frames; t++) {
            double d = mel[(size_t)t * TV_N_MELS + b] - mean;
            var += d * d;
        }
        var /= frames;
        float scale = 1.0f / (float)(sqrt(var) + 1e-5);
        for (int t = 0; t < frames; t++)
            mel[(size_t)t * TV_N_MELS + b] = (float)(mel[(size_t)t * TV_N_MELS + b] - mean) * scale;
        for (int t = frames; t < TV_MEL_FRAMES; t++)
            mel[(size_t)t * TV_N_MELS + b] = 0.0f;
    }

    free(padded); free(window); free(re); free(im); free(power);
}

int tv_subsampled_length(int frames)
{
    for (int i = 0; i < 2; i++) frames = (frames - 3) / 2 + 1;
    return frames < 0 ? 0 : frames;
}

/* ----------------------------------------------------------------- assets */

float *tv_load_filterbank(const char *path)
{
    size_t expected = (size_t)TV_N_MELS * TV_N_FREQ;
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    float *data = (float *)malloc(expected * sizeof(float));
    if (!data || fread(data, sizeof(float), expected, fp) != expected) {
        fprintf(stderr, "%s: expected %zu float32 values\n", path, expected);
        free(data); fclose(fp); return NULL;
    }
    fclose(fp);
    return data;
}

/* One token per line, UTF-8.  Tokens can be a space, so only the newline is
 * stripped -- nothing else may be trimmed. */
char **tv_load_vocab(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return NULL; }
    char **vocab = (char **)calloc(TV_VOCAB_SIZE, sizeof(char *));
    char line[256];
    int n = 0;
    while (n < TV_VOCAB_SIZE && fgets(line, sizeof(line), fp)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        vocab[n] = strdup(line);
        n++;
    }
    fclose(fp);
    if (n != TV_VOCAB_SIZE) {
        fprintf(stderr, "%s: got %d tokens, model has %d\n", path, n, TV_VOCAB_SIZE);
        free(vocab);
        return NULL;
    }
    return vocab;
}
