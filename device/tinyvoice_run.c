/* Standalone on-device speech recogniser for the Allwinner NPU.
 *
 *   ./tinyvoice_run model.nb filterbank.bin vocab.txt audio.wav [options]
 *
 *     --repeat N   run the pipeline N times and report per-stage timings
 *     --quiet      suppress the transcription (keeps the timing summary)
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
 * Benchmark output goes to stderr, because the SDK's awnn wrapper printf()s
 * its own per-call timings to stdout unconditionally -- send stdout to
 * /dev/null to see just the summary.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <awnn_lib.h>

#include "tinyvoice_frontend.h"
#include "tinyvoice_model.h"

/* ----------------------------------------------------------------- timing */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

typedef struct { double total, min, max; int n; } Stat;

static void stat_add(Stat *s, double value)
{
    if (!s->n || value < s->min) s->min = value;
    if (!s->n || value > s->max) s->max = value;
    s->total += value;
    s->n++;
}

static void stat_report(const char *label, const Stat *s, double wall)
{
    if (!s->n) return;
    fprintf(stderr, "  %-18s %7.2f ms  (min %6.2f  max %6.2f)  %5.1f%% of wall\n",
            label, s->total / s->n, s->min, s->max, 100.0 * s->total / wall);
}

/* Sustained throughput is partly a thermal question, so read the sensor rather
 * than assume the first iteration's rate holds. */
static int npu_temp_millicelsius(void)
{
    for (int zone = 0; zone < 32; zone++) {
        char path[128], type[64] = {0};
        FILE *fp;

        snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/type", zone);
        if (!(fp = fopen(path, "r"))) continue;
        if (!fgets(type, sizeof(type), fp)) { fclose(fp); continue; }
        fclose(fp);
        if (!strstr(type, "npu")) continue;

        snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", zone);
        if (!(fp = fopen(path, "r"))) continue;
        int value = -1;
        if (fscanf(fp, "%d", &value) != 1) value = -1;
        fclose(fp);
        return value;
    }
    return -1;
}

/* ------------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s model.nb filterbank.bin vocab.txt audio.wav [--repeat N] [--quiet]\n",
                argv[0]);
        return 1;
    }

    int repeat = 1, quiet = 0;
    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else { fprintf(stderr, "unknown option %s\n", argv[i]); return 1; }
    }
    if (repeat < 1) repeat = 1;

    float *filterbank = tv_load_filterbank(argv[2]);
    char **vocab = tv_load_vocab(argv[3]);
    if (!filterbank || !vocab) return 1;

    size_t samples = 0;
    float *audio = tv_load_wav(argv[4], &samples);
    if (!audio || samples == 0) { fprintf(stderr, "no audio decoded\n"); return 1; }

    double load_start = now_ms();
    awnn_init();
    Awnn_Context_t *context = awnn_create(argv[1]);
    if (!context) { fprintf(stderr, "awnn_create failed on %s\n", argv[1]); awnn_uninit(); return 1; }
    double load_ms = now_ms() - load_start;

    float *mel = (float *)malloc((size_t)TV_MEL_FRAMES * TV_N_MELS * sizeof(float));
    int16_t *mel_q = (int16_t *)malloc((size_t)TV_MEL_FRAMES * TV_N_MELS * sizeof(int16_t));
    int16_t *mask_q = (int16_t *)malloc((size_t)TV_SEQ_LEN * sizeof(int16_t));
    if (!mel || !mel_q || !mask_q) { fprintf(stderr, "out of memory\n"); return 1; }

    const float mel_scale = (float)(1 << TV_MEL_FL);
    const float mask_scale = (float)(1 << TV_MASK_FL);

    /* The graph has a fixed sequence length, so longer audio is cut into
     * window-sized pieces.  Attention cannot see across a boundary. */
    size_t windows = (samples + TV_WINDOW_SAMPLES - 1) / TV_WINDOW_SAMPLES;
    int *collected = (int *)malloc(windows * TV_SEQ_LEN * sizeof(int));
    if (!collected) { fprintf(stderr, "out of memory\n"); return 1; }

    Stat frontend = {0}, quantise = {0}, encoder = {0}, decode = {0};
    int temp_before = npu_temp_millicelsius();
    double wall_start = now_ms();

    for (int iteration = 0; iteration < repeat; iteration++) {
        int total_tokens = 0;

        for (size_t w = 0; w < windows; w++) {
            size_t offset = w * (size_t)TV_WINDOW_SAMPLES;
            size_t length = samples - offset;
            if (length > (size_t)TV_WINDOW_SAMPLES) length = TV_WINDOW_SAMPLES;

            double t0 = now_ms();
            int valid_mel = 0;
            tv_log_mel(audio + offset, length, filterbank, mel, &valid_mel);
            int valid_out = tv_subsampled_length(valid_mel);
            if (valid_out < 1) valid_out = 1;
            if (valid_out > TV_SEQ_LEN) valid_out = TV_SEQ_LEN;

            double t1 = now_ms();
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

            double t2 = now_ms();
            void *inputs[] = { mel_q, mask_q };
            awnn_set_input_buffers(context, inputs);
            awnn_run(context);
            float **outputs = awnn_get_output_buffers(context);

            double t3 = now_ms();
            for (int t = 0; t < valid_out; t++) {
                const float *row = outputs[0] + (size_t)t * TV_VOCAB_SIZE;
                int best = 0;
                for (int v = 1; v < TV_VOCAB_SIZE; v++) if (row[v] > row[best]) best = v;
                collected[total_tokens++] = best;
            }
            double t4 = now_ms();

            stat_add(&frontend, t1 - t0);
            stat_add(&quantise, t2 - t1);
            stat_add(&encoder, t3 - t2);
            stat_add(&decode, t4 - t3);
        }

        /* Greedy CTC: collapse repeats, drop blanks. */
        if (!quiet && iteration == 0) {
            printf("transcription: ");
            int previous = TV_BLANK_ID;
            for (int i = 0; i < total_tokens; i++) {
                int id = collected[i];
                if (id != previous && id != TV_BLANK_ID && id != TV_PAD_ID) fputs(vocab[id], stdout);
                previous = id;
            }
            printf("\n");
            fflush(stdout);
        }
    }

    double wall = now_ms() - wall_start;
    int temp_after = npu_temp_millicelsius();

    if (repeat > 1) {
        double audio_seconds = (double)samples / TV_SAMPLE_RATE * repeat;
        double window_seconds = (double)TV_WINDOW_SAMPLES / TV_SAMPLE_RATE;

        fprintf(stderr, "\n=== %d iterations x %zu window(s) ===\n", repeat, windows);
        fprintf(stderr, "  model load        %7.2f ms  (once, excluded below)\n", load_ms);
        stat_report("frontend (STFT)", &frontend, wall);
        stat_report("quantise input", &quantise, wall);
        stat_report("NPU encoder", &encoder, wall);
        stat_report("argmax + collect", &decode, wall);
        fprintf(stderr, "  ---\n");
        fprintf(stderr, "  wall              %7.2f s\n", wall / 1e3);
        fprintf(stderr, "  audio processed   %7.2f s\n", audio_seconds);
        fprintf(stderr, "  realtime factor   %7.1fx  (this clip)\n",
                audio_seconds / (wall / 1e3));
        fprintf(stderr, "  throughput        %7.2f windows/s\n",
                (double)(repeat * windows) / (wall / 1e3));
        /* A window costs the same however much speech it holds, so a full one
         * is the number to quote for capacity planning. */
        fprintf(stderr, "  full-window rate  %7.1fx  (%.0f s audio/s if windows were full)\n",
                (double)(repeat * windows) * window_seconds / (wall / 1e3),
                (double)(repeat * windows) * window_seconds / (wall / 1e3));
        if (temp_before >= 0 && temp_after >= 0)
            fprintf(stderr, "  NPU temperature   %7.1f C -> %.1f C\n",
                    temp_before / 1000.0, temp_after / 1000.0);
    }

    free(collected); free(mask_q); free(mel_q); free(mel);
    free(audio); free(filterbank);
    for (int i = 0; i < TV_VOCAB_SIZE; i++) free(vocab[i]);
    free(vocab);

    awnn_destroy(context);
    awnn_uninit();
    return 0;
}
