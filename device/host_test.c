/* Host-side check for the device frontend.
 *
 * The DSP in tinyvoice_frontend.c is a hand port of conformer/frontend_np.py,
 * and a silent drift there would show up as unexplained accuracy loss on the
 * board with nothing to point at.  This builds the same code natively (no
 * VIPLite involved), dumps the mel it produces, and lets
 * scripts/check_device_frontend.py diff it against the Python reference.
 *
 *   cc -O2 -o host_test host_test.c tinyvoice_frontend.c -I<device assets> -lm
 *   ./host_test filterbank.bin audio.wav mel.bin
 */

#include <stdio.h>
#include <stdlib.h>

#include "tinyvoice_frontend.h"
#include "tinyvoice_model.h"

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s filterbank.bin audio.wav out_mel.bin\n", argv[0]);
        return 1;
    }

    float *filterbank = tv_load_filterbank(argv[1]);
    if (!filterbank) return 1;

    size_t samples = 0;
    float *audio = tv_load_wav(argv[2], &samples);
    if (!audio || samples == 0) { fprintf(stderr, "no audio decoded\n"); return 1; }
    if (samples > (size_t)TV_WINDOW_SAMPLES) samples = TV_WINDOW_SAMPLES;

    float *mel = (float *)malloc((size_t)TV_MEL_FRAMES * TV_N_MELS * sizeof(float));
    if (!mel) return 1;

    int valid = 0;
    tv_log_mel(audio, samples, filterbank, mel, &valid);

    FILE *fp = fopen(argv[3], "wb");
    if (!fp) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    fwrite(mel, sizeof(float), (size_t)TV_MEL_FRAMES * TV_N_MELS, fp);
    fclose(fp);

    printf("samples=%zu valid_mel_frames=%d encoder_frames=%d\n",
           samples, valid, tv_subsampled_length(valid));

    free(mel); free(audio); free(filterbank);
    return 0;
}
