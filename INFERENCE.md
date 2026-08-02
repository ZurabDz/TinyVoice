# Inference and NPU deployment

Taking a trained checkpoint to a transcription running on the NPU of a Radxa
Cubie A7S (Allwinner A733).

Verified end to end on that board: VIPLite driver `2.0.3.2-AW-2024-08-30`,
Debian 13 aarch64, `/dev/vipcore` present. Output is identical to the float
model on the workstation, and the NPU sustains **45× realtime single-stream or
62× with two workers** — see [Step 8](#step-8-benchmark-and-stress-test).

Each step says whether it runs on the **workstation** or the **board**: the
export and conversion need the SDK container and the checkpoint, while the
build and run happen on the board.

- [How the model is split](#how-the-model-is-split)
- [Step 1. Export to ONNX](#step-1-export-to-onnx)
- [Step 2. Convert to a network binary](#step-2-convert-to-a-network-binary)
- [Step 3. Check the accuracy cost](#step-3-check-the-accuracy-cost)
- [Step 4. Generate the device assets](#step-4-generate-the-device-assets)
- [Step 5. Copy to the board](#step-5-copy-to-the-board)
- [Step 6. Build on the board](#step-6-build-on-the-board)
- [Step 7. Run](#step-7-run)
- [Step 8. Benchmark and stress test](#step-8-benchmark-and-stress-test)
- [Reference](#reference)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

---

## How the model is split

```
    audio ──▶ STFT ─▶ mel ─▶ log ─▶ normalise ──▶ encoder ──▶ CTC decode ─▶ text
              └──────────── CPU ────────────┘    └─ NPU ─┘    └── CPU ──┘
```

The frontend stays on the CPU because it is numerically sensitive — a log of
very small magnitudes, then a mean/variance reduction over the valid frames
only — and quantising it would dominate the error budget. It is also cheap next
to sixteen encoder blocks.

Two consequences worth knowing before you start:

**Sequence length is fixed at export time.** The NPU requires static shapes, so
the window (default 11 s, the training maximum) is baked into the graph. The
NPU spends the same 175 ms whether the window holds 2 s of speech or 11 s.
Longer audio is split into non-overlapping chunks and attention cannot see
across a boundary.

**A frame mask is a second input.** Utterances shorter than the window are
padded, and the mask marks which encoder frames are real. Masking is applied
after the softmax and renormalised rather than as an additive `-inf` bias
before it — mathematically identical, but the additive form puts a large
negative constant into the score tensor and wrecks its quantisation range.

The ONNX graph is built directly from the NNX weights
(`conformer/onnx_builder.py`) rather than traced. The ACUITY importer accepts a
narrow operator set — no `Einsum`, no fused attention — so emitting the graph by
hand pins the operators to ones it handles, and lets the RMS-norm pattern match
ACUITY's fusion rule exactly so it lowers to one native layer instead of seven.

---

## Step 1. Export to ONNX

> Runs on the **workstation**.

```bash
cd ~/Documents/projects/asr-ka/TinyVoice
python scripts/export_onnx.py --seconds 11 --calib-samples 64
```

Writes `artifacts/npu/tinyvoice/`:

| File | Contents |
|---|---|
| `tinyvoice.onnx` | the fixed-shape encoder graph (~89 MB) |
| `tinyvoice_meta.json` | frame geometry, vocabulary, frontend parameters |
| `filterbank.npy` | mel filterbank lifted from the trained model |
| `calib_mel.npy`, `calib_mask.npy` | calibration tensors for the quantiser |
| `inputs_outputs.txt` | import arguments for the toolkit |

Calibration uses real utterances from the packed dev split, or `--calib-dir`
with a folder of audio files. Silence or noise would mis-scale the encoder.

Before writing anything, the script runs the ONNX graph against the JAX model:

```
ONNX graph  artifacts/npu/tinyvoice/tinyvoice.onnx
  mel   (1, 1, 1101, 128)
  mask  (1, 1, 274, 1)
  logits(1, 274, 36)
  window 11.00s, 16 layers, 2091 nodes

Verifying ONNX against the JAX model:
  frames= 130  rel_err=9.25e-03  argmax_agreement=100.000%
  worst relative error: 9.25e-03  [OK]
```

Relative error around `1e-2` with 100 % argmax agreement is expected — the JAX
model computes in bfloat16, ONNX in float32. Anything materially larger means
the graph has drifted; investigate before quantising.

---

## Step 2. Convert to a network binary

> Runs on the **workstation**.

Needs the `ubuntu-npu` SDK container running (ACUITY 6.30.22 + Vivante tools).

```bash
docker ps                                    # find the container id
scripts/npu_convert.sh <container-id> a733 int16
```

**Use `a733`** for the Cubie A7S. The platform selects the hardware target
baked into the binary; a mismatch produces a `.nb` the driver will reject.

Five stages run inside the container — import ONNX, generate and patch the
input metadata, quantise, simulate float and quantised, pack the binary — and
these come back:

| File | Contents |
|---|---|
| `tinyvoice_int16.nb` | the network binary for the board (~49 MB) |
| `tinyvoice_int16_nbg_meta.json` | the quantisation the `.nb` expects at each tensor |
| `tinyvoice_int16.quantize` | per-tensor quantisation parameters |
| `inf/float/`, `inf/int16/` | simulator logit dumps for comparison |

Confirm the binary targets your chip:

```bash
python -c "
import struct; d=open('artifacts/npu/tinyvoice/tinyvoice_int16.nb','rb').read(12)
print(d[:4].decode(), hex(struct.unpack('<I',d[8:12])[0]))"
# VPMN 0x1000003b      <- a733; t527/ai985/mr527 would be 0x10000016
```

Arguments are `<container> <platform> [qtype] [model-dir]`. Environment:

- `RESUME=1` — reuse the container workspace and re-run only the export stage.
  Import and quantisation are the slow parts and neither depends on the
  platform, so this is the fast way to retarget a different chip.
- `PACK_FLAG` — defaults to `--pack-nbg-unify`, which is what this ACUITY
  version offers and what the A733's driver accepts.

### Choosing a quantisation type

| `qtype` | Quantiser | Notes |
|---|---|---|
| `int16` | `dynamic_fixed_point` | **default.** Verified on hardware, no accuracy loss observed |
| `bf16` | `qbfloat16` | closest to the float graph, where the target supports it |
| `pcq` | `perchannel_symmetric_affine` | int8 per channel |
| `uint8` | `asymmetric_affine` | usually too coarse for a 16-layer Conformer |

Per-tensor `uint8` struggles here because attention scores and the residual
stream both have wide dynamic range. Rather than take that on trust, measure it.

---

## Step 3. Check the accuracy cost

> Runs on the **workstation**.

```bash
python scripts/npu_compare.py artifacts/npu/tinyvoice
```

```
comparing 130 valid frames of 274

float32 : სკოლაში მეჩიურ ფეხბურთის გუნდის წევრი და მოსწავლეთა საბჭოს პრეზიდენტი იყო

int16   : სკოლაში მეჩიურ ფეხბურთის გუნდის წევრი და მოსწავლეთა საბჭოს პრეზიდენტი იყო
          cosine=0.999896  frame_argmax_agreement=100.00%  max_abs_err=0.5794
```

Both logit sets are decoded, so what you compare is the transcription
difference rather than a tensor metric that may or may not matter.

Optionally transcribe on the workstation before deploying:

```bash
python scripts/npu_infer.py sample.wav                       # ONNX Runtime, float
python scripts/npu_infer.py sample.wav --backend acuity \
    --container <container-id>                               # quantised simulator
```

The `acuity` backend is bit-accurate to the NPU but slow (~30 s per utterance —
each call pays for a container round trip and a graph rebuild).

---

## Step 4. Generate the device assets

> Runs on the **workstation**.

```bash
python scripts/export_device_assets.py --qtype int16
python scripts/check_device_frontend.py sample.wav
```

Writes `artifacts/npu/tinyvoice/device/`:

| File | Contents |
|---|---|
| `filterbank.bin` | float32 mel filterbank, `n_mels × (n_fft/2+1)` row major |
| `vocab.txt` | one UTF-8 token per line, in token-id order |
| `tinyvoice_model.h` | shapes and the fixed-point exponents the `.nb` expects |

The `.nb` takes **pre-quantised int16** at its inputs —
`awnn_set_input_buffers()` is a raw `memcpy`, so the runner converts. The
exponents come from the toolkit's own `nbg_meta.json`:

```
mel                [1, 1, 1101, 128] i16 fl=10      value = q / 2^10
mask               [1, 1, 274, 1]    i16 fl=15      value = q / 2^15
attach_logits/out0 [1, 274, 36]      i16 fl=10      dequantised by awnn
```

`check_device_frontend.py` builds the device C frontend natively and diffs the
mel it produces against the Python reference:

```
valid frames        : 526
max abs difference  : 1.473e-04  (8.79e-06 relative)
int16 input step    : 9.766e-04  [OK]
```

The difference must stay well under the int16 input step, or the C port has
drifted and the board will lose accuracy for reasons that are hard to attribute
later.

---

## Step 5. Copy to the board

Assuming `TinyVoice` and `ai-sdk` are cloned at `~/Documents/zura` on the
board. From the workstation:

```bash
BOARD=radxa@192.168.100.6
DEST=~/Documents/zura/TinyVoice

ssh $BOARD 'mkdir -p ~/Documents/zura/TinyVoice/artifacts/npu/tinyvoice/device'

scp artifacts/npu/tinyvoice/tinyvoice_int16.nb        $BOARD:$DEST/artifacts/npu/tinyvoice/
scp artifacts/npu/tinyvoice/device/filterbank.bin \
    artifacts/npu/tinyvoice/device/vocab.txt \
    artifacts/npu/tinyvoice/device/tinyvoice_model.h  $BOARD:$DEST/artifacts/npu/tinyvoice/device/
scp device/*.c device/*.h device/Makefile             $BOARD:$DEST/device/
scp sample.wav                                        $BOARD:$DEST/artifacts/npu/tinyvoice/
```

The `.nb` is ~49 MB, so this takes a moment.

---

## Step 6. Build on the board

> Runs on the **board**.

The Cubie A7S runs a full aarch64 Debian with gcc, so this builds **natively** —
no cross toolchain, no Tina SDK.

One prerequisite: gcc ships without the libc headers on this image.

```bash
sudo apt install -y libc6-dev
```

Then:

```bash
cd ~/Documents/zura/TinyVoice/device

make AI_SDK_ROOT=/home/radxa/Documents/zura/ai-sdk \
     AI_SDK_PLATFORM=a733 \
     MODEL_DIR=/home/radxa/Documents/zura/TinyVoice/artifacts/npu/tinyvoice/device
```

The Makefile reads `machinfo/a733/config.mk` for the driver generation (`v2.0`)
and links `-lNBGlinker -lVIPhal` from the ai-sdk checkout, with an RPATH so no
`LD_LIBRARY_PATH` is needed at run time. Confirm nothing is unresolved:

```bash
ldd tinyvoice_run | grep "not found"     # expect no output
```

---

## Step 7. Run

> Runs on the **board**.

Audio must be **16 kHz, 16-bit PCM, mono** — the reader refuses other sample
rates rather than resampling badly. To record on the board:

```bash
arecord -f S16_LE -r 16000 -c 1 -d 5 ~/test.wav
```

```bash
cd ~/Documents/zura/TinyVoice/artifacts/npu/tinyvoice

../../../device/tinyvoice_run \
    tinyvoice_int16.nb \
    device/filterbank.bin \
    device/vocab.txt \
    sample.wav
```

```
VIPLite driver software version 2.0.3.2-AW-2024-08-30
viplite init OK.
VIP cid=0x1000003b, device_count=1
awnn_create total: 95.47 ms.
  vip_run_network: 174.79 ms.
transcription: სკოლაში მეჩიურ ფეხბურთის გუნდის წევრი და მოსწავლეთა საბჭოს პრეზიდენტი იყო
```

`VIP cid` must match the `hex` you checked in Step 2. The diagnostic lines come
from the SDK's `awnn` wrapper, which logs unconditionally; for just the result:

```bash
../../../device/tinyvoice_run tinyvoice_int16.nb device/filterbank.bin \
    device/vocab.txt sample.wav 2>&1 | grep '^transcription:'
```

### Timings measured on the A7S

| Stage | Time | Note |
|---|---|---|
| `awnn_init` | 3.9 ms | once per process |
| `awnn_create` | 95 ms | loads and prepares the 49 MB network, once |
| `vip_run_network` | **175 ms** | per 11 s window → ~63× realtime |
| input quantise + copy | 0.1 ms | |
| output dequantise | 0.2 ms | |

The 175 ms is fixed per window regardless of how much speech it holds, so short
clips get proportionally less benefit.

---

## Step 8. Benchmark and stress test

> Runs on the **board**.

`--repeat N` runs the pipeline N times against one already-loaded model and
reports where the time goes. The summary is written to stderr, so sending
stdout to `/dev/null` drops the SDK's own per-call logging and leaves just the
numbers:

```bash
cd ~/Documents/zura/TinyVoice/artifacts/npu/tinyvoice

../../../device/tinyvoice_run tinyvoice_int16.nb device/filterbank.bin \
    device/vocab.txt sample_1.wav --repeat 100 --quiet >/dev/null
```

```
=== 100 iterations x 1 window(s) ===
  model load          99.56 ms  (once, excluded below)
  frontend (STFT)      42.71 ms  (min  32.05  max  49.53)   19.6% of wall
  quantise input        0.37 ms  (min   0.36  max   0.38)    0.2% of wall
  NPU encoder         174.83 ms  (min 174.32  max 176.38)   80.2% of wall
  argmax + collect      0.07 ms  (min   0.02  max   0.09)    0.0% of wall
  ---
  wall                21.80 s
  audio processed    608.40 s
  realtime factor      27.9x  (this clip)
  throughput           4.59 windows/s
  full-window rate     50.5x  (50 s audio/s if windows were full)
  ---
  NPU hardware time  174.34 ms/window  (driver reports)
  NPU cycles         175.68 M/window   (1008 MHz effective)
  NPU busy             80.0% of wall clock
  NPU temperature      38.8 C -> 44.3 C
```

The NPU stage is reproducible to well under a percent. Two independent runs of
the command above, one from a cool board and one after half an hour of
continuous benchmarking:

| | cool start (38.8 C) | warm start (48.7 C) |
|---|---|---|
| NPU encoder | 174.83 ms | 174.55 ms |
| NPU hardware time | 174.34 ms | 174.51 ms |
| NPU cycles | 175.68 M | 175.87 M |
| NPU busy | 80.0 % | 79.5 % |
| throughput | 4.59 windows/s | 4.56 windows/s |
| frontend (CPU) | 42.71 ms | 44.52 ms |

The 0.15 % spread on the NPU stage is the useful signal here: the accelerator's
timing does not depend on thermal state at this duty cycle. The CPU frontend is
the noisier of the two, which is what you would expect from a scheduler moving
work between the A55 and A76 clusters.

Quote **windows/s**, not the realtime factor of one clip. A window costs the
same whether it holds 2 s of speech or 11 s, so a short test clip understates
capacity — `sample_1.wav` is 6 s, which is why its realtime factor (27.7×) is
far below the full-window rate (50×).

### Where the time goes

Measured on full 11 s windows (a 66 s file, 6 windows, 10 iterations):

| Stage | Time per window | Share |
|---|---|---|
| frontend (STFT + mel + normalise) | 70.4 ms | 28.6 % |
| quantise input to int16 | 0.37 ms | 0.1 % |
| **NPU encoder** | **175.1 ms** | **71.2 %** |
| argmax + CTC collect | 0.12 ms | 0.1 % |

The NPU time is remarkably stable — 174.31 to 175.84 ms across 160 runs. The
frontend is the only variable part, and it scales with how much real speech the
window holds (44 ms for a 6 s clip, 70 ms for a full one).

### The NPU is the ceiling, so run two workers

The NPU takes a fixed 175 ms per window and has a single core (`device_count=1,
core_count=1`). The CPU frontend runs *before* it, serially, so one process
leaves the NPU idle while it computes the STFT. Running two processes overlaps
one's frontend with the other's NPU work:

| Workers | Throughput | Realtime | NPU time per call |
|---|---|---|---|
| 1 | 4.06 windows/s | 44.6× | 175 ms |
| **2** | **5.64 windows/s** | **62.0×** | 303 ms (queued) |
| 3 | 5.66 windows/s | 62.2× | 477 ms (queued) |

Two workers reach 99 % of the theoretical ceiling (1 / 0.175 s = 5.71
windows/s = 62.8×). A third adds nothing: the NPU is already saturated and the
only thing that grows is queueing latency. Reproduce with:

```bash
R=../../../device/tinyvoice_run
for N in 1 2 3; do
  start=$(date +%s.%N)
  for i in $(seq 1 $N); do
    $R tinyvoice_int16.nb device/filterbank.bin device/vocab.txt \
       long_66s.wav --repeat 6 --quiet >/dev/null 2>/dev/null &
  done
  wait
  t=$(echo "$(date +%s.%N) - $start" | bc)
  echo "workers=$N  $(echo "scale=2; $N*36/$t" | bc) windows/s"
done
```

### Confirming the encoder really runs on the NPU

Worth checking rather than assuming, since a silent CPU fallback would look
like "it works, just slowly". Compare wall clock against CPU actually consumed:

```bash
TIMEFORMAT="real %R s   user %U s   sys %S s"
time ../../../device/tinyvoice_run tinyvoice_int16.nb device/filterbank.bin \
     device/vocab.txt sample_1.wav --repeat 100 --quiet >/dev/null 2>/dev/null
```

```
real 21.799 s   user 4.141 s   sys 0.071 s
```

The process burns 4.2 s of CPU across 21.8 s of wall clock -- it is blocked for
81 % of the run. And 4.141 s / 100 = 41 ms per iteration, which is the frontend
time; the 175 ms encoder consumes essentially no CPU. Were it running on the
CPU, user time would track wall clock instead.

Three other things corroborate it:

- the process holds `/dev/vipcore` open (`ls -l /proc/<pid>/fd`)
- the driver enumerates real hardware at startup: `VIP cid=0x1000003b,
  device_count=1, core_count=1`, and that id matches what the `.nb` was
  compiled for
- throughput plateaus at two workers. On an 8-core board a CPU-bound encoder
  would keep scaling; plateauing means one shared serial resource, the NPU's
  single core

### What runs where

| Stage | Runs on | Cost per 11 s window |
|---|---|---|
| WAV decode | CPU | negligible |
| STFT, mel, normalise | CPU | 70 ms |
| quantise to int16 | CPU | 0.4 ms |
| **16-layer Conformer encoder** | **NPU** | **175 ms** |
| argmax + CTC collapse | CPU | 0.1 ms |

The NPU is a VeriSilicon VIP9000 block inside the A733 -- not a TPU, which is
Google hardware and only relevant to training this model, not running it.

### Monitoring the NPU

**btop and htop will not show it.** They cover CPU, memory, disk, network and
NVIDIA/AMD/Intel GPUs; nothing in the distro knows about a VeriSilicon VIP9000.
The `vipcore` driver exposes no utilisation counter in sysfs either, so there is
no file to read for a "% busy" gauge. Three things are available instead.

**1. Live system view.** `device/npu_top.sh` shows temperature, clock and which
processes hold `/dev/vipcore` open:

```bash
~/Documents/zura/TinyVoice/device/npu_top.sh 1
```

```
17:34:24  NPU 54 C  clk 1008 MHz (performance)  CPU 55 C  GPU 53 C  in-use: tinyvoice_run(57209)
```

Note the governor is `performance`: the NPU sits at 1008 MHz permanently and
never scales, so the clock reading tells you nothing about load. `in-use` and
the temperature are the useful columns. Run it as root to see processes
belonging to other users.

**2. Real utilisation, from the hardware itself.** VIPLite exposes a per-network
profiling counter (`VIP_NETWORK_PROP_PROFILING` → `inference_time`,
`total_cycle`), which `tinyvoice_run --repeat` reads and reports:

```
  NPU hardware time  174.34 ms/window  (driver reports)
  NPU cycles         175.68 M/window   (1008 MHz effective)
  NPU busy             80.0% of wall clock
```

This is the number to trust. The driver's own timer (174.34 ms) agrees with the
wall clock measured around `awnn_run()` (174.83 ms) to within half a
millisecond, which is the driver and buffer overhead. 175.68 M cycles at
174.34 ms works out to 1008 MHz, exactly the devfreq clock -- so the counter is
a genuine hardware cycle count, not a software estimate.

"NPU busy 80.0 %" is the utilisation figure a btop-style gauge would show. The
missing 20 % is the CPU frontend, which runs serially ahead of it -- the reason
[two workers](#step-8-benchmark-and-stress-test) push it to ~99 %.

**3. Driver internals, as root.** `/sys/kernel/debug/viplite/` has more, though
most of it is aimed at debugging rather than monitoring:

| Node | Contents |
|---|---|
| `vip_info` | `pid=0x1000003b, date=0x20230518, ver1=0x9000, ver2=0x9202` |
| `vip_freq` | core and PPU clocks; reads back the DVFS percentage while busy |
| `rt_net_profile` | per-layer runtime profiling |
| `mem_profile`, `mem_mapping` | memory pool accounting |
| `pc_value`, `register_rw` | program counter and raw register access |

```bash
sudo cat /sys/kernel/debug/viplite/vip_info
```

Do not read `register_rw` or `pc_value` on a live workload unless you know what
you are poking.

### Headline numbers

| | |
|---|---|
| Peak throughput | **5.6 windows/s = 62× realtime** (2 workers) |
| Single-stream throughput | 4.1 windows/s = 45× realtime |
| Single-stream latency | 245 ms per 11 s window |
| Cold start | ~100 ms to load the 49 MB network, once per process |
| NPU busy (1 worker) | 80.0 % of wall clock, hardware-measured |
| Sustained NPU temperature | 38 °C idle → 60 °C under continuous load, no throttling |

In practical terms: one A7S transcribes roughly **an hour of audio per minute**,
or keeps up with about 60 simultaneous realtime audio streams.

### Thermals

No throttling observed. The NPU sensor rose from 38 °C to 49 °C over several
minutes of continuous inference and the per-call time never drifted — the
spread stayed inside 1.5 ms. Watch it during a longer soak with:

```bash
watch -n1 'for z in /sys/class/thermal/thermal_zone*/; do \
    printf "%s %s\n" "$(cat $z/type)" "$(cat $z/temp)"; done'
```

### If you need more throughput

- **Two worker processes** is the single biggest win and needs no code change.
- **A shorter `--seconds` window** at export time. The 175 ms is dominated by
  the 274-frame attention, which is quadratic in window length — halving the
  window to 5.5 s should more than halve NPU time, at the cost of more chunk
  boundaries.
- **Overlap inside one process.** A producer thread computing the next window's
  mel while the NPU runs the current one gets the two-worker benefit without
  two model copies in memory (~50 MB each).
- **Parallelise the STFT.** It is single-threaded over 1101 frames on a board
  with 8 cores. Only worth doing if you also do the above, since the NPU would
  still cap you at 62×.

---

## Reference

### Platforms

From the SDK's `machinfo` table; the optimise target is derived automatically.

| Platform | NPU | Driver | Hardware PID |
|---|---|---|---|
| `ai985`, `mr527`, `t527` | v2 | v1.13 | `0x10000016` |
| `a733`, `mr536`, `t536`, `t736` | v3 | v2.0 | `0x1000003b` |

**Radxa Cubie A7S → `a733`.** Confirm with
`tr '\0' ' ' < /proc/device-tree/compatible`, which prints
`radxa,cubie-a7s arm,sun60iw2p1 allwinner,sun60i-a733`.

### Scripts

| Script | Runs | Purpose |
|---|---|---|
| `scripts/export_onnx.py` | workstation | checkpoint → ONNX, calibration, metadata |
| `scripts/npu_convert.sh` | workstation → container | ONNX → quantised `.nb` |
| `scripts/npu_inputmeta.py` | container | rewrite inputmeta for tensor inputs |
| `scripts/npu_compare.py` | workstation | float vs quantised, decoded |
| `scripts/npu_infer.py` | workstation | transcribe (ONNX or simulator) |
| `scripts/export_device_assets.py` | workstation | filterbank, vocab, generated header |
| `scripts/check_device_frontend.py` | workstation | diff the C frontend against Python |

### Source layout

| Path | Purpose |
|---|---|
| `conformer/onnx_builder.py` | builds the ONNX graph from NNX parameters |
| `conformer/frontend_np.py` | NumPy log-mel matching the training frontend |
| `conformer/npu_runtime.py` | NumPy-only preprocessing and CTC decode |
| `device/tinyvoice_run.c` | board: WAV → quantise → NPU → CTC → text |
| `device/tinyvoice_frontend.c` | board: STFT, mel, normalisation (port of `frontend_np.py`) |
| `device/host_test.c` | builds the frontend natively for verification |

### Shapes at the default 11 s window

```
mel     (1, 1, 1101, 128)     1101 mel frames × 128 mel bins
mask    (1, 1,  274,   1)     274 encoder frames after 4× subsampling
logits  (1,  274,  36)        36-token vocabulary
```

Inputs are declared 4-D NCHW with a single channel — see
[Troubleshooting](#troubleshooting).

---

## Troubleshooting

### On the board

**`fatal error: math.h: No such file or directory`**
gcc is installed but the libc headers are not: `sudo apt install -y libc6-dev`.

**`fatal error: log/log.h: No such file or directory`**
`awnn_internal.h` includes it from the ai-sdk root, so `-I$(AI_SDK_ROOT)` has to
be on the include path. The Makefile does this; the error means `AI_SDK_ROOT`
is wrong or unset.

**`error while loading shared libraries: libVIPhal.so`**
`libNBGlinker.so` pulls in `libVIPhal.so` from the same directory, and RUNPATH
is *not* inherited when resolving a dependency's own dependencies. The Makefile
passes `-Wl,--disable-new-dtags` so the linker emits RPATH, which is. If you
link by hand instead, set `LD_LIBRARY_PATH` to the viplite lib directory.

**`Cannot read .../machinfo//config.mk`**
`AI_SDK_ROOT` and `AI_SDK_PLATFORM` were not passed to `make`.

**`awnn_create` fails or `VIP cid` differs from the `.nb`**
The binary was built for the wrong chip. Re-run Step 2 with `a733`.

### On the workstation

**`pegasus: error: unrecognized arguments`**
`inputs_outputs.txt` quotes multi-word arguments (`--inputs 'mel mask'`), so it
has to go through `eval`, the way the SDK's own convert scripts do.

**`ValueError: Cannot load file containing pickled data`** during quantisation
ACUITY's NPY provider `np.load()`s the database path directly and handles one
port per database. Each input needs its own single stacked `.npy`, not a text
file listing per-sample arrays.

**`cannot reshape array of size 1 into shape (1,1101,1)`** during quantisation
ACUITY reads dimension 1 as the channel count when applying the inputmeta's
per-channel mean and scale, so a 3-D `(1, T, F)` input makes it treat `T` as
channels. Declaring inputs as 4-D NCHW with one channel avoids it.

**`libvdtproxy.so: cannot open shared object file`** during export
The `gen_nbg` helper ACUITY compiles and runs links against the Vivante
simulator libraries, and pegasus spawns it *without* passing the environment
down — so `LD_LIBRARY_PATH` never reaches it and the libraries must be
registered with `ldconfig`.

**`ImportError: libc10_cuda.so: undefined symbol`** after fixing the above
`cmdtools/common/lib` ships its own `libc10.so`/`libtorch_cpu.so`, which shadow
the toolkit's pip-installed torch — and pegasus imports torch at startup. Only
`libvdtproxy.so` is needed from there, so link exactly that one into a directory
of its own. `npu_convert.sh` already does this.

**`Error in PredictCost() for the op: Softmax`** during simulation
Benign TensorFlow grappler noise from the simulator's backend, not a failure.

---

## Known limitations

**Fixed 11 s window.** The NPU spends 175 ms per window regardless of how much
speech it contains, so a 2 s clip costs the same as a full one. Exporting with a
shorter `--seconds` helps if your utterances are consistently short -- see
[Step 8](#step-8-benchmark-and-stress-test) for the measured cost.

**Chunk boundaries.** Audio longer than the window is split into
non-overlapping chunks and attention cannot cross a boundary, so accuracy dips
around the seams. Overlapping chunks with logit stitching would reduce this.

**Greedy decoding only.** No beam search and no language model. Both would help
word error rate and both belong on the CPU side.

**Single-threaded frontend.** The STFT is a straightforward radix-2 FFT per
frame, costing 70 ms against the NPU's 175 ms. It is not the ceiling, but it
does serialise ahead of the NPU in a single process -- running two workers
recovers the difference.

**Only `int16` is wired through to the device runner.** The other quantisation
types convert and simulate fine, but `export_device_assets.py` refuses anything
else because the C runner assumes int16 fixed-point inputs.
