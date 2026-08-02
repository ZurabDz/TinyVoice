#!/usr/bin/env bash
#
# Convert the exported ONNX encoder into an Allwinner NPU network binary (.nb)
# using the ACUITY toolkit inside the SDK container.
#
#   scripts/export_onnx.py            ->  artifacts/npu/<name>/<name>.onnx
#   scripts/npu_convert.sh            ->  artifacts/npu/<name>/<name>_<qtype>.nb
#
# Usage:
#   scripts/npu_convert.sh <container> <platform> [qtype] [model-dir]
#
#   container : docker container id/name running the ai-sdk image
#   platform  : target chip -- a733 ai985 mr527 mr536 t527 t536 t736
#   qtype     : int16 (default) | bf16 | pcq | uint8
#   model-dir : exported directory (default artifacts/npu/tinyvoice)
#
# Environment:
#   RESUME=1     reuse the container-side workspace and re-run only the export
#                step (import + quantisation are the slow parts)
#   PACK_FLAG    --pack-nbg-unify (default) targets the unified galcore driver;
#                boards running the lightweight VIPLite driver want
#                --pack-nbg-viplite instead
#
# On quantisation type: this is a 16-layer Conformer feeding a CTC head, and
# per-tensor uint8 is usually too coarse for it -- attention scores and the
# residual stream both have wide dynamic range.  int16 (dynamic fixed point) is
# the safe default; bf16 is closest to the float graph where the target
# supports it.  Step 4 dumps float and quantised logits so the accuracy cost is
# measured rather than assumed -- see scripts/npu_compare.py.

set -euo pipefail

CONTAINER="${1:?container id required}"
PLATFORM="${2:?platform required (a733 ai985 mr527 mr536 t527 t536 t736)}"
QTYPE="${3:-int16}"
MODEL_DIR="${4:-artifacts/npu/tinyvoice}"

NAME="$(basename "$MODEL_DIR")"
SDK=/home/ai-sdk
REMOTE="$SDK/models/$NAME"
ACUITY=/usr/local/acuity_command_line_tools
VIV_SDK=/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools/vsimulator
PACK_FLAG="${PACK_FLAG:---pack-nbg-unify}"

case "$QTYPE" in
  uint8) QUANTIZER=asymmetric_affine;           QDTYPE=uint8;     POSTFIX=uint8 ;;
  int16) QUANTIZER=dynamic_fixed_point;         QDTYPE=int16;     POSTFIX=int16 ;;
  pcq)   QUANTIZER=perchannel_symmetric_affine; QDTYPE=int8;      POSTFIX=pcq   ;;
  bf16)  QUANTIZER=qbfloat16;                   QDTYPE=qbfloat16; POSTFIX=bf16  ;;
  *) echo "unknown qtype '$QTYPE' (uint8 / int16 / pcq / bf16)" >&2; exit 1 ;;
esac

[ -f "$MODEL_DIR/$NAME.onnx" ] || {
  echo "missing $MODEL_DIR/$NAME.onnx -- run scripts/export_onnx.py first" >&2; exit 1; }

# The per-chip NPU generation lives in the SDK's machinfo table; deriving the
# optimize target from it keeps this in step with whatever the SDK supports.
NPU_VERSION="$(docker exec "$CONTAINER" sh -c \
  "sed -n 's/^ *NPU_VERSION *= *//p' $SDK/machinfo/$PLATFORM/config.mk" 2>/dev/null | tr -d '[:space:]')"
case "$NPU_VERSION" in
  v1) OPTIMIZE=VIP9000PICO_PID0XEE ;;
  v2) OPTIMIZE=VIP9000NANOSI_PLUS_PID0X10000016 ;;
  v3) OPTIMIZE=VIP9000NANODI_PLUS_PID0X1000003B ;;
  *) echo "unknown platform '$PLATFORM' (no $SDK/machinfo/$PLATFORM/config.mk)" >&2; exit 1 ;;
esac

echo "=== $NAME -> $PLATFORM (NPU $NPU_VERSION, $OPTIMIZE), quantised as $QTYPE ==="

if [ "${RESUME:-0}" != "1" ]; then
  docker exec "$CONTAINER" rm -rf "$REMOTE"
  docker exec "$CONTAINER" mkdir -p "$SDK/models"
  docker cp "$MODEL_DIR" "$CONTAINER:$REMOTE"
  docker cp "$(dirname "$0")/npu_inputmeta.py" "$CONTAINER:$REMOTE/npu_inputmeta.py"

  # No LD_LIBRARY_PATH here on purpose: the Vivante simulator libraries shadow
  # the C++ runtime that the toolkit's bundled torch links against, and the
  # ONNX importer imports torch at module load.
  docker exec -e ACUITY_PATH="$ACUITY" "$CONTAINER" bash -eu -c "
cd '$REMOTE'
PEGASUS='python3 $ACUITY/pegasus.py'

echo '--- 1/5 import onnx ---'
# inputs_outputs.txt quotes multi-word arguments (--inputs 'mel mask'), so it
# has to go through eval the way the SDK's own convert scripts do.
eval \"\$PEGASUS import onnx \
  --model '$NAME.onnx' \
  --output-model '$NAME.json' \
  --output-data '$NAME.data' \
  \$(cat inputs_outputs.txt)\"

echo '--- 2/5 input meta ---'
\$PEGASUS generate inputmeta --model '$NAME.json' \
  --separated-database --input-meta-output '${NAME}_inputmeta.yml'
python3 npu_inputmeta.py '${NAME}_inputmeta.yml' mel=calib_mel.npy mask=calib_mask.npy

echo '--- 3/5 quantize ($QTYPE) ---'
\$PEGASUS quantize \
  --model '$NAME.json' \
  --model-data '$NAME.data' \
  --device CPU \
  --with-input-meta '${NAME}_inputmeta.yml' \
  --rebuild \
  --model-quantize '${NAME}_${POSTFIX}.quantize' \
  --quantizer $QUANTIZER \
  --qtype $QDTYPE

echo '--- 4/5 simulate float vs $QTYPE ---'
\$PEGASUS inference --model '$NAME.json' --model-data '$NAME.data' \
  --dtype float32 --device CPU --iterations 1 \
  --with-input-meta '${NAME}_inputmeta.yml' --output-dir ./inf/float
\$PEGASUS inference --model '$NAME.json' --model-data '$NAME.data' \
  --dtype quantized --model-quantize '${NAME}_${POSTFIX}.quantize' \
  --device CPU --iterations 1 \
  --with-input-meta '${NAME}_inputmeta.yml' --output-dir ./inf/$POSTFIX
"
fi

# The export step compiles a gen_nbg helper and runs it, and that helper links
# against the Vivante simulator libraries.  Two wrinkles:
#
#   * pegasus spawns gen_nbg without passing its environment down, so
#     LD_LIBRARY_PATH on the docker exec does not reach it -- the libraries have
#     to be registered with ldconfig instead.
#   * cmdtools/common cannot be registered wholesale: it ships its own
#     libc10.so/libtorch_cpu.so which shadow the toolkit's pip-installed torch,
#     and pegasus imports torch at startup.  gen_nbg needs exactly one library
#     from there (libvdtproxy.so), so link just that into a directory of its own.
NBG_LIBS=/opt/nbg-libs
docker exec "$CONTAINER" sh -c "
  mkdir -p $NBG_LIBS
  ln -sf $(dirname "$VIV_SDK")/common/lib/libvdtproxy.so $NBG_LIBS/
  printf '%s\n%s\n' '$VIV_SDK/lib' '$NBG_LIBS' > /etc/ld.so.conf.d/vivante-nbg.conf
  ldconfig 2>/dev/null || true
"

docker exec -e ACUITY_PATH="$ACUITY" -e VIV_SDK="$VIV_SDK" \
  -e VSIMULATOR_SHADER_CORE_COUNT=1 -e VSIMULATOR_CONFIG="$OPTIMIZE" \
  "$CONTAINER" bash -eu -c "
cd '$REMOTE'
echo '--- 5/5 export network binary ---'
rm -rf './wksp/${NAME}_${POSTFIX}'
python3 $ACUITY/pegasus.py export ovxlib \
  --model '$NAME.json' \
  --model-data '$NAME.data' \
  --model-quantize '${NAME}_${POSTFIX}.quantize' \
  --dtype quantized \
  --with-input-meta '${NAME}_inputmeta.yml' \
  --optimize '$OPTIMIZE' \
  --viv-sdk '$VIV_SDK' \
  $PACK_FLAG \
  --target-ide-project linux64 \
  --output-path './wksp/${NAME}_${POSTFIX}/${NAME}_${POSTFIX}'

find ./wksp -name 'network_binary.nb' -exec cp {} './${NAME}_${POSTFIX}.nb' \;
# nbg_meta.json states the quantisation the .nb actually expects at its inputs
# and produces at its output -- the device runner needs those scales.
find ./wksp -name 'nbg_meta.json' -exec cp {} './${NAME}_${POSTFIX}_nbg_meta.json' \;
"

for artefact in "${NAME}_${POSTFIX}.nb" "${NAME}_${POSTFIX}.quantize" \
                "${NAME}_${POSTFIX}_nbg_meta.json" "${NAME}_inputmeta.yml" inf; do
  docker cp "$CONTAINER:$REMOTE/$artefact" "$MODEL_DIR/" 2>/dev/null || true
done

if [ -f "$MODEL_DIR/${NAME}_${POSTFIX}.nb" ]; then
  echo
  echo "network binary : $MODEL_DIR/${NAME}_${POSTFIX}.nb"
  echo "size           : $(du -h "$MODEL_DIR/${NAME}_${POSTFIX}.nb" | cut -f1)"
  echo "generated C    : $REMOTE/wksp/${NAME}_${POSTFIX}  (in the container)"
  echo
  echo "Measure the quantisation cost:"
  echo "  python scripts/npu_compare.py $MODEL_DIR"
else
  echo "NBG packing did not produce a .nb -- see the log above" >&2
  exit 1
fi
