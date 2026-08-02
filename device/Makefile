# Build the on-device recogniser.
#
# The Radxa boards run a full aarch64 Debian with gcc, so this builds natively
# on the board -- no cross toolchain needed.  To cross-compile instead, set
# CROSS_COMPILE to your toolchain prefix.
#
#   make AI_SDK_ROOT=/home/radxa/Documents/zura/ai-sdk \
#        AI_SDK_PLATFORM=a733 \
#        MODEL_DIR=/home/radxa/Documents/zura/TinyVoice/artifacts/npu/tinyvoice/device
#
#   AI_SDK_ROOT      the ai-sdk checkout (supplies the VIPLite libraries)
#   AI_SDK_PLATFORM  target chip: a733 ai985 mr527 mr536 t527 t536 t736
#   MODEL_DIR        directory holding the generated tinyvoice_model.h
#   CROSS_COMPILE    toolchain prefix; empty for a native build

TARGET = tinyvoice_run

AI_SDK_ROOT     ?=
AI_SDK_PLATFORM ?=
MODEL_DIR       ?= ../artifacts/npu/tinyvoice/device

CONFIG = $(AI_SDK_ROOT)/machinfo/$(AI_SDK_PLATFORM)/config.mk
-include $(CONFIG)

# Only the build needs the SDK; `make clean` should work without it.
ifeq ($(filter clean,$(MAKECMDGOALS)),)
  ifndef NPU_SW_VERSION
    $(error Cannot read $(CONFIG). Set AI_SDK_ROOT to your ai-sdk checkout and \
AI_SDK_PLATFORM to your chip, e.g. AI_SDK_ROOT=$$HOME/Documents/zura/ai-sdk AI_SDK_PLATFORM=a733)
  endif
endif

CC       = $(CROSS_COMPILE)gcc
EXAMPLES = $(AI_SDK_ROOT)/examples
VIPLITE  = $(AI_SDK_ROOT)/viplite-tina/lib/aarch64-none-linux-gnu/$(NPU_SW_VERSION)

# -I$(AI_SDK_ROOT) is what resolves <log/log.h>, which awnn_internal.h pulls in.
INCLUDE += -I. -I$(MODEL_DIR) -I$(AI_SDK_ROOT) \
           -I$(EXAMPLES)/libawnn_viplite -I$(EXAMPLES)/libawutils -I$(VIPLITE)/inc
SRCS    += tinyvoice_run.c tinyvoice_frontend.c \
           $(EXAMPLES)/libawnn_viplite/awnn_lib.c $(EXAMPLES)/libawnn_viplite/awnn_quantize.c
CFLAGS  += -O2 -Wall

# v2.0 boards link the NBG linker/HAL pair; v1.13 boards the older VIPLite pair.
# The rpath lets the binary find them without LD_LIBRARY_PATH -- they live in
# the SDK checkout, not /usr/lib.  --disable-new-dtags emits RPATH rather than
# RUNPATH deliberately: RUNPATH is not inherited when resolving a dependency's
# own dependencies, and libNBGlinker.so pulls in libVIPhal.so beside it.
RPATH = -Wl,-rpath,$(abspath $(VIPLITE)) -Wl,--disable-new-dtags
ifeq ("$(NPU_SW_VERSION)", "v2.0")
  LIBS_VIP += -L$(VIPLITE) $(RPATH) -lNBGlinker -lVIPhal
  CFLAGS   += -DNPU_SW_VERSION=2
else
  LIBS_VIP += -L$(VIPLITE) $(RPATH) -lVIPlite -lVIPuser
  CFLAGS   += -DNPU_SW_VERSION=1
endif

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(INCLUDE) $(LIBS) $(LIBS_VIP) -lm -lpthread

all: $(TARGET)

install: $(TARGET)
	-@mkdir -p $(INSTALL_PREFIX)/usr/bin
	@cp $(TARGET) $(INSTALL_PREFIX)/usr/bin

clean:
	rm -rf $(TARGET) *.o *~

.PHONY: all install clean
