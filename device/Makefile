# Cross-build the on-device recogniser.  Modelled on the SDK's
# examples/lenet/Makefile, so it expects the same environment:
#
#   AI_SDK_ROOT      checkout of the ai-sdk tree (the one in the SDK container)
#   AI_SDK_PLATFORM  target chip: a733 ai985 mr527 mr536 t527 t536 t736
#   CROSS_COMPILE    aarch64 toolchain prefix from your Tina/Longan SDK
#   INSTALL_PREFIX   staging dir (optional, for `make install`)
#
# The container has no cross compiler -- build this inside your Tina/Longan
# SDK, which is where that toolchain lives.
#
#   make AI_SDK_ROOT=/path/to/ai-sdk AI_SDK_PLATFORM=t527 \
#        CROSS_COMPILE=/path/to/toolchain/bin/aarch64-linux-gnu-

TARGET  = tinyvoice_run
AI_SDK_ROOT     ?= ../
AI_SDK_PLATFORM ?= t527
MODEL_DIR       ?= ../artifacts/npu/tinyvoice/device

include $(AI_SDK_ROOT)/machinfo/$(AI_SDK_PLATFORM)/config.mk

CC  = $(CROSS_COMPILE)gcc
EXAMPLES = $(AI_SDK_ROOT)/examples
VIPLITE  = $(AI_SDK_ROOT)/viplite-tina/lib/aarch64-none-linux-gnu/$(NPU_SW_VERSION)

INCLUDE += -I. -I$(MODEL_DIR) -I$(EXAMPLES)/libawnn_viplite -I$(EXAMPLES)/libawutils -I$(VIPLITE)/inc
SRCS    += tinyvoice_run.c tinyvoice_frontend.c $(EXAMPLES)/libawnn_viplite/awnn_lib.c $(EXAMPLES)/libawnn_viplite/awnn_quantize.c
CFLAGS  += -O2 -Wall

# v2.0 boards link the NBG linker/HAL pair; v1.13 boards the older VIPLite pair.
ifeq ("$(NPU_SW_VERSION)", "v2.0")
  LIBS_VIP += -L$(VIPLITE) -lNBGlinker -lVIPhal
  CFLAGS   += -DNPU_SW_VERSION=2
else
  LIBS_VIP += -L$(VIPLITE) -lVIPlite -lVIPuser
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
