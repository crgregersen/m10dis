# Out-of-tree kernel module Makefile for the ARCTIC Fan Controller driver.
# Operates on the generated, Quilt-patched source tree in build/.
#
# Typical use:
#   ./scripts/prepare.sh    # generate build/ from upstream/ + patches/
#   make                    # build against the running kernel
#   make KDIR=/path/to/linux build dir
#   make install
#   make clean

obj-m += arctic_fan_controller.o

KDIR ?= /lib/modules/$(shell uname -r)/build
BUILD_DIR := $(CURDIR)/build

.PHONY: all prepare clean install

all: prepare
	$(MAKE) -C $(KDIR) M=$(BUILD_DIR) modules

prepare:
	./scripts/prepare.sh

clean:
	if [ -d "$(BUILD_DIR)" ]; then \
		$(MAKE) -C $(KDIR) M=$(BUILD_DIR) clean || true; \
	fi
	rm -rf $(BUILD_DIR)

install:
	$(MAKE) -C $(KDIR) M=$(BUILD_DIR) INSTALL_MOD_PATH=$(INSTALL_MOD_PATH) modules_install
