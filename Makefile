# SPDX-License-Identifier: GPL-2.0
#
# Builds the three pieces of xTier: the executor kernel module, the BPF
# program, and the userspace loader. See README.md for prerequisites.

ifneq ($(KERNELRELEASE),)
# --- Kbuild pass (invoked by the kernel build system) ---
    obj-m := xtier_executor.o
    KBUILD_CFLAGS += -g -O2
else
# --- Normal pass ---

PWD  := $(shell pwd)
SRC  := $(PWD)/src
KDIR ?= /lib/modules/$(shell uname -r)/build

CLANG   ?= clang
CC      ?= gcc
BPFTOOL ?= bpftool

# Which model header gets compiled into the loader. No model ships with this
# repository -- you train one on your own workload (see ml/README.md) and
# point MODEL_HDR at the result.
MODEL_HDR ?= $(PWD)/ml/models/mlp_q8.h

# libbpf. By default we build against the copy in the kernel tree, which is
# guaranteed to match the headers we compile the BPF program against. Set
# LIBBPF_SRC= (empty) to link the distribution's libbpf instead:
#     make LIBBPF_SRC=
LIBBPF_SRC ?= $(KDIR)/tools/lib/bpf

ifeq ($(strip $(LIBBPF_SRC)),)
    LIBBPF_A     :=
    LIBBPF_HDRS  :=
    LIBBPF_STAMP :=
    LIBBPF_CFLAGS :=
    LIBBPF_LIBS  := -lbpf
else
    LIBBPF_A     := $(LIBBPF_SRC)/libbpf.a
    # <bpf/*.h> staged out of the same tree as the .a, rather than taken from
    # the distribution's libbpf-dev, whose version is tied to the distribution
    # kernel and need not match what we link against.
    LIBBPF_HDRS  := $(SRC)/.libbpf-include
    LIBBPF_STAMP := $(LIBBPF_HDRS)/include/bpf/bpf_helpers.h
    LIBBPF_CFLAGS := -I$(LIBBPF_HDRS)/include -I$(LIBBPF_SRC) -I$(LIBBPF_SRC)/include
    LIBBPF_LIBS  := $(LIBBPF_A)
endif

# -mcpu=v3 and -fno-stack-protector are both required: without them the
# program is rejected at load time. Do not drop either.
BPF_CFLAGS := -g -O2 -target bpf -D__TARGET_ARCH_x86 -mcpu=v3 \
              -fno-stack-protector \
              -I$(KDIR)/include \
              -I$(KDIR)/include/uapi \
              -I$(KDIR)/arch/x86/include \
              -I$(KDIR)/arch/x86/include/uapi \
              -I$(KDIR)/arch/x86/include/generated \
              -I$(KDIR)/arch/x86/include/generated/uapi \
              -I$(KDIR)/include/generated \
              $(LIBBPF_CFLAGS) \
              -I$(SRC)

USER_CFLAGS  := -g -O2 -Wall -I$(SRC) -I$(dir $(MODEL_HDR)) $(LIBBPF_CFLAGS) \
                -DXTIER_MODEL_HDR='"$(notdir $(MODEL_HDR))"'
USER_LDFLAGS := $(LIBBPF_LIBS) -lelf -lz -lm

MODULE    := $(SRC)/xtier_executor.ko
BPF_PROG  := $(SRC)/pebs_mlp_kern.o
LOADER    := $(SRC)/page_profiler_user
VMLINUX_H := $(SRC)/vmlinux.h

# A failed recipe must not leave its half-written target behind: a bpftool that
# cannot read this kernel's BTF exits non-zero after creating an empty
# vmlinux.h, which the next make would then accept as up to date.
.DELETE_ON_ERROR:

.PHONY: all module bpf bootstrap-model clean install install-migrate uninstall \
        restart teardown show-stats help

all: module bpf

module:
	$(MAKE) -C $(KDIR) M=$(SRC) modules

bpf: $(BPF_PROG) $(LOADER)

# Generated from the running kernel's BTF, so it is machine-specific and not
# committed. Needs CONFIG_DEBUG_INFO_BTF=y.
$(VMLINUX_H):
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

$(BPF_PROG): $(SRC)/pebs_mlp_kern.c $(SRC)/common_kern.h $(VMLINUX_H) $(LIBBPF_STAMP)
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@
	# Strip debug symbols to stay inside BPF's instruction and size limits.
	llvm-strip -g $@

$(LIBBPF_A):
	$(MAKE) -C $(LIBBPF_SRC)

$(LIBBPF_STAMP):
	$(MAKE) -C $(LIBBPF_SRC) install_headers DESTDIR=$(LIBBPF_HDRS) prefix=

# The loader #includes the model header, so the weights are compiled in
# rather than loaded at runtime. It must therefore be a prerequisite:
# without it, swapping models and re-running make rebuilds nothing and the
# binary silently keeps the previous weights.
$(MODEL_HDR):
	@echo "No model at $(MODEL_HDR)."
	@echo ""
	@echo "Models are not shipped -- train one on your workload:"
	@echo "    python3 ml/train_and_export.py        # writes ml/models/mlp_q8.h"
	@echo ""
	@echo "Then build with that model:"
	@echo "    make MODEL_HDR=\$$PWD/ml/models/<name>.h"
	@echo ""
	@echo "Collecting those features needs the loader, which needs a model."
	@echo "To break that loop, build against placeholder weights first:"
	@echo "    make bootstrap-model"
	@echo "    make MODEL_HDR=\$$PWD/ml/models/mlp_q8_bootstrap.h"
	@echo ""
	@echo "See ml/README.md."
	@false

# Placeholder weights, so the loader builds before any model exists and
# `collect` mode can produce the features one is trained from.
bootstrap-model:
	python3 $(PWD)/ml/bootstrap_model.py $(PWD)/ml/models/mlp_q8_bootstrap.h

$(LOADER): $(SRC)/page_profiler_user.c $(MODEL_HDR) $(BPF_PROG) $(LIBBPF_A) $(LIBBPF_STAMP)
	$(CC) $(USER_CFLAGS) $< -o $@ $(USER_LDFLAGS)

clean:
	$(MAKE) -C $(KDIR) M=$(SRC) clean
	rm -f $(BPF_PROG) $(LOADER) $(VMLINUX_H)
	rm -f $(SRC)/*.o $(SRC)/*.mod.c $(SRC)/*.mod $(SRC)/*.ko \
	      $(SRC)/.*.cmd $(SRC)/Module.symvers $(SRC)/modules.order
	rm -rf $(SRC)/.tmp_versions $(SRC)/.libbpf-include

install: module
	sudo insmod $(MODULE) enable_migration=0 worker_interval_ms=100

install-migrate: module
	sudo insmod $(MODULE) enable_migration=1 worker_interval_ms=100

uninstall:
	@sudo rmmod xtier_executor 2>/dev/null || echo "Module not loaded"

restart: uninstall
	@sudo rm -f /sys/fs/bpf/mprof_region_queue_map /sys/fs/bpf/mprof_stats_map \
	            /sys/fs/bpf/mprof_page_state_map
	@sleep 1
	sudo insmod $(MODULE) enable_migration=1 worker_interval_ms=1 \
	     promote_node=0 demote_node=1
	@echo "Ready. Run: sudo $(LOADER) <pid> <freq> <epoch_ms> evaluate"

teardown:
	@sudo rmmod xtier_executor 2>/dev/null || true
	@sudo rm -f /sys/fs/bpf/mprof_region_queue_map /sys/fs/bpf/mprof_stats_map \
	            /sys/fs/bpf/mprof_page_state_map
	@echo "Clean."

show-stats:
	@sudo dmesg | grep xtier | tail -50
	@sudo bpftool map dump name stats_map 2>/dev/null || echo "BPF stats not available"

help:
	@echo "Targets:"
	@echo "  all              module + bpf (default)"
	@echo "  bootstrap-model  placeholder weights, so the loader builds"
	@echo "  module           build src/xtier_executor.ko"
	@echo "  bpf              build the BPF object and the userspace loader"
	@echo "  install          insmod with migration disabled (observe only)"
	@echo "  install-migrate  insmod with migration enabled"
	@echo "  restart          rmmod, clear pinned maps, insmod for a run"
	@echo "  teardown         rmmod and clear pinned maps"
	@echo "  show-stats       dmesg tail plus a BPF stats map dump"
	@echo "  clean            remove build products"
	@echo ""
	@echo "Variables:"
	@echo "  KDIR=<path>        kernel build tree (default: the running kernel)"
	@echo "  LIBBPF_SRC=<path>  libbpf source; set empty to link -lbpf instead"
	@echo "  MODEL_HDR=<path>   model header to compile in"
	@echo "                     (default: ml/models/mlp_q8.h)"

endif
