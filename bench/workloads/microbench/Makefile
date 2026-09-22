# SPDX-License-Identifier: Apache-2.0
#
# Microbenchmarks with known access patterns, used to exercise specific
# behaviours the real workloads only produce incidentally.
#
# NUMA calls go through syscall(2) directly, so libnuma is not needed. The
# chase workloads use OpenMP.

CC      ?= gcc
CFLAGS  ?= -O2 -Wall -g
CFLAGS  += -fopenmp
LDLIBS  := -lm -lpthread -fopenmp

BINS := phase_chase phase_zipf zipf_chase remote_stream placement_sampler mof_bind

.PHONY: all clean
all: $(BINS)

%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(BINS)
