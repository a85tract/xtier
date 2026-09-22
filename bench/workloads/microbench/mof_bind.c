// SPDX-License-Identifier: Apache-2.0
/* mof_bind <node> <cmd...>
 *
 * Pre-execs a memory policy of MPOL_BIND on the given node WITH
 * MPOL_F_NUMA_BALANCING (which the kernel ORs MPOL_F_MOF into per
 * mm/mempolicy.c::sanitize_mpol_flags). Then execs the command.
 *
 * Why: numactl --membind=N sets MPOL_BIND but does NOT set
 * MPOL_F_NUMA_BALANCING, so MOF stays off, so task_numa_work skips
 * every VMA with NUMAB_SKIP_UNSUITABLE, so AN/TPP do nothing.
 * This wrapper forces MOF on so the scanner works.
 *
 * Build: gcc -O2 -o mof_bind mof_bind.c
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>

#ifndef MPOL_F_NUMA_BALANCING
#define MPOL_F_NUMA_BALANCING (1 << 13)
#endif

static long set_mempolicy_(int mode, const unsigned long *mask, unsigned long maxnode) {
    return syscall(SYS_set_mempolicy, mode, mask, maxnode);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <node> <cmd> [args...]\n", argv[0]);
        return 2;
    }
    int node = atoi(argv[1]);
    if (node < 0 || node > 63) {
        fprintf(stderr, "node must be 0..63\n");
        return 2;
    }

    unsigned long mask = 1UL << node;
    /* MPOL_PREFERRED_MANY (allows migration off the preferred node -- vs
     * MPOL_BIND which is a hard restriction that blocks promotion).
     * MPOL_F_NUMA_BALANCING tells the kernel to OR MPOL_F_MOF in
     * (per sanitize_mpol_flags), so the numa_balancing scanner won't
     * skip our VMAs as NUMAB_SKIP_UNSUITABLE. */
    int mode = MPOL_PREFERRED_MANY | MPOL_F_NUMA_BALANCING;

    if (set_mempolicy_(mode, &mask, 64) != 0) {
        fprintf(stderr, "set_mempolicy(MPOL_BIND|MPOL_F_NUMA_BALANCING, mask=0x%lx) failed: %s\n",
                mask, strerror(errno));
        return 1;
    }

    /* Optional sanity print */
    if (getenv("MOF_BIND_VERBOSE"))
        fprintf(stderr, "[mof_bind] MPOL_BIND on node %d with MOF set\n", node);

    execvp(argv[2], &argv[2]);
    fprintf(stderr, "execvp(%s) failed: %s\n", argv[2], strerror(errno));
    return 127;
}
