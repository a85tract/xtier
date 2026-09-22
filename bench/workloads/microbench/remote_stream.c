// SPDX-License-Identifier: Apache-2.0
/* remote_stream: pointer-chase over a large buffer to force LLC misses to a
 * chosen NUMA node. Used to probe which raw PEBS event codes actually fire. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
int main(int argc, char **argv) {
    size_t gb = argc > 1 ? atoi(argv[1]) : 4;
    int secs  = argc > 2 ? atoi(argv[2]) : 10;
    size_t n = gb * (1UL<<30) / sizeof(uint64_t);
    uint64_t *buf = malloc(n * sizeof(uint64_t));
    if (!buf) { perror("malloc"); return 1; }
    memset(buf, 0, n * sizeof(uint64_t));
    /* random permutation cycle -> dependent chain, no prefetch */
    for (size_t i = 0; i < n; i++) buf[i] = i;
    srand(12345);
    for (size_t i = n - 1; i > 0; i--) { size_t j = rand() % (i+1); uint64_t t = buf[i]; buf[i]=buf[j]; buf[j]=t; }
    /* turn permutation into a cycle-ish chain by index-follow */
    uint64_t idx = 0; volatile uint64_t sink = 0;
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned long iters = 0;
    for (;;) {
        for (int k = 0; k < 1000000; k++) { idx = buf[idx]; sink += idx; }
        iters += 1000000;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (t1.tv_sec - t0.tv_sec >= secs) break;
    }
    printf("iters=%lu sink=%lu\n", iters, (unsigned long)sink);
    return 0;
}
