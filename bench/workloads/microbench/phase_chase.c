// SPDX-License-Identifier: Apache-2.0
// phase_chase.c -- segmented phase-change microbenchmark for tiering studies.
//
//   gcc -O2 -fopenmp -o phase_chase phase_chase.c -lm
//   ./phase_chase <gb> <seg_gb> <mops_per_thread> <threads> <trials> <phases>
//
// Allocates <gb> GB of anonymous memory divided into disjoint segments of
// <seg_gb> GB. At any time exactly ONE segment is HOT: every op is a chain of
// 16 dependent pointer loads at a uniformly random page inside the active
// segment. Every trials/phases trials the active segment advances round-robin
// (PHASE_MARKER phaseN_start epoch_unix=... printed at each boundary).
//
// The point: hotness is binary and uniform (no ranking subtlety) and the hot
// set fits in DRAM, so the IDEAL tiering behavior is fully known -- promote
// the active segment once, go quiescent, repeat at each boundary. Cumulative
// migrations should be an equal-step staircase; per-trial latency should
// spike at each boundary and recover as fast as the system adapts.
// Deterministic for fixed args.
//
// adapt10 addition: a sidecar thread prints "THROUGHPUT ts=<unix> ops=<N>"
// once a second (N = cumulative ops summed over threads), giving a 1 Hz
// windowed-throughput signal at phase boundaries -- much finer than the
// ~25 s Trial Time lines, which are kept unchanged for continuity.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>

#define PG 4096
#define SLOTS (PG/8)

static inline double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec + ts.tv_nsec*1e-9; }
static inline double unix_s(void){ struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts); return ts.tv_sec + ts.tv_nsec*1e-9; }
static inline uint64_t xr(uint64_t *s){ uint64_t x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

// 1 Hz throughput monitor (adapt10): per-thread cumulative op counters on
// private cache lines, summed and printed once a second by a sidecar thread.
// Purely additive -- no change to the access pattern or trial semantics; the
// per-op cost is one store to an exclusive line vs 16 dependent DRAM loads.
#define MAXTHR 256
static struct { volatile uint64_t v; char pad[56]; } g_ops[MAXTHR];
static volatile int g_nthr = 0, g_mon_done = 0;
static void *throughput_monitor(void *arg){
    (void)arg;
    while (!g_mon_done){
        struct timespec r = {1, 0};
        nanosleep(&r, NULL);
        uint64_t s = 0;
        for (int i = 0; i < g_nthr; i++) s += g_ops[i].v;
        printf("THROUGHPUT ts=%.3f ops=%llu\n", unix_s(), (unsigned long long)s);
        fflush(stdout);
    }
    return NULL;
}

int main(int argc, char **argv){
    if (argc < 7){ fprintf(stderr,"usage: %s <gb> <seg_gb> <mops_per_thread> <threads> <trials> <phases>\n", argv[0]); return 1; }
    size_t gb = atoll(argv[1]), seg_gb = atoll(argv[2]);
    uint64_t mops = atoll(argv[3]); int nthr = atoi(argv[4]);
    int trials = atoi(argv[5]), phases = atoi(argv[6]);
    size_t bytes = gb<<30, seg_bytes = seg_gb<<30;
    int nseg = bytes/seg_bytes;
    size_t seg_pages = seg_bytes/PG;
    fprintf(stderr,"[init] %zuGB, %d segments x %zuGB, %d threads, %lu Mops/thr, %d trials, %d phases\n",
            gb, nseg, seg_gb, nthr, (unsigned long)mops, trials, phases);

    uint8_t *mem = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (mem == MAP_FAILED){ perror("mmap"); return 1; }
    double t0 = now_s();
    #pragma omp parallel for num_threads(nthr) schedule(static)
    for (size_t p = 0; p < bytes/PG; p++){
        uint64_t *pg = (uint64_t*)(mem + p*PG);
        uint16_t perm[SLOTS]; for (int i=0;i<SLOTS;i++) perm[i]=i;
        uint64_t seed = 0x9e3779b97f4a7c15ULL ^ p;
        for (int i=SLOTS-1;i>0;i--){ int j = xr(&seed)%(i+1); uint16_t t=perm[i]; perm[i]=perm[j]; perm[j]=t; }
        for (int i=0;i<SLOTS;i++) pg[perm[i]] = perm[(i+1)%SLOTS];
    }
    fprintf(stderr,"[init] faulted+cycled in %.1fs\n", now_s()-t0);

    uint64_t ops_per_trial = mops*1000000ULL/trials;
    int tpp_ = trials/phases;
    const int CHASE = 16;
    double total = 0;
    g_nthr = nthr < MAXTHR ? nthr : MAXTHR;
    pthread_t mon;
    int have_mon = (pthread_create(&mon, NULL, throughput_monitor, NULL) == 0);
    printf("THROUGHPUT ts=%.3f ops=0\n", unix_s());
    printf("PHASE_MARKER phase0_start epoch_unix=%.3f\n", unix_s()); fflush(stdout);
    for (int tr=0; tr<trials; tr++){
        int seg = (tr / tpp_) % nseg;
        if (tr > 0 && tr % tpp_ == 0){
            printf("PHASE_MARKER phase%d_start epoch_unix=%.3f\n", tr/tpp_, unix_s()); fflush(stdout);
        }
        double ts = now_s();
        volatile uint64_t sink = 0;
        #pragma omp parallel num_threads(nthr) reduction(+:sink)
        { int t = omp_get_thread_num();
          uint64_t seed = 0xabcdef + t + 31337*tr, v = 0;
          for (uint64_t i=0;i<ops_per_trial;i++){
              size_t pgi = xr(&seed) % seg_pages;
              uint64_t *pg = (uint64_t*)(mem + (size_t)seg*seg_bytes + pgi*PG + ((v&0)*8));
              for (int c=0;c<CHASE;c++) v = pg[v & (SLOTS-1)];
              if (t < MAXTHR) g_ops[t].v++;   // cumulative, read by the 1Hz monitor
          }
          sink += v; }
        double dt = now_s()-ts; total += dt;
        printf("Trial Time:          %.5f\n", dt); fflush(stdout);
    }
    if (have_mon){ g_mon_done = 1; pthread_join(mon, NULL); }
    printf("Average Time:        %.5f\n", total/trials);
    return 0;
}
