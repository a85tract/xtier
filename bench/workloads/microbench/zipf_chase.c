// SPDX-License-Identifier: Apache-2.0
// zipf_chase.c -- zipfian region-granular pointer-chase benchmark for tiering studies.
//
//   gcc -O2 -fopenmp -o zipf_chase zipf_chase.c -lm
//   ./zipf_chase <gb> <theta> <ops_per_thread_millions> <threads> <trials>
//
// Allocates <gb> GB of anonymous memory carved into 2MB blocks. Block hotness
// follows a zipf(theta) distribution over a random (seeded) block permutation:
// hot "objects" are contiguous 2MB blocks, matching how hot application data
// structures (arrays, buffers, histogram bins) are laid out in practice -- and
// matching the 2MB migration granularity of the system under test.
// Each op walks C dependent pointer hops inside one 4KB page of the chosen
// block (pages hold a random 512-entry u64 cycle), so every hop is a
// serialized cache miss: memory latency is exposed by construction.
// Prints one "Trial Time: <s>" line per trial (gapbs-compatible parsing) and
// "Average Time: <s>" at the end. Deterministic for a fixed seed.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <time.h>
#include <omp.h>

#define PG 4096
#define BLOCK (2u<<20)
#define PPB (BLOCK/PG)          // pages per block = 512
#define SLOTS (PG/8)            // u64 slots per page = 512
#define SEQLEN (1u<<22)         // per-thread zipf sequence length

static inline double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec + ts.tv_nsec*1e-9; }
static inline uint64_t xr(uint64_t *s){ uint64_t x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

int main(int argc, char **argv){
    if (argc < 6){ fprintf(stderr,"usage: %s <gb> <theta> <mops_per_thread> <threads> <trials> [scatter=1]\n", argv[0]); return 1; }
    size_t gb = atoll(argv[1]); double theta = atof(argv[2]);
    uint64_t mops = atoll(argv[3]); int nthr = atoi(argv[4]); int trials = atoi(argv[5]);
    size_t bytes = gb<<30; uint32_t nblk = bytes/BLOCK;
    const int CHASE = 16;
    fprintf(stderr,"[init] %zu GB, %u blocks, theta=%.2f, %d threads, %lu Mops/thread, %d trials\n",
            gb, nblk, theta, nthr, (unsigned long)mops, trials);

    uint8_t *mem = mmap(NULL, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE, -1, 0);
    if (mem == MAP_FAILED){ perror("mmap"); return 1; }

    // init: every page gets a seeded random pointer cycle (also faults pages in)
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

    // zipf CDF over blocks, ranks scattered by a seeded permutation
    double *cdf = malloc(nblk*sizeof(double));
    uint32_t *blkperm = malloc(nblk*sizeof(uint32_t));
    { double s=0; for (uint32_t r=0;r<nblk;r++){ s += 1.0/pow(r+1,theta); cdf[r]=s; }
      for (uint32_t r=0;r<nblk;r++) cdf[r]/=s;
      for (uint32_t i=0;i<nblk;i++) blkperm[i]=i;
      if (argc < 7 || atoi(argv[6])) { uint64_t seed=42; for (uint32_t i=nblk-1;i>0;i--){ uint32_t j=xr(&seed)%(i+1); uint32_t t=blkperm[i]; blkperm[i]=blkperm[j]; blkperm[j]=t; } } }

    // per-thread zipf block sequences (binary search over cdf; init-time only)
    uint32_t **seq = malloc(nthr*sizeof(void*));
    #pragma omp parallel num_threads(nthr)
    { int t = omp_get_thread_num();
      uint32_t *s = malloc(SEQLEN*sizeof(uint32_t));
      uint64_t seed = 0xdeadbeef + 7919*t;
      for (uint32_t i=0;i<SEQLEN;i++){
          double u = (double)(xr(&seed)>>11) / 9007199254740992.0;
          uint32_t lo=0, hi=nblk-1;
          while (lo<hi){ uint32_t mid=(lo+hi)/2; if (cdf[mid]<u) lo=mid+1; else hi=mid; }
          s[i] = blkperm[lo];
      }
      seq[t]=s; }
    fprintf(stderr,"[init] sequences ready\n");

    uint64_t ops_per_trial = mops*1000000ULL/trials;
    double total=0;
    for (int tr=0; tr<trials; tr++){
        double ts = now_s();
        volatile uint64_t sink=0;
        #pragma omp parallel num_threads(nthr) reduction(+:sink)
        { int t = omp_get_thread_num();
          uint64_t seed = 0xabcdef + t + 31337*tr, v = 0;
          uint32_t *s = seq[t];
          for (uint64_t i=0;i<ops_per_trial;i++){
              // v feeds the index (always-zero term) => ops are serialized
              uint32_t blk = s[(i + (v&1)) & (SEQLEN-1)];
              uint32_t pgi = xr(&seed) % PPB;
              uint64_t *pg = (uint64_t*)(mem + (size_t)blk*BLOCK + (size_t)pgi*PG);
              for (int c=0;c<CHASE;c++) v = pg[v & (SLOTS-1)];
          }
          sink += v; }
        double dt = now_s()-ts; total += dt;
        printf("Trial Time:          %.5f\n", dt); fflush(stdout);
    }
    printf("Average Time:        %.5f\n", total/trials);
    return 0;
}
