// SPDX-License-Identifier: Apache-2.0
// phase_zipf.c -- phase_chase with SKEWED (zipfian) intra-segment hotness.
//
//   gcc -O2 -fopenmp -o phase_zipf phase_zipf.c -lm
//   ./phase_zipf <gb> <seg_gb> <mops_per_thread> <threads> <trials> <phases> [theta]
//
// This is exp/scripts/phase_chase.c with exactly ONE change: the intra-segment
// access distribution. Everything that makes the two comparable is preserved
// verbatim -- the segmented arena, the round-robin hot-segment relocation at
// each phase boundary, the PHASE_MARKER lines, the 1 Hz THROUGHPUT sidecar,
// the 16-hop dependent pointer chase, the deterministic seeding, the
// "Trial Time:" / "Average Time:" output contract.
//
// WHY (see Findings 42/43): phase_chase's hotness is BINARY and UNIFORM. Every
// page of the active segment is equally hot, so a frequency-ranking system has
// nothing to separate -- HybridTier's TinyLFU sketch saturates and its cold
// sweep finds no victims ("pages_demoted 0 ... go to monitor mode"), and
// Memtis's __adjust_active_threshold() has no histogram spread to threshold on.
// Both flatline at ~20-24% against a 21% random-placement floor. That result is
// open to the objection that the workload structurally cripples the baselines.
// phase_zipf removes the objection: hotness is now rankable.
//
// THE DISTRIBUTION (lifted from zipf_chase.c, unchanged in form so the two
// zipf workloads stay directly comparable):
//   * the active segment is carved into 2 MB blocks (the migration granularity
//     of the system under test, and the granularity at which real hot data
//     structures are laid out);
//   * block RANK follows zipf(theta), theta = 0.99 by default (the
//     zipf_chase / YCSB convention);
//   * rank -> block goes through a per-segment seeded permutation, so the hot
//     blocks are SCATTERED through the segment. A contiguous hot range would be
//     unrealistically prefetch- and THP-friendly and would let a system win by
//     accident of layout rather than by ranking.
//   * one rank sequence per thread is drawn once at init and reused across
//     phases; only the permutation changes with the segment, so each phase
//     presents a fresh, differently-scattered hot set with identical skew.
//
// SIZING, which is the point of the experiment: run it with a segment LARGER
// than the fast tier (36 GB arena / 12 GB segments at a 7.9 GB node 0 => hot
// set ~1.5x DRAM). Then no system can simply promote the whole active segment;
// it must choose WHICH pages, which is exactly the capability under test. Peak
// achievable segment coverage is node0/seg_gb, not 100%.
//
// PZ_HOTSET_OUT=<path> dumps the hot-decile block map (one "seg byte_offset"
// line per top-10%-by-rank block, for every segment) so an external sampler can
// measure hot-set residency rather than whole-segment coverage.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <time.h>
#include <omp.h>
#include <pthread.h>

#define PG 4096
#define SLOTS (PG/8)
#define BLOCK (2u<<20)          // 2 MB hot-object granularity (zipf_chase)
#define PPB (BLOCK/PG)          // pages per block = 512
#define SEQLEN (1u<<22)         // per-thread zipf sequence length (zipf_chase)

static inline double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec + ts.tv_nsec*1e-9; }
static inline double unix_s(void){ struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts); return ts.tv_sec + ts.tv_nsec*1e-9; }
static inline uint64_t xr(uint64_t *s){ uint64_t x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

// 1 Hz throughput monitor -- identical to phase_chase.c.
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
    if (argc < 7){ fprintf(stderr,"usage: %s <gb> <seg_gb> <mops_per_thread> <threads> <trials> <phases> [theta]\n", argv[0]); return 1; }
    size_t gb = atoll(argv[1]), seg_gb = atoll(argv[2]);
    uint64_t mops = atoll(argv[3]); int nthr = atoi(argv[4]);
    int trials = atoi(argv[5]), phases = atoi(argv[6]);
    double theta = (argc > 7) ? atof(argv[7]) : 0.99;
    size_t bytes = gb<<30, seg_bytes = seg_gb<<30;
    int nseg = bytes/seg_bytes;
    uint32_t nblk = seg_bytes/BLOCK;          // blocks PER SEGMENT
    fprintf(stderr,"[init] %zuGB, %d segments x %zuGB, %u blocks/seg, theta=%.2f, %d threads, %lu Mops/thr, %d trials, %d phases\n",
            gb, nseg, seg_gb, nblk, theta, nthr, (unsigned long)mops, trials, phases);

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

    // zipf CDF over block RANKS (zipf_chase.c lines 58-64, verbatim in form).
    double *cdf = malloc((size_t)nblk*sizeof(double));
    { double s=0; for (uint32_t r=0;r<nblk;r++){ s += 1.0/pow(r+1,theta); cdf[r]=s; }
      for (uint32_t r=0;r<nblk;r++) cdf[r]/=s; }

    // Per-SEGMENT rank->block permutation: same skew in every phase, different
    // (seeded, reproducible) scatter, so a relocation genuinely moves the hot
    // pages rather than re-exposing the same ones.
    uint32_t **segperm = malloc(nseg*sizeof(void*));
    for (int sgi=0; sgi<nseg; sgi++){
        uint32_t *bp = malloc((size_t)nblk*sizeof(uint32_t));
        for (uint32_t i=0;i<nblk;i++) bp[i]=i;
        uint64_t seed = 42 + 1000003ULL*(uint64_t)sgi;
        for (uint32_t i=nblk-1;i>0;i--){ uint32_t j=xr(&seed)%(i+1); uint32_t t=bp[i]; bp[i]=bp[j]; bp[j]=t; }
        segperm[sgi]=bp;
    }

    // Hot-decile map for the external placement sampler (optional).
    const char *hs = getenv("PZ_HOTSET_OUT");
    if (hs && *hs){
        FILE *fh = fopen(hs, "w");
        if (fh){
            uint32_t top = nblk/10 ? nblk/10 : 1;
            for (int sgi=0; sgi<nseg; sgi++)
                for (uint32_t r=0; r<top; r++)
                    fprintf(fh, "%d %llu\n", sgi, (unsigned long long)segperm[sgi][r]*BLOCK);
            fclose(fh);
            fprintf(stderr,"[init] hotset map -> %s (%u blocks/seg = top decile by rank)\n", hs, nblk/10);
        }
    }

    // Per-thread zipf RANK sequences (drawn once; the permutation supplies the
    // phase-dependent scatter at access time).
    uint32_t **seq = malloc(nthr*sizeof(void*));
    #pragma omp parallel num_threads(nthr)
    { int t = omp_get_thread_num();
      uint32_t *s = malloc(SEQLEN*sizeof(uint32_t));
      uint64_t seed = 0xdeadbeef + 7919*t;
      for (uint32_t i=0;i<SEQLEN;i++){
          double u = (double)(xr(&seed)>>11) / 9007199254740992.0;
          uint32_t lo=0, hi=nblk-1;
          while (lo<hi){ uint32_t mid=(lo+hi)/2; if (cdf[mid]<u) lo=mid+1; else hi=mid; }
          s[i] = lo;                       // RANK, not block
      }
      seq[t]=s; }
    fprintf(stderr,"[init] sequences ready\n");

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
        uint32_t *bp = segperm[seg];
        #pragma omp parallel num_threads(nthr) reduction(+:sink)
        { int t = omp_get_thread_num();
          uint64_t seed = 0xabcdef + t + 31337*tr, v = 0;
          uint32_t *s = seq[t];
          for (uint64_t i=0;i<ops_per_trial;i++){
              // v feeds the index through an always-zero/one term => serialized
              uint32_t blk = bp[ s[(i + (v&1)) & (SEQLEN-1)] ];
              uint32_t pgi = xr(&seed) % PPB;
              uint64_t *pg = (uint64_t*)(mem + (size_t)seg*seg_bytes + (size_t)blk*BLOCK + (size_t)pgi*PG);
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
