// SPDX-License-Identifier: Apache-2.0
// placement_sampler.c -- ground-truth page-placement time series for tiering
// adaptation studies (adapt9).
//
//   gcc -O2 -o placement_sampler placement_sampler.c
//   sudo ./placement_sampler <pid> <seg_gb> <interval_ms> <samples_per_seg> <out_csv> [hotset_file]
//
// Locates the target's largest anonymous mapping (the phase_chase arena),
// divides it into <seg_gb> segments, and every <interval_ms> queries the NUMA
// node of <samples_per_seg> uniformly random pages per segment using
// move_pages(2) in query mode (nodes=NULL). Uniform hotness within a segment
// makes the random sample an unbiased estimator of per-segment DRAM residency
// (+/-~2pp at 768 samples).
//
// Passive by construction: no PMU use (per-task PEBS stays untouched), no
// page-data access (no A-bit pollution beyond a PT walk), and immune to the
// pgmigrate_success compaction pollution -- it reads where
// pages ARE, not how they got there. Works identically under any placement
// system, or none. Runs as root (cross-process move_pages
// query needs PTRACE_MODE_READ).
//
// CSV: ts_unix,seg0_dram_frac,...,segN_dram_frac,seg0_valid,...,segN_valid
//      [,hot0_dram_ratio,...,hot0_valid,...]   <- only with a hotset_file
//
// UNITS: every *_dram_frac and *_dram_ratio column is a FRACTION in [0,1]
// (or -1 when no page in that region was resident). It is NOT a percentage.
// These columns were originally emitted as `hot<N>_dram_pct`, which caused a
// real misreading: 0.9689 was read as "1%" instead of 97%. If you have a CSV
// with `_dram_pct` headers, the VALUES are still fractions -- only the header
// was wrong; multiply by 100 to get percent. Do not rename these to
// `*_dram_frac`: adapt9_figure.placement() infers the segment count by
// counting columns whose name ends in `_dram_frac`.
// Exits when the target dies.
//
// hotset_file (adapt12 / phase_zipf): optional. Text, one "<seg> <byte_offset>"
// line per 2MB HOT block, as written by phase_zipf's PZ_HOTSET_OUT. Whole-
// segment coverage is the right metric only when hotness is uniform inside the
// segment; under a zipf segment it is not, and a segment larger than the fast
// tier CANNOT reach 100% anyway. With this file the sampler additionally draws
// uniformly from the hot blocks of the CURRENTLY-hottest segment set and
// reports their DRAM residency, which is what a ranker is actually judged on.
// The extra columns go at the END so the existing positional parser
// (adapt9_figure.placement) is unaffected.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/syscall.h>

#define PG 4096UL
#define MAX_SEG 16

static inline double unix_s(void){ struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts); return ts.tv_sec + ts.tv_nsec*1e-9; }
static inline uint64_t xr(uint64_t *s){ uint64_t x=*s; x^=x<<13; x^=x>>7; x^=x<<17; return *s=x; }

static long q_move_pages(pid_t pid, unsigned long n, void **pages, int *status){
    return syscall(SYS_move_pages, pid, n, pages, NULL, status, 0);
}

// Largest anonymous mapping of pid. Returns size 0 if none found.
static void find_arena(pid_t pid, uint64_t *base, uint64_t *size){
    char path[64], line[512];
    snprintf(path, sizeof path, "/proc/%d/maps", pid);
    FILE *f = fopen(path, "r");
    *base = 0; *size = 0;
    if (!f) return;
    while (fgets(line, sizeof line, f)){
        uint64_t s, e, off; char perms[8], dev[16]; unsigned long ino; char rest[256];
        rest[0] = 0;
        int k = sscanf(line, "%lx-%lx %7s %lx %15s %lu %255[^\n]", &s, &e, perms, &off, dev, &ino, rest);
        if (k < 6) continue;
        if (ino != 0 || rest[0] != 0) continue;        // anonymous, pathless only
        if (perms[0] != 'r' || perms[1] != 'w') continue;
        if (e - s > *size){ *base = s; *size = e - s; }
    }
    fclose(f);
}

#define HOT_BLOCK (2UL<<20)

int main(int argc, char **argv){
    if (argc < 6){
        fprintf(stderr, "usage: %s <pid> <seg_gb> <interval_ms> <samples_per_seg> <out_csv> [hotset_file]\n", argv[0]);
        return 1;
    }
    pid_t pid = atoi(argv[1]);
    uint64_t seg_bytes = (uint64_t)atoll(argv[2]) << 30;
    long interval_ms = atol(argv[3]);
    int nsamp = atoi(argv[4]);
    const char *out = argv[5];

    // Wait (up to 5 min) for the arena to appear and reach full size: phase_chase
    // mmaps the whole arena at startup, but be tolerant of launch wrappers.
    uint64_t base = 0, size = 0, prev = 0;
    for (int i = 0; i < 600; i++){
        find_arena(pid, &base, &size);
        if (size >= seg_bytes && size == prev) break;   // stable across 500ms
        prev = size;
        usleep(500 * 1000);
        if (kill(pid, 0) != 0){ fprintf(stderr, "[sampler] target %d died before arena found\n", pid); return 1; }
    }
    if (size < seg_bytes){ fprintf(stderr, "[sampler] no arena >= 1 segment found for pid %d\n", pid); return 1; }
    int nseg = size / seg_bytes;
    if (nseg > MAX_SEG) nseg = MAX_SEG;
    fprintf(stderr, "[sampler] pid=%d arena=0x%lx size=%.1fGB nseg=%d samples/seg=%d interval=%ldms\n",
            pid, base, size / 1073741824.0, nseg, nsamp, interval_ms);

    // optional hot-block map (phase_zipf PZ_HOTSET_OUT): "<seg> <byte_offset>"
    uint64_t **hot = NULL; int *nhot = NULL;
    if (argc >= 7 && argv[6][0]){
        FILE *fh = fopen(argv[6], "r");
        if (!fh){ perror("hotset_file"); return 1; }
        int cap[MAX_SEG]; hot = calloc(nseg, sizeof(void*)); nhot = calloc(nseg, sizeof(int));
        for (int s = 0; s < nseg; s++){ cap[s] = 1024; hot[s] = malloc(cap[s]*sizeof(uint64_t)); }
        int sg; unsigned long long off;
        while (fscanf(fh, "%d %llu", &sg, &off) == 2){
            if (sg < 0 || sg >= nseg) continue;
            if (nhot[sg] == cap[sg]){ cap[sg] *= 2; hot[sg] = realloc(hot[sg], cap[sg]*sizeof(uint64_t)); }
            hot[sg][nhot[sg]++] = off;
        }
        fclose(fh);
        for (int s = 0; s < nseg; s++)
            fprintf(stderr, "[sampler] hotset seg%d: %d blocks (%.2f GB)\n",
                    s, nhot[s], nhot[s]*HOT_BLOCK/1073741824.0);
    }

    FILE *fo = fopen(out, "w");
    if (!fo){ perror("out_csv"); return 1; }
    fprintf(fo, "ts_unix");
    for (int s = 0; s < nseg; s++) fprintf(fo, ",seg%d_dram_frac", s);
    for (int s = 0; s < nseg; s++) fprintf(fo, ",seg%d_valid", s);
    if (hot){
        for (int s = 0; s < nseg; s++) fprintf(fo, ",hot%d_dram_ratio", s);
        for (int s = 0; s < nseg; s++) fprintf(fo, ",hot%d_valid", s);
    }
    fprintf(fo, "\n"); fflush(fo);

    void **pages = malloc(sizeof(void*) * nsamp);
    int *status  = malloc(sizeof(int)  * nsamp);
    uint64_t seed = 0xc0ffee ^ (uint64_t)pid;
    uint64_t seg_pages = seg_bytes / PG;

    while (kill(pid, 0) == 0){
        double ts = unix_s();
        double frac[MAX_SEG]; int valid[MAX_SEG];
        for (int s = 0; s < nseg; s++){
            for (int i = 0; i < nsamp; i++){
                uint64_t p = xr(&seed) % seg_pages;
                pages[i] = (void*)(base + (uint64_t)s * seg_bytes + p * PG);
            }
            long rc = q_move_pages(pid, nsamp, pages, status);
            int n0 = 0, ok = 0;
            if (rc == 0){
                for (int i = 0; i < nsamp; i++){
                    if (status[i] >= 0){ ok++; if (status[i] == 0) n0++; }
                }
            }
            frac[s] = ok ? (double)n0 / ok : -1.0;
            valid[s] = ok;
        }
        double hfrac[MAX_SEG]; int hvalid[MAX_SEG];
        if (hot){
            for (int s = 0; s < nseg; s++){
                if (nhot[s] == 0){ hfrac[s] = -1.0; hvalid[s] = 0; continue; }
                for (int i = 0; i < nsamp; i++){
                    uint64_t b = hot[s][xr(&seed) % (uint64_t)nhot[s]];
                    uint64_t p = xr(&seed) % (HOT_BLOCK / PG);
                    pages[i] = (void*)(base + (uint64_t)s * seg_bytes + b + p * PG);
                }
                long rc = q_move_pages(pid, nsamp, pages, status);
                int n0 = 0, ok = 0;
                if (rc == 0) for (int i = 0; i < nsamp; i++){
                    if (status[i] >= 0){ ok++; if (status[i] == 0) n0++; }
                }
                hfrac[s] = ok ? (double)n0 / ok : -1.0;
                hvalid[s] = ok;
            }
        }
        fprintf(fo, "%.3f", ts);
        for (int s = 0; s < nseg; s++) fprintf(fo, ",%.4f", frac[s]);
        for (int s = 0; s < nseg; s++) fprintf(fo, ",%d", valid[s]);
        if (hot){
            for (int s = 0; s < nseg; s++) fprintf(fo, ",%.4f", hfrac[s]);
            for (int s = 0; s < nseg; s++) fprintf(fo, ",%d", hvalid[s]);
        }
        fprintf(fo, "\n"); fflush(fo);
        double spent = unix_s() - ts;
        long rem_us = interval_ms * 1000 - (long)(spent * 1e6);
        if (rem_us > 0) usleep(rem_us);
    }
    fprintf(stderr, "[sampler] target exited, done\n");
    fclose(fo);
    return 0;
}
