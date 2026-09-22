// SPDX-License-Identifier: GPL-2.0
//
// xTier userspace loader and control plane: opens per-thread PEBS events,
// loads the BPF program and pushes the quantized model into its maps, runs
// the per-epoch controller (dynamic threshold, top-K admission, adaptive
// sampling), and reports placement statistics.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <inttypes.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <sys/stat.h> // Added for chmod
#include <dirent.h>   // /proc/PID/task enumeration
#include <sched.h>    // sched_setaffinity
#include <limits.h>   // PATH_MAX
#include <linux/perf_event.h>
#include <math.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#include "common_kern.h"

// The Makefile passes MODEL_HDR through as -DXTIER_MODEL_HDR; hardcoding
// "mlp_q8.h" would pick up whatever sat in the include path instead.
#ifndef XTIER_MODEL_HDR
#define XTIER_MODEL_HDR "mlp_q8.h"
#endif
#include XTIER_MODEL_HDR

#ifndef __NR_perf_event_open
#define __NR_perf_event_open 298
#endif

// Define the struct
struct batch_ctrl {
    uint32_t idx;
    uint32_t epoch;
};

static volatile int stop_flag = 0;
static int mode = 0; // 0=monitor, 1=collect, 2=evaluate, 3=production

#define DRAM_PRESSURE_THRESHOLD_MB 670   // fallback ~10% of 6730MB DRAM
#define EVICT_BATCH_SIZE 512
#define EVICT_STRIDE     5    // evaluate pressure every 5 epochs
#define PROMO_TRACK_WINDOW 50 // rolling window for promotion rate tracking
#define HOT_THRESHOLD    1    // must match pebs_mlp_kern.c

// Top-K budget gate.
//
// K is a per-epoch fraction of that epoch's observed MLP scores: admit only
// the top `topk_admit_fraction` by score. Being dimensionless is the point.
// Deriving K from the absolute DRAM budget does not work -- at a 1:20 ratio
// the budget is ~985K pages while a whole run produces under 60K histogram
// entries, so the cutoff never leaves the bottom bucket and no admission
// control happens at all. The DRAM budget still bounds total promotions,
// via promote_max_pages on the kernel side; only the ranking is per-epoch.
//
// Usage:
//   XTIER_DRAM_BUDGET_MB=3850      (nonzero enables the feature)
//   XTIER_TOPK_FRACTION=0.15       (optional; default 0.15 = top 15%)
static uint64_t dram_budget_pages    = 0;    // configured via env; 0 = disabled
static double   topk_admit_fraction  = 0.15; // admit top 15% per epoch by default
static int32_t  out_zpC_cache        = 32;   // captured after load_model_weights
#define TOPK_WARMUP_EPOCHS 3                  // first N epochs use static thresh
#define TOPK_THRESH_MAX    253                // cls_u8 can reach 255 (raw=223); cap at
                                              // 253 so we can still filter deep into the
                                              // top buckets when the MLP saturates. 200
                                              // was too low -- at this workload 99.96%
                                              // of inferences had cls_u8 > 232.

// NUMA node identities -- must match xtier_executor.c module params
// (promote_node=0, demote_node=1 by default).
static const int promote_node_id = 0;
static const int demote_node_id  = 1;

static uint64_t total_evict_batches = 0;
static uint64_t total_evict_pages   = 0;
static uint64_t total_promo_scan_batches = 0;
static uint64_t total_promo_scan_pages   = 0;
static int      migration_stride_ctr = 0;

// Rolling promote/demote balance
static uint32_t rolling_promotes = 0;
static uint32_t rolling_demotes  = 0;

static void sig_handler(int sig) { (void)sig; stop_flag = 1; }

// -------------------- Adaptive Profiling State Machine --------------------
enum adapt_state { ADAPT_ACTIVE = 0, ADAPT_CRUISE = 1, ADAPT_MONITOR = 2 };
static const char *adapt_state_name[] = { "ACTIVE", "CRUISE", "MONITOR" };

// Multipliers for each state
#define CRUISE_PEBS_MULT   5
#define CRUISE_EPOCH_MULT  3
#define MONITOR_PEBS_MULT  5    // same as CRUISE -- need enough samples for promotion scan
#define MONITOR_EPOCH_MULT 10

// Transition thresholds (now based on tier churn + promotion scan, not just enqueue rate)
#define ENQ_WINDOW          20
#define ACTIVE_TO_CRUISE_ENQ_THRESH   5
#define ACTIVE_TO_CRUISE_EPOCHS       30     /* require 30 stable epochs before cruise to avoid premature converge */
#define ACTIVE_TO_CRUISE_EVICT_THRESH 10
#define ACTIVE_TO_CRUISE_EVICT_CYCLES 3

#define CRUISE_TO_MONITOR_ENQ_THRESH  2
#define CRUISE_TO_MONITOR_EPOCHS      50

#define WAKEUP_ENQ_THRESH   8       /* phase-change demo: only real spikes wake us, not gc noise */
#define WAKEUP_ENQ_EPOCHS   3       /* need 3 consecutive high-enq epochs */
#define WAKEUP_FLIP_THRESH  100     /* moderate flip threshold */
#define MONITOR_MAX_SECS    60

// Promotion/demotion balance
#define PROMO_DEMOTE_WINDOW  50
#define WARMUP_SKIP_SECS     0    // start migration immediately -- no warmup hold-off

struct adaptive_ctx {
    enum adapt_state state;
    uint64_t base_pebs_period;   // original command-line value
    int      base_epoch_ms;      // original command-line value

    // Enqueue rate tracking (rolling window)
    uint32_t enq_history[ENQ_WINDOW];
    int      enq_idx;
    uint64_t last_ring_tail;

    // Consecutive low-enqueue epoch counter
    int      low_enq_streak;
    // Consecutive high-enqueue epoch counter (for wakeup)
    int      high_enq_streak;

    // Eviction yield tracking
    int      low_evict_streak;
    int      evict_cycles_since_active; // total eviction cycles since entering ACTIVE

    // Tier flip tracking
    uint64_t last_tier_flips;
    uint64_t flip_history[10];
    int      flip_idx;

    // Time tracking for MONITOR timeout
    struct timespec state_entered;
    struct timespec last_transition;

    // Promotion scan tracking for convergence
    int      low_promo_scan_streak;  // consecutive scans with few candidates
    int      zero_promo_scan_streak; // consecutive scans with zero candidates
    int      last_promo_scan_found;  // candidates found in last scan

    // Stats
    uint64_t epochs_in_active;
    uint64_t epochs_in_cruise;
    uint64_t epochs_in_monitor;
};

static struct adaptive_ctx adapt;

static void adapt_init(uint64_t base_period, int base_ms) {
    memset(&adapt, 0, sizeof(adapt));
    adapt.state = ADAPT_ACTIVE;
    adapt.base_pebs_period = base_period;
    adapt.base_epoch_ms = base_ms;
    clock_gettime(CLOCK_MONOTONIC, &adapt.state_entered);
    adapt.last_transition = adapt.state_entered;
}

static uint64_t adapt_current_pebs_period(void) {
    switch (adapt.state) {
        case ADAPT_CRUISE:  return adapt.base_pebs_period * CRUISE_PEBS_MULT;
        case ADAPT_MONITOR: return adapt.base_pebs_period * MONITOR_PEBS_MULT;
        default:            return adapt.base_pebs_period;
    }
}

static int adapt_current_epoch_ms(void) {
    switch (adapt.state) {
        case ADAPT_CRUISE:  return adapt.base_epoch_ms * CRUISE_EPOCH_MULT;
        case ADAPT_MONITOR: return adapt.base_epoch_ms * MONITOR_EPOCH_MULT;
        default:            return adapt.base_epoch_ms;
    }
}

// Update PEBS period on all perf FDs via ioctl (uses globals declared below)
static void adapt_set_pebs_period(uint64_t new_period);  // defined after pid_perf globals

static void adapt_transition(enum adapt_state new_state, uint32_t epoch,
                             float enq_rate, int evict_yield, int flip_delta) {
    if (new_state == adapt.state) return;
    printf("[ADAPTIVE] Epoch %u: %s -> %s (enq_rate=%.1f, evict_yield=%d, flip_delta=%d)\n",
           epoch, adapt_state_name[adapt.state], adapt_state_name[new_state],
           enq_rate, evict_yield, flip_delta);
    adapt.state = new_state;
    clock_gettime(CLOCK_MONOTONIC, &adapt.state_entered);
    adapt.last_transition = adapt.state_entered;
    adapt.low_enq_streak = 0;
    adapt.high_enq_streak = 0;
    adapt.low_evict_streak = 0;
    adapt.evict_cycles_since_active = 0;

    // Apply new PEBS period and log it
    uint64_t new_period = adapt_current_pebs_period();
    adapt_set_pebs_period(new_period);
    printf("[ADAPTIVE] PEBS period -> %lu, epoch interval -> %dms\n",
           (unsigned long)new_period, adapt_current_epoch_ms());
}

// Called once per epoch with current metrics
static void adapt_update(uint32_t epoch, uint32_t enq_this_epoch,
                         int evict_yield, uint64_t cur_tier_flips) {
    // Track enqueue rate
    adapt.enq_history[adapt.enq_idx % ENQ_WINDOW] = enq_this_epoch;
    adapt.enq_idx++;

    // Compute rolling average enqueue rate
    int window = adapt.enq_idx < ENQ_WINDOW ? adapt.enq_idx : ENQ_WINDOW;
    float enq_avg = 0;
    for (int i = 0; i < window; i++) enq_avg += adapt.enq_history[i];
    enq_avg /= window;

    // Track tier flip delta
    uint64_t flip_delta_raw = cur_tier_flips - adapt.last_tier_flips;
    adapt.flip_history[adapt.flip_idx % 10] = flip_delta_raw;
    adapt.flip_idx++;
    adapt.last_tier_flips = cur_tier_flips;

    // Compute flip delta over last 10 epochs
    int flip_window = adapt.flip_idx < 10 ? adapt.flip_idx : 10;
    uint64_t flip_sum = 0;
    for (int i = 0; i < flip_window; i++) flip_sum += adapt.flip_history[i];
    int flip_delta = (int)flip_sum;

    // Track streaks
    if (enq_this_epoch < ACTIVE_TO_CRUISE_ENQ_THRESH)
        adapt.low_enq_streak++;
    else
        adapt.low_enq_streak = 0;

    if (enq_this_epoch > WAKEUP_ENQ_THRESH)
        adapt.high_enq_streak++;
    else
        adapt.high_enq_streak = 0;

    if (evict_yield >= 0) {
        adapt.evict_cycles_since_active++;
        if (evict_yield < ACTIVE_TO_CRUISE_EVICT_THRESH)
            adapt.low_evict_streak++;
        else
            adapt.low_evict_streak = 0;
    }
    // evict_yield == -1 means no eviction this epoch (stride gating); don't reset streak

    // Count time in state
    switch (adapt.state) {
        case ADAPT_ACTIVE:  adapt.epochs_in_active++;  break;
        case ADAPT_CRUISE:  adapt.epochs_in_cruise++;  break;
        case ADAPT_MONITOR: adapt.epochs_in_monitor++; break;
    }

    // --- Transition logic ---
    // XTIER_FORCE_ACTIVE pins the controller in ADAPT_ACTIVE. CRUISE raises
    // the PEBS period and the epoch, and the model is trained at a fixed
    // period, so pin this when training and serving have to match.
    {
        static int force_active = -1;
        if (force_active < 0) {
            const char *e = getenv("XTIER_FORCE_ACTIVE");
            force_active = (e && *e && strcmp(e, "0")) ? 1 : 0;
            if (force_active)
                printf("[*] XTIER_FORCE_ACTIVE: adaptive controller pinned to ACTIVE "
                       "(constant PEBS period/epoch for train/serve consistency)\n");
        }
        if (force_active) return;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    double secs_in_state = (now.tv_sec - adapt.state_entered.tv_sec) +
                           (now.tv_nsec - adapt.state_entered.tv_nsec) / 1e9;

    switch (adapt.state) {
    case ADAPT_ACTIVE:
        // Transition to CRUISE when enqueue rate is low AND promotion scan
        // isn't finding many candidates (placement is converging)
        if (adapt.low_enq_streak >= ACTIVE_TO_CRUISE_EPOCHS &&
            adapt.low_promo_scan_streak >= 3 &&
            (adapt.low_evict_streak >= ACTIVE_TO_CRUISE_EVICT_CYCLES ||
             adapt.evict_cycles_since_active == 0)) {
            adapt_transition(ADAPT_CRUISE, epoch, enq_avg, evict_yield, flip_delta);
        }
        break;

    case ADAPT_CRUISE:
        // CRUISE is the terminal low-overhead state. MONITOR is disabled --
        // it sampled too sparsely to keep up with workload phase changes.
        // Wake back up to ACTIVE on a spike or when the promotion scan finds
        // fresh hot CXL pages.
        if (adapt.high_enq_streak >= WAKEUP_ENQ_EPOCHS ||
            flip_delta > WAKEUP_FLIP_THRESH ||
            adapt.last_promo_scan_found >= 5) {
            adapt_transition(ADAPT_ACTIVE, epoch, enq_avg, evict_yield, flip_delta);
        }
        break;

    case ADAPT_MONITOR:
        // Disabled -- kept for enum compatibility. Bounce immediately to CRUISE.
        adapt_transition(ADAPT_CRUISE, epoch, enq_avg, evict_yield, flip_delta);
        break;
    }
}

static int perf_event_open_wrap(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    return (int)syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

// -------------------- /proc maps -> meta_map --------------------
// Forward decl: userspace-policy mode keeps a local mirror of meta_map
// (defined in the USERSPACE-POLICY section below; no-op when disabled).
static void us_meta_put_fwd(uint64_t reg_base, const struct meta_val *mv);
static int bucket_from_size(unsigned long sz) {
    if (sz < 4UL * 1024UL) return 0;
    if (sz < 64UL * 1024UL) return 1;
    if (sz < 2UL * 1024UL * 1024UL) return 2;
    if (sz < 1024UL * 1024UL * 1024UL) return 3;
    return 4;
}
static int page_type_from(const char *perms, const char *path) {
    if (path && strstr(path, "[stack]")) return 3;
    if (path && strstr(path, "[heap]"))  return 4;
    if (perms && strchr(perms, 'x'))     return 1;
    if (perms && strchr(perms, 'w'))     return 2;
    return 0;
}
// ============================================================================
// MULTI-TENANCY (static partition)
//
// Each managed process gets a SLOT (0..XTIER_MAX_TENANTS-1). The slot is
// stored as the target_pids value and packed into the low PAGE_SHIFT bits of
// every page/region key, which are structurally zero on a 4 KB-aligned
// address. That namespaces page_state_map, feature_row_map, pred_map,
// dedup_cache and meta_map per tenant without changing any key TYPE.
//
// "Static partition" = each tenant carries its OWN dram budget and therefore
// its own Top-K threshold, computed from its own slice of the score
// histogram. A shared histogram would let whichever tenant emits more samples
// set the admission bar for both.
//
// Configured via XTIER_TENANTS="<pid>:<budget_mb>[,<pid>:<budget_mb>...]".
// Unset => single tenant (argv[1], XTIER_DRAM_BUDGET_MB) in slot 0, which
// reproduces the pre-multi-tenancy key layout exactly.
struct tenant {
    pid_t    pid;
    uint8_t  slot;
    uint64_t budget_pages;
    uint32_t threshold;   // last published Top-K threshold
};
static struct tenant tenants[XTIER_MAX_TENANTS];
static int n_tenants = 0;

static int tenant_slot_of_pid(pid_t pid) {
    for (int i = 0; i < n_tenants; i++) if (tenants[i].pid == pid) return tenants[i].slot;
    return -1;
}

// Parse XTIER_TENANTS. Returns tenant count, 0 if unset.
static int parse_tenants(uint64_t default_budget_pages) {
    const char *env = getenv("XTIER_TENANTS");
    if (!env || !*env) return 0;
    char buf[512]; snprintf(buf, sizeof(buf), "%s", env);
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save); tok && n_tenants < XTIER_MAX_TENANTS;
         tok = strtok_r(NULL, ",", &save)) {
        int pid = 0; unsigned long mb = 0;
        if (sscanf(tok, "%d:%lu", &pid, &mb) < 1 || pid <= 0) continue;
        tenants[n_tenants].pid          = (pid_t)pid;
        tenants[n_tenants].slot         = (uint8_t)n_tenants;
        tenants[n_tenants].budget_pages = mb ? (uint64_t)mb * 256ull : default_budget_pages;
        n_tenants++;
    }
    return n_tenants;
}

static int populate_meta_map(int meta_map_fd, int pid, int slot) {
    char path[64]; snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    FILE *fp = fopen(path, "r"); if (!fp) return -1;
    char *line = NULL; size_t cap = 0;
    while (getline(&line, &cap, fp) > 0) {
        unsigned long start = 0, end = 0; char perms[8] = {0};
        if (sscanf(line, "%lx-%lx %4s", &start, &end, perms) != 3) continue;
        char *pathname = strchr(line, '/');
        if (!pathname) { pathname = strstr(line, "["); if (!pathname) pathname = (char *)""; }
        struct meta_val mv = {
            .perm_r = (uint8_t)(strchr(perms, 'r') != NULL),
            .perm_w = (uint8_t)(strchr(perms, 'w') != NULL),
            .perm_x = (uint8_t)(strchr(perms, 'x') != NULL),
            .page_type = (uint8_t)page_type_from(perms, pathname),
            .vma_size_bucket = (uint8_t)bucket_from_size(end - start),
            // anon=1 if no file backing (no '/' path: unnamed, [heap], [stack], [anon:...], etc.)
            .anon = (uint8_t)(pathname[0] != '/'),
            // shared=1 if mapping has 's' in perms (shared), vs 'p' (private/COW)
            .shared = (uint8_t)(strchr(perms, 's') != NULL),
        };
        unsigned long reg_s = start >> REG_SHIFT; unsigned long reg_e = (end - 1) >> REG_SHIFT;
        for (unsigned long rg = reg_s; rg <= reg_e; rg++) {
            __u64 key = (__u64)(rg << REG_SHIFT) | (__u64)(slot & (int)TENANT_SLOT_MASK);
            bpf_map_update_elem(meta_map_fd, &key, &mv, BPF_ANY);
            us_meta_put_fwd(key, &mv);   // userspace-policy local mirror (no-op unless enabled)
        }
    }
    free(line); fclose(fp); return 0;
}
// The target_pids VALUE is the tenant slot. Children inherit the parent's
// slot so a whole process tree shares one partition.
static void add_process_tree(int pid_map_fd, int meta_map_fd, int pid, int slot) {
    uint32_t p = (uint32_t)pid; uint8_t v = (uint8_t)slot;
    bpf_map_update_elem(pid_map_fd, &p, &v, BPF_ANY); populate_meta_map(meta_map_fd, pid, slot);
    char cmd[64], buf[32]; snprintf(cmd, sizeof(cmd), "pgrep -P %d", pid); FILE *fp = popen(cmd, "r");
    if (!fp) return;
    while (fgets(buf, sizeof(buf), fp)) { int child = atoi(buf); if (child > 0) add_process_tree(pid_map_fd, meta_map_fd, child, slot); } pclose(fp);
}

// Pin this process to the CPUs on the given NUMA node so the daemon doesn't
// contend with the workload on node-0 CPUs (0-9, 20-29 on this box). Default
// pin target is node 1; override with env var XTIER_PROFILER_CPU_NODE=0|1.
// Setting XTIER_PROFILER_CPU_NODE=-1 disables pinning entirely (old behavior).
static void pin_profiler_to_node(void) {
    const char *env = getenv("XTIER_PROFILER_CPU_NODE");
    int node = env ? atoi(env) : 1;
    if (node < 0) { printf("[*] Profiler CPU pinning DISABLED (XTIER_PROFILER_CPU_NODE=%s)\n", env); return; }
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "[!] profiler pin: can't read %s (errno=%d), skipping\n", path, errno); return; }
    char buf[256] = {0};
    if (!fgets(buf, sizeof(buf), fp)) { fclose(fp); return; }
    fclose(fp);
    cpu_set_t set;
    CPU_ZERO(&set);
    int count = 0;
    char *p = buf;
    while (*p && *p != '\n') {
        int lo = 0, hi = 0;
        if (sscanf(p, "%d-%d", &lo, &hi) == 2) {
            for (int c = lo; c <= hi; c++) { CPU_SET(c, &set); count++; }
        } else if (sscanf(p, "%d", &lo) == 1) {
            CPU_SET(lo, &set); count++; hi = lo;
        }
        while (*p && *p != ',' && *p != '\n') p++;
        if (*p == ',') p++;
    }
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "[!] profiler pin: sched_setaffinity failed errno=%d (%s)\n", errno, strerror(errno));
        return;
    }
    printf("[*] Profiler pinned to node %d CPUs (%d CPUs from %s)\n", node, count, buf);
}

// -------------------- Per-task PEBS attachment --------------------
struct pid_perf_entry { pid_t pid; int fd; };
static struct pid_perf_entry *pid_perf_entries = NULL;
static int pid_perf_count = 0;
static int pid_perf_cap   = 0;

static void adapt_set_pebs_period(uint64_t new_period) {
    // No-op in freq mode.
    //
    // PERF_EVENT_IOC_PERIOD with a non-zero period clears attr.freq in the
    // kernel and switches the event back to fixed-period sampling. On memory-
    // dense workloads (memcached, lightgbm) this produced runaway sample
    // rates (observed 297k/s aggregate at period=50000 on memcached, vs the
    // ~40k/s cap that freq=1 sample_freq=2000 enforces). The CRUISE/MONITOR
    // "slow down PEBS" trick from the fixed-period era is actively harmful
    // under freq mode, so we skip it entirely and let the kernel's PID
    // controller hold sample_freq.
    //
    // To tune overhead, set XTIER_PEBS_SAMPLE_FREQ at profiler launch.
    (void)new_period;
    (void)pid_perf_entries;
}

static bool pid_has_perf_fd(pid_t pid) {
    for (int i = 0; i < pid_perf_count; i++)
        if (pid_perf_entries[i].pid == pid) return true;
    return false;
}
static void open_perf_for_pid(pid_t pid, int stage0_fd, const struct perf_event_attr *pe_tmpl) {
    if (pid_has_perf_fd(pid)) return;
    struct perf_event_attr local_pe = *pe_tmpl;
    int fd = perf_event_open_wrap(&local_pe, pid, -1, -1, PERF_FLAG_FD_CLOEXEC);
    if (fd < 0) return; // PID may have already exited; skip silently
    ioctl(fd, PERF_EVENT_IOC_SET_BPF, stage0_fd);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    if (pid_perf_count >= pid_perf_cap) {
        pid_perf_cap = pid_perf_cap ? pid_perf_cap * 2 : 64;
        pid_perf_entries = realloc(pid_perf_entries, (size_t)pid_perf_cap * sizeof(*pid_perf_entries));
    }
    pid_perf_entries[pid_perf_count++] = (struct pid_perf_entry){ .pid = pid, .fd = fd };
}
// Open per-task PEBS for every THREAD in this process. perf_event_open(pid=X)
// attaches to a single task (one thread), not to all tasks in the thread
// group, so we must enumerate /proc/PID/task/* and open one fd per TID.
// Without this we only sample the main thread -- which for an OpenMP workload
// like XGBoost is the orchestrator that does almost no memory work, while all
// the actual compute happens in worker threads we never touch.
static void open_perf_for_threads(pid_t pid, int stage0_fd, const struct perf_event_attr *pe_tmpl) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/task", (int)pid);
    DIR *d = opendir(path);
    if (!d) return;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        pid_t tid = (pid_t)atoi(ent->d_name);
        if (tid > 0) open_perf_for_pid(tid, stage0_fd, pe_tmpl);
    }
    closedir(d);
}

// Diagnostic: per-CPU attach mode. Opens perf events as (pid=-1, cpu=N) for each
// online CPU on node 0. Useful to distinguish "OpenMP thread inheritance breaks
// per-task attach" from "kernel PEBS->BPF delivery is broken." BPF filters by
// tgid anyway via target_pids map, so per-CPU mode still only reacts to our
// workload's samples.
static void open_perf_percpu(int stage0_fd, const struct perf_event_attr *pe_tmpl) {
    // Read node 0's CPU list from /sys, e.g. "0-9,20-29"
    FILE *fp = fopen("/sys/devices/system/node/node0/cpulist", "r");
    char buf[256] = {0};
    if (fp) { fgets(buf, sizeof(buf), fp); fclose(fp); }
    // Parse ranges
    int cpus[256]; int ncpus = 0;
    char *p = buf;
    while (*p && *p != '\n') {
        int lo = 0, hi = 0;
        if (sscanf(p, "%d-%d", &lo, &hi) == 2) {
            for (int c = lo; c <= hi && ncpus < 256; c++) cpus[ncpus++] = c;
        } else if (sscanf(p, "%d", &lo) == 1) {
            cpus[ncpus++] = lo; hi = lo;
        }
        while (*p && *p != ',' && *p != '\n') p++;
        if (*p == ',') p++;
    }
    printf("[*] XTIER_ATTACH_PERCPU: opening perf events on %d node-0 CPUs: ", ncpus);
    for (int i = 0; i < ncpus; i++) printf("%d%s", cpus[i], i+1<ncpus ? "," : "\n");
    int succeeded = 0;
    for (int i = 0; i < ncpus; i++) {
        int cpu = cpus[i];
        struct perf_event_attr local_pe = *pe_tmpl;
        int fd = perf_event_open_wrap(&local_pe, -1, cpu, -1, PERF_FLAG_FD_CLOEXEC);
        if (fd < 0) { fprintf(stderr, "  cpu=%d open FAILED errno=%d (%s)\n", cpu, errno, strerror(errno)); continue; }
        int set_bpf_rc = ioctl(fd, PERF_EVENT_IOC_SET_BPF, stage0_fd);
        int enable_rc = ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        if (set_bpf_rc < 0 || enable_rc < 0) fprintf(stderr, "  cpu=%d fd=%d set_bpf=%d enable=%d errno=%d\n", cpu, fd, set_bpf_rc, enable_rc, errno);
        if (pid_perf_count >= pid_perf_cap) {
            pid_perf_cap = pid_perf_cap ? pid_perf_cap * 2 : 64;
            pid_perf_entries = realloc(pid_perf_entries, (size_t)pid_perf_cap * sizeof(*pid_perf_entries));
        }
        pid_perf_entries[pid_perf_count++] = (struct pid_perf_entry){ .pid = -cpu-1, .fd = fd };
        succeeded++;
    }
    printf("[+] per-CPU attach: %d/%d CPUs OK\n", succeeded, ncpus);
}

static void open_perf_for_tree(pid_t pid, int stage0_fd, const struct perf_event_attr *pe_tmpl) {
    if (getenv("XTIER_ATTACH_PERCPU")) {
        open_perf_percpu(stage0_fd, pe_tmpl);
        return;
    }
    open_perf_for_pid(pid, stage0_fd, pe_tmpl);
    open_perf_for_threads(pid, stage0_fd, pe_tmpl);
    char cmd[64], buf[32];
    snprintf(cmd, sizeof(cmd), "pgrep -P %d", (int)pid);
    FILE *fp = popen(cmd, "r");
    if (!fp) return;
    while (fgets(buf, sizeof(buf), fp)) {
        int child = atoi(buf);
        if (child > 0) open_perf_for_tree((pid_t)child, stage0_fd, pe_tmpl);
    }
    pclose(fp);
}

// -------------------- DRAM pressure eviction --------------------
static long read_node_free_mb(int node) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    long free_kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, " Node %*d MemFree: %ld kB", &free_kb) == 1) break;
    }
    fclose(f);
    return free_kb / 1024;
}

static long read_node_total_mb(int node) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    long total_kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, " Node %*d MemTotal: %ld kB", &total_kb) == 1) break;
    }
    fclose(f);
    return total_kb / 1024;
}

// Returns number of pages enqueued for eviction.
static int evict_coldest_from_dram(int state_map_fd, int queue_map_fd,
                                   int meta_map_fd, uint32_t cur_epoch,
                                   int batch_size) {
    struct {
        uint64_t page;
        uint32_t pid;
        uint32_t recent; // sum of hits_prev[0..2] -- evict lowest first
    } candidates[EVICT_BATCH_SIZE];
    int n_candidates = 0;
    uint32_t max_recent_in_batch = 0;
    int max_recent_idx = 0;

    uint64_t key = 0, next_key = 0;
    struct bpf_page_state pst;

    if (bpf_map_get_next_key(state_map_fd, NULL, &next_key) != 0) return 0;
    key = next_key;

    while (1) {
        if (bpf_map_lookup_elem(state_map_fd, &key, &pst) == 0) {
            if (pst.tier == 1) {
                // Never evict stack pages -- they're pinned to DRAM
                {
                    uint64_t rg = key & ~((uint64_t)((1ull << REG_SHIFT) - 1ull));
                    struct meta_val mv;
                    if (bpf_map_lookup_elem(meta_map_fd, &rg, &mv) == 0 && mv.page_type == 3)
                        goto next;
                }
                // Skip recently migrated pages to prevent thrashing
                if (pst.last_mig_epoch > 0 && cur_epoch > pst.last_mig_epoch &&
                    (cur_epoch - pst.last_mig_epoch) < 20) goto next;

                // Protect genuinely hot pages (2+ hits) in either recent epoch
                if (pst.hits_prev[0] >= HOT_THRESHOLD || pst.hits_prev[1] >= HOT_THRESHOLD) goto next;

                // Recent activity score -- lower is colder (better eviction candidate)
                uint32_t recent = pst.hits_prev[0] + pst.hits_prev[1] + pst.hits_prev[2];

                if (n_candidates < batch_size) {
                    candidates[n_candidates].page   = key;
                    candidates[n_candidates].pid    = pst.owner_pid;
                    candidates[n_candidates].recent = recent;
                    if (recent > max_recent_in_batch) {
                        max_recent_in_batch = recent;
                        max_recent_idx = n_candidates;
                    }
                    n_candidates++;
                } else if (recent < max_recent_in_batch) {
                    // Replace the warmest candidate with this colder page
                    candidates[max_recent_idx].page   = key;
                    candidates[max_recent_idx].pid    = pst.owner_pid;
                    candidates[max_recent_idx].recent = recent;
                    max_recent_in_batch = 0;
                    for (int i = 0; i < n_candidates; i++) {
                        if (candidates[i].recent > max_recent_in_batch) {
                            max_recent_in_batch = candidates[i].recent;
                            max_recent_idx = i;
                        }
                    }
                }
            }
        }
next:
        if (bpf_map_get_next_key(state_map_fd, &key, &next_key) != 0) break;
        key = next_key;
    }

    if (n_candidates == 0) return 0;

    // Read queue, append demotion entries, write back
    uint32_t qkey = 0;
    struct region_queue *q = malloc(sizeof(struct region_queue));
    if (!q) return 0;
    if (bpf_map_lookup_elem(queue_map_fd, &qkey, q) != 0) { free(q); return 0; }

    int enqueued = 0;
    for (int i = 0; i < n_candidates; i++) {
        uint32_t tail = q->tail;
        uint32_t head = q->head;
        if (tail - head >= MAX_REGIONS_PER_EPOCH) break; // ring full
        uint32_t mask = MAX_REGIONS_PER_EPOCH - 1;
        struct region_entry *e = &q->entries[tail & mask];
        e->addr  = candidates[i].page;
        e->pid   = candidates[i].pid;
        e->score = -500;
        e->pred  = 0;  // demote
        e->epoch = cur_epoch;
        e->seq   = tail + 1;
        q->tail  = tail + 1;
        enqueued++;

        // Fix 4: record the eviction decision in page_state so eval.csv captures it
        struct bpf_page_state pst_upd;
        if (bpf_map_lookup_elem(state_map_fd, &candidates[i].page, &pst_upd) == 0) {
            pst_upd.last_final_action = 2; // EVAL_A_DEMOTE
            pst_upd.last_reason       = 7; // EVAL_R_PRESSURE_EVICT
            bpf_map_update_elem(state_map_fd, &candidates[i].page, &pst_upd, BPF_ANY);
        }
    }

    bpf_map_update_elem(queue_map_fd, &qkey, q, BPF_ANY);
    free(q);
    return enqueued;
}

// -------------------- Tier resolver --------------------
// Walk page_state_map, find pages with tier==0 (unknown), and resolve their
// actual physical NUMA node via move_pages(). This replaces the broken BPF
// bootstrap that assumed --preferred=N always lands on node N (false once
// DRAM fills and first-touch spills to the other node).
//
// Why this matters: without ground-truth tier info, the heuristic filter,
// the noop guard, and the promotion scanner are all flying blind. The
// promotion scanner in particular only ever sees tier==2 pages, which means
// hot CXL pages we've never touched are invisible to it.
//
// Cost: one syscall per batch of up to RESOLVE_BATCH pages, plus N hash
// updates. Run periodically, not every epoch -- placement is sticky enough
// that we don't need fresh ground truth more than once per second.

#define RESOLVE_BATCH 1024
#ifndef __NR_move_pages
#define __NR_move_pages 279  // x86_64
#endif

static long sys_move_pages(int pid, unsigned long count, void **pages,
                           const int *nodes, int *status, int flags) {
    return syscall(__NR_move_pages, pid, count, pages, nodes, status, flags);
}

/* Top-K placement + weighted DRAM coverage Q.
 *
 * Walks page_state_map. Per page:
 *   score = cur_hits + sum(hits_prev[0..5])     (~recent access count)
 *   tier  = pst.tier  (1=DRAM, 2=CXL, 0=unresolved -- populated by the
 *           1Hz resolver call we already make from the main loop)
 *
 * Emits per second:
 *   K_oracle              = oracle DRAM capacity in pages (env XTIER_TOPK_K)
 *   hot_on_n0/n1/other    = page counts in the top-K hottest among sampled
 *   total_score           = sum of score over all sampled pages
 *   dram_score            = sum of score over pages with tier==1
 *   topk_score            = sum of score over the top-K hottest pages
 *   Q                     = dram_score / topk_score
 *
 * Q is "weighted DRAM coverage of the sampled hot set" -- bounded [0, 1],
 * intensity-independent, both numerator and denominator come from the same
 * (potentially biased) PEBS sampler so most bias cancels. When sampled set
 * <= K_oracle, Q reduces to "fraction of sampled access weight on DRAM",
 * which is still useful but not a strict oracle ratio.
 *
 * Reads tier from BPF state (no per-tick move_pages call) so the cost is
 * just the map walk. */
struct topk_entry { __u64 page; __u64 score; __u8 tier; };
static int topk_cmp(const void *a, const void *b) {
    __u64 sa = ((const struct topk_entry *)a)->score;
    __u64 sb = ((const struct topk_entry *)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return 1;
    return 0;
}
static void emit_topk_placement(int state_map_fd, int target_pid, int K,
                                FILE *csv, struct timespec t0_real)
{
    if (!csv || K <= 0) return;
    static struct topk_entry *buf = NULL;
    static int buf_cap = 0;
    if (!buf) {
        buf_cap = 1 << 20;
        buf = (struct topk_entry *)malloc(buf_cap * sizeof(*buf));
        if (!buf) return;
    }

    int n = 0;
    __u64 total_score = 0;
    __u64 key = 0, next_key = 0;
    struct bpf_page_state pst;
    if (bpf_map_get_next_key(state_map_fd, NULL, &next_key) != 0) goto emit;
    key = next_key;
    while (n < buf_cap) {
        if (bpf_map_lookup_elem(state_map_fd, &key, &pst) == 0) {
            __u64 score = pst.cur_hits;
            for (int i = 0; i < HIST_STEPS; i++) score += pst.hits_prev[i];
            if (score > 0) {
                buf[n].page = key;
                buf[n].score = score;
                buf[n].tier = 255;  /* will be filled by move_pages below */
                total_score += score;
                n++;
            }
        }
        if (bpf_map_get_next_key(state_map_fd, &key, &next_key) != 0) break;
        key = next_key;
    }

    /* Query actual node placement for ALL sampled pages via batched
     * move_pages. Tier from BPF state was wrong here -- it's only set after
     * resolver/migration, not for default-allocated pages. We need actual
     * placement to compute dram_score honestly. */
    enum { CHUNK = 4096 };
    void *pages_buf[CHUNK];
    int   status_buf[CHUNK];
    __u64 dram_score = 0;
    int n_classified = 0;
    for (int off = 0; off < n; off += CHUNK) {
        int cnt = (n - off) < CHUNK ? (n - off) : CHUNK;
        for (int i = 0; i < cnt; i++) pages_buf[i] = (void *)(uintptr_t)PAGE_ADDR(buf[off + i].page);
        long r = sys_move_pages(target_pid, cnt, pages_buf, NULL, status_buf, 0);
        if (r < 0 && r != -ENOENT) continue;
        for (int i = 0; i < cnt; i++) {
            int s = status_buf[i];
            if (s == 0) {
                buf[off + i].tier = 1;  /* DRAM (N0) */
                dram_score += buf[off + i].score;
                n_classified++;
            } else if (s == 1) {
                buf[off + i].tier = 2;  /* CXL (N1) */
                n_classified++;
            } else {
                buf[off + i].tier = 0;  /* not mapped or other */
            }
        }
    }

    /* Compact: keep only mapped pages (tier 1=N0 or 2=N1). Unmapped pages
     * (status<0 from move_pages -- typically sklearn temp allocations that
     * were sampled briefly then freed) are not part of the workload's
     * current placement question. */
    int n_mapped = 0;
    for (int i = 0; i < n; i++) {
        if (buf[i].tier == 1 || buf[i].tier == 2) {
            buf[n_mapped++] = buf[i];
        }
    }

    /* Recompute total_score and dram_score over the mapped set only. */
    total_score = 0; dram_score = 0;
    for (int i = 0; i < n_mapped; i++) {
        total_score += buf[i].score;
        if (buf[i].tier == 1) dram_score += buf[i].score;
    }

    int hot_n0 = 0, hot_n1 = 0, hot_other = 0;
    int top = 0;
    __u64 topk_score = 0;
    if (n_mapped > 0) {
        qsort(buf, n_mapped, sizeof(*buf), topk_cmp);
        top = (n_mapped < K) ? n_mapped : K;
        for (int i = 0; i < top; i++) {
            topk_score += buf[i].score;
            if (buf[i].tier == 1)      hot_n0++;
            else if (buf[i].tier == 2) hot_n1++;
            else                       hot_other++;
        }
    }
    /* `n` reported below is the original sampled count (with unmapped);
     * `n_mapped` lets the plotter check how much was filtered. */

emit: {
        struct timespec wall; clock_gettime(CLOCK_REALTIME, &wall);
        double t_rel = (wall.tv_sec - t0_real.tv_sec) +
                       (wall.tv_nsec - t0_real.tv_nsec) / 1e9;
        double Q = (topk_score > 0) ? (double)dram_score / (double)topk_score : 0.0;
        fprintf(csv,
            "%ld.%06ld,%.3f,%d,%d,%d,%d,%d,%llu,%llu,%llu,%.6f,%d\n",
            (long)wall.tv_sec, (long)wall.tv_nsec / 1000, t_rel,
            top, hot_n0, hot_n1, hot_other, n,
            (unsigned long long)total_score,
            (unsigned long long)dram_score,
            (unsigned long long)topk_score,
            Q, n_mapped);
        fflush(csv);
    }
}

// Per-tenant. page_state_map keys are tenant-TAGGED, so we must (a) skip keys
// belonging to other tenants -- move_pages against the wrong mm would resolve
// a valid-looking but meaningless node -- and (b) strip the slot before the
// syscall. Called once per tenant from the main loop.
static int resolve_unknown_tiers_slot(int state_map_fd, int target_pid, int slot,
                                      int promote_node, int demote_node) {
    void *pages[RESOLVE_BATCH];
    uint64_t page_keys[RESOLVE_BATCH];
    int status[RESOLVE_BATCH];
    int n = 0, total_resolved = 0;

    uint64_t key = 0, next_key = 0;
    struct bpf_page_state pst;

    if (bpf_map_get_next_key(state_map_fd, NULL, &next_key) != 0) return 0;
    key = next_key;

    while (1) {
        if (bpf_map_lookup_elem(state_map_fd, &key, &pst) == 0) {
            // Only resolve pages with unknown tier AND recent activity.
            // Cold tier==0 pages don't need ground truth -- they're not
            // migration candidates anyway, and walking millions of them
            // through move_pages() would stall the workload on mmap_lock.
            if (pst.tier == 0 && (int)TENANT_SLOT(key) == slot &&
                (pst.hits_prev[0] > 0 || pst.hits_prev[1] > 0)) {
                pages[n] = (void *)(uintptr_t)PAGE_ADDR(key);
                page_keys[n] = key;   /* tagged: map writes below use this */
                n++;
                if (n == RESOLVE_BATCH) {
                    long ret = sys_move_pages(target_pid, n, pages, NULL, status, 0);
                    if (ret >= 0 || ret == -ENOENT) {
                        for (int i = 0; i < n; i++) {
                            if (status[i] < 0) continue;  // not mapped, skip
                            struct bpf_page_state st_upd;
                            if (bpf_map_lookup_elem(state_map_fd, &page_keys[i], &st_upd) != 0)
                                continue;
                            if (st_upd.tier != 0) continue;  // raced
                            if (status[i] == promote_node)      st_upd.tier = 1;
                            else if (status[i] == demote_node)  st_upd.tier = 2;
                            else continue;
                            bpf_map_update_elem(state_map_fd, &page_keys[i], &st_upd, BPF_ANY);
                            total_resolved++;
                        }
                    }
                    n = 0;
                }
            }
        }
        if (bpf_map_get_next_key(state_map_fd, &key, &next_key) != 0) break;
        key = next_key;
    }
    if (n > 0) {
        long ret = sys_move_pages(target_pid, n, pages, NULL, status, 0);
        if (ret >= 0 || ret == -ENOENT) {
            for (int i = 0; i < n; i++) {
                if (status[i] < 0) continue;
                struct bpf_page_state st_upd;
                if (bpf_map_lookup_elem(state_map_fd, &page_keys[i], &st_upd) != 0)
                    continue;
                if (st_upd.tier != 0) continue;
                if (status[i] == promote_node)      st_upd.tier = 1;
                else if (status[i] == demote_node)  st_upd.tier = 2;
                else continue;
                bpf_map_update_elem(state_map_fd, &page_keys[i], &st_upd, BPF_ANY);
                total_resolved++;
            }
        }
    }
    return total_resolved;
}
static int resolve_unknown_tiers(int state_map_fd, int target_pid,
                                 int promote_node, int demote_node) {
    return resolve_unknown_tiers_slot(state_map_fd, target_pid, 0, promote_node, demote_node);
}

// -------------------- Promotion scan: find hot CXL pages --------------------
// Returns number of pages enqueued for promotion.
static int promote_hot_from_cxl(int state_map_fd, int queue_map_fd,
                                int meta_map_fd, uint32_t cur_epoch,
                                int batch_size) {
    struct {
        uint64_t page;
        uint32_t pid;
        uint32_t heat;  // sum of recent hits -- promote hottest first
    } candidates[EVICT_BATCH_SIZE];
    int n_candidates = 0;
    uint32_t min_heat_in_batch = UINT32_MAX;
    int min_heat_idx = 0;

    uint64_t key = 0, next_key = 0;
    struct bpf_page_state pst;

    if (bpf_map_get_next_key(state_map_fd, NULL, &next_key) != 0) return 0;
    key = next_key;

    while (1) {
        if (bpf_map_lookup_elem(state_map_fd, &key, &pst) == 0) {
            // Consider tier==2 (known CXL) AND tier==0 (unresolved) pages.
            // The executor will gup-check before migrating, so any tier==0
            // page that turns out to already be on DRAM gets filtered out
            // there as a noop. This makes the promotion scanner robust to
            // resolver lag.
            if (pst.tier == 2 || pst.tier == 0) {
                // Skip recently migrated pages (cooldown)
                if (pst.last_mig_epoch > 0 && cur_epoch > pst.last_mig_epoch &&
                    (cur_epoch - pst.last_mig_epoch) < 20) goto next_promo;

                // Check if page is hot: hits_prev[0] >= HOT_THRESHOLD
                // OR actively accessed in 2 consecutive epochs
                int hot = 0;
                if (pst.hits_prev[0] >= HOT_THRESHOLD) hot = 1;
                else if (pst.hits_prev[0] > 0 && pst.hits_prev[1] > 0) hot = 1;

                if (hot) {
                    uint32_t heat = pst.hits_prev[0] + pst.hits_prev[1] + pst.hits_prev[2];

                    if (n_candidates < batch_size) {
                        candidates[n_candidates].page = key;
                        candidates[n_candidates].pid  = pst.owner_pid;
                        candidates[n_candidates].heat = heat;
                        if (heat < min_heat_in_batch) {
                            min_heat_in_batch = heat;
                            min_heat_idx = n_candidates;
                        }
                        n_candidates++;
                    } else if (heat > min_heat_in_batch) {
                        // Replace the coolest candidate with this hotter page
                        candidates[min_heat_idx].page = key;
                        candidates[min_heat_idx].pid  = pst.owner_pid;
                        candidates[min_heat_idx].heat = heat;
                        min_heat_in_batch = UINT32_MAX;
                        for (int i = 0; i < n_candidates; i++) {
                            if (candidates[i].heat < min_heat_in_batch) {
                                min_heat_in_batch = candidates[i].heat;
                                min_heat_idx = i;
                            }
                        }
                    }
                }
            }
        }
next_promo:
        if (bpf_map_get_next_key(state_map_fd, &key, &next_key) != 0) break;
        key = next_key;
    }

    if (n_candidates == 0) return 0;

    // Enqueue promotions into the ring buffer
    uint32_t qkey = 0;
    struct region_queue *q = malloc(sizeof(struct region_queue));
    if (!q) return 0;
    if (bpf_map_lookup_elem(queue_map_fd, &qkey, q) != 0) { free(q); return 0; }

    int enqueued = 0;
    for (int i = 0; i < n_candidates; i++) {
        uint32_t tail = q->tail;
        uint32_t head = q->head;
        if (tail - head >= MAX_REGIONS_PER_EPOCH) break;
        uint32_t mask = MAX_REGIONS_PER_EPOCH - 1;
        struct region_entry *e = &q->entries[tail & mask];
        e->addr  = candidates[i].page;
        e->pid   = candidates[i].pid;
        e->score = 500;   // positive score = promote
        e->pred  = 1;     // promote
        e->epoch = cur_epoch;
        e->seq   = tail + 1;
        q->tail  = tail + 1;
        enqueued++;

        // Record promotion decision in page_state
        struct bpf_page_state pst_upd;
        if (bpf_map_lookup_elem(state_map_fd, &candidates[i].page, &pst_upd) == 0) {
            pst_upd.last_final_action = 1; // EVAL_A_PROMOTE
            pst_upd.last_reason       = 3; // EVAL_R_HEUR_HOT
            bpf_map_update_elem(state_map_fd, &candidates[i].page, &pst_upd, BPF_ANY);
        }
    }

    bpf_map_update_elem(queue_map_fd, &qkey, q, BPF_ANY);
    free(q);
    return enqueued;
}

// -------------------- model weights --------------------
static void get_scale_and_shift(float real_multiplier, int32_t *m, int32_t *s) {
    if (real_multiplier == 0.0f) { *m = 0; *s = 0; return; }
    int exp; float sig = frexpf(real_multiplier, &exp); *m = (int32_t)(sig * 2147483648.0f); *s = 31 - exp;
}

static int load_model_weights(struct bpf_object *obj) {
    printf("[*] Loading Model Weights...\n");
    int fd_qpm = bpf_object__find_map_fd_by_name(obj, "qpm"); if (fd_qpm < 0) return -1;
    struct qparams qp = { .in_zp0 = L0_IN_ZP[0], .out_zp0 = L0_OUT_ZP[0], .out_zp1 = L1_OUT_ZP[0], .out_zpC = L2_OUT_ZP[0], .cls_thr_q = CLS_THR_QINT8[0] };
    out_zpC_cache = qp.out_zpC;  // top-K threshold compute needs this
    __u32 k0 = 0; bpf_map_update_elem(fd_qpm, &k0, &qp, BPF_ANY);
    int fds[] = { bpf_object__find_map_fd_by_name(obj,"l0_wb"), bpf_object__find_map_fd_by_name(obj,"l0_mul"), bpf_object__find_map_fd_by_name(obj,"l1_wb"), bpf_object__find_map_fd_by_name(obj,"l1_mul"), bpf_object__find_map_fd_by_name(obj,"lc_wb"), bpf_object__find_map_fd_by_name(obj,"lc_mul") };
    for (int i=0; i<6; i++) if (fds[i] < 0) return -1;
    for (int i=0; i<L0_OUT; i++) { struct row0 r={.b=L0_B_INT32[i]}; for(int j=0; j<L0_IN; j++) r.w[j]=L0_W[i*L0_IN+j]; struct mul m; get_scale_and_shift((L0_IN_SCALE[0]*L0_W_SCALE[i])/L0_OUT_SCALE[0], &m.M, &m.shift); bpf_map_update_elem(fds[0], &i, &r, 0); bpf_map_update_elem(fds[1], &i, &m, 0); }
    for (int i=0; i<L1_OUT; i++) { struct row1 r={.b=L1_B_INT32[i]}; for(int j=0; j<L0_OUT; j++) r.w[j]=L1_W[i*L0_OUT+j]; struct mul m; get_scale_and_shift((L1_IN_SCALE[0]*L1_W_SCALE[i])/L1_OUT_SCALE[0], &m.M, &m.shift); bpf_map_update_elem(fds[2], &i, &r, 0); bpf_map_update_elem(fds[3], &i, &m, 0); }
    struct rowO rc={.b=L2_B_INT32[0]}; for(int j=0; j<L1_OUT; j++) rc.w[j]=L2_W[j]; struct mul mc; get_scale_and_shift((L2_IN_SCALE[0]*L2_W_SCALE[0])/L2_OUT_SCALE[0], &mc.M, &mc.shift); bpf_map_update_elem(fds[4], &k0, &rc, 0); bpf_map_update_elem(fds[5], &k0, &mc, 0);
    return 0;
}

static int setup_tail_calls(struct bpf_object *obj) {
    int fd_jmp = bpf_object__find_map_fd_by_name(obj, "jmp_table"); if (fd_jmp < 0) return -1;
    int fd0 = bpf_program__fd(bpf_object__find_program_by_name(obj, "run_inference_l0"));
    int fd1 = bpf_program__fd(bpf_object__find_program_by_name(obj, "run_inference_l1"));
    int fdb = bpf_program__fd(bpf_object__find_program_by_name(obj, "run_inference_batch"));
    __u32 k1=1, k2=2, k3=3;
    bpf_map_update_elem(fd_jmp, &k1, &fd0, BPF_ANY); bpf_map_update_elem(fd_jmp, &k2, &fd1, BPF_ANY); bpf_map_update_elem(fd_jmp, &k3, &fdb, BPF_ANY);
    return 0;
}

// -------------------- CSV --------------------
static void print_fp_hex(FILE *out, const __u64 *w, int nwords) { for (int i = nwords - 1; i >= 0; --i) fprintf(out, "%016llx", (unsigned long long)w[i]); }
static void print_csv_header_once(FILE *csv) {
    static int p=0; if(p)return; p=1;
    fprintf(csv, "epoch,page,hits,lat_sum,hits_prev_1,lat_prev_1,delta_hits_1,hits_prev_2,lat_prev_2,delta_hits_2,hits_prev_3,lat_prev_3,delta_hits_3,hits_prev_4,lat_prev_4,delta_hits_4,hits_prev_5,lat_prev_5,delta_hits_5,hits_prev_6,lat_prev_6,delta_hits_6,recency_epochs,epochs_since_promotion,fp1_hex,fp2_hex,fp3_hex,fp4_hex,perm_r,perm_w,perm_x,page_type,vma_size_bucket,anon,shared,off_bucket\n");
}
static void emit_feature_row_csv(FILE *csv, const struct feature_row *r) {
    if(!csv)return; print_csv_header_once(csv);
    fprintf(csv, "%u,%#llx,%u,%u", r->epoch, (unsigned long long)r->page, r->hits, r->lat_sum);
    for (int i=0; i<HIST_STEPS; i++) fprintf(csv, ",%u,%u,%d", r->hits_prev[i], r->lat_prev[i], r->delta_hits[i]);
    fprintf(csv, ",%u,%u,", r->recency_epochs, r->epochs_since_promotion);
    for (int j=0; j<FP_HIST; j++) { print_fp_hex(csv, r->fp[j], FP_WORDS_PER_SLOT); fputc(j==FP_HIST-1?',':',', csv); }
    fprintf(csv, "%u,%u,%u,%u,%u,%u,%u,%u\n", r->perm_r, r->perm_w, r->perm_x, r->page_type, r->vma_size_bucket, r->anon, r->shared, r->off_bucket);
}
static void emit_eval_row(FILE *csv, const struct feature_row *r, const struct pred_row *p, const struct bpf_page_state *st) {
    if(!csv)return;
    static int ph=0; if(!ph){ fprintf(csv, "epoch,page,pfn,pid,hits,lat_sum,score,pred,final_action,reason,tier,last_mig_epoch,page_type\n"); ph=1; }
    fprintf(csv, "%u,%#llx,%llu,%u,%u,%u,%d,%d,%u,%u,%u,%u,%u\n", r->epoch, (unsigned long long)r->page, (unsigned long long)(r->page>>PAGE_SHIFT), r->pid, r->hits, r->lat_sum, p?p->score:0, p?p->pred:-1, st?st->last_final_action:0, st?st->last_reason:0, st?st->tier:0, r->last_mig_epoch, r->page_type);
}

// -------------------- TOP-K BUDGET GATE --------------------
// Reads the now-inactive histogram (the one BPF just stopped writing to),
// finds the smallest cls_u8 cutoff C such that sum(hist[C..255]) <= budget
// * fill_target (so only the top-K-scoring pages admit), then converts C
// back to the raw-score threshold that BPF compares with (`raw > thresh`
// where `raw = cls_u8 - out_zpC`). Zeros the buffer on the way out so
// the next epoch starts clean.
//
// Returns the computed promote_threshold_dyn (clamped to [1, TOPK_THRESH_MAX]).
// Returns 0 if the feature is disabled or histogram was empty (caller should
// keep the previous threshold or fall back to the static default).
// Core math, shared verbatim by the in-BPF path (histogram read from the BPF
// map) and the userspace-policy path (local histogram). Identical semantics
// is load-bearing for the userspace-policy counterfactual: both modes must
// compute the same admission threshold from the same score distribution.
static uint32_t topk_threshold_from_hist(const uint64_t hist[SCORE_HIST_BUCKETS],
                                         uint64_t budget_pages) {
    uint64_t total = 0;
    for (int i = 0; i < SCORE_HIST_BUCKETS; i++) total += hist[i];
    if (total == 0 || budget_pages == 0) return 0;

    // K is a per-epoch fraction of that epoch's observed inferences.
    uint64_t K = (uint64_t)((double)total * topk_admit_fraction);
    if (K < 1) K = 1;
    if (K > budget_pages) K = budget_pages;

    // Diagnostic: every ~200 threshold computes, dump top-10 histogram
    // buckets so we can see whether the MLP is saturated or well-ranked.
    static int hist_dump_counter = 0;
    static int hist_dump_every = -1;
    if (hist_dump_every < 0) {
        const char *e = getenv("XTIER_HIST_DUMP_EVERY");
        hist_dump_every = (e && atoi(e) > 0) ? atoi(e) : 200;
    }
    /* With XTIER_HIST_DUMP_EVERY=1 this emits the full score
     * distribution every epoch, which is how we measure whether a scorer
     * actually RANKS (many populated buckets) or merely saturates (one
     * bucket holding everything). A saturated scorer makes top-K arbitrary
     * and cannot be credited with selection quality. */
    if ((hist_dump_counter++ % hist_dump_every) == 0) {
        {   /* distinct-bucket + top-bucket-share summary, cheap to parse */
            int nz = 0; uint64_t top = 0;
            for (int b = 0; b < SCORE_HIST_BUCKETS; b++) { if (hist[b]) { nz++; if (hist[b] > top) top = hist[b]; } }
            fprintf(stdout, "[SCOREDIST] total=%llu buckets=%d top_share=%.4f\n",
                    (unsigned long long)total, nz, total ? (double)top / (double)total : 0.0);
        }
        int top_buckets[10] = {0};
        int tb_ptr = 0;
        // Find non-empty buckets from the top down
        for (int b = SCORE_HIST_BUCKETS - 1; b >= 0 && tb_ptr < 10; b--) {
            if (hist[b] > 0) top_buckets[tb_ptr++] = b;
        }
        fprintf(stdout, "[TOPK_HIST] total=%llu K=%llu top10=[",
                (unsigned long long)total, (unsigned long long)K);
        for (int i = 0; i < tb_ptr; i++) {
            fprintf(stdout, "%d:%llu%s",
                    top_buckets[i], (unsigned long long)hist[top_buckets[i]],
                    i + 1 < tb_ptr ? "," : "");
        }
        fprintf(stdout, "]\n");
    }

    // Walk from top bucket down, accumulate until we exceed K.
    uint64_t acc = 0;
    int cutoff = 0;  // smallest cls_u8 that still admits
    for (int b = SCORE_HIST_BUCKETS - 1; b >= 0; b--) {
        acc += hist[b];
        if (acc >= K) { cutoff = b; break; }
        cutoff = b;  // running min
    }

    // raw > thresh_dyn passes when cls_u8 > out_zpC + thresh_dyn
    // We want this to match cls_u8 >= cutoff, i.e. cls_u8 > cutoff - 1
    // So: out_zpC + thresh_dyn = cutoff - 1  =>  thresh_dyn = cutoff - 1 - out_zpC
    int32_t dyn = cutoff - 1 - out_zpC_cache;
    if (dyn < 1) dyn = 1;
    if (dyn > TOPK_THRESH_MAX) dyn = TOPK_THRESH_MAX;
    return (uint32_t)dyn;
}

// Reads ONE tenant's slice of the histogram ([slot][bucket]) and drains it.
static uint32_t topk_compute_threshold_slot(int fd_hist, uint64_t budget_pages, int slot) {
    uint64_t hist[SCORE_HIST_BUCKETS] = {0};
    __u32 base = (__u32)slot * SCORE_HIST_BUCKETS;
    for (__u32 i = 0; i < SCORE_HIST_BUCKETS; i++) {
        uint64_t v = 0; __u32 k = base + i;
        bpf_map_lookup_elem(fd_hist, &k, &v);
        hist[i] = v;
    }
    // Zero the buffer for next use (we just drained it)
    uint64_t zero = 0;
    for (__u32 i = 0; i < SCORE_HIST_BUCKETS; i++) {
        __u32 k = base + i;
        bpf_map_update_elem(fd_hist, &k, &zero, BPF_ANY);
    }
    return topk_threshold_from_hist(hist, budget_pages);
}
static uint32_t topk_compute_threshold(int fd_hist, uint64_t budget_pages) {
    return topk_compute_threshold_slot(fd_hist, budget_pages, 0);
}

// ============================================================================
// USERSPACE-POLICY COUNTERFACTUAL
//
// "How significant is the in-kernel design choice, end to end?" This mode
// answers it: the SAME trained INT8 MLP, the SAME features, filters, top-K
// budget gate, admission dice and cooldown -- but executed in USERSPACE, on a
// sample stream forwarded out of the kernel through a ringbuf, with decisions
// re-injected through the same enqueue_migration()/executor path.
//
// Everything downstream of scoring is byte-identical (userspace_enqueue BPF
// program -> same dedup cache -> same CAS ring -> same kernel worker).
// Everything upstream (PEBS event, freq, epoch cadence) is identical.
// What moves across the boundary: per-sample state tracking, the heuristic
// filter, the MLP forward pass, the histogram/threshold, the gates.
//
// Known, documented deviations from the in-BPF path (all inherent to a
// userspace agent, i.e. part of what is being measured):
//   - Samples can be LOST when the ring overruns (stats.us_fwd_drop). The
//     in-BPF path never loses a sample it decided to process.
//   - Tier/cooldown feedback: the in-BPF path gets ground truth pushed into
//     page_state_map by the executor; this agent sets tier=1/last_mig
//     optimistically at admit time and corrects via a 1 Hz move_pages()
//     readback of recent admits (syscall round-trip = the userspace tax).
//   - The admission dice uses a userspace PRNG (same distribution).
// Enabled by XTIER_USERSPACE_POLICY=1; supports the mlp scorer only.
// ============================================================================

static int us_mode = 0;

// ---- verbatim copies of the BPF fixed-point helpers (must stay in sync
// ---- with pebs_mlp_kern.c; any drift breaks the "same policy" claim) ----
#define US_WARMUP_EPOCHS     3
#define US_K_PERSISTENT      0
#define US_COLD_EPOCHS       10
#define US_VAR_THRESHOLD_INT 25

static inline uint8_t us_clamp_u8(int64_t x) { if (x < 0) return 0; if (x > 255) return 255; return (uint8_t)x; }
static inline uint8_t us_clamp_relu_u8(int64_t x, int32_t zp) {
    if (x < (int64_t)zp) return (uint8_t)zp;
    if (x > 255) return 255;
    return (uint8_t)x;
}
static inline int64_t us_mul_shift_rnd(int64_t acc, int32_t M, int32_t sh) {
    int64_t t = (M != 0) ? (acc * (int64_t)M) : acc;
    if (sh < 0) sh = 0; if (sh > 63) sh = 63;
    if (sh > 0) { int64_t add = (t >= 0) ? ((int64_t)1 << (sh - 1)) : -((int64_t)1 << (sh - 1)); t += add; t >>= sh; }
    return t;
}
static inline uint8_t us_pack_delta(int32_t d) { if (d > 127) d = 127; if (d < -128) d = -128; return (uint8_t)(d + 128); }
static inline uint8_t us_log_scale(uint32_t val) {
    if (val == 0) return 32; if (val <= 5) return (uint8_t)(32 + val * 4);
    if (val <= 20) return (uint8_t)(52 + (val-5)/2); if (val <= 100) return (uint8_t)(60 + (val-20)/10);
    { uint32_t r = 70u + (val - 100u) / 50u; return (uint8_t)(r > 255u ? 255u : r); }
}
static inline uint16_t us_off_bucket_calc(uint64_t page) { return (uint16_t)(((page >> PAGE_SHIFT) & 511u) & (OFF_BUCKETS - 1u)); }

// cur_tier: 0=unknown, 1=fast, 2=slow -- verbatim from pebs_mlp_kern.c
static int us_check_heuristic_filter(struct feature_row *row, uint8_t cur_tier, uint32_t cd_epochs) {
    if (cd_epochs != 0 && row->last_mig_epoch != 0 && row->epoch > row->last_mig_epoch && (row->epoch - row->last_mig_epoch) < cd_epochs) return -3;
    int persistent = 1; if (row->hits < HOT_THRESHOLD) persistent = 0;
    for (int i = 0; i < US_K_PERSISTENT - 1; i++) { if (row->hits_prev[i] < HOT_THRESHOLD) persistent = 0; }
    if (persistent) {
        if (cur_tier == 1) return -4;
        return 1;
    }
    if (row->epoch >= US_WARMUP_EPOCHS) {
        int cold = 1; if (row->hits > 0) cold = 0;
        for (int i = 0; i < US_COLD_EPOCHS - 1; i++) { if (row->hits_prev[i] > 0) cold = 0; }
        if (cold) {
            if (cur_tier == 2) return -5;
            return 2;
        }
        uint64_t sum = row->hits; uint64_t sum_sq = (uint64_t)row->hits * row->hits; int N = 1;
        for (int i=0; i<HIST_STEPS; i++) { sum += row->hits_prev[i]; sum_sq += (uint64_t)row->hits_prev[i] * (uint64_t)row->hits_prev[i]; N++; }
        uint64_t variance_scaled = (sum_sq * N - sum * sum) * 100; variance_scaled /= (N * N);
        if (variance_scaled < US_VAR_THRESHOLD_INT) {
            if (cur_tier == 1 && row->hits <= 1) return -1;
            return -2;
        }
    }
    return -1;
}

// ---- INT8 MLP replica: same rows/muls the loader pushes into BPF maps ----
static struct row0 usw_l0[L0_OUT];  static struct mul usw_l0m[L0_OUT];
static struct row1 usw_l1[L1_OUT];  static struct mul usw_l1m[L1_OUT];
static struct rowO usw_lc;          static struct mul usw_lcm;
static struct qparams usw_qp;

static void us_build_weights(void) {
    usw_qp = (struct qparams){ .in_zp0 = L0_IN_ZP[0], .out_zp0 = L0_OUT_ZP[0], .out_zp1 = L1_OUT_ZP[0], .out_zpC = L2_OUT_ZP[0], .cls_thr_q = CLS_THR_QINT8[0] };
    for (int i=0; i<L0_OUT; i++) { usw_l0[i].b = L0_B_INT32[i]; for (int j=0; j<L0_IN; j++) usw_l0[i].w[j] = L0_W[i*L0_IN+j]; get_scale_and_shift((L0_IN_SCALE[0]*L0_W_SCALE[i])/L0_OUT_SCALE[0], &usw_l0m[i].M, &usw_l0m[i].shift); }
    for (int i=0; i<L1_OUT; i++) { usw_l1[i].b = L1_B_INT32[i]; for (int j=0; j<L0_OUT; j++) usw_l1[i].w[j] = L1_W[i*L0_OUT+j]; get_scale_and_shift((L1_IN_SCALE[0]*L1_W_SCALE[i])/L1_OUT_SCALE[0], &usw_l1m[i].M, &usw_l1m[i].shift); }
    usw_lc.b = L2_B_INT32[0]; for (int j=0; j<L1_OUT; j++) usw_lc.w[j] = L2_W[j];
    get_scale_and_shift((L2_IN_SCALE[0]*L2_W_SCALE[0])/L2_OUT_SCALE[0], &usw_lcm.M, &usw_lcm.shift);
}

// Same arithmetic as run_inference_l0/l1 (s64 accumulate, requantize, ReLU
// clamp at the zero point, plain clamp on the head).
static uint8_t us_mlp_forward(const uint8_t in[L0_IN]) {
    uint8_t l0o[L0_OUT], l1o[L1_OUT];
    for (int i = 0; i < L0_OUT; i++) {
        int64_t acc = usw_l0[i].b;
        for (int j = 0; j < L0_IN; j++) acc += (int64_t)((int32_t)in[j] - usw_qp.in_zp0) * usw_l0[i].w[j];
        l0o[i] = us_clamp_relu_u8(us_mul_shift_rnd(acc, usw_l0m[i].M, usw_l0m[i].shift) + usw_qp.out_zp0, usw_qp.out_zp0);
    }
    for (int i = 0; i < L1_OUT; i++) {
        int64_t acc = usw_l1[i].b;
        for (int j = 0; j < L0_OUT; j++) acc += (int64_t)((int32_t)l0o[j] - usw_qp.out_zp0) * usw_l1[i].w[j];
        l1o[i] = us_clamp_relu_u8(us_mul_shift_rnd(acc, usw_l1m[i].M, usw_l1m[i].shift) + usw_qp.out_zp1, usw_qp.out_zp1);
    }
    int64_t accC = usw_lc.b;
    for (int i = 0; i < L1_OUT; i++) accC += (int64_t)((int32_t)l1o[i] - usw_qp.out_zp1) * usw_lc.w[i];
    return us_clamp_u8(us_mul_shift_rnd(accC, usw_lcm.M, usw_lcm.shift) + usw_qp.out_zpC);
}

// ---- shadow page-state table (the userspace mirror of page_state_map) ----
struct us_pstate {
    uint64_t page;          // 0 = empty slot
    uint64_t cur_hits;
    uint32_t hits_prev[HIST_STEPS];
    uint32_t last_epoch_seen;
    uint32_t last_mig_epoch;   // set optimistically at admit; corrected by readback
    uint32_t last_enq_epoch;
    uint32_t owner_pid;
    uint16_t recency_epochs;
    uint8_t  tier;             // 0 unknown, 1 DRAM, 2 CXL
    uint8_t  _pad;
};
#define US_STATE_BITS 22                    // 4M slots ~ the BPF 1M-entry LRU, with headroom
#define US_STATE_SLOTS (1u << US_STATE_BITS)
#define US_PROBE_MAX 32
static struct us_pstate *us_state;          // calloc'd lazily; ~200MB virtual, touched pages only
static uint64_t us_state_evictions;

static inline uint64_t us_hash64(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x;
}
static struct us_pstate *us_state_get(uint64_t page, int create) {
    uint32_t idx = (uint32_t)(us_hash64(page) & (US_STATE_SLOTS - 1));
    uint32_t first = idx;
    for (int p = 0; p < US_PROBE_MAX; p++) {
        struct us_pstate *e = &us_state[idx];
        if (e->page == page) return e;
        if (e->page == 0) {
            if (!create) return NULL;
            memset(e, 0, sizeof(*e));
            e->page = page;
            return e;
        }
        idx = (idx + 1) & (US_STATE_SLOTS - 1);
    }
    if (!create) return NULL;
    // Probe chain full: overwrite the first slot (crude eviction, mirrors the
    // BPF LRU_HASH throwing out an old entry under pressure).
    us_state_evictions++;
    struct us_pstate *e = &us_state[first];
    memset(e, 0, sizeof(*e));
    e->page = page;
    return e;
}

// ---- local VMA metadata (userspace mirror of meta_map, keyed by 2MB region) ----
struct us_meta_ent { uint64_t reg; struct meta_val mv; };
#define US_META_BITS 18
#define US_META_SLOTS (1u << US_META_BITS)
static struct us_meta_ent us_meta[US_META_SLOTS];
static void us_meta_put(uint64_t reg_base, const struct meta_val *mv) {
    uint32_t idx = (uint32_t)(us_hash64(reg_base | 1) & (US_META_SLOTS - 1));
    for (int p = 0; p < US_PROBE_MAX; p++) {
        struct us_meta_ent *e = &us_meta[idx];
        if (e->reg == reg_base || e->reg == 0) { e->reg = reg_base; e->mv = *mv; return; }
        idx = (idx + 1) & (US_META_SLOTS - 1);
    }
    us_meta[(uint32_t)(us_hash64(reg_base | 1) & (US_META_SLOTS - 1))] = (struct us_meta_ent){ .reg = reg_base, .mv = *mv };
}
static void us_meta_put_fwd(uint64_t reg_base, const struct meta_val *mv) {
    if (us_mode) us_meta_put(reg_base, mv);
}
static const struct meta_val *us_meta_get(uint64_t page) {
    uint64_t reg = page & ~((uint64_t)((1ull << REG_SHIFT) - 1ull));
    uint32_t idx = (uint32_t)(us_hash64(reg | 1) & (US_META_SLOTS - 1));
    for (int p = 0; p < US_PROBE_MAX; p++) {
        struct us_meta_ent *e = &us_meta[idx];
        if (e->reg == reg) return &e->mv;
        if (e->reg == 0) return NULL;
        idx = (idx + 1) & (US_META_SLOTS - 1);
    }
    return NULL;
}

// ---- per-epoch working set + candidate batch ----
static uint64_t us_epoch_pages[MAX_PAGES_PER_EPOCH];
static uint32_t us_epoch_cnt;
static struct region_entry us_cands[MAX_PAGES_PER_EPOCH];
static uint64_t us_hist[SCORE_HIST_BUCKETS];
static uint32_t us_thresh_dyn = 0;   // 0 = static fallback (thresh 1), same as BPF
static uint32_t us_cur_epoch = 1;

// recent-admit ring for the 1 Hz move_pages tier readback (rotating cursor)
#define US_ADMIT_RING 65536
static uint64_t us_admit_pages[US_ADMIT_RING];
static uint32_t us_admit_head;
static uint32_t us_admit_rb_cursor;
// tier-verification queue: samples that land on shadow-tier-1 pages nominate
// them (subsampled) for ground-truth re-resolution. This is how the userspace
// agent finds out the kernel cold scanner demoted a page it thought was on
// DRAM -- the in-BPF path gets that feedback for free via page_state_map.
#define US_VERIFY_RING 16384
static uint64_t us_verify_pages[US_VERIFY_RING];
static uint32_t us_verify_head, us_verify_tail;

// cumulative counters (reported per tick + at exit)
static uint64_t us_samples_rx, us_inferences, us_admits, us_epochs_done;
static uint64_t us_proc_ns_total, us_proc_ns_max;
static uint64_t us_enq_syscalls;

static uint32_t us_rng_state = 0x9E3779B9u;
static inline uint32_t us_prandom(void) {  // xorshift32; same role as bpf_get_prandom_u32
    uint32_t x = us_rng_state; x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return us_rng_state = x;
}

// libbpf ringbuf plumbing + BPF fds used by the userspace policy
static struct ring_buffer *us_rb;
static int us_fd_cand = -1, us_fd_cand_cnt = -1, us_fd_enq_prog = -1;

// Per-sample handler: mirrors on_pebs_stage0's state tracking exactly, minus
// the pieces that only make sense in-kernel (fp bloom ring, feature_row_map).
static int us_sample_cb(void *ctx, void *data, size_t sz) {
    (void)ctx;
    if (sz < sizeof(struct us_sample)) return 0;
    const struct us_sample *smp = (const struct us_sample *)data;
    us_samples_rx++;

    uint64_t page = smp->page;
    // page-type early skip (stack pinned to DRAM, exec pinned to CXL) --
    // stage0 does this in BPF mode; the forwarder is dumb so we do it here.
    const struct meta_val *mv = us_meta_get(page);
    if (mv && (mv->page_type == 3 || mv->page_type == 1)) return 0;

    struct us_pstate *st = us_state_get(page, 1);
    // roll_state_if_needed, verbatim semantics
    if (st->last_epoch_seen != us_cur_epoch) {
        if (st->cur_hits == 0 && st->last_epoch_seen != 0) {
            uint32_t gap = us_cur_epoch - st->last_epoch_seen;
            st->recency_epochs = (st->recency_epochs + gap < 0xFFFF) ? st->recency_epochs + gap : 0xFFFF;
        }
        for (int i = HIST_STEPS - 1; i > 0; --i) st->hits_prev[i] = st->hits_prev[i-1];
        st->hits_prev[0] = (uint32_t)st->cur_hits;
        st->cur_hits = 0;
        st->last_epoch_seen = us_cur_epoch;
    }
    st->cur_hits++;
    st->owner_pid = smp->pid;

    if (st->last_enq_epoch == us_cur_epoch) return 0;   // fast path, same as BPF
    if (st->tier == 1) {
        // DRAM-resident early skip -- but our tier is SHADOW state, and the
        // kernel cold scanner may have demoted this page behind our back.
        // Subsample 1/32 of these hits into the verify queue so ground truth
        // catches up (in-BPF gets this from the executor's tier write-back).
        if ((us_prandom() & 31u) == 0 &&
            (us_verify_head - us_verify_tail) < US_VERIFY_RING)
            us_verify_pages[us_verify_head++ & (US_VERIFY_RING - 1)] = page;
        return 0;
    }
    st->last_enq_epoch = us_cur_epoch;
    if (us_epoch_cnt < MAX_PAGES_PER_EPOCH) us_epoch_pages[us_epoch_cnt++] = page;
    return 0;
}

// Push the admitted candidates through the SAME kernel enqueue path.
static void us_push_candidates(uint32_t n) {
    unsigned char pkt[64]; memset(pkt, 0, sizeof(pkt));
    uint32_t k0 = 0;
    for (uint32_t off = 0; off < n; off += US_CAND_MAX) {
        uint32_t cnt = n - off; if (cnt > US_CAND_MAX) cnt = US_CAND_MAX;
        for (uint32_t i = 0; i < cnt; i++) {
            bpf_map_update_elem(us_fd_cand, &i, &us_cands[off + i], BPF_ANY);
            us_enq_syscalls++;
        }
        bpf_map_update_elem(us_fd_cand_cnt, &k0, &cnt, BPF_ANY);
        struct bpf_test_run_opts opts = { .sz = sizeof(opts), .data_in = pkt, .data_size_in = 64, .repeat = 1 };
        bpf_prog_test_run_opts(us_fd_enq_prog, &opts);
        us_enq_syscalls += 2;
    }
}

// The per-epoch decision pass: the userspace twin of run_inference_batch +
// run_inference_l0/l1 + the top-K threshold update.
static void us_process_epoch(uint32_t ep_fin, uint32_t admit_prob_q16,
                             uint32_t cd_epochs, uint64_t budget_pages,
                             FILE *uslog) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Drain whatever is still sitting in the ring so this epoch's decisions
    // see this epoch's samples (the in-BPF path has them by construction).
    ring_buffer__consume(us_rb);

    uint32_t n_cand = 0;
    uint32_t inf_this = 0;
    for (uint32_t i = 0; i < us_epoch_cnt; i++) {
        uint64_t page = us_epoch_pages[i];
        struct us_pstate *st = us_state_get(page, 0);
        if (!st) continue;

        struct feature_row row = { .epoch = ep_fin, .page = page, .pid = st->owner_pid,
                                   .last_mig_epoch = st->last_mig_epoch,
                                   .recency_epochs = st->recency_epochs,
                                   .epochs_since_promotion = st->recency_epochs,
                                   .off_bucket = us_off_bucket_calc(page) };
        if (st->last_epoch_seen == ep_fin) {
            row.hits = (uint32_t)st->cur_hits; row.lat_sum = (uint32_t)st->cur_hits;
            for (int h=0; h<HIST_STEPS; h++) { row.hits_prev[h] = st->hits_prev[h]; row.lat_prev[h] = st->hits_prev[h]; }
        } else if (ep_fin == st->last_epoch_seen + 1) {
            row.hits_prev[0] = (uint32_t)st->cur_hits; row.lat_prev[0] = (uint32_t)st->cur_hits;
            for (int h=1; h<HIST_STEPS; h++) { row.hits_prev[h] = st->hits_prev[h-1]; row.lat_prev[h] = st->hits_prev[h-1]; }
        }
        row.delta_hits[0] = (int32_t)row.hits - (int32_t)row.hits_prev[0];
        for (int d=1; d<HIST_STEPS; d++) row.delta_hits[d] = (int32_t)row.hits_prev[d-1] - (int32_t)row.hits_prev[d];

        const struct meta_val *mv = us_meta_get(page);
        if (mv) { row.perm_r = mv->perm_r; row.perm_w = mv->perm_w; row.perm_x = mv->perm_x;
                  row.page_type = mv->page_type; row.vma_size_bucket = mv->vma_size_bucket;
                  row.anon = mv->anon; row.shared = mv->shared; }

        // page-type pinning (same as run_inference_batch)
        if (mv && mv->page_type == 3) {
            if (st->tier != 1 && n_cand < MAX_PAGES_PER_EPOCH) {
                us_cands[n_cand++] = (struct region_entry){ .addr = page, .pid = st->owner_pid, .score = 999, .pred = 1, .epoch = ep_fin };
                st->tier = 1; st->last_mig_epoch = ep_fin;
            }
            continue;
        }
        if (mv && mv->page_type == 1) continue;

        int heuristic = us_check_heuristic_filter(&row, st->tier, cd_epochs);
        if (heuristic == -3 || heuristic == -4 || heuristic == -5) continue;

        uint8_t input[L0_IN];
        memset(input, 64, sizeof(input));
        input[0] = (uint8_t)(ep_fin & 0xFF);
        input[1] = us_log_scale(row.hits);
        input[2] = us_log_scale(row.hits);
        for (int h = 0; h < HIST_STEPS; h++) {
            input[3+h*3+0] = us_log_scale(row.hits_prev[h]);
            input[3+h*3+1] = us_log_scale(row.hits_prev[h]);
            input[3+h*3+2] = us_pack_delta(row.delta_hits[h]);
        }
        input[21] = us_log_scale((uint32_t)row.recency_epochs);
        input[22] = us_log_scale((uint32_t)row.epochs_since_promotion);
        if (mv) {
            input[23] = mv->anon; input[24] = mv->shared;
            input[27] = mv->perm_r; input[28] = mv->perm_w; input[29] = mv->perm_x;
            input[30] = mv->page_type; input[31] = mv->vma_size_bucket;
        }

        uint8_t cls_u8 = us_mlp_forward(input);
        inf_this++;
        us_hist[cls_u8]++;

        int32_t raw = (int32_t)cls_u8 - usw_qp.out_zpC;
        uint32_t thresh_dyn = (us_thresh_dyn > 0) ? us_thresh_dyn : 1;
        int32_t score = raw - (int32_t)thresh_dyn;
        if (score > 0 && admit_prob_q16 > 0 && admit_prob_q16 < 65536) {
            if ((us_prandom() & 0xFFFF) >= admit_prob_q16) score = 0;
        }

        if (score > 0 && st->tier != 1) {
            if (n_cand < MAX_PAGES_PER_EPOCH) {
                us_cands[n_cand++] = (struct region_entry){ .addr = page, .pid = st->owner_pid, .score = score, .pred = 1, .epoch = ep_fin };
                st->tier = 1; st->last_mig_epoch = ep_fin;   // optimistic; corrected by readback
                us_admit_pages[us_admit_head++ & (US_ADMIT_RING - 1)] = page;
                us_admits++;
            }
        }
    }
    us_inferences += inf_this;

    if (n_cand > 0) us_push_candidates(n_cand);

    // top-K threshold from the local histogram -- identical math to the in-BPF
    // path's topk_compute_threshold (shared core).
    if (budget_pages > 0 && ep_fin >= TOPK_WARMUP_EPOCHS) {
        uint32_t new_thresh = topk_threshold_from_hist(us_hist, budget_pages);
        if (new_thresh > 0) us_thresh_dyn = new_thresh;
    }
    memset(us_hist, 0, sizeof(us_hist));

    clock_gettime(CLOCK_MONOTONIC, &t1);
    uint64_t dur_ns = (uint64_t)(t1.tv_sec - t0.tv_sec) * 1000000000ull + (t1.tv_nsec - t0.tv_nsec);
    us_proc_ns_total += dur_ns;
    if (dur_ns > us_proc_ns_max) us_proc_ns_max = dur_ns;
    us_epochs_done++;

    printf("[USPOL] epoch=%u pages=%u inf=%u admits=%u thresh=%u proc_ms=%.2f\n",
           ep_fin, us_epoch_cnt, inf_this, n_cand, us_thresh_dyn, dur_ns / 1e6);
    if (uslog) {
        fprintf(uslog, "%u,%u,%u,%u,%u,%.3f,%llu,%llu\n",
                ep_fin, us_epoch_cnt, inf_this, n_cand, us_thresh_dyn, dur_ns / 1e6,
                (unsigned long long)us_samples_rx, (unsigned long long)us_admits);
        fflush(uslog);
    }
    us_epoch_cnt = 0;
}

// 1 Hz tier readback: ground-truth the recent admits via move_pages(NULL).
// This corrects the optimistic tier=1 for pages whose migration failed (they
// go back to tier=2 and can be retried after cooldown expires).
static void us_tier_readback(pid_t target_pid) {
    void *pages[RESOLVE_BATCH];
    uint64_t keys[RESOLVE_BATCH];
    int status[RESOLVE_BATCH];
    uint32_t n = 0;
    // 1) drain the verify queue (pages we thought were on DRAM but keep
    //    getting sampled -- candidates for a cold-scanner demotion we missed)
    while (us_verify_tail != us_verify_head && n < RESOLVE_BATCH / 2) {
        uint64_t pg = us_verify_pages[us_verify_tail++ & (US_VERIFY_RING - 1)];
        if (pg) { pages[n] = (void *)(uintptr_t)pg; keys[n] = pg; n++; }
    }
    // 2) rotate a cursor over the recent-admit ring so every admit eventually
    //    gets ground-truthed (failed migrations flip back to tier 2)
    uint32_t total = us_admit_head < US_ADMIT_RING ? us_admit_head : US_ADMIT_RING;
    for (uint32_t scanned = 0; scanned < total && n < RESOLVE_BATCH; scanned++) {
        uint64_t pg = us_admit_pages[us_admit_rb_cursor++ % (total ? total : 1)];
        if (pg) { pages[n] = (void *)(uintptr_t)pg; keys[n] = pg; n++; }
    }
    if (n == 0) return;
    long ret = sys_move_pages(target_pid, n, pages, NULL, status, 0);
    if (ret < 0 && ret != -ENOENT) return;
    for (uint32_t i = 0; i < n; i++) {
        if (status[i] < 0) continue;
        struct us_pstate *st = us_state_get(keys[i], 0);
        if (!st) continue;
        if (status[i] == promote_node_id)      st->tier = 1;
        else if (status[i] == demote_node_id)  st->tier = 2;
    }
}

// -------------------- MAIN --------------------
int main(int argc, char **argv) {
    if (argc < 5) { printf("Usage: %s <pid> <freq> <ep_ms> <collect|evaluate|run>\n", argv[0]); return 1; }
    pid_t target_pid = atoi(argv[1]); int interval_ms = atoi(argv[3]);
    if (!strcmp(argv[4], "collect")) mode = 1;
    else if (!strcmp(argv[4], "evaluate")) mode = 2;
    else if (!strcmp(argv[4], "run") || !strcmp(argv[4], "production")) mode = 3;
    else return 1;

    // Pin the profiler to node 1 CPUs by default so it does not contend
    // with the workload on node 0. Left unpinned, the contention alone cost
    // PR kron_26 an 84% slowdown with zero migrations performed.
    pin_profiler_to_node();

    // Top-K budget gate. XTIER_DRAM_BUDGET_MB=0 or unset falls back to the
    // static PROMOTE_THRESH=1 in BPF. XTIER_TOPK_FRACTION overrides the
    // per-epoch admit fraction (default 0.15 = top 15% of scored pages).
    {
        const char *bstr = getenv("XTIER_DRAM_BUDGET_MB");
        const char *fstr = getenv("XTIER_TOPK_FRACTION");
        if (fstr && *fstr) {
            double v = strtod(fstr, NULL);
            if (v > 0.0 && v < 1.0) topk_admit_fraction = v;
        }
        if (bstr && *bstr) {
            uint64_t mb = (uint64_t)strtoull(bstr, NULL, 10);
            dram_budget_pages = mb * (1024 * 1024) / 4096;  // MB -> 4KB pages
            printf("[*] Top-K budget gate ENABLED: budget=%llu MB (%llu pages), "
                   "per-epoch admit fraction=%.3f\n",
                   (unsigned long long)mb, (unsigned long long)dram_budget_pages,
                   topk_admit_fraction);
        } else {
            printf("[*] Top-K budget gate disabled (XTIER_DRAM_BUDGET_MB unset) -- using static PROMOTE_THRESH\n");
        }
    }
    
    signal(SIGINT, sig_handler); struct rlimit rl = { RLIM_INFINITY, RLIM_INFINITY }; setrlimit(RLIMIT_MEMLOCK, &rl);

    // 1. Force Unpin
    printf("[*] Unpinning old maps...\n");
    unlink("/sys/fs/bpf/mprof_region_queue_map");
    unlink("/sys/fs/bpf/mprof_stats_map");
    unlink("/sys/fs/bpf/mprof_page_state_map");

    // 2. Load
    // Resolved next to this binary: the documented invocation is
    // `src/page_profiler_user ...` from the repository root.
    char objpath[PATH_MAX];
    const char *env_obj = getenv("XTIER_BPF_OBJ");
    if (env_obj && *env_obj) {
        snprintf(objpath, sizeof(objpath), "%s", env_obj);
    } else {
        ssize_t n = readlink("/proc/self/exe", objpath, sizeof(objpath) - 1);
        if (n > 0) {
            objpath[n] = '\0';
            char *slash = strrchr(objpath, '/');
            if (slash && (size_t)(slash - objpath) + sizeof("/pebs_mlp_kern.o") < sizeof(objpath))
                strcpy(slash + 1, "pebs_mlp_kern.o");
            else
                snprintf(objpath, sizeof(objpath), "pebs_mlp_kern.o");
        } else {
            snprintf(objpath, sizeof(objpath), "pebs_mlp_kern.o");
        }
    }
    struct bpf_object *obj = bpf_object__open_file(objpath, NULL);
    if (!obj || bpf_object__load(obj)) {
        fprintf(stderr, "ERR: load failed (%s)\n", objpath);
        return 1;
    }

    // 3. Pin & Permission Fix
    int fd_q = bpf_object__find_map_fd_by_name(obj, "region_queue_map");
    if (fd_q >= 0) {
        if (bpf_obj_pin(fd_q, "/sys/fs/bpf/mprof_region_queue_map") != 0) { perror("pin"); return 1; }
        chmod("/sys/fs/bpf/mprof_region_queue_map", 0666); 
        printf("[+] Region Queue pinned & chmod 0666 (fd=%d)\n", fd_q);
    }

    int fd_stats = bpf_object__find_map_fd_by_name(obj, "stats_map");
    if (fd_stats >= 0) {
        bpf_obj_pin(fd_stats, "/sys/fs/bpf/mprof_stats_map");
        chmod("/sys/fs/bpf/mprof_stats_map", 0666);
    }
    
    int fd_state = bpf_object__find_map_fd_by_name(obj, "page_state_map");
    if (fd_state >= 0) {
        bpf_obj_pin(fd_state, "/sys/fs/bpf/mprof_page_state_map");
        chmod("/sys/fs/bpf/mprof_page_state_map", 0666);
    }

    // Pin config_map so executor can read exec_sleep flag
    int fd_tenant = bpf_object__find_map_fd_by_name(obj, "tenant_cfg_map");
    int fd_cfg_pin = bpf_object__find_map_fd_by_name(obj, "config_map");
    if (fd_cfg_pin >= 0) {
        if (fd_tenant >= 0) {
            unlink("/sys/fs/bpf/mprof_tenant_cfg_map");
            bpf_obj_pin(fd_tenant, "/sys/fs/bpf/mprof_tenant_cfg_map");
            chmod("/sys/fs/bpf/mprof_tenant_cfg_map", 0666);
        }
        unlink("/sys/fs/bpf/mprof_config_map");
        bpf_obj_pin(fd_cfg_pin, "/sys/fs/bpf/mprof_config_map");
        chmod("/sys/fs/bpf/mprof_config_map", 0666);
    }

    load_model_weights(obj);
    setup_tail_calls(obj);

    int fd_cfg = bpf_object__find_map_fd_by_name(obj, "config_map"), fd_feat = bpf_object__find_map_fd_by_name(obj, "feature_row_map"), fd_prog = bpf_object__find_map_fd_by_name(obj, "batch_progress"), fd_cnt0 = bpf_object__find_map_fd_by_name(obj, "count_0"), fd_cnt1 = bpf_object__find_map_fd_by_name(obj, "count_1"), fd_pgs0 = bpf_object__find_map_fd_by_name(obj, "epoch_pages_0"), fd_pgs1 = bpf_object__find_map_fd_by_name(obj, "epoch_pages_1"), fd_pred = bpf_object__find_map_fd_by_name(obj, "pred_map"), fd_pid = bpf_object__find_map_fd_by_name(obj, "target_pids"), fd_meta = bpf_object__find_map_fd_by_name(obj, "meta_map");
    int fdb = bpf_program__fd(bpf_object__find_program_by_name(obj, "run_inference_batch"));
    int stage0_fd = bpf_program__fd(bpf_object__find_program_by_name(obj, "on_pebs_stage0"));
    // Top-K histograms (double-buffered)
    int fd_hist0 = bpf_object__find_map_fd_by_name(obj, "score_hist_0");
    int fd_hist1 = bpf_object__find_map_fd_by_name(obj, "score_hist_1");
    if (dram_budget_pages > 0 && (fd_hist0 < 0 || fd_hist1 < 0)) {
        fprintf(stderr, "WARN: top-K enabled but score_hist_{0,1} maps not found -- disabling\n");
        dram_budget_pages = 0;
    }

    __u32 admit_prob_q16_init = 0;
    if (dram_budget_pages > 0 && topk_admit_fraction > 0.0 && topk_admit_fraction < 1.0) {
        admit_prob_q16_init = (__u32)(topk_admit_fraction * 65536.0);
        if (admit_prob_q16_init < 1) admit_prob_q16_init = 1;
        if (admit_prob_q16_init >= 65536) admit_prob_q16_init = 0; // 0 = disabled
        printf("[*] Probabilistic admission gate: admit_prob_q16=%u (~%.1f%% pass)\n",
               admit_prob_q16_init, admit_prob_q16_init * 100.0 / 65536.0);
    }
    /* Scorer ablation. XTIER_SCORER selects the feature->score function;
     * every gate downstream is unchanged, so differences isolate ranking
     * quality. Default (unset) is the shipping MLP. */
    __u32 scorer_id_init = 0;
    {
        const char *sc = getenv("XTIER_SCORER");
        static const char *names[] = { "mlp", "freq", "recency", "ewma", "random", "oracle" };
        if (sc && *sc) {
            int found = -1;
            for (int i = 0; i < (int)(sizeof(names)/sizeof(names[0])); i++)
                if (!strcmp(sc, names[i])) { found = i; break; }
            if (found < 0) {
                fprintf(stderr, "[!] XTIER_SCORER='%s' unrecognized; valid: mlp freq recency ewma random oracle\n", sc);
                return 1;
            }
            scorer_id_init = (__u32)found;
        }
        printf("[*] Scorer: %s (id=%u)%s\n", names[scorer_id_init], scorer_id_init,
               scorer_id_init == 0 ? "" : "  [ABLATION -- MLP output discarded, cost retained]");
    }
    __u32 cooldown_epochs_init = 15;   /* == COOLDOWN_EPOCHS default */
    { const char *ce = getenv("XTIER_COOLDOWN_EPOCHS");
      if (ce && *ce) { cooldown_epochs_init = (__u32)atoi(ce);
          printf("[*] Cooldown epochs: %u%s\n", cooldown_epochs_init,
                 cooldown_epochs_init == 0 ? "  [FILTER OFF]" : ""); } }
    /* Userspace-policy counterfactual. XTIER_USERSPACE_POLICY=1 runs the
     * identical policy (same INT8 MLP, filters, gates) in userspace on a
     * forwarded sample stream; BPF degrades to a plain forwarder. */
    { const char *up = getenv("XTIER_USERSPACE_POLICY");
      us_mode = (up && *up && strcmp(up, "0")) ? 1 : 0;
      if (us_mode) {
          if (scorer_id_init != 0) {
              fprintf(stderr, "[!] XTIER_USERSPACE_POLICY supports the mlp scorer only\n");
              return 1;
          }
          if (mode != 3) {
              fprintf(stderr, "[!] XTIER_USERSPACE_POLICY requires run/production mode\n");
              return 1;
          }
          printf("[*] USERSPACE POLICY MODE: BPF = dumb sample forwarder; "
                 "filter+MLP+gates run in userspace; enqueue via userspace_enqueue\n");
      } }
    /* Closed-loop admit controller: when
     * XTIER_TARGET_ADMIT_PCT is set, admit_prob_q16 is adjusted each epoch by
     * feedback so admitted/inferences converges to the target REGARDLESS of the
     * scorer's score distribution. Rationale: the threshold
     * and the dice COMPOUND differently per scorer -- a uniform (random) scorer
     * pays both gates (0.15*0.15=2.3%) while a coarse-bucketed model passes the
     * threshold wholesale and pays only the dice (14%). That 6x effective-
     * budget asymmetry is what made the model "do more migrations". With a
     * common target both spend the same small budget and differ only in
     * WHICH pages they pick. */
    double target_admit = -1.0;
    { const char *ta = getenv("XTIER_TARGET_ADMIT_PCT");
      if (ta && *ta) { target_admit = atof(ta) / 100.0;
          printf("[*] Closed-loop admit controller: target=%.2f%% of inferences\n", target_admit*100.0); } }
    __u32 k0 = 0, zero = 0; struct config_val cfg = { .pid = target_pid, .cur_epoch = 1, .mode = mode, .promote_threshold_dyn = 0, .active_hist_idx = 0, .admit_prob_q16 = admit_prob_q16_init, .scorer_id = scorer_id_init, .cooldown_epochs = cooldown_epochs_init, .userspace_policy = (__u32)us_mode }; bpf_map_update_elem(fd_cfg, &k0, &cfg, 0);

    // Userspace-policy setup: shadow state, weight replica, ringbuf consumer,
    // candidate maps + enqueue program fds, per-epoch CSV.
    FILE *uslog = NULL;
    if (us_mode) {
        us_state = calloc(US_STATE_SLOTS, sizeof(struct us_pstate));
        if (!us_state) { fprintf(stderr, "ERR: us_state alloc failed\n"); return 1; }
        us_build_weights();
        int fd_usring = bpf_object__find_map_fd_by_name(obj, "us_ring");
        us_fd_cand     = bpf_object__find_map_fd_by_name(obj, "us_cand");
        us_fd_cand_cnt = bpf_object__find_map_fd_by_name(obj, "us_cand_cnt");
        struct bpf_program *enq_prog = bpf_object__find_program_by_name(obj, "userspace_enqueue");
        us_fd_enq_prog = enq_prog ? bpf_program__fd(enq_prog) : -1;
        if (fd_usring < 0 || us_fd_cand < 0 || us_fd_cand_cnt < 0 || us_fd_enq_prog < 0) {
            fprintf(stderr, "ERR: userspace-policy maps/prog missing (ring=%d cand=%d cnt=%d prog=%d)\n",
                    fd_usring, us_fd_cand, us_fd_cand_cnt, us_fd_enq_prog);
            return 1;
        }
        us_rb = ring_buffer__new(fd_usring, us_sample_cb, NULL, NULL);
        if (!us_rb) { fprintf(stderr, "ERR: ring_buffer__new failed\n"); return 1; }
        const char *uslog_path = getenv("XTIER_USPOL_CSV");
        if (!uslog_path) uslog_path = "us_policy.csv";
        uslog = fopen(uslog_path, "w");
        if (uslog) fprintf(uslog, "epoch,pages,inferences,admits,thresh_dyn,proc_ms,samples_rx_cum,admits_cum\n");
        printf("[+] Userspace policy engine ready (zpC=%d, budget_pages=%llu)\n",
               usw_qp.out_zpC, (unsigned long long)dram_budget_pages);
    }
    // Multi-tenancy: XTIER_TENANTS overrides the single argv[1] target. Every
    // tenant is registered with its own slot before PEBS attach so no sample
    // can land with an unassigned slot.
    if (parse_tenants(dram_budget_pages) > 0) {
        printf("[*] MULTI-TENANT static partition: %d tenants\n", n_tenants);
        for (int i = 0; i < n_tenants; i++) {
            add_process_tree(fd_pid, fd_meta, (int)tenants[i].pid, tenants[i].slot);
            printf("      slot %u  pid %d  budget %llu pages (%llu MB)\n",
                   tenants[i].slot, (int)tenants[i].pid,
                   (unsigned long long)tenants[i].budget_pages,
                   (unsigned long long)(tenants[i].budget_pages / 256));
        }
    } else {
        tenants[0].pid = target_pid; tenants[0].slot = 0;
        tenants[0].budget_pages = dram_budget_pages;
        n_tenants = 1;
        add_process_tree(fd_pid, fd_meta, (int)target_pid, 0);
    }

    // PEBS event: MEM_UOPS_RETIRED.ALL_LOADS (0xD0 umask 0x81).
    // Per Intel Haswell SDM Vol 3B Table 19-9: 0xD0 is MEM_UOPS_RETIRED (precise,
    // ALL_LOADS umask=0x81); 0xD1 is MEM_LOAD_UOPS_RETIRED (umasks 0x01..0x40
    // only -- 0x81 is undefined there and silently mis-decoded on Haswell-EP).
    uint64_t pebs_event = 0x81D0;  // MEM_UOPS_RETIRED.ALL_LOADS
    const char *event_name = "MEM_UOPS_RETIRED.ALL_LOADS (0x81D0)";
    // Override knob: XTIER_PEBS_EVENT lets us A/B-test event codes without
    // recompiling. Format: hex like "0x81D1". Pre-fix code used 0x81D1.
    const char *env_event = getenv("XTIER_PEBS_EVENT");
    if (env_event && env_event[0] != '\0') {
        uint64_t v = strtoull(env_event, NULL, 0);
        if (v) { pebs_event = v; event_name = env_event; }
    }

    // Frequency mode: kernel auto-adjusts the period via its PID controller to
    // maintain ~sample_freq samples/sec/event, avoiding the per-event throttle
    // that fixed-period mode trips on dense workloads (max_samples_per_tick
    // = perf_event_max_sample_rate/HZ = 100000/250 = 400 on this kernel).
    int pebs_freq = 2000;
    const char *env_freq = getenv("XTIER_PEBS_SAMPLE_FREQ");
    if (env_freq && atoi(env_freq) > 0) pebs_freq = atoi(env_freq);

    struct perf_event_attr pe = {
        .type = PERF_TYPE_RAW,
        .size = sizeof(pe),
        .config = pebs_event,
        .freq = 1,
        .sample_freq = pebs_freq,
        .sample_type = PERF_SAMPLE_ADDR,
        .precise_ip = 2,               // request PEBS
        .exclude_kernel = 1,
        .disabled = 1,
    };

    printf("[*] PEBS event: %s, freq=%d samples/sec (argv[2]=%s ignored in freq mode)\n",
           event_name, pebs_freq, argv[2]);

    // Determine the effective event config by probing on the target PID.
    // Per-task attachment (pid=target_pid, cpu=-1) fires only for target process,
    // eliminating NMI overhead from all other processes on the system.
    {
        int probe_fd = perf_event_open_wrap(&pe, target_pid, -1, -1, PERF_FLAG_FD_CLOEXEC);
        if (probe_fd < 0 && (errno == EINVAL || errno == ENOENT)) {
            // Fallback: try 0x81D1 (old event, undefined umask but historically
            // accepted by the kernel even if semantics differ)
            uint64_t fallback = 0x81D1;
            pe.config = fallback;
            probe_fd = perf_event_open_wrap(&pe, target_pid, -1, -1, PERF_FLAG_FD_CLOEXEC);
            if (probe_fd >= 0) printf("[*] Fallback: using event 0x%lX\n", (unsigned long)fallback);
        }
        if (probe_fd < 0) {
            fprintf(stderr, "WARN: PEBS memory events not available (errno=%d), falling back to HW_CACHE_MISSES.\n", errno);
            pe.type = PERF_TYPE_HARDWARE;
            pe.config = PERF_COUNT_HW_CACHE_MISSES;
            probe_fd = perf_event_open_wrap(&pe, target_pid, -1, -1, PERF_FLAG_FD_CLOEXEC);
        }
        if (probe_fd < 0) {
            fprintf(stderr, "ERR: Could not open any perf event for PID %d (errno=%d)\n", (int)target_pid, errno);
            return 1;
        }
        close(probe_fd); // re-opened below via open_perf_for_tree
    }
    // Open one perf event per PID in the process tree (per-task, not system-wide)
    // Every tenant needs its own PEBS attachment: a pid registered in
    // target_pids but never attached would be tracked in the maps and
    // never sampled, which looks exactly like an idle workload.
    for (int ti = 0; ti < n_tenants; ti++)
        open_perf_for_tree(tenants[ti].pid, stage0_fd, &pe);
    printf("[+] Opened per-task PEBS for %d PID(s)\n", pid_perf_count);

    // Open CSV only for modes that emit one (mode 3 runs the full pipeline
    // with no CSV output). Paths can be overridden via XTIER_FEATURES_CSV and
    // XTIER_EVAL_CSV; both default to the cwd.
    FILE *csv = NULL;
    if (mode == 1) {
        const char *p = getenv("XTIER_FEATURES_CSV");
        csv = fopen(p ? p : "features.csv", "w");
        if (!csv) { perror("features csv"); return 1; }
    } else if (mode == 2) {
        const char *p = getenv("XTIER_EVAL_CSV");
        csv = fopen(p ? p : "eval.csv", "w");
        if (!csv) { perror("eval csv"); return 1; }
    }

    // Per-second DRAM/CXL hit CSV. Reads eval_samples_t1 (DRAM) / t2 (CXL)
    // from BPF stats_map every 1 s wall-clock and writes deltas. Path can be
    // overridden via XTIER_DRAM_HITS_CSV; defaults to dram_hits.csv in cwd.
    const char *dram_csv_path = getenv("XTIER_DRAM_HITS_CSV");
    if (!dram_csv_path) dram_csv_path = "dram_hits.csv";
    FILE *dram_csv = fopen(dram_csv_path, "w");
    if (dram_csv) {
        fprintf(dram_csv, "wall_unix,t_rel_s,dram_hits,cxl_hits,unknown_hits,total_samples,cum_mig_success,cum_mig_fail\n");
        fflush(dram_csv);
    }
    struct timespec dram_t0; clock_gettime(CLOCK_REALTIME, &dram_t0);
    struct timespec dram_last; clock_gettime(CLOCK_MONOTONIC, &dram_last);
    uint64_t dram_last_t1 = 0, dram_last_t2 = 0, dram_last_unk = 0;

    // Top-K hot-page placement CSV.
    // Once per second: walk page_state_map, score each page by sum of recent
    // hit counts (cur_hits + hits_prev[0..5] ~ ~2 s look-back in CRUISE),
    // take the top-K, sys_move_pages to query their actual node, count
    // how many sit on N0 (good) vs N1 (bad) vs other. Output:
    //   wall_unix, t_rel_s, K, hot_on_n0, hot_on_n1, hot_other, sampled_pages
    const char *topk_csv_path = getenv("XTIER_TOPK_CSV");
    if (!topk_csv_path) topk_csv_path = "top_k.csv";
    FILE *topk_csv = fopen(topk_csv_path, "w");
    if (topk_csv) {
        fprintf(topk_csv,
            "wall_unix,t_rel_s,K,hot_on_n0,hot_on_n1,hot_other,sampled_pages,"
            "total_score,dram_score,topk_score,Q,n_mapped\n");
        fflush(topk_csv);
    }
    int topk_k = 10000;
    {
        const char *e = getenv("XTIER_TOPK_K");
        if (e && atoi(e) > 0) topk_k = atoi(e);
    }
    struct timespec topk_last; clock_gettime(CLOCK_MONOTONIC, &topk_last);

    struct timespec last_ep, last_st; clock_gettime(CLOCK_MONOTONIC, &last_ep); clock_gettime(CLOCK_MONOTONIC, &last_st);
    uint64_t last_pebs = 0, last_inf = 0;
    struct batch_ctrl ctrl = {0xDEADBEEF, 0xDEADBEEF};
    int refresh = 0; // REFRESH VARIABLE ADDED HERE

    const char *mode_name = (mode == 1) ? "COLLECT" : (mode == 2) ? "EVALUATE" : (mode == 3) ? "PRODUCTION" : "MONITOR";
    printf("[+] v10 Started. Console Output: V4 Style. Mode: %s\n", mode_name);

    // Initialize adaptive profiling (active for all modes, but only adapts in evaluate/production)
    adapt_init((uint64_t)atoi(argv[2]), interval_ms);
    int last_evict_yield = -1; // -1 = no eviction this epoch

    while (!stop_flag) {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);

        // ---- per-second DRAM/CXL hit emission ----
        // Reads eval_samples_t1 (DRAM-tier) and t2 (CXL-tier) from BPF stats
        // and writes a delta row per wall-clock second to dram_csv.
        // The cheap path (just stats_map read + cum_mig_success column) runs
        // always. The expensive resolver call (walks 1M LRU entries +
        // sys_move_pages over thousands of pages) is gated behind
        // XTIER_HEAVY_INSTRUMENT=1 -- without it, t1/t2/unknown columns will
        // be 0, but cum_mig_success and walltime data are still valid.
        static int heavy_instrument = -1;
        if (heavy_instrument < 0) {
            const char *e = getenv("XTIER_HEAVY_INSTRUMENT");
            heavy_instrument = (e && atoi(e) > 0) ? 1 : 0;
        }
        if (dram_csv) {
            double dt_dram = (now.tv_sec - dram_last.tv_sec) + (now.tv_nsec - dram_last.tv_nsec) / 1e9;
            if (dt_dram >= 1.0) {
                if (heavy_instrument)
                    for (int ti = 0; ti < n_tenants; ti++)
                        resolve_unknown_tiers_slot(fd_state, (int)tenants[ti].pid,
                                                   tenants[ti].slot,
                                                   promote_node_id, demote_node_id);
                if (us_mode)
                    us_tier_readback(target_pid);   // 1 Hz ground-truth correction of optimistic tiers
                struct stats_val s_now;
                if (bpf_map_lookup_elem(fd_stats, &k0, &s_now) == 0) {
                    uint64_t d_t1  = s_now.eval_samples_t1      - dram_last_t1;
                    uint64_t d_t2  = s_now.eval_samples_t2      - dram_last_t2;
                    uint64_t d_unk = s_now.eval_samples_unknown - dram_last_unk;
                    struct timespec wall; clock_gettime(CLOCK_REALTIME, &wall);
                    double t_rel = (wall.tv_sec - dram_t0.tv_sec) + (wall.tv_nsec - dram_t0.tv_nsec) / 1e9;
                    fprintf(dram_csv, "%ld.%06ld,%.3f,%llu,%llu,%llu,%llu,%llu,%llu\n",
                            (long)wall.tv_sec, (long)wall.tv_nsec / 1000, t_rel,
                            (unsigned long long)d_t1, (unsigned long long)d_t2,
                            (unsigned long long)d_unk,
                            (unsigned long long)(d_t1 + d_t2 + d_unk),
                            (unsigned long long)s_now.eval_mig_success,
                            (unsigned long long)s_now.eval_mig_fail);
                    fflush(dram_csv);
                    dram_last_t1  = s_now.eval_samples_t1;
                    dram_last_t2  = s_now.eval_samples_t2;
                    dram_last_unk = s_now.eval_samples_unknown;
                }
                dram_last = now;
            }
        }

        // ---- per-second top-K hot-page placement emission ----
        // Heavy: walks page_state_map + sys_move_pages over all sampled
        // pages. Gated behind XTIER_HEAVY_INSTRUMENT=1 to avoid CPU steal
        // from the workload (~1 core/sec).
        if (heavy_instrument && topk_csv) {
            double dt_topk = (now.tv_sec - topk_last.tv_sec) +
                             (now.tv_nsec - topk_last.tv_nsec) / 1e9;
            if (dt_topk >= 1.0) {
                emit_topk_placement(fd_state, target_pid, topk_k,
                                    topk_csv, dram_t0);
                topk_last = now;
            }
        }

        int cur_interval_ms = (mode >= 2) ? adapt_current_epoch_ms() : interval_ms;
        if (((now.tv_sec - last_ep.tv_sec) * 1000.0 + (now.tv_nsec - last_ep.tv_nsec) / 1e6) >= cur_interval_ms) {
            __u32 ep_fin = cfg.cur_epoch; 
            struct batch_ctrl init = { .idx = 0, .epoch = ep_fin }; bpf_map_update_elem(fd_prog, &zero, &init, 0);
            int fd_c = (ep_fin % 2 == 0) ? fd_cnt0 : fd_cnt1, fd_p = (ep_fin % 2 == 0) ? fd_pgs0 : fd_pgs1;
            __u32 n = 0; bpf_map_lookup_elem(fd_c, &k0, &n); 
            unsigned char pkt[64]; memset(pkt, 0, 64); memcpy(pkt, &ep_fin, 4);
            struct bpf_test_run_opts opts = { .sz = sizeof(opts), .data_in = pkt, .data_size_in = 64, .repeat = 1 };

            if (us_mode) {
                // Userspace-policy path: the whole decision pass (features,
                // filter, MLP, gates, enqueue) runs here instead of the pump.
                us_process_epoch(ep_fin, cfg.admit_prob_q16, cfg.cooldown_epochs,
                                 dram_budget_pages, uslog);
                n = 0;   // skip the BPF-side drain below
            } else {
            // --- BATCH PUMP ---
            // Timed: this is the in-BPF path's per-epoch decision latency
            // (feature build + MLP + gates + enqueue for every epoch page),
            // directly comparable to [USPOL] proc_ms in userspace mode.
            struct timespec pump_t0, pump_t1;
            clock_gettime(CLOCK_MONOTONIC, &pump_t0);
            uint32_t last_idx = 0xFFFFFFFF; int stalls = 0;
            for (int pump = 0; pump < 4000; pump++) {
                if (bpf_prog_test_run_opts(fdb, &opts)) break; bpf_map_lookup_elem(fd_prog, &zero, &ctrl);
                if (ctrl.idx == 0 && pump > 0) break;
                if (ctrl.idx == last_idx) { if (++stalls > 20) break; } else { last_idx = ctrl.idx; stalls = 0; }
            }
            clock_gettime(CLOCK_MONOTONIC, &pump_t1);
            if (mode >= 2) {
                double pump_ms = (pump_t1.tv_sec - pump_t0.tv_sec) * 1e3 +
                                 (pump_t1.tv_nsec - pump_t0.tv_nsec) / 1e6;
                printf("[PUMP] epoch=%u pages=%u pump_ms=%.2f\n", ep_fin, n, pump_ms);
            }
            }

            if (n > MAX_PAGES_PER_EPOCH) n = MAX_PAGES_PER_EPOCH;
            for (__u32 i=0; i<n; i++) {
                __u64 pg; bpf_map_lookup_elem(fd_p, &i, &pg); struct feature_key fkey = {.epoch = ep_fin, .page = pg}; struct feature_row row;
                if (bpf_map_lookup_elem(fd_feat, &fkey, &row) == 0) {
                    if (mode == 1) emit_feature_row_csv(csv, &row);
                    else { struct pred_row pr={0,0}; struct bpf_page_state st={0}; bpf_map_lookup_elem(fd_pred,&fkey,&pr); bpf_map_lookup_elem(fd_state,&pg,&st); emit_eval_row(csv, &row, &pr, &st); bpf_map_delete_elem(fd_pred, &fkey); }
                    bpf_map_delete_elem(fd_feat, &fkey);
                }
            }

            // --- COLD PAGE SAMPLING (collect mode only) ---
            // Sample pages from page_state_map that were NOT accessed this epoch.
            // These become negative training examples so the model learns what cold looks like.
            if (mode == 1 && csv && n > 0) {
                __u64 cold_key = 0, next_key = 0;
                int cold_sampled = 0;
                int cold_target = (int)n; // Emit roughly 1:1 hot:cold ratio
                if (cold_target > 2048) cold_target = 2048; // Cap to avoid slow epochs
                int skip_counter = 0;
                // Walk page_state_map using get_next_key, emit pages with zero hits this epoch
                if (bpf_map_get_next_key(fd_state, NULL, &next_key) == 0) {
                    cold_key = next_key;
                    while (cold_sampled < cold_target) {
                        struct bpf_page_state pst = {0};
                        if (bpf_map_lookup_elem(fd_state, &cold_key, &pst) == 0) {
                            // Only emit if this page was NOT accessed this epoch (last_epoch_seen != current)
                            if (pst.last_epoch_seen != ep_fin && pst.last_epoch_seen > 0) {
                                // Subsample: skip some to spread across address space
                                if (++skip_counter % 3 == 0) {
                                    // Build a synthetic feature row for this cold page
                                    struct feature_row crow = {0};
                                    crow.epoch = ep_fin;
                                    crow.page = cold_key;
                                    crow.hits = 0;
                                    crow.lat_sum = 0;
                                    crow.pid = pst.owner_pid;
                                    crow.recency_epochs = pst.recency_epochs;
                                    crow.epochs_since_promotion = pst.recency_epochs;
                                    // Copy history from page state
                                    for (int h = 0; h < 6; h++) {
                                        crow.hits_prev[h] = pst.hits_prev[h];
                                        crow.lat_prev[h] = pst.lat_prev[h];
                                    }
                                    crow.delta_hits[0] = -(int32_t)pst.hits_prev[0];
                                    for (int d = 1; d < 6; d++)
                                        crow.delta_hits[d] = (int32_t)pst.hits_prev[d-1] - (int32_t)pst.hits_prev[d];
                                    // Lookup VMA metadata
                                    __u64 reg_base = cold_key & ~((__u64)((1ULL << REG_SHIFT) - 1ULL));
                                    struct meta_val mv = {0};
                                    if (bpf_map_lookup_elem(fd_meta, &reg_base, &mv) == 0) {
                                        crow.perm_r = mv.perm_r;
                                        crow.perm_w = mv.perm_w;
                                        crow.perm_x = mv.perm_x;
                                        crow.page_type = mv.page_type;
                                        crow.vma_size_bucket = mv.vma_size_bucket;
                                        crow.anon = mv.anon;
                                        crow.shared = mv.shared;
                                    }
                                    emit_feature_row_csv(csv, &crow);
                                    cold_sampled++;
                                }
                            }
                        }
                        if (bpf_map_get_next_key(fd_state, &cold_key, &next_key) != 0) break;
                        cold_key = next_key;
                    }
                }
            }
            // ---- PROMOTION SCAN + BALANCED EVICTION ----
            if (mode >= 2 && ++migration_stride_ctr >= EVICT_STRIDE) {
                migration_stride_ctr = 0;

                // Check warmup: skip all migration for first WARMUP_SKIP_SECS
                struct timespec warmup_now;
                clock_gettime(CLOCK_MONOTONIC, &warmup_now);
                static struct timespec warmup_start = {0, 0};
                if (warmup_start.tv_sec == 0) warmup_start = warmup_now;
                double elapsed_secs = (warmup_now.tv_sec - warmup_start.tv_sec) +
                                      (warmup_now.tv_nsec - warmup_start.tv_nsec) / 1e9;

                if (elapsed_secs < WARMUP_SKIP_SECS) {
                    // During warmup: no promotion, no eviction. Preserve initial placement.
                } else {
                    // -----------------------------------------------------------------
                    // Userspace map-walking scanners are DISABLED.
                    //
                    // resolve_unknown_tiers, promote_hot_from_cxl, and
                    // evict_coldest_from_dram each walked the entire page_state_map
                    // (now 1M LRU entries) via per-key bpf_map_get_next_key +
                    // bpf_map_lookup_elem syscalls. That cost ~1s of profiler CPU
                    // per scan call, with 3 scans per 500ms -- pinning the profiler
                    // at >100% of one core and stealing scheduling time from the
                    // workload's OpenMP threads (since both run with --cpunodebind=0).
                    //
                    // Their work is now covered by other paths:
                    //   - Hot-CXL identification: BPF stage1 heuristic enqueues every
                    //     CXL page that gets any PEBS sample (HOT_THRESHOLD=1).
                    //   - Cold-DRAM identification: kernel-side PTE A-bit scanner.
                    //   - tier=0 -> tier=1 resolution: executor sets tier on
                    //     successful migration; with --preferred=1 every page starts
                    //     on CXL so this covers the entire working set.
                    // -----------------------------------------------------------------
                    int n_promoted = 0;
                    adapt.last_promo_scan_found = 0;
                    adapt.low_promo_scan_streak++;
                    adapt.zero_promo_scan_streak++;
                    last_evict_yield = -1;

                    /* Old code, kept for reference (compiled out):
                    int n_resolved = resolve_unknown_tiers(...);
                    n_promoted = promote_hot_from_cxl(...);
                    int n_evicted = evict_coldest_from_dram(...);
                    */
                    if (0) {  /* compiled out -- see comment above */
                    long node0_total = read_node_total_mb(0);
                    long node0_free  = read_node_free_mb(0);
                    long threshold = (node0_total > 0) ? node0_total / 100 : DRAM_PRESSURE_THRESHOLD_MB;

                    int skip_evict = 0;
                    if (rolling_demotes > rolling_promotes * 2 + 100) skip_evict = 1;
                    if (node0_free >= threshold) skip_evict = 1;

                    if (!skip_evict) {
                        int evict_target = n_promoted + 16;
                        if (evict_target > EVICT_BATCH_SIZE) evict_target = EVICT_BATCH_SIZE;

                        int n_evicted = evict_coldest_from_dram(fd_state, fd_q, fd_meta, ep_fin, evict_target);
                        last_evict_yield = n_evicted;
                        if (n_evicted > 0) {
                            total_evict_batches++;
                            total_evict_pages += (uint64_t)n_evicted;
                            rolling_demotes += (uint32_t)n_evicted;
                            printf("[EVICT] %d cold DRAM pages (room for %d promotions) "
                                   "free=%ldMB balance: P=%u D=%u\n",
                                   n_evicted, n_promoted, node0_free,
                                   rolling_promotes, rolling_demotes);
                        }
                    }
                    }  /* close if (0) compiled-out block */
                }
            }
            // --- ADAPTIVE PROFILING UPDATE ---
            // Only in production mode (3); evaluate mode (2) runs at full rate for data quality
            if (mode >= 3) {
                // Read ring tail to compute enqueue rate
                struct stats_val adapt_s;
                uint32_t enq_this_epoch = 0;
                uint64_t cur_flips = 0;
                if (bpf_map_lookup_elem(fd_stats, &k0, &adapt_s) == 0) {
                    uint64_t cur_tail = adapt_s.xt_ring_tail;
                    if (adapt.last_ring_tail > 0)
                        enq_this_epoch = (uint32_t)(cur_tail - adapt.last_ring_tail);
                    adapt.last_ring_tail = cur_tail;
                    cur_flips = adapt_s.eval_tier_flips;
                }
                adapt_update(ep_fin, enq_this_epoch, last_evict_yield, cur_flips);
                last_evict_yield = -1; // reset until next eviction cycle
            }

            if (csv) fflush(csv); bpf_map_update_elem(fd_c, &k0, &zero, 0); cfg.cur_epoch = ep_fin + 1;
            // Signal executor: only deep sleep in MONITOR (now disabled). CRUISE keeps
            // the worker responsive so promotions/demotions still execute promptly.
            cfg.exec_sleep = (adapt.state == ADAPT_MONITOR) ? 1 : 0;

            // Top-K budget gate: flip the active histogram, let in-flight
            // inferences settle, snapshot & zero the now-inactive buffer,
            // compute the Kth-percentile threshold, publish to BPF.
            if (!us_mode && dram_budget_pages > 0 && ep_fin >= TOPK_WARMUP_EPOCHS) {
                uint32_t old_idx = cfg.active_hist_idx;
                uint32_t new_idx = old_idx ^ 1;
                cfg.active_hist_idx = new_idx;
                bpf_map_update_elem(fd_cfg, &k0, &cfg, 0);
                // Settle: a single inference takes ~60 us; 10 ms is ~160x headroom.
                struct timespec ts = { .tv_sec = 0, .tv_nsec = 10 * 1000 * 1000 };
                nanosleep(&ts, NULL);
                int fd_inactive = (old_idx == 0) ? fd_hist0 : fd_hist1;
                // STATIC PARTITION: each tenant's threshold comes from its own
                // histogram slice against its own budget. Nothing is shared, so
                // a noisy tenant cannot raise the bar for a quiet one.
                for (int ti = 0; ti < n_tenants; ti++) {
                    uint64_t bp = tenants[ti].budget_pages ? tenants[ti].budget_pages
                                                           : dram_budget_pages;
                    uint32_t th = topk_compute_threshold_slot(fd_inactive, bp, tenants[ti].slot);
                    if (th > 0) tenants[ti].threshold = th;
                    struct tenant_cfg tc = {
                        .pid = (__u32)tenants[ti].pid,
                        .promote_threshold_dyn = tenants[ti].threshold,
                        .admit_prob_q16 = cfg.admit_prob_q16,
                        .budget_pages = bp,
                    };
                    __u32 sk = tenants[ti].slot;
                    if (fd_tenant >= 0) bpf_map_update_elem(fd_tenant, &sk, &tc, BPF_ANY);
                    printf("[TOPK] epoch=%u slot=%u pid=%d thresh_dyn=%u budget_pages=%llu\n",
                           ep_fin, sk, (int)tenants[ti].pid, tenants[ti].threshold,
                           (unsigned long long)bp);
                }
                // Slot 0 also drives the legacy global field so single-tenant
                // runs and any reader of config_map see the same value as before.
                if (tenants[0].threshold > 0) cfg.promote_threshold_dyn = tenants[0].threshold;
            }

            /* closed-loop admit controller (see init above; BPF mode only --
             * userspace mode reads cfg.admit_prob_q16 directly and we don't
             * run the controller there) */
            if (!us_mode && target_admit > 0.0) {
                static uint64_t cl_last_inf = 0, cl_last_pos = 0;
                struct stats_val cs; memset(&cs, 0, sizeof(cs));
                if (bpf_map_lookup_elem(fd_stats, &k0, &cs) == 0) {
                    uint64_t d_inf = cs.total_inferences - cl_last_inf;
                    uint64_t d_pos = cs.ml_score_pos   - cl_last_pos;
                    cl_last_inf = cs.total_inferences; cl_last_pos = cs.ml_score_pos;
                    if (d_inf > 100) {
                        double rate = (double)d_pos / (double)d_inf;
                        double cur  = (cfg.admit_prob_q16 ? cfg.admit_prob_q16 : 65536) / 65536.0;
                        double next = cur;
                        if (rate > 1e-9) next = cur * (target_admit / rate);
                        else next = cur * 2.0;                 /* starved: open up */
                        if (next > 1.0) next = 1.0; if (next < 1.0/65536.0) next = 1.0/65536.0;
                        uint32_t q = (uint32_t)(next * 65536.0);
                        if (q >= 65536) q = 65535;
                        cfg.admit_prob_q16 = q ? q : 1;
                    }
                }
            }
            bpf_map_update_elem(fd_cfg, &k0, &cfg, 0); last_ep = now;
            us_cur_epoch = cfg.cur_epoch;   // sample callback stamps into the new epoch
            if (++refresh > 50) {
                for (int ti = 0; ti < n_tenants; ti++)
                    add_process_tree(fd_pid, fd_meta, (int)tenants[ti].pid, tenants[ti].slot);
                // Open perf events for any new child PIDs spawned since last refresh
                for (int ti = 0; ti < n_tenants; ti++)
                    open_perf_for_tree(tenants[ti].pid, stage0_fd, &pe);
                refresh = 0;
            }
        }
        
        // --- CONSOLE OUTPUT (V4 STYLE) ---
        if (((now.tv_sec - last_st.tv_sec) * 1000.0 + (now.tv_nsec - last_st.tv_nsec) / 1e6) >= 2000.0) {
            struct stats_val s; if (bpf_map_lookup_elem(fd_stats, &k0, &s) == 0) {
                double pebs_rate = (s.pebs_seen - last_pebs) / 2.0; 
                uint64_t inf_delta = s.total_inferences - last_inf;
                uint32_t backlog = (uint32_t)(s.xt_ring_tail - s.xt_ring_head);
                
                uint64_t avg_s0 = s.pebs_triggered ? s.t_stage0_accum_ns / s.pebs_triggered : 0;
                uint64_t avg_s1 = s.total_inferences ? s.t_stage1_accum_ns / s.total_inferences : 0;
                uint64_t avg_mig = s.nr_migrations_timed ? s.t_migration_accum_ns / s.nr_migrations_timed : 0;

                printf("[e%u %s] PEBS: %llu (%.1fk/s period=%lu) | Inf: %llu (+%llu) | Promote: %llu/%llu | Demote: %llu/%llu\n",
                    cfg.cur_epoch, adapt_state_name[adapt.state],
                    (unsigned long long)s.pebs_seen, pebs_rate / 1000.0,
                    (unsigned long)adapt_current_pebs_period(),
                    (unsigned long long)s.total_inferences, (unsigned long long)inf_delta,
                    (unsigned long long)s.nr_promote_succ, (unsigned long long)s.nr_promote_att,
                    (unsigned long long)s.nr_demote_succ, (unsigned long long)s.nr_demote_att);

                printf("\tLatency: Stage0(Collect)=%lluns | Stage1(ML)=%lluns | Mig(Kernel)=%lluus\n",
                    (unsigned long long)avg_s0, (unsigned long long)avg_s1, (unsigned long long)(avg_mig / 1000));

                printf("\tML Shape: Pos=%llu | Neg=%llu | Zero=%llu\n",
                    (unsigned long long)s.ml_score_pos, (unsigned long long)s.ml_score_neg, (unsigned long long)s.ml_score_zero);

                printf("\tML Drops: Cooldown=%llu\n", (unsigned long long)s.ml_drop_cooldown);

                printf("\tEnqueue: OK=%llu | Fail: RingFull=%llu CAS_Retries=%llu Lookup=%llu Dedup=%llu\n",
                    (unsigned long long)s.enq_ok_ml_total, (unsigned long long)s.enq_fail_ring_full, (unsigned long long)s.enq_fail_cas_retries, (unsigned long long)s.enq_fail_lookup, (unsigned long long)s.filter_dedup);

                printf("\tWorker: BatchLimit=%llu | Ring: Head=%llu Tail=%llu Backlog=%u\n",
                    (unsigned long long)s.xt_hit_batch_limit, (unsigned long long)s.xt_ring_head, (unsigned long long)s.xt_ring_tail, backlog);

                printf("\tExec Fail: NoOp=%llu | Err/MM=%llu\n",
                    (unsigned long long)s.xt_skip_noop, (unsigned long long)s.xt_skip_mm_or_gup_fail);

                if (us_mode) {
                    double loss = (s.us_fwd_ok + s.us_fwd_drop) ?
                        100.0 * s.us_fwd_drop / (double)(s.us_fwd_ok + s.us_fwd_drop) : 0.0;
                    printf("\t[USPOL] Fwd: ok=%llu drop=%llu (%.3f%% lost) | Rx=%llu Inf=%llu Admit=%llu | proc avg=%.1fms max=%.1fms | evict=%llu\n",
                        (unsigned long long)s.us_fwd_ok, (unsigned long long)s.us_fwd_drop, loss,
                        (unsigned long long)us_samples_rx, (unsigned long long)us_inferences,
                        (unsigned long long)us_admits,
                        us_epochs_done ? us_proc_ns_total / 1e6 / us_epochs_done : 0.0,
                        us_proc_ns_max / 1e6,
                        (unsigned long long)us_state_evictions);
                }
                {   // profiler CPU self-accounting (both modes; the userspace
                    // agent's CPU is part of what the counterfactual measures)
                    struct rusage ru;
                    if (getrusage(RUSAGE_SELF, &ru) == 0) {
                        double cpu_s = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec/1e6 +
                                       ru.ru_stime.tv_sec + ru.ru_stime.tv_usec/1e6;
                        printf("\t[CPU] profiler cumulative cpu=%.2fs\n", cpu_s);
                    }
                }

                if (mode >= 2) {
                    printf("\t[EVAL] Samples: T1=%llu T2=%llu Unk=%llu | Thrash1s: %llu | Flips: %llu\n",
                        (unsigned long long)s.eval_samples_t1, (unsigned long long)s.eval_samples_t2, (unsigned long long)s.eval_samples_unknown, (unsigned long long)s.eval_thrashes_1s, (unsigned long long)s.eval_tier_flips);
                    printf("\t[EVAL] Dec: P=%llu D=%llu N=%llu | Mig: S=%llu F=%llu N=%llu\n",
                        (unsigned long long)s.eval_dec_promote, (unsigned long long)s.eval_dec_demote, (unsigned long long)s.eval_dec_noop,
                        (unsigned long long)s.eval_mig_success, (unsigned long long)s.eval_mig_fail, (unsigned long long)s.eval_mig_noop);
                    printf("\tPromoScan: Batches=%llu TotalPages=%llu | Eviction: Batches=%llu TotalPages=%llu | Balance: P=%u D=%u\n",
                        (unsigned long long)total_promo_scan_batches,
                        (unsigned long long)total_promo_scan_pages,
                        (unsigned long long)total_evict_batches,
                        (unsigned long long)total_evict_pages,
                        rolling_promotes, rolling_demotes);
                    printf("\t[ADAPT] %s | period=%lu epoch=%dms | A/C/M epochs: %llu/%llu/%llu\n",
                        adapt_state_name[adapt.state],
                        (unsigned long)adapt_current_pebs_period(),
                        adapt_current_epoch_ms(),
                        (unsigned long long)adapt.epochs_in_active,
                        (unsigned long long)adapt.epochs_in_cruise,
                        (unsigned long long)adapt.epochs_in_monitor);
                }
                
                // PEBS-delivery watchdog: if pebs_seen grew by less than
                // THRESH since the last 2-second tick while perf fds are
                // still open, the per-event throttle has probably stuck our
                // events. The kernel throttles an event past
                // max_samples_per_tick and the unthrottle path sometimes
                // misses it, which shows up as 10-20 samples trickling
                // through per tick rather than a clean zero. Toggling
                // DISABLE/ENABLE resets hwc->interrupts and re-arms.
                //
                // Skip the watchdog for one tick after a toggle: time_running
                // needs a tick to restart cleanly after enable, so checking
                // immediately would re-trip on the recovery itself.
                {
                    static int frozen_ticks = 0;
                    static int grace_ticks = 0;
                    #define WATCHDOG_THRESH 50   // <50 samples in 2s = stuck
                    if (grace_ticks > 0) {
                        grace_ticks--;
                    } else {
                        uint64_t delta = s.pebs_seen - last_pebs;
                        if (delta < WATCHDOG_THRESH && pid_perf_count > 0 && cfg.cur_epoch > 20) {
                            frozen_ticks++;
                            if (frozen_ticks >= 2) {  // 4s of <50 samples/2s = stuck
                                fprintf(stderr, "[WATCHDOG] PEBS rate %llu/2s (< %d) -- toggling %d perf fds (epoch=%u pebs=%llu)\n",
                                        (unsigned long long)delta, WATCHDOG_THRESH,
                                        pid_perf_count, cfg.cur_epoch,
                                        (unsigned long long)s.pebs_seen);
                                for (int i = 0; i < pid_perf_count; i++) {
                                    ioctl(pid_perf_entries[i].fd, PERF_EVENT_IOC_DISABLE, 0);
                                    ioctl(pid_perf_entries[i].fd, PERF_EVENT_IOC_ENABLE, 0);
                                }
                                frozen_ticks = 0;
                                grace_ticks = 1;
                            }
                        } else {
                            frozen_ticks = 0;
                        }
                    }
                }
                last_pebs = s.pebs_seen;
                last_inf = s.total_inferences;
            }
            last_st = now;
        }
        if (us_mode) {
            // Consume forwarded samples; 1ms max block keeps the epoch timer
            // responsive. This wakeup latency + scheduler exposure is the
            // userspace control loop being measured.
            ring_buffer__poll(us_rb, 1);
        } else {
            usleep(1000);
        }
    }
    if (us_mode) {
        struct stats_val s_fin;
        if (bpf_map_lookup_elem(fd_stats, &k0, &s_fin) == 0) {
            printf("[USPOL FINAL] fwd_ok=%llu fwd_drop=%llu rx=%llu inf=%llu admits=%llu "
                   "epochs=%llu proc_avg_ms=%.2f proc_max_ms=%.2f enq_syscalls=%llu evictions=%llu\n",
                   (unsigned long long)s_fin.us_fwd_ok, (unsigned long long)s_fin.us_fwd_drop,
                   (unsigned long long)us_samples_rx, (unsigned long long)us_inferences,
                   (unsigned long long)us_admits, (unsigned long long)us_epochs_done,
                   us_epochs_done ? us_proc_ns_total / 1e6 / us_epochs_done : 0.0,
                   us_proc_ns_max / 1e6, (unsigned long long)us_enq_syscalls,
                   (unsigned long long)us_state_evictions);
        }
        if (uslog) fclose(uslog);
        ring_buffer__free(us_rb);
        free(us_state);
    }
    if (csv) fclose(csv);
    for (int i = 0; i < pid_perf_count; i++) close(pid_perf_entries[i].fd);
    free(pid_perf_entries);
    bpf_object__close(obj); return 0;
}
