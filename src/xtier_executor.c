// xtier_executor.c
// SPDX-License-Identifier: GPL-2.0

#define pr_fmt(fmt) "xtier: " fmt

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/sched/task.h>
#include <linux/sched/mm.h>
#include <linux/mm.h>
#include <linux/migrate.h>
#include <linux/mempolicy.h>
#include <linux/llist.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/pid.h>
#include <linux/delay.h>
#include <linux/bpf.h>
#include <linux/btf.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/sort.h>
#include <linux/nodemask.h>
#include <linux/topology.h>
#include <linux/string.h>
#include <linux/minmax.h>
#include <linux/timekeeping.h>
#include <linux/uaccess.h>
#include <linux/align.h>
#include <linux/pgtable.h>
#include <asm/pgtable_types.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Research Team");
MODULE_DESCRIPTION("xtier memory tiering executor");
MODULE_VERSION("6.8");

// --- CONSTANTS ---
#define MAX_REGIONS_PER_EPOCH 32768
#define REG_SHIFT_BITS 21
#define REG_SIZE (1ULL << REG_SHIFT_BITS)
static struct bpf_map *tenant_map;   /* per-tenant table; NULL = single tenant */
#define BPF_MAP_PATH "/sys/fs/bpf/mprof_region_queue_map"
#define STATS_MAP_PATH "/sys/fs/bpf/mprof_stats_map"
#define STATE_MAP_PATH "/sys/fs/bpf/mprof_page_state_map"
#define MIGRATE_BATCH_MAX 256

// --- TUNABLES ---
static unsigned int worker_interval_ms = 10;
module_param(worker_interval_ms, uint, 0644);

static bool enable_migration = false;
module_param(enable_migration, bool, 0644);

static unsigned int batch_limit = 2048;
module_param(batch_limit, uint, 0644);

static int promote_node = 0;
module_param(promote_node, int, 0644);

static int demote_node = 1;
module_param(demote_node, int, 0644);

// DRAM capacity cap: max net pages to promote before stopping.
// Set to ~90% of node 0 capacity in pages. E.g. for 7GB DRAM: 7*1024*1024/4 * 0.9 ~ 1,612,800
// Default 0 = disabled (no cap). Set via module param at load time.
static unsigned long promote_max_pages = 0;
module_param(promote_max_pages, ulong, 0644);

// Max migrations per worker cycle (prevents burst stalls)
static unsigned int mig_budget_per_cycle = 64;
module_param(mig_budget_per_cycle, uint, 0644);

// --- EXTERNAL KERNEL API ---
extern long xtier_migrate_range(struct mm_struct *mm, unsigned long start, unsigned long end, int target_node);

// --- STRUCTS ---
struct region_entry {
    u64 addr;
    u32 pid;
    s32 score;
    s32 pred;
    u32 epoch;
    u32 seq;
    u32 _pad;
};

struct region_queue {
    u32 head;
    u32 tail;
    u32 _pad[2];
    struct region_entry entries[MAX_REGIONS_PER_EPOCH];
};

struct xtier_stats {
    atomic64_t intents_processed;
    atomic64_t seq_retries;
    atomic64_t nr_invalid;
    atomic64_t nr_stale_clear_skip;
    atomic64_t queue_process_time_ns;
};
static struct xtier_stats stats;

// Mirror of BPF structs
struct config_val {
    u32 pid;
    u32 cur_epoch;
    u32 mode;
    u32 exec_sleep; // 0=normal, 1=deep sleep (profiler in MONITOR/CRUISE)
};

/* Mirrors struct tenant_cfg in common_kern.h -- layout must match exactly. */
struct tenant_cfg {
    u32 pid;
    u32 promote_threshold_dyn;
    u32 admit_prob_q16;
    u32 _pad;
    u64 budget_pages;
};
#define XTIER_MAX_TENANTS 8

struct stats_val {
    u64 pebs_seen; u64 drops;
    u64 t_stage0_accum_ns; u64 t_stage1_accum_ns;
    u64 t_migration_accum_ns; u64 nr_migrations_timed;
    u64 total_inferences; u64 heuristic_filtered; u64 queue_full; u64 pebs_triggered;
    u64 lock_dropped; u64 queue_epoch_conflict;
    u64 nr_promote_att; u64 nr_promote_succ; u64 nr_promote_fail;
    u64 nr_demote_att; u64 nr_demote_succ; u64 nr_demote_fail;
    u64 nr_node_mismatch; u64 nr_err_fault;
    u64 collect_rows_emitted;
    u64 dec_hot_heur; u64 dec_cold_heur; u64 skip_cooldown; u64 skip_stable; u64 run_ml;
    u64 ml_score_pos; u64 ml_score_neg; u64 ml_score_zero;
    u64 ml_drop_cooldown;
    u64 filter_dedup; u64 enq_fail_lookup; u64 enq_fail_ring_full; u64 enq_fail_cas_retries; u64 enq_ok_ml_total;
    u64 xt_hit_batch_limit; u64 xt_ring_head; u64 xt_ring_tail;
    u64 xt_clear_success; u64 xt_clear_stale_skip;
    u64 xt_prom_seen; u64 xt_dem_seen; u64 xt_prom_ok; u64 xt_dem_ok;
    u64 xt_skip_noop; u64 xt_skip_mm_or_gup_fail;
    u64 eval_samples_total; u64 eval_samples_t1; u64 eval_samples_t2; u64 eval_samples_unknown;
    u64 eval_dec_total; u64 eval_dec_promote; u64 eval_dec_demote; u64 eval_dec_noop;
    u64 eval_filter_cooldown; u64 eval_filter_stable;
    u64 eval_tier_flips; u64 eval_thrashes_1s;
    u64 eval_mig_success; u64 eval_mig_fail; u64 eval_mig_noop;
};

#define HIST_STEPS 6
struct bpf_page_state {
    u64 cur_hits; u64 cur_lat;
    u32 hits_prev[HIST_STEPS]; u32 lat_prev[HIST_STEPS];
    u32 last_epoch_seen; u32 last_hot_epoch;
    u32 last_mig_epoch; u32 last_enq_epoch;
    u32 owner_pid; u16 recency_epochs;
    u8 last_decision; u8 _pad;
    u8 tier; u8 last_final_action; u8 last_reason; u8 _pad2;
    u64 last_tier_change_ns;
    u32 last_exec_epoch; s32 last_exec_moved;
    u16 last_from_nid; u16 last_to_nid;
    u64 last_exec_dur_ns;
};

// Pending migration entry
struct pending_mig {
    u64 addr;
    u32 pid;
    u32 epoch;
    s32 score;
};

// --- STATE ---
static struct task_struct *worker_task;
static struct bpf_map *queue_map = NULL;
static struct bpf_map *stats_map = NULL;
static struct bpf_map *state_map = NULL;
static struct bpf_map *config_map = NULL;
static int empty_cycles = 0;

// Adaptive backoff: exponential backoff when ring is empty
#define BACKOFF_MIN_MS   5    // start at 5ms (= worker_interval_ms)
#define BACKOFF_MAX_MS   1000 // cap at 1 second
#define DEEP_SLEEP_MS    2000 // sleep interval when profiler signals MONITOR
static unsigned int current_sleep_ms;

// Pre-allocated batch buffers
static struct pending_mig *promote_batch;
static struct pending_mig *demote_batch;

// Net promotion tracking: promotes - demotes. When this exceeds promote_max_pages, stop promoting.
static atomic64_t net_promoted = ATOMIC64_INIT(0);

// Check if we've promoted too many pages (DRAM is likely full)
static bool promote_allowed(void)
{
    if (promote_max_pages == 0)
        return true; // cap disabled

    return atomic64_read(&net_promoted) < (s64)promote_max_pages;
}

// --- HELPERS ---

static struct bpf_map *acquire_map(const char *path)
{
    struct file *f;
    struct bpf_map *map;
    f = filp_open(path, O_RDWR, 0);
    if (IS_ERR(f)) return NULL;
    if (!f->f_inode || !f->f_inode->i_private) { filp_close(f, NULL); return NULL; }
    map = f->f_inode->i_private;
    bpf_map_inc(map);
    filp_close(f, NULL);
    return map;
}

// --- STATS HELPERS ---
static void update_xt_stats_seen(bool promote) {
    if (!stats_map) return;
    u32 key = 0; struct stats_val *s;
    rcu_read_lock();
    s = stats_map->ops->map_lookup_elem(stats_map, &key);
    if (s) {
        if (promote) __sync_fetch_and_add(&s->xt_prom_seen, 1);
        else __sync_fetch_and_add(&s->xt_dem_seen, 1);
    }
    rcu_read_unlock();
}

static void update_xt_stats_exec(bool promote, long moved, u64 dur_ns) {
    if (!stats_map) return;
    u32 key = 0; struct stats_val *s;
    rcu_read_lock();
    s = stats_map->ops->map_lookup_elem(stats_map, &key);
    if (s) {
        if (promote) {
            __sync_fetch_and_add(&s->nr_promote_att, 1);
            if (moved > 0) { __sync_fetch_and_add(&s->xt_prom_ok, 1); __sync_fetch_and_add(&s->nr_promote_succ, moved); }
            else __sync_fetch_and_add(&s->nr_promote_fail, 1);
        } else {
            __sync_fetch_and_add(&s->nr_demote_att, 1);
            if (moved > 0) { __sync_fetch_and_add(&s->xt_dem_ok, 1); __sync_fetch_and_add(&s->nr_demote_succ, moved); }
            else __sync_fetch_and_add(&s->nr_demote_fail, 1);
        }
        __sync_fetch_and_add(&s->t_migration_accum_ns, dur_ns);
        __sync_fetch_and_add(&s->nr_migrations_timed, 1);
        if (moved > 0) __sync_fetch_and_add(&s->eval_mig_success, 1);
        else __sync_fetch_and_add(&s->eval_mig_fail, 1);
    }
    rcu_read_unlock();
}

static void update_xt_stats_noop(void) {
    if (!stats_map) return;
    u32 key = 0; struct stats_val *s;
    rcu_read_lock();
    s = stats_map->ops->map_lookup_elem(stats_map, &key);
    if (s) {
        __sync_fetch_and_add(&s->xt_skip_noop, 1);
        __sync_fetch_and_add(&s->eval_mig_noop, 1);
    }
    rcu_read_unlock();
}

static void update_xt_stats_mm_fail(void) {
    if (!stats_map) return;
    u32 key = 0; struct stats_val *s;
    rcu_read_lock();
    s = stats_map->ops->map_lookup_elem(stats_map, &key);
    if (s) {
        __sync_fetch_and_add(&s->xt_skip_mm_or_gup_fail, 1);
        __sync_fetch_and_add(&s->eval_mig_fail, 1);
    }
    rcu_read_unlock();
}

static void update_xt_stats_batch_limit(void) {
    if (!stats_map) return;
    u32 key = 0; struct stats_val *s;
    rcu_read_lock();
    s = stats_map->ops->map_lookup_elem(stats_map, &key);
    if (s) __sync_fetch_and_add(&s->xt_hit_batch_limit, 1);
    rcu_read_unlock();
}

static void update_xt_stats_ring_debug(u32 head, u32 tail) {
    if (!stats_map) return;
    u32 key = 0; struct stats_val *s;
    rcu_read_lock();
    s = stats_map->ops->map_lookup_elem(stats_map, &key);
    if (s) {
        s->xt_ring_head = head;
        s->xt_ring_tail = tail;
    }
    rcu_read_unlock();
}

static void update_page_cooldown(u64 page, u32 epoch) {
    if (!state_map) return;
    struct bpf_page_state *st;
    rcu_read_lock();
    st = state_map->ops->map_lookup_elem(state_map, &page);
    if (st) st->last_mig_epoch = epoch;
    rcu_read_unlock();
}

static void update_page_exec_info(u64 page, u32 epoch, int from_nid, int to_nid, long moved, u64 dur_ns) {
    if (!state_map) return;
    struct bpf_page_state *st;
    u64 now_ns = ktime_get_ns();
    rcu_read_lock();
    st = state_map->ops->map_lookup_elem(state_map, &page);
    if (st) {
        st->last_exec_epoch = epoch;
        st->last_exec_moved = (s32)moved;
        st->last_from_nid = (u16)(from_nid < 0 ? 0xFFFF : from_nid);
        st->last_to_nid = (u16)(to_nid < 0 ? 0xFFFF : to_nid);
        st->last_exec_dur_ns = dur_ns;
        // Even on a noop (page already on target / migration declined),
        // record the discovered physical node into tier so the heuristic
        // filter and promotion scanner stop wasting work on it. This is the
        // ground truth that corrects the slow-tier seed BPF starts pages at.
        if (from_nid >= 0) {
            if (from_nid == promote_node)      st->tier = 1;
            else if (from_nid == demote_node)  st->tier = 2;
        }
        if (moved > 0) {
            u8 new_tier = (to_nid == promote_node) ? 1 : 2;
            u8 old_tier = st->tier;
            st->tier = new_tier;
            if (old_tier != 0 && old_tier != new_tier) {
                if (stats_map) {
                    u32 key = 0;
                    struct stats_val *s = stats_map->ops->map_lookup_elem(stats_map, &key);
                    if (s) {
                        __sync_fetch_and_add(&s->eval_tier_flips, 1);
                        if (st->last_tier_change_ns && (now_ns - st->last_tier_change_ns) < 1000000000ULL)
                            __sync_fetch_and_add(&s->eval_thrashes_1s, 1);
                    }
                }
                st->last_tier_change_ns = now_ns;
            } else if (old_tier == 0) {
                st->last_tier_change_ns = now_ns;
            }
        }
    }
    rcu_read_unlock();
}

static int gup_one_page_nid(struct mm_struct *mm, unsigned long addr)
{
    struct page *page = NULL;
    int locked = 0;
    long ret;
    int nid = -1;
    ret = get_user_pages_remote(mm, addr, 1, FOLL_GET, &page, &locked);
    if (ret > 0 && page) {
        nid = page_to_nid(page);
        put_page(page);
    }
    return nid;
}

// --- BATCHED MIGRATION WITH CONTIGUOUS RANGE MERGING ---

static int addr_cmp(const void *a, const void *b)
{
    const struct pending_mig *ma = a, *mb = b;
    if (ma->addr < mb->addr) return -1;
    if (ma->addr > mb->addr) return 1;
    return 0;
}

static void execute_batch(struct pending_mig *batch, int count, int target_node, bool is_promote)
{
    struct pid *pid_struct;
    struct task_struct *task;
    struct mm_struct *mm;
    int i;

    if (count == 0) return;

    // Sort by address for contiguous detection
    sort(batch, count, sizeof(struct pending_mig), addr_cmp, NULL);

    pid_struct = find_get_pid(batch[0].pid);
    if (!pid_struct) {
        for (i = 0; i < count; i++) update_xt_stats_mm_fail();
        return;
    }
    task = get_pid_task(pid_struct, PIDTYPE_PID);
    if (!task) {
        put_pid(pid_struct);
        for (i = 0; i < count; i++) update_xt_stats_mm_fail();
        return;
    }
    mm = get_task_mm(task);
    if (!mm) {
        put_task_struct(task);
        put_pid(pid_struct);
        for (i = 0; i < count; i++) update_xt_stats_mm_fail();
        return;
    }

    // Merge contiguous pages into ranges and migrate each range in one call
    i = 0;
    while (i < count) {
        unsigned long range_start = (unsigned long)batch[i].addr;
        unsigned long range_end = range_start + PAGE_SIZE;
        int range_count = 1;

        // Extend range while pages are contiguous
        while (i + range_count < count &&
               (unsigned long)batch[i + range_count].addr == range_end) {
            range_end += PAGE_SIZE;
            range_count++;
        }

        // No pre-migration gup probe. Each gup_one_page_nid() takes the
        // mmap_read_lock and a refcount on the page -- doing it for every
        // candidate range puts the worker in lock contention with the
        // workload. The kernel's migrate_pages path inside xtier_migrate_range
        // already short-circuits pages that are already on the target node.
        if (target_node >= 0 && target_node < MAX_NUMNODES && node_online(target_node)) {
            ktime_t ts1 = ktime_get();
            long moved = xtier_migrate_range(mm, range_start, range_end, target_node);
            ktime_t ts2 = ktime_get();
            u64 dur = ktime_to_ns(ktime_sub(ts2, ts1));

            // Update stats for the range
            update_xt_stats_exec(is_promote, moved, dur);

            // Track net promotions for capacity cap
            if (moved > 0) {
                if (is_promote)
                    atomic64_add(range_count, &net_promoted);
                else
                    atomic64_sub(range_count, &net_promoted);
            }

            // Update per-page bookkeeping. We don't have a from_nid anymore
            // (no gup probe), pass -1 to indicate unknown.
            int j;
            for (j = 0; j < range_count; j++) {
                if (moved > 0) update_page_cooldown(batch[i + j].addr, batch[i + j].epoch);
                update_page_exec_info(batch[i + j].addr, batch[i + j].epoch,
                                     -1, target_node, moved > 0 ? 1 : 0, dur);
            }
        } else {
            int j;
            for (j = 0; j < range_count; j++) {
                update_xt_stats_mm_fail();
                update_page_exec_info(batch[i + j].addr, batch[i + j].epoch, -1, target_node, 0, 0);
            }
        }

        i += range_count;
        cond_resched();
    }

    mmput(mm);
    put_task_struct(task);
    put_pid(pid_struct);
}

// --- CORE LOGIC ---

static int process_ring(void)
{
    int drained = 0;
    int n_promote = 0, n_demote = 0;
    bool can_promote;
    ktime_t t0 = ktime_get();

    u32 key = 0;
    rcu_read_lock();
    struct region_queue *q = queue_map->ops->map_lookup_elem(queue_map, &key);
    if (!q) {
        rcu_read_unlock();
        return 0;
    }

    u32 head = READ_ONCE(q->head);
    u32 tail = READ_ONCE(q->tail);

    update_xt_stats_ring_debug(head, tail);

    // Check if we can still promote (haven't hit the cap)
    can_promote = promote_allowed();

    while (head != tail) {
        if (drained >= batch_limit)  {
            update_xt_stats_batch_limit();
            break;
        }
        // Enforce per-cycle migration budget
        if ((n_promote + n_demote) >= mig_budget_per_cycle)
            break;
        if (n_promote >= MIGRATE_BATCH_MAX || n_demote >= MIGRATE_BATCH_MAX)
            break;

        u32 mask = MAX_REGIONS_PER_EPOCH - 1;
        struct region_entry *e = &q->entries[head & mask];

        u32 seq = READ_ONCE(e->seq);
        int spin_count = 0;

        while (seq != head + 1 && spin_count < 100) {
            cpu_relax();
            seq = READ_ONCE(e->seq);
            spin_count++;
        }

        if (seq != head + 1)
            break;

        smp_rmb();

        update_xt_stats_seen(e->pred > 0);

        if (enable_migration && e->pid != 0) {
            struct pending_mig pm = {
                .addr = e->addr,
                .pid = e->pid,
                .epoch = e->epoch,
                .score = e->score,
            };

            if (e->pred > 0) {
                // Only enqueue promotion if DRAM has room
                if (can_promote && n_promote < MIGRATE_BATCH_MAX) {
                    promote_batch[n_promote++] = pm;
                } else {
                    // DRAM full -- skip this promotion silently
                    update_xt_stats_noop();
                }
            } else if (n_demote < MIGRATE_BATCH_MAX) {
                demote_batch[n_demote++] = pm;
            }
        }

        WRITE_ONCE(e->seq, head + MAX_REGIONS_PER_EPOCH + 1);
        smp_wmb();

        head++;
        WRITE_ONCE(q->head, head);

        drained++;
        atomic64_inc(&stats.intents_processed);
    }

    rcu_read_unlock();

    // Execute batched migrations outside rcu lock
    if (n_promote > 0)
        execute_batch(promote_batch, n_promote, promote_node, true);
    if (n_demote > 0)
        execute_batch(demote_batch, n_demote, demote_node, false);

    ktime_t t1 = ktime_get();
    atomic64_add(ktime_to_ns(ktime_sub(t1, t0)), &stats.queue_process_time_ns);

    return drained;
}

// =============================================================================
// PTE Accessed-bit cold scanner
// =============================================================================
//
// PEBS-based cold detection ("the page hasn't been sampled recently") is
// statistical and unreliable: at modest sample rates, warm pages get evicted
// because they happened to dodge the sampling interval. AutoNUMA gets cold
// detection for free from the hardware Accessed bit in the PTE.
//
// We do the same here. A kernel-side walker traverses the workload's page
// tables, and for each PTE pointing to a page on promote_node (DRAM):
//   - if the A bit is set, clear it (the page was touched since last scan,
//     treat as warm).
//   - if the A bit is already clear, the page hasn't been touched since the
//     previous scan -- it's cold. Add to demotion list.
//
// Cold candidates flow into execute_batch as demotions. The walker is
// chunked (COLD_SCAN_MAX_PTES per call) and resumes from where it left off,
// so a single scan call costs only a few ms -- never blocks the workload.
// =============================================================================

#define COLD_SCAN_MAX_RESULTS 1024
#define COLD_SCAN_MAX_PTES   50000    // ~3-5ms of work per call
#define COLD_SCAN_INTERVAL_CYCLES 200 // run every 200 worker cycles (~1s)
// Target: ~50K PTEs/sec scan rate = ~200 MB/s through the address space.
// On a 5M-page working set that's a full pass every ~100s. Matches the
// "AutoNUMA takes minutes" cadence and minimizes mmap_lock + PTE-lock
// contention with the workload's OpenMP threads. The pressure gate below
// further suppresses the scan when DRAM has free capacity.
// Only run the cold scanner when DRAM occupancy crosses this fraction of
// promote_max_pages. When DRAM has plenty of room, scanning + demoting is
// pure waste -- there's no need to make room for promotions that aren't
// blocked. Self-regulating watermark: scanner sleeps until pressure rises.
#define COLD_SCAN_PRESSURE_NUM 7
#define COLD_SCAN_PRESSURE_DEN 10

static u64 *cold_addrs_buffer;
static unsigned long cold_scan_resume_addr;
static u64 cold_scan_total_scanned;
static u64 cold_scan_total_cold;
static u64 cold_scan_total_passes;

/* Manual page-table walker. We can't use walk_page_range / vm_normal_page /
 * ptep_test_and_clear_young from a module (none are EXPORT_SYMBOL'd in this
 * kernel), so we walk by hand using only inlines and exported symbols. */
static int scan_dram_for_cold_pages(struct mm_struct *mm,
                                    u64 *out_addrs, int max_out)
{
    struct vm_area_struct *vma;
    unsigned long addr;
    unsigned long end;
    int n_cold = 0;
    int scanned = 0;

    if (!mm) return 0;
    if (!mmap_read_trylock(mm)) return 0;

    end = TASK_SIZE_MAX;
    addr = cold_scan_resume_addr;
    if (addr >= end) addr = 0;

    while (addr < end && scanned < COLD_SCAN_MAX_PTES && n_cold < max_out) {
        unsigned long vma_end;

        vma = find_vma(mm, addr);
        if (!vma) break;
        if (addr < vma->vm_start) addr = vma->vm_start;

        /* Skip stack (pinned to DRAM via page_type rule), and any mapping
         * that doesn't represent normal RAM-backed user memory. */
        if (vma->vm_flags & (VM_GROWSDOWN | VM_PFNMAP | VM_IO | VM_HUGETLB)) {
            addr = vma->vm_end;
            continue;
        }

        vma_end = vma->vm_end;
        if (vma_end > end) vma_end = end;

        while (addr < vma_end && scanned < COLD_SCAN_MAX_PTES && n_cold < max_out) {
            pgd_t *pgd;
            p4d_t *p4d;
            pud_t *pud;
            pmd_t *pmd;
            pte_t *pte_ptr;
            spinlock_t *ptl;
            pte_t entry;
            unsigned long pfn;
            struct page *page;

            scanned++;

            pgd = pgd_offset(mm, addr);
            if (pgd_none(*pgd) || pgd_bad(*pgd)) {
                addr = (addr + PGDIR_SIZE) & PGDIR_MASK;
                continue;
            }
            p4d = p4d_offset(pgd, addr);
            if (p4d_none(*p4d) || p4d_bad(*p4d)) {
                addr = (addr + P4D_SIZE) & P4D_MASK;
                continue;
            }
            pud = pud_offset(p4d, addr);
            if (pud_none(*pud) || pud_bad(*pud)) {
                addr = (addr + PUD_SIZE) & PUD_MASK;
                continue;
            }
            pmd = pmd_offset(pud, addr);
            if (pmd_none(*pmd) || pmd_bad(*pmd)) {
                addr = (addr + PMD_SIZE) & PMD_MASK;
                continue;
            }
            /* Skip transparent huge pages -- handle 4K only for now. */
            if (pmd_trans_huge(*pmd) || pmd_devmap(*pmd)) {
                addr = (addr + PMD_SIZE) & PMD_MASK;
                continue;
            }

            /* pte_offset_kernel is an inline that returns the PTE pointer
             * directly from the pmd's linear mapping. Safe to use here under
             * mmap_read_lock: munmap takes the write lock and can't race
             * with us, page faults only set PTEs (don't tear them down),
             * and we re-check pte_present below. */
            pte_ptr = pte_offset_kernel(pmd, addr);
            if (!pte_ptr) {
                addr += PAGE_SIZE;
                continue;
            }

            entry = ptep_get(pte_ptr);
            if (pte_present(entry)) {
                pfn = pte_pfn(entry);
                if (pfn_valid(pfn)) {
                    page = pfn_to_page(pfn);
                    if (page && page_to_nid(page) == promote_node) {
                        if (pte_young(entry)) {
                            /* Touched since last scan -- clear A bit
                             * atomically. No TLB flush needed; hardware
                             * sets the A bit again on next access. */
                            test_and_clear_bit(_PAGE_BIT_ACCESSED,
                                               (unsigned long *)&pte_ptr->pte);
                        } else {
                            /* A bit was already clear -- cold candidate. */
                            out_addrs[n_cold++] = addr;
                        }
                    }
                }
            }

            (void)ptl;  /* unused with lockless walk */
            addr += PAGE_SIZE;
        }
    }

    mmap_read_unlock(mm);

    cold_scan_total_scanned += scanned;
    cold_scan_total_cold += n_cold;

    if (addr >= end) {
        cold_scan_resume_addr = 0;
        cold_scan_total_passes++;
    } else {
        cold_scan_resume_addr = addr;
    }
    return n_cold;
}

/* Cold scanner only runs when DRAM is close to its cap. With plenty of free
 * DRAM there's no value in evicting anything -- just makes room we don't need
 * and feeds a thrash loop with the promotion side. */
static bool cold_scan_should_run(void)
{
    s64 net;
    if (promote_max_pages == 0) return true;  /* uncapped: always scan */
    net = atomic64_read(&net_promoted);
    if (net < 0) return false;
    return (u64)net > (promote_max_pages * COLD_SCAN_PRESSURE_NUM
                                          / COLD_SCAN_PRESSURE_DEN);
}

/* Scan ONE process for cold DRAM pages and demote them. Split out of
 * run_cold_scan_and_demote so multi-tenant runs can sweep every managed
 * process: with a single hard-wired pid, tenant B was never scanned, so its
 * cold pages were never demoted and it held DRAM it had stopped using --
 * which would have silently broken the static partition. */
static void cold_scan_one_pid(pid_t target_pid)
{
    struct pid *pid_struct;
    struct task_struct *task;
    struct mm_struct *mm;
    int n_cold;
    struct pending_mig *batch;
    int i;

    if (!cold_addrs_buffer || target_pid <= 0) return;

    pid_struct = find_get_pid(target_pid);
    if (!pid_struct) return;
    task = get_pid_task(pid_struct, PIDTYPE_PID);
    if (!task) { put_pid(pid_struct); return; }
    mm = get_task_mm(task);
    if (!mm) { put_task_struct(task); put_pid(pid_struct); return; }

    n_cold = scan_dram_for_cold_pages(mm, cold_addrs_buffer, COLD_SCAN_MAX_RESULTS);

    mmput(mm);
    put_task_struct(task);
    put_pid(pid_struct);

    if (n_cold == 0) return;

    /* Hand the cold list to execute_batch as a demotion. execute_batch will
     * re-acquire the mm itself and apply the existing contiguous-range
     * merging in xtier_migrate_range. */
    batch = kvmalloc_array(n_cold, sizeof(*batch), GFP_KERNEL);
    if (!batch) return;
    for (i = 0; i < n_cold; i++) {
        batch[i].addr = cold_addrs_buffer[i];
        batch[i].pid = target_pid;
        batch[i].epoch = 0;
        batch[i].score = -500;
    }
    execute_batch(batch, n_cold, demote_node, false);
    kvfree(batch);
}

static void run_cold_scan_and_demote(void)
{
    u32 slot;
    int n_scanned = 0;

    if (!config_map || !cold_addrs_buffer) return;
    if (!cold_scan_should_run()) return;

    /* Multi-tenant: sweep every occupied slot. Each tenant is scanned against
     * its own mm, so a page is only ever demoted out of the process that owns
     * it. */
    if (tenant_map) {
        for (slot = 0; slot < XTIER_MAX_TENANTS; slot++) {
            struct tenant_cfg *tc;
            pid_t pid = 0;

            rcu_read_lock();
            tc = tenant_map->ops->map_lookup_elem(tenant_map, &slot);
            if (tc)
                pid = (pid_t)tc->pid;
            rcu_read_unlock();

            if (pid > 0) {
                cold_scan_one_pid(pid);
                n_scanned++;
            }
        }
    }

    /* Fall back to the single global pid when the tenant table is absent or
     * still empty (userspace has not published a threshold yet). */
    if (n_scanned == 0) {
        u32 key = 0;
        struct config_val *cfg;
        pid_t pid = 0;

        rcu_read_lock();
        cfg = config_map->ops->map_lookup_elem(config_map, &key);
        if (cfg)
            pid = (pid_t)cfg->pid;
        rcu_read_unlock();

        if (pid > 0)
            cold_scan_one_pid(pid);
    }
}

// Check if profiler has signaled deep sleep (MONITOR mode)
static bool check_deep_sleep(void)
{
    if (!config_map) return false;
    u32 key = 0;
    struct config_val *cfg;
    bool deep = false;
    rcu_read_lock();
    cfg = config_map->ops->map_lookup_elem(config_map, &key);
    if (cfg && cfg->exec_sleep)
        deep = true;
    rcu_read_unlock();
    return deep;
}

static int worker_thread(void *data)
{
    int cold_scan_counter = 0;

    pr_info("xtier: worker started (budget=%u, max_pages=%lu, worker_ms=%u, adaptive backoff)\n",
            mig_budget_per_cycle, promote_max_pages, worker_interval_ms);

    current_sleep_ms = worker_interval_ms;

    while (!kthread_should_stop()) {
        if (!queue_map) {
            queue_map = acquire_map(BPF_MAP_PATH);
            stats_map = acquire_map(STATS_MAP_PATH);
            state_map = acquire_map(STATE_MAP_PATH);
            config_map = acquire_map("/sys/fs/bpf/mprof_config_map");
            /* Optional: absent on single-tenant runs, in which case the cold
             * scanner falls back to config_map->pid exactly as before. */
            tenant_map = acquire_map("/sys/fs/bpf/mprof_tenant_cfg_map");
            if (!queue_map) {
                if (empty_cycles++ > 25) {
                    pr_info("xtier: waiting for map at %s\n", BPF_MAP_PATH);
                    empty_cycles = 0;
                }
            } else {
                if (stats_map) pr_info("xtier: Stats map attached\n");
                if (config_map) pr_info("xtier: Config map attached (deep sleep support)\n");
            }
        }
        if (queue_map) {
            int work_done = process_ring();

            /* Periodic PTE A-bit cold scan. Runs every COLD_SCAN_INTERVAL_CYCLES
             * worker iterations regardless of whether the ring had work -- this
             * is the only mechanism that finds cold DRAM pages now that we've
             * stopped relying on PEBS-absence. */
            if (++cold_scan_counter >= COLD_SCAN_INTERVAL_CYCLES) {
                cold_scan_counter = 0;
                run_cold_scan_and_demote();
            }

            if (work_done > 0) {
                // Found work -- reset backoff to minimum
                current_sleep_ms = worker_interval_ms;
                cond_resched();
                continue;
            }
            // No work -- apply adaptive backoff
            if (check_deep_sleep()) {
                // Profiler is in MONITOR mode: sleep long
                current_sleep_ms = DEEP_SLEEP_MS;
            } else {
                // Exponential backoff: double sleep time, cap at BACKOFF_MAX_MS
                current_sleep_ms = min(current_sleep_ms * 2, (unsigned int)BACKOFF_MAX_MS);
            }
        }
        if (kthread_should_stop()) break;
        msleep_interruptible(current_sleep_ms);
    }
    if (queue_map) { bpf_map_put(queue_map); queue_map = NULL; }
    if (stats_map) { bpf_map_put(stats_map); stats_map = NULL; }
    if (state_map) { bpf_map_put(state_map); state_map = NULL; }
    if (config_map) { bpf_map_put(config_map); config_map = NULL; }
    return 0;
}

static int __init xtier_init(void)
{
    memset(&stats, 0, sizeof(stats));

    promote_batch = kvmalloc_array(MIGRATE_BATCH_MAX, sizeof(struct pending_mig), GFP_KERNEL);
    demote_batch = kvmalloc_array(MIGRATE_BATCH_MAX, sizeof(struct pending_mig), GFP_KERNEL);
    cold_addrs_buffer = kvmalloc_array(COLD_SCAN_MAX_RESULTS, sizeof(u64), GFP_KERNEL);
    if (!promote_batch || !demote_batch || !cold_addrs_buffer) {
        kvfree(promote_batch);
        kvfree(demote_batch);
        kvfree(cold_addrs_buffer);
        return -ENOMEM;
    }

    worker_task = kthread_run(worker_thread, NULL, "xtier_worker");
    if (IS_ERR(worker_task)) {
        kvfree(promote_batch);
        kvfree(demote_batch);
        kvfree(cold_addrs_buffer);
        return PTR_ERR(worker_task);
    }
    return 0;
}

static void __exit xtier_exit(void)
{
    if (worker_task) kthread_stop(worker_task);
    kvfree(promote_batch);
    kvfree(demote_batch);
    kvfree(cold_addrs_buffer);
    pr_info("=== XTier Stats ===\n");
    pr_info("Intents: %lld | SeqRetries: %lld | CAS-Skips: %lld\n",
        atomic64_read(&stats.intents_processed),
        atomic64_read(&stats.seq_retries),
        atomic64_read(&stats.nr_stale_clear_skip));
    pr_info("Cold scan: scanned=%llu cold_found=%llu full_passes=%llu\n",
        cold_scan_total_scanned,
        cold_scan_total_cold,
        cold_scan_total_passes);
}

module_init(xtier_init);
module_exit(xtier_exit);
