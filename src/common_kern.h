/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Shared ABI between the BPF program, the kernel module and the userspace
 * loader: model dimensions, page/region layout, and the config and stats
 * structs that live in shared BPF maps.
 */
#ifndef __COMMON_KERN_H__
#define __COMMON_KERN_H__

// Define types for userspace/kernel compatibility
typedef __u8 u8;
typedef __u16 u16;
typedef __u32 u32;
typedef __u64 u64;
typedef __s32 s32;
typedef __s64 s64;

// --- MODEL DIMENSIONS (Fixed) ---
#define L0_IN   32
#define L0_OUT  192
#define L1_OUT  96

// --- CONFIGURATION ---
#define PAGE_SHIFT  12u       
#define REG_SHIFT   21u       
#define HIST_STEPS       6        
#define FP_HIST          4        
#define FP_SLOT_BITS     256      
#define FP_WORDS_PER_SLOT (FP_SLOT_BITS / 64) 
#define OFF_BUCKETS      128        

// Buffer sizing. A workload with 20-40GB RSS touches well over 8192 unique
// pages in an epoch, and the overflow is silent, so size for the large case.
#define MAX_PAGES_PER_EPOCH 32768
#define MAX_REGIONS_PER_EPOCH 32768
#define REGION_QUEUE_MAP_SIZE 2

// Top-K budget-gated admission: BPF tallies MLP output scores
// into a 256-bucket histogram per epoch (cls_u8 ranges 0..255). Userspace
// reads the histogram, finds the Kth-percentile score where K tracks the
// DRAM budget, and writes the threshold back via config_val.promote_threshold_dyn.
// Double-buffered: BPF reads active_hist_idx each inference to pick which
// bucket array to bump; userspace flips the index before snapshotting the
// inactive buffer, so reads are race-free.
#define SCORE_HIST_BUCKETS 256
#define SCORE_HIST_COUNT   2          // double-buffered

// --- MULTI-TENANCY (static partition) ---
// Two processes managed at once collide in every page-keyed map, because the
// key is a bare virtual address. With randomize_va_space=0 (which we set for
// reproducibility) two instances of the same binary get IDENTICAL layouts, so
// they would collide on essentially every page: hit counts merge and
// owner_pid becomes last-writer-wins, which can enqueue a migration against
// the wrong mm.
//
// Fix without changing any map key TYPE: a page key is always 4 KB-aligned,
// so its low PAGE_SHIFT bits are structurally zero. We store a small tenant
// SLOT there. Region keys (REG_SHIFT) have even more room. Everything keyed
// by "page" therefore becomes per-tenant for free, and the only discipline
// required is to mask the slot off wherever the value is used as a real
// address (enqueue_migration, region_entry.addr, move_pages).
//
// Slot is an index, not a pid: pids do not fit in 12 bits. The pid<->slot
// mapping lives in target_pids (value = slot).
#define XTIER_MAX_TENANTS  8
#define TENANT_SLOT_MASK   ((__u64)((1ull << PAGE_SHIFT) - 1ull))
#define PAGE_ADDR_MASK     (~TENANT_SLOT_MASK)
#define TENANT_SLOT(key)   ((__u32)((key) & TENANT_SLOT_MASK))
#define PAGE_ADDR(key)     ((key) & PAGE_ADDR_MASK)
#define TENANT_KEY(pg, sl) (PAGE_ADDR(pg) | ((__u64)(sl) & TENANT_SLOT_MASK))

// Per-tenant admission state. The Top-K gate is what "static partition"
// actually means: each tenant gets its own DRAM budget, its own score
// histogram, and therefore its own threshold. A shared histogram would let
// whichever tenant emits more samples set the threshold for both.
struct tenant_cfg {
    __u32 pid;                   // 0 = slot unused
    __u32 promote_threshold_dyn; // per-tenant Top-K threshold
    __u32 admit_prob_q16;        // per-tenant probabilistic gate
    __u32 _pad;
    __u64 budget_pages;          // this tenant's DRAM partition
};

// --- EVALUATION ENUMS ---
enum eval_reason {
  EVAL_R_NONE = 0,
  EVAL_R_FILTER_COOLDOWN = 1,
  EVAL_R_FILTER_STABLE   = 2,
  EVAL_R_HEUR_HOT        = 3,
  EVAL_R_HEUR_COLD       = 4,
  EVAL_R_ML              = 5,
  EVAL_R_PRESSURE_EVICT  = 7,
};

enum eval_final_action {
  EVAL_A_NOOP    = 0,
  EVAL_A_PROMOTE = 1,
  EVAL_A_DEMOTE  = 2,
};

// --- STRUCTS ---

struct config_val {
    __u32 pid;
    __u32 cur_epoch;
    __u32 mode; // 0=monitor, 1=collect, 2=evaluate, 3=production (run)
    __u32 exec_sleep; // 0=normal polling, 1=deep sleep (MONITOR mode, near-zero work)
    // Top-K budget gate. promote_threshold_dyn=0 falls back to the static
    // PROMOTE_THRESH=1. When the profiler sets it > 0, BPF promotes only
    // when raw > promote_threshold_dyn.
    __u32 promote_threshold_dyn; // 0..253, clamped by userspace
    __u32 active_hist_idx;       // 0 or 1 -- picks which score_hist buffer BPF bumps
    // Probabilistic admission gate. Q16 fixed-point probability: 0 or 65536
    // means 100% pass (disabled), N < 65536 admits the fraction N/65536 of
    // otherwise-eligible pages. This is the fallback for when MLP output
    // saturates and the score carries no ranking signal, where the only
    // remaining defence against thrashing is to limit migration volume.
    __u32 admit_prob_q16;
    // Scorer selection, for ablation. Picks which function maps features to
    // the 0..255 score that feeds the histogram / top-K / admission path.
    // Everything downstream is identical across scorers; only the ranking
    // changes, which is what isolates the model's contribution.
    //   0 = mlp (default, ships as xTier)  3 = ewma
    //   1 = freq                           4 = random (gates-only control)
    //   2 = recency                        5 = oracle (future-benefit table)
    __u32 scorer_id;
    // Runtime cooldown. 0 = cooldown OFF.
    // Set from XTIER_COOLDOWN_EPOCHS; default COOLDOWN_EPOCHS (15).
    __u32 cooldown_epochs;
    // Userspace-policy counterfactual, for isolating the benefit of running
    // the policy in-kernel from the quality of the policy itself. When 1,
    // on_pebs_stage0 degrades to a plain sample forwarder (ringbuf to
    // userspace, no state, no filters, no inference) and the identical
    // policy runs in userspace (the us_* section of page_profiler_user.c),
    // re-injecting decisions through userspace_enqueue into the same
    // enqueue_migration() and executor path.
    // Set from XTIER_USERSPACE_POLICY=1. 0 = normal in-BPF policy.
    __u32 userspace_policy;
};

// Sample record forwarded to userspace in userspace-policy mode.
struct us_sample {
    __u64 page;    // 4K-aligned virtual address
    __u64 ts_ns;   // bpf_ktime_get_ns at PEBS handler entry
    __u32 pid;     // tgid
    __u32 _pad;
};

// Candidate batch userspace pushes back for enqueue (drained per test_run).
// 8, not more: the verifier walks the drain loop times enqueue_migration's
// unrolled 20-way CAS loop (early-return branches multiply the explored
// states), and both 1024 and 128 iterations exceeded the 1M-insn complexity
// budget (E2BIG at load). Userspace loops over chunks, so a small batch only
// costs extra syscalls per epoch -- which is part of the userspace tax the
// counterfactual measures anyway.
#define US_CAND_MAX 8

enum xtier_scorer {
    XTIER_SCORER_MLP     = 0,
    XTIER_SCORER_FREQ    = 1,
    XTIER_SCORER_RECENCY = 2,
    XTIER_SCORER_EWMA    = 3,
    XTIER_SCORER_RANDOM  = 4,
    XTIER_SCORER_ORACLE  = 5,
    XTIER_SCORER__MAX
};

// The "Huge" Stats Struct (Matches xtier_executor.c)
struct stats_val { 
    __u64 pebs_seen; 
    __u64 drops; 
    
    // Timings
    __u64 t_stage0_accum_ns;    
    __u64 t_stage1_accum_ns;    
    __u64 t_migration_accum_ns; 
    __u64 nr_migrations_timed;  
    
    __u64 total_inferences;
    __u64 heuristic_filtered;
    __u64 queue_full;            
    __u64 pebs_triggered;        

    // Legacy
    __u64 lock_dropped;
    __u64 queue_epoch_conflict;

    // Migration Stats (Legacy/Summary)
    __u64 nr_promote_att;
    __u64 nr_promote_succ;
    __u64 nr_promote_fail;
    __u64 nr_demote_att;
    __u64 nr_demote_succ;
    __u64 nr_demote_fail;
    __u64 nr_node_mismatch; 
    __u64 nr_err_fault;       
    
    // NEW INSTRUMENTATION COUNTERS
    __u64 collect_rows_emitted;

    // DECISION HISTOGRAM
    __u64 dec_hot_heur;
    __u64 dec_cold_heur;
    __u64 skip_cooldown;
    __u64 skip_stable;
    __u64 run_ml;

    // --- PIPELINE DEBUG COUNTERS ---
    
    // Group A: ML Output Shape
    __u64 ml_score_pos;        
    __u64 ml_score_neg;        
    __u64 ml_score_zero;       

    // Group B: ML Drops
    __u64 ml_drop_cooldown;    

    // Group C: Enqueue Mechanics
    __u64 filter_dedup;            
    __u64 enq_fail_lookup;         
    __u64 enq_fail_ring_full;        
    __u64 enq_fail_cas_retries;      
    __u64 enq_ok_ml_total;           

    // Group D: Worker Health
    __u64 xt_hit_batch_limit;
    __u64 xt_ring_head;            
    __u64 xt_ring_tail;            
    __u64 xt_clear_success;        
    __u64 xt_clear_stale_skip;       

    // Group E: Execution Outcomes
    __u64 xt_prom_seen;        
    __u64 xt_dem_seen;         
    __u64 xt_prom_ok;          
    __u64 xt_dem_ok;           
    __u64 xt_skip_noop;              
    __u64 xt_skip_mm_or_gup_fail;    

    // --- EVALUATION COUNTERS ---
    __u64 eval_samples_total;
    __u64 eval_samples_t1;
    __u64 eval_samples_t2;
    __u64 eval_samples_unknown;

    __u64 eval_dec_total;
    __u64 eval_dec_promote;
    __u64 eval_dec_demote;
    __u64 eval_dec_noop;

    __u64 eval_filter_cooldown;
    __u64 eval_filter_stable;

    __u64 eval_tier_flips;
    __u64 eval_thrashes_1s;
    __u64 eval_mig_success;
    __u64 eval_mig_fail;
    __u64 eval_mig_noop;

    // --- USERSPACE-POLICY FORWARDER HEALTH (appended; executor's mirror
    // --- struct omits these, which is safe: earlier offsets are unchanged).
    __u64 us_fwd_ok;    // samples successfully shipped to the ringbuf
    __u64 us_fwd_drop;  // ringbuf reserve failures = userspace fell behind
};

// Metadata
struct meta_val { __u8 perm_r, perm_w, perm_x; __u8 page_type; __u8 vma_size_bucket; __u8 anon; __u8 shared; __u8 _pad; };

struct bpf_page_state { 
    __u64 cur_hits; 
    __u64 cur_lat; 
    __u32 hits_prev[HIST_STEPS]; 
    __u32 lat_prev[HIST_STEPS]; 
    __u32 last_epoch_seen; 
    __u32 last_hot_epoch; 
    __u32 last_mig_epoch;   
    __u32 last_enq_epoch;   
    __u32 owner_pid;        
    __u16 recency_epochs; 
    __u8 last_decision;      
    __u8 _pad; 

    // --- EVALUATION STATE ---
    __u8  tier;              
    __u8  last_final_action; 
    __u8  last_reason;       
    __u8  _pad2;

    __u64 last_tier_change_ns;
    __u32 last_exec_epoch;
    __s32 last_exec_moved;    
    __u16 last_from_nid;
    __u16 last_to_nid;
    __u64 last_exec_dur_ns;
};

struct feature_key { __u32 epoch; __u32 _pad; __u64 page; };

struct feature_row {
    __u32 epoch; 
    __u64 page; 
    __u32 hits; 
    __u32 lat_sum;
    __u32 hits_prev[HIST_STEPS]; 
    __u32 lat_prev[HIST_STEPS]; 
    __s32 delta_hits[HIST_STEPS];
    __u16 recency_epochs; 
    __u16 epochs_since_promotion;
    __u64 fp[FP_HIST][FP_WORDS_PER_SLOT];
    __u8 perm_r, perm_w, perm_x;
    __u8 page_type;
    __u8 vma_size_bucket;
    __u8 anon;    // 1 if anonymous (no file backing: [heap]/[stack]/[anon:...]/unnamed)
    __u8 shared;  // 1 if shared mapping (perms contain 's')
    __u16 off_bucket;
    __u64 reg_id;
    __u32 pid; 
    __u32 last_mig_epoch; 
};

struct inference_args { 
    __u32 epoch; 
    __u32 pid; 
    __u64 page; 
};

struct region_entry {
    __u64 addr; 
    __u32 pid;
    __s32 score;
    __s32 pred; 
    __u32 epoch; 
    __u32 seq;   
    __u32 _pad;  
};

struct region_queue {
    __u32 head;   
    __u32 tail;   
    __u32 _pad[2];
    struct region_entry entries[MAX_REGIONS_PER_EPOCH];
};

struct qparams { __s32 in_zp0; __s32 out_zp0; __s32 out_zp1; __s32 out_zpC; __s32 out_zpR; __s32 cls_thr_q; };
struct row0 { __s32 w[L0_IN];  __s32 b; };
struct row1 { __s32 w[L0_OUT]; __s32 b; }; // Note: input to L1 is L0_OUT
struct rowO { __s32 w[L1_OUT]; __s32 b; };
struct mul  { __s32 M; __s32 shift; };
struct pred_row{ __s32 score; __s32 pred; };

#endif
