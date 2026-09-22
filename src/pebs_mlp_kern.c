// SPDX-License-Identifier: GPL-2.0
//
// xTier BPF program: handles PEBS memory-access samples, maintains per-page
// access state, runs the quantized INT8 MLP forward pass in-kernel, and
// enqueues migration intents for the executor module to carry out.

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
#include <stdbool.h>
#include "common_kern.h"

char LICENSE[] SEC("license") = "GPL";

#ifndef READ_ONCE
#define READ_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif
#ifndef WRITE_ONCE
#define WRITE_ONCE(x, val) ((*(volatile typeof(x) *)&(x)) = (val))
#endif
#define smp_wmb() asm volatile("" ::: "memory")

// --- CONFIG ---
// HOT_THRESHOLD: at PEBS period=5000, each sample stands for ~5000 loads, so
// even one sample in an epoch is a meaningful signal. Requiring two misses
// genuinely hot pages, because sampling this sparse rarely catches the same
// page twice in one epoch.
#define HOT_THRESHOLD       1
#define WARMUP_EPOCHS       3
// COOLDOWN_EPOCHS: prevent re-migration churn. A recently migrated page
// must wait this many epochs before being reconsidered. Higher = fewer
// total migrations = less overhead, but slower reaction to changes.
#define COOLDOWN_EPOCHS     15
#define VAR_THRESHOLD_INT   25
#define K_PERSISTENT        0   // restored to 0: heuristic generates candidates, ML filters them (see below)
#define SAFETY_MARGIN       2
// COLD_EPOCHS: consecutive zero-hit epochs before demotion.
// At 3ms epochs, 10 epochs = 30ms of inactivity.
// At 1000ms epochs, 10 epochs = 10s of inactivity.
#define COLD_EPOCHS         10

// --- MAPS ---
struct { __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 1024); __type(key, __u32); __type(value, __u8); } target_pids SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, struct config_val); } config_map SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, struct stats_val); } stats_map SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 32768); __type(key, __u64); __type(value, struct meta_val); } meta_map SEC(".maps");
// LRU_HASH: when full, evict least-recently-used. Critical: a plain HASH
// silently rejects inserts past max_entries, which means after the first
// ~131k pages we observe, every new PEBS sample to a new page is dropped on
// the floor -- invisible to the pipeline forever. That capped DRAM placement
// at ~520 MB regardless of how aggressively we promoted. With LRU and 1M
// entries (~4 GB of trackable pages, ~160 MB kernel memory) we can hold the
// active working set and let cold pages naturally roll out.
struct { __uint(type, BPF_MAP_TYPE_LRU_HASH); __uint(max_entries, 1048576); __type(key, __u64); __type(value, struct bpf_page_state); } page_state_map SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 1 << 20); __type(key, struct feature_key); __type(value, struct feature_row); } feature_row_map SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_HASH); __uint(max_entries, 131072); __type(key, struct feature_key); __type(value, struct pred_row); } pred_map SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 2); __type(key, __u32); __type(value, struct region_queue); } region_queue_map SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, struct qparams); } qpm SEC(".maps");

// Weights
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, L0_OUT); __type(key, __u32); __type(value, struct row0); } l0_wb SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, L0_OUT); __type(key, __u32); __type(value, struct mul); } l0_mul SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, L1_OUT); __type(key, __u32); __type(value, struct row1); } l1_wb SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, L1_OUT); __type(key, __u32); __type(value, struct mul); } l1_mul SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, struct rowO); } lc_wb SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, struct mul); } lc_mul SEC(".maps");

struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, MAX_PAGES_PER_EPOCH); __type(key, __u32); __type(value, __u64); } epoch_pages_0 SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, __u32); } count_0 SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, MAX_PAGES_PER_EPOCH); __type(key, __u32); __type(value, __u64); } epoch_pages_1 SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, __u32); } count_1 SEC(".maps");

struct { __uint(type, BPF_MAP_TYPE_PROG_ARRAY); __uint(max_entries, 4); __uint(key_size, 4); __uint(value_size, 4); } jmp_table SEC(".maps");

struct nn_scratch {
    __u8 input[32]; __u8 l0_out[192]; __u8 l1_out[96];
    struct inference_args meta; __u64 start_ts;
    // Raw (un-log-compressed) features for the alternate scorers, stashed at
    // the same site where input[] is built so that run_inference_l1 sees
    // exactly the same page, epoch and gates whichever scorer is selected.
    // Raw rather than log_scale'd on purpose: the MLP only ever sees
    // log-compressed inputs, so handing freq/ewma the raw values makes those
    // baselines stronger than the model's own view, and any MLP win measured
    // against them is a conservative one.
    __u32 f_hits_now;      // this epoch's accesses
    __u32 f_hits_sum;      // this epoch + HIST_STEPS previous epochs
    __u32 f_ewma;          // exponentially weighted, recent epochs dominate
    __u32 f_recency;       // epochs since last sampled access
};
struct { __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, struct nn_scratch); } scratch_map SEC(".maps");

struct batch_ctrl { __u32 idx; __u32 epoch; };
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, struct batch_ctrl); } batch_progress SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_LRU_HASH); __uint(max_entries, 16384); __type(key, __u64); __type(value, __u32); } dedup_cache SEC(".maps");

struct fp_slot { __u64 bm[FP_WORDS_PER_SLOT]; };
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, FP_HIST); __type(key, __u32); __type(value, struct fp_slot); } fp_ring SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, FP_HIST); __type(key, __u32); __type(value, __u32); } fp_epoch SEC(".maps");

// --- USERSPACE-POLICY COUNTERFACTUAL ---
// Sample forwarder ring: 8 MB (~330K in-flight samples at 24B/record),
// comparable to a generously sized perf mmap ring. bpf_ringbuf_output uses
// spin_trylock in NMI context, so under contention or overrun the output
// fails rather than blocking PEBS; the failure is counted in
// stats.us_fwd_drop.
struct { __uint(type, BPF_MAP_TYPE_RINGBUF); __uint(max_entries, 1 << 23); } us_ring SEC(".maps");
// Decision re-injection: userspace writes up to US_CAND_MAX region_entry
// candidates + a count, then test-runs userspace_enqueue to drain them
// through the SAME enqueue_migration() the in-BPF policy uses.
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, US_CAND_MAX); __type(key, __u32); __type(value, struct region_entry); } us_cand SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, 1); __type(key, __u32); __type(value, __u32); } us_cand_cnt SEC(".maps");

// Top-K budget gate: two histograms, double-buffered. BPF reads
// config_val.active_hist_idx each inference and bumps the chosen buffer's
// bucket at index cls_u8 (the quantized MLP output, 0..255). Userspace
// flips active_hist_idx before reading/zeroing the now-inactive buffer.
// Per-tenant score histograms: laid out as [slot][bucket], indexed
// slot*SCORE_HIST_BUCKETS + bucket. Each tenant's Top-K threshold is computed
// from its OWN slice against its OWN budget -- that is the static partition.
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, SCORE_HIST_BUCKETS * XTIER_MAX_TENANTS); __type(key, __u32); __type(value, __u64); } score_hist_0 SEC(".maps");
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, SCORE_HIST_BUCKETS * XTIER_MAX_TENANTS); __type(key, __u32); __type(value, __u64); } score_hist_1 SEC(".maps");
// Per-tenant admission state, indexed by slot. Written by userspace each epoch.
struct { __uint(type, BPF_MAP_TYPE_ARRAY); __uint(max_entries, XTIER_MAX_TENANTS); __type(key, __u32); __type(value, struct tenant_cfg); } tenant_cfg_map SEC(".maps");

// --- HELPERS ---
static __always_inline __u8 clamp_u8(__s64 x) { if (x < 0) return 0; if (x > 255) return 255; return (__u8)x; }
/*
 * Quantized ReLU. The trained network is
 *     x = relu0(fc0(x)); x = relu1(fc1(x)); x = fc_head(x)
 * but this program only ever applied clamp_u8(), i.e. clamp to [0,255].
 *
 * In affine quantization the real value is (q - zero_point) * scale, so
 * "output >= 0" is "q >= zero_point". Clamping to [0,255] therefore admits the
 * whole band BELOW the zero point -- values encoding NEGATIVE activations that
 * ReLU should have zeroed. With L0_OUT_ZP=101 that is ~40% of the range
 * flowing through with the wrong sign; it overflows layer 1 (17.7% of its
 * outputs pin at 255) and saturates the final score to a single value for
 * every page.
 *
 * Verified offline against PyTorch on 3000 real feature vectors:
 *   clamp [0,255]  -> 1 distinct output, 100% at 255
 *   clamp [zp,255] -> 25 distinct outputs, 1% at 255
 *
 * fc_head has NO ReLU, so the final layer keeps plain clamp_u8().
 */
static __always_inline __u8 clamp_relu_u8(__s64 x, __s32 zp) {
    /* zp comes from the model header, always 0..255 -- no bounds check needed.
     * Kept to ONE extra compare over clamp_u8(): run_inference_l1 sits close to
     * the eBPF complexity limit and a 4-branch version was rejected with E2BIG. */
    if (x < (__s64)zp) return (__u8)zp;
    if (x > 255) return 255;
    return (__u8)x;
}
static __always_inline __s64 mul_shift_rnd(__s64 acc, __s32 M, __s32 sh) {
    __s64 t = (M != 0) ? (acc * (__s64)M) : acc;
    if (sh < 0) sh = 0; if (sh > 63) sh = 63;
    if (sh > 0) { __s64 add = (t >= 0) ? ((__s64)1 << (sh - 1)) : -((__s64)1 << (sh - 1)); t += add; t >>= sh; }
    return t;
}
static __always_inline __u8 pack_delta(__s32 d) { if (d > 127) d = 127; if (d < -128) d = -128; return (__u8)(d + 128); }
static __always_inline __u8 log_scale(__u32 val) {
    if (val == 0) return 32; if (val <= 5) return (__u8)(32 + val * 4);
    if (val <= 20) return (__u8)(52 + (val-5)/2); if (val <= 100) return (__u8)(60 + (val-20)/10);
    /* SATURATE. This returns __u8: 70 + (val-100)/50 exceeds 255 above ~9350
     * hits and truncates, so the hottest pages used to encode COLDER than cold
     * ones (12.9M hits -> byte 3 vs 36 for a single hit). Harmless at the old
     * throttled sample rate; material at 4M samples/s. log_scale_np() in
     * train_and_export.py clips identically -- keep the two in sync. */
    {
        __u32 r = 70u + (val - 100u) / 50u;
        return (__u8)(r > 255u ? 255u : r);
    }
}
/*
 * Quantize an unbounded counter into 0..255 while
 * preserving rank. 16 levels per octave over 16 octaves -- far finer than
 * log_scale(), which saturates hard above ~100 and would manufacture ties.
 * Tie rate per scorer is reported alongside every ablation run; if a scorer
 * ties much more than the MLP it wins/loses on resolution, not judgement.
 */
static __always_inline __u8 q255_log2(__u32 v) {
    if (v == 0) return 0;
    __u32 hi = 0;
    #pragma unroll
    for (int b = 31; b >= 1; b--) { if (hi == 0 && (v >> b)) hi = (__u32)b; }
    __u32 mant = (hi >= 4) ? ((v >> (hi - 4)) & 0xF) : ((v << (4 - hi)) & 0xF);
    __u32 q = hi * 16u + mant;
    return (__u8)(q > 255u ? 255u : q);
}
static __always_inline __u16 off_bucket_calc(__u64 page) { return (__u16)(((page >> PAGE_SHIFT) & 511u) & (OFF_BUCKETS - 1u)); }
static __always_inline __u32 mix_to_bit(__u64 page_num) {
    __u64 x = page_num ^ (page_num >> 33); x *= 0xff51afd7ed558ccdULL; x ^= (x >> 33);
    return (__u32)x & (FP_SLOT_BITS - 1u);
}

static __always_inline void set_fp_bit(__u64 page, __u32 cur_epoch) {
    __u32 slot = cur_epoch & (FP_HIST - 1u);
    struct fp_slot *fs = bpf_map_lookup_elem(&fp_ring, &slot);
    __u32 *se = bpf_map_lookup_elem(&fp_epoch, &slot);
    if (!fs || !se) return;
    if (*se != cur_epoch) { 
#pragma unroll
        for (int i = 0; i < FP_WORDS_PER_SLOT; i++) { fs->bm[i] = 0; }
        *se = cur_epoch; 
    }
    __sync_fetch_and_or(&fs->bm[mix_to_bit(page >> PAGE_SHIFT) >> 6], 1ull << (mix_to_bit(page >> PAGE_SHIFT) & 63));
}

static __always_inline void roll_state_if_needed(struct bpf_page_state *st, __u32 cur_epoch) {
    if (st->last_epoch_seen == cur_epoch) return;
    if (st->cur_hits == 0 && st->last_epoch_seen != 0) {
        __u32 gap = cur_epoch - st->last_epoch_seen;
        st->recency_epochs = (st->recency_epochs + gap < 0xFFFF) ? st->recency_epochs + gap : 0xFFFF;
    }
    for (int i = HIST_STEPS - 1; i > 0; --i) { st->hits_prev[i] = st->hits_prev[i-1]; st->lat_prev[i] = st->lat_prev[i-1]; }
    st->hits_prev[0] = (__u32)st->cur_hits; st->lat_prev[0] = (__u32)st->cur_lat;
    st->cur_hits = 0; st->cur_lat = 0; st->last_epoch_seen = cur_epoch;
}

static __always_inline bool try_mark_enqueued_this_epoch(struct bpf_page_state *st, __u32 cur_epoch) {
    __u32 prev = READ_ONCE(st->last_enq_epoch);
    if (prev == cur_epoch) return false;
    if (__sync_val_compare_and_swap(&st->last_enq_epoch, prev, cur_epoch) == prev) return true;
    return false;
}

static __always_inline void update_decision_bookkeeping(struct bpf_page_state *st, __s32 pred, __u8 reason) {
    st->last_final_action = (__u8)pred;
    st->last_reason = reason;
}

// cur_tier: 0=unknown, 1=fast(promote_node), 2=slow(demote_node)
static __always_inline int check_heuristic_filter(struct feature_row *row, __u8 cur_tier, __u32 cd_epochs) {
    /* cd_epochs==0 disables the cooldown filter. */
    if (cd_epochs != 0 && row->last_mig_epoch != 0 && row->epoch > row->last_mig_epoch && (row->epoch - row->last_mig_epoch) < cd_epochs) return -3;
    int persistent = 1; if (row->hits < HOT_THRESHOLD) persistent = 0;
    for (int i = 0; i < K_PERSISTENT - 1; i++) { if (row->hits_prev[i] < HOT_THRESHOLD) persistent = 0; }
    if (persistent) {
        if (cur_tier == 1) return -4; // hot, already on fast tier -- no action needed
        return 1;
    }
    if (row->epoch >= WARMUP_EPOCHS) {
        int cold = 1; if (row->hits > 0) cold = 0;
        for (int i = 0; i < COLD_EPOCHS - 1; i++) { if (row->hits_prev[i] > 0) cold = 0; }
        if (cold) {
if (cur_tier == 2) return -5; // cold, already on slow tier -- no action needed
            return 2;
        }
        
        __u64 sum = row->hits; __u64 sum_sq = row->hits * row->hits; int N = 1;
        for (int i=0; i<HIST_STEPS; i++) { sum += row->hits_prev[i]; sum_sq += (__u64)row->hits_prev[i] * (__u64)row->hits_prev[i]; N++; }
        __u64 variance_scaled = (sum_sq * N - sum * sum) * 100; variance_scaled /= (N * N);
        if (variance_scaled < VAR_THRESHOLD_INT) {
            // DRAM-resident low-activity pages: let ML decide (may demote)
            if (cur_tier == 1 && row->hits <= 1) return -1;
            return -2;
        }
    }
    return -1;
}

static __always_inline void enqueue_migration(__u64 page, __u32 epoch, __u32 pid, __s32 pred, __s32 score) {
    __u32 k0 = 0; struct stats_val *s_debug = bpf_map_lookup_elem(&stats_map, &k0);
    __u32 *last_enq = bpf_map_lookup_elem(&dedup_cache, &page);
    if (last_enq && *last_enq == epoch) { 
        if (s_debug) __sync_fetch_and_add(&s_debug->filter_dedup, 1); 
        return; 
    }
    
    __u32 idx_map = 0; struct region_queue *q = bpf_map_lookup_elem(&region_queue_map, &idx_map);
    if (!q) { 
        if (s_debug) __sync_fetch_and_add(&s_debug->enq_fail_lookup, 1); 
        return; 
    }
#pragma unroll
    for (int i = 0; i < 20; i++) {
        __u32 head = READ_ONCE(q->head), tail = READ_ONCE(q->tail);
        if (tail - head >= MAX_REGIONS_PER_EPOCH) {
            if (s_debug) __sync_fetch_and_add(&s_debug->enq_fail_ring_full, 1);
            return;
        }
        if (__sync_val_compare_and_swap(&q->tail, tail, tail + 1) == tail) {
            __u32 mask = MAX_REGIONS_PER_EPOCH - 1; struct region_entry *e = &q->entries[tail & mask];
            // page is a TENANT-TAGGED key; the executor migrates this address
            // in the target mm, so strip the slot bits here. dedup_cache below
            // deliberately keeps the tagged key so two tenants touching the
            // same virtual address do not dedup each other away.
            e->addr = PAGE_ADDR(page); e->pid = pid; e->score = score; e->pred = pred; e->epoch = epoch;
            smp_wmb(); WRITE_ONCE(e->seq, tail + 1);
            bpf_map_update_elem(&dedup_cache, &page, &epoch, BPF_ANY);
            
            if (s_debug) {
                __sync_fetch_and_add(&s_debug->enq_ok_ml_total, 1);
                __sync_fetch_and_add(&s_debug->eval_mig_success, 1); 
            }
            return;
        }
    }
    if (s_debug) __sync_fetch_and_add(&s_debug->enq_fail_cas_retries, 1);
}

// Userspace-policy mode: drain the candidate batch userspace computed,
// through the identical enqueue path (dedup cache, CAS ring, stats) the
// in-BPF policy uses. Invoked via bpf_prog_test_run once per pushed batch.
SEC("tc")
int userspace_enqueue(struct __sk_buff *skb) {
    __u32 k0 = 0;
    __u32 *cnt = bpf_map_lookup_elem(&us_cand_cnt, &k0);
    if (!cnt) return BPF_OK;
    __u32 n = *cnt;
    if (n > US_CAND_MAX) n = US_CAND_MAX;
#pragma clang loop unroll(disable)
    for (__u32 i = 0; i < US_CAND_MAX; i++) {
        if (i >= n) break;
        __u32 key = i;
        struct region_entry *e = bpf_map_lookup_elem(&us_cand, &key);
        if (!e) continue;
        enqueue_migration(e->addr, e->epoch, e->pid, e->pred, e->score);
    }
    *cnt = 0;
    return BPF_OK;
}

// --- MLP PIPELINE ---

SEC("tc")
int run_inference_l1(struct __sk_buff *skb) {
    __u32 k0 = 0; struct nn_scratch *scr = bpf_map_lookup_elem(&scratch_map, &k0);
    const struct qparams *qp = bpf_map_lookup_elem(&qpm, &k0);
    struct batch_ctrl *ctrl = bpf_map_lookup_elem(&batch_progress, &k0);
    if (!scr || !qp || !ctrl) return BPF_OK;

#pragma clang loop unroll(disable)
    for (int i = 0; i < L1_OUT; i++) {
        __u32 key = i; struct row1 *r = bpf_map_lookup_elem(&l1_wb, &key); struct mul *m = bpf_map_lookup_elem(&l1_mul, &key);
        if (!r || !m) continue;
        __s64 acc = (__s64)r->b;
#pragma clang loop unroll(disable)
        for (int j = 0; j < L0_OUT; j++) { __s32 xi = ((__s32)scr->l0_out[j]) - qp->out_zp0; __s32 wi = r->w[j]; acc += (__s64)xi * (__s64)wi; }
        scr->l1_out[i] = clamp_relu_u8(mul_shift_rnd(acc, m->M, m->shift) + qp->out_zp1, qp->out_zp1);
    }

    const struct rowO *rc = bpf_map_lookup_elem(&lc_wb, &k0); const struct mul *mc = bpf_map_lookup_elem(&lc_mul, &k0);
    if (rc && mc) {
        __s64 accC = (__s64)rc->b;
#pragma clang loop unroll(disable)
        for (int i = 0; i < L1_OUT; i++) { __s32 xi = ((__s32)scr->l1_out[i]) - (__s32)qp->out_zp1; __s32 wi = rc->w[i]; accC += (__s64)xi * (__s64)wi; }
        __u8 cls_u8 = clamp_u8(mul_shift_rnd(accC, mc->M, mc->shift) + qp->out_zpC);

        /*
         * ======================= SCORER ABLATION =======================
         * Swap ONLY the feature->score function. Everything after this point
         * -- histogram, dynamic threshold, top-K, probabilistic admission,
         * cooldown, capacity gate, executor, adaptive controller -- is
         * byte-identical across scorers. So any behavioural difference is
         * attributable to ranking quality and nothing else.
         *
         * The MLP still runs in every variant (its cost stays in the
         * measurement, and l0/l1 must execute to reach here), we just
         * discard its output when scorer_id != 0. That keeps CPU overhead
         * constant across scorers too, so one cannot "win" by being cheap.
         */
        {
            struct config_val *cfg_s = bpf_map_lookup_elem(&config_map, &k0);
            __u32 sid = cfg_s ? READ_ONCE(cfg_s->scorer_id) : 0;

            if (sid == XTIER_SCORER_FREQ) {
                cls_u8 = q255_log2(scr->f_hits_sum);
            } else if (sid == XTIER_SCORER_RECENCY) {
                /* inverted: recently touched == hot */
                __u8 r = q255_log2(scr->f_recency);
                cls_u8 = (__u8)(255u - r);
            } else if (sid == XTIER_SCORER_EWMA) {
                cls_u8 = q255_log2(scr->f_ewma);
            } else if (sid == XTIER_SCORER_RANDOM) {
                /* gates-only control: if this matches the MLP, the filters
                 * are doing the work and the learned ranker is not. */
                cls_u8 = (__u8)(bpf_get_prandom_u32() & 0xFFu);
            }
            /* XTIER_SCORER_MLP and ORACLE fall through: ORACLE is injected
             * by userspace via pred_map, handled outside this program. */
        }

        // Top-K budget gate: bump the MLP-output histogram for this epoch so
        // userspace can compute the Kth-percentile admission threshold. Uses
        // the active buffer (flipped by userspace before each snapshot).
        {
            struct config_val *cfg_h = bpf_map_lookup_elem(&config_map, &k0);
            // Per-tenant slice: slot*SCORE_HIST_BUCKETS + bucket. Each tenant's
            // threshold comes from its own distribution against its own budget.
            __u32 hslot = TENANT_SLOT(scr->meta.page);
            if (hslot >= XTIER_MAX_TENANTS) hslot = 0;
            __u32 bucket = hslot * SCORE_HIST_BUCKETS + (__u32)cls_u8;
            __u64 *cell;
            if (cfg_h && READ_ONCE(cfg_h->active_hist_idx) == 1) {
                cell = bpf_map_lookup_elem(&score_hist_1, &bucket);
            } else {
                cell = bpf_map_lookup_elem(&score_hist_0, &bucket);
            }
            if (cell) __sync_fetch_and_add(cell, 1);
        }

        // raw = distance from zero-point. 0 means "zero predicted future hotness" (cold page).
        // Positive means the model predicts future accesses. Negative shouldn't happen with ReLU
        // but we handle it anyway.
        __s32 raw = (__s32)cls_u8 - (__s32)qp->out_zpC;

        // Promote threshold: page must have raw > threshold to promote.
        // The threshold is set by userspace each epoch to target the top-K
        // pages by MLP score, where K tracks the configured DRAM budget.
        // When promote_threshold_dyn == 0, falls back to the legacy static
        // value of 1 (admit nearly everything -- the pre-Finding-11 behavior).
        // Userspace clamps promote_threshold_dyn to [0, 200] so the
        // promote path can never be fully starved.
        __u32 thresh_dyn = 1;
        __u32 admit_prob = 0;  // 0 = disabled (100% pass above threshold)
        {
            // Per-tenant first (static partition); fall back to the global
            // config so single-tenant runs behave exactly as before.
            __u32 tslot = TENANT_SLOT(scr->meta.page);
            if (tslot >= XTIER_MAX_TENANTS) tslot = 0;
            struct tenant_cfg *tc = bpf_map_lookup_elem(&tenant_cfg_map, &tslot);
            if (tc && READ_ONCE(tc->pid) != 0) {
                __u32 dyn = READ_ONCE(tc->promote_threshold_dyn);
                if (dyn > 0) thresh_dyn = dyn;
                admit_prob = READ_ONCE(tc->admit_prob_q16);
            } else {
                struct config_val *cfg_t = bpf_map_lookup_elem(&config_map, &k0);
                if (cfg_t) {
                    __u32 dyn = READ_ONCE(cfg_t->promote_threshold_dyn);
                    if (dyn > 0) thresh_dyn = dyn;
                    admit_prob = READ_ONCE(cfg_t->admit_prob_q16);
                }
            }
        }
        __s32 score = raw - (__s32)thresh_dyn;

        // Probabilistic admission gate, active when admit_prob is in
        // (0, 65536). This is the fallback for a saturated MLP, where the
        // score carries no ranking signal and the only lever left is to
        // rate-limit volume. Applied after the threshold check, so it only
        // thins pages that already passed; pages below threshold still NOOP.
        if (score > 0 && admit_prob > 0 && admit_prob < 65536) {
            __u32 r = bpf_get_prandom_u32() & 0xFFFF;
            if (r >= admit_prob) {
                score = 0;  // force NOOP -- didn't win the probabilistic admission
            }
        }
        struct bpf_page_state *st = bpf_map_lookup_elem(&page_state_map, &scr->meta.page);
        struct stats_val *s = bpf_map_lookup_elem(&stats_map, &k0);

        // ML outputs a continuous benefit score. Only PROMOTE when score > 0.
        // When score <= 0, NOOP (skip) -- do NOT enqueue a demotion. Demotions
        // are handled by the kernel PTE cold scanner which is cheaper (no ring
        // buffer, no per-page TLB shootdown in the data path). ML-driven
        // demotions added 47K extra migrations and 600K TLB shootdowns in
        // testing -- more overhead than they saved.
        struct feature_key fkey = { .epoch = scr->meta.epoch, .page = scr->meta.page };
        struct pred_row pv = { .score = score, .pred = (score > 0) ? 1 : 0 };
        bpf_map_update_elem(&pred_map, &fkey, &pv, BPF_ANY);

        if (score > 0) {
            // ML says: this page has high predicted benefit -- promote it.
            __u8 cur_tier = st ? READ_ONCE(st->tier) : 0;
            if (cur_tier == 1) {
                // Already on DRAM -- noop
                if (st) update_decision_bookkeeping(st, 0, 6);
                struct config_val *cfg = bpf_map_lookup_elem(&config_map, &k0);
                if (s && cfg && cfg->mode >= 2) {
                    __sync_fetch_and_add(&s->eval_dec_total, 1);
                    __sync_fetch_and_add(&s->eval_dec_noop, 1);
                }
            } else {
                enqueue_migration(scr->meta.page, scr->meta.epoch, scr->meta.pid, 1, score);
                if (st) update_decision_bookkeeping(st, 1, 3);
                struct config_val *cfg = bpf_map_lookup_elem(&config_map, &k0);
                if (s && cfg && cfg->mode >= 2) {
                    __sync_fetch_and_add(&s->eval_dec_total, 1);
                    __sync_fetch_and_add(&s->eval_dec_promote, 1);
                }
            }
        } else {
            // ML says: low/zero benefit -- skip (noop). Don't demote.
            // The PTE cold scanner handles DRAM eviction independently.
            if (st) update_decision_bookkeeping(st, 0, 6);
            struct config_val *cfg = bpf_map_lookup_elem(&config_map, &k0);
            if (s && cfg && cfg->mode >= 2) {
                __sync_fetch_and_add(&s->eval_dec_total, 1);
                __sync_fetch_and_add(&s->eval_dec_noop, 1);
            }
        }
        
        if (s) { 
            __sync_fetch_and_add(&s->total_inferences, 1); 
            if (score > 0) __sync_fetch_and_add(&s->ml_score_pos, 1); 
            else if (score < 0) __sync_fetch_and_add(&s->ml_score_neg, 1); 
            else __sync_fetch_and_add(&s->ml_score_zero, 1);
            
            // [FIX] Stage 1 Timing
            __u64 now = bpf_ktime_get_ns();
            if (now > scr->start_ts) {
                __sync_fetch_and_add(&s->t_stage1_accum_ns, (now - scr->start_ts));
            }
        }
    }
    ctrl->idx++; bpf_tail_call(skb, &jmp_table, 3);
    return BPF_OK;
}

SEC("tc")
int run_inference_l0(struct __sk_buff *skb) {
    __u32 k0 = 0; struct nn_scratch *scr = bpf_map_lookup_elem(&scratch_map, &k0);
    const struct qparams *qp = bpf_map_lookup_elem(&qpm, &k0);
    if (!scr || !qp) return BPF_OK;
#pragma clang loop unroll(disable)
    for (int i = 0; i < L0_OUT; i++) {
        __u32 key = i; struct row0 *r = bpf_map_lookup_elem(&l0_wb, &key); struct mul *m = bpf_map_lookup_elem(&l0_mul, &key);
        if (!r || !m) continue;
        __s64 acc = (__s64)r->b;
#pragma clang loop unroll(disable)
        for (int j = 0; j < L0_IN; j++) { __s32 xi = ((__s32)scr->input[j]) - qp->in_zp0; __s32 wi = ((__s32)r->w[j]); acc += (__s64)xi * (__s64)wi; }
        scr->l0_out[i] = clamp_relu_u8(mul_shift_rnd(acc, m->M, m->shift) + qp->out_zp0, qp->out_zp0);
    }
    bpf_tail_call(skb, &jmp_table, 2); 
    return BPF_OK;
}

SEC("tc")
int run_inference_batch(struct __sk_buff *skb) {
    __u32 k0 = 0; struct nn_scratch *scr = bpf_map_lookup_elem(&scratch_map, &k0);
    struct batch_ctrl *ctrl = bpf_map_lookup_elem(&batch_progress, &k0);
    if (!scr || !ctrl) return BPF_OK;

    __u32 *epoch_ptr = (void *)(long)skb->data;
    if ((void *)(epoch_ptr + 1) > (void *)(long)skb->data_end) return BPF_OK;
    __u32 incoming_epoch = *epoch_ptr;
    if (ctrl->epoch != incoming_epoch) { ctrl->idx = 0; ctrl->epoch = incoming_epoch; }

    __u32 epoch = ctrl->epoch;
    __u32 *cnt_ptr = (epoch % 2 == 0) ? bpf_map_lookup_elem(&count_0, &k0) : bpf_map_lookup_elem(&count_1, &k0);
    if (!cnt_ptr) return BPF_OK;
    __u32 n = *cnt_ptr; if (n > MAX_PAGES_PER_EPOCH) n = MAX_PAGES_PER_EPOCH;
    __u32 i = ctrl->idx;
    if (i >= n) { ctrl->idx = 0; return BPF_OK; } 

    __u64 *pg_ptr = (epoch % 2 == 0) ? bpf_map_lookup_elem(&epoch_pages_0, &i) : bpf_map_lookup_elem(&epoch_pages_1, &i);
    if (!pg_ptr || !*pg_ptr) { ctrl->idx++; bpf_tail_call(skb, &jmp_table, 3); return BPF_OK; }

    __u64 page = *pg_ptr;
    struct bpf_page_state *st = bpf_map_lookup_elem(&page_state_map, &page);
    if (!st) { ctrl->idx++; bpf_tail_call(skb, &jmp_table, 3); return BPF_OK; }

    struct feature_row row = { .epoch = epoch, .page = PAGE_ADDR(page), .pid = st->owner_pid, .last_mig_epoch = st->last_mig_epoch, .recency_epochs = st->recency_epochs, .epochs_since_promotion = st->recency_epochs, .off_bucket = off_bucket_calc(PAGE_ADDR(page)) };
    if (st->last_epoch_seen == epoch) {
        row.hits = (__u32)st->cur_hits; row.lat_sum = (__u32)st->cur_lat;
        for (int h=0; h<6; h++) { row.hits_prev[h] = st->hits_prev[h]; row.lat_prev[h] = st->lat_prev[h]; }
    } else if (epoch == st->last_epoch_seen + 1) {
        row.hits_prev[0] = (__u32)st->cur_hits; row.lat_prev[0] = (__u32)st->cur_lat;
        for(int h=1; h<6; h++) { row.hits_prev[h] = st->hits_prev[h-1]; row.lat_prev[h] = st->lat_prev[h-1]; }
    }
    row.delta_hits[0] = (__s32)row.hits - (__s32)row.hits_prev[0];
    for(int d=1; d<6; d++) row.delta_hits[d] = (__s32)row.hits_prev[d-1] - (__s32)row.hits_prev[d];

    // Must reproduce stage0's tagging exactly: region base of the REAL address,
    // then the slot re-applied. Masking `page` directly would strip the slot
    // (REG_SHIFT > PAGE_SHIFT) and miss meta_map for every tenant.
    __u64 reg_base = (PAGE_ADDR(page) & ~((__u64)((1ull << REG_SHIFT) - 1ull))) | TENANT_SLOT(page);
    struct meta_val *mv = bpf_map_lookup_elem(&meta_map, &reg_base);
    if (mv) { row.perm_r = mv->perm_r; row.perm_w = mv->perm_w; row.perm_x = mv->perm_x; row.page_type = mv->page_type; row.vma_size_bucket = mv->vma_size_bucket; row.anon = mv->anon; row.shared = mv->shared; }

    struct feature_key fkey = { .epoch = epoch, .page = page };
    bpf_map_update_elem(&feature_row_map, &fkey, &row, BPF_ANY);

    struct config_val *cfg = bpf_map_lookup_elem(&config_map, &k0);
    if (cfg && cfg->mode == 1) { ctrl->idx++; bpf_tail_call(skb, &jmp_table, 3); return BPF_OK; }

    __u8 cur_tier = READ_ONCE(st->tier);

    // Page-type pinning safety net (normally caught in stage0 early-skip)
    if (mv) {
        if (mv->page_type == 3) {
            // Stack: pin to DRAM. Force promote if on CXL, noop if already DRAM.
            if (cur_tier == 2 || cur_tier == 0) {
                enqueue_migration(page, epoch, st->owner_pid, 1, 999);
                struct pred_row pv = {999, 1}; bpf_map_update_elem(&pred_map, &fkey, &pv, BPF_ANY);
                update_decision_bookkeeping(st, 1, 1);
            } else {
                update_decision_bookkeeping(st, 0, 6);
            }
            ctrl->idx++; bpf_tail_call(skb, &jmp_table, 3); return BPF_OK;
        }
        if (mv->page_type == 1) {
            // Executable: pin to CXL. Never promote.
            update_decision_bookkeeping(st, 0, 6);
            ctrl->idx++; bpf_tail_call(skb, &jmp_table, 3); return BPF_OK;
        }
    }

    __u32 cd_epochs = COOLDOWN_EPOCHS;
    { struct config_val *cfg_cd = bpf_map_lookup_elem(&config_map, &k0);
      if (cfg_cd) cd_epochs = READ_ONCE(cfg_cd->cooldown_epochs); }
    int heuristic = check_heuristic_filter(&row, cur_tier, cd_epochs);
    // HEURISTICS DISABLED -- ML is the sole decision-maker for promotions.
    // Only keep safety guards:
    //   -3 = cooldown (prevent thrash on recently migrated pages)
    //   -4 = hot + already on DRAM (noop, nothing to do)
    //   -5 = cold + already on CXL (noop, nothing to do)
    // Everything else (promote candidates, cold candidates, ambiguous,
    // stable variance) falls through to ML.
    if (heuristic == -3 || heuristic == -4 || heuristic == -5) {
        update_decision_bookkeeping(st, 0, (heuristic == -3 ? 5 : 6));
        ctrl->idx++; bpf_tail_call(skb, &jmp_table, 3); return BPF_OK;
    }
    // ALL other pages -> ML decides (promote, demote, or noop)

    scr->meta.epoch = epoch; scr->meta.pid = st->owner_pid; scr->meta.page = page;
    scr->start_ts = bpf_ktime_get_ns();
    __builtin_memset(scr->input, 64, 32);
    scr->input[0] = (__u8)(epoch & 0xFF);
    scr->input[1] = log_scale(row.hits);
    scr->input[2] = log_scale(row.hits); // lat_sum == hits in current impl; keep consistent with training
    #pragma unroll
    for (int h = 0; h < 6; h++) {
        scr->input[3+h*3+0] = log_scale(row.hits_prev[h]);
        scr->input[3+h*3+1] = log_scale(row.hits_prev[h]); // lat_prev == hits_prev in current impl
        scr->input[3+h*3+2] = pack_delta(row.delta_hits[h]);
    }
    scr->input[21] = log_scale((__u32)row.recency_epochs);
    scr->input[22] = log_scale((__u32)row.epochs_since_promotion);

    // --- stash raw features for the alternate scorers ---
    {
        __u32 sum = row.hits;
        __u32 ew  = row.hits << 5;   // weight 32 for the current epoch
        #pragma unroll
        for (int h = 0; h < 6; h++) {
            sum += row.hits_prev[h];
            ew  += row.hits_prev[h] << (4 - (h > 4 ? 4 : h));  // 16,8,4,2,1,1
        }
        scr->f_hits_now = row.hits;
        scr->f_hits_sum = sum;
        scr->f_ewma     = ew;
        scr->f_recency  = (__u32)row.recency_epochs;
    }
    // Slots 23-24: anon/shared VMA flags from meta_map.
    // Slots 25-26: unused, remain 64 (neutral) from memset above.
    // Slots 27-31: permissions and page type from meta_map.
    if (mv) {
        scr->input[23] = mv->anon;
        scr->input[24] = mv->shared;
        scr->input[27] = mv->perm_r;
        scr->input[28] = mv->perm_w;
        scr->input[29] = mv->perm_x;
        scr->input[30] = mv->page_type;
        scr->input[31] = mv->vma_size_bucket;
    }

    bpf_tail_call(skb, &jmp_table, 1);
    return BPF_OK;
}

SEC("perf_event")
int on_pebs_stage0(struct bpf_perf_event_data *ctx) {
    __u64 t_start = bpf_ktime_get_ns();
    __u32 tgid = (__u32)(bpf_get_current_pid_tgid() >> 32);
    __u32 k0 = 0; struct stats_val *s = bpf_map_lookup_elem(&stats_map, &k0);
    
    if (s) {
        __sync_fetch_and_add(&s->pebs_seen, 1);
    }
    
    // With per-task PEBS attachment (pid=target, cpu=-1), only target-process loads
    // reach this handler. This check is now redundant but kept as a safety guard.
    // target_pids value is this tenant's SLOT, used to namespace every
    // page-keyed map below. Single-tenant runs get slot 0, which makes every
    // key identical to the pre-multi-tenancy layout.
    __u8 *slot_p = bpf_map_lookup_elem(&target_pids, &tgid);
    if (!slot_p) {
        if (s) __sync_fetch_and_add(&s->lock_dropped, 1);
        return 0;
    }
    __u32 slot = *slot_p;
    if (slot >= XTIER_MAX_TENANTS) slot = 0;
    
    // [FIX] Stage 0 Trigger Count (Divisor for latency)
    if (s) __sync_fetch_and_add(&s->pebs_triggered, 1);

    struct config_val *cfg = bpf_map_lookup_elem(&config_map, &k0); if (!cfg) return 0;
    if (!ctx->addr) return 0;


    // NOTE: `page` is the tenant-TAGGED key from here down, not a bare
    // address. Its low PAGE_SHIFT bits carry the slot. Use PAGE_ADDR(page)
    // anywhere it must be a real virtual address.
    __u64 page = TENANT_KEY(ctx->addr, slot);

    // Userspace-policy counterfactual: degrade to a plain sample
    // forwarder. Everything the in-BPF path does per sample (state roll,
    // early skips, epoch queue) and per epoch (filters, MLP, gates) is done
    // in userspace instead, on this forwarded stream. A failed reserve means
    // the ring overran because userspace fell behind -- count it and move on.
    if (READ_ONCE(cfg->userspace_policy)) {
        struct us_sample smp = { .page = PAGE_ADDR(page), .ts_ns = t_start, .pid = tgid };
        if (bpf_ringbuf_output(&us_ring, &smp, sizeof(smp), 0) == 0) {
            if (s) __sync_fetch_and_add(&s->us_fwd_ok, 1);
        } else {
            if (s) __sync_fetch_and_add(&s->us_fwd_drop, 1);
        }
        if (s) { __u64 t_end = bpf_ktime_get_ns(); __sync_fetch_and_add(&s->t_stage0_accum_ns, (t_end - t_start)); }
        return 0;
    }

    // Page-type-aware early skip: stack pages are pinned to DRAM and
    // executable pages are pinned to CXL. Neither needs tracking, so
    // skip all BPF processing (bloom filter, epoch queue, ML inference).
    // We can't avoid the hardware PEBS interrupt, but this eliminates
    // the per-sample BPF overhead (~1-2us saved per skipped sample).
    {
        __u64 rg = (PAGE_ADDR(page) & ~((__u64)((1ull << REG_SHIFT) - 1ull))) | slot;
        struct meta_val *mv = bpf_map_lookup_elem(&meta_map, &rg);
        if (mv && (mv->page_type == 3 || mv->page_type == 1)) {
            if (s) { __u64 t_end = bpf_ktime_get_ns(); __sync_fetch_and_add(&s->t_stage0_accum_ns, (t_end - t_start)); }
            return 0;
        }
    }

    struct bpf_page_state *st = bpf_map_lookup_elem(&page_state_map, &page);
    if (!st) { struct bpf_page_state zero = { .tier = 2 }; bpf_map_update_elem(&page_state_map, &page, &zero, BPF_NOEXIST); st = bpf_map_lookup_elem(&page_state_map, &page); if (!st) return 0; }

    // Seeded to the slow tier: the run configuration allocates there and
    // promotes upward. The executor rewrites it from the real node on contact.

    if (s && cfg->mode >= 2) {
        __sync_fetch_and_add(&s->eval_samples_total, 1);
        __u8 tier = READ_ONCE(st->tier);
        if (tier == 1) __sync_fetch_and_add(&s->eval_samples_t1, 1);
        else if (tier == 2) __sync_fetch_and_add(&s->eval_samples_t2, 1);
        else __sync_fetch_and_add(&s->eval_samples_unknown, 1);
    }

    roll_state_if_needed(st, cfg->cur_epoch);
    st->cur_hits++; st->cur_lat += 1; st->owner_pid = tgid;

    // FAST PATH: page already enqueued this epoch -- skip bloom filter + CAS enqueue.
    // The hit is counted above; no need to re-add to epoch queue or update fp ring.
    // This saves ~500ns per re-hit (set_fp_bit + try_mark ops) for hot pages.
    if (READ_ONCE(st->last_enq_epoch) == cfg->cur_epoch) {
        if (s) { __u64 t_end = bpf_ktime_get_ns(); __sync_fetch_and_add(&s->t_stage0_accum_ns, (t_end - t_start)); }
        return 0;
    }

    // EARLY SKIP for tier==1 (DRAM-resident) pages.
    // Promotion would be a noop, and cold detection is now handled by the
    // kernel-side PTE Accessed-bit scanner in xtier_executor.c -- which is
    // strictly more accurate than the PEBS-absence signal we used to rely on.
    // Skipping enqueue here saves the entire stage0->stage1 ML pipeline cost
    // for the (large) majority of samples that land on pages already in DRAM.
    if (READ_ONCE(st->tier) == 1) {
        if (s) { __u64 t_end = bpf_ktime_get_ns(); __sync_fetch_and_add(&s->t_stage0_accum_ns, (t_end - t_start)); }
        return 0;
    }

    set_fp_bit(page, cfg->cur_epoch);
    if (try_mark_enqueued_this_epoch(st, cfg->cur_epoch)) {
        __u32 *cnt = (cfg->cur_epoch % 2 == 0) ? bpf_map_lookup_elem(&count_0, &k0) : bpf_map_lookup_elem(&count_1, &k0);
        if (cnt) { __u32 n = *cnt; if (n < MAX_PAGES_PER_EPOCH) { if (cfg->cur_epoch % 2 == 0) bpf_map_update_elem(&epoch_pages_0, &n, &page, BPF_ANY); else bpf_map_update_elem(&epoch_pages_1, &n, &page, BPF_ANY); __sync_fetch_and_add(cnt, 1); } }
    } else if (s) __sync_fetch_and_add(&s->ml_drop_cooldown, 1);
    
    // [FIX] Stage 0 Timing - Accumulate here
    if (s) { __u64 t_end = bpf_ktime_get_ns(); __sync_fetch_and_add(&s->t_stage0_accum_ns, (t_end - t_start)); }
    return 0;
}
