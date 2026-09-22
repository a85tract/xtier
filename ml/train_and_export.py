#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Trains the page-benefit MLP, quantizes it to INT8 with fixed quantization
# parameters, and emits the C header that the loader compiles in and pushes
# into the BPF program's maps.
#
# The emitted header is GPL-2.0 -- see LICENSING.md.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization as tq
from torch.ao.quantization.observer import FixedQParamsObserver
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- CONFIGURATION ---
INPUT_CSV = "features.csv"
OUTPUT_HEADER = os.environ.get(
    "XTIER_OUTPUT_HEADER",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "mlp_q8.h"),
)
EPOCHS = int(os.environ.get('XTIER_EPOCHS', '8'))   # XTIER_EPOCHS to train longer
PATIENCE = 10            
BATCH_SIZE = 1024        
LR = 0.005               

# Constants
L0_IN = 32
L0_OUT = 192
L1_OUT = 96

# --- 1. DATA PREP ---
def log_scale_np(arr):
    arr = arr.fillna(0).astype(np.int32)
    res = 70 + (arr - 100) // 50
    mask = arr <= 100
    res[mask] = 60 + (arr[mask] - 20) // 10
    mask = arr <= 20
    res[mask] = 52 + (arr[mask] - 5) // 2
    mask = arr <= 5
    res[mask] = 32 + arr[mask] * 4
    mask = arr == 0
    res[mask] = 32
    # CLIP to the byte range. The BPF side returns __u8, so a value above 255
    # TRUNCATES there (hits=12963171 -> 259331 -> byte 3, i.e. the hottest page
    # reads COLDER than a page with one hit). Training previously fed the raw
    # unclamped int, which the input quantizer then clamped to 255 -- so the
    # same page encoded as 255 in training and 3 at runtime. Both sides now
    # saturate at 255. Keep in sync with log_scale() in pebs_mlp_kern.c.
    return np.clip(res, 0, 255)

def parse_hex_col(series):
    s = series.fillna('0').astype(str).str.slice(0, 16)
    return s.apply(lambda x: int(x, 16) if x != '' else 0)

def prepare_data(csv_path, label_mode="benefit"):
    print(f"[*] Loading {csv_path} (Vectorized Mode)...")
    try:
        # Cap rows to keep training tractable on 5+ GB feature dumps.
        df = pd.read_csv(csv_path, nrows=5_000_000)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        exit(1)

    df = df.sort_values(['page', 'epoch'])

    if label_mode == "binary":
        # --- LEGACY: Binary classification ---
        print("[*] Using BINARY labels (legacy mode)")
        HOT_THRESHOLD_TRAIN = 2
        indexer = df.groupby('page')['hits']
        df['next_hits'] = indexer.shift(-1).fillna(0) + indexer.shift(-2).fillna(0) + indexer.shift(-3).fillna(0)
        df = df.dropna(subset=['next_hits'])
        df['target'] = (df['next_hits'] >= HOT_THRESHOLD_TRAIN).astype(np.float32) * 200.0
        scale_factor = 1

        n_hot = (df['target'] > 0).sum()
        n_cold = (df['target'] == 0).sum()
        print(f"[*] Binary split -- hot (target=200): {n_hot}, cold (target=0): {n_cold}")

        # Balanced 1:1 sampling
        pos = df[df['target'] > 0]
        neg = df[df['target'] == 0]
        n_min = min(len(pos), len(neg))
        pos = pos.sample(n=n_min, random_state=42)
        neg = neg.sample(n=n_min, random_state=42)
        df_bal = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"    Final Training Size: {len(df_bal)} ({n_min} hot + {n_min} cold)")

    else:
        # --- NEW: Continuous benefit score ---
        # For each (page, epoch) observation, compute:
        #   reuse_intensity = sum of hits on this page in epochs E+1..E+10
        #   tenure = how many of those 10 epochs have hits > 0
        #   reuse_speed = epochs until first re-access (1..10, cap at 10)
        #   benefit = reuse_intensity * (tenure/10) / (1 + reuse_speed)
        #
        # This scores high for pages re-accessed soon, frequently, and
        # persistently. Pages touched once and never again score ~0.
        # The model learns "how valuable is promoting this page" instead
        # of "is this page hot or cold."
        print("[*] Using BENEFIT SCORE labels (continuous)")

        # VECTORIZED benefit computation.
        # The original used df.iterrows() over every row with a 10-deep inner
        # lookup plus a per-page dict built from groupby -- O(minutes-to-hours)
        # and heavy on RAM at multi-million-row scale. This produces the SAME
        # numbers via 10 MultiIndex reindex operations.
        #
        # Semantics preserved exactly:
        #   reuse_intensity = sum of hits at (page, e+d) for d in 1..10
        #   tenure          = count of d in 1..10 with hits > 0
        #   reuse_speed     = smallest d in 1..9 with hits > 0, else 10
        #                     (note: d == 10 does NOT set reuse_speed, matching
        #                      the original's `if reuse_speed == 10 and delta < 10`)
        #   benefit = reuse_intensity * (tenure/10) / (1 + reuse_speed)
        import time as _t
        _t0 = _t.time()
        # `page` arrives as object dtype (hex address strings). Hashing object
        # keys in a 3.87M-row MultiIndex is superlinear in distinct-key count
        # and dominates runtime. factorize() maps them to int64 codes once, so
        # the reindex below is integer-keyed. Codes are per-run identifiers,
        # only ever compared for equality, so this cannot change the labels.
        _pg = pd.factorize(df['page'])[0].astype(np.int64)
        _ep = df['epoch'].astype(np.int64).to_numpy()
        # dict(zip(...)) keeps the LAST value for duplicate (page, epoch); match that.
        _lookup = pd.Series(df['hits'].to_numpy(),
                            index=pd.MultiIndex.from_arrays([_pg, _ep])
                            ).groupby(level=[0, 1]).last()

        _intensity = np.zeros(len(df), dtype=np.float64)
        _tenure    = np.zeros(len(df), dtype=np.float64)
        _speed     = np.full(len(df), 10, dtype=np.float64)
        for _d in range(1, 11):
            _key = pd.MultiIndex.from_arrays([_pg, _ep + _d])
            _fh = _lookup.reindex(_key).to_numpy(dtype=np.float64, na_value=0.0)
            _hit = _fh > 0
            _intensity += np.where(_hit, _fh, 0.0)
            _tenure    += _hit
            if _d < 10:
                _speed = np.where(_hit & (_speed == 10), float(_d), _speed)

        df['benefit_raw'] = _intensity * (_tenure / 10.0) / (1.0 + _speed)
        print(f"[*] benefit computed (vectorized) in {_t.time()-_t0:.1f}s")

        # Scale benefit to quantized output range [0, 200].
        #
        # The benefit distribution is a POINT MASS AT ZERO with a heavy tail
        # (median 0, max ~2.6e5, ~99.5% of rows have no future reuse in the
        # 10-epoch window). Clipping at the 99.9th percentile and scaling
        # linearly therefore collapses ~all targets to 0, and the model learns
        # a constant function -- no ranking signal. This saturation is the
        # central difficulty in learning from this label.
        #
        # log1p compresses the tail. It is MONOTONE, so ranking by predicted
        # log-benefit is identical to ranking by benefit, which is all top-K
        # needs. Balancing (below) is what actually supplies the signal.
        # XTIER_LABEL_TF selects the target transform:
        #   log1p : log1p(benefit) scaled to 0..200      (compresses the tail)
        #   rank  : percentile rank of benefit * 200     (DEFAULT)
        #
        # 'rank' exists because top-K is a RANKING consumer, not a regression
        # consumer. With log1p the target is still ~95% near-zero, so the model
        # learns the (true but useless) fact that almost every page is cold and
        # emits a near-constant score -- measured at runtime as 1-3 populated
        # score buckets vs 16 for trivial frequency counting. A percentile-rank
        # target is uniform by construction, so the model is forced to spread
        # its output across the range and top-K gets something to cut on.
        # Rank is monotone in benefit, so it preserves the ordering that matters.
        _tf = os.environ.get("XTIER_LABEL_TF", "rank")
        _b = df['benefit_raw'].to_numpy(dtype=np.float64)
        if _tf == "log1p":
            _lb = np.log1p(_b)
            _mx = _lb.max() if _lb.max() > 0 else 1.0
            df['target'] = np.clip(_lb / _mx * 200.0, 0, 200).astype(np.float32)
            scale_factor = _mx / 200.0
        else:
            # Rank ONLY within the non-zero rows. Ranking the whole column
            # would collapse the ~95% tied zeros onto a single rank and give a
            # bimodal target, not a uniform one. Zeros stay at 0 (genuinely
            # cold); reused pages spread uniformly over 20..200 so the model
            # has to discriminate AMONG promote candidates, which is exactly
            # what top-K consumes.
            _r = np.zeros(len(_b), dtype=np.float64)
            _nzm = _b > 0
            if _nzm.sum() > 0:
                _r[_nzm] = pd.Series(_b[_nzm]).rank(method='average', pct=True).to_numpy()
            df['target'] = np.where(_nzm, 20.0 + _r * 180.0, 0.0).astype(np.float32)
            scale_factor = 1.0 / 200.0  # quantized -> percentile rank among reused
            _mx = 1.0
        print(f"[*] Label transform: {_tf}")

        print(f"[*] Benefit score stats (raw):")
        print(f"    min={df['benefit_raw'].min():.2f}, median={df['benefit_raw'].median():.2f}, "
              f"mean={df['benefit_raw'].mean():.2f}, max={df['benefit_raw'].max():.2f}")
        print(f"    log1p scale: max_log={_mx:.3f} (target = log1p(benefit)/max_log*200)")
        print(f"[*] Target stats (scaled to 0-200):")
        print(f"    min={df['target'].min():.1f}, median={df['target'].median():.1f}, "
              f"mean={df['target'].mean():.1f}, max={df['target'].max():.1f}")

        # No balanced sampling needed -- the continuous distribution IS the signal.
        # Just shuffle and use all data.
        n_zero = (df['target'] < 1).sum()
        n_nonzero = (df['target'] >= 1).sum()
        print(f"    Near-zero (target < 1): {n_zero} ({100*n_zero/len(df):.1f}%)")
        print(f"    Non-zero (target >= 1): {n_nonzero} ({100*n_nonzero/len(df):.1f}%)")

        # Balance. The original code asserted "no balanced sampling needed --
        # the continuous distribution IS the signal", which holds only if the
        # distribution is genuinely continuous. At 99.5% zeros it is not, and
        # training on it yields a constant predictor. Keep every non-zero row
        # and subsample zeros to XTIER_BAL_RATIO x that count.
        _bal_ratio = float(os.environ.get("XTIER_BAL_RATIO", "3"))
        _nz = df[df['target'] >= 1]
        _z  = df[df['target'] < 1]
        _keep_z = min(len(_z), int(len(_nz) * _bal_ratio))
        if len(_nz) > 0 and _keep_z > 0:
            df_bal = pd.concat([_nz, _z.sample(n=_keep_z, random_state=42)])
            df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"    Balanced {_bal_ratio:.0f}:1 -> {len(_nz)} non-zero + {_keep_z} zero")
        else:
            df_bal = df.sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"    WARNING: cannot balance (non-zero={len(_nz)}); using all rows")
        print(f"    Target spread after balancing: "
              f"p50={df_bal['target'].quantile(0.5):.1f} p90={df_bal['target'].quantile(0.9):.1f} "
              f"p99={df_bal['target'].quantile(0.99):.1f} max={df_bal['target'].max():.1f}")
        print(f"    Final Training Size: {len(df_bal)}")





    # --- FEATURE MATRIX ---
    print(f"[*] Constructing Feature Matrix...")
    N = len(df_bal)
    X_np = np.full((N, 32), 64, dtype=np.float32)
    
    def process_delta(series):
        v = series.fillna(0).values.astype(int)
        v = np.clip(v, -128, 127)
        return (v + 128) & 0xFF

    # 1. Basics
    X_np[:, 0] = df_bal['epoch'].values & 0xFF
    X_np[:, 1] = log_scale_np(df_bal['hits'])
    X_np[:, 2] = log_scale_np(df_bal['lat_sum'])
    
    # 2. History
    for i in range(6): 
        base = 3 + (i * 3)
        csv_idx = i + 1 
        X_np[:, base + 0] = log_scale_np(df_bal[f'hits_prev_{csv_idx}'])
        X_np[:, base + 1] = log_scale_np(df_bal[f'lat_prev_{csv_idx}'])
        X_np[:, base + 2] = process_delta(df_bal[f'delta_hits_{csv_idx}'])

    # 3. Timers
    X_np[:, 21] = log_scale_np(df_bal['recency_epochs'])
    X_np[:, 22] = log_scale_np(df_bal['epochs_since_promotion'])
    
    # 4. anon/shared VMA flags (slots 23-24); slots 25-26 unused, remain at default 64
    X_np[:, 23] = df_bal['anon'].fillna(0).astype(int).values & 0xFF
    X_np[:, 24] = df_bal['shared'].fillna(0).astype(int).values & 0xFF
    # slots 25-26 intentionally left at 64 (neutral); BPF also provides 64 for these

    # 5. Metadata
    X_np[:, 27] = df_bal['perm_r'].fillna(0).astype(int).values & 0xFF
    X_np[:, 28] = df_bal['perm_w'].fillna(0).astype(int).values & 0xFF
    X_np[:, 29] = df_bal['perm_x'].fillna(0).astype(int).values & 0xFF
    X_np[:, 30] = df_bal['page_type'].fillna(0).astype(int).values & 0xFF
    X_np[:, 31] = df_bal['vma_size_bucket'].fillna(0).astype(int).values & 0xFF

    Y = df_bal['target'].values.astype(np.float32)
    return X_np, Y, scale_factor

# --- 2. MODEL ---
class RegMLP(nn.Module):
    def __init__(self):
        super(RegMLP, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.fc0 = nn.Linear(L0_IN, L0_OUT)
        self.relu0 = nn.ReLU()
        self.drop0 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(L0_OUT, L1_OUT)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.1)
        
        self.fc_head = nn.Linear(L1_OUT, 1)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu0(self.fc0(x))
        x = self.drop0(x)
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)
        x = self.fc_head(x) 
        x = self.dequant(x)
        return x

# --- 3. TRAINING WITH WEIGHTED LOSS ---
def train_and_eval(X_train, X_val, Y_train, Y_val):
    Xt = torch.tensor(X_train)
    Yt = torch.tensor(Y_train).unsqueeze(1)
    Xv = torch.tensor(X_val)
    Yv = torch.tensor(Y_val).unsqueeze(1)
    
    model = RegMLP()
    
    default_qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    model.qconfig = default_qconfig
    
    input_observer = FixedQParamsObserver.with_args(
        scale=1.0, zero_point=0, dtype=torch.quint8, quant_min=0, quant_max=255
    )
    model.quant.qconfig = torch.ao.quantization.QConfig(activation=input_observer, weight=default_qconfig.weight)

    output_observer = FixedQParamsObserver.with_args(
        scale=1.0, zero_point=32, dtype=torch.quint8, quant_min=0, quant_max=255
    )
    model.fc_head.qconfig = torch.ao.quantization.QConfig(activation=output_observer, weight=default_qconfig.weight)
    
    torch.ao.quantization.prepare_qat(model, inplace=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # --- KEY CHANGE: WEIGHTED LOSS ---
    # Since we have 9x more negatives, we must weight positives higher 
    # so the model doesn't just predict 0 for everything.
    criterion = nn.HuberLoss(delta=5.0, reduction='none') 
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("-" * 65)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val RMSE':<8} | {'LR':<8}")
    print("-" * 65)
    
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(Xt.size()[0])
        train_loss = 0
        
        for i in range(0, Xt.size()[0], BATCH_SIZE):
            indices = permutation[i:i+BATCH_SIZE]
            batch_x, batch_y = Xt[indices], Yt[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # Unweighted loss: the 1:9 class balance already supplies the
            # signal, and up-weighting hot pages on top of it overfits them.
            loss = criterion(outputs, batch_y).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= (len(Xt) / BATCH_SIZE)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        model.eval()
        with torch.no_grad():
            val_preds = model(Xv)
            # Validation loss doesn't need weighting, we want real error
            val_loss = nn.HuberLoss(delta=5.0)(val_preds, Yv).item()
            mse = mean_squared_error(Yv.numpy(), val_preds.numpy())
            rmse = math.sqrt(mse)
            
        print(f"{epoch+1:<6} | {train_loss:<10.4f} | {val_loss:<10.4f} | {rmse:<8.4f} | {current_lr:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
                
    model.load_state_dict(best_model_state)
    return model

# --- 4. EXPORT ---
def get_quant_params(layer, input_scale):
    w = layer.weight()
    w_int8 = w.int_repr().numpy().astype(np.int8)
    if w.qscheme() == torch.per_tensor_affine:
        w_scales = np.array([w.q_scale()] * layer.out_features, dtype=np.float32)
    else:
        w_scales = w.q_per_channel_scales().numpy().astype(np.float32)
    
    bias_float = layer.bias().detach().numpy()
    safe_scales = np.where(w_scales == 0, 1e-9, w_scales)
    bias_int32 = np.round(bias_float / (input_scale * safe_scales)).astype(np.int32)
    return w_int8, bias_int32, w_scales, layer.scale, int(layer.zero_point)

def export_header(model_int8, output_file, scale_factor):
    print(f"[*] Exporting weights to {output_file}...")
    l0 = model_int8.fc0
    l1 = model_int8.fc1
    lc = model_int8.fc_head
    
    in_scale = model_int8.quant.scale.item()
    in_zp = int(model_int8.quant.zero_point.item())
    
    w0, b0, s0, out_s0, out_z0 = get_quant_params(l0, in_scale)
    w1, b1, s1, out_s1, out_z1 = get_quant_params(l1, out_s0) 
    wc, bc, sc, out_sC, out_zC = get_quant_params(lc, out_s1) 
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        # GPL-2.0 because this header is compiled into the loader and its
        # weights are pushed into the BPF program. The licence of the
        # generator does not bind the licence of what it generates.
        f.write("/* SPDX-License-Identifier: GPL-2.0 */\n")
        f.write("/*\n")
        f.write(" * Generated by ml/train_and_export.py -- do not edit.\n")
        f.write(" *\n")
        f.write(" * INT8-quantized MLP weights. Compiled into the userspace loader and\n")
        f.write(" * pushed into the BPF program's maps at startup, so this file is part\n")
        f.write(" * of the GPL-licensed build even though its generator is not.\n")
        f.write(" */\n")
        f.write("#ifndef MLP_Q8_H\n#define MLP_Q8_H\n\n")
        f.write("#include \"common_kern.h\"\n\n")
        f.write(f"static const __s32 L0_IN_ZP[] = {{ {in_zp} }};\n")
        f.write(f"static const __u32 L0_OUT_ZP[] = {{ {out_z0} }};\n")
        f.write(f"static const __u32 L1_OUT_ZP[] = {{ {out_z1} }};\n")
        f.write(f"static const __u32 L2_OUT_ZP[] = {{ {out_zC} }};\n")
        threshold = int(0) 
        f.write(f"static const __s32 CLS_THR_QINT8[] = {{ {threshold} }};\n\n")
        f.write(f"static const float L0_IN_SCALE[] = {{ {in_scale:.10f} }};\n")
        f.write(f"static const float L0_OUT_SCALE[] = {{ {out_s0:.10f} }};\n")
        f.write(f"static const float L1_IN_SCALE[] = {{ {out_s0:.10f} }};\n")
        f.write(f"static const float L1_OUT_SCALE[] = {{ {out_s1:.10f} }};\n")
        f.write(f"static const float L2_IN_SCALE[] = {{ {out_s1:.10f} }};\n")
        f.write(f"static const float L2_OUT_SCALE[] = {{ {out_sC:.10f} }};\n\n")
        f.write(f"static const __s8 L0_W[] = {{ " + ", ".join(map(str, w0.flatten())) + " };\n")
        f.write(f"static const __s32 L0_B_INT32[] = {{ " + ", ".join(map(str, b0)) + " };\n")
        f.write(f"static const float L0_W_SCALE[] = {{ " + ", ".join([f"{v:.10f}" for v in s0]) + " };\n\n")
        f.write(f"static const __s8 L1_W[] = {{ " + ", ".join(map(str, w1.flatten())) + " };\n")
        f.write(f"static const __s32 L1_B_INT32[] = {{ " + ", ".join(map(str, b1)) + " };\n")
        f.write(f"static const float L1_W_SCALE[] = {{ " + ", ".join([f"{v:.10f}" for v in s1]) + " };\n\n")
        f.write(f"static const __s8 L2_W[] = {{ " + ", ".join(map(str, wc.flatten())) + " };\n")
        f.write(f"static const __s32 L2_B_INT32[] = {{ " + ", ".join(map(str, bc)) + " };\n")
        f.write(f"static const float L2_W_SCALE[] = {{ " + ", ".join([f"{v:.10f}" for v in sc]) + " };\n\n")
        f.write("#endif\n")
    print("[+] Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train xtier MLP and export quantized weights")
    parser.add_argument("--input", default=INPUT_CSV, help=f"Input features CSV (default: {INPUT_CSV})")
    parser.add_argument("--output", default=OUTPUT_HEADER, help=f"Output header path (default: {OUTPUT_HEADER})")
    parser.add_argument("--name", default=None, help="Short name for the model (e.g., 'redis'). Sets output to export/mlp_q8_<name>.h")
    parser.add_argument("--label-mode", default="benefit", choices=["binary", "benefit"],
                        help="Label mode: 'binary' (legacy hot/cold) or 'benefit' (continuous benefit score, default)")
    args = parser.parse_args()

    input_csv = args.input
    output_header = args.output
    if args.name:
        output_header = f"export/mlp_q8_{args.name}.h"

    print(f"[*] Input:  {input_csv}")
    print(f"[*] Output: {output_header}")

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        exit(1)
    X, Y, scale = prepare_data(input_csv, label_mode=args.label_mode)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = train_and_eval(X_train, X_val, Y_train, Y_val)
    

    # --- DIAGNOSTIC ---
    import copy
    model_diag = copy.deepcopy(model)
    model_int8_diag = torch.ao.quantization.convert(model_diag.cpu())
    model_int8_diag.eval()
    with torch.no_grad():
        preds = model_int8_diag(torch.tensor(X_val))
        print(f"\n[DIAG] Quantized output stats:")
        print(f"  min={preds.min().item():.2f} max={preds.max().item():.2f} mean={preds.mean().item():.2f}")
        # score = cls_u8 - out_zpC - PROMOTE_THRESH = cls_u8 - 32 - 3 = cls_u8 - 35
        # demote zone: score < 0  -> cls_u8 < 35
        # promote zone: score > 0 -> cls_u8 > 35
        print(f"  Pct in demote zone (cls_u8 <= 32): {(preds <= 32).float().mean().item()*100:.1f}%")
        print(f"  Pct in promote zone (cls_u8 > 35): {(preds > 35).float().mean().item()*100:.1f}%")
        print(f"  Pct <= 32 (demote zone):  {(preds <= 32).float().mean().item()*100:.1f}%")
        print(f"  Pct > 35 (promote zone):  {(preds > 35).float().mean().item()*100:.1f}%")
        for t in [0, 10, 20, 32, 35, 50, 100, 150, 200]:
            pct = (preds <= t).float().mean().item() * 100
            print(f"  <= {t:3d}: {pct:6.1f}%")


    model_int8 = torch.ao.quantization.convert(model.cpu())
    export_header(model_int8, output_header, scale)
