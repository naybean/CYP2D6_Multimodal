#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import math
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---- our modules ----
from data.variants import load_onehot_bank, dataframe_to_variant_tensors
from data.dataset import MultimodalDataset, collate_fn
from data.drugs import smiles_to_graph
from models import SimpleCNN, GCNModel, MultimodalModel


# ------------------ helpers ------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_num_node_features_from_sample(smiles_series):
    for s in smiles_series:
        d = smiles_to_graph(s)
        if hasattr(d, "x") and d.x is not None and d.x.numel() > 0:
            return int(d.x.size(1))
    raise RuntimeError("Unable to obtain node feature dimensions from valid SMILES.")

def compute_weighted_aupr(y_true_np, y_probs_np, num_classes=3):
    vals = []
    for i in range(num_classes):
        y_bin = (y_true_np == i).astype(int)
        if y_bin.sum() == 0:
            vals.append(np.nan)
        else:
            vals.append(average_precision_score(y_bin, y_probs_np[:, i]))
    return float(np.nanmean(vals))

def compute_per_class_auc(y_true_np, y_probs_np, num_classes=3):
    out = []
    for i in range(num_classes):
        y_bin = (y_true_np == i).astype(int)
        pos, neg = y_bin.sum(), (y_bin == 0).sum()
        if pos == 0 or neg == 0:
            out.append(np.nan)
        else:
            out.append(roc_auc_score(y_bin, y_probs_np[:, i]))
    return out

def compute_per_class_aupr(y_true_np, y_probs_np, num_classes=3):
    out = []
    for i in range(num_classes):
        y_bin = (y_true_np == i).astype(int)
        if y_bin.sum() == 0:
            out.append(np.nan)
        else:
            out.append(average_precision_score(y_bin, y_probs_np[:, i]))
    return out

def per_class_ovr_accuracy(y_true_np, y_pred_np, num_classes=3):
    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(num_classes)))
    N = cm.sum()
    accs = []
    for c in range(num_classes):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = N - (TP + FP + FN)
        accs.append((TP + TN) / N if N > 0 else 0.0)
    return accs

def safe_multiclass_roc_auc(y_true_np, y_probs_np):
    try:
        return roc_auc_score(y_true_np, y_probs_np, multi_class='ovr')
    except Exception:
        return 0.5


# ------------------ main eval ------------------
def evaluate_test(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 데이터 로드 + 라벨 인코딩(훈련과 동일한 매핑을 위해 train_csv도 받음)
    train_df = pd.read_csv(args.train_csv)
    test_df  = pd.read_csv(args.test_csv)

    le = LabelEncoder()
    train_df["Class"] = le.fit_transform(train_df["Class"])
    test_df["Class"]  = le.transform(test_df["Class"])
    print("[Label mapping]", {i: c for i, c in enumerate(le.classes_)})

    # 2) 원핫 bank + 텐서 변환
    bank = load_onehot_bank(args.onehot_dir, target_rows=args.h, target_cols=args.w, verbose=False)

    X_test, y_test = dataframe_to_variant_tensors(
        test_df, bank, variant_col="Variant", class_col="Class"
    )
    X_test = X_test.unsqueeze(1)  # (N,1,2H,W)
    test_smiles = test_df["isomeric"].values

    test_ds = MultimodalDataset(X_test, test_smiles, y_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    # 3) GCN 입력 피처 차원
    num_node_features = get_num_node_features_from_sample(test_smiles)
    print(f"[GCN] num_node_features={num_node_features}")

    # 4) 폴드별 평가
    agg = defaultdict(list)

    # per-class 저장 버킷
    per_class_metrics = {
        "acc": [[] for _ in range(args.num_classes)],
        "prec": [[] for _ in range(args.num_classes)],
        "rec": [[] for _ in range(args.num_classes)],
        "f1": [[] for _ in range(args.num_classes)],
        "auc": [[] for _ in range(args.num_classes)],
        "aupr": [[] for _ in range(args.num_classes)],
    }

    criterion = nn.CrossEntropyLoss().to(device)

    for fold in range(args.folds):
        ckpt_path = os.path.join(args.ckpt_dir, f"best_model_fold_{fold+1}.pth")
        if not os.path.exists(ckpt_path):
            # fold 인덱싱을 0부터 저장했다면 아래 줄로 대체:
            # ckpt_path = os.path.join(args.ckpt_dir, f"best_model_fold_{fold}.pth")
            ckpt_path = os.path.join(args.ckpt_dir, f"best_model_fold_{fold}.pth")
            if not os.path.exists(ckpt_path):
                print(f"[WARN] checkpoint not found for fold {fold+1}. Skipped.")
                continue

        # 모델 구성 (아키텍처는 train과 동일해야 함)
        cnn = SimpleCNN().to(device)
        gcn = GCNModel(num_node_features=num_node_features, hidden_channels=args.gcn_hidden).to(device)
        model = MultimodalModel(cnn, gcn,
                                cnn_output_dim=args.cnn_out,
                                gcn_output_dim=args.gcn_out,
                                num_classes=args.num_classes).to(device)

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        test_loss, correct, total = 0.0, 0, 0
        all_lab, all_log = [], []

        with torch.no_grad():
            for cnn_inputs, gcn_batch, labels in test_loader:
                cnn_inputs = cnn_inputs.to(device)
                gcn_batch = gcn_batch.to(device)
                labels = labels.to(device)

                logits = model(cnn_inputs, gcn_batch)
                loss = criterion(logits, labels)

                test_loss += loss.item()
                _, pred = logits.max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()

                all_lab.append(labels.detach().cpu())
                all_log.append(logits.detach().cpu())

        if len(all_lab) == 0:
            print(f"[WARN] empty test loader on fold {fold+1}")
            continue

        y_true = torch.cat(all_lab).numpy()
        logits = torch.cat(all_log)
        probs = F.softmax(logits, dim=1).numpy()
        y_pred = logits.argmax(1).numpy()

        # 전체 지표
        loss_avg = test_loss / max(len(test_loader), 1)
        acc = 100.0 * correct / max(total, 1)
        auc = safe_multiclass_roc_auc(y_true, probs)
        aupr = compute_weighted_aupr(y_true, probs, num_classes=args.num_classes)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"[Fold {fold+1}] loss {loss_avg:.4f} | acc {acc:6.2f}% | "
              f"AUC {auc:.4f} | AUPR {aupr:.4f} | P {prec:.4f} R {rec:.4f} F1 {f1:.4f}")

        agg["test_loss"].append(loss_avg)
        agg["accuracy"].append(acc)
        agg["auc"].append(auc)
        agg["aupr"].append(aupr)
        agg["precision"].append(prec)
        agg["recall"].append(rec)
        agg["f1"].append(f1)

        # per-class 지표
        report = classification_report(
            y_true, y_pred, labels=list(range(args.num_classes)),
            output_dict=True, zero_division=0
        )
        acc_ovr = per_class_ovr_accuracy(y_true, y_pred, num_classes=args.num_classes)
        auc_c   = compute_per_class_auc (y_true, probs, num_classes=args.num_classes)
        aupr_c  = compute_per_class_aupr(y_true, probs, num_classes=args.num_classes)

        for c in range(args.num_classes):
            per_class_metrics["acc"][c].append(acc_ovr[c])
            per_class_metrics["prec"][c].append(report[str(c)]["precision"])
            per_class_metrics["rec"][c].append(report[str(c)]["recall"])
            per_class_metrics["f1"][c].append(report[str(c)]["f1-score"])
            per_class_metrics["auc"][c].append(auc_c[c])
            per_class_metrics["aupr"][c].append(aupr_c[c])

    # 요약 출력/저장
    def mean_sd(x): return float(np.mean(x)), float(np.std(x))
    def nmean_nsd(x): return float(np.nanmean(x)), float(np.nanstd(x))

    print("\n=== Overall (k-fold) metrics ===")
    for k in ["test_loss","accuracy","auc","aupr","precision","recall","f1"]:
        if k in agg and len(agg[k]):
            m, s = mean_sd(agg[k])
            suf = "%" if k=="accuracy" else ""
            print(f"{k:>10}: {m:.4f}{suf} ± {s:.4f}{suf}")

    print("\n=== Per-class metrics (avg over folds) ===")
    rows = []
    for c in range(args.num_classes):
        acc_m, acc_s = mean_sd(per_class_metrics["acc"][c])
        pr_m,  pr_s  = mean_sd(per_class_metrics["prec"][c])
        rc_m,  rc_s  = mean_sd(per_class_metrics["rec"][c])
        f1_m,  f1_s  = mean_sd(per_class_metrics["f1"][c])
        auc_m, auc_s = nmean_nsd(per_class_metrics["auc"][c])
        apr_m, apr_s = nmean_nsd(per_class_metrics["aupr"][c])

        print(f"Class {c}: Acc(ovr) {acc_m:.4f}±{acc_s:.4f} | "
              f"P {pr_m:.4f}±{pr_s:.4f} R {rc_m:.4f}±{rc_s:.4f} F1 {f1_m:.4f}±{f1_s:.4f} | "
              f"AUC {auc_m:.4f}±{auc_s:.4f} AUPR {apr_m:.4f}±{apr_s:.4f}")

        rows.append({
            "class": c,
            "acc_ovr_mean": acc_m, "acc_ovr_std": acc_s,
            "precision_mean": pr_m, "precision_std": pr_s,
            "recall_mean": rc_m, "recall_std": rc_s,
            "f1_mean": f1_m, "f1_std": f1_s,
            "auc_mean": auc_m, "auc_std": auc_s,
            "aupr_mean": apr_m, "aupr_std": apr_s,
        })

    # CSV 저장 (옵션)
    if args.save_csv:
        pd.DataFrame(agg).to_csv(os.path.join(args.out_dir, "test_overall_metrics.csv"), index=False)
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "test_per_class_metrics.csv"), index=False)
        print(f"[Saved] {args.out_dir}/test_overall_metrics.csv, test_per_class_metrics.csv")


# ------------------ cli ------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Evaluate multimodal model on test set using k-fold checkpoints")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv",  type=str, required=True)
    p.add_argument("--onehot_dir", type=str, required=True)
    p.add_argument("--h", type=int, default=4330)
    p.add_argument("--w", type=int, default=7)
    p.add_argument("--num_classes", type=int, default=3)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--gcn_hidden", type=int, default=128)
    p.add_argument("--cnn_out", type=int, default=128)
    p.add_argument("--gcn_out", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--ckpt_dir", type=str, default="outputs")
    p.add_argument("--out_dir",  type=str, default="outputs")
    p.add_argument("--save_csv", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    evaluate_test(args)

