#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from data.variants import load_onehot_bank, dataframe_to_variant_tensors
from data.dataset import MultimodalDataset, collate_fn
from models import SimpleCNN, GCNModel, MultimodalModel

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Subset
import pandas as pd

# ================================
# 데이터 로드 및 전처리
# ================================
train_df = pd.read_csv("train_aug.csv")
test_df  = pd.read_csv("test.csv")

# one-hot bank 로드
bank = load_onehot_bank("onehot7", target_rows=4330, target_cols=7)

# 라벨 인코딩
le = LabelEncoder()
train_df["Class"] = le.fit_transform(train_df["Class"])
test_df["Class"]  = le.transform(test_df["Class"])
print({i: c for i, c in enumerate(le.classes_)})

# variant → 텐서 변환
train_X, train_y = dataframe_to_variant_tensors(train_df, bank)
test_X,  test_y  = dataframe_to_variant_tensors(test_df,  bank)

# CNN 입력 모양 맞추기
train_X = train_X.unsqueeze(1)
test_X  = test_X.unsqueeze(1)

# SMILES 데이터
train_smiles = train_df["isomeric"].values
test_smiles  = test_df["isomeric"].values

# Dataset / DataLoader
train_ds = MultimodalDataset(train_X, train_smiles, train_y)
test_ds  = MultimodalDataset(test_X,  test_smiles,  test_y)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, collate_fn=collate_fn)

# ================================
# 학습 코드
# ================================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_weighted_aupr(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int = 3) -> float:
    """ OvR 방식의 macro(혹은 단순 평균) AUPR """
    auprs = []
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        aupr = average_precision_score(y_true_binary, y_probs[:, i])
        auprs.append(aupr)
    return float(np.mean(auprs))


def safe_multiclass_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    일부 fold/epoch에서 검증셋에 특정 클래스가 1개 이하로 나타나면
    sklearn roc_auc_score가 에러를 던질 수 있어 예외를 안전 처리.
    """
    try:
        return roc_auc_score(y_true, y_probs, multi_class='ovr')
    except Exception:
        # 클래스 부족 등으로 계산 불가할 때는 NaN 대신 0.5(무작위)로 대체
        return 0.5


def get_num_node_features_from_sample(smiles_series) -> int:
    """
    SMILES 한 개를 그래프로 변환해서 노드 피처 차원을 동적으로 산정
    """
    for s in smiles_series:
        data = smiles_to_graph(s)
        if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
            return int(data.x.size(1))
    raise RuntimeError("유효한 SMILES로부터 노드 피처 차원을 얻을 수 없습니다.")


# ------------------------------
# Training / Evaluation
# ------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for cnn_inputs, gcn_batch, labels in loader:
        cnn_inputs = cnn_inputs.to(device)
        gcn_batch = gcn_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(cnn_inputs, gcn_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        total_correct += pred.eq(labels).sum().item()

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * total_correct / max(total, 1)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int = 3):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_labels, all_logits = [], []

    for cnn_inputs, gcn_batch, labels in loader:
        cnn_inputs = cnn_inputs.to(device)
        gcn_batch = gcn_batch.to(device)
        labels = labels.to(device)

        outputs = model(cnn_inputs, gcn_batch)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        total_correct += pred.eq(labels).sum().item()

        all_labels.append(labels.cpu())
        all_logits.append(outputs.cpu())

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100.0 * total_correct / max(total, 1)

    if len(all_labels) == 0:
        # 빈 로더 보호
        return avg_loss, acc, 0.5, 0.5, 0.0, 0.0, 0.0

    y_true = torch.cat(all_labels).numpy()
    logits = torch.cat(all_logits)
    probs = F.softmax(logits, dim=1).numpy()

    auc = safe_multiclass_roc_auc(y_true, probs)
    aupr = compute_weighted_aupr(y_true, probs, num_classes=num_classes)
    y_pred = logits.argmax(dim=1).numpy()

    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return avg_loss, acc, auc, aupr, prec, rec, f1


def make_datasets(train_df, onehot_bank, label_encoder, device=None):
    """
    - Variant -> (N, 2H, W) 텐서
    - Class   -> (N,) 텐서 (라벨 인코딩 완료 가정)
    - CNN 입력 모양 (N, 1, 2H, W)
    - SMILES 배열
    - MultimodalDataset 리턴
    """
    X, y = dataframe_to_variant_tensors(train_df, onehot_bank,
                                        variant_col="Variant", class_col="Class",
                                        device=device)
    X = X.unsqueeze(1)  # (N, 1, 2H, W)
    smiles = train_df["isomeric"].values
    ds = MultimodalDataset(X, smiles, y)
    return ds, y.cpu().numpy() if y.is_cuda else y.numpy()


# ------------------------------
# Main Cross-Validation Routine
# ------------------------------
def run_cv(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # 1) Load data
    train_df = pd.read_csv(args.train_csv)
    # (선택) test_df를 별도로 쓰려면 필요시 읽기
    # test_df  = pd.read_csv(args.test_csv) if args.test_csv else None

    # 2) Label encoding (fit on train)
    le = LabelEncoder()
    train_df["Class"] = le.fit_transform(train_df["Class"])
    print("[Label mapping]", {i: c for i, c in enumerate(le.classes_)})

    # 3) One-hot bank
    bank = load_onehot_bank(args.onehot_dir, target_rows=args.h, target_cols=args.w, verbose=True)

    # 4) Build dataset (will build tensors once; we'll index by folds)
    full_ds, y_full = make_datasets(train_df, bank, le, device=None)

    # 5) Node feature dim (from a sample SMILES)
    num_node_features = get_num_node_features_from_sample(train_df["isomeric"].values)
    print(f"[GCN] num_node_features = {num_node_features}")

    # 6) StratifiedKFold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # 7) Aggregation buckets
    fold_results = defaultdict(list)

    # 8) Folds
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_full)), y_full), start=1):
        print(f"\n===== Fold {fold}/{args.folds} =====")

        # Subsets and loaders
        tr_subset = Subset(full_ds, tr_idx)
        va_subset = Subset(full_ds, va_idx)

        train_loader = DataLoader(tr_subset, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
        val_loader   = DataLoader(va_subset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

        # Models
        cnn_model = SimpleCNN().to(device)
        gcn_model = GCNModel(num_node_features=num_node_features, hidden_channels=args.gcn_hidden).to(device)
        model = MultimodalModel(cnn_model, gcn_model,
                                cnn_output_dim=args.cnn_out,
                                gcn_output_dim=args.gcn_out,
                                num_classes=args.num_classes).to(device)

        # Optimizer with parameter groups
        optimizer = torch.optim.Adam([
            {"params": cnn_model.parameters(), "lr": args.cnn_lr},
            {"params": gcn_model.parameters(), "lr": args.gcn_lr},
            {"params": model.fc1.parameters(), "lr": args.head_lr},
            {"params": model.fc2.parameters(), "lr": args.head_lr},
        ], weight_decay=args.weight_decay)

        # (선택) Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=1, verbose=True
        )

        criterion = nn.CrossEntropyLoss().to(device)

        best_val_auc = -math.inf
        patience_counter = 0

        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_acc, va_auc, va_aupr, va_prec, va_rec, va_f1 = evaluate(
                model, val_loader, criterion, device, num_classes=args.num_classes
            )

            # step scheduler on metric
            scheduler.step(va_auc)

            print(f"[Fold {fold:02d}][Epoch {epoch:03d}] "
                  f"Train loss {tr_loss:.4f} | acc {tr_acc:6.2f}% || "
                  f"Val loss {va_loss:.4f} | acc {va_acc:6.2f}% | "
                  f"AUC {va_auc:.4f} | AUPR {va_aupr:.4f} | "
                  f"P {va_prec:.4f} R {va_rec:.4f} F1 {va_f1:.4f}")

            # early stopping (by AUC)
            if va_auc > best_val_auc + 1e-8:
                best_val_auc = va_auc
                patience_counter = 0
                save_path = os.path.join(args.out_dir, f"best_model_fold_{fold}.pth")
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[Fold {fold}] Early stopping.")
                    break

        # one more evaluation with best checkpoint (optional)
        # (여기서는 마지막 에폭 성능을 기록했지만, 필요 시 best ckpt 로드 후 재평가 가능)

        # Fold results (마지막 에폭 기준 기록; 필요 시 best 시점 기록으로 바꿔도 됨)
        fold_results['train_loss'].append(tr_loss)
        fold_results['train_acc'].append(tr_acc)
        fold_results['val_loss'].append(va_loss)
        fold_results['val_acc'].append(va_acc)
        fold_results['val_auc'].append(va_auc)
        fold_results['val_aupr'].append(va_aupr)
        fold_results['precision'].append(va_prec)
        fold_results['recall'].append(va_rec)
        fold_results['f1'].append(va_f1)

    # 9) Print CV summary
    def mean_std(x): return (float(np.mean(x)), float(np.std(x)))

    print("\n==== Cross-Validation Results ====")
    for k, v in fold_results.items():
        m, s = mean_std(v)
        print(f"{k:>10}: {m:.4f} ± {s:.4f}")

    # (선택) CSV로 저장
    if args.save_csv:
        out_csv = os.path.join(args.out_dir, "cv_results.csv")
        pd.DataFrame(fold_results).to_csv(out_csv, index=False)
        print(f"[Saved] {out_csv}")


# ------------------------------
# Entry
# ------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Multimodal CNN+GCN training with Stratified K-Fold")
    # Data
    p.add_argument("--train_csv", type=str, required=True, help="Path to training CSV with columns: Variant, isomeric, Class")
    p.add_argument("--onehot_dir", type=str, required=True, help="Directory of one-hot CSV files")
    p.add_argument("--h", type=int, default=4330, help="One-hot height (per haplotype)")
    p.add_argument("--w", type=int, default=7, help="One-hot width (channels)")
    # Model
    p.add_argument("--gcn_hidden", type=int, default=128)
    p.add_argument("--cnn_out", type=int, default=128)
    p.add_argument("--gcn_out", type=int, default=128)
    p.add_argument("--num_classes", type=int, default=3)
    # Train
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    # Optim
    p.add_argument("--cnn_lr", type=float, default=1e-3)
    p.add_argument("--gcn_lr", type=float, default=1e-3)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    # System
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--save_csv", action="store_true")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_cv(args)

