#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from typing import Dict, Tuple, Optional, List

import pandas as pd
import torch


# In[ ]:


def _normalize_key(filename: str) -> str:
    """
    'CYP2D6_1.001.csv' -> 'CYP2D6.1.001'
    """
    base = filename[:-4] if filename.endswith(".csv") else filename
    return base.replace("_", ".")

def pad_matrix_np_to_tensor(arr, target_rows=4330, target_cols=7,
                            dtype: torch.dtype = torch.float32,
                            device: Optional[torch.device] = None) -> torch.Tensor:
    """
    np.array(H, W) -> torch.Tensor(target_rows, target_cols) with zero padding
    """
    t = torch.zeros((target_rows, target_cols), dtype=dtype, device=device)
    r, c = arr.shape
    t[:r, :c] = torch.as_tensor(arr, dtype=dtype, device=device)
    return t

def load_onehot_bank(folder: str,
                     target_rows: int = 4330,
                     target_cols: int = 7,
                     dtype: torch.dtype = torch.float32,
                     device: Optional[torch.device] = None,
                     verbose: bool = True) -> Dict[str, torch.Tensor]:
    """
    폴더 내 *.csv 원핫 매트릭스를 모두 읽어 패딩 후 딕셔너리로 반환.
    key = 'CYP2D6.1.001' 형태.
    """
    bank: Dict[str, torch.Tensor] = {}
    for f in os.listdir(folder):
        if not f.endswith(".csv"):
            continue
        key = _normalize_key(f)
        path = os.path.join(folder, f)
        try:
            arr = pd.read_csv(path, header=None).values
            bank[key] = pad_matrix_np_to_tensor(arr, target_rows, target_cols, dtype, device)
            if verbose:
                print(f"[onehot] loaded {key}: shape {arr.shape}")
        except Exception as e:
            print(f"[onehot] ERROR reading {f}: {e}")
    if verbose:
        print(f"[onehot] total loaded: {len(bank)}")
    return bank

def parse_variant_string(variant: str, sep: str = ";") -> List[str]:
    """
    'CYP2D6.1.001; CYP2D6.2.001' -> ['CYP2D6.1.001', 'CYP2D6.2.001']
    공백은 자동 트림.
    """
    return [v.strip() for v in variant.split(sep) if v.strip()]

def variant_to_onehot(variant: str,
                      bank: Dict[str, torch.Tensor],
                      max_hap: int = 2,
                      h: int = 4330,
                      w: int = 7,
                      dtype: torch.dtype = torch.float32,
                      device: Optional[torch.device] = None,
                      strict: bool = False) -> torch.Tensor:
    """
    Variant 문자열을 받아 bank에서 텐서를 가져와 세로로 이어붙임(2H x W).
    bank에 없는 키는 zero 텐서로 대체( strict=True면 예외).
    """
    names = parse_variant_string(variant, sep=";")
    mats: List[torch.Tensor] = []

    for v in names[:max_hap]:
        if v in bank:
            mats.append(bank[v])
        else:
            msg = f"[onehot] WARNING: variant not found in bank: {v}"
            if strict:
                raise KeyError(msg)
            else:
                print(msg)
                mats.append(torch.zeros((h, w), dtype=dtype, device=device))

    while len(mats) < max_hap:
        mats.append(torch.zeros((h, w), dtype=dtype, device=device))

    return torch.vstack(mats)  # shape = (max_hap * h, w)

def dataframe_to_variant_tensors(
    df: pd.DataFrame,
    bank: Dict[str, torch.Tensor],
    variant_col: str = "Variant",
    class_col: str = "Class",
    max_hap: int = 2,
    h: int = 4330,
    w: int = 7,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    DataFrame에서 Variant 컬럼을 one-hot 텐서로 변환하고, Class를 y로 반환.
    X shape: (N, 2H, W), y shape: (N,)
    """
    mapped = []
    for v in df[variant_col].tolist():
        mapped.append(
            variant_to_onehot(
                v, bank, max_hap=max_hap, h=h, w=w, dtype=dtype, device=device
            )
        )
    X = torch.stack(mapped)  # (N, 2H, W)
    y = torch.as_tensor(df[class_col].values, dtype=torch.long, device=device)
    return X, y


# In[ ]:





# In[ ]:




