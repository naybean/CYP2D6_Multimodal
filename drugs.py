#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from rdkit import Chem

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

def atom_to_onehot_features(atom):
    symbol_one_hot = one_of_k_encoding(
        atom.GetSymbol(),
        ['H','C','N','O','F','P','S','Cl','Br','I','other']
    )
    degree_one_hot = one_of_k_encoding(atom.GetDegree(), [0,1,2,3,4,5,'other'])
    explicit_valence_one_hot = one_of_k_encoding(atom.GetExplicitValence(), [0,1,2,3,4,5,6,'other'])
    implicit_valence_one_hot = one_of_k_encoding(atom.GetImplicitValence(), [0,1,2,3,4,5,'other'])
    formal_charge_one_hot = one_of_k_encoding(atom.GetFormalCharge(), [-1,0,1,'other'])
    hybridization_one_hot = one_of_k_encoding(
        atom.GetHybridization(),
        [Chem.rdchem.HybridizationType.SP,
         Chem.rdchem.HybridizationType.SP2,
         Chem.rdchem.HybridizationType.SP3,
         Chem.rdchem.HybridizationType.SP3D,
         Chem.rdchem.HybridizationType.SP3D2,
         'other']
    )
    aromatic = [int(atom.GetIsAromatic())]
    chiral_tag_one_hot = one_of_k_encoding(
        atom.GetChiralTag(),
        [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
         Chem.rdchem.ChiralType.CHI_OTHER]
    )

    return (symbol_one_hot + degree_one_hot + explicit_valence_one_hot +
            implicit_valence_one_hot + formal_charge_one_hot +
            hybridization_one_hot + aromatic + chiral_tag_one_hot)

def bond_to_onehot_features(bond):
    bt = bond.GetBondType()
    bond_type = one_of_k_encoding(bt, [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        'other'
    ])
    conj = [int(bond.GetIsConjugated())]
    stereo = one_of_k_encoding(bond.GetStereo(), [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOANY
    ])
    return bond_type + conj + stereo

def smiles_to_graph(smiles, add_hs=False, use_edge_attr=True, add_self_loops=False):
    """
    Returns
    -------
    Data(x, edge_index[, edge_attr])
      x: (N, F_node), edge_index: (2, E), edge_attr: (E, F_edge) if use_edge_attr
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 빈 그래프를 반환하거나 예외 발생 중 선택
        # raise ValueError(f"Invalid SMILES: {smiles}")
        return Data()  # 빈 Data를 반환 (collate에서 걸러지도록)
    if add_hs:
        mol = Chem.AddHs(mol)

    # 노드 특징
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_to_onehot_features(atom))
    x = torch.tensor(x, dtype=torch.float) if len(x) else torch.empty((0, 0), dtype=torch.float)

    # 엣지 인덱스 / 엣지 특성
    rows, cols, eattr = [], [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [a, b]
        cols += [b, a]
        if use_edge_attr:
            f = bond_to_onehot_features(bond)
            eattr += [f, f]  # 양방향

    edge_index = torch.tensor([rows, cols], dtype=torch.long) if rows else torch.empty((2,0), dtype=torch.long)
    if use_edge_attr:
        edge_attr = torch.tensor(eattr, dtype=torch.float) if eattr else torch.empty((0, 0), dtype=torch.float)

    # self-loop 옵션
    if add_self_loops and edge_index.numel() > 0:
        from torch_geometric.utils import add_self_loops
        if use_edge_attr and edge_attr.numel() > 0:
            # self-loop용 edge_attr는 zero vector로 추가
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            loop_attr = torch.zeros((x.size(0), edge_attr.size(1)), dtype=edge_attr.dtype)
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        else:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    if use_edge_attr:
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return Data(x=x, edge_index=edge_index)

