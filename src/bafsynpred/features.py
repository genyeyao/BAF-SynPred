import numpy as np
import rdkit
import torch
from rdkit import Chem
from rdkit.Chem import Mol, AllChem
from torch_geometric.data import Data

# ==========================================
# 1. 化学特征常量定义
# ==========================================
atom_types = ['B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'Pt', 'S']
formal_charges = [-2, -1, 0, 1, 2]
degree = [0, 1, 2, 3, 4, 5, 6]
num_hs = [0, 1, 2, 3, 4]
hybridization = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
]
bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
chirality = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]


# ==========================================
# 2. 辅助函数
# ==========================================
def one_hot_embedding(value, options):
    """生成 One-Hot 编码，未匹配项归入最后一位"""
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding


def get_gasteiger_partial_charges(mol, n_iter=12):
    """计算 Gasteiger 偏电荷"""
    try:
        AllChem.ComputeGasteigerCharges(mol, nIter=n_iter, includeHA=True)
    except Exception:
        pass


# ==========================================
# 3. 特征提取核心
# ==========================================
def get_node_features(atoms, mol):
    """提取纯 2D 原子特征 (55维)"""
    get_gasteiger_partial_charges(mol)
    node_features = []
    for node in atoms:
        features = one_hot_embedding(node.GetSymbol(), atom_types)  # 13
        features += one_hot_embedding(node.GetTotalDegree(), degree)  # 8
        features += one_hot_embedding(node.GetFormalCharge(), formal_charges)  # 6
        features += one_hot_embedding(node.GetTotalNumHs(), num_hs)  # 6
        features += one_hot_embedding(node.GetHybridization(), hybridization)  # 8
        features += one_hot_embedding(node.GetChiralTag(), chirality)  # 5
        features += [int(node.GetIsAromatic())]  # 1
        features += [node.GetMass() * 0.01]  # 1

        try:
            charge = float(node.GetProp('_GasteigerCharge'))
            if np.isnan(charge) or np.isinf(charge): charge = 0.0
        except (KeyError, ValueError, TypeError):
            charge = 0.0
        features += [charge]  # 1

        features += [int(node.IsInRingSize(3))]  # 1
        features += [int(node.IsInRingSize(4))]  # 1
        features += [int(node.IsInRingSize(5))]  # 1
        features += [int(node.IsInRingSize(6))]  # 1
        features += [int(node.IsInRingSize(7))]  # 1
        features += [int(node.IsInRingSize(8))]  # 1

        node_features.append(features)
    return np.array(node_features, dtype=np.float32)


def get_bond_features_2d(bond):
    """提取纯 2D 化学键特征 (7维)"""
    features = one_hot_embedding(str(bond.GetBondType()), bond_types)  # 5
    features += [int(bond.GetIsConjugated())]  # 1
    features += [int(bond.IsInRing())]  # 1
    return np.array(features, dtype=np.float32)


# ==========================================
# 4. 图构建流程
# ==========================================
def mol_to_data_atom_graph(mol: Mol) -> Data:
    """构建【全拓扑】原子图 (Atom Graph)"""
    if mol.GetNumAtoms() == 0:
        return None

    atoms = mol.GetAtoms()
    node_features = get_node_features(atoms, mol)

    edges = []
    edge_feats = []

    if mol.GetNumBonds() > 0:
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            feat = get_bond_features_2d(bond)

            # 双向添加
            edges.append([u, v])
            edge_feats.append(feat)
            edges.append([v, u])
            edge_feats.append(feat)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        dim = len(bond_types) + 1 + 2  # 7维
        edge_attr = torch.empty((0, dim), dtype=torch.float32)

    return Data(
        x=torch.as_tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=mol.GetNumAtoms()
    )


def to_line_graph_2d(data: Data) -> Data:
    """构建【纯拓扑】键图 (Bond Graph / Line Graph)"""
    if data.edge_index.numel() == 0:
        return Data(x=torch.empty(0, 0), x_old=data.x,
                    edge_index=torch.empty(2, 0, dtype=torch.long),
                    edge_attr=torch.empty(0, 0), edge_attr_old=data.edge_attr)

    # 1. 键图节点特征: 原节点(u) + 边特征(uv) + 原节点(v)
    src_indices = data.edge_index[0]
    dst_indices = data.edge_index[1]

    new_node_features = torch.cat([
        data.x[src_indices],
        data.edge_attr,
        data.x[dst_indices]
    ], dim=1)

    # 2. 键图边连接 (Directed: k->u -> u->v, k!=v)
    u_vals = dst_indices.unsqueeze(1)
    v_vals = src_indices.unsqueeze(0)
    connectivity = (u_vals == v_vals)

    k_vals = src_indices.unsqueeze(1)
    Target_vals = dst_indices.unsqueeze(0)
    no_backtrack = (k_vals != Target_vals)

    valid_adj = connectivity & no_backtrack
    new_edge_index = valid_adj.nonzero(as_tuple=False).t()

    # 3. 键图边特征: 边1特征 + 中间原子特征 + 边2特征
    if new_edge_index.size(1) > 0:
        idx_i = new_edge_index[0]
        idx_j = new_edge_index[1]
        u_indices = dst_indices[idx_i]

        new_edge_attr = torch.cat([
            data.edge_attr[idx_i],
            data.x[u_indices],
            data.edge_attr[idx_j]
        ], dim=1)
    else:
        dim = data.edge_attr.size(1) * 2 + data.x.size(1)
        new_edge_attr = torch.empty(0, dim, dtype=torch.float32)

    return Data(
        x=new_node_features,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        x_old=data.x,
        edge_index_old=data.edge_index,
        edge_attr_old=data.edge_attr
    )


def smiles_to_data_pure_2d(smiles: str, add_hs: bool = False) -> Data:
    """SMILES -> 纯 2D 图数据 Pipeline
    :param add_hs: 是否显式添加氢原子 (全原子图为 True，重原子图为 False)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # 提取最大碎片（去盐）
        mol = Chem.MolFromSmiles(max(Chem.MolToSmiles(mol).split('.'), key=len))
    except (ValueError, RuntimeError):
        pass

    # 根据配置决定是否加氢
    if add_hs:
        mol = Chem.AddHs(mol)

    raw_data = mol_to_data_atom_graph(mol)
    if raw_data is None:
        return None

    bond_data = to_line_graph_2d(raw_data)
    return bond_data
