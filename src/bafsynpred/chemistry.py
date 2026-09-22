from collections import deque
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS, Mol

from .features import mol_to_data_atom_graph, to_line_graph_2d


HYPEREDGE_TYPES: Dict[str, int] = {
    "bond": 0,
    "ring": 1,
    "functional_group": 2,
    "brics_fragment": 3,
    "conjugated_system": 4,
    "atom_self": 5,
}
NUM_HYPEREDGE_TYPES = len(HYPEREDGE_TYPES)
HYPEREDGE_ATTR_DIM = NUM_HYPEREDGE_TYPES + 5


FUNCTIONAL_GROUP_SMARTS: Sequence[Tuple[str, str]] = (
    ("hydroxyl", "[OX2H]"),
    ("primary_secondary_amine", "[NX3;H2,H1;!$(NC=O)]"),
    ("tertiary_amine", "[NX3;H0;!$(NC=O)]"),
    ("carbonyl", "[CX3]=[OX1]"),
    ("carboxyl", "[CX3](=O)[OX2H1]"),
    ("amide", "[NX3][CX3](=[OX1])"),
    ("ester", "[CX3](=O)[OX2][#6]"),
    ("ether", "[OD2]([#6])[#6]"),
    ("sulfonamide", "S(=O)(=O)N"),
    ("sulfone", "S(=O)(=O)[#6]"),
    ("phosphate", "P(=O)(O)(O)"),
    ("nitrile", "[CX2]#N"),
    ("halogen_substituent", "[#6][F,Cl,Br,I]"),
)


def _largest_fragment_mol(smiles: str) -> Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        largest = max(Chem.MolToSmiles(mol).split("."), key=len)
        mol = Chem.MolFromSmiles(largest)
    except Exception:
        pass
    return mol


def _normalize_atoms(atoms: Iterable[int], num_atoms: int) -> Tuple[int, ...]:
    valid = sorted({int(idx) for idx in atoms if 0 <= int(idx) < num_atoms})
    return tuple(valid)


def _hyperedge_attr(mol: Mol, atoms: Sequence[int], type_name: str) -> torch.Tensor:
    num_atoms = max(mol.GetNumAtoms(), 1)
    atoms = list(atoms)
    atom_objs = [mol.GetAtomWithIdx(int(idx)) for idx in atoms]
    size = len(atom_objs)

    type_one_hot = torch.zeros(NUM_HYPEREDGE_TYPES, dtype=torch.float32)
    type_one_hot[HYPEREDGE_TYPES[type_name]] = 1.0

    if size == 0:
        stats = torch.zeros(5, dtype=torch.float32)
    else:
        aromatic_ratio = sum(atom.GetIsAromatic() for atom in atom_objs) / size
        hetero_ratio = sum(atom.GetAtomicNum() not in (1, 6) for atom in atom_objs) / size
        ring_ratio = sum(atom.IsInRing() for atom in atom_objs) / size
        formal_charge = sum(atom.GetFormalCharge() for atom in atom_objs) / 5.0
        stats = torch.tensor(
            [
                size / num_atoms,
                float(aromatic_ratio),
                float(hetero_ratio),
                float(ring_ratio),
                float(formal_charge),
            ],
            dtype=torch.float32,
        )
    return torch.cat([type_one_hot, stats], dim=0)


def _add_hyperedge(records: List[dict], seen: set, mol: Mol, atoms: Iterable[int], type_name: str):
    atom_tuple = _normalize_atoms(atoms, mol.GetNumAtoms())
    if len(atom_tuple) == 0:
        return
    key = (type_name, atom_tuple)
    if key in seen:
        return
    seen.add(key)
    records.append(
        {
            "atoms": atom_tuple,
            "type": type_name,
            "attr": _hyperedge_attr(mol, atom_tuple, type_name),
        }
    )


def _bond_hyperedges(mol: Mol, records: List[dict], seen: set):
    for bond in mol.GetBonds():
        _add_hyperedge(
            records,
            seen,
            mol,
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            "bond",
        )


def _ring_hyperedges(mol: Mol, records: List[dict], seen: set):
    for ring in Chem.GetSymmSSSR(mol):
        ring_atoms = tuple(int(idx) for idx in ring)
        if len(ring_atoms) >= 3:
            _add_hyperedge(records, seen, mol, ring_atoms, "ring")


def _functional_group_hyperedges(mol: Mol, records: List[dict], seen: set):
    for _, smarts in FUNCTIONAL_GROUP_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            if len(match) >= 2:
                _add_hyperedge(records, seen, mol, match, "functional_group")


def _connected_components(num_atoms: int, edges: Iterable[Tuple[int, int]]) -> List[Tuple[int, ...]]:
    adjacency = [[] for _ in range(num_atoms)]
    for u, v in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)

    visited = [False] * num_atoms
    components = []
    for start in range(num_atoms):
        if visited[start] or not adjacency[start]:
            continue
        queue = deque([start])
        visited[start] = True
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for nxt in adjacency[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        components.append(tuple(sorted(component)))
    return components


def _brics_fragment_hyperedges(mol: Mol, records: List[dict], seen: set):
    try:
        cut_bonds = {
            tuple(sorted((int(pair[0]), int(pair[1]))))
            for pair, _ in BRICS.FindBRICSBonds(mol)
        }
    except Exception:
        cut_bonds = set()

    if not cut_bonds:
        return

    kept_edges = []
    for bond in mol.GetBonds():
        pair = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        if pair not in cut_bonds:
            kept_edges.append(pair)

    for component in _connected_components(mol.GetNumAtoms(), kept_edges):
        if 2 <= len(component) < mol.GetNumAtoms():
            _add_hyperedge(records, seen, mol, component, "brics_fragment")


def _conjugated_system_hyperedges(mol: Mol, records: List[dict], seen: set):
    conjugated_edges = []
    for bond in mol.GetBonds():
        if bond.GetIsConjugated() or bond.GetIsAromatic():
            conjugated_edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    for component in _connected_components(mol.GetNumAtoms(), conjugated_edges):
        if len(component) >= 3:
            _add_hyperedge(records, seen, mol, component, "conjugated_system")


def build_chemistry_hypergraph(mol: Mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    records: List[dict] = []
    seen = set()

    _bond_hyperedges(mol, records, seen)
    _ring_hyperedges(mol, records, seen)
    _functional_group_hyperedges(mol, records, seen)
    _brics_fragment_hyperedges(mol, records, seen)
    _conjugated_system_hyperedges(mol, records, seen)

    if not records:
        for atom_idx in range(mol.GetNumAtoms()):
            _add_hyperedge(records, seen, mol, (atom_idx,), "atom_self")

    incidence = []
    hyperedge_types = []
    hyperedge_attrs = []
    for edge_id, item in enumerate(records):
        hyperedge_types.append(HYPEREDGE_TYPES[item["type"]])
        hyperedge_attrs.append(item["attr"])
        for atom_idx in item["atoms"]:
            incidence.append((atom_idx, edge_id))

    if incidence:
        hyperedge_index = torch.tensor(incidence, dtype=torch.long).t().contiguous()
    else:
        hyperedge_index = torch.empty((2, 0), dtype=torch.long)

    hyperedge_type = torch.tensor(hyperedge_types, dtype=torch.long)
    hyperedge_attr = torch.stack(hyperedge_attrs, dim=0).to(torch.float32)
    return hyperedge_index, hyperedge_type, hyperedge_attr


def smiles_to_data_chem_hypergraph(smiles: str) -> torch.Tensor:
    mol = _largest_fragment_mol(smiles)
    if mol is None:
        return None

    raw_data = mol_to_data_atom_graph(mol)
    if raw_data is None:
        return None

    hyperedge_index, hyperedge_type, hyperedge_attr = build_chemistry_hypergraph(mol)
    bond_data = to_line_graph_2d(raw_data)

    bond_data.edge_index_old = raw_data.edge_index
    bond_data.edge_attr_old = raw_data.edge_attr
    bond_data.atom_hyperedge_index = hyperedge_index
    bond_data.atom_hyperedge_type = hyperedge_type
    bond_data.atom_hyperedge_attr = hyperedge_attr
    bond_data.num_atom_hyperedges = int(hyperedge_type.numel())
    bond_data.num_atom_nodes = int(raw_data.x.size(0))
    return bond_data
