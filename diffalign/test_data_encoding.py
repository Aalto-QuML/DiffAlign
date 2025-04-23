from diffalign.utils import mol
from hydra import compose, initialize
from omegaconf import DictConfig
from diffalign.utils import graph
from torch_geometric.data import Data
import torch
# Initialize the Hydra context
initialize(config_path="../configs")
# Compose the configuration, referring to the config file name without the .yaml extension
cfg = compose(config_name="default")

r = "CC(C)C1=CC(OC2=C(Br)C=C(CC(O)C(=O)O)C=C2Br)=CC(/C=C/C2=CC=CC=C2)=C1O"
p = "CC(C)C1=CC(OC2=C(Br)C=C(CC(O)C(=O)O)C=C2Br)=CC(/C=C/C2=CC=CC=C2)=C1O"
cfg.dataset.atom_types = ['none', 'O', 'C', 'N', 'I', 'Cl', 'Si', 'F', 'Br', 'S', 'B', 'Cu', 'Sn', 'P', 'Se', 'Zn', 'Mg', 'U', 'Au', 'SuNo']
cfg.dataset.with_formal_charge_in_atom_symbols = False
cfg.dataset.use_stereochemistry = True
cfg.dataset.use_charges_as_features = True

nodes, atom_charges, atom_chiral, edge_index, bond_types, bond_dirs, atom_map = mol.smiles_to_graph_with_stereochem(smi=r, cfg=cfg)
mol_assignment = torch.zeros_like(nodes)
atom_map_numbers = torch.zeros_like(nodes)
y = torch.zeros((1, 0), dtype=torch.float)

nodes_p, atom_charges_p, atom_chiral_p, edge_index_p, bond_types_p, bond_dirs_p, atom_map_p = mol.smiles_to_graph_with_stereochem(smi=p, cfg=cfg)
mol_assignment_p = torch.ones_like(nodes_p)
atom_map_numbers_p = torch.zeros_like(nodes_p)
y = torch.zeros((1, 0), dtype=torch.float)

# # TODO: This is a bit clunky -> we should have a method that directly goes to the Data() object
# g = Data(x=nodes, edge_index=edge_index,
#                      edge_attr=bond_types, y=y, idx=0,
#                      mol_assignment=mol_assignment,
#                      atom_map_numbers=atom_map_numbers,
#                      smiles=r,
#                      atom_charges=atom_charges,
#                      atom_chiral=atom_chiral,
#                      bond_dirs=bond_dirs)

g = Data(x=torch.cat([nodes, nodes_p], dim=0), 
                     edge_index=torch.cat([edge_index, edge_index_p + len(nodes)], dim=1),
                     edge_attr=torch.cat([bond_types, bond_types_p], dim=0), y=y, idx=0,
                     mol_assignment=torch.cat([mol_assignment, mol_assignment_p], dim=0),
                     atom_map_numbers=torch.cat([atom_map_numbers, atom_map_numbers_p], dim=0),
                     smiles=r + ">>" + p,
                     atom_charges=torch.cat([atom_charges, atom_charges_p], dim=0),
                     atom_chiral=torch.cat([atom_chiral, atom_chiral_p], dim=0),
                     bond_dirs=torch.cat([bond_dirs, bond_dirs_p], dim=0))

dense = graph.to_dense(g)
print("as")

dense.mask(collapse=True)

smi_recovered = mol.get_cano_smiles_from_dense_with_stereochem(dense, cfg)
print(smi_recovered[0].split(">>")[0])
print(r)