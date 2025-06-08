import torch
import torch.nn as nn
from diffalign_old.utils.graph import PlaceHolder
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ExtraMolecularFeatures(nn.Module):
    def __init__(self, dataset_infos):
        super(ExtraMolecularFeatures, self).__init__()
        self.charge = ChargeFeature(remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies,
                                    bond_orders=dataset_infos.bond_orders)
        self.valency = ValencyFeature(bond_orders=dataset_infos.bond_orders)
        self.weight = WeightFeature(max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights)

    def forward(self, X, E):
        charge = self.charge(X, E).unsqueeze(-1)      # (bs, n, 1)
        valency = self.valency(X, E).unsqueeze(-1)    # (bs, n, 1)
        weight = self.weight(X)                    # (bs, 1)

        extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)

        X_res = torch.cat((charge, valency), dim=-1)     # (bs, n, dx+2)
        E_res = extra_edge_attr
        y_res = weight
        return X_res, E_res, y_res

        # return PlaceHolder(X=torch.cat((charge, valency), dim=-1), E=extra_edge_attr, y=weight)


class ChargeFeature(nn.Module):
    def __init__(self, remove_h, valencies, bond_orders):
        super(ChargeFeature, self).__init__()
        self.remove_h = remove_h
        self.valencies = valencies
        self.bond_orders = bond_orders

    def __call__(self, X, E):

        # TODO: The hydrogens are removed in our data, meaning that the calculations here are not correct
        # Could fill in the hydrogens with rdkit as an improvement
        bond_orders = torch.tensor(self.bond_orders, device=E.device).reshape(1, 1, 1, -1)
        weighted_E = E * bond_orders      # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)   # (bs, n)

        valencies = torch.tensor(self.valencies, device=X.device).reshape(1, 1, -1)
        X_ = X * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X_, dim=-1)               # (bs, n) # <- wait isn't this entirely incorrect? what does the argmax have to do with the valencies here

        return (normal_valencies - current_valencies).type_as(X)


class ValencyFeature(nn.Module):
    def __init__(self, bond_orders):
        super(ValencyFeature, self).__init__()
        self.bond_orders = bond_orders

    def __call__(self, X, E):
        orders = torch.tensor(self.bond_orders, device=E.device).reshape(1, 1, 1, -1)
        E_ = E * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies.type_as(X)


class WeightFeature(nn.Module):
    def __init__(self, max_weight, atom_weights):
        super(WeightFeature, self).__init__()
        self.max_weight = max_weight
        self.atom_weight_list = list(atom_weights.values())

    def __call__(self, X):
        X_ = torch.argmax(X, dim=-1)     # (bs, n)
        X_weights = torch.tensor(self.atom_weight_list, device=X_.device)[X_]           # (bs, n)
        return X_weights.sum(dim=-1).unsqueeze(-1).type_as(X) / self.max_weight     # (bs, 1)
