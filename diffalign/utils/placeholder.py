import copy
from dataclasses import dataclass, fields
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch


def encode_no_element(A):
    '''Turns no elements (e.g. from dense padding) to one-hot encoded vectors. Works on X and E.'''
    assert len(A.shape) >= 3
    if A.shape[-1] == 0:
        return A
    no_elt = torch.sum(A, dim=-1) == 0
    first_elt = A[..., 0]
    first_elt[no_elt] = 1
    A[..., 0] = first_elt
    return A


@dataclass
class PlaceHolder:
    X: torch.Tensor
    E: torch.Tensor
    y: torch.Tensor
    node_mask: torch.Tensor = None
    atom_map_numbers: torch.Tensor = None
    mol_assignments: torch.Tensor = None

    def _apply(self, fn):
        """Apply fn to every tensor field, return a new PlaceHolder."""
        return PlaceHolder(**{
            f.name: fn(getattr(self, f.name)) if isinstance(getattr(self, f.name), torch.Tensor) else None
            for f in fields(self)
        })

    def _apply_inplace(self, fn):
        """Apply fn to every tensor field in place, return self."""
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, fn(val))
        return self

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of all tensor fields."""
        return self._apply_inplace(lambda t: t.type_as(x))

    def to_device(self, device):
        return self._apply_inplace(lambda t: t.to(device))

    def to_cpu(self):
        return self._apply_inplace(lambda t: t.detach().cpu())

    def to_numpy(self):
        return self._apply_inplace(lambda t: t.detach().cpu().numpy())

    def flatten(self, start_dim, end_dim):
        return self._apply(lambda t: t.flatten(start_dim=start_dim, end_dim=end_dim))

    def select_subset(self, selection):
        return self._apply(lambda t: t.clone()[selection])

    def slice_by_idx(self, idx):
        return self._apply(lambda t: t.clone()[:idx])

    def subset_by_idx(self, start_idx, end_idx):
        return self._apply(lambda t: t.clone()[start_idx:end_idx])

    def get_new_object(self, **kwargs):
        """Return a new PlaceHolder with specified fields replaced, all tensor fields cloned."""
        return PlaceHolder(**{
            f.name: kwargs[f.name].clone() if isinstance(kwargs.get(f.name), torch.Tensor)
                    else getattr(self, f.name).clone() if isinstance(getattr(self, f.name), torch.Tensor)
                    else None
            for f in fields(self)
        })

    def mask(self, node_mask=None, collapse=False):
        if node_mask is None:
            node_mask = self.node_mask
        assert node_mask is not None, 'node_mask is None.'

        x_node_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_node_mask1 = x_node_mask.unsqueeze(2)        # bs, n, 1, 1
        e_node_mask2 = x_node_mask.unsqueeze(1)        # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)
            self.X[node_mask == 0] = 0
            self.E[(e_node_mask1 * e_node_mask2).squeeze(-1) == 0] = 0
        else:
            self.X = self.X * x_node_mask
            self.E = self.E * e_node_mask1 * e_node_mask2
            diag = torch.eye(self.E.shape[1], dtype=torch.bool).unsqueeze(0).expand(self.E.shape[0], -1, -1)
            self.E[diag] = 0
            self.X = encode_no_element(self.X)
            self.E = encode_no_element(self.E)
        return self

    def reshape_bs_n_samples(self, bs, n_samples, n):
        self.X = self.X.reshape(bs, n_samples, n)
        self.E = self.E.reshape(bs, n_samples, n, n)
        self.y = torch.empty((bs, n_samples))
        self.node_mask = self.node_mask.reshape(bs, n_samples, n)
        self.atom_map_numbers = self.atom_map_numbers.reshape(bs, n_samples, n)
        self.mol_assignments = self.mol_assignments.reshape(bs, n_samples, n)

    def cat_by_batchdim(self, placeh):
        for f in fields(self):
            setattr(self, f.name, torch.cat((getattr(self, f.name), getattr(placeh, f.name)), dim=0))

    def cat_by_batchdim_with_padding(self, placeh):
        if self.X.shape[1] > placeh.X.shape[1]:
            to_pad, ready = placeh, self
        else:
            to_pad, ready = self, placeh
        pad_size = ready.X.shape[1] - to_pad.X.shape[1]
        to_pad.pad_nodes(pad_size)
        ready.cat_by_batchdim(to_pad)
        return ready

    def pad_nodes(self, pad_size):
        padding_tuple_X = (0, pad_size) if self.X.ndim == 2 else (0, 0, 0, pad_size)
        padding_tuple_E = (0, pad_size, 0, pad_size) if self.E.ndim == 3 else (0, 0, 0, pad_size, 0, pad_size)
        padding_tuple_v = (0, pad_size)
        self.X = F.pad(self.X, padding_tuple_X, value=0)
        self.E = F.pad(self.E, padding_tuple_E, value=0)
        self.node_mask = F.pad(self.node_mask, padding_tuple_v, value=0)
        self.atom_map_numbers = F.pad(self.atom_map_numbers, padding_tuple_v, value=0)
        self.mol_assignments = F.pad(self.mol_assignments, padding_tuple_v, value=0)

    def select_by_batch_idx(self, idx):
        return PlaceHolder(**{
            f.name: copy.deepcopy(getattr(self, f.name)[idx:idx+1])
            for f in fields(self)
        })

    def select_by_batch_and_sample_idx(self, bs, n_samples, batch_idx, sample_idx):
        assert self.X.ndim == 2, f'Expected X of shape (bs, n), got X.shape={self.X.shape}.'
        assert self.E.ndim == 3, f'Expected E of shape (bs, n, n), got E.shape={self.E.shape}.'
        return PlaceHolder(
            X=self.X.reshape(bs, n_samples, self.X.shape[1])[batch_idx:batch_idx+1, sample_idx],
            E=self.E.reshape(bs, n_samples, self.E.shape[2], -1)[batch_idx:batch_idx+1, sample_idx],
            y=self.y.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx],
            node_mask=self.node_mask.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx],
            atom_map_numbers=self.atom_map_numbers.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx],
            mol_assignments=self.mol_assignments.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx],
        )

    def serialize(self):
        return {f.name: getattr(self, f.name).detach().cpu().numpy().tolist() for f in fields(self)}

    def pyg(self):
        """Turns back into a pytorch geometric DataBatch() object."""
        return_data = []
        for i in range(self.X.shape[0]):
            E_idx, E_attr = dense_to_sparse(adj=self.E[i])
            X = self.X[i] if self.X.dim() == 2 else self.X[i].argmax(-1)
            assert X.dim() == 1
            return_data.append(Data(
                x=X.to(torch.uint8), edge_index=E_idx.to(torch.int16),
                edge_attr=E_attr.to(torch.uint8), y=self.y.to(torch.uint8),
                node_mask=self.node_mask[i].to(torch.uint8),
                mask_atom_mapping=self.atom_map_numbers[i].to(torch.uint8),
                mol_assignment=self.mol_assignments[i].to(torch.uint8),
            ))
        return Batch.from_data_list(return_data)
