import copy
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch


def encode_no_element(A):
    '''
        Turns no elements (e.g. from dense padding) to one-hot encoded vectors.
        Works on X and E.
    '''
    assert len(A.shape) >= 3
    if A.shape[-1]==0:
        return A
    no_elt = torch.sum(A, dim=-1) == 0
    first_elt = A[..., 0]
    first_elt[no_elt] = 1
    A[..., 0] = first_elt
    return A


class PlaceHolder:
    def __init__(self, X, E, y, node_mask=None, atom_map_numbers=None, mol_assignments=None):
        self.X = X
        self.E = E
        self.y = y
        self.node_mask = node_mask
        self.atom_map_numbers = atom_map_numbers
        self.mol_assignments = mol_assignments

    def flatten(self, start_dim, end_dim):
        '''
            return a placeholder object with the first idx batch elements.
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]

        new_kwargs = {at: getattr(self, at).flatten(start_dim=start_dim, end_dim=end_dim) if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}

        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]

        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'

        new_obj = PlaceHolder(**new_kwargs)

        return new_obj

    def reshape_bs_n_samples(self, bs, n_samples, n):
        self.X = self.X.reshape(bs, n_samples, n)
        self.E = self.E.reshape(bs, n_samples, n, n)
        self.y = torch.empty((bs, n_samples))
        self.node_mask = self.node_mask.reshape(bs, n_samples, n)
        self.atom_map_numbers = self.atom_map_numbers.reshape(bs, n_samples, n)
        self.mol_assignments = self.mol_assignments.reshape(bs, n_samples, n)

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.type_as(x)
        if isinstance(self.mol_assignments, torch.Tensor):
            self.mol_assignments = self.mol_assignments.type_as(x)
        return self

    def to_device(self, device):
        self.X = self.X.to(device)
        self.E = self.E.to(device)
        self.y = self.y.to(device)
        self.node_mask = self.node_mask.to(device)
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.to(device)
        if isinstance(self.mol_assignments, torch.Tensor):
            self.mol_assignments = self.mol_assignments.to(device)
        return self

    def to_numpy(self):
        self.X = self.X.detach().cpu().numpy()
        self.E = self.E.detach().cpu().numpy()
        self.y = self.y.detach().cpu().numpy()
        self.node_mask = self.node_mask.detach().cpu().numpy()
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.detach().cpu().numpy()
        if isinstance(self.mol_assignments, torch.Tensor):
            self.mol_assignments = self.mol_assignments.detach().cpu().numpy()
        return self

    def to_cpu(self):
        self.X = self.X.detach().cpu()
        self.E = self.E.detach().cpu()
        self.y = self.y.detach().cpu()
        self.node_mask = self.node_mask.detach().cpu()
        if isinstance(self.atom_map_numbers, torch.Tensor):
            self.atom_map_numbers = self.atom_map_numbers.detach().cpu()
        if isinstance(self.mol_assignments, torch.Tensor):
            self.mol_assignments = self.mol_assignments.detach().cpu()
        return self

    def mask(self, node_mask=None, collapse=False):
        if node_mask==None:
            node_mask = self.node_mask

        assert node_mask is not None, 'node_mask is None.'

        x_node_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_node_mask1 = x_node_mask.unsqueeze(2)            # bs, n, 1, 1
        e_node_mask2 = x_node_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1) # (bs, n)
            self.E = torch.argmax(self.E, dim=-1) # (bs, n, n)

            self.X[node_mask == 0] = 0
            self.E[(e_node_mask1 * e_node_mask2).squeeze(-1) == 0] = 0
        else:
            # always mask by node, masking by subgraph is a subset of that
            '''
                X_0 = NN(noisy.X) => (bs, n_max, v)
                => e.g.: true nodes: (0, n<n_max, v) = [0.9, 0.8, .....]
                         fake nodes: (0, n>n_max, v) = [0.9, 0.8, .....] => [1, 0, 0, 0....]

                => how to get correct fake nodes?
                => node_mask: (bs, n_max, 1)
                X_0 * node_mask => X_0' = (bs, n<n_max, v) = [0.9, 0.8, .....]
                                   X_0' = (bs, n>n_max, v) = [0, 0, .....] (doesn't exist)

                => last step: fix the [0, ...] to [1, 0, ...]
                => last step for other masks: X_0' = X_0 * node_mask + X_orig * (~node_mask)
                    => X_0': (bs, n<n_max, v) = [0.9, 0.8, .....] (e.g. output of NN)
                    => X_0': (bs, n>n_max, v) = [1, 0, .....] (e.g. orig one_hot) => perks: already proba dist, already one-hot...
            '''
            self.X = self.X * x_node_mask
            self.E = self.E * e_node_mask1 * e_node_mask2
            diag = torch.eye(self.E.shape[1], dtype=torch.bool).unsqueeze(0).expand(self.E.shape[0], -1, -1)
            self.E[diag] = 0
            self.X = encode_no_element(self.X)
            self.E = encode_no_element(self.E)

            # adjacency matrix of undirected graph => mirrored over the diagonal
            # assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def get_new_object(self, **kwargs):
        '''
            returns a new placeholder object with X, E or y changed
            and the other features copied from the current placeholder object.
        '''
        # get all attributes that are not functions
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]
        # logic below:
        # ... if we're given a new variable in the kwargs for the attribute,
        # ...... clone it and use it as the value for the new object
        # ... if the current object has a value for att, clone that instead.
        # ... if no value for the at anywhere, assign None.
        # We assume all attributes are tensors
        new_kwargs = {at: kwargs.get(at).clone() if isinstance(kwargs.get(at), torch.Tensor) else \
                          getattr(self, at).clone() if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}

        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]

        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors.'

        new_obj = PlaceHolder(**new_kwargs)

        return new_obj

    def select_subset(self, selection):
        '''
            return a placeholder object with the selection in the form of a boolean mask of shape (bs,)
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]

        new_kwargs = {at: getattr(self, at).clone()[selection] if isinstance(getattr(self, at), torch.Tensor) else None for at in attributes}

        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]

        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'

        new_obj = PlaceHolder(**new_kwargs)

        return new_obj

    def slice_by_idx(self, idx):
        '''
            return a placeholder object with the first idx batch elements.
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]

        new_kwargs = {at: getattr(self, at).clone()[:idx] if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}

        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]

        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'

        new_obj = PlaceHolder(**new_kwargs)

        return new_obj

    def subset_by_idx(self, start_idx, end_idx):
        '''
            return a placeholder object with the first idx batch elements.
        '''
        attributes = [a for a in self.__dir__() if '__' not in a and not callable(getattr(self, a))]

        new_kwargs = {at: getattr(self, at).clone()[start_idx:end_idx] if isinstance(getattr(self, at), torch.Tensor) else \
                          None \
                          for at in attributes}

        not_tensor_attribute = [not isinstance(getattr(self, at), torch.Tensor) and\
                                getattr(self, at) is not None for at in attributes]

        assert sum(not_tensor_attribute)==0, 'PlaceHolder object has attributes that are not tensors. These will be set to None in the new object.'

        new_obj = PlaceHolder(**new_kwargs)

        return new_obj

    def cat_by_batchdim(self, placeh):
        self.X = torch.cat((self.X, placeh.X), dim=0)
        self.E = torch.cat((self.E, placeh.E), dim=0)
        self.node_mask = torch.cat((self.node_mask, placeh.node_mask), dim=0)
        self.atom_map_numbers = torch.cat((self.atom_map_numbers, placeh.atom_map_numbers), dim=0)
        self.mol_assignments = torch.cat((self.mol_assignments, placeh.mol_assignments), dim=0)
        self.y = torch.cat((self.y, placeh.y), dim=0)

    def cat_by_batchdim_with_padding(self, placeh):
        # 1. choose which object to pad
        if self.X.shape[1] > placeh.X.shape[1]:
            to_pad = placeh
            ready = self
        else:
            to_pad = self
            ready = placeh

        # 2. pad object
        pad_size = ready.X.shape[1]-to_pad.X.shape[1]
        to_pad.pad_nodes(pad_size)

        # 3. cat
        ready.cat_by_batchdim(to_pad)

        return ready

    def pad_nodes(self, pad_size):
        padding_tuple_X = (0, pad_size) if self.X.ndim==2 else (0, 0, 0, pad_size)
        padding_tuple_E = (0, pad_size, 0, pad_size) if self.E.ndim==3 else (0, 0, 0, pad_size, 0, pad_size)
        padding_tuple_v = (0, pad_size)
        self.X = F.pad(self.X, padding_tuple_X, value=0)
        self.E = F.pad(self.E, padding_tuple_E, value=0)
        self.node_mask = F.pad(self.node_mask, padding_tuple_v, value=0)
        self.atom_map_numbers = F.pad(self.atom_map_numbers, padding_tuple_v, value=0)
        self.mol_assignments = F.pad(self.mol_assignments, padding_tuple_v, value=0)

    def select_by_batch_idx(self, idx):
        '''
            Return a placeholder graph specified by the batch idx given as input.
            The returned graph does not share same memory with the original graph.
            idx: batch idx given
        '''
        return PlaceHolder(X=copy.deepcopy(self.X[idx:idx+1]), E=copy.deepcopy(self.E[idx:idx+1]), y=copy.deepcopy(self.y[idx:idx+1]), node_mask=copy.deepcopy(self.node_mask[idx:idx+1]),
                           atom_map_numbers=copy.deepcopy(self.atom_map_numbers[idx:idx+1]), mol_assignments=copy.deepcopy(self.mol_assignments[idx:idx+1]))

    def select_by_batch_and_sample_idx(self, bs, n_samples, batch_idx, sample_idx):
        assert self.X.ndim==2, f'Expected X of shape (bs, n), got X.shape={self.X.shape}. Use mask(node_mask, collapse=True) before calling this function.'
        assert self.E.ndim==3, f'Expected E of shape (bs, n, n), got E.shape={self.E.shape}. Use mask(node_mask, collapse=True) before calling this function.'

        return PlaceHolder(X=self.X.reshape(bs, n_samples, self.X.shape[1])[batch_idx:batch_idx+1, sample_idx],
                           E=self.E.reshape(bs, n_samples, self.E.shape[2], -1)[batch_idx:batch_idx+1, sample_idx],
                           y=self.y.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx],
                           node_mask=self.node_mask.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx],
                           atom_map_numbers=self.atom_map_numbers.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx],
                           mol_assignments=self.mol_assignments.reshape(bs, n_samples, -1)[batch_idx:batch_idx+1, sample_idx])

    def serialize(self):
        return {"X": self.X.detach().cpu().numpy().tolist(), "E": self.E.detach().cpu().numpy().tolist(),
                "y": self.y.detach().cpu().numpy().tolist(), "node_mask": self.node_mask.detach().cpu().numpy().tolist(),
                "atom_map_numbers": self.atom_map_numbers.detach().cpu().numpy().tolist(),
                "mol_assignments": self.mol_assignments.detach().cpu().numpy().tolist()}

    def pyg(self):
        """Turns back into a pytorch geometric DataBatch() object, also with lesser precision for easier saving.
        To turn back to higher precision, there exists a function for that. pyg_to_full_precision_expanded() """

        # TODO: This or the other iteration are unnecessary, but it's okay
        return_data = []
        for i in range(self.X.shape[0]):
            # Concatenate the X as well
            E_idx, E_attr = dense_to_sparse(adj=self.E[i])

            X = self.X[i] if self.X.dim() == 2 else self.X[i].argmax(-1)
            assert X.dim() == 1

            # X = self.X[i].reshape(-1, self.X.shape[1]) if self.X.dim() == 2 else self.X[i].argmax(-1).reshape(-1, self.X.shape[1])
            atom_map_numbers = self.atom_map_numbers[i]
            node_mask = self.node_mask[i]
            mol_assignment = self.mol_assignments[i]

            # NOTE: atom mappings and mol_assignment have a different field names in the Data() objects and in the PlaceHolder objects. Needs to be accommodated here.
            return_data.append(Data(x=X.to(torch.uint8), edge_index=E_idx.to(torch.int16), edge_attr=E_attr.to(torch.uint8), y=self.y.to(torch.uint8), node_mask=node_mask.to(torch.uint8),
                        mask_atom_mapping=atom_map_numbers.to(torch.uint8), mol_assignment=mol_assignment.to(torch.uint8)))

        return Batch.from_data_list(return_data)
