from syntheseus import BackwardReactionModel
from typing import Sequence
from syntheseus import Bag, Molecule, SingleProductReaction
import os
import logging
log = logging.getLogger(__name__)
import torch
import numpy as np
import sys
import pathlib
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
from rdkit import Chem

class OurModel(BackwardReactionModel):
    def __init__(self, wandb_id, epoch_num, repeat_elbo, loss_0_repeat, num_diffusion_steps, n_samples_per_condition, dataset_nb,
                 add_supernode_edges, add_supernodes):
        from diffalign.utils import setup
        import hydra

        # create an empty Hydra config with standard Hydra:
        hydra.initialize(config_path="../../configs/")
        cfg = hydra.compose(config_name="default.yaml")
        cfg.general.wandb.run_id = wandb_id

        # load the default config: (this sets 
        # run, cfg = setup.setup_wandb(cfg, job_type='ranking')

        super().__init__()

        # load the run config and override the default config:
        run_config = setup.load_wandb_config(cfg)
        entity = cfg.general.wandb.entity
        project = cfg.general.wandb.project
        cfg = setup.merge_configs(default_cfg=cfg, new_cfg=run_config, cli_overrides={})
        cfg.general.wandb.entity = entity
        cfg.general.wandb.project = project
        cfg.general.wandb.run_id = wandb_id
        cfg.test.repeat_elbo = repeat_elbo
        cfg.test.loss_0_repeat= loss_0_repeat
        cfg.diffusion.diffusion_steps_eval = num_diffusion_steps
        cfg.test.n_samples_per_condition = n_samples_per_condition
        cfg.dataset.dataset_nb = dataset_nb
        cfg.dataset.add_supernode_edges = add_supernode_edges
        cfg.dataset.add_supernodes = add_supernodes

        datamodule, dataset_infos = setup.get_dataset(cfg=cfg, dataset_class=setup.task_to_class_and_model[cfg.general.task]['data_class'],
                                                    shuffle=cfg.dataset.shuffle, return_datamodule=True, recompute_info=False, 
                                                    slices={'train': None, 'val': None, 'test': None})
        
        device_count = torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f'device_count: {device_count}, device: {device}\n')

        model, optimizer, scheduler, scaler, start_epoch = setup.get_model_and_train_objects(cfg, model_class=setup.task_to_class_and_model[cfg.general.task]['model_class'], 
                                                                                            model_kwargs={'dataset_infos': dataset_infos, 
                                                                                                        'node_type_counts_unnormalized': dataset_infos.node_types_unnormalized,
                                                                                                        'edge_type_counts_unnormalized': dataset_infos.edge_types_unnormalized,
                                                                                                        'use_data_parallel': device_count>1},
                                                                                            parent_path=parent_path, savedir=os.path.join(parent_path, 'experiments'), 
                                                                                            load_weights_bool=False, device=device, device_count=device_count)
        
        # 4. load the weights to the model
        savedir = os.path.join(parent_path, "experiments", "trained_models", cfg.general.wandb.run_id)
        model, optimizer, scheduler, scaler, artifact_name_in_wandb = setup.load_weights_from_wandb_no_download(cfg, epoch_num, savedir, model, optimizer, 
                                                                                                            scheduler, scaler, device_count=device_count)
        
        # self.default_num_results = 10
        # self._cache = []
        
        self.model = model

    def _get_reactions(
        self, inputs: list[Molecule], num_results: int
    ) -> list[Sequence[SingleProductReaction]]:
        return [
            self._get_reactions_single(mol)[:num_results]
            for mol in inputs
        ]

    def _get_reaction_score(self, i: int, n_atoms: int) -> float:
        # Give higher score to reactions which break the input into
        # equal-sized pieces.
        return float(min(i, n_atoms - i))

    def _get_reactions_single(
        self, mol: Molecule
    ) -> Sequence[SingleProductReaction]:
        n = len(mol.smiles)

        reactants, probs = self.model.product_smiles_to_reactant_smiles(mol.smiles)

        # filter out invalid ones
        reactants, probs = zip(*[(r,p) for r,p in zip(reactants, probs) if Chem.MolFromSmiles(r)!=None])

        reactions = []

        for i, reactant in enumerate(reactants):
            r_objs = [Molecule(r) for r in reactant.split('.')]
            reactions.append(
                SingleProductReaction(
                    reactants=Bag(r_objs),
                    product=mol,
                    metadata={"probability": probs[i]},
                )
            )
    
        return sorted(
            reactions,
            key=lambda r: r.metadata["probability"],
            reverse=True,
        )
    
from syntheseus.search.analysis.route_extraction import (
    iter_routes_time_order,
)

from syntheseus.search.mol_inventory import SmilesListInventory
from syntheseus.search.algorithms.best_first.retro_star import (
    RetroStarSearch
)
from syntheseus.search.node_evaluation.common import (
    ConstantNodeEvaluator,
    ReactionModelLogProbCost,
)

def get_routes(model, starting_smiles, purchasable_smiles):
    search_algorithm = RetroStarSearch(
        reaction_model=model,
        mol_inventory=SmilesListInventory(smiles_list=purchasable_smiles),
        limit_iterations=100,  # max number of algorithm iterations
        limit_reaction_model_calls=100,  # max number of model calls
        time_limit_s=600.0,  # max runtime in seconds
        value_function=ConstantNodeEvaluator(0.0),
        and_node_cost_fn=ReactionModelLogProbCost(),
    )

    output_graph, _ = search_algorithm.run_from_mol(
        Molecule(starting_smiles)
    )
    routes = list(
        iter_routes_time_order(output_graph, max_routes=100)
    )

    print(f"Found {len(routes)} routes")
    return output_graph, routes

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp_dim', type=int, default=2048)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--wandb_id', type=str, default='7ckmnkvc')
    parser.add_argument('--epoch_num', type=int, default=0)
    parser.add_argument('--num_diffusion_steps', type=int, default=10)
    parser.add_argument('--samples_per_condition', type=int, default=50)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--repeat_elbo', type=int, default=1)
    parser.add_argument('--loss_0_repeat', type=int, default=1)
    parser.add_argument('--dataset_nb', type=str, default=None)
    parser.add_argument('--add_supernode_edges', type=bool, default=False)
    parser.add_argument('--add_supernodes', type=bool, default=False)
    parser.add_argument("-k", "--beam_size", help="beam size", type=int, default=5)
    args, unparsed = parser.parse_known_args()
    beam_size = args.beam_size
    wandb_id = args.wandb_id
    epoch_num = args.epoch_num
    repeat_elbo = args.repeat_elbo
    num_diffusion_steps = args.num_diffusion_steps
    loss_0_repeat = args.loss_0_repeat
    n_samples_per_condition = args.samples_per_condition
    dataset_nb = args.dataset_nb
    add_supernode_edges = args.add_supernode_edges
    add_supernodes = args.add_supernodes

    dataset_nb = "dummy15-stereo-charges-in-atom-types"
    add_supernodes = True

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    import pickle
    routes = pickle.load(open("/u/38/rissans2/unix/Work/RetroDiffuser/src/syntheseus/routes_possible_test_hard.pkl", 'rb'))

    starting_smiles = [r[0].split('>>')[0] for r  in routes]
    purchasable_smiles_file = "/u/38/rissans2/unix/Work/RetroDiffuser/src/syntheseus/origin_dict.csv"
    purchasable_smiles = pd.read_csv(purchasable_smiles_file)
    purchasable_smiles = purchasable_smiles['mol'].tolist()
    # with open(purchasable_smiles_file, "r") as f:
    #     purchasable_smiles = f.readlines()

    backwardmodel = OurModel(wandb_id, epoch_num, repeat_elbo, loss_0_repeat, 
                             num_diffusion_steps, n_samples_per_condition, dataset_nb, 
                             add_supernode_edges, add_supernodes)
    
    # def print_predictions(backwardmodel, smiles: str):
    #     [reactions] = backwardmodel([Molecule(smiles)])
    #     for reaction in reactions:
    #         probability = reaction.metadata["probability"]
    #         print(f"{reaction} (probability: {probability:.3f})")

    # print_predictions(backwardmodel, "CCCC")
    # print(backwardmodel.num_calls())

    all_routes = []

    for smiles in starting_smiles:
        output_graph, routes = get_routes(backwardmodel, smiles, purchasable_smiles)
        all_routes.append(routes)

    # save all_routes to file
    with open("all_routes.pkl", "wb") as f:
        pickle.dump(all_routes, f)

    # from syntheseus.search.visualization import visualize_andor

    # for name, idx in [("first", 0), ("last", -1)]:
    #     visualize_andor(
    #         output_graph, filename=f"route_{name}.pdf", nodes=routes[idx]
    #     )
    pass