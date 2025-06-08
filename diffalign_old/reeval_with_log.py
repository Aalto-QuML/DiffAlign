'''
Evaluate the samples saved as wandb artifacts.
'''
# A script that takes a run id, (a list of epochs) that we have good evaluations for, and updates new re-ranked evaluations to wandb. Also the lambda re-ranking value.
# What we need:
# 1. The run id
# 2. The list of epochs for which we have results from wandb automatically (TODO: How to do this?)
# retrieve the correct data from wandb for a given run id and epoch
# get the config file for the run, and create a model based on it (from diffusion.diffusion_rxn import DiscreteDenoisingDiffusionRxn)
# Then transform the samples.txt data into the elbo_ranked format
# Then get weighted_prob_sorted_rxns = graph.reactions_sorted_with_weighted_prob(elbo_sorted_rxns, self.cfg.test.sort_lambda_value)
# true_rcts, true_prods = graph.split_reactions_to_reactants_and_products(true_rxn_smiles)
# topk_weighted = graph.calculate_top_k(self, weighted_prob_sorted_rxns, true_rcts, true_prods)
import os
from diffalign_old.utils import wandb_utils, io_utils, graph
import wandb
from omegaconf import OmegaConf, DictConfig
import numpy as np
import pickle
import hydra
import random
import torch
import logging
import pathlib
import re
import sys
from diffalign_old.utils import setup
from diffalign_old.utils import mol
from collections import Counter


# A logger for this file
log = logging.getLogger(__name__)
parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. The run id, etc.
@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):      
    # TODO: Change this such that it uses the correct stuff
    
    file_path = '/Users/laabidn1/RetroDiffuser/src/samples_epoch320_steps100_cond4949_sampercond100_1705921479.txt'
    condition_range = [0, 512]
    data = open(file_path, 'r').read()
    reactions = io_utils.read_saved_reaction_only_data(data)
    topk = [1, 3, 5, 10, 50, 100]
    res = {k:0 for k in topk}
    
    for i, t in enumerate(reactions):
        orig = t[0]
        gen = t[1]
        # get unique rxn conf
        conf_dict = {r:np.log(c/len(gen)+1e-6) for r,c in Counter(gen).items()}
        sorted_dict = sorted(conf_dict.items(), key=lambda item: item[1], reverse=True)
        # print(f'sorted_dict: {sorted_dict}\n')
        for k in topk:
            if orig in [rxn[0] for rxn in sorted_dict[:k]]:
                res[k] += 1
                
    res = {k:v/len(reactions) for k,v in res.items()}
    print(f'res: {res}\n')

if __name__ == "__main__":
    main()
    # try:
    #     # save_default_config()
    #     main()
    # except Exception as e:
    #     log.exception("main crashed. Error: %s", e)
