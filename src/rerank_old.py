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
from src.utils import wandb_utils
from src.utils import io_utils
from src.utils import graph
import wandb
from omegaconf import OmegaConf
import time
import numpy as np
import sys
import re

def main():
    # Parse arguments
    run_id = sys.argv[1] # e.g., c64gs0eh
    sort_lambda_value = float(sys.argv[2]) # a float, usually 0.9
    num_conditions = int(sys.argv[3]) # an integer, usually 128
    repeat_elbo = int(sys.argv[4]) # an integer, usually 1
    clfg_weight = float(sys.argv[5]) # a float, usually 0.1
    lambda_test = sys.argv[6]
    conditional_set = sys.argv[7]

    project = "retrodiffuser"
    entity = "najwalb"
    # whether to use the new or old key system (for choosing samples to reeval)
    # new is noduplicates
    new_key = False 
    # whether the numbers for each reaction contains the duplicates count
    # True for new samples (with noduplicates)
    with_count = new_key

    # sort_lambda_values = [0.9]
    # num_conditions = 128

    # The list of epochs for which we have results from wandb automatically (TODO: How to do this?)

    run_path = wandb_utils.get_run_path(run_id, project, entity)
    run = wandb_utils.get_run(run_path)
    cfg = OmegaConf.create(run.config)
    cfg.test.n_conditions = num_conditions # TODO: This is a hack to get the correct key for the results. 
    cfg.test.repeat_elbo = repeat_elbo # TODO: This is a hack to get the correct key for the results.
    cfg.diffusion.classifier_free_guidance_weight = clfg_weight
    cfg.diffusion.lambda_test=lambda_test
    cfg.diffusion.edge_conditional_set=conditional_set

    # This designates which sampling thing to load the results from 
    old_key = wandb_utils.eval_key(cfg, new_key=False) # TODO: E.g., "eval_ncond_val_128", corresponds to new setup eval_noduplicates_ncond_val_128_elborep_1
    new_key = wandb_utils.eval_key(cfg, new_key=True) # TODO: E.g., "eval_noduplicates_ncond_val_128_elborep_1"
    savedir = os.path.join("experiments", "trained_models", run.name + run_id)

    # TODO: Figure out what to do when the duplicates are already removed (in the newer runs)
    # TODO: Can we combine the results if there are some things with old and new runs? Nah too complicated

    # Get the samples.txt files for the run and also the epochs where we have evaluations in the first place
    files = run.files()
    epochs = []

    # print(new_key)
    # Check if any file starts with the new key (and thus )
    load_data_in_new_format = any([file.name.startswith(os.path.join(savedir, new_key)) for file in files])
    if load_data_in_new_format: key_for_loading = new_key
    else: key_for_loading = old_key

    # print(load_data_in_new_format)
    # print(key_for_loading)
    print(os.path.join(savedir, key_for_loading))

    for file in files:
        # print(file)
        # print("<File", os.path.join(savedir, new_key))
        if file.name.startswith(os.path.join(savedir, key_for_loading)):
            filename = os.path.basename(file.name)
            pattern = r'samples_epoch(\d+)\.txt'
            match = re.match(pattern, filename)
            if match:
                # Download the file
                os.makedirs(os.path.dirname(file.name), exist_ok=True)
                file.download(replace=True)
                # TODO: This might now work now
                # epoch = os.path.basename(file.name).split("_")[-1].split(".")[0][5:] # remove the "samples_epoch" part and the ".txt" part
                epoch = match.group(1)
                epochs.append(int(epoch))

    all_scores = []

    epochs = sorted(epochs)

    all_scores = []

    for i, epoch in enumerate(epochs):
        print(f'epoch {epoch}\n')
        
        # Load the downloaded data
        with open(os.path.join(savedir, key_for_loading + f"samples_epoch{epoch}.txt")) as f:
            samples = f.read()
        
        # Then transform the samples.txt data into the elbo_ranked format (with the counts if they were in the data)
        reactions = io_utils.read_saved_reaction_data(samples)
        
        try:# This may fail if the data is in the old format
            print("EPOCH ____ ", epoch)
            elbo_sorted_rxns_no_duplicates = io_utils.restructure_reactions(reactions, with_count=True)
        except:
            elbo_sorted_rxns = io_utils.restructure_reactions(reactions, with_count=False)
            # we need gen_rxn_smiles, which is a list of lists of strings, where each string is a reaction SMILES
            gen_rxn_smiles = []
            for key in elbo_sorted_rxns.keys():
                gen_rxn_smiles.extend([".".join(reaction['rcts']) + ">>" + reaction['prod'][0] for reaction in elbo_sorted_rxns[key]])
            # unique_indices, counts, is_unique = graph.get_unique_indices_from_reaction_list(gen_rxn_smiles)
            elbo_sorted_rxns_no_duplicates = io_utils.remove_duplicates_and_select_random(elbo_sorted_rxns)
        
        # Then rerank them with the new lambda value
        weighted_prob_sorted_rxns = graph.reactions_sorted_with_weighted_prob(elbo_sorted_rxns_no_duplicates, sort_lambda_value)
        # Then get the top k
        true_rcts = [r[0].split('>>')[0].split('.') for r in reactions]
        true_prods = [[r[0].split('>>')[1]] for r in reactions]
        topk = graph.calculate_top_k(cfg, elbo_sorted_rxns_no_duplicates, true_rcts, true_prods)
        topk_weighted = graph.calculate_top_k(cfg, weighted_prob_sorted_rxns, true_rcts, true_prods)
        print(topk.shape)
        print(topk_weighted.shape)
        print(topk_weighted[:,1])
        # Then prepare the results for a format that can be saved to wandb
        scores = {'epoch': epoch}
        for j, k_ in enumerate(cfg.test.topks):
            scores[f'top-{k_}'] = topk[:,j].mean().item()
            # for i, sort_lambda_value in enumerate(sort_lambda_values):
            scores[f'top-{k_}_weighted_{sort_lambda_value}'] = topk_weighted[:,j].mean().item()
        all_scores.append(scores)

        # Then save the non-duplicated and sorted samples locally
        key = wandb_utils.eval_key(cfg)
        wandb_path = os.path.join(savedir, key)
        os.makedirs(wandb_path, exist_ok=True)
        graph.save_samples_to_file_without_weighted_prob(os.path.join(wandb_path, f'samples_epoch{epoch}_noduplicates.txt'), 0, elbo_sorted_rxns_no_duplicates, true_rcts, true_prods)
        # for i, sort_lambda_value in enumerate(sort_lambda_values):
            # weighted_prob_sorted_rxns = weighted_prob_sorted_rxns_[i]
        graph.save_samples_to_file(os.path.join(wandb_path, f'samples_epoch{epoch}_resorted_{sort_lambda_value}.txt'), 0, weighted_prob_sorted_rxns, true_rcts, true_prods)
    
    print(all_scores)
    print(new_key)

    # Log the results to wandb, with custom fields to group the training run and this evaluation run together
    # ... in the future, we should use groups for this
    run.config["experiment_group"] = run.name
    run.update()
    cfg["experiment_group"] = run.name
    
    # Automatically check whether we can use groups or not
    group_name = run.group
    cfg = OmegaConf.to_container(cfg)
    if group_name:
        wandb_kwargs = {'name': f'{key}', 'project': project, 'entity': entity, 'config': cfg, 'job_type': 'eval'}
    else:
        wandb_kwargs = {'name': f'{key}', 'project': project, 'entity': entity, 'config': cfg, 'job_type': 'eval',
                        'group': group_name}

    with wandb.init(**wandb_kwargs) as run:
        for idx, epoch in enumerate(epochs):    
            # Save to wandb
            scores = all_scores[idx]
            
            # Move back to the original directory so that wandb.save works properly
            wandb.save(os.path.join(wandb_path, "all_scores.txt"))
            wandb.save(os.path.join(wandb_path, f'samples_epoch{epoch}_noduplicates.txt'))
            wandb.save(os.path.join(wandb_path, f'samples_epoch{epoch}_resorted_{sort_lambda_value}.txt'))
            wandb.log({key: scores})

if __name__ == '__main__':
    # run_ids = ["2z7h3qx5"]
    main()
