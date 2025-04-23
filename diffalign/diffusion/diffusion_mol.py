import os
import pathlib
import wandb
import logging
import time
import torch
from datetime import date

from rdkit import RDLogger
from fcd_torch import FCD
from diffalign.utils import graph, mol, setup
from diffalign.utils.diffusion.helpers import mol_diagnostic_chains
from diffalign.diffusion.diffusion_abstract import DiscreteDenoisingDiffusion

# A logger for this file
log = logging.getLogger(__name__)
RDLogger.DisableLog('rdApp.*') # Disable rdkit warnings

parent_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_NODES = 100

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class DiscreteDenoisingDiffusionMol(DiscreteDenoisingDiffusion):
    def __init__(self, cfg, dataset_infos, node_type_counts_unnormalized=None, edge_type_counts_unnormalized=None):
        super().__init__(cfg=cfg, dataset_infos=dataset_infos, node_type_counts_unnormalized=node_type_counts_unnormalized, 
                         edge_type_counts_unnormalized=edge_type_counts_unnormalized)
        
    @torch.no_grad()
    def evaluate(self, epoch, train_dataloader, test_dataloader, additional_dataloader=None, inpaint_node_idx=None, inpaint_edge_idx=None):
        
        assert additional_dataloader!=None or self.cfg.diffusion.diffuse_edges==True, "If conditional on edges, need to provide the additional dataloader."
        assert self.cfg.test.n_conditions>=self.cfg.train.chains_to_save, f"Sampling {self.cfg.test.n_conditions} precursors but asking to plot {self.cfg.train.chains_to_save} chains. # of chains should be <= # of samples."

        eval_start_time = time.time()
        ## TODO: UNCOMMENT THIS BACK
        elbo_of_data_time = time.time()
        log.info("calculating ELBO...")
        elbo_test = self.get_elbo_of_data(test_dataloader, n_samples=self.cfg.test.elbo_samples, device=device)
        elbo_train = self.get_elbo_of_data(train_dataloader, n_samples=self.cfg.test.elbo_samples, device=device)
        log.info(f"ELBO train: {elbo_train}, ELBO test: {elbo_test}. Time taken: {elbo_of_data_time - time.time()}")
        if self.cfg.train.log_to_wandb: wandb.log({'sample_eval/': {'train_elbo': elbo_train.item(), 'test_elbo': elbo_test.item()}})
        
        # evaluate final samples
        log.info("scoring samples...")
        sample_start_time = time.time()
        scores = self.score_samples(dataloader=additional_dataloader, epoch=epoch, inpaint_node_idx=inpaint_node_idx, inpaint_edge_idx=inpaint_edge_idx)
        log.info(f"===== all scores: {scores}")
        log.info(f"Time taken for scoring samples: {time.time()-sample_start_time}")
        
        # save scores and samples to wandb
        dir_= os.path.join(parent_path, f'experiments/rxn/results/{date.today()}/')
        setup.mkdir_p(dir_)
        open(os.path.join(dir_, f'scores-{self.cfg.test.testfile}.txt'), 'w').writelines(f'{scores}')
        if self.cfg.train.log_to_wandb: 
            wandb.log({'sample_eval/': scores})
            if os.path.exists(f'samples_epoch{epoch}.txt'): wandb.save(f'samples_epoch{epoch}.txt')

        log.info(f"Time taken for evaluation: {time.time()-eval_start_time}")
        
        return scores

    def score_samples(self, dataloader, data_class=None, save_samples_as_smiles=False, epoch=0):
        '''
            Score samples from the model, either with conditions from the full dataset.
        '''
        validity, uniqueness, novelty = 0, 0, 0
        mol_stability, atom_stability = 0, 0
        fcd, fcd_custom = 0, 0

        dataiter = iter(dataloader)
        # num_batches = n_samples/batch_size or n_batches in full dataloader
        if self.cfg.test.full_dataset: 
            num_batches = len(dataiter)
        else:
            num_batches = self.cfg.test.n_samples // self.cfg.test.batch_size + int(self.cfg.test.n_samples%self.cfg.test.batch_size >0)
            num_batches = max(num_batches, 1)

        for _ in range(num_batches):
            # try/except handling the case where n_samples>len(data)
            try:
                data = next(dataiter)
            except:
                dataiter = iter(dataloader)
                data = next(dataiter)
                
            data = data.to(device)
            scores_ = self.score_batch(data, data_class=data_class, save_samples_as_smiles=save_samples_as_smiles)
            validity += scores_['validity']/num_batches
            uniqueness += scores_['uniqueness']/num_batches
            novelty += scores_['novelty']/num_batches
            mol_stability += scores_['mol_stability']/num_batches
            atom_stability += scores_['atom_stability']/num_batches
            fcd += scores_['fcd']/num_batches
            fcd_custom += scores_['fcd_custom']/num_batches
        
            scores = {'validity': validity, 'uniqueness': uniqueness, 
                      'novelty': novelty, 'mol_stability': mol_stability,
                      'atom_stability': atom_stability, 'fcd': fcd, 'fcd_custom': fcd_custom}

        return scores
    
    def score_batch(self, data, data_class=None, save_samples_as_smiles=False):
        assert data_class is not None, 'data_class needed to compute custom_fcd.'
        # TODO: check_valid_mol_batch and check_stable_mols are super slow right now,
        # could try at least CPU parallelization
        dense_data = graph.to_dense(data).to_device(device)
        samples = self.sample_one_batch(data=dense_data, get_chains=False)
        
        validity, all_valid_smis, all_smiles = mol.check_valid_mol_batch(X=samples.X.argmax(-1), E=samples.E.argmax(-1),
                                                         atom_types=self.dataset_info.atom_decoder, bond_types=self.dataset_info.bond_decoder)
        # unique mols needs to be unique among valid
        uniqueness, unique_smiles = mol.check_unique_mols(all_smis=all_valid_smis)
        dataset_name = self.cfg.dataset.dataset+'-'+self.cfg.dataset.dataset_nb if self.cfg.dataset.dataset_nb!='' else self.cfg.dataset.dataset
        dataset_path = os.path.join(parent_path, 'data', dataset_name, 'raw', 'trainmols.csv')
        novelty = mol.check_novel_mols(smiles=all_smiles, train_dataset_path=dataset_path)
        mol_stability, atom_stability = mol.check_stable_mols(atoms=samples.X.argmax(-1), edges=samples.E.argmax(-1), 
                                                              atom_types=self.dataset_info.atom_decoder, bond_types=self.dataset_info.bond_decoder)
        # FCD score
        # The score is expected increase for smaller sample sizes
        # The sample statistic should probably be precalculated on the full data set
        from diffalign.utils.setup import get_custom_fcd_statistic_precalculated, get_standard_fcd_statistic_precalculated
        fcd_precalc_stats_custom = get_custom_fcd_statistic_precalculated(cfg=self.cfg, data_class=data_class)
        # The following is the actual statistic that was used in the original FCD paper, not based on our data or fcd_torch
        # ... but also it is based on the tensorflow implementation, so this is correct if the torch implementation is exactly correct
        fcd_stats = get_standard_fcd_statistic_precalculated(cfg=self.cfg)
        # canonicalize tries to canonicalize the molecules, but raises error if our smiles list contains totally invalid molecules
        # ... not sure if we should still have it, maybe with training we would get good enough molecules for RdKit to not throw an error?
        # Apparently it can make a difference, according to the github of the original FCD paper. 
        # UPDATE: Trying on test data, it definitely makes a difference :) Can only reach FCD of like 16 without canonize, and 0.45 with canonize for 5000 samples
        # .. SOLUTION: The following line canonicalizes the smiles for the convention used in FCD calculations
        canon_smiles = mol.canonicalize_smiles_fcd(all_smiles)
        fcd_class = FCD(device=device, n_jobs=0, canonize=False)
        fcd = fcd_class(pref=fcd_stats, gen=canon_smiles)
        fcd_custom = fcd_class(pref=fcd_precalc_stats_custom, gen=canon_smiles)

        scores = {'validity': validity, 'uniqueness': uniqueness, 'novelty': novelty, 
                  'mol_stability': mol_stability, 'atom_stability': atom_stability, 
                  'fcd': fcd, 'fcd_custom': fcd_custom}

        return scores
    
    def plot_diagnostics(self, dataloader, epoch=0):
        if dataloader!=None and (not self.cfg.diffusion.diffuse_edges or not self.cfg.diffusion.diffuse_nodes):
            adjacency_samples = next(iter(dataloader))
            adjacency_samples = adjacency_samples.to(device)
            dense_data = graph.to_dense(data=adjacency_samples)
            data = graph.PlaceHolder(X=dense_data.X[:self.cfg.train.chains_to_save], 
                                     E=dense_data.E[:self.cfg.train.chains_to_save],
                                     node_mask=dense_data.node_mask[:self.cfg.train.chains_to_save],
                                     y=dense_data.y)
        else:
            data = None
        
        dense_data = graph.to_dense(data).to_device(device)
        final_samples, actual_sample_chains, prob_s_chains, pred_0_chains = self.sample_one_batch(data=data, 
                                                                                                  n_samples=self.cfg.train.chains_to_save, 
                                                                                                  get_chains=True)
        
        # plot denoiser output
        log.info("Plot denoiser output")
        prob_s_chains = mol_diagnostic_chains(chains=prob_s_chains, atom_types=self.dataset_info.atom_decoder, 
                                          bond_types=self.dataset_info.bond_decoder, 
                                          chain_name=f'epoch{epoch}_denoiser')
        if self.cfg.train.log_to_wandb:
            for i, (chain_vid_path, mol_img_path, smi) in enumerate(prob_s_chains):
                wandb.log({f'sample_chains/epoch{epoch}_denoiser(x_t-1|x_t)_chain{i}': wandb.Video(chain_vid_path, fps=1, format='mp4')})
                #wandb.log({f'sample_mols/epoch{epoch}_denoiser(x_t-1|x_t)_mol{i}': wandb.Image(mol_img_path)})
                open(os.path.join(os.getcwd(), f'sampled_molecules.txt'), 'a').writelines(f'epoch{epoch}_denoiser(x_t-1|x_t)_smi{i}: {smi}\n')

        # # plot NN output
        log.info("Plot NN output")
        pred_0_chains = mol_diagnostic_chains(chains=pred_0_chains, atom_types=self.dataset_info.atom_decoder, 
                                          bond_types=self.dataset_info.bond_decoder, chain_name=f'epoch{epoch}_nn')
        if self.cfg.train.log_to_wandb:
            for i, (chain_vid_path, mol_img_path, smi) in enumerate(pred_0_chains):
                wandb.log({f'sample_chains/epoch{epoch}_NN(x_0|x_t)_chain{i}': wandb.Video(chain_vid_path, fps=1, format='mp4')})
                #wandb.log({f'sample_mols/epoch{epoch}_NN(x_0|x_t)_mol{i}': wandb.Image(mol_img_path)})
                open(os.path.join(os.getcwd(), f'sampled_molecules.txt'), 'a').writelines(f'epoch{epoch}_NN(x_t-1|x_t)_smi{i}: {smi}\n')

        # # plot the samples directly
        log.info("Plot sample chains")
        actual_sample_chains = mol_diagnostic_chains(chains=actual_sample_chains, atom_types=self.dataset_info.atom_decoder, 
                                          bond_types=self.dataset_info.bond_decoder, chain_name=f'epoch{epoch}_actual_sample')
        
        for i, (chain_vid_path, mol_img_path, smi) in enumerate(actual_sample_chains):
            open(os.path.join(os.getcwd(), f'sampled_molecules.txt'), 'a').writelines(f'epoch{epoch}_actual_sample_smi{i}: {smi}\n')
                      
        if self.cfg.train.log_to_wandb:
            for i, (chain_vid_path, mol_img_path, smi) in enumerate(actual_sample_chains):
                wandb.log({f'sample_chains/epoch{epoch}_actual_sample_chain{i}': wandb.Video(chain_vid_path, fps=1, format='mp4')})
                #wandb.log({f'sample_mols/epoch{epoch}_actual_sample_mol{i}': wandb.Image(mol_img_path)})

                