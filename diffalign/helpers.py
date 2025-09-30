'''
This file contains helper functions for the multiguide package.
NOTE: best to keep this package as independent of third-party packages as possible. 
The goal is to make it callable by scripts from any conda environment.
'''

import os
from pathlib import Path
import torch

PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DotDict(dict):
    def __getattr__(self, key):
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]


def accumulate_rxn_scores(acc_scores, new_scores, total_iterations):
    '''
        Updates the acc_scores with new metric averages taking into account the new_scores.
        
        input:
            acc_scores: accumulated scores state
            new_scores: new_scores to add to the accumulation
            total_iterations: total number of batches considered. 
        output:
            acc_scores: accumulated scores state with the new_scores added.
    '''
    for metric in new_scores.keys():
        if type(new_scores[metric])==list: # accumulates the plots
            if acc_scores[metric]==0:
                acc_scores[metric] = new_scores[metric]
            else:
                acc_scores[metric].extend(new_scores[metric])
        else:
            acc_scores[metric] += new_scores[metric].mean()/total_iterations
        
    return acc_scores
    

def average_rxn_scores(scores_list, counts_of_samples_in_list_elements):
    '''
        Averages together the scores in scores_list. 
        
        input:
            scores_list: list of dicts containing the scores
            counts_of_samples_in_list_elements: list of integers with the number of samples used to calculate the scores in scores_list
        output:
            avg: averaged scores
    '''
    total_samples = sum(counts_of_samples_in_list_elements)
    avg_scores = {}
    for i, scores in enumerate(scores_list):
        for metric in scores_list[0].keys():
            if metric not in avg_scores.keys():
                if type(scores[metric])==list:
                    avg_scores[metric] = [scores[metric]]
                else:
                    avg_scores[metric] = scores[metric] * counts_of_samples_in_list_elements[i] / total_samples
            else:
                if type(avg_scores[metric])==list:
                    avg_scores[metric].extend(scores[metric])
                else:
                    avg_scores[metric] += scores[metric]  * counts_of_samples_in_list_elements[i] / total_samples
    return avg_scores


