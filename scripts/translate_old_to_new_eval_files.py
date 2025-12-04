import os
import sys
import re
import pandas as pd

from setup_path import *
from diffalign.helpers import PROJECT_ROOT
from diffalign.data.helpers import compare_reactant_smiles

def deduplicate_exact_matches(df):
    # Create a mask for duplicates only where is_exact_match=1
    mask = (df['is_exact_match'] == 1) & df.duplicated(
        subset=['product', 'true_reactants'], 
        keep='last'
    )

    # Drop those rows
    out_df = df[~mask]
    return out_df

def translate_old_to_new_eval_files(old_eval_file):
    old_eval_path = os.path.join(PROJECT_ROOT, 'experiments', old_eval_file)
    new_eval_file = old_eval_file.split('.txt')[0] + '_new.csv'
    new_eval_path = os.path.join(PROJECT_ROOT, 'experiments', new_eval_file)

    old_eval = open(old_eval_path, 'r').read()

    # Regex to match entire blocks
    block_pattern = r'\(cond\s+\d+\)\s+[^>]+>>[^:]+:(?:\n\s+\([^\)]+\))*'
    block_pattern = r'\(cond\s+(\d+)\)\s+(.+?>>.+?):((?:\n\s+\(.+?,\s*\[.+?\]\))+)'
    data_line_pattern = r"\('(.+?)',\s*\[(.+?)\]\)"
    # block_pattern = r'\(cond\s+(\d+)\)\s+(.+?>>.+?):((?:\n\s+\(.+?\))+)'
    # data_line_pattern = r"\('(.+?)',\s*\[(.+?)\]\)"

    # data = "    ('O=CC(Br)=CN=CNC1CC2=CC=CC=C2C1>>O=CC1=CN=CN1C1CC2=CC=CC=C2C1', [0.23111647367477417, 0.2306087166070938, -0.0005096630193293095, 20, 0.07795154919754205])"
    # data_matches = re.findall(data_line_pattern, data)
    # print(f'len(data_matches): {len(data_matches)}')
    # exit()
    blocks = re.findall(block_pattern, old_eval)
    records = []

    for cond_num, true_rxn, data_lines in blocks:
        true_product = true_rxn.split('>>')[1]
        true_reactants = true_rxn.split('>>')[0]
        data_matches = re.findall(data_line_pattern, data_lines)
        print(f'len(data_matches): {len(data_matches)}, for cond_num: {cond_num}')
        for predicted_rxn, metrics_str in data_matches:
            metrics = [float(x.strip()) for x in metrics_str.split(',')]
            predicted_product = predicted_rxn.split('>>')[1]
            predicted_reactants = predicted_rxn.split('>>')[0]
            assert predicted_product == true_product, f"Predicted product {predicted_product} does not match true product {true_product}"
            is_exact_match, true_reactants_canonicalized, predicted_reactants_canonicalized = compare_reactant_smiles(true_reactants, predicted_reactants, return_reactants_canonicalized=True)
            # NOTE: could add more metrics here like reaction naming, round trip accuracy, etc.
            # NOTE: hmm looks like these old evals are not well canonicalized, there are still some duplicates.
            # TODO: deduplicate before sharing final file... can add the count but not sure what to do about the other metrics.
                # NOTE: for now keep the first duplicate only
            records.append({
                'condition': int(cond_num),
                #'true_reaction': true_rxn,
                "product": true_product,
                "true_reactants": true_reactants,
                "true_reactants_canonicalized": true_reactants_canonicalized,
                'pred_reactants': predicted_reactants,
                "pred_reactants_canonicalized": predicted_reactants_canonicalized,
                "is_exact_match": is_exact_match,
                # NOTE: will probably drop these metrics in latest evaluation script
                # x["elbo"], x["loss_t"], x["loss_0"], x["count"], x["weighted_prob"]
                'elbo': metrics[0],
                'loss_t': metrics[1],
                'loss_0': metrics[2],
                'count': metrics[3],
                'weighted_prob': metrics[4]
            })

    df = pd.DataFrame(records)
    #df = deduplicate_exact_matches(df)d
    #df = df.drop_duplicates(subset=['product', 'true_reactants_canonicalized', 'pred_reactants_canonicalized'], keep='first')
    df.to_csv(new_eval_path, index=False)
    print(f"Saved to {new_eval_path}")

if __name__ == "__main__":
    # old_eval_file = "7ck/eval_epoch720_steps100_resorted_0.9_cond4992_sampercond100_test_lam0.9.txt"
    # old_eval_file = "7ck/eval_epoch760_steps100_resorted_0.9_cond4992_sampercond100_val_lam0.9.txt"
    #experiment_dir = os.path.join(PROJECT_ROOT, 'experiments', 'testing_sample_array_job_20251019_000836')
    experiment_dir = os.path.join(PROJECT_ROOT, 'experiments', 'testing_sample_array_job_20251022_154711')
    for f in os.listdir(experiment_dir):
        if f.startswith('eval_epoch') and 'resorted' in f:
            old_eval_file = os.path.join(experiment_dir, f)
            translate_old_to_new_eval_files(old_eval_file)