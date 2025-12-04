import os
import pandas as pd

from setup_path import *
from diffalign.helpers import PROJECT_ROOT

def readout_metrics(df, topk_key='is_exact_match'):
    # eval_path = os.path.join(PROJECT_ROOT, 'experiments', eval_file)
    # df = pd.read_csv(eval_path)
    num_products = df['product'].nunique()
    # find one product with exact matches more than 1
    # exact_matches_df = df[df['is_exact_match'] == True]
    # exact_matches_df = exact_matches_df.groupby('product').filter(lambda x: x.shape[0] > 1)
    # print(exact_matches_df)
    # exit()
    topk = {1: 0, 3: 0, 5: 0, 10: 0, 50: 0, 100: 0}
    # Original logic for exact matches
    topk_with_rank = df.groupby('product').apply(
            lambda x: pd.DataFrame({topk_key: x.reset_index(drop=True)[topk_key]==True})
        ).reset_index()
    topk_matches_df = topk_with_rank[topk_with_rank[topk_key]]
    # NOTE: temp solution, need to deduplicate
    #topk_matches_df = topk_matches_df.groupby('product').min().reset_index()
    for k in topk:
        topk[k] = topk_matches_df[topk_matches_df['level_1']+1<=k].shape[0]/num_products
    print(topk)

if __name__ == "__main__":
    # eval_file = "7ck/eval_epoch720_steps100_resorted_0.9_cond4992_sampercond100_test_lam0.9_new.csv"
    # eval_file = "7ck/eval_epoch760_steps100_resorted_0.9_cond4992_sampercond100_val_lam0.9_new.csv"
    #experiment_dir = os.path.join(PROJECT_ROOT, 'experiments', 'testing_sample_array_job_20251019_000836')
    experiment_dir = os.path.join(PROJECT_ROOT, 'experiments', 'testing_sample_array_job_20251022_154711')
    dfs = []
    for f in os.listdir(experiment_dir):
        if f.startswith('eval_epoch') and 'resorted' in f and 'new' in f:
            eval_file = os.path.join(experiment_dir, f)
            df = pd.read_csv(eval_file)
            dfs.append(df)
    df = pd.concat(dfs)
    readout_metrics(df)