import os
import hydra
from pathlib import Path
import logging
import pandas as pd
log = logging.getLogger(__name__)
PROJECT_ROOT = Path(os.path.realpath(__file__)).parents[0]

@hydra.main(config_path="configs", config_name="default.yaml")
def original_to_raw(cfg):
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    subsets = ['train', 'val', 'test']
    datadir = cfg.dataset.datadir
    if cfg.dataset.dataset_nb!='':
        datadir += '-'+str(cfg.dataset.dataset_nb)
    original_dir = os.path.join(PROJECT_ROOT, datadir, 'original')
    raw_dir = os.path.join(PROJECT_ROOT, datadir, 'raw')

    for subset in subsets:
        log.info(f'===== Converting {subset} =====')
        original_file = os.path.join(original_dir, f"raw_{subset}.csv")
        df = pd.read_csv(original_file)
        reactions_only = df['reactants>reagents>production'].tolist()
        log.info(f"read {len(reactions_only)} reactions from {original_file}")
        raw_file = os.path.join(raw_dir, f"{subset}.csv")
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.writelines([f"{i}\n" for i in reactions_only])
        log.info(f"wrote {len(reactions_only)} reactions to {raw_file}")

if __name__ == "__main__":
    original_to_raw()
