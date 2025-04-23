
# DiffAlign

Official implementation of the paper [Equivariant Denoisers Cannot Copy Graphs: Align your Graph Diffusion Models](https://openreview.net/forum?id=onIro14tHv&referrer=%5Bthe%20profile%20of%20Najwa%20Laabid%5D(%2Fprofile%3Fid%3D~Najwa_Laabid1))

# Installation

- Install the packages in ´req.txt´ in a custom environment of your choice (e.g. conda or pip environments)
    - I usually create a conda environmnet using e.g.: ´conda create -n diffalign python=3.10´
    - activate it: ´conda activate diffalign´
    - [optional] install pip in case I need it: ´conda install pip´
    - install the packages one by one to account for dependencies
- Install the diffalign editable packages by running (in the home directory where ´setup.py´is): 
        'pip install -e .'
- Notes: 
    - we need python 3.10 because ptyroch geometric only supports pytorch 2.2.0 (latest version) which is only available in older python versions
    - make sure you install the right torch/torch_geometric and their dependencies for your platform. More information on [torch](https://pytorch.org/get-started/locally/), [torch_geometric](https://pytorch-geometric.readthedocs.io/en/2.5.2/notes/installation.html)


# Running the dummy copy graph example
We generate data on the fly for this task. Checkpoints for the experiments reported in the paper coming soon.

- to train the model from scratch
- to evaluate the model on existing checkpoints

# Running the model on retrosynthesis
## Preparing the dataset and checkpoints 
We provide initial data and checkpoints below. More checkpoints coming soon.

<details>
<summary>Links to available data and checkpoints</summary>

- [Data](https://www.dropbox.com/scl/fo/swuggv6qf8ombw914yxh8/AEwUgTxowsq2vrnv0D2xRNg/schneider50k?dl=0&rlkey=1ed5tqauj7udn5n2olvw1looi&subfolder_nav_tracking=1): we use the same data split as [GLN](https://github.com/Hanjun-Dai/GLN?tab=readme-ov-file) called schneider_50k
- [DiffAlign_PE+Skip](https://figshare.com/articles/software/diffalign_pe_skip_connection_epoch620/28838489?file=53887232): our best checkpoint for our best performing model using positional encoding and skip connection.

</details>
To use the checkpoints:
- simply download the checkpoint of your choice and place it under `DiffAlign/checkpoints`.

To use the data splits:
1. Place the data subsets (`raw_train.csv`, `raw_val.csv`, and `raw_test.csv`) under `DiffAlign/data/<dataset_custom_name>/original`. 
2. run `python3 original_to_raw.py +experiment=<experiment-config.yaml> general.wandb.mode=offline`: choose a suitable experiment config from `configs/experiment/`, e.g.:
    - for the PE+Skip model: `python3 original_to_raw.py +experiment=diffalign_pe_skip.yaml general.wandb.mode=offline`

## Training 
- The first step in training is to process the reactions in the data splits to valid graph formats: 
    - this takes about ...

## Evaluation
