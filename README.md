
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

<details>
<summary>Collapsible</summary>

Your content goes here. This can include text, code, images, or any markdown elements.

</details>

# Data and checkpoints

# Running the dummy copy graph example

- to train the model from scratch
- to evaluate the model on existing checkpoints

# Running the model on retrosynthesis
- training
- evaluation
