# Debug
- the plotting with imageio ffpmeg on puhti

# Puhti info
- DiffAlign: cd /scratch/project_2015607/DiffAlign/

# Plan
Options: 
- Get the corresponding transition matrices from my thing and implement them as an StateTransition object in noise_schedule.py. Mainly the init could be that we pre-calculate all the transition matrices, and this requires some new code -> after that, smooth sailing?
	- The beta_t and alpha_bar_t interface doesn't quite match with what we have, we just have the timestep t and do some custom stuff based on that
	- But the PredefinedNoiseScheduleDiscrete might just be enough here
		- Wait what do you do with the alpha_bars and betas in a discrete diffusion model?
		- Okay you do something
		- But anyways, combining these two should be enough for our purposes here
- What is the AbsorbingStateTransition currently doing in noise_schedule.py? is it usable? 
- Another option is to do replace the statetransition object with something else

- Functions that need some changing in DiscreteDiffusion thing: 
	- __init__: Need to redefine the noise_schedule, limit_dist and transition_model. Maybe turn those into a single thing in the implementation in noise_schedule (or create an alternative file where all have the same form), since those are not really needed as separate objects here anyways: always used in conjunction
	- apply_noise: Definition of the Qtb matrices redefined!
	- KL_prior: same thing
	- Compute_Lt: same thing
	- reconstruction_logp: noise_schedule and transition_model as usual
	- sample_p_zs_given_zt: same as others 

- In noise_schedule:
	- Maybe just define a new object that takes care of both scheduling (no limiting to alpha and beta) and also the transition matrices, and plug those into all the functions mentioned before. The alphas and betas can't really stay, in any case. 
	
- Aside from that: 
	- Need to hook up my diffusions to generate these transition matrices. Start with maskdiffusion
	- How to generate a transition matrix? How to place the generating code? I guess I could add a function in MaskDiffusion.py to generate the corresponding transition matrices. -> Then define these MaskDiffusion objects in the noise_schedule.py class
		- could also check how they match with the thing implemented already in noise_schedule.py
	- Then we should be good to go? 
	
- Alright also change: the time normalization to [0,1], doesn't make that much sense

- Try implementing the mask diffusion such that empty edges are mask state
- Also try fixing the standard tokenwisediffusion code on the frequency calculations: What is the problem?
	- The empty edge is taken away instantly: Not calculated correctly I think
- But doesn't matter if we define it to be mask!

# Notes on tech
- Problem of video codec on wandb, e.g. when saving the diagnostic plots during training:
    - Issue: mozilla (and other browsers) do not support all video codec (compression algorithm used in the creation of mp4 videos):
        - List of codecs supported by mozilla: https://support.mozilla.org/en-US/kb/html5-audio-and-video-firefox#w_supported-formats
        - General info on codecs: https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Video_codecs
    - Solution: need to find a codec that works for: wandb on key browsers (mozilla, chrome), vscode on mac/linux/slurm cluster + how to pass it to imageio/ffmpeg from mac and linux.
        - How to pass codec to imageio: https://imageio.readthedocs.io/en/v2.8.0/format_ffmpeg.html
        - codec supported by imageio: https://imageio.readthedocs.io/en/stable/formats/video_formats.html
        - codec supported by vscode: https://code.visualstudio.com/api/extension-guides/webview#supported-media-formats

    - Codec value that worked is: 'h264', passed to imageio get_writer kwargs as {'codec':'h264'}
    - Codec values tried:
        - vp9: not supported by vscode, worked on mozilla/wandb, can still open file locally using vlc:
        - vp8: supported by vscode but couldn't pass it to imageio
          - TODO: ask why vp8 gives error by imageio when supported by ffmpeg:
            https://github.com/imageio/imageio/issues/1010

# LUMI instruction links
https://docs.csc.fi/apps/by_system/#lumi
https://docs.csc.fi/apps/pytorch/
https://docs.lumi-supercomputer.eu/software/installing/container-wrapper/

To set up the Slingshot interconnect thing (after loading the pytorch module)
module load LUMI/22.08 partition/G
module load EasyBuild-user
eb aws-ofi-rccl-66b3b31-cpeGNU-22.08.eb -r

# Running GPU on sinteractive
sinteractive --account project_2006174 --cores 4 --gpu 1 --time 2:00:00 --mem 30G --tmp 100
sinteractive --account project_2006174 --cores 4 --gpu 1 --time 12:00:00 --mem 30G --tmp 100
sinteractive --account project_2006950 --cores 4 --gpu 1 --time 12:00:00 --mem 30G --tmp 100
## On LUMI
salloc --gres=gpu:1 --account=project_462000276 --partition=dev-g --time=02:00:00 --mem-per-cpu=10G --cpus-per-task=7
srun --overlap --pty --jobid=<jobid> $SHELL

# To see rocm-smi on a node
srun --interactive --pty --jobid=<jobid> rocm-smi

# Loading the environment
export PATH="/projappl/project_2006950/retrodiffuser/bin:$PATH" 
export PATH="/projappl/project_2006174/retrodiffuser/bin:$PATH"

# Making git work on Arno puhti
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# Resource benchmarking
- Train uniform, ce, e200, seed5: time=10:15:00, memory= 60.49GB, GPU, 10 cores per node (triton)
- Test uniform, seed 1, 500 steps, atom accuracy: time=3:30:00, memory= 6GB

# Data preprocessing
- Only consider discrete features for atoms/bonds (for now only one hot encoding)
- Digress one hot encoding edges:
    - no edge is an edge type

# Code

```
        # if masks:
            #split_idx = masks['node_mask_prod_and_sn'].sum(dim=-1).cumsum(dim=-1)
            # node_mask_r = node_mask[masks['node_mask_prod_and_sn']]
            # print(f'node_mask_r.shape {node_mask_r.shape}\n')
            # mask_split = torch.tensor_split(node_mask_r, split_idx)
            # mask_nested = torch.nested.nested_tensor(list(mask_split)[:-1])
            # node_mask_r = torch.nested.to_padded_tensor(input=mask_nested, padding=False)

            # X_r = X[masks['node_mask_prod_and_sn']][...,:-1]
            # X_ = X.flatten(start_dim=0, end_dim=1)
            # print(f'X.shape {X.shape}\n')
            # print(f'X_.shape {X_.shape}\n')
            # print(f'X_r.shape {X_r.shape}\n')
            
            # mask_split = torch.tensor_split(X_r, split_idx)
            # mask_nested = torch.nested.nested_tensor(list(mask_split)[:-1])
            # X_r = torch.nested.to_padded_tensor(input=mask_nested, padding=False)[...,:-1]

            # E_r = E[masks['node_mask_prod_and_sn']].flatten(start_dim=0, end_dim=1)[...,:-3]
            # E_ = E.flatten(start_dim=0, end_dim=2)
            # print(f'E.shape {E.shape}\n')
            # print(f'E_r.shape {E_r.shape}\n')
            # print(f'E_.shape {E_.shape}\n')
            # exit()

            # mask_split = torch.tensor_split(E_r_, split_idx)
            # mask_nested = torch.nested.nested_tensor(list(mask_split)[:-1])
            # E_r_ = torch.nested.to_padded_tensor(input=mask_nested, padding=False)
            # E_r_p = E_r_.permute(0, 2, 1, 3)

            # E_r_ = E_r_p[masks['node_mask_prod_and_sn']]
            # mask_split = torch.tensor_split(E_r_, split_idx)
            # mask_nested = torch.nested.nested_tensor(list(mask_split)[:-1])
            # E_r = torch.nested.to_padded_tensor(input=mask_nested, padding=False)[...,:-3]

        # print(f'masks["mask_product_and_sn"].shape {masks["mask_product_and_sn"].shape}\n')
        # print(f'node_mask.shape {node_mask.shape}\n')
        # print(f'node_mask_r.shape {node_mask_r.shape}\n')
        # print(f'X.shape {X.shape}\n')
        # print(f'X_r.shape {X_r.shape}\n')
        # print(f'E.shape {E.shape}\n')
        # print(f'E_r_.shape {E_r_.shape}\n')
        # print(f'E_r.shape {E_r.shape}\n')
```

# refactoring recommendations for train.py

Your script seems well-organized and follows many best practices for Python development. Here's a detailed review based on readability, maintainability, and adherence to software engineering best practices:

### Readability:
1. **Comments and Docstrings**: There are no comments or docstrings in your code. Including them would help other developers (and future you) understand the purpose of different parts of the code. For instance, explaining the purpose of the `main` function, what `cfg` is expected to contain, etc.
   
2. **Variable Naming**: Your variable names are mostly clear and descriptive which is great. However, there are some variables like `t0` whose purpose might not be immediately clear to someone reading the code. A more descriptive name could enhance readability.

3. **Print Statements**: The print statement for training epochs could be converted to a log statement for consistency and better control over where your output goes.

4. **Code Structure**: Your `main` function is quite long. Consider breaking it down into smaller functions to enhance readability.

### Maintainability:
1. **Modularity**: Parts of your script could be made into functions to improve modularity. For example, the training loop could be its own function, evaluation could be its own function, etc.

2. **Hardcoded Values**: There are a few hardcoded values (like the `.pt` file extension and the sample filename format `'samples_epoch{epoch}.txt'`). Consider defining them as constants at the beginning of your script or including them in the configuration file.

3. **Error Handling**: You have a try-except block around your main function call, which is good for catching unexpected errors. However, you could also include more specific error handling throughout your code.

### Software Engineering Best Practices:
1. **Configuration Management**: You’re using Hydra for configuration management, which is a good practice.

2. **Logging**: You are using logging, which is great. Ensure that you configure your logger appropriately elsewhere in your codebase.

3. **Dependency Management**: Ensure that all your dependencies are listed in a requirements file or equivalent to make setting up the environment easy for any other developers (or yourself in the future).

4. **Use of Warnings**: You've used `warnings.filterwarnings("ignore", category=PossibleUserWarning)` to suppress warnings of a specific category. Make sure that this is absolutely necessary, as warnings are usually there for a reason.

5. **Import Statements**: You have two separate import statements for `datetime`. You might want to clean that up.

6. **Reproducibility**: You’ve set random seeds, which is good for reproducibility.

7. **Error Messages**: Your assert statements have error messages, which is good. Make sure all of them are informative enough to help diagnose issues when they arise.

8. **Resource Management**: Ensure that all resources (like file handles if you are opening files elsewhere in your code) are managed properly, using context managers (`with` statements) where applicable.

9. **Code Duplication**: There is a bit of code duplication in the model saving section. This could be abstracted into a function.

10. **Use of Global Variables**: `device` is defined as a global variable, which might not be the best practice. Consider passing it as a parameter to functions that need it.

Overall, your script is quite well-structured, but paying attention to the details mentioned above could help enhance its readability, maintainability, and adherence to best practices.


# Notes on adding the entire test set validation, implementation
New things: Contains code for evaluating on the entire test/validation set efficiently. 

Notes on evaluating on the entire test set:
- Previously the pipeline was: eval_from_wandb_wrapper.py followed by rerank_old.py. eval_from_wandb_wrapper parallelized the evaluation with respect to different checkpoints of the same training run. 
- Now we want to parallelize with respect to different subsets of the dataset, for the same checkpoint (a single epoch). To do this, I created eval_from_wandb_wrapper_single_epoch.py, which starts subprocesses for different subsets of the data. The subset splitting is achieved by the following lines in eval_from_wandb_utils.evaluate:

```
# additional_dataloader = datamodule.val_dataloader()
    if condition_range: # take only a slice of the 'true' edge conditional set
        # datamodule.datasets[cfg.diffusion.edge_conditional_set] = datamodule.datasets[cfg.diffusion.edge_conditional_set][condition_range[0]:condition_range[1]]
        data_slices={'train': None, 'val': None, 'test': None}
        data_slices[cfg.diffusion.edge_conditional_set] = condition_range
        datamodule.prepare_data(datamodule.datasets, slices=data_slices)
```

(I changed the prepare_data method for the reaction data set to accommodate the slicing)
        
condition_range is defined with the following code in eval_from_wandb_wrapper. The condition_ranges are then passed to the respective subprocesses, that consequently call eval_from_wandb_utils.evaluate. 

```
# Split into multiple processes based on cfg.test.n_conditions
conditions_per_gpu = cfg.test.n_conditions // num_gpus # Give this many to the first cft.test.n_conditions-1 gpus, and the rest to the last gpu
condition_ranges = [(conditions_per_gpu*i, conditions_per_gpu*(i+1)) for i in range(num_gpus)]
condition_ranges[-1] = (conditions_per_gpu*(num_gpus-1), cfg.test.n_conditions) # the actual last index
```    

To evaluate on the full test set, we set cfg.test.n_conditions=4949 (test set size). We also need to set diffusion.edge_conditional_set=test since that's how we specify the conditioning set. 

Another detail is that eval_from_wandb_wrapper_single_epoch.py creates the files where samples will be saved to, before splitting into subprocesses. This is because now we want all subprocesses to write to the same files, but we don't want that they all try to create the file at the same time. This is avoided by creating it in advance, and only appending to that file during generation in the different subprocesses. In eval_from_wandb_wrapper_single_epoch.py:
```
# Create empty files at this stage so that the different processes don't overwrite each other
  f = open(f"samples_epoch{epoch}.txt", "w")
  f.close()
  f = open(f"samples_epoch{epoch}_resorted_{cfg.test.sort_lambda_value}.txt", "w")
  f.close()
```