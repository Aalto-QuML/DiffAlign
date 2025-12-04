import os
import sys
import multiprocessing
import wandb
import yaml
from omegaconf import OmegaConf
import traceback
import pathlib
parent_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
os.environ["WANDB_WATCH"] = "false"
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# log.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# log.addHandler(handler)
# Add your custom handler
handler = logging.StreamHandler()
log.addHandler(handler)
# Optionally, set a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from src.utils import wandb_utils

# os.environ['WANDB_SERVICE_WAIT'] = 1000

# 1. Access the run using the run path
def get_run(run_path):
    api = wandb.Api()
    run = api.run(run_path)
    return run

def download_shared_files(run, run_id):
    # Downloads the run config (shared between processes) and creates the directory structure
    savedir = os.path.join("experiments", "trained_models", run.name + run_id)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    config_path = os.path.join(savedir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(run.config, f)
    return savedir

def main_subprocess(queue, project, entity, run_id, save_models_dir, epoch, gpu_id, cfg, save_results_dir):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # log.info("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    # The following are imported here to make sure that the GPU binding for torch is correct (after os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id))
    import torch # Must be done after setting CUDA_VISIBLE_DEVICES
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(gpu_id)
    a = torch.tensor([1.], device='cuda')
    log.info(f"Device: {device}. Available GPUs: {torch.cuda.device_count()}.")
    from src.eval_from_wandb_utils import create_model_from_config, evaluate, load_weights, download_files

    # Log the number of GPUs and other details for debugging purposes
    log.info("\n" \
     + "Evaluating epoch " + str(epoch)  + " on GPU " + str(gpu_id) + "\n"\
     + '__Python VERSION:' + str(sys.version) + "\n"\
     + '__pyTorch VERSION:' + str(torch.__version__) + "\n"\
     + '__CUDNN VERSION:' + str(torch.backends.cudnn.version()) + "\n"\
     + '__Number CUDA Devices:' + str(torch.cuda.device_count()) + "\n"\
     + '__Devices'+ "\n"
     + 'Active CUDA Device: GPU' + str(torch.cuda.current_device()) + "\n")

    try:
        run_path = f"{entity}/{project}/{run_id}"
        run = get_run(run_path)
        download_files(save_models_dir, run, epoch)

        model, datamodule, data_class, dataset_infos = create_model_from_config(cfg, device)
        checkpoint_path = save_models_dir + f'/eval_epoch{epoch}.pt'
        log.info(f"Checkpoint path: {checkpoint_path}")
        model = load_weights(model, checkpoint_path, device)

        # Change the working directory so that samples get saved to the correct place, as they would be with Hydra
        os.chdir(save_results_dir)

        scores = evaluate(cfg, data_class, datamodule, epoch, model, device)

        for key, value in scores.items():
            if isinstance(value, torch.Tensor):
                scores[key] = value.detach().cpu().numpy()
        log.info(scores)
        
        queue.put({epoch: scores})

    except Exception as e:
        log.info(f"Exception occurred for epoch {epoch} on GPU {gpu_id}: {e}")
        traceback.print_exc() 

def main():
    # Parse arguments
    run_id = sys.argv[1] # e.g., c64gs0eh
    epochs = sys.argv[2] # e.g., "19,39,59,..."
    # Get the number of gpus & their ids
    num_gpus = int(sys.argv[3])
    gpu_ids = list(range(num_gpus))
    epochs = [int(epoch) for epoch in epochs.split(",")]
    # assert len(epochs) <= num_gpus, "Number of epochs must be less than the number of GPUs."
    cli_overrides = OmegaConf.from_cli(sys.argv[4:])

    # Download the files shared between processes and set up directory structure
    project = "retrodiffuser"
    entity = "najwalb"
    run_path = f"{entity}/{project}/{run_id}"
    run = get_run(run_path)
    savedir = download_shared_files(run, run_id) # Download the config file and create the directory structure
    orig_dir = os.getcwd() # This was previously used when doing multiple checkpoints sequentially, to undo the effect of os.chdir(modified_cfg_dir)

    # The default Hydra config
    # This context manager ensures that we're working with a clean slate (Hydra's global state is reset upon exit)
    log.info("Current working dir: " + os.getcwd())
    with initialize(config_path="../configs"):
        # Compose the configuration using the default config name
        default_cfg = compose(config_name="default")
        OmegaConf.set_struct(default_cfg, False) # Allow adding new fields (the experiment files sometimes have incorrectly added new stuff and not updated the default)
    # The context is closed, and GlobalHydra is cleared, ensuring there are no lingering Hydra states
    GlobalHydra.instance().clear()
    
    config_path = os.path.join(os.getcwd(), savedir, 'config.yaml')
    # Default config
    base_config = OmegaConf.load(config_path)
    # Override based on the config used for the run
    cfg = OmegaConf.merge(default_cfg, base_config)
    # Override based on the command line arguments
    cfg = OmegaConf.merge(cfg, cli_overrides)
    cfg.diffusion.classifier_free_guidance_weight = float(cfg.diffusion.classifier_free_guidance_weight)
    cfg.diffusion.edge_conditional_set = 'val'
    # save the modified config to the artifact dir
    key = wandb_utils.eval_key(cfg, new_key=True)
    # key = f'eval_noduplicates_ncond_{cfg.diffusion.edge_conditional_set}_{cfg.test.n_conditions}'\
    #     + (f"_clfgw_{cfg.diffusion.classifier_free_guidance_weight}" if cfg.diffusion.classifier_free_guidance_weight != 0.1 else "")\
    #     + (f"_lambdatest_{cfg.diffusion.lambda_test}" if cfg.diffusion.lambda_test != 1 else "") \
    #     + (f"_elborep_{cfg.test.repeat_elbo}") + '/'
    save_results_dir = os.path.join(os.getcwd(), savedir, key)
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    dir_for_wandb = os.path.join(savedir, key)
    log.info(f"Dir for wandb: {dir_for_wandb}")
    modified_cfg_file = os.path.join(save_results_dir, 'modified_config.yaml')
    OmegaConf.save(cfg, modified_cfg_file)

    # Create processes to run the training on each GPU
    q = multiprocessing.Queue() # To aggregate the results in the end
    num_concurrent_runs = len(epochs) // num_gpus + (1 if len(epochs) % num_gpus != 0 else 0)
    for cr in range(num_concurrent_runs):
        processes = []
        for i in range(num_gpus):
            if cr * num_gpus + i >= len(epochs):
                break
            epoch = epochs[cr * num_gpus + i]
            p = multiprocessing.Process(target=main_subprocess, args=(q, project, entity, run_id, savedir, epoch, gpu_ids[i], cfg, save_results_dir))
            p.start()
            processes.append(p)
        # Wait for all processes to complete
        for p in processes:
            p.join()

    # Get the results from the queue
    results = {}
    while not q.empty():
        results.update(q.get())

    log.info(results)

    epochs_that_worked = list(results.keys())
    epochs_that_worked.sort() # This to make sure that epochs are logged in an ascending order (in case the epoch numbers are not usable due to wandb bug)

    with wandb.init(id=run_id, project=project, entity=entity, resume="allow") as run:
        # Move back to the original directory so that wandb.save works properly
        wandb.save(os.path.join(dir_for_wandb, "modified_config.yaml"))
        for i, epoch in enumerate(epochs_that_worked):
            # os.chdir(orig_dir)
            # wandb.log({key:results[epoch]})
            # wandb.log({key: })
            if os.path.exists(os.path.join(dir_for_wandb, f'samples_epoch{epoch}.txt')): # This only saves the non-resorted samples
                wandb.save(os.path.join(dir_for_wandb, f'samples_epoch{epoch}.txt'))
            # wandb.log(metrics)
# TO RUN:
# python src/eval_from_wandb_wrapper.py [RUN_ID] [CHECKPOINT_EPOCHS_COMMA_SEPARATED] [NB_OF_GPUS] [HYDRA_OVERRIDES] 
# NOTE: Only have a single script running for a single run at a time, otherwise wandb will drop some of the results
# e.g., python src/eval_from_wandb_wrapper.py c64gs0eh 19,39,59 0 test.n_conditions=128

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # without this, we get often into problems with CUDA and multiprocessing on Unix-systems. 
    main()
