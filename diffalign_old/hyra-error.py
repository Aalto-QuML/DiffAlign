'''
    Sampling from a trained model.
'''
from omegaconf import DictConfig
import hydra

@hydra.main(version_base='1.1', config_path='../configs', config_name=f'default')
def main(cfg: DictConfig):
    print(f'epochs {cfg.general.wandb.checkpoint_epochs}\n')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("main crashed. Error: %s", e)