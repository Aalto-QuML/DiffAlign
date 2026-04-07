"""Per-platform SLURM configuration."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlatformConfig:
    platform: str
    project: str
    partition_gpu: str
    partition_cpu: str
    module: str
    venv_path: str
    scratch_base: str
    gpu_type: str = ''
    extra_modules: list = field(default_factory=list)
    extra_env: dict = field(default_factory=dict)


PLATFORMS = {
    'puhti': PlatformConfig(
        platform='puhti', project='project_2015607',
        partition_gpu='gpu', partition_cpu='small',
        module='pytorch/2.4',
        venv_path='/projappl/project_2015607/diffalign',
        scratch_base='/scratch/project_2015607',
        gpu_type='v100',
        extra_modules=['gcc/13.2.0'],
        extra_env={'LD_PRELOAD': '/appl/spack/v022/install-tree/gcc-8.5.0/gcc-13.2.0-hgaeyz/lib64/libstdc++.so.6:$LD_PRELOAD'},
    ),
    'mahti': PlatformConfig(
        platform='mahti', project='project_2015607',
        partition_gpu='gpusmall', partition_cpu='small',
        module='pytorch/2.4',
        venv_path='/projappl/project_2015607/diffalign',
        scratch_base='/scratch/project_2015607',
        gpu_type='a100',
        extra_modules=['gcc/13.2.0'],
        extra_env={'LD_PRELOAD': '/appl/spack/v022/install-tree/gcc-8.5.0/gcc-13.2.0-hgaeyz/lib64/libstdc++.so.6:$LD_PRELOAD'},
    ),
    'lumi': PlatformConfig(
        platform='lumi', project='project_462001028',
        partition_gpu='small-g', partition_cpu='small',
        module='pytorch/2.4',
        venv_path='/projappl/project_462001028/multiguide',
        scratch_base='/scratch/project_462001028',
    ),
}


def get_platform_config(platform_name: str, use_gpu: bool = False) -> PlatformConfig:
    if platform_name not in PLATFORMS:
        raise ValueError(f'Platform {platform_name} not supported. Choose from: {list(PLATFORMS.keys())}')
    return PLATFORMS[platform_name]
