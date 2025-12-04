from setuptools import setup, find_packages

setup(
    name='diffalign',
    version='1.0.0',
    url=None,
    author='N.L',
    author_email='najwa.laabid@aalto.fi',
    description='Diffusion for retrosynthesis',
    packages=find_packages(),
    install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1']
)