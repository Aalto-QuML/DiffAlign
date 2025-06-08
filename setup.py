from setuptools import setup, find_namespace_packages
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

setup(
    name="diffalign",
    version="0.1.0",
    packages=find_namespace_packages(include=['diffalign*']),
    package_dir={"": "."}
)