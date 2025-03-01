from setuptools import find_packages, setup

setup(
    name="data_preprocessing",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "hydra-core",
        "wandb",
        "numpy",
        "zarr"
    ],
)