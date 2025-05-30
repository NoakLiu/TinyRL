from setuptools import setup, find_packages

setup(
    name="tinyrl",
    version="0.1.0",
    description="Flash-Attn, Linear-Attn Integrated RL framework",
    author="TinyRL Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "gymnasium>=0.28.0",
        "flash-attn>=2.0.0",
        "transformers>=4.20.0",
        "wandb>=0.13.0",
        "tensorboard>=2.10.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "hydra-core>=1.2.0",
        "omegaconf>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
        ],
        "atari": [
            "ale-py>=0.8.0",
            "gymnasium[atari]>=0.28.0",
        ],
        "mujoco": [
            "mujoco>=2.3.0",
            "gymnasium[mujoco]>=0.28.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 