# Generative Adversarial Networks on OASIS dataset

## Overview

This Generative Adversarial Network's (GANs) application is to generate realistic like brain scans using the the Preprocessed OASIS dataset. This project is to explore the capabilities of GANs and working with High Performance Computing (HPC) Clusters, mainly UQ's Rangpur High Performance Computing Cluster and it's SLURM queuing system.

The preprocessed OASIS dataset was given, examples are attached below.

## Table of Contents

- [Generative Adversarial Networks on OASIS dataset](#generative-adversarial-networks-on-oasis-dataset)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [File Structure](#file-structure)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

## File Structure

Folder contains the following files:

  - `train.py`: Main file to train the model.
  - 'gan.py': Contains the GAN model.
  - 'dataset.py': Contains the dataset class.



## Installation

1. Download preprocessed OASIS dataset from [here](https://www.oasis-brains.org/).
2. Clone the repository.
3. Install the required packages.
4. Run the code.


## Requirements

- Anaconda3 was used for it's intra-library compatibility and ease of use. The following packages are required to run the project:

| Package | Version |
| --- | --- |
|pytorch | 2.0.1 |
|torchvision | 0.15.2 |
|tqdm | 4.66.5 |
|numpy | 1.25.2 |
|matplotlib | 3.8.0 |
|pillow | 9.4.0 |

- Note: An NVIDIA GPU is advise to train the model.The model was trained on a NVIDIA A100 GPU with 40GB of memory and 64GB of RAM.

## Usage

Note: The following commands are to be run on the Rangpur HPC Cluster but can be modified to run on any other HPC Cluster. or local machine.

Netowrk is trained from scratch, no pre-trained model is used. To train the model, run the following command:

```
> python train.py
```

## Contributing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
