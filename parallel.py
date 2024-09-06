import argparse

import torch
import os
import numpy as np
import sys
from logic.utils import set_seed
import run
import itertools
import run
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output

import sys

if __name__ == "__main__":

    arguments = sys.argv

    device = torch.device("cuda")
    log_path = os.path.realpath("../")
    sys.path.append(log_path)

    parameters = {
        "optimizer": arguments[1],
        "train_batch_size": int(arguments[2]),
        "lr": np.power(10., int(arguments[3])),
    }

    set_seed()
    print(f"Current seed {np.random.get_state()[1][0]}")
    print(">> Starting hyperparameter tuning run")

    embeds_path = "data/3000_smells_graphcodebert_pooler_output.npy"
    label_path = "data/raw/3000_Smell.json"

    current_params = {
        "seed": 42,
        "num_epochs": 2000,
        "patience": 300
    }

    for index, key in enumerate(parameters.keys()):
        current_params[key] = parameters[key]

    print(current_params)
    runner = run.Runner(label_path, embeds_path, 0.2, device=device, params=current_params, output_folder="results/hyperparam_run15")
    runner.run((0, 6), shuffle=True)
    clear_output(wait=True)