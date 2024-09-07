import itertools
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait
import copy

def call_run(parameters):
    print("Starting run with parameters:", parameters)
    os.system(
        f'python parallel.py {parameters["embeds_path"]} {parameters["optimizer"]} {parameters["train_batch_size"]} {parameters["lr"]}')


def multi_embeds_parallel():
    embeds = [
        "/home/user/Desktop/Triplet-net-keras/Test/embeds_graphcodebert_3000_1.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/3000_smells_graphcodebert_hidden_state.npy",
    ]
    parameters = {
        "optimizer": "Adam",
        "train_batch_size": 512,
        "lr": -4,
    }
    with ProcessPoolExecutor() as executor:
        for i, p in enumerate(embeds):
            current_params = copy.copy(parameters)
            current_params["embeds_path"] = p

            executor.submit(call_run, current_params)


def hyper_parallel_hyper():
    parameters = {
        "optimizer": ["SGD"],
        "train_batch_size": list(range(32, 81, 16)),
        "lr": np.arange(-5, -2, 1)
    }
    s = [np.arange(len(i)) for i in parameters.values()]
    perm = list(itertools.product(*s))

    for i in range(0, len(perm), 7):
        curr_perms = perm[i * 7: (i + 1) * 7]
        futures = []
        with ProcessPoolExecutor() as executor:
            for i, p in enumerate(curr_perms):
                current_params = {
                    "embeds_path": "data/3000_smells_graphcodebert_hidden_state.npy"
                }
                for index, key in enumerate(parameters.keys()):
                    current_params[key] = parameters[key][p[index]]

                futures.append(executor.submit(call_run, current_params))
        wait(futures)


def hyper_parallel():
    # parameters = {
    #     "optimizer": ["SGD", "Adam"],
    #     "train_batch_size": list(range(32, 81, 16)),
    #     "lr": np.arange(-5, -4, 1)
    # }
    parameters = {
        "optimizer": ["Adam"],
        "train_batch_size": [128, 256, 512],
        "lr": [-4]
    }
    s = [np.arange(len(i)) for i in parameters.values()]
    perm = list(itertools.product(*s))
    with ProcessPoolExecutor() as executor:
        for i, p in enumerate(perm):
            current_params = {
                "embeds_path": "data/3000_smells_graphcodebert_hidden_state.npy"
            }
            for index, key in enumerate(parameters.keys()):
                current_params[key] = parameters[key][p[index]]

            executor.submit(call_run, current_params)


if __name__ == "__main__":
    multi_embeds_parallel()
