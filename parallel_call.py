import itertools
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, wait, as_completed
import copy
from logic.utils import save_annotation
import contextlib


def call_run(parameters):
    print("Starting run with parameters:", parameters)
    os.system(
        f'python parallel.py {parameters["embeds_path"]} {parameters["optimizer"]} '
        f'{parameters["train_batch_size"]} {parameters["lr"]} {parameters["label_path"]}')


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


def hyper_parallel_hyper(embeds_path, label_path):
    parameters = {
        "optimizer": ["SGD", "Adam"],
        "train_batch_size": [32, 64, 128, 256, 512],
        "lr": np.arange(-6, -3, 1)
    }
    s = [np.arange(len(i)) for i in parameters.values()]
    perm = list(itertools.product(*s))

    for i in range(0, len(perm), 7):
        curr_perms = perm[i * 7: (i + 1) * 7]
        futures = []
        with ProcessPoolExecutor() as executor:
            for _, p in enumerate(curr_perms):
                current_params = {
                    "embeds_path": embeds_path,
                    "label_path": label_path
                }
                for index, key in enumerate(parameters.keys()):
                    current_params[key] = parameters[key][p[index]]

                futures.append(executor.submit(call_run, current_params))
        wait(futures)
    save_annotation("hyperparam_run_" + os.path.basename(embeds_path), "Finished parallel hyperparam run for " + embeds_path)


def hyper_parallel(embeds_path, label_path):
    parameters = {
        "optimizer": ["SGD", "Adam"],
        "train_batch_size": [32, 64, 128, 256, 512],
        "lr": np.arange(-6, -3, 1)
    }
    s = [np.arange(len(i)) for i in parameters.values()]
    perm = list(itertools.product(*s))

    futures = []
    with ProcessPoolExecutor() as executor:
        for _, p in enumerate(perm):
            current_params = {
                "embeds_path": embeds_path,
                "label_path": label_path
            }
            for index, key in enumerate(parameters.keys()):
                current_params[key] = parameters[key][p[index]]

            futures.append(executor.submit(call_run, current_params))

    wait(futures)
    save_annotation("hyperparam_run_" + os.path.basename(embeds_path), "Finished parallel hyperparam run for " + embeds_path)


if __name__ == "__main__":
    embeds_paths = """data/1500_smells_bert_nli_mean_token_pooler_output.npy
            data/1500_smells_codebert_pooler_output.npy
            data/1500_smells_graphcodebert_pooler_output.npy""".split("\n")
    graphcodebert_embeds = "data/1500_smells_graphcodebert_hidden_state.npy"

    label_path_ = "data/raw/7500_smells_test.json"

    for i in embeds_paths:
        print(i)
        save_annotation("hyperparam_run_" + os.path.basename(i), "Starting parallel hyperparam run for " + i)
        hyper_parallel(i, label_path_)
    save_annotation("hyperparam_run_" + os.path.basename(graphcodebert_embeds), "Starting parallel hyperparam run for " + graphcodebert_embeds)
    hyper_parallel_hyper(graphcodebert_embeds, label_path_)