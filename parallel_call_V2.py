import itertools
import os
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np

param_keys = ["optimizer", "train_batch_size", "lr"]


def call_run(call_str):
    print(call_str)
    os.system(call_str)


def call_parallel(embeds_paths, label_path, parameters=None):
    if parameters is None:
        parameters = {
            "optimizer": ["SGD", "Adam"],
            "train_batch_size": [32, 64, 128, 256, 512],
            "lr": np.arange(-6, -3, 1)
        }
    else:
        assert all(i in param_keys for i in parameters.keys())

    output_folder = [i for i in os.listdir("results/") if i.startswith("hyperparam_run")]
    output_folder = "results/hyperparam_run" + str(len(output_folder) + 1)
    print(">> At folder", output_folder)

    with ProcessPoolExecutor() as executor:
        for embeds_path in embeds_paths:
            s = [np.arange(len(i)) for i in parameters.values()]
            perm = list(itertools.product(*s))
            for _, p in enumerate(perm):
                current_params = {
                    "embeds_path": embeds_path,
                    "label_path": label_path,
                    "smell_range": ",".join(str(i) for i in (0, 7)),
                    "shuffle": 1,
                    "num_epochs": 2000,
                    "patience": 500
                }
                for index, key in enumerate(parameters.keys()):
                    current_params[key] = parameters[key][p[index]]

                call_str = (f"python run_test.py -ep {embeds_path} -lp {label_path} -op {current_params['optimizer']} "
                            f"-lr {current_params['lr']} -bs {current_params['train_batch_size']} "
                            f"-sr {current_params['smell_range']} -sh {current_params['shuffle']} "
                            f"-e {current_params['num_epochs']} -pt {current_params['patience']} "
                            f"-o {output_folder}")
                executor.submit(call_run, call_str)


def call_siamese_triplets():
    embeds_folders = """/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_21_01_7500_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_21_43_7500_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_22_31_7500_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_22_45_7500_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_23_51_7500_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_29_31_7500_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_50_04_7500_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_53_48_7500_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_53_59_7500_smells_bert_nli_mean_token_pooler_output""".split("\n")

    embed_type = "test"
    embed_name = embed_type + "_embeds_triplet_output.npy"
    label_path = f"/home/user/PycharmProjects/Model_Scratch/data/raw/7500_smells_{embed_type}.json"

    embeds_paths = []
    for i in embeds_folders:
        embeds_paths.append(os.path.join(i, embed_name))

    parameters = {
        "optimizer": ["Adam"],
        "train_batch_size": [256],
        "lr": [-4]
    }

    call_parallel(embeds_paths, label_path, parameters=parameters)


if __name__ == '__main__':
    call_siamese_triplets()
