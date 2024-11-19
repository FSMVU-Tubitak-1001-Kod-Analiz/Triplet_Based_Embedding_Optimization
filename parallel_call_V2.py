import itertools
import os
from asyncio import as_completed
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np

param_keys = ["optimizer", "train_batch_size", "lr"]


def call_run(call_str):
    print(call_str)
    os.system(call_str)


def call_parallel(embeds_paths, label_path, parameters=None, max_workers=None, non_perm_parameters=None, wait_end=True):
    if non_perm_parameters is None:
        non_perm_parameters = {}

    if parameters is None:
        parameters = {
            "optimizer": ["SGD", "Adam"],
            "train_batch_size": [32, 64, 128, 256, 512],
            "lr": np.arange(-6, -3, 1)
        }
    else:
        assert all(i in parameters.keys() for i in param_keys)

    if "output_folder" not in non_perm_parameters.keys():
        output_folder = [i for i in os.listdir("results/") if i.startswith("hyperparam_run")]
        output_folder = "results/hyperparam_run" + str(len(output_folder) + 1)
    else:
        output_folder = non_perm_parameters["output_folder"]
    print(">> At folder", output_folder)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
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
                    "patience": 2000
                }
                current_params = current_params | non_perm_parameters
                for index, key in enumerate(parameters.keys()):
                    current_params[key] = parameters[key][p[index]]

                call_str = (f"python run_test.py -ep {embeds_path} -lp {label_path} -op {current_params['optimizer']} "
                            f"-lr {current_params['lr']} -bs {current_params['train_batch_size']} "
                            f"-sr {current_params['smell_range']} -sh {current_params['shuffle']} "
                            f"-e {current_params['num_epochs']} -pt {current_params['patience']} "
                            f"-o {output_folder}")
                futures.append(executor.submit(call_run, call_str))
        if wait_end:
            wait(futures)
        else:
            executor.shutdown(False, cancel_futures=False)


def call_siamese_triplets():
    embeds_folders = """/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_21_01_7500_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_21_43_7500_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_22_31_7500_smells_bert_nli_mean_token_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_22_45_7500_smells_graphcodebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_23_51_7500_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_29_31_7500_smells_codebert_pooler_output
/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_50_04_7500_smells_graphcodebert_pooler_output
|/home/eislamoglu/PycharmProjects/siamese-triplet/results/13/triplet_2024_11_16__21_53_48_7500_smells_codebert_pooler_output
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
        "lr": [-4],
    }

    call_parallel(embeds_paths, label_path, parameters=parameters)


def call_offline():
    java_embeds = """/home/user/Desktop/Triplet-net-keras/Test/revision/java_bert_nli_mean_token_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision/java_codebert_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision/java_graphcodebert_pooler_output.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_bert_nli_mean_token_pooler_output_java.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_codebert_pooler_output_java.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_pooler_output_java.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_hidden_state_java.npy""".split("\n")

    php_embeds = """/home/user/Desktop/Triplet-net-keras/Test/revision/php_bert_nli_mean_token_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision/php_codebert_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision/php_graphcodebert_pooler_output.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_bert_nli_mean_token_pooler_output_php.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_codebert_pooler_output_php.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_pooler_output_php.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_hidden_state_php.npy""".split("\n")

    py_embeds = """/home/user/Desktop/Triplet-net-keras/Test/revision/py_bert_nli_mean_token_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision/py_codebert_pooler_output.npy
/home/user/Desktop/Triplet-net-keras/Test/revision/py_graphcodebert_pooler_output.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_bert_nli_mean_token_pooler_output_py.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_codebert_pooler_output_py.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_pooler_output_py.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_hidden_state_py.npy""".split("\n")

    gb_java = """/home/user/Desktop/Triplet-net-keras/Test/revision/java_graphcodebert_hidden_state.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_hidden_state_java.npy""".split("\n")

    gb_php = """/home/user/Desktop/Triplet-net-keras/Test/revision/php_graphcodebert_hidden_state.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_hidden_state_php.npy""".split("\n")

    gb_py = """/home/user/Desktop/Triplet-net-keras/Test/revision/py_graphcodebert_hidden_state.npy
/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_hidden_state_py.npy""".split("\n")

#     java_embeds = """/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_bert_nli_mean_token_pooler_output_java.npy
# /home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_codebert_pooler_output_java.npy
# /home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_pooler_output_java.npy
# /home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_hidden_state_java.npy""".split("\n")
#
#     php_embeds = """/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_bert_nli_mean_token_pooler_output_php.npy
# /home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_codebert_pooler_output_php.npy
# /home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_pooler_output_php.npy
# /home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_hidden_state_php.npy""".split("\n")
#
#     py_embeds = """/home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_bert_nli_mean_token_pooler_output_py.npy
# /home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_codebert_pooler_output_py.npy
# /home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_pooler_output_py.npy
# /home/user/PycharmProjects/Model_Scratch/data/revision/500_smells_graphcodebert_hidden_state_py.npy""".split("\n")

    parameters = {
        "optimizer": ["Adam"],
        "train_batch_size": [256],
        "lr": [-4],
    }

    non_perm_parameters = {
        "output_folder": "results/revision/v2"
    }

    java_labels_path = "data/raw/7500_smells_test_java.json"
    call_parallel(gb_java, java_labels_path, parameters=parameters, non_perm_parameters=non_perm_parameters, wait_end=False)

    php_labels_path = "data/raw/7500_smells_test_php.json"
    call_parallel(gb_php, php_labels_path, parameters=parameters, non_perm_parameters=non_perm_parameters, wait_end=False)

    py_labels_path = "data/raw/7500_smells_test_py.json"
    call_parallel(gb_py, py_labels_path, parameters=parameters, non_perm_parameters=non_perm_parameters, wait_end=False)


def call_combined():
    original_embeds = [
        "/home/user/PycharmProjects/Model_Scratch/data/7500_smells_bert_nli_mean_token_pooler_output.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/7500_smells_codebert_pooler_output.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/7500_smells_graphcodebert_pooler_output.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/7500_smells_graphcodebert_hidden_state.npy"
    ]
    triplet_embeds = [
        "/home/user/Desktop/Triplet-net-keras/Test/embeds_nli_pooler_1500_1.npy",
        "/home/user/Desktop/Triplet-net-keras/Test/embeds_codebert_pooler_1500_1.npy",
        "/home/user/Desktop/Triplet-net-keras/Test/embeds_graphcodebert_pooler_1500_1.npy",
        "/home/user/Desktop/Triplet-net-keras/Test/embeds_graphcodebert_1500_1.npy"
    ]

    parameters = {
        "optimizer": ["Adam"],
        "train_batch_size": [256],
        "lr": [-4],
    }

    non_perm_parameters = {
        "output_folder": "results/revision/v4"
    }

    label_path = "data/raw/7500_smells_test.json"
    call_parallel(original_embeds, label_path, parameters, non_perm_parameters=non_perm_parameters, wait_end=False)
    call_parallel(triplet_embeds, label_path, parameters, non_perm_parameters=non_perm_parameters, wait_end=False)


if __name__ == '__main__':
    call_combined()
