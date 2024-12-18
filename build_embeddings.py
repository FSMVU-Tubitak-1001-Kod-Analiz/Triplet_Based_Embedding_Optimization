import numpy as np
from tqdm import tqdm

import logic.embeds as embeds
import pathlib
import code_utils as cu


def build_embeddings(data_path):
    codebert_file_name = f"{pathlib.Path(data_path).stem}_codebert_pooler_output.npy"
    embeds.build_codebert(data_path, codebert_file_name)

    graphcodebert_pooler_file_name = f"{pathlib.Path(data_path).stem}_graphcodebert_pooler_output.npy"
    embeds.build_graphcodebert(data_path, graphcodebert_pooler_file_name)

    graphcodebert_hidden_state_file_name = f"{pathlib.Path(data_path).stem}_graphcodebert_hidden_state.npy"
    embeds.build_graphcodebert(data_path, graphcodebert_hidden_state_file_name)

    bert_file_name = f"{pathlib.Path(data_path).stem}_bert_nli_mean_token_pooler_output.npy"
    embeds.build_bert_nli_mean(data_path, bert_file_name)


def separate_train_test_embeddings(embeds, train_path, test_path):
    train_df = cu.open_smell_file(train_path)

    test_df = cu.open_smell_file(test_path)
    for embed in tqdm(embeds):
        embed_path = pathlib.Path(embed)
        embed_content = np.load(embed, "r")

        train_file_name = f"{str(embed_path.parent)}/{pathlib.Path(embed).stem}_train.npy"
        test_file_name = f"{str(embed_path.parent)}/{pathlib.Path(embed).stem}_test.npy"

        with open(train_file_name, "wb") as train_file:
            np.save(train_file, embed_content[train_df["index"]])

        with open(test_file_name, "wb") as test_file:
            np.save(test_file, embed_content[test_df["index"]])


if __name__ == '__main__':
    separate_train_test_embeddings("""/home/user/PycharmProjects/Model_Scratch/data/9000_smells_bert_nli_mean_token_pooler_output.npy
/home/user/PycharmProjects/Model_Scratch/data/9000_smells_codebert_pooler_output.npy
/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_hidden_state.npy
/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_pooler_output.npy""".split("\n"),
                                   "data/raw/9000_smells_train.json",
                                   "data/raw/9000_smells_test.json"
                                   )