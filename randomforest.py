import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from threading import Thread
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from dataset import BPE, TokenVocab, TreeBERTDataset
from torch.utils.data import DataLoader
import torch
import json
import pandas as pd
import numpy as np
import tqdm

labels = open('pretrain_data_code/java_custom_label/bert_output.json', 'r').readlines()

smells = []

for i in labels:
    test_line = json.loads(i)
    smells.append(test_line["smellKey"])

label_series = pd.Series(smells)

# vocab = TokenVocab.load_vocab("data/vocab.large")
# labeled_dataset = TreeBERTDataset(vocab, "pretrain_data_AST_tmp/java_custom_label/bert_output.json_ast", path_num=100,
#                                   node_num=20, code_len=200, is_fine_tune=False, corpus_lines=None)

# batch_size = 32

# labeled_data_loader = DataLoader(labeled_dataset, batch_size=batch_size)


# device = torch.device("cuda:0")
# model = torch.load('data/output_rerun_rerun_extended.ep49')
# model = model.to(device)
# model.eval()

# data_iter = tqdm.tqdm(enumerate(labeled_data_loader), total=len(labeled_data_loader))
#
# code_data_list = []
#
# for i, data in data_iter:
#     item = {key: value.to(device) for key, value in data.items()}
#
#     with torch.no_grad():
#         code_data_list.append(model(item["encoder_input"], item["node_pos_em_coeff"], item["decoder_input"]))


less_than_10 = label_series.value_counts().where(lambda x:x <= 40).dropna().index
to_drop = label_series[label_series.apply(lambda x:x in less_than_10)].index.to_numpy()
mask = np.ones(len(label_series), dtype=bool)
mask[to_drop] = False

ohe = OneHotEncoder(sparse_output=False)
le = LabelEncoder()

label_numpy = ohe.fit_transform(label_series.to_numpy().reshape(-1, 1))
le.fit_transform(label_series.to_numpy())
label_count = len(ohe.categories_[0])


k_split = KFold(n_splits=5, shuffle=True)

# cpu_tensors = [i.cpu() for i in code_data_list]
# flattened = [torch.flatten(i, start_dim=1) for i in cpu_tensors]
# before_drop = np.vstack(flattened)

print("Opening tensor file")
fromfile = np.fromfile("dump.txt")
print("Opened tensor file")

assert fromfile.ndim == 1 and (fromfile.size % len(label_series)) == 0
before_drop = fromfile.reshape(len(label_series), -1)
final_data = before_drop[mask, ...]

final_labels = label_numpy[mask, ...]
encoded_labels = le.fit_transform(ohe.inverse_transform(final_labels).ravel())

k_iterator = k_split.split(final_data, final_labels)

total_score = 0


def thread_func(train_index_, test_index_, results_, i_):
    rf_classifier = RandomForestClassifier(verbose=2)
    rf_classifier.fit(final_data[train_index_], final_labels[train_index_])
    score = rf_classifier.score(final_data[test_index_], final_labels[test_index_])
    results_[i_] = score
    print(f"Score of fold {i_}: {score}")


def thread_func_xgboost(train_index_, test_index_, results_, i_):
    xrf_classifier = xgb.XGBClassifier(verbosity=2, n_estimators=10, n_jobs=4)
    xrf_classifier.fit(final_data[train_index_], encoded_labels[train_index_])
    score = xrf_classifier.score(final_data[test_index_], encoded_labels[test_index_])
    results_[i_] = score
    print(f"Score of fold {i_}: {score}")


thread_list = [None] * 5
results = [None] * 5

train_xgboost = True

for i, (train_index, test_index) in enumerate(k_iterator):
    # thread = Thread(target=thread_func if not train_xgboost else thread_func_xgboost, args=(train_index, test_index, results, i))
    # Can't thread xgboost, too much memory required
    thread_func_xgboost(train_index, test_index, results, i)
    # thread_list[i] = thread
    # thread.start()

# for thread in thread_list:
#     thread.join()

print("Total average:", np.sum(results) / 5)
