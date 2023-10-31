import more_itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def peek(loader):
    gen = more_itertools.peekable(loader)
    peek_ = gen.peek()
    return peek_


bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}"


def label_wise_accuracy(output, target):
    pred = (output > 0.5).astype(np.float32)
    correct = (pred == target).astype(np.float32)
    label_accuracy = np.mean(correct, axis=0)
    return label_accuracy


def predict(model, X, y: np.ndarray, is_multi_label):
    assert issubclass(X.__class__, DataLoader)
    assert issubclass(y.__class__, np.ndarray)
    predicted_list = []
    iterator = tqdm(enumerate(X), total=len(X))
    model.eval()
    with torch.no_grad():
        for i, data_ in iterator:
            code_data_1d = torch.flatten(data_, start_dim=1) if data_.ndim > 2 else data_
            result = model(code_data_1d)
            predicted_list.append(result.detach().cpu().numpy())

    predicted_array = np.vstack(predicted_list)
    predicted = np.argmax(predicted_array, axis=1)
    target = None

    if y.ndim > 1:
        assert y.ndim == 2
        target = np.argmax(y, axis=1)

    if is_multi_label:
        assert y.ndim == 2
        accuracy = label_wise_accuracy(predicted_array, y)
    else:
        accuracy = np.sum(target == predicted) / len(y)

    print("Accuracy is:", accuracy)
    return predicted_array


def append_to_path(relative_path):
    import os
    import sys

    tb_path = os.path.realpath(relative_path)
    sys.path.append(tb_path)
