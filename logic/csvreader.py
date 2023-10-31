import pandas as pd
from .label import Label


def read_labels(path):
    smells = pd.read_csv(path)
    label_series = smells["smellKey"]
    return Label(label_series)


def read_functions(path):
    smells = pd.read_csv(path)
    functions = smells["function"]
    return functions.tolist()


