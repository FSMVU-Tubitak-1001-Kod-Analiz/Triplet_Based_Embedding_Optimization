import json

import pandas as pd
from .label import Label


def read_labels(path):
    smells = pd.read_csv(path)
    label_series = smells["smellKey"]
    return Label(label_series)


def read_functions(path, to_json=False):
    if path.endswith('.csv'):
        smells = pd.read_csv(path)
        functions = smells["function"]
        if not to_json:  # return list
            return functions.tolist()
        else:  # return df
            return functions.to_json(orient="records", lines=True)

    else:
        assert path.endswith('.json')
        labels_file = open(path, 'r').readlines()
        functions = []
        for i in labels_file:
            test_line = json.loads(i)
            functions.append(test_line["smellKey"])

        return functions
