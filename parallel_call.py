import itertools
import os

import numpy as np
from logic.utils import set_seed
from concurrent.futures import ProcessPoolExecutor


def call_run(parameters):
    print("Starting run with parameters:", parameters)
    # return Popen(['python', 'parallel.py', parameters["optimizer"], parameters["train_batch_size"], parameters["lr"]], stdout=STDOUT, stderr=STDOUT, shell=True)
    os.system(f'python parallel.py {parameters["optimizer"]} {parameters["train_batch_size"]} {parameters["lr"]}')


if __name__ == "__main__":
    parameters = {
        "optimizer": ["Adam"],
        "train_batch_size": list(range(16, 81, 16)),
        "lr": np.arange(-7, -3, 1),
    }
    s = [np.arange(len(i)) for i in parameters.values()]
    perm = list(itertools.product(*s))
    set_seed()
    print(f"Current seed {np.random.get_state()[1][0]}")

    with ProcessPoolExecutor() as executor:
        for i, p in enumerate(perm):
            current_params = {}
            for index, key in enumerate(parameters.keys()):
                current_params[key] = parameters[key][p[index]]

            executor.submit(call_run, current_params)
