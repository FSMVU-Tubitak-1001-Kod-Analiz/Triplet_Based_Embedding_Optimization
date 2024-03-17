import numpy as np
import torch
import os


def save_to_file(embeds: list, file_name, save_path=None):
    embeds_tensor = torch.vstack([i["pooler_output"].detach().cpu() for i in embeds])
    embeds_numpy = embeds_tensor.cpu().detach().numpy()
    if save_path is None:
        parent = 0
        found = False
        curdir = os.curdir
        while os.path.abspath(curdir) != '/':
            if os.path.exists(check_path := (os.path.join(curdir, "data"))) and os.path.isdir(check_path):
                curdir = check_path
                found = True
                break
            else:
                curdir = os.path.join(os.path.pardir, curdir)
        if found:
            save_path = curdir
        else:
            raise FileNotFoundError("Couldn't locate folder data")

    np.save(open(os.path.join(save_path, file_name), "wb"), embeds_numpy)