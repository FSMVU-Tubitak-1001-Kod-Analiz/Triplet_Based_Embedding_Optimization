import pandas as pd
import torch.utils.data as data_utils
import torch
import tqdm
from transformers import RobertaModel
from logic.embeds.graphcodebert_impl import Model
from logic.embeds.graphcodebert_impl import build_dataset
from logic import csvreader
import tempfile
from logic.embeds.utils import save_to_file


def build(code_path, file_name, batch_size=64):
    if code_path.endswith('.csv'):
        functions = csvreader.read_functions(code_path, True)
        with tempfile.NamedTemporaryFile("w") as tmp:
            tmp.write(functions)
            dataset = build_dataset(tmp.name)
    else:
        assert code_path.endswith('.json')
        dataset = build_dataset(code_path)

    model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")

    device = "cuda"
    model = Model(model)
    model.to(device)

    embeds = []

    loader = data_utils.DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for step, batch in enumerate(tqdm.tqdm(loader)):
            code_inputs = batch[0].to(device)
            attn_mask = batch[1].to(device)
            position_idx = batch[2].to(device)
            # nl_inputs = batch[3].to(device)

            code_vec = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx)
            embeds.append(code_vec)

    print("Shape of embeddings is", embeds[0][0].shape if batch_size != 1 else embeds[0].shape)

    if file_name is None:
        file_name = "graphcodebert_hidden_state_temp"

    save_to_file(embeds, file_name)


# build("../../data/raw/unique_data_setV3.csv", "test", 64)