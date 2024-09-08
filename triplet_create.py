#!/usr/bin/env python
from sklearn.preprocessing import LabelBinarizer
from torch import optim
import copy
from tqdm import tqdm
import numpy as np
from torch import nn
import torch.utils.data as dt
import torch
import pandas as pd


def create_triplet(embeds_path, save_path, train_batch_size, epoch):

    device = torch.device("cuda")
    # 3000 is hardcoded deal with it
    series_data = pd.Series(np.arange(0, 3000))
    series_data = (series_data // 500).astype(int)

    loaded_ast_embeddings = np.load(embeds_path)
    loaded_ast_embeddings = loaded_ast_embeddings.reshape(-1, np.prod(loaded_ast_embeddings.shape[1:]))

    X = loaded_ast_embeddings
    y = np.array(series_data)

    def decode(coded):
        coded = int(coded)
        n = coded % 3000
        p = (coded // 3000) % 3000
        a = (coded // 3000 // 3000) % 3000
        return a,p,n

    def encode(a, p, n):
        return (a * 3000 * 3000) + (p * 3000) + n

    X_, y_ = X, y
    data_xy = tuple([X_, y_])

    ind_list = []

    for data_class in sorted(set(data_xy[1])):

        same_class_idx = np.where((data_xy[1] == data_class))[0]
        diff_class_idx = np.where(data_xy[1] != data_class)[0]

        a = np.random.randint(np.min(same_class_idx), np.max(same_class_idx), int(1.5e6), dtype=np.int64)
        p = np.random.randint(np.min(same_class_idx), np.max(same_class_idx), int(1.5e6), dtype=np.int64)
        n = np.random.randint(np.min(diff_class_idx), np.max(diff_class_idx), int(1.5e6), dtype=np.int64)
        ind_list.append(encode(a, p, n))

    X_train = np.array(list(set(np.concatenate(ind_list))))

    class TripletDataset(dt.Dataset):
        def __init__(self, X_, triplet_indices, device):
            self.X_tensor = torch.from_numpy(X_).to(device).float()
            self.triplet_indices = triplet_indices
            self.device = device

        def __len__(self):
            return len(self.triplet_indices)

        def __getitem__(self, index):
            a, p, n = decode(self.triplet_indices[index])
            return self.X_tensor[a], self.X_tensor[p], self.X_tensor[n]

    class BaseNetwork(nn.Module):
        def __init__(self, input_size):
            super(BaseNetwork, self).__init__()
            self.linear1 = nn.Linear(input_size, 1000)
            self.linear2 = nn.Linear(1000, 500)
            self.linear3 = nn.Linear(500, input_size)
            self.actRelu = nn.LeakyReLU()
            self.drop = nn.Dropout(0.15)

        def forward(self, x):
            out = self.linear1(x)
            out = self.actRelu(out)
            out = self.drop(out)
            out = self.linear2(out)
            out = self.actRelu(out)
            out = self.drop(out)
            out = self.linear3(out)
            return out

    class TripletArchitecture(nn.Module):
        def __init__(self, input_size):
            super(TripletArchitecture, self).__init__()
            self.bn = BaseNetwork(input_size)

        def forward(self, a, p, n):
            a_out = self.bn(a)
            p_out = self.bn(p)
            n_out = self.bn(n)
            return a_out, p_out, n_out

    import math
    triplet_model = TripletArchitecture(X.shape[1]).to(device)

    triplet_optim = optim.Adam(triplet_model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    triplet_criterion = nn.TripletMarginLoss()

    triplet_dataset = TripletDataset(X, X_train, device)

    triplet_loader = dt.DataLoader(triplet_dataset, shuffle=True, batch_size=train_batch_size)

    triplet_model.train()
    triplet_model.to(device)
    best_model = None
    best_loss = 1000
    default_patience = 2
    patience = default_patience

    for i in range(epoch):
        data_iter = tqdm(enumerate(triplet_loader),
                                      desc="EP_%s:%d" % ("test", i),
                                      total=len(triplet_loader),
                                      bar_format="{l_bar}{r_bar}")
        total_loss = 0

        for j, (a_, p_, n_) in data_iter:
            # NEW STUFF START
            a_.require_grad = False
            p_.require_grad = False
            n_.require_grad = False
            # minibatch_size = len(a_)
            # pos_dists = torch.linalg.norm(a_ - p_, dim=1).to("cpu")
            # neg_dists = torch.linalg.norm(a_ - n_, dim=1).to("cpu")
            #
            # valid_triplets_mask = torch.less(pos_dists, neg_dists)
            #
            # pos_mask = torch.zeros(minibatch_size, dtype=torch.bool)
            # pos_mask[[torch.sort(pos_dists).indices.split(math.floor(minibatch_size * 0.3))[0]]] = True
            #
            # neg_mask = torch.zeros(minibatch_size, dtype=torch.bool)
            # neg_mask[[torch.sort(neg_dists).indices.split(math.floor(minibatch_size * 0.3))[0]]] = True
            #
            # indices = torch.arange(minibatch_size)[valid_triplets_mask & pos_mask & neg_mask]
            # input_size = len(indices)

            # a_ = a_[indices]
            # p_ = p_[indices]
            # n_ = n_[indices]
            a_o, p_o, n_o = triplet_model(a_, p_, n_)
            loss = triplet_criterion(a_o, p_o, n_o)
            loss.backward()
            total_loss += loss.item()
            # data_iter.set_postfix({"Loss": total_loss / (j + 1), "Input Size": input_size})
            data_iter.set_postfix({"Loss": total_loss / (j + 1)})
            triplet_optim.step()

        total_loss /= len(triplet_loader)
        print(f'Epoch [{i + 1}/{100}], Loss: {total_loss:.4f}')
        if total_loss < best_loss - 0.0001:
            best_loss = total_loss
            best_model = copy.deepcopy(triplet_model)
            patience = default_patience
        else:
            patience -= 1
        if patience == 0:
            break

    # torch.save(best_model, "Test/test_best_model.pth")

    best_model.eval()
    y_tensor = torch.from_numpy(np.arange(len(y)))
    x_tensor = torch.from_numpy(X).to(device).float()
    triplet_test_loader = dt.DataLoader(dt.TensorDataset(x_tensor, y_tensor), batch_size=256)
    labels = []
    output_embeds = []
    with torch.no_grad():
        for (data_, label_) in tqdm(triplet_test_loader, total=len(triplet_test_loader)):
            labels.append(label_)
            a, _, _ = best_model(data_, data_, data_)
            output_embeds.append(a.cpu().detach().numpy())

    embeds = np.vstack(output_embeds)

    np.save(save_path, embeds)
