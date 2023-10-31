import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import copy
import tqdm
import math

from logic import Label, peek, MultiLabelClassifier, bar_format


class ValidationModel:
    def __init__(self, label, dataset: data.Dataset, batch_size, device=None, val_ratio=0.1):
        # labels
        assert isinstance(label, Label)
        self.label: Label = label
        self.val_ratio = val_ratio

        # data
        self.batch_size = batch_size
        self.dataset = dataset

        data_len = len(self.label.labels)
        val_len = math.floor(data_len * val_ratio)

        indices = list(range(data_len))
        val_indices = np.random.choice(indices, size=val_len, replace=False)
        train_indices = list(set(indices) - set(val_indices))

        self.train_sampler = data.SubsetRandomSampler(train_indices)
        self.val_sampler = data.SubsetRandomSampler(val_indices)

        self.train_loader = data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        self.val_loader = data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.val_sampler)

        # device
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # model
        self.classifier = None
        self.best_model = None

    def __initialize_multi_label__(self):
        input_shape = peek(self.val_loader).shape

        input_size_ = np.prod(input_shape[1:])  # Multiply shape values after batch size to get input size
        output_size_ = self.label.label_count  # Number of classes

        self.classifier = MultiLabelClassifier(input_size_, output_size_)
        self.classifier.to(self.device)
        self.classifier.train()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(self.classifier.parameters(), lr=1e-3)

        return criterion, optimizer

    # Define the model
    def train_multi_label(self):
        criterion, optimizer = self.__initialize_multi_label__()

        min_loss = 100
        num_epochs = 100

        for epoch in range(num_epochs):

            for phase in ["train", "val"]:
                optimizer.zero_grad()

                if phase == "train":
                    self.classifier.train()
                else:
                    self.classifier.eval()

                with torch.no_grad() if phase == "val" else torch.enable_grad():

                    total_loss = 0
                    accuracy = 0
                    count = 0

                    loader = self.train_loader if phase == 'train' else self.val_loader

                    data_iter = tqdm.tqdm(enumerate(loader),
                                          desc="Epoch %s:%d" % (phase, epoch),
                                          total=len(loader),
                                          bar_format=bar_format)

                    for i, data_ in data_iter:
                        code_data = data_

                        batch_interval = slice(len(data_) * i, len(data_) * (i + 1))
                        label_indices = self.train_sampler.indices[batch_interval] if phase == "train" else self.val_sampler.indices[batch_interval]

                        target = torch.tensor(self.label.labels[label_indices], dtype=torch.float32, device=self.device)

                        code_data_1d = torch.flatten(code_data, start_dim=1) if code_data.ndim > 2 else code_data

                        outputs = self.classifier(code_data_1d)  # Forward pass 32x768 32x3
                        loss = criterion(outputs, target)  # Calculate the loss
                        count += len(data_)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                        total_loss += loss.item()


                        data_iter.set_postfix_str(f"Loss: {total_loss / count :.4f}")

                    total_loss = total_loss / len(loader)

                    accuracy /= len(loader)

                    if phase == "train" and min_loss > total_loss:
                        min_loss = total_loss
                        self.best_model = copy.deepcopy(self.classifier)

        print("Training finished!")
        return self.best_model