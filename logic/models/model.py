import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import copy
import tqdm
import math

from logic import Label, peek, SimpleClassifier, MultiLabelClassifier


# noinspection PyTypeChecker
class Model:
    def __init__(self, label, dataset: data.Dataset, batch_size, device=None):
        # labels
        assert isinstance(label, Label)
        self.label: Label = label

        # data
        self.batch_size = batch_size
        self.dataset = dataset
        self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size)

        # device
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # model
        self.classifier = None
        self.best_model = None

        self.history = None

    def __initialize__(self, multi_label=False):
        shape = peek(self.loader).shape

        input_size_ = np.prod(shape[1:])  # Multiply shape values after batch size to get input size
        output_size_ = self.label.label_count  # Number of classes

        if not multi_label:
            self.classifier = SimpleClassifier(input_size_, output_size_)
        else:
            self.classifier = MultiLabelClassifier(input_size_, output_size_)

        self.classifier.to(self.device)
        self.classifier.train()

        if not multi_label:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.SGD(self.classifier.parameters(), lr=1e-4)

        return criterion, optimizer

    def train(self):
        criterion, optimizer = self.__initialize__(True)
        min_loss = 100
        num_epochs = 100

        for epoch in range(num_epochs):
            total_loss = 0
            accuracy = 0

            data_iter = tqdm.tqdm(enumerate(self.loader),
                                  desc="EP_%s:%d" % ("test", epoch),
                                  total=len(self.loader),
                                  bar_format="{l_bar}{r_bar}")

            for i, data_ in data_iter:
                code_data = data_
                code_data.requires_grad = False

                batch_interval = slice(len(data_) * i, len(data_) * (i + 1))
                target = torch.tensor(self.label.labels[batch_interval], dtype=torch.long, device=self.device)
                target = torch.argmax(target, dim=1)

                optimizer.zero_grad()

                # FORMAT TO MODEL:
                code_data_1d = torch.flatten(code_data, start_dim=1) if code_data.ndim > 2 else code_data

                # TRAIN
                outputs = self.classifier(code_data_1d)  # Forward pass 32x768 32x3
                loss = criterion(outputs, target)  # Calculate the loss
                loss.backward()  # Backpropagation
                total_loss += loss.item()
                accuracy += (torch.sum(torch.argmax(outputs, dim=1) == target) / len(target)).item()

                optimizer.step()  # Update weights

            total_loss = total_loss / len(self.loader)
            accuracy = accuracy / len(self.loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.5f}')

            if min_loss > total_loss:
                min_loss = total_loss
                self.best_model = copy.deepcopy(self.classifier)
            # PUT EARLY STOP BELOW:

        print("Training finished!")
        return self.best_model

    def train_multi_label(self):
        criterion, optimizer = self.__initialize__(True)

        min_loss = 100
        num_epochs = 100
        prev_loss = 100
        default_patience = 15
        patience = default_patience

        self.history = {
            "losses": [],
            "accuracy": [],
            "epoch_time": [],
            "patience": default_patience,
            "max_epochs": num_epochs,
            "optimizer": str(optimizer),
            "criterion": str(criterion),
            "batch_size": self.batch_size,
            "class_count": self.label.labels.shape[1],
            "sample_size": len(self.dataset),
            "classifier": str(self.classifier),
            "device": str(self.device),
            "dataset": str(self.dataset.__class__),
            "loader": str(self.loader)
        }

        for epoch in range(num_epochs):
            total_loss = 0
            accuracy = 0
            optimizer.zero_grad()

            data_iter = tqdm.tqdm(enumerate(self.loader),
                                  desc="EP_%s:%d" % ("test", epoch),
                                  total=len(self.loader),
                                  bar_format="{l_bar}{r_bar}")
            for i, data_ in data_iter:

                code_data = data_
                code_data.requires_grad = False

                batch_interval = slice(len(data_) * i, len(data_) * (i + 1))
                target = torch.tensor(self.label.labels[batch_interval], dtype=torch.float32, device=self.device)

                # FORMAT TO MODEL:
                code_data_1d = torch.flatten(code_data, start_dim=1) if code_data.ndim > 2 else code_data
                # 32 %40 -> 32 %30 = 64 %35 -> 32 % 50 -> 64 + 32 -> 96 -> %35 + %35 + %50 -> 40
                # TRAIN
                outputs = self.classifier(code_data_1d)  # Forward pass 32x768 32x3
                loss = criterion(outputs, target)  # Calculate the loss
                loss.backward()  # Backpropagation
                total_loss += loss.item()

                accuracy += torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(target, dim=1)).item()

                optimizer.step()  # Update weights

            total_loss = total_loss / len(self.loader)
            accuracy = accuracy / len(self.dataset)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy: .4f}')

            self.history["accuracy"].append(accuracy)
            self.history["losses"].append(total_loss)
            self.history["epoch_time"].append(data_iter.format_dict["elapsed"])

            if min_loss > total_loss:
                min_loss = total_loss
                self.best_model = copy.deepcopy(self.classifier)

            if total_loss > prev_loss - 0.00001:
                if patience == 0:
                    print("Training finished early!")
                    self.history["early_finish"] = "True"
                    return self.best_model
                else:
                    patience -= 1
            else:
                prev_loss = total_loss
                patience = default_patience

        print("Training finished!")
        self.history["early_finish"] = "False"
        return self.best_model
