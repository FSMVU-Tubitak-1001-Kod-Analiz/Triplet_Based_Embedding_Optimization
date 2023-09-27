import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import copy
import tqdm
import json
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import itertools
import more_itertools
import pandas as pd
from torch.utils.data.dataset import T_co
import math

def _peek(loader):
    gen = more_itertools.peekable(loader)
    peek_ = gen.peek()
    return peek_


# Define the neural network class
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleClassifier, self).__init__()
        hidden_input_size1 = 256 # input_size // 2
        hidden_input_size2 = 128  # input_size // 4

        self.linear = nn.Linear(input_size, hidden_input_size1)
        self.linear2 = nn.Linear(hidden_input_size1, hidden_input_size2)
        self.linear3 = nn.Linear(hidden_input_size2, hidden_input_size2)
        self.linear4 = nn.Linear(hidden_input_size2, output_size)
        self.drop = nn.Dropout(0.1)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        out = self.act(out)
        out = self.linear2(out)
        out = self.act(out)
        # out = self.drop(out)
        out = self.linear3(out)
        out = self.act(out)
        # out = self.drop(out)
        out = self.linear4(out)
        return out


class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiLabelClassifier, self).__init__()
        hidden_input_size1 = 256 # input_size // 2
        hidden_input_size2 = 128  # input_size // 4

        self.linear = nn.Linear(input_size, hidden_input_size1)
        self.linear2 = nn.Linear(hidden_input_size1, hidden_input_size2)
        self.linear3 = nn.Linear(hidden_input_size2, hidden_input_size2)
        self.linear4 = nn.Linear(hidden_input_size2, output_size)
        self.drop = nn.Dropout(0.1)
        self.act = torch.nn.ReLU()
        self.act2 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.linear3(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.linear4(x)
        x = self.act2(x)
        return x


class Label:
    def __init__(self, labels):
        self.ohe = None
        self.le = None

        self.labels = None
        self.label_count = None

        if isinstance(labels, str):
            self.label_path = labels
        else:
            self.label_path = None

        self.__read_labels__(labels)

    def __read_labels__(self, labels):
        label_series = None

        if isinstance(labels, str):  # labels is path
            labels_file = open(labels, 'r').readlines()
            smells = []
            for i in labels_file:
                test_line = json.loads(i)
                smells.append(test_line["smellKey"])

                label_series = pd.Series(smells)
        else:
            label_series = labels.copy()

        self.ohe = OneHotEncoder(sparse_output=False)
        self.le = LabelEncoder()

        self.labels = self.ohe.fit_transform(label_series.to_numpy().reshape(-1, 1))
        self.le.fit(label_series.to_numpy())
        self.label_count = len(self.ohe.categories_[0])


class Model:
    def __init__(self, label, dataset: data.Dataset, batch_size, device=None, validate=False, val_ratio=0.1):
        # labels
        assert isinstance(label, Label)
        self.label: Label = label
        self.validate = validate
        self.val_ratio = val_ratio

        # data
        self.batch_size = batch_size
        self.dataset = dataset
        self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size)

        # device
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # model
        self.classifier = None
        self.best_model = None

    def __initialize__(self):
        shape = _peek(self.loader).shape

        input_size_ = np.prod(shape[1:])  # Multiply shape values after batch size to get input size
        output_size_ = self.label.label_count  # Number of classes

        self.classifier = SimpleClassifier(input_size_, output_size_)
        self.classifier.to(self.device)
        self.classifier.train()

        # Define a loss function and an optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.classifier.parameters(), lr=1e-1)

        return criterion, optimizer

    def __initialize_multi_label__(self):
        shape = _peek(self.loader).shape

        input_size_ = np.prod(shape[1:])  # Multiply shape values after batch size to get input size
        output_size_ = self.label.label_count  # Number of classes

        self.classifier = MultiLabelClassifier(input_size_, output_size_)
        self.classifier.to(self.device)
        self.classifier.train()

        # Define a loss function and an optimizer
        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.classifier.parameters(), lr=1e-3)

        return criterion, optimizer

    # Define the model
    def train(self):
        criterion, optimizer = self.__initialize__()

        min_loss = 100
        num_epochs = 100

        for epoch in range(num_epochs):
            total_loss = 0
            accuracy = 0
            train_accuracy = 0
            train_total_loss = 0

            train_count = 0
            val_count = 0

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

                if self.validate:
                    assert 0 < self.val_ratio <= 1
                    # TARGET:
                    # TODO: CAN BE REPLACED WITH A LABEL DATASET

                    boundary = math.floor(len(target) * (1 - self.val_ratio))
                    train_interval = slice(0, boundary)
                    val_interval = slice(boundary, -1)

                    train_target = target[train_interval]
                    val_target = target[val_interval]

                    train_data = code_data_1d[train_interval]
                    val_data = code_data_1d[val_interval]

                    assert len(val_data) == len(val_target)
                    val_count += len(val_target)
                    train_count += len(train_target)

                    train_outputs = self.classifier(train_data)  # Forward pass 32x768 32x3
                    train_loss = criterion(train_outputs, train_target)  # Calculate the loss

                    val_outputs = self.classifier(val_data)  # Forward pass 32x768 32x3
                    val_loss = criterion(val_outputs, val_target)  # Calculate the loss

                    train_total_loss += train_loss.item()
                    total_loss += val_loss.item() if not np.isnan(val_loss.item()) else 0
                    val_loss.backward()

                    train_accuracy += (torch.sum(torch.argmax(train_outputs, dim=1) == train_target)
                                       / len(train_target)).item()
                    accuracy += a if not np.isnan(a := (torch.sum(torch.argmax(val_outputs, dim=1) == val_target)
                                                        / len(val_target)).item()) else 0
                    # if np.isnan((torch.sum(torch.argmax(val_outputs, dim=1) == val_target)
                    #                                     / len(val_target)).item()):
                    #     print("pfff")
                else:
                    # TRAIN
                    outputs = self.classifier(code_data_1d)  # Forward pass 32x768 32x3
                    loss = criterion(outputs, target)  # Calculate the loss
                    loss.backward()  # Backpropagation
                    total_loss += loss.item()
                    accuracy += (torch.sum(torch.argmax(outputs, dim=1) == target) / len(target)).item()

                optimizer.step()  # Update weights

            if self.validate:
                train_total_loss = train_total_loss / train_count
                total_loss = total_loss / val_count

                train_accuracy = train_accuracy / train_count
                accuracy = accuracy / val_count
                if epoch % 1000 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_total_loss:.4f}, Val Loss: {total_loss:.4f}, '
                          f'Accuracy: {accuracy:.5f}, Val Accuracy: {train_accuracy:.4f}')

            else:
                total_loss = total_loss / len(self.loader)
                accuracy = accuracy / len(self.loader)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Accuracy: {accuracy:.5f}')

            if min_loss > total_loss:
                min_loss = total_loss
                self.best_model = copy.deepcopy(self.classifier)
            # PUT EARLY STOP BELOW:

        print("Training finished!")
        return self.best_model

    def predict(self, X, y: np.ndarray):
        assert issubclass(X.__class__, data.DataLoader)
        assert issubclass(y.__class__, np.ndarray)
        predicted_list = []
        iterator = tqdm.tqdm(enumerate(X), total=len(X))
        self.best_model.eval()
        with torch.no_grad():
            for i, data_ in iterator:
                code_data_1d = torch.flatten(data_, start_dim=1) if data_.ndim > 2 else data_
                result = self.best_model(code_data_1d)
                predicted_list.append(result.detach().cpu().numpy())

        predicted_array = np.vstack(predicted_list)
        predicted = np.argmax(predicted_array, axis=1)
        if y.ndim > 1:
            assert y.ndim == 2
            y = np.argmax(y, axis=1)

        accuracy = np.sum(y == predicted) / len(y)
        print("Accuracy is:", accuracy)
        return predicted_array

    def train_multi_label(self):
        criterion, optimizer = self.__initialize_multi_label__()

        min_loss = 100
        num_epochs = 100
        prev_loss = 100
        patience = 3

        for epoch in range(num_epochs):
            total_loss = 0
            # accuracy = 0
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

                # TRAIN
                outputs = self.classifier(code_data_1d)  # Forward pass 32x768 32x3
                loss = criterion(outputs, target)  # Calculate the loss
                loss.backward()  # Backpropagation
                total_loss += loss.item()
                # accuracy += (torch.sum(torch.argmax(outputs, dim=1) == target) / len(target)).item()

                optimizer.step()  # Update weights

            total_loss = total_loss / len(self.loader)
            # accuracy = accuracy / len(self.loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}')

            if min_loss > total_loss:
                min_loss = total_loss
                self.best_model = copy.deepcopy(self.classifier)

            if total_loss > prev_loss - 0.0008:
                if patience == 0:
                    print("Training finished early!")
                    return self.best_model
                else:
                    patience -= 1
            else:
                prev_loss = total_loss
                patience = 3


        print("Training finished!")
        return self.best_model


class SingleTensorDataset(data.Dataset):
    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, index) -> T_co:
        return self.tensor[index]

    def __add__(self, other: 'data.Dataset[T_co]') -> 'data.ConcatDataset[T_co]':
        raise NotImplementedError
