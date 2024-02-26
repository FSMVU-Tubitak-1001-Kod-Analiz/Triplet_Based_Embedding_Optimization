import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import logic
import logic.datasets
import logic.models
import logic.utils


class Runner:
    def __init__(self, label_path, file_path, val_ratio, params, force=False, device=None, title=None, output_folder=None):
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.label_path = label_path
        self.file_path = file_path
        self.title = title
        self.output_folder = output_folder

        self.val_ratio = val_ratio
        self.validate = (val_ratio != 0)

        assert params is not None
        self.params = params

        label_extension = os.path.splitext(label_path)[1]
        try:
            if label_extension == '.csv' or label_extension == '.json':
                self.labels = logic.Label(label_path)
            else:
                raise TypeError('Illegal label path')
        except KeyError as e:
            print("Smells should be under the key 'smellKey'")
            raise e

        file_extension = os.path.splitext(file_path)[1]
        if file_extension != '.npy' and not force:
            raise TypeError('Extension of data file should be .npy and the file should be embeddings saved using the '
                            'function \'numpy.save\'. If the file is a numpy array, change extension or set argument '
                            'force=True.')
        else:
            self.dataset = logic.datasets.LazyDataset(file_path, device)

    r'''Args: smell_range (int or (int, int)) : which smells to test based on value counts. Single parameter means 
    test only the x most frequent smells, tuple parameter means test the smells between x most frequent and y.'''
    def __pre_run__(self, smells, shuffle):
        if isinstance(smells, tuple):
            assert len(smells) == 2
            smell_range = smells
        else:
            smell_range = (0, smells)

        label_series = self.labels.label_series
        most_freq = label_series.value_counts().index[smell_range[0]:smell_range[1]]
        top_indices = np.asarray(label_series[label_series.apply(lambda x: x in list(most_freq))].index)
        
        if shuffle:
            print(f"Current seed {np.random.get_state()[1][0]}")
            np.random.shuffle(top_indices)

        self.smell_range = smell_range
        self.smell_names = label_series.value_counts().index[smell_range[0]:smell_range[1]].tolist()

        return top_indices

    def run(self, smells: int | tuple[int, int], shuffle=False, fold_size=5, test_batch_size=32):
        top_indices = self.__pre_run__(smells, shuffle)

        folds = np.array_split(top_indices, fold_size)
        folds_as_array = np.asarray(folds, dtype=np.ndarray)
        predictions_list = []
        history_list = []
        indices = []

        for i, _ in enumerate(folds):
            print(("-" * 20), "Start of fold", i, ("-" * 20))

            folds_as_array = np.ma.array(folds_as_array, mask=False)
            folds_as_array.mask[i] = True

            train = folds_as_array.compressed() if len(top_indices) % fold_size == 0 else np.concatenate(folds_as_array.compressed())
            test = folds[i]
            indices.extend(test.tolist())

            data_train = logic.datasets.LazyDataset(self.file_path, self.device, indices=train)
            data_test = logic.datasets.LazyDataset(self.file_path, self.device, indices=test)

            train_target = logic.Label(self.labels.label_series.iloc[train])
            test_target = logic.Label(self.labels.label_series.iloc[test])

            if "writer" in self.params.keys():
                self.params["writer"] = SummaryWriter()

            model_ = logic.models.ValidationModel(train_target, data_train, self.params,
                                                  self.val_ratio, self.device)

            model_.train()

            predicts, test_accuracy = logic.utils.predict(model_.best_model, DataLoader(data_test, batch_size=test_batch_size),
                                           test_target.labels, False)

            penultimate_history = {
                "fold": i,
                "test_accuracy": test_accuracy
            }

            penultimate_history.update(model_.history)

            history_list.append(penultimate_history)

            folds_as_array.mask[i] = False
            predictions_list.append(predicts)

            self.params["writer"].close()

        final_history = {
            "smell_range": str(self.smell_range),
            "smell_names": str(self.smell_names),
            "label_path": str(self.label_path),
            "file_path": str(self.file_path),
            "indices": indices,
            "parameters": {
                "val_ratio": str(self.val_ratio),
                "train_batch_size": str(self.params["train_batch_size"]),
                "lr": str(self.params["lr"])
            },
            "folds": history_list
        }

        from datetime import datetime
        import json

        now = datetime.now()

        now = now.strftime("%Y_%m_%d__%H_%M")

        if self.output_folder is not None:
            if not os.path.exists(self.output_folder):
                os.mkdir(self.output_folder)
            folder = os.path.join(self.output_folder, now) + "/"
        elif os.path.exists("results") and os.path.isdir("results"):
            folder = os.path.join("results", now) + "/"
        elif os.path.exists("../results") and os.path.isdir("../results"):
            folder = os.path.join("../results", now) + "/"
        else:
            folder = now + "/"

        if self.title is not None:
            folder = folder[:-1] + "_" + self.title + folder[-1]

        os.mkdir(folder)
        save_history_at = folder + "metadata.json"
        save_at = folder + "predictions.npy"

        try:
            np.save(open(save_at, "wb"), np.vstack(predictions_list))
            json.dump(final_history, open(save_history_at, "w"), indent=4)
        except Exception as e:
            print(e)
            print("Emergency dumping results and history...")
            print(final_history)
            np.save(open("emergency_dump.npy", "wb"), np.vstack(predictions_list))
            json.dump(final_history, open("emergency_history_dump.json", "w"), indent=4)
            raise e

        print("Prediction results saved at", save_at)
        print("Prediction metadata saved at", save_history_at)
        return folder
