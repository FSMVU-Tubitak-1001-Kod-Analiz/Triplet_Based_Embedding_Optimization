import numpy as np
import pandas as pd
import os
import logic
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib as pl

class Results:
    @staticmethod
    def get_latest(folder, label_path, at=-1):
        files = []
        mtimes = []

        for i in os.listdir(folder):
            current_file = pl.Path(os.path.join(os.path.abspath(folder), i))
            current_time = current_file.stat().st_mtime

            files.append(current_file)
            mtimes.append(current_time)

        return Results(np.array(files)[np.array(mtimes).argsort()][at], label_path)

    def __init__(self, result_folder_path, label_path):
        self.result_folder_path = result_folder_path
        if isinstance(label_path, str):
            self.label_path = label_path

        self.labels = logic.Label(label_path)

        self._predictions_path = os.path.join(self.result_folder_path, "predictions.npy")
        self._metadata_path = os.path.join(self.result_folder_path, "metadata.json")

        self.metadata = json.load(open(self._metadata_path, "r"))

        self.results_matrix = np.load(self._predictions_path)
        results_array = np.argmax(self.results_matrix, axis=1)

        tested_labels = logic.Label(self.labels.label_series.iloc[self.metadata["indices"]])
        labels_array = np.argmax(tested_labels.labels, axis=1)

        assert results_array.shape[0] == tested_labels.labels.shape[0]

        predicted = pd.Series(results_array, name="predicted")
        target = pd.Series(labels_array, name="target")

        self.result_df = pd.concat([predicted, target], axis=1)

        self.class_count = len(target.unique())
        self.foldc = len(self.metadata["folds"])

    def __accuracy_score__(self, target: pd.Series, predicted: pd.Series):
        accuracy = np.zeros(self.class_count, dtype=np.float32)
        unique_vals = target.unique()
        for i in unique_vals:
            accuracy[i] = (target[(a := target.where(lambda x: x == i).dropna().index)] == predicted[a]).sum() / len(a)

        return pd.Series(accuracy, name="Accuracy")

    def __precision__score(self, target: pd.Series, predicted: pd.Series):
        from sklearn.metrics import precision_score
        return pd.Series([precision_score(target, predicted, average="macro", labels=[i], zero_division=0) for i in
                          range(self.class_count)], name="Precision")

    def __recall_score__(self, target: pd.Series, predicted: pd.Series):
        from sklearn.metrics import recall_score
        return pd.Series(
            [recall_score(target, predicted, average="macro", labels=[i]) for i in range(self.class_count)],
            name="Recall")

    def __f1_score__(self, target: pd.Series, predicted: pd.Series):
        from sklearn.metrics import f1_score
        return pd.Series([f1_score(target, predicted, average="macro", labels=[i]) for i in range(self.class_count)],
                         name="F1")

    def print_confusion_matrix(self, target=None, predicted=None):
        target = self.result_df["target"] if target is None else target
        predicted = self.result_df["predicted"] if predicted is None else predicted
        accuracy = self.__accuracy_score__(target, predicted)
        precision = self.__precision__score(target, predicted)
        recall = self.__recall_score__(target, predicted)
        f1 = self.__f1_score__(target, predicted)

        print(pd.concat([accuracy, precision, recall, f1], axis=1).T)
        crosstab = pd.crosstab(target, predicted, margins=True)
        crosstab.columns = list(crosstab.columns[:-1]) + ["Target"]
        crosstab.index = list(crosstab.index[:-1]) + ["Predicted"]
        return crosstab

    @staticmethod
    def __init_axis__(nrows, ncols, figsize=(10, 5), height_ratios=None, pad=None, fontsize=15, xrotate=None):
        scale = 0.75
        figsize = tuple(i * scale for i in figsize)
        fontsize *= scale
        if pad is not None:
            pad *= scale
        fig_, ax_ = plt.subplots(nrows, ncols, figsize=figsize, height_ratios=height_ratios)
        ax_: list[plt.Axes] = ax_
        if pad is not None:
            fig_.tight_layout(pad=pad)
        for i in np.array(ax_).ravel():
            i.tick_params(axis='x', labelsize=fontsize, rotation=xrotate)
            i.tick_params(axis='y', labelsize=fontsize)
            i.set_xlabel(i.get_xlabel(), fontsize=fontsize)
            i.set_ylabel(i.get_ylabel(), fontsize=fontsize)

        return fig_, ax_

    def plot_result_graph(self):
        target = self.result_df["target"]
        predicted = self.result_df["predicted"]
        df_melted = pd.concat([target.value_counts()._set_name("target"),
                               predicted.value_counts()._set_name("predicted")],
                              axis=1).reset_index().melt(id_vars="index")
        # sns.catplot()
        _, ax = Results.__init_axis__(1, 1, figsize=(15, 7.5))
        sns.barplot(df_melted, x="index", y="value", hue="variable", ax=ax)
        plt.show()

    def total_accuracy(self):
        return np.sum(self.result_df["predicted"] == self.result_df["target"]) / self.result_df.__len__()

    def __plot_per_epoch_for_folds__(self, function):
        _, ax = Results.__init_axis__(1, 1, figsize=(15, 7.5))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss" if "loss" in function.__name__ else "Accuracy")
        for i in range(self.foldc):
            data = function(i)
            sns.lineplot(data, ax=ax, label=i)
        ax.legend()
        return ax

    def plot_loss_per_epoch_for_folds(self):
        return self.__plot_per_epoch_for_folds__(self.loss_per_epoch_for_fold)

    def plot_acc_per_epoch_for_folds(self):
        return self.__plot_per_epoch_for_folds__(self.acc_per_epoch_for_fold)

    def loss_per_epoch_for_fold(self, fold):
        fold_data = self.metadata["folds"][fold]
        losses = fold_data["losses"]
        return losses

    def val_loss_per_epoch_for_fold(self, fold):
        fold_data = self.metadata["folds"][fold]
        losses = fold_data["val_losses"]
        return losses

    def acc_per_epoch_for_fold(self, fold):
        fold_data = self.metadata["folds"][fold]
        losses = fold_data["accuracy"]
        return losses

    def val_acc_per_epoch_for_fold(self, fold):
        fold_data = self.metadata["folds"][fold]
        losses = fold_data["val_accuracy"]
        return losses

    def __extract_fold__(self, fold):
        fold_indices = np.cumsum(
            np.pad([len(i) for i in np.array_split(np.arange(len(self.result_df["target"])), self.foldc)], (1, 0), "constant"))
        return self.result_df.iloc[fold_indices[fold]:fold_indices[fold + 1]]

    def get_class_ratios_for_fold(self, fold):
        fold_df = self.__extract_fold__(fold)
        fold_df = fold_df["target"].value_counts(normalize=True)
        return fold_df

    def print_confusion_matrix_for_fold(self, fold):
        fold_df = self.__extract_fold__(fold)
        self.print_confusion_matrix(target=fold_df["target"], predicted=fold_df["predicted"])

