import pathlib
import zipfile
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd
import os

import sklearn.metrics
from matplotlib.ticker import NullFormatter, FixedLocator

import logic
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib as pl
from enum import Enum

import tsne
from logic import utils
import sklearn.metrics as metrics

class Metric(Enum):
    LOSS = "loss"
    ACCURACY = "accuracy"


class Results:
    @staticmethod
    def get_latest(folder, at=-1):
        files = []
        mtimes = []

        for i in os.listdir(folder):
            current_file = pl.Path(os.path.join(os.path.abspath(folder), i))

            if i.startswith("2024_"):
                current_time = current_file.stat().st_mtime
                files.append(current_file)
                mtimes.append(current_time)

        return Results(np.array(files)[np.array(mtimes).argsort()][at])

    def __init__(self, result_folder_path):
        self.result_folder_path = result_folder_path

        self._predictions_path = os.path.join(self.result_folder_path, "predictions.npy")
        self._metadata_path = os.path.join(self.result_folder_path, "metadata.json")

        self.metadata = json.load(open(self._metadata_path, "r"))

        self.label_path = self.metadata["label_path"]

        self.labels = logic.Label(self.label_path)

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

    def print_confusion_matrix(self, target=None, predicted=None, rename_cols=False):
        target = self.result_df["target"] if target is None else target
        predicted = self.result_df["predicted"] if predicted is None else predicted
        accuracy = self.__accuracy_score__(target, predicted)
        precision = self.__precision__score(target, predicted)
        recall = self.__recall_score__(target, predicted)
        f1 = self.__f1_score__(target, predicted)

        if rename_cols:
            print((a := pd.concat([accuracy, precision, recall, f1], axis=1).T).rename(columns={i:j for i, j in enumerate(self.labels.le.inverse_transform(a.columns))}))
        else:
            print(pd.concat([accuracy, precision, recall, f1], axis=1).T)
        crosstab = pd.crosstab(target, predicted, margins=True)
        if rename_cols:
            crosstab.columns = list(self.labels.le.inverse_transform([int(i) for i in crosstab.columns[:-1]])) + ["Target"]
            crosstab.index = list(self.labels.le.inverse_transform([int(i) for i in crosstab.index[:-1]])) + ["Predicted"]
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

    def f1(self):
        return metrics.f1_score(self.result_df["target"], self.result_df["predicted"])

    # def __plot_per_epoch_for_folds__(self, function):
    #     _, ax = Results.__init_axis__(1, 1, figsize=(15, 7.5))
    #     ax.set_xlabel("Epoch")
    #     ax.set_ylabel("Loss" if "loss" in function.__name__ else "Accuracy")
    #     for i in range(self.foldc):
    #         data = function(i)
    #         sns.lineplot(data, ax=ax, label=i)
    #     ax.legend()
    #     return ax
    #
    # def plot_loss_per_epoch_for_folds(self):
    #     return self.__plot_per_epoch_for_folds__(self.loss_per_epoch_for_fold)
    #
    # def plot_acc_per_epoch_for_folds(self):
    #     return self.__plot_per_epoch_for_folds__(self.acc_per_epoch_for_fold)

    def reflect_for_fold(self, title: Metric, is_val):
        if title == Metric.LOSS:
            if is_val:
                return self.val_loss_per_epoch_for_fold
            else:
                return self.loss_per_epoch_for_fold
        elif title == Metric.ACCURACY:
            if is_val:
                return self.val_acc_per_epoch_for_fold
            else:
                return self.acc_per_epoch_for_fold
        else:
            raise ValueError("Illegal title")

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
            np.pad([len(i) for i in np.array_split(np.arange(len(self.result_df["target"])), self.foldc)], (1, 0),
                   "constant"))
        return self.result_df.iloc[fold_indices[fold]:fold_indices[fold + 1]]

    def get_class_ratios_for_fold(self, fold):
        fold_df = self.__extract_fold__(fold)
        fold_df = fold_df["target"].value_counts(normalize=True)
        return fold_df

    def print_confusion_matrix_for_fold(self, fold):
        fold_df = self.__extract_fold__(fold)
        self.print_confusion_matrix(target=fold_df["target"], predicted=fold_df["predicted"])


def print_result(lh_result: Results, rh_result: Results, lh_title=None, rh_title=None):
    save_path = "results/print/" + utils.get_now() + ".txt"
    with open(save_path, "w") as f:
        with redirect_stdout(f):
            print("LEFTHAND" if lh_title is None else lh_title.upper())
            print("PATH", lh_result.result_folder_path)
            print("ACC", lh_result.total_accuracy())
            print("PRINT")
            print(lh_result.print_confusion_matrix())
            print()
            print("RIGHTHAND" if rh_title is None else rh_title.upper())
            print("PATH", rh_result.result_folder_path)
            print("ACC", rh_result.total_accuracy())
            print("PRINT")
            print(rh_result.print_confusion_matrix())

    return save_path


def plot_result(metric: Metric, do_val, lh_result, rh_result, lh_title=None, rh_title=None):
    def forward(a):
        return np.power(np.abs(a), 1 / 3)

    def inverse(a):
        return np.power(a, 3)

    def style(ax_, title, lim):
        ax_.set_xscale("function", functions=(forward, inverse))
        ax_.xaxis.set_minor_formatter(NullFormatter())
        ax_.set_xlim([0, lim])

        xticks = np.round(np.linspace(0, lim ** (1/3), 4) ** 3).astype(int)
        # xticks = np.concatenate([xticks[xticks < lim], [lim]])

        ax_.xaxis.set_major_locator(FixedLocator(xticks))
        ax_: plt.Axes = ax_

        ax_.legend(fontsize=13)
        ax_.set_title(title, fontdict={"fontsize": 16})

    _, ax = Results.__init_axis__(1, 1, (12, 8.5), fontsize=18)

    lh_title = lh_title if lh_title is not None else "lefthand embeddings"
    rh_title = rh_title if rh_title is not None else "righthand embeddings"

    assert metric == Metric.LOSS or metric == metric.ACCURACY

    fold_count_min_lh = np.min([len(lh_result.reflect_for_fold(metric, False)(i)) for i in range(5)])
    fold_count_min_rh = np.min([len(rh_result.reflect_for_fold(metric, False)(i)) for i in range(5)])
    fold_count_min = min(fold_count_min_lh, fold_count_min_rh)

    lh_values = np.mean(
        np.array([lh_result.reflect_for_fold(metric, False)(i)[:fold_count_min] for i in range(5)]), axis=0)
    rh_values = np.mean(
        np.array([rh_result.reflect_for_fold(metric, False)(i)[:fold_count_min] for i in range(5)]), axis=0)

    plot_title = metric.value

    sns.lineplot(lh_values, label=lh_title + " train " + plot_title, linewidth=1, ax=ax)
    sns.lineplot(rh_values, label=rh_title + " train " + plot_title, linewidth=1, ax=ax)

    if do_val:
        lh_val_values = np.mean(
            np.array([lh_result.reflect_for_fold(metric, True)(i)[:fold_count_min] for i in range(5)]), axis=0)
        rh_val_values = np.mean(
            np.array([rh_result.reflect_for_fold(metric, True)(i)[:fold_count_min] for i in range(5)]),
            axis=0)
        sns.lineplot(lh_val_values, label=lh_title + " validation " + plot_title, ax=ax)
        sns.lineplot(rh_val_values, label=rh_title + " validation " + plot_title, ax=ax)

    style(ax, "Training " + plot_title.title() + " per Epoch", fold_count_min)

    now = utils.get_now()
    # save_path
    save_path = ("/home/eislamoglu/Pictures/accs/accuracy_graph_" + now
                 if metric == Metric.ACCURACY else "/home/eislamoglu/Pictures/losses/loss_graph_" + now) + ".png"
    plt.savefig(save_path, dpi=500)

    return save_path


def save_results(result_lh: Results, result_rh: Results, lh_title=None, rh_title=None, do_val=False, do_tsne=False):
    acc_graph = plot_result(Metric.ACCURACY, do_val, result_lh, result_rh, lh_title, rh_title)
    loss_graph = plot_result(Metric.LOSS, do_val, result_lh, result_rh)
    result_path = print_result(result_lh, result_rh)

    if do_tsne:
        lh_path = result_lh.metadata["file_path"]
        rh_path = result_rh.metadata["file_path"]
        lh_tsne, rh_tsne = tsne.tsne_plot(lh_path, rh_path, result_lh.label_path)

        compress(acc_graph, loss_graph, result_path, lh_tsne, rh_tsne, zipfile_name=pathlib.Path(result_lh.metadata["file_path"]).stem)

    else:
        compress(acc_graph, loss_graph, result_path, zipfile_name=pathlib.Path(result_lh.metadata["file_path"]).stem)


def compress(*file_names, zipfile_name):
    print("File Paths:")
    print(file_names)

    # Select the compression mode ZIP_DEFLATED for compression
    # or zipfile.ZIP_STORED to just store the file
    compression = zipfile.ZIP_DEFLATED

    # create the zip file first parameter path/name, second mode
    now = utils.get_now()
    zf = zipfile.ZipFile("results/archives/" + now + "_" + zipfile_name + ".zip", mode="w")
    try:
        for file_name in file_names:
            # Add file to the zip file
            # first parameter file to zip, second filename in zip
            zf.write(file_name, os.path.basename(file_name), compress_type=compression)
        print("Zip file created at:", "results/archives/" + now + "_" + zipfile_name + ".zip")
    except FileNotFoundError as e:
        print("An error occurred")
        raise e
    finally:
        zf.close()
