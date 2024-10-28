import json
from datetime import datetime
from logic.utils import save_annotation
import numpy as np
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from pathlib import Path

from sklearn.preprocessing import LabelEncoder


class TSNECreator:
    @staticmethod
    def _scatter(x, labels, subtitle=None):
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", 10))
        plt.tight_layout()
        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                        c=palette[labels.astype(int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # We add the labels for each digit.
        txts = []
        for i in range(6):
            # Position of each label.
            xtext, ytext = np.median(x[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        #if subtitle != None:
        #    plt.suptitle(subtitle)

        now = datetime.now()
        now = now.strftime("%Y_%m_%d__%H_%M")
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        save_path = "/home/eislamoglu/Pictures/tsne/tsne_" + now + "_" + subtitle + ".png"
        plt.savefig(save_path, dpi=500, bbox_inches=0)

        return save_path



    @staticmethod
    def create_from_pair(original_data_path, new_data_path, label_path):
        """original_data must have same shape as new data
        """

        original_data_path = Path(original_data_path)
        new_data_path = Path(new_data_path)

        original_data = np.load(original_data_path, "r")
        original_data = original_data.reshape(-1, np.prod(original_data.shape[1:]))

        new_data = np.load(new_data_path, "r")
        new_data = new_data.reshape(-1, np.prod(new_data.shape[1:]))

        assert original_data.shape == new_data.shape

        labels = []
        with open(label_path, "r") as label:
            for line in label.readlines():
                labels.append(json.loads(line))
        label_df = pd.DataFrame(labels)
        y_tsne = label_df.smellKey
        le = LabelEncoder()
        y_tsne = le.fit_transform(y_tsne)

        tsne_new = TSNE()
        eval_new_tsne_embeds = tsne_new.fit_transform(new_data)

        tsne_original = TSNE()
        eval_original_tsne_embeds = tsne_original.fit_transform(original_data)

        original_tsne = TSNECreator._scatter(eval_new_tsne_embeds, y_tsne, new_data_path.stem)
        triplet_tsne = TSNECreator._scatter(eval_original_tsne_embeds, y_tsne, original_data_path.stem)
        save_annotation("tsne_" + new_data_path.stem, "Created tsne. Class codes for values are like" + str(le.classes_) +
                        "\nBy which I mean that given array is the order in which classes are coded. "
                        "So 0th element in the array is coded to 0 in the tsne. You get it.")
        return original_tsne, triplet_tsne
