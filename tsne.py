#!/usr/bin/env python
# coding: utf-8
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

from logic import Label


# original_data_path = Path("/home/user/PycharmProjects/Model_Scratch/data/2500_smells_bert_nli_mean_pooler_output.npy")
# new_data_path = Path("/home/user/Desktop/Triplet-net-keras/Test/embeds_nli_pooler_1.npy")
#
# original_data = np.load(original_data_path, "r")
# original_data = original_data.reshape(-1, np.prod(original_data.shape[1:]))
# new_data = np.load(new_data_path, "r")
# new_data = new_data.reshape(-1, np.prod(new_data.shape[1:]))
#
# series_data = pd.Series(np.arange(0, 2500, dtype=np.float64))
# for i in range(len(series_data)):
#     if i < 500:
#         series_data[i] = 0
#     elif i < 1000:
#         series_data[i] = 1
#     elif i < 1500:
#         series_data[i] = 2
#     elif i < 2000:
#         series_data[i] = 3
#     else:
#         series_data[i] = 4
# y_tsne = np.array(series_data)


# Define our own plot function
def scatter(x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

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
    for i in range(5):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.suptitle(subtitle)

    now = datetime.now()
    now = now.strftime("%Y_%m_%d__%H_%M")

    save_path = "/home/eislamoglu/Pictures/tsne/tsne_" + now + "_" + subtitle + ".png"
    plt.savefig(save_path, dpi=500)

    return save_path


def tsne_plot(triplet_path, not_triplet_path, labels):
    triplet_path = Path(triplet_path)
    not_triplet_path = Path(not_triplet_path)

    triplet_embeddings = np.load(triplet_path, "r")
    triplet_embeddings = triplet_embeddings.reshape(-1, np.prod(triplet_embeddings.shape[1:]))
    not_triplet_embeddings = np.load(not_triplet_path, "r")
    not_triplet_embeddings = not_triplet_embeddings.reshape(-1, np.prod(not_triplet_embeddings.shape[1:]))

    tsne = TSNE()
    labels = Label(labels).labels
    print("Starting TSNE plotting of triplets...")
    train_tsne_embeds = tsne.fit_transform(triplet_embeddings)
    triplet_tsne = scatter(train_tsne_embeds, np.argmax(labels, axis=1), triplet_path.stem)

    print("Starting TSNE plotting of not triplets...")
    eval_tsne_embeds = tsne.fit_transform(not_triplet_embeddings)
    not_triplet_tsne = scatter(eval_tsne_embeds, np.argmax(labels, axis=1), not_triplet_path.stem)

    return triplet_tsne, not_triplet_tsne
