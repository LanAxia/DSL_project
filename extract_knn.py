import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
from collections import Counter
from tqdm import tqdm

import blosum

bl62 = blosum.BLOSUM(62)


def calculate_blosum62(sequence1, sequence2):
    l = len(sequence1)
    score = 0
    for i in range(l):
        score = score + bl62[sequence1[i]][sequence2[i]]

    return score


threshold = 1.65

train_idx = pd.read_csv("Cache/train_indices.csv", header=None).to_numpy(dtype=int).squeeze()

df = pd.read_csv("Data/processed_peptides10.csv", sep=',')

merops_df = pd.read_csv("Data/MMP3_unique_sequence.csv", header=None)

# df = df.loc[train_idx]

sequences = df["Sequence"].to_list()
merops_sequences = merops_df[0].to_list()


"""blosum = np.zeros((len(sequences), len(merops_sequences)), dtype=int)
for i in tqdm(range(len(sequences))):
    s1 = sequences[i]
    for j in range(len(merops_sequences)):
        s2 = merops_sequences[j]
        blosum[i, j] = int(calculate_blosum62(s1, s2))

np.save("blosum_MMP9.npy", blosum)"""

blosum_all = np.load("blosum_MMP3.npy")

num_sequences = blosum_all.shape[1]

blosum_all = pd.DataFrame(blosum_all)
a = 1
cols = df.columns[1:19].to_list()
# cols = ['MMP2', 'MMP7', 'MMP8', 'MMP9', 'MMP10', 'MMP11', 'MMP12', 'MMP13', 'MMP14', 'MMP15', 'MMP16', 'MMP17', 'MMP16']

ks = [0.01, 0.03, 0.05, 0.07, 0.09]

for col_i, col in enumerate(cols):
    knn = np.zeros((num_sequences, len(ks)))

    score_this = df[["Sequence", col]]
    print("===========", col, "============")
    for k_i, k in enumerate(ks):
        for j in tqdm(range(num_sequences)):
            bl_scores = blosum_all.loc[:, j].copy()
            score_this.loc[:, "blosum"] = bl_scores
            score_this_copy = score_this.copy().sort_values(by="blosum", ascending=False).reset_index(drop=True)
            score_this_copy["mask"] = score_this_copy.loc[:, col] > threshold
            knn[j, k_i] = score_this_copy.loc[0:int(len(df) * k), "mask"].mean()
        a = 1
    np.save(f"Data/knn_MMP3_{col}_prediction.npy", knn)

a = 1
# k = 0.01

"""for col in cols:
    scores_this = df[["Sequence", col]]
    scores_this["positive"] = (scores_this[col] >= threshold)
    scores_this["k"] = None
    for index, row in scores_this.iterrows():
        scores_others = scores_this.drop(index=index)
        scores_others["blosum"] = None
        s1 = row["Sequence"][1:9]
        for i, row1 in tqdm(scores_others.iterrows()):
            s2 = row1["Sequence"][1:9]
            scores_others.loc[i, "blosum"] = calculate_blosum62(s1, s2)
        scores_others = scores_others.sort_values(by="blosum", ascending=False).reset_index(drop=True)
        topk = scores_others.loc[0:int(len(scores_others) * k), "positive"].sum()
        scores_this.loc[index, "k"] = topk
    a = 1

a = 1"""