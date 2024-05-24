import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


cleave_scores = pd.read_csv("Data/processed_peptides10.csv")
cleave_scores = cleave_scores.iloc[:, 1:].values
row_norms = np.linalg.norm(cleave_scores, axis=1, keepdims=True)
cleave_scores = cleave_scores / row_norms
correlation_matrix = np.dot(cleave_scores, cleave_scores.T)

blosum_scores = np.load("blosum_no_repeat.npy")
print("finish loading")

num_peptides = blosum_scores.shape[0]

upper_tri_indices = np.triu_indices(num_peptides, k=1)

correlation_1d = correlation_matrix[upper_tri_indices]
blosum_scores_1d = blosum_scores[upper_tri_indices]
del blosum_scores
del correlation_matrix
"""sorted_index = np.argsort(blosum_scores_1d)

print("finish sorting")

blosum_scores_1d = blosum_scores_1d[sorted_index]
correlation_1d = correlation_1d[sorted_index]"""

blosum_min = -10
blosum_max = 43

blosum_avg = np.zeros((blosum_max - blosum_min + 1, ))
corr_avg = np.zeros((blosum_max - blosum_min + 1, ))
corr_std = np.zeros((blosum_max - blosum_min + 1, ))
corr_num = np.zeros((blosum_max - blosum_min + 1, ))

for i, score in tqdm(enumerate(range(blosum_min, blosum_max + 1))):
    corr_avg[i] = np.mean(correlation_1d[blosum_scores_1d == score])
    corr_std[i] = np.std(correlation_1d[blosum_scores_1d == score])
    corr_num[i] = correlation_1d[blosum_scores_1d == score].shape[0]

plt.plot([i for i in range(blosum_min, blosum_max + 1)], corr_avg)
plt.fill_between([i for i in range(blosum_min, blosum_max + 1)], corr_avg-corr_std, corr_avg+corr_std, color='grey', alpha=0.5)
plt.xlabel("blosum score", fontsize=16)
plt.ylabel("cleavage pattern correlation", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True, which='both')

plt.show()

a = 1