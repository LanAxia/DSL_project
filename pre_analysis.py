import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import Counter

from utils import *


def corr_blosum_cleavage():
    """
    Generate the Figure 2: Relationship between average blosum score and cleavage correlation for two proteases
    """
    # load the cleavage scores and calculate the correlation
    cleave_scores = pd.read_csv("Data/processed_peptides10.csv")
    cleave_scores = cleave_scores.iloc[:, 1:].values
    row_norms = np.linalg.norm(cleave_scores, axis=1, keepdims=True)
    cleave_scores = cleave_scores / row_norms
    correlation_matrix = np.dot(cleave_scores, cleave_scores.T)

    file_path = "Data/blosum_no_repeat.npy"
    if os.path.exists(file_path):
        print("Blosum scores already calculated")
        blosum_scores = np.load("Data/blosum_no_repeat.npy")
        print("Finish load")
    else:
        print("Need to calculate blosum scores")
        calculate_all_blosum()
        print("Finish calculation")
        blosum_scores = np.load("Data/blosum_no_repeat.npy")
        print("Finish load")

    num_peptides = blosum_scores.shape[0]

    upper_tri_indices = np.triu_indices(num_peptides, k=1)

    correlation_1d = correlation_matrix[upper_tri_indices]
    blosum_scores_1d = blosum_scores[upper_tri_indices]
    del blosum_scores
    del correlation_matrix

    blosum_min = -10
    blosum_max = 43

    blosum_avg = np.zeros((blosum_max - blosum_min + 1,))
    corr_avg = np.zeros((blosum_max - blosum_min + 1,))
    corr_std = np.zeros((blosum_max - blosum_min + 1,))
    corr_num = np.zeros((blosum_max - blosum_min + 1,))

    # calculate the average correlation for every blosum scores between ``blosum_min`` and ``blosum_max``
    for i, score in tqdm(enumerate(range(blosum_min, blosum_max + 1))):
        corr_avg[i] = np.mean(correlation_1d[blosum_scores_1d == score])
        corr_std[i] = np.std(correlation_1d[blosum_scores_1d == score])
        corr_num[i] = correlation_1d[blosum_scores_1d == score].shape[0]

    plt.plot([i for i in range(blosum_min, blosum_max + 1)], corr_avg)
    plt.xlabel("blosum score", fontsize=16)
    plt.ylabel("cleavage pattern correlation", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig("Figures/Figure2.pdf")
    plt.close()


def corr_MMP():
    """
    Generate the Figure 3: Correlation of cleavage efficiency between different MMP family proteases
    """
    file_path = "Data/processed_peptides10.csv"
    cleavage_score = pd.read_csv(file_path)
    proteases = cleavage_score.columns.to_list()[1:]
    scores = cleavage_score[proteases].values
    corr = np.corrcoef(scores, rowvar=False)

    plt.imshow(corr, cmap='viridis')
    plt.colorbar()
    plt.xticks(np.arange(len(proteases)), labels=proteases, rotation=45, ha='center')
    plt.yticks(np.arange(len(proteases)), labels=proteases)
    plt.tight_layout()
    plt.savefig("Figures/Figure3.pdf")
    plt.close()


def corr_family():
    """
    Generate the Figure 1: Average blosum62 score between peptides that can be cleaved by different proteases
    """
    file_path = "Data/prot_sequences_df.csv"
    if os.path.exists(file_path):
        print("Preprocessed Merops data exists, load the file")
        prot_sequences_df = pd.read_csv("Data/prot_sequences_df.csv")
        print("Finish load")
    else:
        print("Preprocessing Merops data")
        prot_sequences_df = preprocess_merops()
        print("Finish preprocess")

    human_protease = pd.read_csv("Data/human_protease.txt", header=None).iloc[:, 0].values.tolist()
    print("Load human proteases")

    unique_prot = [p for p in prot_sequences_df["prot"].unique().tolist() if p in human_protease]

    print("Extract peptides of each Merops protease")
    prot_peptides_dict, prot_peptides_num = extract_peptides(unique_prot, prot_sequences_df, 20)

    families = extract_family(prot_peptides_num.index.to_list())

    family_peptides_dict = {family: [] for family in families}
    for protease in prot_peptides_dict.keys():
        family_peptides_dict[protease[0]].extend(prot_peptides_dict[protease])
    for family in families:
        family_peptides_dict[family] = list(set(family_peptides_dict[family]))

    # extract all peptides
    peptides_all = list(set(sum(family_peptides_dict.values(), [])))

    # calculate or load all blosum scores between every two peptides
    if os.path.exists("Data/peptides_all_blosum.npy"):
        peptides_all_blosum = np.load("Data/peptides_all_blosum.npy")
    else:
        peptides_all_blosum = np.zeros((len(peptides_all), len(peptides_all)))

        for i in tqdm(range(len(peptides_all))):
            for j in range(i, len(peptides_all)):
                bl_score = calculate_blosum62(peptides_all[i], peptides_all[j])
                peptides_all_blosum[i, j] = bl_score
                peptides_all_blosum[j, i] = bl_score

        peptides_all_blosum = peptides_all_blosum.astype(int)
        peptides_all_blosum = pd.DataFrame(data=peptides_all_blosum, index=peptides_all, columns=peptides_all)
        np.save("Data/peptides_all_blosum.csv", peptides_all_blosum)

    unique_prot_valid = list(prot_peptides_dict.keys())
    unique_prot_valid.sort()

    # calculate the average blosum scores between every two proteases
    if os.path.exists("Data/protease_corr.csv"):
        protease_corr = pd.read_csv("Data/protease_corr.csv", index_col=0)
    else:
        protease_corr = np.zeros((len(unique_prot_valid), len(unique_prot_valid)))
        for i in range(len(unique_prot_valid)):
            peptides_list_1 = prot_peptides_dict[unique_prot_valid[i]]
            for j in range(i, len(unique_prot_valid)):
                peptides_list_2 = prot_peptides_dict[unique_prot_valid[j]]
                blosum_scores_this = peptides_all_blosum.loc[peptides_list_1, :].loc[:, peptides_list_2].values
                protease_corr[i, j] = np.mean(blosum_scores_this)

        protease_corr = pd.DataFrame(data=protease_corr, index=unique_prot_valid, columns=unique_prot_valid)
        protease_corr.to_csv("Data/protease_corr.csv")

    for i in range(1, 106):
        for j in range(0, i):
            protease_corr.iloc[i, j] = protease_corr.iloc[j, i]

    norm = Normalize(vmin=-8, vmax=5)
    plt.imshow(protease_corr.values, cmap='viridis', norm=norm)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.savefig("Figures/Figure1.pdf")
    plt.close()


def calculate_all_blosum():
    sequences = pd.read_csv("Data/processed_peptides10.csv").iloc[:, 0].tolist()
    blosum_all = np.zeros((len(sequences), len(sequences)), dtype=int)

    print("\nStart to calculate blosum scores between every two peptides.")
    print("It may take 10 minutes to half an hour.")
    for i in tqdm(range(len(sequences))):
        s1 = sequences[i]
        for j in range(i, len(sequences)):
            s2 = sequences[j]
            bl_score = calculate_blosum62(s1, s2)
            blosum_all[i, j] = bl_score
            blosum_all[j, i] = bl_score

    np.save("Data/blosum_no_repeat.npy", blosum_all)


def main():
    # generate the Figure 1
    print("------------generate Figure1-------------")
    corr_family()
    # generate the Figure 2
    print("------------generate Figure2-------------")
    corr_blosum_cleavage()
    # generate the Figure 3
    print("------------generate Figure3-------------")
    corr_MMP()


if __name__ == "__main__":
    main()