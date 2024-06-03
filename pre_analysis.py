import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import Counter

from utils import *


def main():
    """file_path = "Data/processed_peptides10.csv"
    cleavage_score = pd.read_csv(file_path)
    proteases = cleavage_score.columns.to_list()[1:]
    scores = cleavage_score[proteases].values
    corr = np.corrcoef(scores, rowvar=False)

    plt.imshow(corr, cmap='viridis')
    plt.colorbar()
    plt.xticks(np.arange(len(proteases)), labels=proteases, rotation=45, ha='center')
    plt.yticks(np.arange(len(proteases)), labels=proteases)
    plt.show()"""
    a = 1

    """file_path = "Data/prot_sequences_df.csv"
    if os.path.exists(file_path):
        print("Preprocessed Merops data exists, load the file")
        prot_sequences_df = pd.read_csv("Data/prot_sequences_df.csv")
    else:"""
    """print("Preprocessing Merops data")
    prot_sequences_df = preprocess_merops(False)
    print("Finish preprocess")"""

    human_protease = pd.read_csv("Data/human_protease.txt", header=None).iloc[:, 0].values.tolist()
    print("Load human proteases")

    proteases = pd.read_csv('Data/txtdata/Substrate_search.txt', sep='\t', header=None, encoding='utf8', low_memory=False,
                                dtype=str).iloc[:, 1].tolist()

    counts = Counter(proteases)

    counts = dict(counts)
    counts = np.sort(list(counts.values()))
    plt.scatter(np.arange(counts.shape[0]), counts)

    plt.show()

    unique_prot = [p for p in prot_sequences_df["prot"].unique().tolist() if p in human_protease]

    print("Extract peptides of each Merops protease")
    prot_peptides_dict, prot_peptides_num = extract_peptides(unique_prot, prot_sequences_df, 0)
    a = 1

    """file_path = "Data/prot_sequences_df.csv"
    if os.path.exists(file_path):
        print("Preprocessed Merops data exists, load the file")
        prot_sequences_df = pd.read_csv("Data/prot_sequences_df.csv")
    else:
        print("Preprocessing Merops data")
        prot_sequences_df = preprocess_merops()
        print("Finish preprocess")

    # prot_sequences_df = prot_sequences_df.sample(frac=0.1)

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

    peptides_all = list(set(sum(family_peptides_dict.values(), [])))

    if os.path.exists("Data/peptides_all_blosum.csv"):
        peptides_all_blosum = np.load("Data/peptides_all_blosum.csv")
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
    #peptides_all_blosum = pd.DataFrame(data=peptides_all_blosum, index=peptides_all, columns=peptides_all)

    if os.path.exists("Data/family_corr.csv"):
        family_corr = pd.read_csv("Data/family_corr.csv", index_col=0)
    else:
        family_corr = np.zeros((len(families), len(families)))

        for i in range(len(families)):
            peptides_list_1 = family_peptides_dict[families[i]]
            for j in range(i, len(families)):
                peptides_list_2 = family_peptides_dict[families[j]]
                blosum_scores_this = peptides_all_blosum.loc[peptides_list_1, :].loc[:, peptides_list_2].values
                family_corr[i, j] = np.mean(blosum_scores_this)
                family_corr[j, i] = np.mean(blosum_scores_this)

        family_corr = pd.DataFrame(data=family_corr, index=families, columns=families)
        family_corr.to_csv("Data/family_corr.csv")

    """"""plt.imshow(family_corr.values, cmap='viridis')
    plt.colorbar()
    plt.show()""""""

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
    plt.show()"""

    a = 1




if __name__ == "__main__":
    main()