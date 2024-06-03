import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import blosum

import matplotlib.pyplot as plt

from utils import *


mmp_family = ['M10.001', 'M10.003', 'M10.005', 'M10.008', 'M10.002', 'M10.004', 'M10.006','M10.007','M10.009','M10.013'
    , 'M10.014', 'M10.015', 'M10.016', 'M10.017', 'M10.021', 'M10.019', 'M10.024']
# target_protease = "MMP3"


def step1(protease):
    """
    The first step of the search pipeline.
    Select peptides which have high cleavage scores for MMP3/MMP9 and low for others.
    The cleavage score difference should be greater than 0.4
    :param protease: ``MMP3`` or ``MMP9``
    :return: none
    """
    df = pd.read_csv("Data/processed_peptides10.csv", sep=',')

    cleave_threshold = 1.65  # score > 1.65 ==> can cleave
    margin_min = 0.4    # score difference between MMP3/MMP9 and other MMP need to be greater than 0.4

    df = df[df[protease] > cleave_threshold]
    target_scores = df[protease].values
    df = df.drop(protease, axis=1)
    other_scores = df.iloc[:, 1:].values
    sequences = df.iloc[:, 0].tolist()

    # calculate the score difference between MMP3/MMP9 and others
    margins = target_scores - np.max(other_scores, axis=1)

    margins_df = pd.DataFrame({'Sequence': sequences,
                               'target scores': target_scores,
                               'margins': margins,
                               'maximum other scores': np.max(other_scores, axis=1)
                               })
    margins_df = margins_df.sort_values('margins', ascending=False).reset_index(drop=True)

    # select peptides which fulfill the requirements and save the result from step 1 in the ``Data`` folder
    margins_df = margins_df[
        (margins_df['maximum other scores'] < cleave_threshold) & (margins_df['margins'] > margin_min)]\
        .reset_index(drop=True)
    margins_df.to_csv(f"Data/{protease}_search_step1.csv", index=None)


def step2(target_protease):
    """
    The second step of the search pipeline.
    It uses the peptides from the 1st step and compare with other peptides in the MEROPS dataset.
    Sort them according to the maximum blosum scores.
    :param target_protease: ``MMP3`` or ``MMP9``
    :return: none
    """
    # load peptides and proteases from the MEROPS dataset
    file_path = "Data/prot_sequences_df.csv"
    if os.path.exists(file_path):
        print("Preprocessed Merops data exists, load the file")
        prot_sequences_df = pd.read_csv("Data/prot_sequences_df.csv")
    else:
        print("Preprocessing Merops data")
        prot_sequences_df = preprocess_merops()
        print("Finish preprocess")

    # load human proteases
    human_protease = pd.read_csv("Data/human_protease.txt", header=None).iloc[:, 0].values.tolist()
    print("Load human proteases")

    # load result from step 1
    target_peptides_df = pd.read_csv(f"Data/{target_protease}_search_step1.csv")
    target_peptides_df.set_index('Sequence', inplace=True)
    target_peptides = target_peptides_df.index.tolist()
    print("Load targeted peptides from step 1")

    unique_prot = [p for p in prot_sequences_df["prot"].unique().tolist() if
                   (p not in mmp_family) & (p in human_protease)]

    # construct the dictionary between proteases and peptides
    print("Extract peptides of each Merops protease")
    prot_peptides_dict, prot_peptides_num = extract_peptides(unique_prot, prot_sequences_df, 0)

    # calculate the maximum, top5 and top10 blosum scores and sort
    print("Calculate the blosum score with MEROPS peptides")
    mmp_merops_compare_top1 = pd.DataFrame(columns=target_peptides, index=unique_prot)
    mmp_merops_compare_top5 = pd.DataFrame(columns=target_peptides, index=unique_prot)
    mmp_merops_compare_top10 = pd.DataFrame(columns=target_peptides, index=unique_prot)

    for protease in tqdm(unique_prot):
        compared_peptides = prot_peptides_dict[protease]
        num_peptides_c = len(compared_peptides)
        for peptide in target_peptides:
            blosum_scores = np.zeros((len(compared_peptides),))
            for i, peptide_c in enumerate(compared_peptides):
                blosum_scores[i] = calculate_blosum62(peptide, peptide_c)
            blosum_scores = np.sort(blosum_scores)[::-1]

            mmp_merops_compare_top1.at[protease, peptide] = blosum_scores[0]
            mmp_merops_compare_top5.at[protease, peptide] = np.mean(blosum_scores[0:min(num_peptides_c, 5)])
            mmp_merops_compare_top10.at[protease, peptide] = np.mean(blosum_scores[0:min(num_peptides_c, 10)])

        mmp_merops_compare_top1 = mmp_merops_compare_top1.astype(float)
        mmp_merops_compare_top5 = mmp_merops_compare_top5.astype(float)
        mmp_merops_compare_top10 = mmp_merops_compare_top10.astype(float)

    target_final_df = pd.DataFrame(columns=['top1', 'top5', 'top10'], index=target_peptides)

    for peptide in target_peptides:
        bl_this = mmp_merops_compare_top1[peptide].values
        bl_this = np.sort(bl_this)[::-1]
        target_final_df.at[peptide, 'top1'] = bl_this[0]
        target_final_df.at[peptide, 'top5'] = np.mean(bl_this[0:5])
        target_final_df.at[peptide, 'top10'] = np.mean(bl_this[0:10])

    target_final_df = pd.merge(target_final_df, target_peptides_df, left_index=True, right_index=True)
    target_final_df = target_final_df.sort_values('top1')

    # save the final result in the ``Result`` folder
    print("Save the final result")
    target_final_df.to_csv(f"Result/{target_protease}_search.csv")


def main():

    for protease in ["MMP3", "MMP9"]:
        print(f"-------------------Search Pipeline for {protease} starts----------------")
        print("-------Step1--------")
        step1(protease)
        print("-------Step2--------")
        step2(protease)


if __name__ == "__main__":
    main()