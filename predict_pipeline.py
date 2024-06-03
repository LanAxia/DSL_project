import os

import numpy as np
import pandas as pd
from utils import *


def step2(target_peptides: list, target_protease, pipeline_type=0):
    """
    The second step of the predict pipeline.
    It load the result from
    :param target_peptides:
    :param target_protease:
    :param pipeline_type: 0 for search pipeline, 1 for predict pipeline
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

    target_final_df = target_final_df.sort_values(by=['top1', 'top5', 'top10'], )

    # save the final result in the ``Result`` folder
    if pipeline_type == 0:
        target_final_df.to_csv(f"Data/{target_protease}_search.csv")
    else:
        target_final_df.to_csv(f"Data/{target_protease}_predict.csv")


def main():
    for target_protease in ["mmp3", "mmp9"]:
        print(f"-------------------Predict Pipeline for {target_protease} starts----------------")
        if target_protease == "mmp3":
            mmp_index = 2
        elif target_protease == "mmp9":
            mmp_index = 5

        # load the result from the 1st step
        precision_threshold = 1.65
        recall_threshold = 0.0

        input_peptides = pd.read_csv(f'Data/{target_protease}_unique_sequence.csv', header=None)
        mmp_scores = np.load(f'Result/pred_{target_protease}.npy')
        valid_mask = (np.max(np.delete(mmp_scores, mmp_index, axis=1), axis=1) <= recall_threshold)

        valid_peptides = input_peptides[valid_mask].values.squeeze().tolist()

        # the 2nd step
        step2(valid_peptides, target_protease, 1)


if __name__ == "__main__":
    main()

