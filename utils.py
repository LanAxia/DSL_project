import os

import pandas as pd
import numpy as np
from tqdm import tqdm
import blosum


bl62 = blosum.BLOSUM(62)
mmp_family = ['M10.001', 'M10.003', 'M10.005', 'M10.008', 'M10.002', 'M10.004', 'M10.006','M10.007','M10.009','M10.013'
    ,'M10.014','M10.015','M10.016','M10.017','M10.021','M10.019','M10.024']

def three_to_one(three_letter_seq):
    # Dictionary to map three-letter codes to one-letter codes
    three_to_one_dict = {
        '-': '-', 'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
    }

    # Split the input sequence into individual three-letter codes
    #three_letter_list = three_letter_seq.split()

    # Convert each three-letter code to its one-letter code
    one_letter_seq = ''.join([three_to_one_dict[aa] for aa in three_letter_seq])

    return one_letter_seq


def calculate_blosum62(sequence1, sequence2):
    l = len(sequence1)
    score = 0
    for i in range(l):
        if sequence2[i] != '-' and sequence1[i] != '-':
            score = score + bl62[sequence1[i]][sequence2[i]]

    return score


def preprocess_merops():
    substrate = pd.read_csv('Data/txtdata/Substrate_search.txt', sep='\t', header=None, encoding='utf8', low_memory=False,
                                dtype=str)
    substrate = substrate.iloc[:, [1] + list(range(4, 12))].replace("'", "", regex=True)
    prot_sequences = []
    for index, row in tqdm(substrate.iterrows()):
        prot = row[1]
        sequence = []
        valid_sequence = True
        for j in range(4, 12):
            if row[j] not in valid_amino:
                valid_sequence = False
                break
            sequence.append(row[j])
        if not valid_sequence:
            continue
        sequence = three_to_one(sequence)
        prot_sequences.append([prot, sequence])

    prot_sequences_df = pd.DataFrame(prot_sequences)
    prot_sequences_df = prot_sequences_df.rename(columns={0: "prot", 1: "sequence"})

    prot_sequences_df.to_csv("Data/prot_sequences_df.csv", index=False)

    return prot_sequences_df


def extract_peptides(proteases, prot_sequences_df, min_num):
    prot_peptides_dict = {}
    counts = []
    indexes = []

    for prot in tqdm(proteases):
        peptides_this = prot_sequences_df[prot_sequences_df['prot'] == prot]['sequence'].unique().tolist()
        if len(peptides_this) >= min_num:
            prot_peptides_dict[prot] = peptides_this
            counts.append(len(peptides_this))
            indexes.append(prot)
    prot_peptides_num = pd.DataFrame({'number of peptides': counts}, index=indexes)
    prot_peptides_num = prot_peptides_num.sort_values(by='number of peptides', ascending=False)

    return prot_peptides_dict, prot_peptides_num


def extract_family(proteases: list):
    families = [f[0] for f in proteases]
    families_unique = set(families)

    return list(families_unique)
