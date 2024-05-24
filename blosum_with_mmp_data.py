import pandas as pd
import numpy as np
from tqdm import tqdm
import blosum

bl62 = blosum.BLOSUM(62)


amino_acid_index = {
    'A': 0,  # Alanine
    'R': 1,  # Arginine
    'N': 2,  # Asparagine
    'D': 3,  # Aspartic acid
    'C': 4,  # Cysteine
    'E': 5,  # Glutamic acid
    'Q': 6,  # Glutamine
    'G': 7,  # Glycine
    'H': 8,  # Histidine
    'I': 9,  # Isoleucine
    'L': 10, # Leucine
    'K': 11, # Lysine
    'M': 12, # Methionine
    'F': 13, # Phenylalanine
    'P': 14, # Proline
    'S': 15, # Serine
    'T': 16, # Threonine
    'W': 17, # Tryptophan
    'Y': 18, # Tyrosine
    'V': 19  # Valine
}

def calculate_blosum62(sequence1, sequence2):
    l = len(sequence1)
    score = 0
    for i in range(l):
        score = score + bl62[sequence1[i]][sequence2[i]]

    return score


df = pd.read_csv("Data/mmp_data.txt", sep='\t')

sequences = ['RIFCCRSG', 'YMACCLLY', 'WCGHCKRL', 'AVAHCKRG', 'WCGHCKKL', 'YVKNCFRM', 'LQANCYEE', 'AAQNCTNV', 'GAINCTNV']

protease = 'MMP3'

cleave_threshold = 1.65
marginal = 0.1

sorted_df = df.sort_values(by=protease, ascending=False).reset_index(drop=True)
# sorted_df = sorted_df[sorted_df[protease] >= cleave_threshold + marginal]

for sequence in sequences:
    sorted_df[sequence] = None

for index, row in sorted_df.iterrows():
    s1 = row['Sequence']
    for s2 in sequences:
        sorted_df.loc[index, s2] = calculate_blosum62(s1[1:9], s2)

a = 1
target_df = sorted_df[['Sequence', protease, 'construct']]
