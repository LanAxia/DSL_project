import pandas as pd
import numpy as np
from tqdm import tqdm


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


df = pd.read_csv("Data/processed_peptides10.csv", sep=',')

protease = 'MMP3'
cleave_threshold = 1.65
margin_min = 0.4

df = df[df[protease] > cleave_threshold]
target_scores = df[protease].values
df = df.drop(protease, axis=1)
other_scores = df.iloc[:, 1:].values
sequences = df['Sequence'].to_list()

margins = target_scores - np.max(other_scores, axis=1)

margins_df = pd.DataFrame({'Sequence': sequences,
                           'target scores': target_scores,
                           'margins': margins,
                           'maximum other scores': np.max(other_scores, axis=1)
                           })

margins_df = margins_df.sort_values('margins', ascending=False).reset_index(drop=True)

margins_df = margins_df[(margins_df['maximum other scores'] < cleave_threshold) & (margins_df['margins'] > margin_min)].reset_index(drop=True)
margins_df.to_csv(f"Data/{protease}_filtered_mmp.csv", index=None)