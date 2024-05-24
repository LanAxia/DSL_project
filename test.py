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


df = pd.read_csv("Data/mmp_data.txt", sep='\t')

protease = 'MMP9'

cleave_threshold = 1.65
margin = 0.1

sorted_df = df.sort_values(by=protease, ascending=False).reset_index(drop=True)
sorted_df = sorted_df[sorted_df[protease] >= cleave_threshold + margin]

target_df = sorted_df[['Sequence', protease, 'construct']]
others_df = sorted_df.drop(columns=['Sequence', protease, 'construct'])

new_columns = ['1', '2', '3', 'name1', 'name2', 'name3']
for column in new_columns:
    target_df.loc[:, column] = None
target_df['1'] = target_df['1'].astype(float)
target_df['2'] = target_df['2'].astype(float)
target_df['3'] = target_df['3'].astype(float)

for index, row in others_df.iterrows():
    top_scores = row.nlargest(3)
    target_df.loc[index, '1'] = top_scores.values[0]
    target_df.loc[index, '2'] = top_scores.values[1]
    target_df.loc[index, '3'] = top_scores.values[2]
    target_df.loc[index, 'name1'] = top_scores.index[0]
    target_df.loc[index, 'name2'] = top_scores.index[1]
    target_df.loc[index, 'name3'] = top_scores.index[2]

target_df = target_df[target_df['3'] < cleave_threshold - margin]
target_df['sum'] = target_df[['1', '3']].sum(axis=1)
target_df = target_df.sort_values(by='2', ascending=True).reset_index(drop=True)


peptides = df['Unnamed: 0']
peptides.column = ['Sequence']

peptides = peptides.str.replace(' ', '', regex=False)
peptides = peptides.str[1:-1]

binary_encoded = []

for i, peptide in tqdm(enumerate(peptides)):
    peptide = list(peptide)
    amino_list = [amino_acid_index[amino] for amino in peptide]
    one_hot = np.zeros((160,), dtype=int)
    for j, amino in enumerate(amino_list):
        one_hot[j * 20 + amino] = 1
    binary_encoded.append(one_hot)

binary = np.stack(binary_encoded)

binary_df = pd.DataFrame(binary)
binary_df.insert(0, None, peptides)

binary_df.to_csv('binary.tsv', index=False, header=False)