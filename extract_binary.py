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


df = pd.read_excel(r"1-s2.0-S1074552115002574-mmc2.xlsx")

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