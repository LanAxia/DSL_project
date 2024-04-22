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

K_MAX = 6

def cksaap(sequence: list, k: int):
    sequence_len = len(sequence)
    encoded = np.zeros((400,))
    for i in range(sequence_len - k):
        a1 = sequence[i]
        a2 = sequence[i + k]
        encoded[a1 * 20 + a2] = encoded[a1 * 20 + a2] + 1
    encoded = encoded / (sequence_len - k)

    return encoded


df = pd.read_excel(r"1-s2.0-S1074552115002574-mmc2.xlsx")

peptides = df['Unnamed: 0']
peptides.column = ['Sequence']

peptides = peptides.str.replace(' ', '', regex=False)
peptides = peptides.str[1:-1]

cksaap_encode = np.zeros((len(peptides), 2400))
for i, peptide in tqdm(enumerate(peptides)):
    print(peptide)
    peptide = list(peptide)
    peptide_index = [amino_acid_index[amino] for amino in peptide]
    for k in range(1, K_MAX + 1):
        encoded = cksaap(peptide_index, k)
        cksaap_encode[i, (k - 1) * 400: k * 400] = encoded

cksaap_df = pd.DataFrame(cksaap_encode)

cksaap_df.insert(0, None, peptides)
cksaap_df.to_csv('cksaap.tsv', index=False, header=False)