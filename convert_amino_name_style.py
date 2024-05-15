import numpy as np
import pandas as pd


def three_to_one(three_letter_seq):
    # Dictionary to map three-letter codes to one-letter codes
    three_to_one_dict = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
    }

    # Split the input sequence into individual three-letter codes
    three_letter_list = three_letter_seq.split()

    # Convert each three-letter code to its one-letter code
    one_letter_seq = ''.join([three_to_one_dict[aa] for aa in three_letter_list])

    return one_letter_seq


file_name = "M10.004 sequences.txt"

original_style = pd.read_csv(file_name)
original_style = original_style.values.tolist()

new_style = []
for s in original_style:
    if '-' in s[0]:
        continue
    new_style.append(three_to_one(s[0]))

new_df = pd.DataFrame(new_style)

new_df.to_csv("MMP3_unique_sequence.csv", index=None, header=None)
a = 1