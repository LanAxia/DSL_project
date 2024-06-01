# This file is used to extract all data features
# import packages
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
from collections import Counter
from tqdm import tqdm

# constant
K_MAX = 6

# construct amino to index map
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
    'L': 10,  # Leucine
    'K': 11,  # Lysine
    'M': 12,  # Methionine
    'F': 13,  # Phenylalanine
    'P': 14,  # Proline
    'S': 15,  # Serine
    'T': 16,  # Threonine
    'W': 17,  # Tryptophan
    'Y': 18,  # Tyrosine
    'V': 19  # Valine
}


# Functions for dealing with sparse matrices
def save_sparse_matrix(mat: pd.DataFrame, path: str) -> None:
    # save sparse matrix to given file path
    mat = coo_matrix(mat.values)
    save_npz(path, mat)


def load_sparse_matrix(path: str) -> pd.DataFrame:
    # load sparse matrix from given file path
    mat = load_npz(path).toarray()
    return pd.DataFrame(mat)


# extract BINARY
def extract_binary_features(peptides: pd.Series) -> pd.DataFrame:
    # extract binary features from peptides
    binary_encoded = []

    for i, peptide in tqdm(enumerate(peptides), desc='Extracting binary features'):
        peptide = list(peptide)
        amino_list = [amino_acid_index[amino] for amino in peptide]
        one_hot = np.zeros((160,), dtype=int)
        for j, amino in enumerate(amino_list):
            one_hot[j * 20 + amino] = 1
        binary_encoded.append(one_hot)

    binary = np.stack(binary_encoded)

    binary_df = pd.DataFrame(binary)
    binary_df.insert(0, None, peptides)
    return binary_df


# extract cksaap
def cksaap(sequence: list, k: int):
    # calculate cksaap features
    sequence_len = len(sequence)
    encoded = np.zeros((400,))
    for i in range(sequence_len - k):
        a1 = sequence[i]
        a2 = sequence[i + k]
        encoded[a1 * 20 + a2] = encoded[a1 * 20 + a2] + 1
    encoded = encoded / (sequence_len - k)

    return encoded


def extract_cksaap_features(peptides: pd.Series) -> pd.DataFrame:
    # extract cksaap features from peptide sequences
    cksaap_encode = np.zeros((len(peptides), 2400))
    for i, peptide in tqdm(enumerate(peptides), desc='Extracting cksaap features'):
        peptide = list(peptide)
        peptide_index = [amino_acid_index[amino] for amino in peptide]
        for k in range(1, K_MAX + 1):
            encoded = cksaap(peptide_index, k)
            cksaap_encode[i, (k - 1) * 400: k * 400] = encoded

    cksaap_df = pd.DataFrame(cksaap_encode)
    cksaap_df.insert(0, None, peptides)
    return cksaap_df


# extract AAC features
def extract_aac_features(peptides: pd.Series) -> pd.DataFrame:
    # extract aac features from peptide sequences
    aac_mat = []
    for i, peptide in enumerate(peptides):
        peptide_aac = [0] * 20
        amino_count = Counter(peptide)
        for amino in amino_count:
            amino_id = amino_acid_index[amino]
            peptide_aac[amino_id] = amino_count[amino] / len(peptide)  # calculate the frequency of each amino acid
        aac_mat.append(peptide_aac)
    aac_df = pd.DataFrame(aac_mat)
    return aac_df


def load_features() -> pd.DataFrame:
    # load all the features at one time
    peptides = pd.read_csv("./Cache/peptides.csv", header=None)
    binary = load_sparse_matrix("./Cache/binary.npz")
    cksaap = load_sparse_matrix("./Cache/cksaap.npz")
    aac = load_sparse_matrix("./Cache/aac.npz")
    knn = []
    for i in list(range(1, 4)) + list(range(7, 18)) + [19, 20, 24, 25]:  # load all KNN features
        mmp_knn = np.load("./Data/MMP{}_no_repeat.npy".format(i))
        knn.append(pd.DataFrame(mmp_knn))
    knn = pd.concat(knn, axis=1)

    features = pd.concat([peptides, binary, cksaap, aac, knn], axis=1)
    return features


def load_features_by_name(features_name: tuple) -> pd.DataFrame:
    # load the specified features.
    # the input features_name is a tuple, e.g. ["binary", "cksaap"]
    peptides = pd.read_csv("./Cache/peptides.csv", header=None)
    binary = load_sparse_matrix("./Cache/binary.npz")
    cksaap = load_sparse_matrix("./Cache/cksaap.npz")
    aac = load_sparse_matrix("./Cache/aac.npz")
    knn = []
    for i in list(range(1, 4)) + list(range(7, 18)) + [19, 20, 24, 25]:  # load all KNN features
        mmp_knn = np.load("./Data/MMP{}_no_repeat.npy".format(i))
        knn.append(pd.DataFrame(mmp_knn))
    knn = pd.concat(knn, axis=1)

    selected_features = [peptides]
    if "binary" in features_name:
        selected_features.append(binary)
    if "cksaap" in features_name:
        selected_features.append(cksaap)
    if "aac" in features_name:
        selected_features.append(aac)
    if "knn" in features_name:
        selected_features.append(knn)
    features = pd.concat(selected_features, axis=1)  # only use the specified features
    return features


# load features (merops dataset)
def load_pred_features(mmp: int) -> pd.DataFrame:
    # load all the features at one time
    mmp_peptides = pd.read_csv("./Data/MMP{}_unique_sequence.csv".format(mmp), header=None).iloc[:, 0]

    binary = extract_binary_features(mmp_peptides).iloc[:, 1:]
    cksaap = extract_cksaap_features(mmp_peptides).iloc[:, 1:]
    aac = extract_aac_features(mmp_peptides)
    knn = []
    for i in list(range(1, 4)) + list(range(7, 18)) + [19, 20, 24, 25]:  # load all KNN features
        mmp_knn = np.load("./Data/knn_MMP{}_MMP{}_prediction.npy".format(mmp, i))
        knn.append(pd.DataFrame(mmp_knn))
    knn = pd.concat(knn, axis=1)

    features = pd.concat([mmp_peptides, binary, cksaap, aac, knn], axis=1)  # return DataFrame
    return features


if __name__ == "__main__":
    # process data
    df = pd.read_excel("./Data/peptides10.xlsx")
    df.iloc[:, 0] = df.iloc[:, 0].map(lambda x: x.strip()).map(
        lambda x: x[1:-1])  # delete the first and last amino, because we need to limit the length of peptide to 8
    df = df.iloc[:, :-1]  # delete the last column

    # process duplicated rows
    peptides = df.iloc[:, 0]
    repeat_peptides = [x for x, v in Counter(peptides.values.tolist()).items() if v > 1]
    repeat_peptides = df[df.iloc[:, 0].isin(repeat_peptides)]
    repeat_peptides = repeat_peptides.groupby("Unnamed: 0").mean()
    repeat_peptides = repeat_peptides.reset_index()
    df = df[~df.iloc[:, 0].isin(repeat_peptides.iloc[:, 0])]
    df = pd.concat([df, repeat_peptides], axis=0).reset_index(drop=True)
    df.to_csv("./Data/processed_peptides10.csv", index=False)  # save the processed data.

    # save peptide sequences
    peptides = df['Unnamed: 0']
    peptides.column = ['Sequence']
    peptides.to_csv('./Cache/peptides.csv', index=False, header=False)

    # extract binary features and save
    binary_df = extract_binary_features(peptides)
    save_sparse_matrix(binary_df.iloc[:, 1:], './Cache/binary.npz')

    # extract cksaap features and save
    cksaap_df = extract_cksaap_features(peptides)
    save_sparse_matrix(cksaap_df.iloc[:, 1:], './Cache/cksaap.npz')

    # extract AAC features and save
    aac_df = extract_aac_features(peptides)
    save_sparse_matrix(aac_df, './Cache/aac.npz')

    # process merops data
    data = pd.read_csv("./Data/Protease_Peptides.csv", sep="\t")
    data = data[(data.iloc[:, 1:] != "-").sum(axis=1) == 8]  # delete the rows including "-"
    # 38727 rows, the number of unique peptides are 26794

    # map the 3-letter amino acid to 1-letter amino acid
    amino_3_to_1 = dict()
    amino_table = pd.read_csv("./Data/amino_table.csv", header=None, sep="\t")
    for i, x in amino_table.iterrows():
        amino_3_to_1[x[3].lower()] = x[2]
    data.iloc[:, 1:] = data.iloc[:, 1:].map(lambda x: amino_3_to_1[x.lower()])

    # save the peptide file
    pred_peptides = ["".join(x.tolist()) for i, x in data.iloc[:, 1:].iterrows()]
    pred_peptides = pd.DataFrame(pred_peptides)
    pred_peptides.to_csv("./Cache/to_predict_peptides.csv", index=False, header=None)

    pred_peptides = pred_peptides.iloc[:, 0]  # only use the first column
    pred_peptides.column = ['Sequence']
    pred_peptides.to_csv('./Cache/pred_peptides.csv', index=False, header=False)

    # extract binary features and save (for merops data)
    pred_binary_df = extract_binary_features(pred_peptides)
    save_sparse_matrix(pred_binary_df.iloc[:, 1:], './Cache/pred_binary.npz')  # 不保存第一列

    # extract cksaap features and save (for merops data)
    pred_cksaap_df = extract_cksaap_features(pred_peptides)
    save_sparse_matrix(pred_cksaap_df.iloc[:, 1:], './Cache/pred_cksaap.npz')  # 不保存第一列

    # extract aac features and save (for merops data)
    pred_aac_df = extract_aac_features(pred_peptides)
    save_sparse_matrix(pred_aac_df, './Cache/pred_aac.npz')
