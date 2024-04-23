import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, save_npz, load_npz
from tqdm import tqdm

# constant
K_MAX = 6

# load data
df = pd.read_excel("./Data/peptides10.xlsx")

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

# preprocess data
peptides = df['Unnamed: 0']
peptides.column = ['Sequence']

peptides = peptides.str.replace(' ', '', regex=False)
peptides = peptides.str[1:-1]


# 存储和读取稀疏矩阵
def save_sparse_matrix(mat: pd.DataFrame, path: str) -> None:
    # 将pd.Dataframe保存为.npz文件
    mat = coo_matrix(mat.values)
    save_npz(path, mat)


def load_sparse_matrix(path: str) -> pd.DataFrame:
    # 读取.npz文件并转换为pd.DataFrame
    mat = load_npz(path).toarray()
    return pd.DataFrame(mat)


# extract features

# extract BINARY
def extract_binary_features(peptides: pd.DataFrame) -> pd.DataFrame:
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

    # 检查index是否正确，这里不用检查
    return binary_df


# extract cksaap
def cksaap(sequence: list, k: int):
    sequence_len = len(sequence)
    encoded = np.zeros((400,))
    for i in range(sequence_len - k):
        a1 = sequence[i]
        a2 = sequence[i + k]
        encoded[a1 * 20 + a2] = encoded[a1 * 20 + a2] + 1
    encoded = encoded / (sequence_len - k)

    return encoded

def extract_cksaap_features(peptides: pd.DataFrame) -> pd.DataFrame:
    cksaap_encode = np.zeros((len(peptides), 2400))
    for i, peptide in tqdm(enumerate(peptides), desc='Extracting cksaap features'):
        peptide = list(peptide)
        peptide_index = [amino_acid_index[amino] for amino in peptide]
        for k in range(1, K_MAX + 1):
            encoded = cksaap(peptide_index, k)
            cksaap_encode[i, (k - 1) * 400: k * 400] = encoded

    cksaap_df = pd.DataFrame(cksaap_encode)
    cksaap_df.insert(0, None, peptides)

    # 检查index是否正确，这里不用检查
    return cksaap_df


if __name__ == "__main__":
    # 保存peptides列
    peptides.to_csv('./Cache/peptides.csv', index=False, header=False)

    # 提取binary特征并保存
    binary_df = extract_binary_features(peptides)
    save_sparse_matrix(binary_df.iloc[:, 1:], './Cache/binary.npz')  # 不保存第一列

    # 提取cksaap特征并保存
    cksaap_df = extract_cksaap_features(peptides)
    save_sparse_matrix(cksaap_df.iloc[:, 1:], './Cache/cksaap.npz')  # 不保存第一列