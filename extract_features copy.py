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


# 存储和读取稀疏矩阵
def save_sparse_matrix(mat: pd.DataFrame, path: str) -> None:
    # 将pd.Dataframe保存为.npz文件
    mat = coo_matrix(mat.values)
    save_npz(path, mat)


def load_sparse_matrix(path: str) -> pd.DataFrame:
    # 读取.npz文件并转换为pd.DataFrame
    mat = load_npz(path).toarray()
    return pd.DataFrame(mat)


# extract BINARY
def extract_binary_features(peptides: pd.Series) -> pd.DataFrame:
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


def extract_cksaap_features(peptides: pd.Series) -> pd.DataFrame:
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


# 提取AAC特征
def extract_aac_features(peptides: pd.Series) -> pd.DataFrame:
    aac_mat = []
    for i, peptide in enumerate(peptides):
        peptide_aac = [0] * 20
        amino_count = Counter(peptide)
        for amino in amino_count:
            amino_id = amino_acid_index[amino]
            peptide_aac[amino_id] = amino_count[amino] / len(peptide)  # 计算每个氨基酸出现的频率
        aac_mat.append(peptide_aac)
    aac_df = pd.DataFrame(aac_mat)
    return aac_df


# 读取所有特征
def load_features() -> pd.DataFrame:
    # 一次性加载所有特征，返回DataFrame
    peptides = pd.read_csv("./Cache/peptides.csv")
    binary = load_sparse_matrix("./Cache/binary.npz")
    cksaap = load_sparse_matrix("./Cache/cksaap.npz")
    aac = load_sparse_matrix("./Cache/aac.npz")
    knn = []
    for i in list(range(1, 4)) + list(range(7, 18)) + [19, 20, 24, 25]:  # 加载所有的knn特征
        mmp_knn = np.load("./Data/MMP{}_no_repeat.npy".format(i))
        knn.append(pd.DataFrame(mmp_knn))
    knn = pd.concat(knn, axis=1)

    features = pd.concat([peptides, binary, cksaap, aac, knn], axis=1)  # 仅使用部分特征
    return features


# 读取所有待预测数据的特征
def load_pred_features(mmp: int) -> pd.DataFrame:
    # 一次性加载所有特征，返回DataFrame
    all_peptides = pd.read_csv("./Cache/pred_peptides.csv", header=None)
    mmp_peptides = pd.read_csv("./MMP{}_unique_sequence.csv".format(mmp), header=None)
    pep_ids = [all_peptides[all_peptides[0] == x[0]].index.to_list()[0] for x in mmp_peptides.values.tolist()]

    binary = load_sparse_matrix("./Cache/pred_binary.npz").iloc[pep_ids].reset_index(drop=True)
    cksaap = load_sparse_matrix("./Cache/pred_cksaap.npz").iloc[pep_ids].reset_index(drop=True)
    aac = load_sparse_matrix("./Cache/pred_aac.npz").iloc[pep_ids].reset_index(drop=True)
    knn = []  # 这部分代码没写完
    for i in list(range(1, 4)) + list(range(7, 18)) + [19, 20, 24, 25]:  # 加载所有的knn特征
        mmp_knn = np.load("./Data/knn_MMP{}_MMP{}_prediction.npy".format(mmp, i))
        knn.append(pd.DataFrame(mmp_knn))
    knn = pd.concat(knn, axis=1)

    features = pd.concat([mmp_peptides, binary, cksaap, aac, knn], axis=1)  # 仅使用部分特征
    return features


if __name__ == "__main__":
    # 处理训练数据
    df = pd.read_excel("./Data/peptides10.xlsx")
    df.iloc[:, 0] = df.iloc[:, 0].map(lambda x: x.strip()).map(lambda x: x[1:-1])  # 删除首尾的氨基酸和空格
    df = df.iloc[:, :-1]  # 删除最后一列

    # 处理重复数据
    peptides = df.iloc[:, 0]
    repeat_peptides = [x for x, v in Counter(peptides.values.tolist()).items() if v > 1]
    repeat_peptides = df[df.iloc[:, 0].isin(repeat_peptides)]
    repeat_peptides = repeat_peptides.groupby("Unnamed: 0").mean()
    repeat_peptides = repeat_peptides.reset_index()
    df = df[~df.iloc[:, 0].isin(repeat_peptides.iloc[:, 0])]
    df = pd.concat([df, repeat_peptides], axis=0).reset_index(drop=True)
    df.to_csv("./Data/processed_peptides10.csv", index=False)  # 保存处理后的数据

    # 保存peptides列
    peptides = df['Unnamed: 0']
    peptides.column = ['Sequence']
    peptides.to_csv('./Cache/peptides.csv', index=False, header=False)

    # 提取binary特征并保存
    binary_df = extract_binary_features(peptides)
    save_sparse_matrix(binary_df.iloc[:, 1:], './Cache/binary.npz')  # 不保存第一列

    # 提取cksaap特征并保存
    cksaap_df = extract_cksaap_features(peptides)
    save_sparse_matrix(cksaap_df.iloc[:, 1:], './Cache/cksaap.npz')  # 不保存第一列

    # 提取AAC特征并保存
    aac_df = extract_aac_features(peptides)
    save_sparse_matrix(aac_df, './Cache/aac.npz')

    # 处理测试数据（并不是训练时用的测试集，而是最后的预测集）
    pred_df = pd.read_csv("./Cache/to_predict_peptides.csv", header=None)
    pred_peptides = pred_df.iloc[:, 0]
    pred_peptides.column = ['Sequence']
    pred_peptides.to_csv('./Cache/pred_peptides.csv', index=False, header=False)

    # 提取binary特征并保存
    pred_binary_df = extract_binary_features(pred_peptides)
    save_sparse_matrix(pred_binary_df.iloc[:, 1:], './Cache/pred_binary.npz')  # 不保存第一列

    # 提取cksaap特征并保存
    pred_cksaap_df = extract_cksaap_features(pred_peptides)
    save_sparse_matrix(pred_cksaap_df.iloc[:, 1:], './Cache/pred_cksaap.npz')  # 不保存第一列

    # 提取AAC特征并保存
    pred_aac_df = extract_aac_features(pred_peptides)
    save_sparse_matrix(pred_aac_df, './Cache/pred_aac.npz')
