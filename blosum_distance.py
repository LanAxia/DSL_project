import numpy as np
import pandas as pd
import pickle as pkl

from tqdm import tqdm
import blosum as bl

SAVE = False


data = pd.read_csv("./Data/Substrate_search_processed.csv", sep='\t', header=None, encoding='utf-8')

if SAVE is True:
    data.to_csv("./Data/Substrate_search_processed.csv", sep='\t', index=False, header=False, encoding="utf-8")
data.head(5)

amino_table = pd.read_csv("./Data/amino_table.csv", sep="\t", header=None)
amino_table.columns = ["chinese", "english", "one_abbr", "three_abbr"]
amino_three2one, amino_one2three = dict(), dict()
for row_i, row in amino_table.iterrows():
    three_abbr = row["three_abbr"].lower()
    one_abbr = row["one_abbr"].lower()
    amino_three2one[three_abbr] = one_abbr
    amino_one2three[one_abbr] = three_abbr

# 只保留蛋白酶名和肽链信息
data = pd.read_csv("./Data/Substrate_search_processed.csv", sep='\t', header=None, encoding='utf-8')

"""protease_peptide = pd.concat((data[[1]], data.iloc[:, 4:12]), axis=1)  # 拼接蛋白酶和肽链信息
protease_peptide = protease_peptide.dropna()  # 删除有nan的行
nan_row_ids = set()  # 获取nan行的索引，部分数据有"NAN"的值，只能手动删除
for row_i, row in protease_peptide.iterrows():
    if "NAN" in row.tolist():
        nan_row_ids |= {row_i}
protease_peptide = protease_peptide.drop(list(nan_row_ids))  # 删除有nan行

protease_peptide.columns = ["protease"] + [i for i in range(8)]  # 修改列名
protease_peptide.iloc[:, 1:] = protease_peptide.iloc[:, 1:].map(lambda x: x.lower())  # 将氨基酸转为小写
protease_peptide.iloc[:, 1:] = protease_peptide.iloc[:, 1:].map(lambda x: "-" if "-" in x else x)  # 将"/-/"转为"-"

# 筛选人体蛋白酶
human_protease = pd.read_csv("./Data/human_protease.txt", sep="\t")
human_protease = set(human_protease["MEROPS ID"].tolist())
human_animos = set(bl.BLOSUM(62).keys())
protease_peptide = protease_peptide[protease_peptide["protease"].isin(human_protease)]

# 筛选只含人体内氨基酸的肽链
valid_row_ids = []
for row_i, row in protease_peptide.iterrows():
    if len(set(row.iloc[1:].map(lambda x: x.lower()).tolist()) - (amino_three2one.keys() | {"-"})) == 0:
        valid_row_ids.append(row_i)
protease_peptide = protease_peptide.loc[valid_row_ids]
"""

protease_peptide = pd.read_csv("./Data/Protease_Peptides.csv", sep="\t")

if SAVE is True:
    protease_peptide.to_csv("./Data/Protease_Peptides.csv", sep='\t', header=True, index=False)  # 保存数据
protease_peptide.head(5)

bl62 = bl.BLOSUM(62)
# 生成相似肽链

# 比较两条肽链之间的相似性
def compare_peptides(pep_source: tuple, pep_target: tuple) -> int:
    # 通过比较替换前和替换后的分数差来比较两条肽链的相似性
    score = 0  # initialize score
    for pep_source_amino, pep_target_amino in zip(pep_source, pep_target):
        # 判断是否为"-":
        if (pep_source_amino == "-") ^ (pep_target_amino == "-"):  # 如果两个肽链有一位是"-"但是另一个不是，则认为距离为负无穷
            valid_amino = list({pep_source_amino, pep_target_amino} - {"-"})[0]
            valid_amino_abbr = amino_three2one[valid_amino].upper()
            score += 0
        else:
            if pep_source_amino == "-":  # 如果两者都是"-"，则分数不变
                pass
            else:
                pep_source_amino_abbr, pep_target_amino_abbr = amino_three2one[pep_source_amino].upper(), amino_three2one[pep_target_amino].upper()  # turn into abbr
                replace_score = bl62[pep_source_amino_abbr][pep_target_amino_abbr]   # 获取原始分数
                score += replace_score
    return score


MMP3_protease = "M10.005"

# 构建从protease到肽链的映射
proteases = set(protease_peptide["protease"].tolist())
protease2peptides = dict()
for protease in proteases:
    peptides = protease_peptide[protease_peptide["protease"] == protease].iloc[:, 1:].values.tolist()
    peptides = {tuple(x) for x in peptides}
    protease2peptides[protease] = peptides

def search_unique_peptides(protease: str, protease2peptides: dict = protease2peptides) -> set:
    peptides = protease2peptides[protease].copy()  # 必须要复制，不然会改变字典内部的值
    for p in protease2peptides.keys():
        if p != protease:
            peptides -= protease2peptides[p]
    return peptides

len(search_unique_peptides(MMP3_protease))

other_peptides = set()
for protease in protease2peptides.keys():
    if protease != MMP3_protease:
        other_peptides |= protease2peptides[protease]

mmp3_out_scores = dict()

for mmp3_peptide in tqdm(search_unique_peptides(MMP3_protease)):
    for peptide in other_peptides:
        score = compare_peptides(mmp3_peptide, peptide)
        mmp3_out_scores[(mmp3_peptide, peptide)] = score

"""error_amino = set()
for row_i, row in protease_peptide.iterrows():
    protease_name = row["protease"]
    peptides = row.iloc[1:]
    for peptide in peptides:
        if peptide not in amino_three2one and peptide != "-":
            error_amino |= {peptide}
            print("Error: {} row, {}".format(row_i, peptide))
"""
a = 1