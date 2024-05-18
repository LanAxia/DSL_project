# import libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

import sklearn
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor


# 导入其他文件
from extract_features import load_features


# 获取cat最优参数
def kf_test(lr: float, iter: int, labels: np.array) -> float:
    kf = KFold(n_splits=5)
    kf_errors = []
    for train_id, test_id in tqdm(kf.split(features_x), desc="Finding best parameters", total=5):
        train_x, train_y = features_x[train_id], labels[train_id]
        test_x, test_y = features_x[test_id], labels[test_id]

        model = CatBoostRegressor(iterations=iter, depth=10, learning_rate=lr, loss_function="RMSE")

        model.fit(train_x, train_y, verbose=False)
        pred = model.predict(test_x)
        error = sklearn.metrics.mean_squared_error(test_y, pred)
        kf_errors.append(error)
    avg_kf_errors = np.average(kf_errors)
    return avg_kf_errors


# 测试cat在全部mmp上的表现
def kf_test_all(lr: float, iter: int, labels: np.array) -> np.ndarray:
    kf = KFold(n_splits=5)
    kf_errors = np.zeros((5, labels.shape[1]))
    for i, (train_id, test_id) in tqdm(enumerate(kf.split(features_x)), desc="Testing on all MMPs", total=5):
        for mmp_i in range(labels.shape[1]):
            train_x, train_y = features_x[train_id], labels[:, mmp_i][train_id]
            test_x, test_y = features_x[test_id], labels[:, mmp_i][test_id]

            model = CatBoostRegressor(iterations=iter, depth=10, learning_rate=lr, loss_function="RMSE")

            model.fit(train_x, train_y, verbose=False)
            pred = model.predict(test_x)
            error = sklearn.metrics.mean_squared_error(test_y, pred)
            kf_errors[i, mmp_i] = np.sqrt(error)
    return kf_errors


if __name__ == "__main__":
    # constant
    SAVE = True

    # import data
    data = pd.read_excel("./Data/peptides10.xlsx")  # load data
    data.iloc[:, 0] = data.iloc[:, 0].map(lambda x: x.strip()).map(lambda x: x[1:-1])  # 删除首尾的氨基酸和空格
    data = data.iloc[:, :-1]  # 删除最后一列

    # 得到氨基酸序列
    peptides = data.iloc[:, 0].values.tolist()  # 肽链的列表（字符串）

    mmp3_y = data.iloc[:, 3].values  # 得到mmp3的y

    # load extracted features
    features_x = load_features().iloc[:, 1:].values

    # 基于mmp3获取最优参数
    iterations = [500]
    lrs = [1, 0.1, 0.01, 0.001]
    errors = dict()
    for iteration in iterations:
        for lr in lrs:
            avg_error = kf_test(lr, iteration, mmp3_y)
            errors[(lr, iteration)] = avg_error

    best_param = sorted([(k, errors[k]) for k in errors], key=lambda x: x[1])
    best_lr, best_iter = best_param[0][0]
    print(best_lr, best_iter)  # 输出最优参数

    # best_lr, best_iter = 0.01, 1500  # 最优参数
    # mmp_labels = data.iloc[:, 1:].values
    # all_mmp_scores = kf_test_all(best_lr, best_iter, mmp_labels)
    #
    # # 保存实验结果
    # all_mmp_scores = pd.DataFrame(all_mmp_scores, columns=data.columns[1:])
    # all_mmp_scores.to_csv("./Cache/scores_xgb.csv", index=False)
