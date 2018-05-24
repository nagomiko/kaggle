import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# テストデータと訓練データ読み込み
train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')
# NaNを全部0.0に変換
train_data = train_data.fillna(0.0)
test_data = test_data.fillna(0.0)


def normalize(x):
    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    return x


# とりあえず数字のやつだけ全部とる
x = train_data[
    ['LotArea', 'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
     'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
     'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
     'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]
test_x = test_data[
    ['LotArea', 'MSSubClass', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
     'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
     'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
     'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
     'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]
y = train_data['SalePrice']

x = normalize(x)
test_x = normalize(test_x)

# Ridge　alphaは0.1がいい感じだった
rdg = Ridge(alpha=0.1)
# 二次で
phi_x = np.hstack([x, x * x])
rdg.fit(phi_x, y)
test_phi_x = np.hstack([test_x, test_x * test_x])
# テストデータに適用
y_test_pred = rdg.predict(test_x)
# 書き出し
df_out = pd.read_csv("./input/test.csv")
df_out["SalePrice"] = y_test_pred

# outputディレクトリに出力する
df_out[["Id", "SalePrice"]].to_csv("./output/submission2.csv", index=False)
