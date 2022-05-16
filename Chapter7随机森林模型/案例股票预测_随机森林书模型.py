import numpy as np
import pandas as pd
import tushare as ts
from sklearn.ensemble import RandomForestClassifier

# 1.股票基本数据获取
token = '7e0f9a0477a8b7c89481ba56e5c4b7e7667676b4ddf888143a9715d5'
pro = ts.pro_api(token)
df1 = pro.daily(ts_code='000001.SZ', start_date='20210501', end_date='20220501')

df1 = df1.set_index('trade_date')

print(df1.head(20))

# **2.简单衍生变量的计算**
df1['close-open'] = (df1['close'] - df1['open']) / df1['open']
df1['high-low'] = (df1['high'] - df1['low']) / df1['low']
df1['pre_close'] = df1['close'].shift(1)
df1['price_change'] = df1['close'] - df1['pre_close']
df1['p_change'] = (df1['close'] - df1['pre_close']) / df1['pre_close'] * 100

# 3.移动平均线相关数据构造
df1['MA5'] = df1['close'].rolling(5).mean()
df1['MA10'] = df1['close'].rolling(10).mean()
df1.dropna(inplace=True)  # 删除空值

# 4.通过Ta_lib库构造衍生变量
import talib

df1['RSI'] = talib.RSI(df1['close'], timeperiod=12)
df1['MOM'] = talib.MOM(df1['close'], timeperiod=5)
df1['EMA12'] = talib.EMA(df1['close'], timeperiod=12)
df1['EMA26'] = talib.EMA(df1['close'], timeperiod=26)
df1['MACD'], df1['MACDsignal'], df1['MACDhist'] = talib.MACD(df1['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df1.dropna(inplace=True)
df1.tail()

# df1.to_excel('data.xls', sheet_name='data2')

# 老一套

# 1、提取特征变量和目标变量
X = df1[['close', 'vol', 'close-open', 'MA5', 'MA10', 'high-low', 'RSI', 'MOM', 'EMA12', 'MACD', 'MACDsignal',
         'MACDhist']]
y = np.where(df1['price_change'].shift(-1) > 0, 1, -1)

# 2、划分训练集和测试集
X_length = X.shape[0]  # shape属性获取X的行数和列数，shape[0]即表示行数
split = int(X_length * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3、模型搭建
model = RandomForestClassifier(max_depth=3, n_estimators=10, min_samples_leaf=10, random_state=1)
model.fit(X_train, y_train)

# 4、模型使用与评估
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head(20))

y_pred_proba = model.predict_proba(X_test)
print(y_pred_proba)

# 变量特征重要性
print(model.feature_importances_)

features = X.columns
importances = model.feature_importances_
b = pd.DataFrame()
b['特征'] = features
b['特征重要性'] = importances
b = b.sort_values('特征重要性', ascending=False)
print(b)
