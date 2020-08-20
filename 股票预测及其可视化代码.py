# 导入pandas等库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# 线性回归库
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 导入数据，输出前几行看一下情况
data = pd.read_csv('D:/青海啤酒股票k线数据.csv')
df = data[['open', 'high', 'low', 'volume', 'close']]
print(df.head())

# 划分特征值和目标值
featureDatas = df[['open', 'high', 'low', 'volume']]
feature = featureDatas.values
target = np.array(df['close'])

# 划分训练集和测试集
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.2)

lrtoot = LinearRegression()  # 创建线性回归对象
lrtoot.fit(feature_train, target_train)  # 训练
# 用测试集预测结果
predictByTest = lrtoot.predict(feature_test)
predictDays = int(math.ceil(0.2 * len(df)))  # 预测的天数，这里设置20%是因为股票数据大部分是2018年到2020年的


# 在前80%的交易日中，设置预测结果和收盘价一致
index = 0
while index < len(data) - predictDays:
    df.loc[index, 'predictValue'] = data.loc[index, 'close']  # 把训练集部分的预测股价设置成收盘价
    df.loc[index, 'date'] = data.loc[index, 'date']  # 训练集部分的日期
    index = index + 1

# 在后20%的交易日中，用测试集推算预测股价
predictedCnt = 0
while predictedCnt < predictDays:
    df.loc[index, 'predictValue'] = predictByTest[predictedCnt]  # 把df中表示测试结果的predictedVal列设置成相应的预测结果
    df.loc[index, 'date'] = data.loc[index, 'date']  # 逐行设置了每条记录中的日期
    predictedCnt = predictedCnt + 1
    index = index + 1


# 可视化
plt.figure(figsize=(10, 5))
df['predictValue'].plot(color='red', label='predict data', fontsize=15)
df['close'].plot(color='blue', label='real data', fontsize=5)
plt.legend(loc='best', fontsize=15)  # 绘制图例

# 设置x坐标的标签
major_index = df.index[df.index % 30 == 0]
major_xtics = df['date'][df.index % 30 == 0]
plt.xticks(major_index, major_xtics)
plt.setp(plt.gca().get_xticklabels(), rotation=10)

# 带网格线，且设置了网格样式
plt.grid(linestyle='-.')
plt.show()


# 把预测值单列在原数据表中
df.to_csv('D:/实际与预测组合数据表.csv')
