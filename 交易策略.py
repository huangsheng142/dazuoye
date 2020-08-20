# 交易策略
import pandas as pd

# 读入本地整合好的数据
df = pd.read_csv('D:/实际与预测组合数据表.csv')

# 设置初始资金money=100万，持股数investment=0，flag为买入卖出标志变量
money = 1000000
investment = 0
flag = 0

# 20日收盘价均线
df['predictValue_avg_5'] = (df['predictValue'] + df.shift(1)['predictValue']+df.shift(2)['predictValue']+df.shift(3)['predictValue']+df.shift(4)['predictValue']+df.shift(5)['predictValue']+df.shift(6)['predictValue']+df.shift(7)['predictValue']+df.shift(8)['predictValue']+df.shift(9)['predictValue']+df.shift(10)['predictValue']+df.shift(11)['predictValue']+df.shift(12)['predictValue']+df.shift(13)['predictValue']+df.shift(14)['predictValue']+df.shift(15)['predictValue']+df.shift(16)['predictValue']+df.shift(17)['predictValue']+df.shift(18)['predictValue']+df.shift(19)['predictValue']) / 20
df['predictValue_avg_5'] = df['predictValue_avg_5'].shift(-20)
df = df.dropna()
# 查看一下df
print(df.head())

# 初始化循环变量i
i = 0
close = 0
close_avg_5 = 0
count = 0

# 用for循环进行比较，高于日均线就卖，低于日均线就买
for i in range (552):
# buy
 if df.loc[i, 'predictValue'] < df.loc[i, 'predictValue_avg_5'] and flag == 0:
   investment = money/df.loc[i, 'close']
   money = 0
   close=df.shift(1).loc[i,'predictValue']
   close_avg_5 = df.shift(1).loc[i, 'predictValue_avg_5']
   flag = 1
   count+= 1
# sell
 elif df.loc[i, 'predictValue'] > df.loc[i, 'predictValue_avg_5'] and flag == 1:
    money = investment*df.loc[i, 'close']
    investment = 0
    close = df.shift(1).loc[i, 'predictValue']
    close_avg_5 = df.shift(1).loc[i, 'predictValue_avg_5']
    flag = 0
    count += 1

# 查看交易详情
print('投入资金:1000000', '\n目前资金:', money, '\n买入（卖出）次数：', count/2, '\n目前持股数量：', investment, '\n最后一次持股收盘价：', close, '\n持股资金：', close*investment)