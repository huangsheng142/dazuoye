import numpy as np
import pandas as pd
import talib as ta 
from jqdatasdk import *
auth("15161023933", "15161023933Wsx")
import matplotlib.pyplot as plt

# 以2018年青岛啤酒（600600）股票数据为测试实例，频率为每天，只要收盘价
price = get_price("600600.XSHG", start_date="2018-01-01", end_date="2018-12-31", frequency="daily", fields=['close'])['close']

# 画出K线图
ret = price.pct_change()
plt.figure(figsize=(18,8))
ax1 = plt.subplot(2,1,1)
ax1.plot(price)
ax2 = plt.subplot(2,1,2)
ax2.plot(ret)
plt.show()

# 用talib库中的相应函数计算MACD指标及EMA12和EMA26
dif, dea, hist = ta.MACD(price)
ema12 = ta.EMA(price, 12)
ema26 = ta.EMA(price, 26)

# 用指标判断买卖信号
sig1 = (hist>0)
sig2 = (hist>0) & (dea>0)
sig3 = (hist>0) & (price>ema26)
plt.figure(figsize=(18,12))
ax1=plt.subplot(4,1,1)
ax1.plot(price)
ax2=plt.subplot(4,1,2)
ax2.bar(x=sig1.index, height=sig1.values)
ax3=plt.subplot(4,1,3)
ax3.bar(x=sig2.index, height=sig2.values)
ax4=plt.subplot(4,1,4)
ax4.bar(x=sig3.index, height=sig3.values)
plt.show()

# 按照文献推荐确定的信号进行买卖并与股价进行比较
sig2_lag = sig2.shift(1).fillna(0).astype(int)
sig2_ret = sig2_lag*ret
cum_sig2_ret = (1+sig2_ret).cumprod()
price_norm = price/price[0]
plt.figure(figsize=(18,8))
plt.plot(price_norm)
plt.plot(cum_sig2_ret)
plt.legend(["benchmark", "strategy cumulative return"], loc="upper left")
plt.show()

# 以100万元作初始资金，
zuizhong=cum_sig2_ret[-1]*1000000
print("initial values:1000,000   now:" + str(zuizhong))
