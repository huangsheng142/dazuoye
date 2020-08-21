# 股票买卖预测代码

气卓1801班-U201815324-陈钰溥

## 一、简介

为完成最终大项目，我们小组根据老师要求进行分工合作，通过学习老师的课件以及上网进行探索学习，完成了老师布置的大作业。
但是我在安装Tensorflow库时遇到了很大的困难，反复报错，在尽所能还是无法顺利运行的情况下，由于时间所剩不多，安装成功无望，只能放弃，转而寻求别的方法编写交易策略，完成后再和组员进行参照对比。
最终我通过依靠较为科学的MACD因子的判断方法进行交易策略代码的编写并成功运行，获得输出。

## 二、具体代码

```python
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
```

## 三、具体说明

首先需要说明的是，由于Tensorflow库安装失败，所以我无法通过编写神经网络学习来进行股票价格的预测，转而选择用金融因子来进行买卖的预测分析。在小组分工中的定位是作为其他方案与我们的主方案作对比参考。

此外，在代码中我选取了2018年全年青岛啤酒的股票数据进行分析，通过相关因子判断买卖，但并不会因为是2018年的已知数据而出现“肯定盈利”的情况（虽然事实上通过科学的因子分析，不出现意外的前提下的确能保证获利）。

1. 导入必要的库和数据。代码包含了numpy、pandas、matplotlib、talib以及数据获取所需的库jqdatasdk。程序通过聚宽网导入青岛啤酒（600600）2018年股票数据，且为了程序的简介可行，我只保留了每日收盘价数据进行分析

```python
price = get_price("600600.XSHG", start_date="2018-01-01", end_date="2018-12-31", frequency="daily", fields=['close'])['close']
```

2. 画出K线图，计算每日收益率。画K线的目的在于便于在后面和交易线作对比，而每日收益率用于计算初始投资100万元条件下的最终收益。

```python
ret = price.pct_change()
plt.figure(figsize=(18,8))
ax1 = plt.subplot(2,1,1)
ax1.plot(price)
ax2 = plt.subplot(2,1,2)
ax2.plot(ret)
plt.show()
```

![K线及每日收益率](https://github.com/lahuan3369/MyPictures/blob/master/jiaoyi_1.png)

3. 判断买入信号。根据不同指标（MACD、EMA12、EMA26）的组合，确定了3种买卖信号。其中sig1只考虑HIST指标，HIST转正时开仓买入，转负时清仓；sig2同时考虑HIST指标和DEA指标，只有当HIST转正，且DEA在0以上时，才开仓买入，任何一个指标变负即清仓；sig3同时考虑HIST和EMA指标，只有当HIST为正，而且当前价格在慢线（26日指数加权平均价）上方时，才开仓买入，任何一个指标转负即清仓。信号的值只有0和1，0代表当天不买入并确保清仓，1代表当天全部买入并持股。对于不同的市场环境，应当选取不同的信号。

```python
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
```

![判断不同种信号](https://github.com/lahuan3369/MyPictures/blob/master/jiaoyi_2.png)

4. 选取信号进行实测。根据相关论文资料，sig2更加科学合理，则用sig2的数组与每日收益率对应相乘，再通过函数cumprod()进行累计连乘，得到收益率曲线，结合100万元的初始资金，最后结果为年获利约154，168元。

```python
sig2_lag = sig2.shift(1).fillna(0).astype(int)
sig2_ret = sig2_lag*ret
cum_sig2_ret = (1+sig2_ret).cumprod()
price_norm = price/price[0]
plt.figure(figsize=(18,8))
plt.plot(price_norm)
plt.plot(cum_sig2_ret)
plt.legend(["benchmark", "strategy cumulative return"], loc="upper left")
plt.show()

zuizhong=cum_sig2_ret[-1]*1000000
print("initial values:1000,000   now:" + str(zuizhong))
```

![收益率折线图](https://github.com/lahuan3369/MyPictures/blob/master/jiaoyi_3.png)

![最终收益](https://github.com/lahuan3369/MyPictures/blob/master/jiaoyi_4.png)