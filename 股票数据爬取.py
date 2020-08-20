import tushare as ts

# 初始化
ts.set_token('32b8468e7397e72da67b1acfe011701bfeffba30aaa26652053703d6')
pro = ts.pro_api('32b8468e7397e72da67b1acfe011701bfeffba30aaa26652053703d6')

# 查看青海啤酒(600600）指定时间内k线数据
print(ts.get_hist_data('600600',start='2011-01-01',end='2018-07-01'))
df1=ts.get_hist_data('600600',start='2011-01-01',end='2018-07-01')
print(ts.get_hist_data('600600',start='2018-07-01',end='2020-07-01'))
df2=ts.get_hist_data('600600',start='2018-07-01',end='2020-07-01')

# 查看赋值是否成功
print(df1.head())
print(df2.head())

# 将股票数据存为csv文件
df1.to_csv('D:/青海啤酒股票k线数据3.csv')
df2.to_csv('D:/青海啤酒股票k线数据4.csv')
