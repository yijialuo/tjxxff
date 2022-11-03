import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# 读取费收数据
df = pd.read_csv("tlf.csv")
# 获取日期列
dates = np.array(df['date'])

# 将日期变为标准yyyy-mm-dd
year_month_day = []
for date in np.array(df['date']):
    # 月份有2位
    date_seq = date.split('/')
    year = date_seq[0]
    month = date_seq[1]
    day = date_seq[2].split(" ")[0]
    if len(month) == 1:
        month = "0" + month
    if len(day) == 1:
        day = "0" + day
    year_month_day.append(year+month+day)
# 给数据添加清洗后的日期
df['year_month_day'] = year_month_day

# 10,9,8月的索引
find_10index=[]
find_9index=[]
find_8index=[]

# 找到10、9、8月的索引
for inx,date in enumerate(dates):
    month=date.split("/")[1]
    if month=='10':
        find_10index.append(inx)
    elif month=='8':
        find_8index.append(inx)
    elif month=='9':
        find_9index.append(inx)
# 8,9的数据
df9=df.iloc[find_9index]
df8=df.iloc[find_8index]
# 删除10月的数据，inplace为True表示直接在原来数据上删除
df.drop(find_10index,inplace=True)
# 删除多余列
df.drop(labels=['date','tugboat','customer'],axis=1,inplace=True)
df9.drop(labels=['date','tugboat','customer'],axis=1,inplace=True)
df8.drop(labels=['date','tugboat','customer'],axis=1,inplace=True)
# 按照日期排序,inplace=True原来数组排序
df.sort_values(by='year_month_day',inplace=True)
df9.sort_values(by='year_month_day',inplace=True)
df8.sort_values(by='year_month_day',inplace=True)

# 按照公司分组
for i, v in df.groupby(['company']):
    if i == 'GZGBRTB':
        date_fee_com1 = v
    else:
        date_fee_com2 = v

for i, v in df8.groupby(['company']):
    if i == 'GZGBRTB':
        date8_fee_com1 = v
    else:
        date8_fee_com2 = v
        
for i, v in df9.groupby(['company']):
    if i == 'GZGBRTB':
        date9_fee_com1 = v
    else:
        date9_fee_com2 = v

print(date8_fee_com1)
print(date8_fee_com2)
print(date9_fee_com1)
print(date9_fee_com2)

# 分组，统计每天的费用
date_fee_all = df.groupby(['year_month_day']).sum(['fee'])
date_fee_com1 = date_fee_com1.groupby(['year_month_day']).sum(['fee'])
date_fee_com2 = date_fee_com2.groupby(['year_month_day']).sum(['fee'])

# 分组后，列索引变平，变成一个正常索引df数据
year_month_fee = date_fee_all.reset_index(['year_month_day'])
date_fee_com1 = date_fee_com1.reset_index(['year_month_day'])
date_fee_com2 = date_fee_com2.reset_index(['year_month_day'])

# ==========================下面是8,9月========================

date8_fee_all = df8.groupby(['year_month_day']).sum(['fee'])
date8_fee_com1 = date8_fee_com1.groupby(['year_month_day']).sum(['fee'])
date8_fee_com2 = date8_fee_com2.groupby(['year_month_day']).sum(['fee'])

# 分组后，列索引变平，变成一个正常索引df数据
year_month8_fee = date8_fee_all.reset_index(['year_month_day'])
date8_fee_com1 = date8_fee_com1.reset_index(['year_month_day'])
date8_fee_com2 = date8_fee_com2.reset_index(['year_month_day'])

date9_fee_all = df9.groupby(['year_month_day']).sum(['fee'])
date9_fee_com1 = date9_fee_com1.groupby(['year_month_day']).sum(['fee'])
date9_fee_com2 = date9_fee_com2.groupby(['year_month_day']).sum(['fee'])

# 分组后，列索引变平，变成一个正常索引df数据
year_month9_fee = date9_fee_all.reset_index(['year_month_day'])
date9_fee_com1 = date9_fee_com1.reset_index(['year_month_day'])
date9_fee_com2 = date9_fee_com2.reset_index(['year_month_day'])

print("============"*100)
print(year_month8_fee)
print("============"*100)
print(date8_fee_com1)
print("============"*100)
print(date8_fee_com2)
print("============"*100)
print(year_month9_fee)
print("============"*100)
print(date9_fee_com1)
print("============"*100)
print(date9_fee_com2)
print("============"*100)
#
# X_all = np.array(year_month_fee['year_month_day'])
# Y_all = np.array(year_month_fee['fee'])
#
# X_com1 = np.array(date_fee_com1['year_month_day'])
# Y_com1 = np.array(date_fee_com1['fee'])
#
# X_com2 = np.array(date_fee_com2['year_month_day'])
# Y_com2 = np.array(date_fee_com2['fee'])
#
# fig, axes = plt.subplots(3, 1)
#
# polt_data_all = pd.Series(Y_all, index=list(X_all))
# polt_data_com1 = pd.Series(Y_com1, index=list(X_com1))
# polt_data_com2 = pd.Series(Y_com2, index=list(X_com2))
#
#
# polt_data_all.plot(ax=axes[0], color='k', alpha=0.7)
# polt_data_com1.plot(ax=axes[1], color='b', alpha=0.7)
# polt_data_com2.plot(ax=axes[2], color='g', alpha=0.7)
# plt.show()
# #
