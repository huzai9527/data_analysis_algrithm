
import itertools
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel('data/date.xlsx', index_col='date', parse_dates=['date'])
sub = df['1984':'2100']

print("======")
print(sub)
train = sub.loc['1984':'2000']
test = sub.loc['2001':'2100']
print("======")
print(train)
plt.plot(train)
plt.show()

## 差分法，使得数据更加平稳
df['Close_diff_1'] = df['data'].diff(1)
df['Close_diff_2'] = df['Close_diff_1'].diff(1)
print("+++++++++++")
print(df['Close_diff_1'].fillna(0))
fig = plt.figure(figsize=(20, 6))
ax1 = fig.add_subplot(131)
ax1.plot(df['data'])
ax2 = fig.add_subplot(132)
ax2.plot(df['Close_diff_1'])
ax3 = fig.add_subplot(133)
ax3.plot(df['Close_diff_2'])
# df['data'] = df['Close_diff_1']
plt.show()

# 遍历，寻找适宜的参数
p_min = 0
d_min = 0
q_min = 0
p_max = 5
d_max = 0
q_max = 5
# Initialize a DataFrame to store the results,，以BIC准则
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                           columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])

for p, d, q in itertools.product(range(p_min, p_max + 1),
                                 range(d_min, d_max + 1),
                                 range(q_min, q_max + 1)):
    if p == 0 and d == 0 and q == 0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue

    try:
        model = sm.tsa.ARIMA(train, order=(p, d, q), #train 数据
                             # enforce_stationarity=False,
                             # enforce_invertibility=False,
                             )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(results_bic,
                 mask=results_bic.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f',
                 )
ax.set_title('BIC')
plt.show()

# 拖尾和截尾
import statsmodels.api as sm

fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train,  ax=ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
plt.show()



# 模型的检验，残差序列的随机性可以通过自相关函数法来检验，即做残差的自相关函数图：
# model = sm.tsa.ARIMA(train, order=(1, 0, 1))
# results = model.fit()
# resid = results.resid #赋值
# fig = plt.figure(figsize=(12, 8))
# fig = sm.graphics.tsa.plot_acf(resid.values.squeeze())
# plt.show()

# sub = df['Close_diff_1'].fillna(0)
##模型的预测
model = sm.tsa.ARIMA(sub, order=(1, 1, 1))
results = model.fit()
predict_sunspots = results.predict()
print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
ax = sub.plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()

results.forecast()[0]