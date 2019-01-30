import itertools
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
df = pd.read_excel('data/va.xlsx', index_col='date', parse_dates=['date'])
sub = df['2011':'2017']


train = sub.loc['2011':'2015']
test = sub.loc['2016':'2017']
import matplotlib.pyplot as plt
model = sm.tsa.ARIMA(sub, order=[0, 0, 1])
results = model.fit()
predict_sunspots = results.predict('2011','2028')
print(predict_sunspots)
fig, ax = plt.subplots(figsize=(12, 8))
ax = df['data'].plot(ax=ax)
predict_sunspots.plot(ax=ax)
print(predict_sunspots)
plt.show()