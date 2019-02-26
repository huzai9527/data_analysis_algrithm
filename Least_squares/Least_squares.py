import numpy as np
from scipy.optimize import leastsq
import pandas as pd
from pandas import DataFrame
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def func(x, z, p):
    theta, x1, x2 = p
    return theta + x1*x + x2*z


def residuals(p, y, x, z):
    return y - func(x, z, p)

df = DataFrame(pd.read_excel("grades.xlsx"))
x1, x2 = df['eng'], df['c++']
X = np.array(x1)
Z = np.array(x2)
Y = np.array(df['total'])

p0 = [10, 1, 1]

plsq = leastsq(residuals, p0, args=(Y,X,Z))

print(plsq[0])
x, y, z = plsq[0]
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X, Y, Z, color="red", label="sample point", linewidths=3)
ax.plot(X, Z, func(X, Z, plsq[0]), label= "拟合数据")
ax.legend()
plt.show()