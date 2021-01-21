import stat

import pandas as pd
import numpy as np
import scipy.stats as scs
import statistics as st
import matplotlib.pyplot as plt

d = {'value': [1, 2, 3, 4, 5, 6],
     'probability': [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]}
df = pd.DataFrame(data=d)


#1
print('max', np.max(df['value']))
print('min', np.min(df['value']))
print("std", np.std(df['value']))
print('mean', np.mean(df['value']))

#2

probka_rozkladu_bernoulliego = scs.bernoulli.rvs(0.3, size=100)
probka_rozkladu_dwumianowego = scs.binom.rvs(1, 0.3, size=100)
probka_rozkladu_poissona = scs.poisson.rvs(0.3, size=100)

#3

print('Bernoulli mean', np.mean(probka_rozkladu_bernoulliego))
print('Binom mean', np.mean(probka_rozkladu_dwumianowego))
print('Poisson mean', np.mean(probka_rozkladu_poissona))

print('\nBernoulli variance', st.variance(probka_rozkladu_bernoulliego))
print('Binom variance', st.variance(probka_rozkladu_dwumianowego))
print('Poisson variance', st.variance(probka_rozkladu_poissona))

print('\nBernoulli kurtosis', scs.kurtosis(probka_rozkladu_bernoulliego))
print('Binom kurtosis', scs.kurtosis(probka_rozkladu_dwumianowego))
print('Poisson kurtosis', scs.kurtosis(probka_rozkladu_poissona))

print('\nBernoulli skewnes', scs.skew(probka_rozkladu_bernoulliego))
print('Binom skewnes', scs.skew(probka_rozkladu_dwumianowego))
print('Poisson skewnes', scs.skew(probka_rozkladu_poissona))

#4

X = np.arange(0, 100)
plt.plot(X, probka_rozkladu_bernoulliego)
plt.plot(X, probka_rozkladu_dwumianowego)
plt.plot(X, probka_rozkladu_poissona)
plt.show()

#5

dwum = scs.binom.pmf(20, 20, 0.4)
print(np.sum(dwum))
print(dwum)

#6

nor = scs.norm.rvs(0, 2, 100)
print('max', np.max(nor))
print('min', np.min(nor))
print("std", np.std(nor))
print('mean', np.mean(nor))