from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

sample = np.hstack((np.random.randn(30), np.random.randn(20)+5))
print(sample)
sample = np.sort(sample)
print(sample)
print(sample.shape)
density = stats.kde.gaussian_kde(sample)
fig, ax = plt.subplots(figsize=(8,4))

x = np.arange(-6,12,0.1)
ax.plot(x, density(x))

ax.plot(sample, [0.01]*len(sample), '|', color='k')
plt.show()