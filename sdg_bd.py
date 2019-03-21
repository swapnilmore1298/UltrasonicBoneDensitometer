import numpy as np
import math
from scipy import stats
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def estimate_coef(x, y):
    n = np.size(x)
    m_x, m_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y*x - n*m_y*m_x)
    SS_xx = np.sum(x*x - n*m_x*m_x)
    b1 = SS_xy / SS_xx
    b0 = m_y - b1*m_x
    return(b0, b1)

def Train():
    x = np.array([0, 0.4, 0.4, 0.5, 0.6, 1, 1.1, 1.2, 1.3, 1.5, 1.7, 1.8, 1.8, 1.9,
                  1.9, 2, 2.1, 2.2, 2.4, 2.5, 2.7, 3, 3.2, 3.4, 3.5, 3.5, 3.7, 3.8, 4, 4])
    y = np.array([0, 0.42, 0.31, 0.4, 0.5, 0.6, 0.62, 0.63, 0.7, 0.97, 0.98, 0.96, 0.96, 0.8,
                  0.13, 1, 1.09, 1.25, 1.25, 1.4, 1.34, 1.6, 1.70, 1.81, 1.8, 1.22, 2.01, 2.07, 2, 2.3])
    b0, b1 = estimate_coef(x, y)
    return b0, b1

def Predict(NTD):
    BoneDensity = r0 + r1 * NTD
    print("The estimated value of Predicted Bone Density = " + str(BoneDensity))
    return BoneDensity

r0, r1 = Train()

x = np.array([0, 0.4, 0.4, 0.5, 0.6, 1, 1.1, 1.2, 1.3, 1.5, 1.7, 1.8, 1.8, 1.9,
              1.9, 2, 2.1, 2.2, 2.4, 2.5, 2.7, 3, 3.2, 3.4, 3.5, 3.5, 3.7, 3.8, 4, 4])
y = np.array([0, 0.42, 0.31, 0.4, 0.5, 0.6, 0.62, 0.63, 0.7, 0.97, 0.98, 0.96, 0.96, 0.8,
              0.83, 1, 1.09, 1.25, 1.25, 1.4, 1.34, 1.6, 1.70, 1.81, 1.8, 1.22, 2.01, 2.07, 2, 2.3])

plt.xlabel('Net-Time Delay')
plt.ylabel('Bone Density (g/cm²)')
plt.title("Co-relation Between NDT and Bone Density",
          loc='left', fontsize=18, fontweight=4, color='Black')
plt.plot(x, y, 'ro', color='green')
plt.axis([0, 6, 0, 3])

fit = np.polyfit(x, y, 1)
fit_fn = np.poly1d(fit)
plt.plot(x, y, 'yo', x, fit_fn(x), color='orange', alpha=0.3)
plt.xlim(0, 6)
plt.ylim(0, 3)
plt.show()

Scanned_Data = np.array([round(float(i)) for i in range(1, len(sys.argv))])
NTD = np.mean(Scanned_Data)
std_dev = np.std(Scanned_Data)
sample_size = Scanned_Data.size
BD = Predict(NTD)

popu_mean, std, N = 1.278, 0.139, 1000
s = pd.Series(np.random.normal(popu_mean, std, N))
p = s.plot(kind='hist', bins=50, color='orange', label='Population')

plt.xlabel('density in g/cm²')
plt.ylabel('frequency')
plt.title("Bone Density Categorical", loc='left',
          fontsize=18, fontweight=4, color='Black')
bar_value_to_label = BD
min_distance = float("inf")  # initialize min_distance with infinity
index_of_bar_to_label = 0
for i, rectangle in enumerate(p.patches):  # iterate over every bar
    tmp = abs(  # tmp = distance from middle of the bar to bar_value_to_label
        (rectangle.get_x() +
            (rectangle.get_width() * (1 / 2))) - bar_value_to_label)
    if tmp < min_distance:  # we are searching for the bar with x cordinate
                            # closest to bar_value_to_label
        min_distance = tmp
        index_of_bar_to_label = i
p.patches[index_of_bar_to_label].set_color('b')
blue_patch = mpatches.Patch(color='blue', label='Patient')
plt.legend(handles=[blue_patch])
plt.show()

print("The mean of the scanned sample set = " + str(BD))
print("The mean of the Whole population = " + str(popu_mean))

t_statistic = (BD - popu_mean)/(std_dev/math.sqrt(sample_size-1))

print("t-score = " + str(t_statistic))

if (t_statistic > -1):
    print("Yout t-score is above -1,you have Normal Bone Density")
elif(t_statistic < -2.5):
    print("Your t-score is below -2.5, you have Osteoporosis")
else:
    print("Your t-score is between -2.5 and -1, you have Low Bone Density")
