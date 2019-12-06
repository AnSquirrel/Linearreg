
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

s = [[40], [50], [70], [80], [90], [100], [115], [120], [130], [135], [150], [165], [180], [185], [200], [210], [225], [240]]
space = np.array(s)

p = [170, 220, 380, 400, 420, 490, 550, 630, 690, 725, 900, 962, 1000, 1210, 1300, 1310, 1395, 1430]
price = np.array(p)

s_train, s_test, p_train, p_test = train_test_split(space, price, random_state=10)

print('Train And Test Space:\n', s_train, 2 * '\n', s_test, '\n')
print('Train And Test Price:\n', p_train, 2 * '\n', p_test, '\n')

linearreg = LinearRegression()
print('Linear Model:\n', linearreg)

linearreg.fit(s_train, p_train)
print('Linearreg Coef:\n', linearreg.coef_)
print('Linearreg Intercept:\n',linearreg.intercept_)

p_predict = linearreg.predict(s_test)
print('Predict Price:\n', p_predict)

plt.figure(figsize=(8, 6))
plt.scatter(s, p, color='k', s=80)
plt.plot(s, s * linearreg.coef_ + linearreg.intercept_, color='r', label='LinearRegression', linewidth=2)
plt.legend()
plt.show()

# s_test = s_test[np.lexsort(s_test[:,::-1].T)]
s_test = np.sort(s_test, axis=0)
p_test = np.sort(p_test, axis=0)
p_predict = np.sort(p_predict, axis=0)
print('S And P Test Sorted:\n', s_test.T, 2 * '\n', p_test)

plt.figure(figsize=(8, 6))
plt.plot(s_test, p_test, color='k', label='P Test', linewidth=2)
plt.plot(s_test, p_predict, color='r', label='P Predict', linewidth=2)
plt.legend()
plt.show()
