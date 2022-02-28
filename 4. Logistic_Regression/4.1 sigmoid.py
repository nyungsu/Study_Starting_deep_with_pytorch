'''
sigmoid(x)
계수가 커질수록 기울기가 가팔라지고
+이면 왼쪽으로
-이면 오른쪽으로
'''

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)

y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

# y1 = sigmoid(x+1)
# y2 = sigmoid(x+2)
# y3 = sigmoid(x+3)

plt.plot(x,y1,'r')
plt.plot(x,y2,'g')
plt.plot(x,y3,'b')

plt.plot([0,0],[0.0,1.0], ':')
# (0,0)과 (0,1)을 잇는다

plt.title('sigmoide function')
plt.show()



