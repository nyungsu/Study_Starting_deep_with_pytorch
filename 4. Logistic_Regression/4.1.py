import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+1)
y2 = sigmoid(x+2)
y3 = sigmoid(x+3)

plt.plot(x,y1,'r')
plt.plot(x,y2,'g')
plt.plot(x,y3,'b')
plt.plot([0,0],[1.0,0.0], ':')
plt.title('sigmoide function')
plt.show()

'''
sigmoid(x) 의 계수가 커질수록 기울기가 가팔라진다
'''

