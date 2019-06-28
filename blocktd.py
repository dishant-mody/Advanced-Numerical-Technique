import numpy as np
import pandas as pd

x1 = 0
x2 = 1
y1_ = 0
y2_ = 1
h = 1/3

def thomas_(a,b,c,d):
    c_ = np.zeros(c.shape)
    d_ = np.zeros(d.shape)
    c_[0] = np.dot(np.linalg.inv(b[0]), c[0] )
    d_[0] = np.dot(np.linalg.inv(b[0]), d[0] )
    for i in range(1, c.shape[0]-1):
        c_[i] = np.dot( np.linalg.inv(b[i] - np.dot(a[i], c_[i-1])), c[i] )
    for i in range(1, d.shape[0]):
        d_[i] = np.dot( np.linalg.inv(b[i] - np.dot(a[i], c_[i-1])), (d[i] - np.dot(a[i], d_[i-1])))
    
    return [c_, d_]

def main_(h=1/3):
    n = int((x2-x1)/h) +1
    a = np.zeros((n-1,2,2))
    b = np.zeros((n-1,2,2))
    c = np.zeros((n-1,2,2))
    d = np.zeros((n-1,2))
    x_f = np.zeros((n-1))
    
    for i in range(0, n-1):
        x_f[i] = x1+(1+i)*h
        a[i][0][0] = -1
        a[i][0][1] = -1*h/2
        a[i][1][0] = 0
        a[i][1][1] = 1/(h*h) - 2/h
    for i in range(0, n-1):
        b[i][0][0] = 1
        b[i][0][1] = -1*h/2
        b[i][1][0] = -6
        b[i][1][1] = -2/(h*h) + 1
    for i in range(0, n-1):
        c[i][0][0] = 0
        c[i][0][1] = 0
        c[i][1][0] = 0
        c[i][1][1] = 1/(h*h) + 2/h
        
    d[0][0] = 0
    d[0][1] = 1
        
    for i in range(1,n-2):
        d[i][0] = 0
        d[i][1] = 1
    
    d[-1] = np.array([0, 1]) - np.dot(c[-1], np.array([0, 1]))
    c_,d_ = thomas_(a,b,c,d)
    res = np.zeros((n-1,2))
    res[-1] = d_[-1]

    for i in range(n-2):
        res[n-3-i] = d_[n-3-i] - np.dot(c_[n-3-i], res[n-2-i])
    
    return [res[:,0], x_f]

a_1, x_1 = main_(1/3)
a_2, x_2 = main_(0.1)
a_3, x_3 = main_(0.05)
    
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x_1, a_1, 'ro', label = 'h = 1/3')
plt.plot(x_2, (a_2), 'gx', label = 'h = 0.1')
plt.plot(x_3, (a_3), 'bo', label = 'h = 0.05')

plt.legend(loc='best')
plt.show()

#print(pd.DataFrame(np.column_stack((x_1, a_1)), columns=["x", "predicted"]))
#print(pd.DataFrame(np.column_stack((x_2, a_2)), columns=["x", "predicted"]))
#print(pd.DataFrame(np.column_stack((x_3, a_3)), columns=["x", "predicted"]))