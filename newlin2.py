import numpy as np
import pandas as pd
x1 = 0
x2 = 10
y1_ = 0
y2_ = 1
h = 1
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

def main_(n=3):
    e = 0.001
    h = (x2-x1)/n
    a = np.zeros((n-1,2,2))
    b = np.zeros((n-1,2,2))
    c = np.zeros((n-1,2,2))
    d = np.zeros((n-1,2))
    x_f = np.zeros((n+1))
    f = np.zeros(n+1)
    F = np.zeros(n+1)
    for i in range(n+1):
        x_f[i] = x1 + i*h
        f[i] = 101*x_f[i]*x_f[i]/20 - x_f[i]*x_f[i]*x_f[i]/3
        F[i] = x_f[i]*(10.1 - x_f[i])
        
    res_final = np.zeros(n-1)
    flag=20
    while flag!=1:
        for i in range(0, n-1):
            a[i][0][0] = -1
            a[i][0][1] = -1*h/2
            a[i][1][0] = 0
            a[i][1][1] = 1/(h*h) - f[i+1]/(2*h)
        for i in range(0, n-1):
            b[i][0][0] = 1
            b[i][0][1] = -1*h/2
            b[i][1][0] = (F[i+2]-F[i])/(2*h)
            b[i][1][1] = -2/(h*h) - 2*F[i+1]
        for i in range(0, n-1):
            c[i][0][0] = 0
            c[i][0][1] = 0
            c[i][1][0] = 0
            c[i][1][1] = 1/(h*h) + f[i+1]/(2*h)
        for i in range(0, n-1):
            d[i][0] = f[i]-f[i+1]+(h/2)*(F[i+1]+F[i])
            d[i][1] = -1 + F[i+1]*F[i+1] - (F[i+2]-2*F[i+1]+F[i])/(h*h) - f[i+1]*(F[i+2]-F[i])/(2*h)
        
        c_,d_ = thomas_(a,b,c,d)
        res = np.zeros((n-1,2))
        res[-1] = d_[-1]
        for i in range(n-2):
            res[n-3-i] = d_[n-3-i] - np.dot(c_[n-3-i], res[n-2-i])
        f[1:-1] = f[1:-1] + res[:,0]
        F[1:-1] = F[1:-1] + res[:,1]

        flag = flag -1
        
    return [f[:-1], x_f[:-1]]

a_1, x_1 = main_(3)
a_2, x_2 = main_(5)
a_3, x_3 = main_(9)
a_4, x_4 = main_(19)

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x_1, a_1, 'ro', label = 'h = 1/3')
plt.plot(x_2, (a_2), 'gx', label = 'h = 0.1')
plt.plot(x_3, (a_3), 'bo', label = 'h = 0.05')
plt.plot(x_4, (a_4), 'y-', label = 'h = 0.1')
plt.legend(loc='best')
plt.show()