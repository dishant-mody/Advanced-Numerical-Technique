import numpy as np

x1 = 0
x2 = 10
y1_ = 0
y2_ = 0
ep = 0.0001
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

def mod(a):
    if a>0:
        return a
    return -1*a

def main_(n):
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
        f[i] = 5*x_f[i]*x_f[i] - x_f[i]*x_f[i]*x_f[i]/3
        F[i] = x_f[i]*(10 - x_f[i])
    res_final = np.zeros(n+1)
    res_final = f
    flag=0
    
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
            b[i][1][1] = -2/(h*h) + 2*F[i+1]
        for i in range(0, n-1):
            c[i][0][0] = 0
            c[i][0][1] = 0
            c[i][1][0] = 0
            c[i][1][1] = 1/(h*h) + f[i+1]/(2*h)
        for i in range(0, n-1):
            d[i][0] = 0
            d[i][1] = F[i+1]*F[i+1] + f[i+1]*(F[i+2]-F[i])/(2*h)

    b[0][1][1] = b[0][1][1] + (4/3)*(1/(h*h) - f[1]/(2*h))
    c[0][1][1] = c[0][1][1] + (-1/3)*(1/(h*h) - f[1]/(2*h))
    
    c_,d_ = thomas_(a,b,c,d)
    res = np.zeros((n-1,2))
    res[-1] = d_[-1]
    for i in range(n-2):
        res[n-3-i] = d_[n-3-i] - np.dot(c_[n-3-i], res[n-2-i])
    flag=1

    for i in range(n-1):
        if mod(f[i+1]-res[i, 0]) > ep or mod(F[i+1]-res[i,1]) > ep:
            flag=0
    
    f[1:-1] = res[:,0]
    F[1:-1] = res[:,1]
    return [res_final, x_f]

a_1, x_1 = main_(3)
a_2, x_2 = main_(6)
a_3, x_3 = main_(11)
a_4, x_4 = main_(13)
a_5, x_5 = main_(17)
a_6, x_6 = main_(23)

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x_1[:-1], a_1[:-1], 'r.', label = 'h = 3.333')
plt.plot(x_2[:-1], (a_2[:-1]), 'k.', label = 'h = 1.666')
plt.plot(x_3[:-1], (a_3[:-1]), 'y.', label = 'h = 0.9091')
plt.plot(x_4[:-1], (a_4[:-1]), 'g.', label = 'h = 0.7692')
plt.plot(x_5[:-1], (a_5[:-1]), 'b.', label = 'h = 0.5882')
plt.plot(x_6[:-1], (a_6[:-1]), 'c.', label = 'h = 0.4347')
plt.legend(loc='best')
plt.show()
