import numpy as np

x1 = 0
x2 = np.pi
y1 = 0.5
y2 = -0.5
ep = 0.0001

def thomas_(a,b,c,d):
    c_ = np.zeros(c.size)
    d_ = np.zeros(d.size)
    c_[0] = c[0]/b[0]
    d_[0] = d[0]/b[0]
    for i in range(1, c.shape[0]-1):
        c_[i] = c[i]/(b[i] - a[i]*c_[i-1])
    for i in range(1, d.shape[0]):
        d_[i] = (d[i] - a[i]*d_[i-1])/(b[i] - a[i]*c_[i-1])

    return [c_, d_]

def mod(a):
    if a>0:
        return a
    return -1*a

def main_(n=3):
    h = np.pi/n
    y = np.zeros(n+1)
    x_f = np.zeros(n+1)

    for i in range(n+1):
        x_f[i] = (i)*h
        y[i] = 0.5 - np.sin(i*h/2)

    flag = 0
    while flag!=1:
        a = np.zeros(n-1)
        b = np.zeros(n-1)
        c = np.zeros(n-1)
        d = np.zeros(n-1)
        res = np.zeros(n-1)
        
        for i in range(n-1):
            a[i] = ( 1/(h*h) + (1/(2*h*h))*(y[i+2]-y[i]) )
        for i in range(n-1):
            b[i] = ( -2/(h*h) - 2*y[i+1] + 1 )
        for i in range(n-1):
            c[i] = ( 1/(h*h) - (1/(2*h*h))*(y[i+2]-y[i]) )
        for i in range(n-1):
            d[i] = ( -1*((y[i+2]-y[i])/(2*h)) * ((y[i+2]-y[i])/(2*h)) - y[i+1]*y[i+1] - 1 )
            d[0] = d[0] - 0.5 * a[0]
            d[-1] = d[-1] + 0.5*c[-1]
        
        c_, d_ = thomas_(a,b,c,d)

        res[-1] = d_[-1]
        for i in range(n-2):
            res[n-3-i] = d_[n-3-i] - res[n-2-i]*c_[n-3-i]
        flag=1
        for i in range(n-1):
            if mod(y[i+1]-res[i]) > ep:
                flag=0
        for i in range(1,n):
            y[i] = res[i-1]

    return [y, x_f]

a_1, x_1 = main_(3)
a_2, x_2 = main_(7)
a_3, x_3 = main_(13)


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x_1, a_1, 'mo', label = 'h = 1.0472')
plt.plot(x_2, (a_2), 'bx', label = 'h = 0.4487')
plt.plot(x_3, (a_3), 'yo', label = 'h = 0.2416')


plt.legend(loc='best')
plt.show()