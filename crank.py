import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import math
def thomas_algorithm(a, b, c, d):

    a = list(a)
    b = list(b)
    c = list(c)
    d = list(d)
    assert len(a) == len(b) == len(c) == len(d)
    N = len(c)
    c_ = [0 for i in range(N)]
    d_ = [0 for i in range(N)]
    f = [0 for i in range(N)]
    c_[0] = c[0]/b[0]
    d_[0] = d[0]/b[0]

    for i in range(1, N):
        if i is not N-1:
            c_[i] = c[i]/(b[i] - a[i]*c_[i-1])
        d_[i] = (d[i] - a[i]*d_[i-1])/(b[i] - a[i]*c_[i-1])

    f[N-1] = d_[N-1]
    for i in range(N-2, -1, -1):
        f[i] = d_[i] - c_[i]*f[i+1]

    return f


def crank(dx):
    rX = 1
    n = 1+int(rX/dx)
    X = np.linspace(0, rX, n)

    dt = 1/96
    m = 10
    U = np.zeros((n, m), dtype=np.float64)

    U[0, :] = 0
    U[n-1, :] = 0
    for j in range (0,n):
        U[j, 0] =(math.sin(dx*j*(np.pi))) 

    v=1
    r=v*(dt/dx**2)
    for k in range(1, m):
        A = [r/2 for i in range(0,n-2)]
        C = [r/2-1 for i in range(0, n-2)]
        B = [(-1)*r for i in range(0, n-2)] 
        a = U[0:n-2, k-1]*(r/(-2))
        b = U[1:n-1, k-1]*(r-1)
        c = U[2:n, k-1]*(r/(-2))
        D = a+b+c
        U[1:n-1, k] = thomas_algorithm(A, B, C, D)
    return U,X,m
    
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
g,X,k=crank(0.01)   
plt.plot(X,g[0:,k-1],'r+',label='0.01')

g,X2,k=crank(0.05)   
plt.plot(X2,g[0:,k-1],'b.',label='0.05')

g,X3,k=crank(0.1)   
plt.plot(X3,g[0:,k-1],'g*',label='0.1')

plt.legend(loc='best')
plt.show()
