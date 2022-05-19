#%%
import numpy as np
import math
from numba import jit
import matplotlib.pyplot as plt
import time

#%%
pi2 = -2*math.pi

def fft(X):
    n = len(X)
    Fr = np.empty(n)
    Fi = np.empty(n)
    A = np.arange(0,n//2)*(-pi2/n)
    C = np.cos(A)
    S = np.sin(A)
    fft2(Fr,Fi,C,S,X,0,0,1,n)
    return Fr,Fi
    
@jit(nopython=True)
def fft2(Fr,Fi,C,S,X,fs,xs,xi,n):
    if n == 2:        
        f2 = fs+1
        x2 = X[xs+xi]
        Fr[f2] = X[xs] - x2
        Fi[f2] = 0
        Fr[fs] = X[xs] + x2
        Fi[fs] = 0
    else:
        n2 = n//2
        xi2 = xi*2
        fft2(Fr,Fi,C,S,X,fs,xs,xi2,n2)
        fft2(Fr,Fi,C,S,X,fs+n2,xs+xi,xi2,n2)
        a = 0
        km = fs + n2
        for k in range(fs,km):
            k2 = k + n2
            qr = C[a]*Fr[k2] - S[a]*Fi[k2]
            qi = C[a]*Fi[k2] + S[a]*Fr[k2]
            Fr[k2] = Fr[k] - qr
            Fi[k2] = Fi[k] - qi
            Fr[k] += qr
            Fi[k] += qi
            a += xi




                

#%%
n = 2**20
X = np.sin(np.linspace(0,2*math.pi,n)*10) + np.random.rand(n)
fft(X)

t = time.process_time()
Fr,Fi = fft(X)
print(time.process_time()-t)

t = time.process_time()
ffta2 = np.fft.fft(X)
print(time.process_time()-t)




ffta1 = np.sqrt(Fr*Fr+Fi*Fi)
ffta2 = np.abs(ffta2)

plt.close('all')
plt.figure()
plt.plot(ffta1)
plt.plot(ffta2,'--')
plt.legend(['my','np'])