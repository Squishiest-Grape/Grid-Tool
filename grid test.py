from numba import jit
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import scipy.signal, scipy.ndimage
import os
import time


@jit(nopython=True)
def squareConvolve(C,O,K):
    h,w = C.shape
    n = len(K)
    for x in range(w):
        for y in range(h):
            s_h = 0
            s_v = 0
            for i in range(n):
                for j in range(n):
                    s_h += K[j,i]*O[y+j,x+i]
                    s_v += K[i,j]*O[y+j,x+i]
            C[y,x] += (s_h*s_h+s_v*s_v)**.5
    return C

@jit(nopython=True)
def colorConvolve(C,O,K):
    h,w = C.shape
    n = len(K)
    for x in range(w):
        for y in range(h):
            s_h = 0
            s_v = 0
            for c in range(3):
                for i in range(n):
                    for j in range(n):
                        s_h += K[j,i]*O[y+j,x+i,c]
                        s_v += K[i,j]*O[y+j,x+i,c]
                C[y,x] += (s_h*s_h+s_v*s_v)**.5
    return C



@jit(nopython=True)
def houghTransform(H,E,C,S,dr,limitL,limitH,offsetX,offsetY,offsetR,Ha):
    # H is the hough-space to put the data
    # Ha is the housh-space to put the adjustment
    # E is the edge data image
    # C is the pre-computed list of cosines for the angles 0-179
    # S is the pre-computed list of sines for the angles 0-179
    # dr is the radius spacing (if I wanted to downsample)
    # limits$ are low and high limits to look at the edges
    # offset& are the index offset to get pixels refrences for the image and hough space
    
    h,w = E.shape  # get hieght and width
    n = len(C)     # get number of angles
    for x in range(w):  # for every point
        for y in range(h):    
            for i_a in range(n):  # for every checked angle
                r = C[i_a]*(x-offsetX) + S[i_a]*(y-offsetY) # calculated the line radius
                r_ = (r + offsetR)/dr                       # offset to fit in hough space
                i_r = int(r_)                               # get index
                di = r_ - i_r                               # get partial index
                e = E[y,x]                                  # get edge reference
                if e >= limitL and e <= limitH:             # check if criteria met
                    H[i_r,i_a] += 1-di                      # apply partial data to first half of index
                    H[i_r+1,i_a] += di                      # apply partial data to second half of index
                Ha[i_r,i_a] += 1-di
                Ha[i_r+1,i_a] += di
    return H,Ha

@jit(nopython=True)
def getAdjustment(D,C,S,Rad,l,t,r,b):
    n = len(C)
    m = len(R)
    for i in range(n):
        c = C[i]
        s = S[i]
        for j in range(m):
            rad = Rad[j]
            if s != 0:
                rs = rad/s
                cs = c/s
                p0 = rs - l*cs
                p2 = rs - r*cs
            else:
                p0 = math.nan
                p2 = math.nan
            if c != 0:
                rc = rad/c
                sc = s/c
                p1 = rc - t*sc
                p3 = rc - b*sc
            else:
                p1 = math.nan
                p3 = math.nan     
            g0 = (p0 >= b) and (p0 < t)
            g1 = (p1 >= l) and (p1 < r)
            g2 = (p2 > b) and (p2 <= t)
            g3 = (p3 > l) and (p3 <= r)    
            if g0 and g1:
                dx = l - p1
                dy = p0 - t
                D[j,i] = (dx*dx+dy*dy)**.5
                continue
            elif g0 and g2:
                dx = l - r
                dy = p0 - p2
                D[j,i] = (dx*dx+dy*dy)**.5
                continue
            elif g0 and g3:
                dx = l - p3
                dy = p0 - b
                D[j,i] = (dx*dx+dy*dy)**.5
                continue
            elif g1 and g2:
                dx = p1 - r
                dy = t - p2
                D[j,i] = (dx*dx+dy*dy)**.5
                continue
            elif g1 and g3:
                dx = p1 - p3
                dy = t - b
                D[j,i] = (dx*dx+dy*dy)**.5
                continue
            elif g2 and g3:
                dx = r - p3
                dy = p2 - b
                D[j,i] = (dx*dx+dy*dy)**.5
                continue
            else:
                D[j,i] = math.nan
                continue
    return D

@jit(nopython=True)
def getBB(a,rad,l,t,r,b):
    c = math.cos(a)
    s = math.sin(a)
    if s != 0:
        rs = rad/s
        cs = c/s
        p0 = rs - l*cs
        p2 = rs - r*cs
    else:
        p0 = math.nan
        p2 = math.nan
    if c != 0:
        rc = rad/c
        sc = s/c
        p1 = rc - t*sc
        p3 = rc - b*sc
    else:
        p1 = math.nan
        p3 = math.nan     
    g0 = (p0 >= b) and (p0 < t)
    g1 = (p1 >= l) and (p1 < r)
    g2 = (p2 > b) and (p2 <= t)
    g3 = (p3 > l) and (p3 <= r)    
    if g0 and g1:
        return [(l,p0),(p1,t)]
    elif g0 and g2:
        return [(l,p0),(r,p2)]
    elif g0 and g3:
        return [(l,p0),(p3,b)]
    elif g1 and g2:
        return [(p1,t),(r,p2)]
    elif g1 and g3:
        return [(p1,t),(p3,b)]
    elif g2 and g3:
        return [(r,p2),(p3,b)]
    else:
        return [(math.nan,math.nan),(math.nan,math.nan)]


def gausePeaks(vals,std):
    peaks = np.ones(len(vals),dtype=bool)
    peaks = findPeaks1(peaks,vals,std)
    return np.argwhere(peaks)[:,0]


@jit(nopython=True)
def findPeaks1(peaks,vals,std):
    n = len(vals)
    n2 = n//2
    d = 2*(std)**2
    for i in range(n):
        if peaks[i]: 
            v = vals[i]
            if v > 0: 
                for x in range(1,n2):
                    y = v*math.e**((-x**2)/d)
                    j = (i-x)%n
                    if y > vals[j]: peaks[j] = False 
                    j = (i+x)%n  
                    if y > vals[j]: peaks[j] = False 
    return peaks


def avg(x,r):
    a = np.empty_like(x)
    avg1(a,x,r)
    return a

@jit(nopython=True)
def avg1(a,x,r):
    n = len(x)
    l = 2*r+1
    for i in range(r):
        a[i] = 0
        for j in range(0,i+r+1): a[i] += x[j]
        a[i] += (r-i)*x[0]
        a[i] /= l
        # a[i] /= (r-i+r+1)
    for i in range(r,n-r):
        a[i] = 0
        for j in range(i-r,i+r+1): a[i] += x[j]
        a[i] /= l
    for i in range(n-r,n):
        a[i] = 0
        for j in range(i-r,n): a[i] += x[j]
        a[i] += (n-i)*x[-1]
        a[i] /= l
        # a[i] /= (n-i+r+1)

def boxPoints(h,w,a,o):
    p1,p2 = getBB(a/180*math.pi,o,-w/2,h/2,w/2,-h/2)
    x = [p1[0]+w/2,p2[0]+w/2]
    y = [p1[1]+h/2,p2[1]+h/2]
    return x,y




#%% 

dp = 1  # distance granularity in pixels
da = 1  # angle granularity in degrees

K = [[1,-1],  # kernal
      [1,-1]]

line_filter = 30 # line filter strength in percent
fft_res = 1

percent = [55,85] # edge detection percentile

plt.close('all')


skip = False


for image_index in [0]: #range(15):


    # %% 
    
    folder = 'test_images'
    
    files = os.listdir(folder)
    file = os.path.join(folder,files[image_index])
    
    
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    t = time.time()
    # image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
    
    K = np.array(K).astype(float)
    offset = len(K)/2
    
    edges = np.zeros(tuple(np.array(image.shape[:-1])-(len(K)-1)),dtype=float)
    
    # edges = squareConvolve(edges,image_gray,K)
    edges = colorConvolve(edges,image,K)
    
    if not skip:
    
        inds = np.sort(edges.flatten())
        limitL = inds[int(percent[0]/100*len(inds))]
        limitH = inds[int(percent[1]/100*len(inds))-1]
        
        A = np.arange(0,180,da)
        C = np.cos(A*np.pi/180)
        S = np.sin(A*np.pi/180)
        nA = len(A)
        
        offsetX = edges.shape[1]/2 -.5
        offsetY = edges.shape[0]/2 -.5
        
        nr = int(np.linalg.norm((offsetX,offsetY))/dp)+1
        offsetR = nr*dp
        R = np.linspace(-dp*nr,dp*nr,nr*2+1)
        
        H = np.zeros((nr*2+1,nA),dtype=float)
        adj = np.zeros((nr*2+1,nA),dtype=float)
        
        # adj = np.zeros_like(H)
        # adj = getAdjustment(adj,C,S,R,-offsetX,offsetY,offsetX,-offsetY)
        
        H,adj = houghTransform(H, edges, C, S, dp, limitL, limitH, offsetX, offsetY, offsetR, adj)
        adj[adj==0] = math.nan
        
        
        adj = adj*(1 - line_filter/100) + R[-1]*(line_filter/200)
        H /= adj
        
        
        
        n = 2**(int(np.ceil(np.log2(H.shape[0]*fft_res)))-1)
        n2 = n*2
        # n34 = int(n*3/4) + 1
        
        fft = np.fft.fft(np.nan_to_num(H),n2,0)[:n,:]
        fft_ang = np.angle(fft)
        fft_amp = np.abs(fft)
        
        f = np.linspace(0,.5,n,endpoint=False)    
        
        fft_avg = []
        rw = math.log(n,6)
        
        
        for i in range(nA):
            fft_avg.append(avg(fft_amp[:,i],int(rw))*rw) 
        
        fft_avg = np.mean(fft_avg,axis=0)
        fft_off = (fft_amp.T-fft_avg).T
        
        
    
        
        # fft_off = fft_off[:n34,:]
        
        fft_off[fft_off<0] = 0
        
        
        
        std = 45*fft_res
        peaks = []
        prom = np.empty(nA)
        for i in range(nA):
            inds = gausePeaks(fft_off[:n//2,i],std)
            peaks.append(inds)
            prom[i] = np.sum(fft_off[inds,i])
        
        
        inds = gausePeaks(prom,nA/4)
        
        
        grid = {}
        for ind in inds:
            y = fft_off[peaks[ind],ind]
            i = np.argmax(y>np.max(y)*0.4)    
            i = peaks[ind][i]
            s = dp*n2/i
            o = (s*fft_ang[i,ind]/(2*math.pi) + dp*nr) % s
            grid[ind] = dict(i=i,s=s,o=o)
    

#%%

    t = time.time()-t
    print(file)
    print(f'{t} seconds')
    if not skip: print(grid)
    
    
    #%%
    
    title = f'Test Image {image_index} :'

    
    plt.figure()
    plt.imshow(image) 
    plt.title(title+' image')
    
    plt.figure()
    plt.imshow(edges)
    plt.title(title+' edges')
    
    plt.figure()
    plt.imshow(np.logical_and(edges>=limitL,edges<=limitH))
    plt.title(title+' edge detection')
    
    plt.figure()
    plt.imshow(adj,extent=[0,180-da,R[0],R[-1]],aspect='auto',interpolation='none')
    plt.title(title+' hough adjustment')
    
    
    plt.figure()
    plt.imshow(H,extent=[0,180-da,R[0],R[-1]],aspect='auto',interpolation='none')
    plt.title(title+' hough transform')
    
    c = 30
    plt.figure()
    plt.plot(H[:,::c])
    if 180/c < 20: plt.legend(np.arange(len(A)//c)*da*c)
    plt.title(title+' hough transform')
    
    
    c = 1
    plt.figure()
    plt.plot(f,fft_amp[:,::c])
    plt.plot(f,fft_avg,'--')
    plt.title(title+' ffts')
    
    c = 30
    plt.figure()
    plt.plot(f,fft_off[:,::c])
    if 180/c < 20: plt.legend(np.arange(len(A)//c)*da*c)
    plt.title(title+' normalized fft')
    

    plt.figure()
    plt.plot(A,prom)
    plt.scatter(A[inds],prom[inds],color='red')
    plt.title(title+' angle detection')
    
    
    plt.figure()
    plt.plot(f,fft_off[:,inds])
    plt.legend(np.arange(0,180,da)[inds])
    for ind in inds:
        i = peaks[ind]
        plt.scatter(f[i],fft_off[i,ind])
    for a,x in grid.items():
        plt.scatter(f[x['i']],fft_off[x['i'],a],color='red',marker='x')
    plt.title(title+' spacing detection')
    
    g = 30
    plt.figure()
    plt.imshow(image)
    if not skip:
        for a,info in grid.items():
            for do in range(-g,g+1):
                x,y = boxPoints(*image.shape[:-1],-a,-info['o']+do*info['s'])
                plt.plot(x,y,'r',lw=0.5)    
    plt.title(title+' grid')
    
