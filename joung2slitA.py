import numpy as np
#import pylab as plt
import matplotlib.pyplot as plt
from PIL import Image
import time as tm

import RALF1FilterX as XFilter

Nx = 100+1 # Number of Grids/steps in x-axis
Ny = 50+1  # Number of Grids/steps in y-axis
dx = 0.1   # Step size
dy = dx
x = np.asarray(range(Nx),float)*dx # x-axis
y = np.asarray(range(Ny),float)*dy # y-axis
mpx = (Nx-1)/2# Mid point of x axis
                # ( Mid pt of 0 to 100 = 50 here )
T = 220+1  # Total number of time steps
f = 1000   # frequency of source
dt = 0.0001# Time-Step
t= np.asarray(range(T),float)*dt
v = 500 # Wave velocity
c = v*(dt/dx) # CFL condition

U = np.zeros((T,Nx,Ny),float)
s1 = T # floor(T/f)  
w = 24+1 # width of the source

pi=np.arccos(0)*2
for d in range(2):
    for ss in range(s1):
        U[ss][int(mpx-w-d)][0] = np.sin(2*pi*f*t[ss])
        U[ss][int(mpx-w-d)][1] = np.sin(2*pi*f*t[ss])
        U[ss][int(mpx+w+d)][0] = np.sin(2*pi*f*t[ss])
        U[ss][int(mpx+w+d)][1] = np.sin(2*pi*f*t[ss])
        
# Finite Difference Scheme
for k in range(2,T-1): 
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            U1 = 2*U[k-1][i][j]-U[k-2][i][j]
            U2 = U[k-1][i-1][j]+U[k-1][i+1][j]+U[k-1][i][j+1]+U[k-1][i][j-1]-4*U[k-1][i][j]
            U[k][i][j] =U1+c*c*U2
    for i in range(Nx):
        U[k+1][i][Ny-1] = 0.5*( U[k][i][Ny-1]+U[k][i][Ny-2])
    for i in range(Ny):
        U[k+1][Nx-1][i] = 0.5*( U[k][Nx-1][i]+U[k][Nx-2][i])
        U[k+1][0][i] = 0.5*( U[k][0][i]+U[k][1][i])

Ux=U.copy()
M=Nx
N=Ny
szN=T*N
UU=np.zeros((szN,M),float)
k=-1
for l in range(N):
    for j in range(T):  
        k=k+1;
        for i in range(M):        
            UU[k][i]=U[j][i][l]  

Uplot0= XFilter.RALF1FilterX(UU,szN,M,0,0)
UUmn=np.mean(UU)
UU=UU-UUmn
UU1=UU-UU*(UU<0)
UU2=UU-UU*(1-(UU<0))
Uplot1= XFilter.RALF1FilterX(UU1,szN,M,1,0)+UUmn
Uplot2= -XFilter.RALF1FilterX(-UU2,szN,M,1,0)+UUmn
UU=Uplot1+Uplot2-Uplot0

k=-1
for l in range(N):
    for j in range(T):  
        k=k+1;
        for i in range(M):  
            U[j][i][l]=UU[k][i] 

wrkdir = r"C:\Work\\"
aname='joung2slit_xxA'
ImApp=[]

Uplot_=[]
Uplotx_=[]
for j in range(T-1):   
    Uplot=U[j].transpose()
    Uplotx=Ux[j].transpose()
    
    Uplot_.append(Uplot[N-1])
    Uplotx_.append(Uplotx[N-1])
    
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    k=0
    for ax in axs.ravel():
        if k==0:
            cs=ax.plot(x,np.asarray(Uplotx_,float).transpose(), 'oy', alpha=0.1)
        if k==1:
            cs=ax.plot(x,np.asarray(Uplot_,float).transpose(), 'oy', alpha=0.1)  
        if k==2:
            cs = ax.contourf(x, y, Uplotx)
        if k==3:
            cs = ax.contourf(x, y, Uplot)
        k=k+1    
        ax.locator_params(nbins=4)

    plt.show()  
    fig.savefig(wrkdir +'dynamic.png',dpi=100,transparent=False,bbox_inches = 'tight')
    frame=Image.open(wrkdir +'dynamic.png')
    ImApp.append(frame)
    ImApp[0].save(wrkdir + aname+'.gif',save_all=True,append_images=ImApp[1:],duration=200,loop=0)

    tm.sleep(.1)



