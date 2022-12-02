import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import pandas as pd
import time as tm
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from PIL import Image 
import cv2 as cv

xmin=433.08723	
xmax=712.8824867	
ymin=276.3070205
ymax=428.5658698
# define grid.
xi = np.linspace(xmin,xmax,100)
yi = np.linspace(ymin,ymax,100)

def fig2img ( fig ):
    fig.savefig(wrkdir +'dynamic.png',dpi=150,transparent=False,bbox_inches = 'tight')
    frame=Image.open(wrkdir +'dynamic.png')
    # fig.canvas.draw()
    # frame=Image.frombytes('RGB', fig.canvas.get_width_height(),
    #                        fig.canvas.tostring_rgb())
    return frame


wrkdir = r"c:\Work\\"
zi=[]
dd_=0
for i in range(13):
    excel_data_df = pd.read_excel('Book1.xls', sheet_name='%s+'%(1900+i*10))
    dat=np.asarray(excel_data_df, float)
    x=dat[:,0]
    y=dat[:,1]
    z=dat[:,2]        
    # grid the data.
    data=griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')

    zi.append(data)

    dd=zi[0].copy()
    dd_=(dd_*i+dd)/(i+1)
        
im = plt.imread('isochr.jpg')

fig = plt.figure()
axy=[int(fig.get_figheight()* fig.dpi),int(fig.get_figwidth()* fig.dpi)]
coef=axy[0]/im.shape[0]*0.76
frm=ndimage.zoom(im[:,:,0],coef)
axy=frm.shape
im_=im[0:axy[0],0:axy[1],:].copy()
     
ImApp=[]

fourcc = cv.VideoWriter_fourcc(*'MP4V')        

for j in range(3):
    aaaa=(ndimage.zoom(im[:,:,j],coef))
    im_[:,:,j]=aaaa.copy()

lvls1= np.linspace(-0.1,0.1,20)
ddx=np.zeros((20,20),float)
for i in range(13):  
    dd=zi[i]-dd_
    dd=dd-gaussian_filter(dd, sigma=0.8)  
    for ii in range(20):
        for jj in range(20):
            dd1=np.amax((-dd[ii*5:(ii+1)*5,jj*5:(jj+1)*5]))
            dd2=np.amin((-dd[ii*5:(ii+1)*5,jj*5:(jj+1)*5]))
            asr1=np.abs(dd1)>np.abs(dd2)
            asr2=np.abs(dd1)<np.abs(dd2)
            ddx[ii,jj]=dd1*asr1+dd2*asr2
    ax = plt.gca()  
    #CS = plt.contour(xi,yi,dd_,levels=lvls2,linewidths=0.5,colors='k')
    CS = ax.contourf(xi[::5],yi[::5],ddx,levels=lvls1,cmap='RdBu')      
    plt.title('Anomaly intensity of magnetic field amplitude %s'%(1900+i*10))
    for c in CS.collections:
        c.set_edgecolor("face")        
    ax.figure.savefig('test.png')
    frame=Image.open('test.png') 
     
    ax.figure.figimage(frame,-axy[0]*.1,-axy[1]*.04)
    ax.figure.figimage(im_,axy[0]*.15,axy[1]*.066,alpha=0.5)
    plt.show()

    ax.figure.savefig('test.png')
    frame=Image.open('test.png')        
    
    cimg = cv.cvtColor(np.array(frame), cv.COLOR_RGB2BGR)        
    gray_sz1=len(cimg[0])
    gray_sz2=len(cimg)
    ImApp.append(frame)
    aDur=1
    out = cv.VideoWriter('mercury.mp4',fourcc, aDur, (gray_sz1,gray_sz2))                   
    for icl in range(len(ImApp)):
        cimgx=(cv.cvtColor(np.array(ImApp[icl]), cv.COLOR_RGB2BGR)) 
        out.write(cimgx[0:gray_sz2,0:gray_sz1,:]) 
    out.release()
    del(out)
    tm.sleep(1) 


 


          
               




 


