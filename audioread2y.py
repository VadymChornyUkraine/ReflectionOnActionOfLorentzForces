#import concurrent.futures
import pylab as plt
import numpy as np
from PIL import Image 
import cv2 as cvv
#import time as tm
import hickle as hkl
import os
import multiprocessing as mp

from scipy import stats as scp
import dill 
#from scipy.signal import savgol_filter
import moviepy.editor as mpv
#import wmi as wm

from RALf1FiltrVID import filterFourierQ
from RALf1FiltrVID import RALf1FiltrQ
from RALf1FiltrVID import RandomQ
import RALF1FilterX as XFilter

wrkdir = r"/home/abdulwahid/Документы/Work/W14_7/"
api_key = 'ONKTYPV6TAMZK464' 

wwrkdir_=wrkdir+r"/W10/"
nama='water1'

Lengt=20000
dsiz=700
Ngroup=3
Nproc=2*Ngroup#*(mp.cpu_count())
Lo=0
aTmStop=6
lSrez=0.99
NIt=3
NIter=100
DT=0.38
Nf_K=3
aDecm=3
dNIt=8
KPP=1    

def decimat(adat_):
    if Lo:
        adat_=np.log(adat_)
    adatx=0
    k=0
    adat__=np.zeros(int(len(adat_)/aDecm),float)
    for i in range(len(adat_)):
        adatx=adatx+adat_[i]
        if int(i/aDecm)*aDecm==i and i>0:
            adat__[k]=adatx/aDecm
            k=k+1
            adatx=0
    if Lo:
        return np.exp(adat__[1:len(adat__)-1])
    else:
        return (adat__[1:int(0.8*len(adat__))-1])

def fig2img ( fig ):
    fig.savefig(wrkdir +'dynamic.png',dpi=150,transparent=False,bbox_inches = 'tight')
    frame=Image.open(wrkdir +'dynamic.png')
    # fig.canvas.draw()
    # frame=Image.frombytes('RGB', fig.canvas.get_width_height(),
    #                        fig.canvas.tostring_rgb())
    return frame

def loaddata(aLengt,key):
    audio = mpv.AudioFileClip(wwrkdir_ +nama+".mp4")
    samples = np.asarray(audio.to_soundarray(),float)
    siz=len(samples)

    siz_=int(siz/dsiz)
    samplesx=np.zeros(siz_,float)
    for i in range(siz_):            
        samplesx[i]=(((np.mean(samples[i*dsiz+0:(i+1)*dsiz,0]))))
    samplesx=np.log(abs(samplesx))
    samplesx_=np.mean(np.asarray(list(filter((-np.Inf).__ne__, samplesx)),float))
    samplesx=samplesx-samplesx_
    siz=len(samplesx)
    samplesy=np.concatenate((samplesx,np.zeros(siz,float)))
    samplesx=samplesy[np.asarray(range(siz),int)+siz*(samplesx==-np.Inf)]  
    samplesx=samplesx-np.amin(samplesx)+np.std(samplesx)
    arrrxx=np.asarray(samplesx,float)
    return arrrxx[1:int(0.99*len(arrrxx))-1]

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

if __name__ == '__main__': 
    # w = wm.WMI(namespace="OpenHardwareMonitor")
    # temperature_infos = w.Sensor()
    # if temperature_infos==[]:
    #     os.startfile(r".\\OpenHardwareMonitor\OpenHardwareMonitor.exe")

    # del(w)
    # del(temperature_infos)
    # del(wm)

    ImApp=[]
    try:
        arrrxx=hkl.load(wrkdir + nama+"dat.rlf1")
    except:
        arrrxx=loaddata(Lengt,1)  
        arrrxx=np.asarray(arrrxx,float)
        arrrxx=decimat(arrrxx)
        try:                
            hkl.dump(arrrxx,wrkdir + nama+"dat.rlf1")
        except:
            os.mkdir(wrkdir)
            hkl.dump(arrrxx,wrkdir + nama+"dat.rlf1")
        
    esz=len(arrrxx)
    arr_rezDzRez=[[] for j in range(esz)]
    kkk=0
    out=0   
    
    interv="%d"%dsiz  

    aname=nama
    arrr=np.asarray(arrrxx).copy()  

    arrr=np.asarray(arrr,float)    
    Lengt=len(arrr)
    Nf=Lengt
    
    nn=int(Nf*DT)             
    NNew=int(Nf*0.6)  
    Nf=Nf+nn        
    ar0=np.asarray(arrr[0:])           
    
    arr_z=np.zeros(Nf,float)
    arr_z[0:Nf-NNew]=arrr[0:Nf-NNew].copy()
    arr_z[Nf-NNew:]=arr_z[Nf-NNew-1]
      
    adat0=''
                           
    Arr_AAA=np.zeros((NIter*Nproc,Nf),float) 
    arr_rezBz=np.zeros(Nf,float)
    arr_rezBzz=arr_rezBz.copy()
    
    all_rezAz=np.zeros((NIter,Nf),float)
    arr_z[Nf-NNew:]=arr_z[Nf-NNew-1]  
    all_RezN=np.zeros((Ngroup,NIter,Nf),float)+1e32
    all_RezM=np.zeros((Ngroup,NIter,Nf),float)-1e32
    dd1a=np.zeros((Ngroup,NIter,Nf),float)-1e32
    dd2a=np.zeros((Ngroup,NIter,Nf),float)+1e32
    all_RezNM=np.zeros((Ngroup,NIter,Nf),float)
    all_RezMM=np.zeros((Ngroup,NIter,Nf),float)
    argss=[[0] for j in range(Nproc)]    

    hh0=0
    hhh=0
    hhh_=0
            
    Koef_=[]
    ZZ=0
    key=0

    fig = plt.figure()
    axes = fig.add_axes([0.1, 0.1, 1.2, 1.2])
    axes_ = fig.add_axes([0, 0, 0.3, 0.3])     
    axes__ = fig.add_axes([0.4, 0, 0.3, 0.3])
    
    if Lo:
        axes.semilogy(ar0, 'r.')
        axes.semilogy(arr_z, 'go-')  #cut data used for model
        axes.grid(True, which="both", ls="-")
    else:
        axes.plot(ar0, 'r.')
        axes.plot(arr_z, 'go-')  #cut data used for model
        axes.grid(True, which="both", ls="-")

    axes.text(4, 4, 'Reflection on Action of Lorentz Forces-1, #2011612714  \n\n Course = %s, start = %s, step = %s * %s'%(aname,adat0,interv,aDecm),
            verticalalignment='bottom', horizontalalignment='right',
            transform=axes_.transAxes,color='blue', fontsize=14)        

    try:
        frame=fig2img(fig)  
    except:
        frame=fig2img(fig) 
    ImApp.append(frame)
    cimg = cvv.cvtColor(np.array(frame), cvv.COLOR_RGB2BGR)        
    gray_sz1=len(cimg[0])
    gray_sz2=len(cimg)
    aDur=2
    fourcc = cvv.VideoWriter_fourcc(*'MP4V')
    out = cvv.VideoWriter(wrkdir + aname+'.mp4',fourcc, aDur, (gray_sz1,gray_sz2))                   
    for icl in range(len(ImApp)):
        cimgx=(cvv.cvtColor(np.array(ImApp[icl]), cvv.COLOR_RGB2BGR)) 
        out.write(cimgx[0:gray_sz2,0:gray_sz1,:]) 
    out.release()
    plt.show()
    del(out)

